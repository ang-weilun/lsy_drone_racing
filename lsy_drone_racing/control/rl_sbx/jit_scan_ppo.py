"""SBX PPO subclass that replaces the per-step host loop with a JAX scan.

Stock :meth:`sbx.PPO.collect_rollouts` dispatches one device call per env
step (policy forward, env step, value forward, log-prob, then back to
host). On 16k vectorized worlds that overhead caps us at roughly
75k env-steps/s. The rl_song stack collects the entire rollout inside a
single ``jax.lax.scan`` and hits ~700k env-steps/s on the same GPU.

:class:`JitScanPPO` overrides only :meth:`collect_rollouts`. The rest of
SBX's training loop (loss, optimizer, callback dispatch) is reused. The
override:

1. Builds an :class:`RLSBXRolloutStaticConfig` from the env wrapper's
   public attributes.
2. Calls :func:`scan_rollout` once.
3. Pulls the device-side rollout result back to host and writes the
   stacked transitions directly into the SB3
   :class:`~stable_baselines3.common.buffers.RolloutBuffer`, bypassing
   the per-step :meth:`RolloutBuffer.add` (which is single-step Python
   and would dominate the runtime if called inside the override).
4. Updates ``self.num_timesteps`` and the env wrapper's carried state
   (``_prev_action``, ``_prev_env_obs``) so the next rollout picks up
   where this one left off.
5. Calls :func:`RolloutBuffer.compute_returns_and_advantage` with the
   bootstrap ``V(s_{T+1})`` produced by the scan.

Compatibility constraints — :class:`JitScanPPO` is **only** valid with:

* :class:`~lsy_drone_racing.control.rl_sbx.env_gym.RLSBXVecEnv` (reads
  ``env.jax_env``, ``env.{actor,critic}_normalizer``, ``env.reward_cfg``,
  ``env.alpha_max``, ``env.{thrust_min,thrust_max}``,
  ``env._prev_action``, and writes back ``env._prev_action`` /
  ``env._prev_env_obs``).
* :class:`~lsy_drone_racing.control.rl_sbx.policy.AsymmetricActorCriticPolicy`
  (reads ``policy.actor_state.params`` / ``policy.vf_state.params``).

Trade-offs deliberately taken for milestone-1:

* **No per-step callback dispatch.** SBX's stock collector calls
  ``callback.on_step()`` every env step (for early-stopping / info
  buffer / wandb pushes). We call ``callback.on_rollout_end()`` only.
  Per-step callbacks are not used in our recipe.
* **No timeout bootstrap.** SBX's stock collector mutates
  ``rewards[idx] += gamma * V(s_terminal)`` for envs that truncated
  (``TimeLimit.truncated and not terminated``). We skip this. The
  bias is bounded by ``gamma * V_max * (truncation_rate per rollout)``
  — small on milestone-1 (lap deadlines rarely fire pre-finish). A
  follow-up commit can fold the bootstrap inside the scan body using
  the post-reset critic forward.

References:
----------
Stable Baselines Jax (SBX), https://github.com/araffin/sbx.
Song, Y. et al. (2023). Reaching the limit in autonomous racing.
    *Science Robotics* 8, eadg1462.
"""

from __future__ import annotations

import time
from functools import partial
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import torch as th
from sbx import PPO
from stable_baselines3.common.utils import explained_variance

from lsy_drone_racing.control.rl_sbx.rollout import (
    RLSBXScanResult,
    make_static_config,
    scan_rollout,
)
from lsy_drone_racing.control.rl_song.rollout import (
    SRC_PHASE1_SEG,
    SRC_PHASE2_REPLAY,
    SRC_TRUE_START,
)
from lsy_drone_racing.envs.race_core import obs as race_core_obs

if TYPE_CHECKING:
    from flax.training.train_state import TrainState
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecEnv

# rl_sbx env wrapper has a single drone per vec env; mirrors
# :data:`rl_sbx.rollout.SINGLE_DRONE_INDEX`.
SINGLE_DRONE_INDEX: int = 0

# Pessimistic value-loss prefactor matching
# :data:`lsy_drone_racing.control.rl_song.train.VALUE_LOSS_SCALE`. Stock SBX
# ``_one_update`` multiplies the mean squared value error only by ``vf_coef``;
# rl_song's hand-rolled PPO additionally scales by 0.5 (the conventional
# ½(V−R)² factor — see Schulman 2017 Eq. 9). Folding it in here so the same
# ``vf_coef`` config knob produces the same effective value-loss magnitude in
# both stacks. Without this, an SBX run at ``vf_coef=0.5`` applies twice the
# effective value-loss weight of an rl_song run at the same setting.
VALUE_LOSS_SCALE: float = 0.5

# Tiny constant added to the policy ratio inside the approx-KL estimator to
# keep ``log(ratio)`` finite when the importance ratio underflows in fp32.
# Mirrors the stock SBX value at ``sbx/ppo/ppo.py:320``.
APPROX_KL_RATIO_EPS: float = 1e-7


class JitScanPPO(PPO):
    """SBX :class:`PPO` subclass using :func:`jax.lax.scan` for rollout collection.

    See module docstring for the throughput rationale and compatibility
    constraints.

    Notes:
    -----
    Inherits the entire ``learn`` / ``train`` / loss / optimizer stack
    from SBX. Only :meth:`collect_rollouts` is overridden.
    """

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect ``n_rollout_steps`` of experience via :func:`scan_rollout`.

        Parameters
        ----------
        env : VecEnv
            Must be a :class:`RLSBXVecEnv` (or the same interface). Read
            for env attributes and the inner JAX env; written for
            ``_prev_action`` / ``_prev_env_obs``.
        callback : BaseCallback
            Per-step callbacks are NOT dispatched (see module docstring).
            ``on_rollout_start`` and ``on_rollout_end`` fire as usual.
        rollout_buffer : RolloutBuffer
            Filled in bulk via direct array writes.
        n_rollout_steps : int
            Number of env steps per env to collect this iteration.

        Returns:
        -------
        bool
            Always ``True`` (we never early-stop the rollout — the
            per-step ``callback.on_step()`` hook that could request it
            isn't dispatched).
        """
        if self._last_obs is None:
            raise RuntimeError(
                "JitScanPPO.collect_rollouts called before env reset populated _last_obs."
            )
        if self._last_episode_starts is None:
            raise RuntimeError(
                "JitScanPPO.collect_rollouts called before _last_episode_starts initialized."
            )

        profile = getattr(self, "profile_throughput", False)
        if profile:
            t_entry = time.perf_counter()
            prev_exit = getattr(self, "_prof_last_exit", None)
            if prev_exit is not None:
                self.logger.record("time/prof_update_plus_log_s", t_entry - prev_exit)

        rollout_buffer.reset()
        callback.on_rollout_start()

        # Pull wrapper state. ``env`` is an ``RLSBXVecEnv`` (or compatible
        # wrapper); raise on missing attrs to surface a config mistake
        # rather than silently fall back to stock SBX behavior.
        for attr in (
            "jax_env",
            "actor_normalizer",
            "critic_normalizer",
            "reward_cfg",
            "alpha_max",
            "thrust_min",
            "thrust_max",
            "seg_init_kwargs",
            "phase2_buffer",
            "episode_source",
            "_prev_action",
        ):
            if not hasattr(env, attr):
                raise AttributeError(
                    f"JitScanPPO requires an env exposing '{attr}'. "
                    "Use RLSBXVecEnv (see lsy_drone_racing.control.rl_sbx.env_gym)."
                )

        static_cfg = make_static_config(
            n_steps=n_rollout_steps,
            n_envs=env.num_envs,
            thrust_min=env.thrust_min,
            thrust_max=env.thrust_max,
            tangent_alpha_max_rad=env.alpha_max,
            reward_cfg=env.reward_cfg,
            **env.seg_init_kwargs,
        )
        # Keep the rollout static config stable across the Phase-2 warmup
        # boundary. The effective probability is a JAX runtime scalar; the
        # scan masks the replay buffer empty while it is zero so we do not
        # recompile when training crosses ``phase2_warmup_steps``.
        phase2_warmup_steps = int(getattr(env, "phase2_warmup_steps", 0))
        phase2_prob = float(getattr(env, "phase2_prob", 0.0))
        effective_phase2_prob = (
            phase2_prob if int(self.num_timesteps) >= phase2_warmup_steps else 0.0
        )

        # ``_last_episode_starts`` lives on SBX as a float32 / bool numpy
        # array. The scan needs a JAX bool of shape (n_envs,) — convert
        # explicitly so a stale dtype on the SBX side doesn't propagate.
        next_done_jax = jnp.asarray(self._last_episode_starts, dtype=jnp.bool_)

        # Advance the wrapper's RNGs once per rollout. The scan does its
        # own per-step splits internally; we only need one fresh key
        # per dispatch. The reset-key stream is independent of the
        # action-sampling stream so toggling seg-init does not bit-shift
        # the policy's exploration trajectory.
        rng_key = self._next_rollout_rng_key()
        reset_rng_key = self._next_reset_rng_key()

        if profile:
            t_scan_start = time.perf_counter()
        scan_result: RLSBXScanResult = scan_rollout(
            env.jax_env.data,
            self.policy.actor_state.params,
            self.policy.vf_state.params,
            env.actor_normalizer,
            env.critic_normalizer,
            env._prev_action,
            rng_key,
            reset_rng_key,
            next_done_jax,
            env.phase2_buffer,
            env.episode_source,
            jnp.asarray(effective_phase2_prob, dtype=jnp.float32),
            env.jax_env._step,
            env.jax_env._reset,
            static_cfg,
        )
        if profile:
            jax.block_until_ready(scan_result)
            t_scan_done = time.perf_counter()
            self.logger.record("time/prof_scan_s", t_scan_done - t_scan_start)

        # Round-trip env state. The wrapper's ``_prev_env_obs`` is only
        # consumed by ``step_wait``; refreshing it here keeps the wrapper
        # in a valid state if external code calls ``step_wait`` between
        # iterations (e.g. a manual eval interleaved with training).
        env.jax_env.data = scan_result.env_data
        env._prev_action = scan_result.prev_action_env_4vec
        env.phase2_buffer = scan_result.phase2_buffer
        env.episode_source = scan_result.source
        final_env_obs_full = race_core_obs(scan_result.env_data)
        env._prev_env_obs = {
            key: value[:, SINGLE_DRONE_INDEX] for key, value in final_env_obs_full.items()
        }

        # Bulk-write the stacked transitions into the SB3 buffer. Bypasses
        # ``RolloutBuffer.add`` (per-step Python; would dominate runtime
        # inside the override). Field shapes / dtypes match the buffer's
        # ``reset()`` allocations (see
        # stable_baselines3/common/buffers.py:391-398).
        outputs = scan_result.outputs
        rollout_buffer.observations[:n_rollout_steps] = np.asarray(
            outputs.observations, dtype=rollout_buffer.observations.dtype
        )
        rollout_buffer.actions[:n_rollout_steps] = np.asarray(
            outputs.actions, dtype=rollout_buffer.actions.dtype
        )
        rollout_buffer.rewards[:n_rollout_steps] = np.asarray(outputs.rewards, dtype=np.float32)
        rollout_buffer.episode_starts[:n_rollout_steps] = np.asarray(
            outputs.episode_starts, dtype=np.float32
        )
        rollout_buffer.values[:n_rollout_steps] = np.asarray(outputs.values, dtype=np.float32)
        rollout_buffer.log_probs[:n_rollout_steps] = np.asarray(outputs.log_probs, dtype=np.float32)
        rollout_buffer.pos = n_rollout_steps
        rollout_buffer.full = True

        # SBX uses np.float32 for ``_last_episode_starts``; match that
        # dtype so the next rollout's start-mask conversion is consistent
        # with what the stock collector would have produced. ``np.array``
        # (not ``np.asarray``) forces a writable copy — JAX device-backed
        # arrays come back read-only, and ``th.as_tensor`` warns on
        # non-writable arrays even though SB3 only reads from it.
        next_done_np = np.array(scan_result.next_done, dtype=np.float32)
        last_values_np = np.array(scan_result.last_values, dtype=np.float32)

        # SB3's ``compute_returns_and_advantage`` wants a torch tensor
        # (it calls ``.clone().cpu().numpy().flatten()`` internally). The
        # method is otherwise pure-numpy.
        rollout_buffer.compute_returns_and_advantage(
            last_values=th.as_tensor(last_values_np), dones=next_done_np
        )

        # ``_last_obs`` is what the next rollout's ``episode_starts[0]``
        # would be derived from if we ever fell back to the stock
        # collector — keep it consistent with the env state. Build the
        # post-rollout flat-concat obs from the wrapper.
        # The wrapper exposes ``_build_obs`` for this.
        self._last_obs = env._build_obs(env._prev_env_obs)
        self._last_episode_starts = next_done_np

        # Bump step counter and surface basic info-buffer entries that
        # SBX's logging hooks expect at iteration boundaries. SBX's
        # ``learn`` uses ``num_timesteps`` to drive log_interval / LR
        # annealing — must reflect the env steps we just collected.
        self.num_timesteps += n_rollout_steps * env.num_envs

        # v125+: aggregate rollout diagnostics into the SB3 logger so the
        # WandbScalarCallback picks them up at on_rollout_end. Per-step
        # arrays are (n_steps, n_envs) — flatten for stats.
        target_gate_arr = np.asarray(outputs.target_gate)
        terminated_arr = np.asarray(outputs.terminated)
        truncated_arr = np.asarray(outputs.truncated)
        finished_arr = np.asarray(outputs.finished)
        done_arr = terminated_arr | truncated_arr
        # ``terminated`` already includes the ``finished`` mask (set in
        # scan_step); a crash is a termination that is not a finish and
        # not a timeout.
        crash_arr = terminated_arr & ~finished_arr & ~truncated_arr

        self.logger.record("env/max_target_gate", int(target_gate_arr.max()))
        self.logger.record("env/mean_target_gate", float(target_gate_arr.mean()))
        n_episodes = int(done_arr.sum())
        self.logger.record("env/n_episodes_in_rollout", n_episodes)
        if n_episodes > 0:
            self.logger.record("env/finish_rate", float(finished_arr.sum() / n_episodes))
            self.logger.record("env/crash_rate", float(crash_arr.sum() / n_episodes))

        # Per-source finish/crash breakdown. The Phase-2 replay buffer
        # respawns drones from successful past states near the back of the
        # track, so a single aggregate ``env/finish_rate`` can be dominated
        # by easy mid-track finishes while true-start performance stays at
        # zero. Breaking this out makes the gap visible during training.
        # See rl_song.rollout for the SRC_* constants (int8 codes).
        source_arr = np.asarray(outputs.source)
        for source_code, source_name in (
            (SRC_TRUE_START, "true_start"),
            (SRC_PHASE1_SEG, "phase1_seg"),
            (SRC_PHASE2_REPLAY, "phase2_replay"),
        ):
            source_mask = source_arr == source_code
            n_done_source = int((done_arr & source_mask).sum())
            self.logger.record(f"env/n_episodes_{source_name}", n_done_source)
            if n_done_source > 0:
                self.logger.record(
                    f"env/finish_rate_{source_name}",
                    float((finished_arr & source_mask).sum() / n_done_source),
                )
                self.logger.record(
                    f"env/crash_rate_{source_name}",
                    float((crash_arr & source_mask).sum() / n_done_source),
                )

        # Phase-2 replay-buffer composition diagnostics. Slot 0 is unused;
        # slots >= 1 hold successful gate-pass states for ``target_gate == g``.
        # Tests the calm-pass-bias hypothesis: at low omega_coef the buffer
        # may be skewed toward survivable-slow gate transitions, biasing the
        # actor toward calm continuation via replay (see codex review
        # 2026-05-26). Lower v130-vs-v128 ``agg_ang_vel_norm_mean`` would
        # support this.
        phase2_buffer = scan_result.phase2_buffer
        phase2_data = phase2_buffer.data
        phase2_fill = phase2_buffer.fill
        n_gates = int(phase2_data.shape[0])
        capacity = int(phase2_data.shape[1])
        row_idx = jnp.arange(capacity)
        fill_host = np.asarray(phase2_fill)

        for gate_idx in range(1, n_gates):
            fill_g_host = int(fill_host[gate_idx])
            if fill_g_host <= 0:
                continue

            fill_g = phase2_fill[gate_idx]
            denom_g = jnp.maximum(fill_g, 1).astype(jnp.float32)
            valid_g = row_idx < fill_g
            data_g = phase2_data[gate_idx]

            vel_norm_g = jnp.linalg.norm(data_g[:, 3:6], axis=-1)
            ang_vel_norm_g = jnp.linalg.norm(data_g[:, 10:13], axis=-1)
            altitude_g = data_g[:, 2]
            prev_thrust_g = data_g[:, 13]

            vel_mean_g = jnp.sum(jnp.where(valid_g, vel_norm_g, 0.0)) / denom_g
            ang_vel_mean_g = jnp.sum(jnp.where(valid_g, ang_vel_norm_g, 0.0)) / denom_g
            vel_std_g = jnp.sqrt(
                jnp.sum(jnp.where(valid_g, (vel_norm_g - vel_mean_g) ** 2, 0.0)) / denom_g
            )
            ang_vel_std_g = jnp.sqrt(
                jnp.sum(jnp.where(valid_g, (ang_vel_norm_g - ang_vel_mean_g) ** 2, 0.0)) / denom_g
            )

            self.logger.record(f"phase2_buffer/g{gate_idx}/fill", fill_g_host)
            self.logger.record(
                f"phase2_buffer/g{gate_idx}/vel_norm_mean", float(np.asarray(vel_mean_g))
            )
            self.logger.record(
                f"phase2_buffer/g{gate_idx}/vel_norm_std", float(np.asarray(vel_std_g))
            )
            self.logger.record(
                f"phase2_buffer/g{gate_idx}/ang_vel_norm_mean", float(np.asarray(ang_vel_mean_g))
            )
            self.logger.record(
                f"phase2_buffer/g{gate_idx}/ang_vel_norm_std", float(np.asarray(ang_vel_std_g))
            )
            self.logger.record(
                f"phase2_buffer/g{gate_idx}/altitude_mean",
                float(np.asarray(jnp.sum(jnp.where(valid_g, altitude_g, 0.0)) / denom_g)),
            )
            self.logger.record(
                f"phase2_buffer/g{gate_idx}/prev_thrust_mean",
                float(np.asarray(jnp.sum(jnp.where(valid_g, prev_thrust_g, 0.0)) / denom_g)),
            )

        gate_idx_all = jnp.arange(n_gates)[:, None]
        row_idx_all = jnp.arange(capacity)[None, :]
        valid_all = (gate_idx_all > 0) & (row_idx_all < phase2_fill[:, None])
        denom_all = jnp.maximum(jnp.sum(valid_all), 1).astype(jnp.float32)
        agg_vel = jnp.linalg.norm(phase2_data[:, :, 3:6], axis=-1)
        agg_ang_vel = jnp.linalg.norm(phase2_data[:, :, 10:13], axis=-1)
        self.logger.record("phase2_buffer/total_fill", int(fill_host[1:].sum()))
        self.logger.record(
            "phase2_buffer/agg_vel_norm_mean",
            float(np.asarray(jnp.sum(jnp.where(valid_all, agg_vel, 0.0)) / denom_all)),
        )
        self.logger.record(
            "phase2_buffer/agg_ang_vel_norm_mean",
            float(np.asarray(jnp.sum(jnp.where(valid_all, agg_ang_vel, 0.0)) / denom_all)),
        )
        self.logger.record(
            "phase2_buffer/agg_altitude_mean",
            float(np.asarray(jnp.sum(jnp.where(valid_all, phase2_data[:, :, 2], 0.0)) / denom_all)),
        )
        self.logger.record(
            "phase2_buffer/agg_prev_thrust_mean",
            float(
                np.asarray(jnp.sum(jnp.where(valid_all, phase2_data[:, :, 13], 0.0)) / denom_all)
            ),
        )

        for name, comp_arr in outputs.reward_components.items():
            self.logger.record(f"reward/{name}", float(np.asarray(comp_arr).mean()))

        callback.on_rollout_end()

        if profile:
            t_exit = time.perf_counter()
            self.logger.record("time/prof_host_s", t_exit - t_scan_done)
            self._prof_last_exit = t_exit
        return True

    @staticmethod
    @partial(jax.jit, static_argnames=["normalize_advantage", "share_features_extractor"])
    def _one_update_clipped_vf(
        actor_state: TrainState,
        vf_state: TrainState,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
        old_log_prob: jnp.ndarray,
        old_values: jnp.ndarray,
        clip_range: float,
        clip_range_vf: float,
        ent_coef: float,
        vf_coef: float,
        normalize_advantage: bool = True,
        share_features_extractor: bool = False,
    ) -> tuple[
        tuple[TrainState, TrainState],
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ]:
        """Minibatch PPO update with pessimistic clipped value loss.

        Mirror of :meth:`sbx.PPO._one_update` (``sbx/ppo/ppo.py:206-268``)
        with the critic loss replaced by the Schulman pessimistic clipped
        form used in :func:`lsy_drone_racing.control.rl_song.train` (lines
        941-951). The policy update is byte-identical to stock SBX so
        ``ratio``, ``clip_fraction``, and ``approx_kl`` downstream metrics
        are comparable to historical SBX runs.

        Parameters
        ----------
        actor_state, vf_state : TrainState
            Flax train states for actor and critic. Returned updated.
        observations : ndarray, shape (batch, obs_dim)
            Minibatch observations (already normalized upstream).
        actions : ndarray, shape (batch, action_dim)
            Minibatch actions sampled by the rollout-time policy.
        advantages : ndarray, shape (batch,)
            GAE(λ) advantages from the rollout buffer.
        returns : ndarray, shape (batch,)
            GAE(λ) returns from the rollout buffer (advantages + V_old).
        old_log_prob : ndarray, shape (batch,)
            Log-probability under the rollout-time policy. Used in the PPO
            ratio.
        old_values : ndarray, shape (batch,)
            Critic predictions at rollout time. Used as the clipping center
            for the pessimistic value loss — without it the critic can move
            arbitrarily far per minibatch (the unclipped MSE failure mode
            this patch addresses).
        clip_range : float
            Policy ratio clip ε in ``clip(ratio, 1-ε, 1+ε)``.
        clip_range_vf : float
            Value-function clip ε in ``V_old + clip(V_new − V_old, −ε, +ε)``.
            Matches ``clip_range`` per rl_song convention.
        ent_coef : float
            Entropy bonus coefficient.
        vf_coef : float
            Critic-loss coefficient (composed with
            :data:`VALUE_LOSS_SCALE`).
        normalize_advantage : bool, optional
            If True, standard-scale advantages within the minibatch (the
            stock SBX default; skipped when the batch is a single sample).
        share_features_extractor : bool, optional
            CNN-policy shared-features hack from stock SBX. Always False in
            our use; kept on the signature so the JIT cache key matches
            stock SBX's argument list.

        Returns:
        -------
        (actor_state, vf_state) : tuple of TrainState
            Updated Flax train states.
        (pg_loss, policy_loss, entropy_loss, vf_loss, ratio) : tuple of arrays
            Same five diagnostics SBX returns; downstream logger code is
            unchanged.

        Notes:
        -----
        The pessimistic max ensures the critic cannot reduce loss by jumping
        far from ``old_values`` in a direction that would otherwise be
        advantageous — exactly the property that bounds critic movement
        per minibatch and (per the redesign brief 2026-05-27) prevents the
        action-saturation cascade the SBX-only runs v112-v133 hit. Same
        construction as ``rl_song/train.py:943-950``.
        """
        if normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def actor_loss(params: dict[str, Any]) -> tuple[jnp.ndarray, tuple]:
            dist = actor_state.apply_fn(params, observations)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()

            ratio = jnp.exp(log_prob - old_log_prob)
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
            policy_loss = -jnp.minimum(policy_loss_1, policy_loss_2).mean()
            entropy_loss = -jnp.mean(entropy)

            total_policy_loss = policy_loss + ent_coef * entropy_loss
            return total_policy_loss, (ratio, policy_loss, entropy_loss)

        (pg_loss_value, (ratio, policy_loss, entropy_loss)), grads = jax.value_and_grad(
            actor_loss, has_aux=True
        )(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        def critic_loss(params: dict[str, Any]) -> jnp.ndarray:
            vf_values = vf_state.apply_fn(params, observations).flatten()
            value_loss_unclipped = jnp.square(vf_values - returns)
            v_clipped = old_values + jnp.clip(vf_values - old_values, -clip_range_vf, clip_range_vf)
            value_loss_clipped = jnp.square(v_clipped - returns)
            value_loss = VALUE_LOSS_SCALE * jnp.mean(
                jnp.maximum(value_loss_unclipped, value_loss_clipped)
            )
            return vf_coef * value_loss

        vf_loss_value, grads = jax.value_and_grad(critic_loss, has_aux=False)(vf_state.params)
        vf_state = vf_state.apply_gradients(grads=grads)

        return (actor_state, vf_state), (
            pg_loss_value,
            policy_loss,
            entropy_loss,
            vf_loss_value,
            ratio,
        )

    def train(self) -> None:
        """Run PPO updates with pessimistic clipped value loss.

        Overrides :meth:`sbx.PPO.train` (``sbx/ppo/ppo.py:270-358``) to
        thread ``old_values`` from the rollout buffer and a value-function
        clip range into :meth:`_one_update_clipped_vf`. All other training
        machinery (learning-rate schedule, n_epochs / batch_size loop,
        adaptive-LR target-KL handling, explained-variance logging) is
        kept byte-identical to stock SBX so historical training metrics
        remain comparable.

        Raises:
        ------
        TypeError
            If :attr:`self.clip_range_vf` is set to a callable (a schedule).
            Only scalar ``float`` is supported; pass the same value as
            :attr:`self.clip_range` to match rl_song convention.
        """
        # Stock SBX gates LR updates on ``target_kl is None`` and otherwise
        # lets the adaptive-LR object drive LR per-minibatch. We mirror that
        # control flow exactly so the override stays a drop-in replacement.
        if self.target_kl is None:
            self._update_learning_rate(
                [self.policy.actor_state.opt_state[1], self.policy.vf_state.opt_state[1]],
                learning_rate=self.lr_schedule(self._current_progress_remaining),
            )
        clip_range = self.clip_range_schedule(self._current_progress_remaining)

        # Resolve ``clip_range_vf`` to a scalar. ``None`` defaults to the
        # policy ratio clip (rl_song convention — see
        # ``rl_song/train.py:925, 945``: both clips use ``ppo_cfg.clip_coef``).
        # Schedule (callable) is rejected to keep the JIT signature stable;
        # if we ever need a schedule, plumb it through analogously to
        # ``clip_range``.
        if self.clip_range_vf is None:
            clip_range_vf_value = clip_range
        elif callable(self.clip_range_vf):
            raise TypeError(
                "JitScanPPO requires clip_range_vf as a scalar float, not a "
                "schedule. Pass a constant or leave it None to default to clip_range."
            )
        else:
            clip_range_vf_value = float(self.clip_range_vf)

        n_updates = 0
        mean_clip_fraction = 0.0
        mean_kl_div = 0.0
        pg_loss = policy_loss = entropy_loss = value_loss = None
        ratio = None

        for _ in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                n_updates += 1
                # Box action space only — JitScanPPO is the only consumer
                # and its env wrapper exposes ``Box(RAW_ACTION_DIM)``. The
                # stock SBX ``isinstance(..., spaces.Discrete)`` branch is
                # dead in our pipeline and omitted here.
                actions = rollout_data.actions.numpy()

                (
                    (self.policy.actor_state, self.policy.vf_state),
                    (pg_loss, policy_loss, entropy_loss, value_loss, ratio),
                ) = self._one_update_clipped_vf(
                    actor_state=self.policy.actor_state,
                    vf_state=self.policy.vf_state,
                    observations=rollout_data.observations.numpy(),
                    actions=actions,
                    advantages=rollout_data.advantages.numpy(),
                    returns=rollout_data.returns.numpy(),
                    old_log_prob=rollout_data.old_log_prob.numpy(),
                    old_values=rollout_data.old_values.numpy().flatten(),
                    clip_range=clip_range,
                    clip_range_vf=clip_range_vf_value,
                    ent_coef=self.ent_coef,
                    vf_coef=self.vf_coef,
                    normalize_advantage=self.normalize_advantage,
                    share_features_extractor=False,
                )

                # Approximate reverse KL for diagnostics / adaptive LR (see
                # Schulman http://joschu.net/blog/kl-approx.html). Same
                # estimator stock SBX uses at sbx/ppo/ppo.py:320-321.
                approx_kl_div = jnp.mean(
                    (ratio - 1.0 + APPROX_KL_RATIO_EPS) - jnp.log(ratio + APPROX_KL_RATIO_EPS)
                ).item()
                clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > clip_range).item()
                mean_clip_fraction += (clip_fraction - mean_clip_fraction) / n_updates
                mean_kl_div += (approx_kl_div - mean_kl_div) / n_updates

                if self.target_kl is not None:
                    self.adaptive_lr.update(approx_kl_div)
                    self._update_learning_rate(
                        [self.policy.actor_state.opt_state[1], self.policy.vf_state.opt_state[1]],
                        learning_rate=self.adaptive_lr.current_adaptive_lr,
                    )

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Last-minibatch scalars (stock SBX behaviour, see comment at
        # sbx/ppo/ppo.py:342). Replace with epoch-mean if any downstream
        # callback needs a smoother signal.
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_gradient_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/approx_kl", mean_kl_div)
        self.logger.record("train/clip_fraction", mean_clip_fraction)
        self.logger.record("train/pg_loss", pg_loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        # Stock SBX has a commented-out clip_range_vf log line because it
        # never wired the value clip through; we always log it since
        # clip_range_vf is now load-bearing.
        self.logger.record("train/clip_range_vf", clip_range_vf_value)

    def _next_rollout_rng_key(self) -> jnp.ndarray:
        """Return a fresh PRNG key for the next rollout scan.

        Splits :attr:`self.key` (SBX's policy noise key) and stores the
        residual back, so subsequent calls are deterministic given the
        ``seed`` argument to :meth:`PPO.__init__`. SBX uses the policy's
        ``noise_key`` for action sampling in the stock collector; here
        we route through our own key for the scan and leave SBX's noise
        key untouched.
        """
        import jax

        # ``self.key`` is set by ``PPOPolicy.build`` via ``self.policy.key``
        # — but SBX stores it on the algo object too via ``set_random_seed``.
        # Fall back to splitting the policy's noise_key if the algo-level
        # key isn't present (older SBX revisions).
        if hasattr(self, "_jit_scan_key"):
            key = self._jit_scan_key
        else:
            # First call: seed from the policy's noise key so a fixed
            # PPO seed produces a deterministic scan trace.
            key = self.policy.noise_key
        key, sub = jax.random.split(key)
        self._jit_scan_key = key
        return sub

    def _next_reset_rng_key(self) -> jnp.ndarray:
        """Return a fresh PRNG key for the in-scan seg-init / perturbation.

        Maintains a stream independent of the action-sampling key so a
        stage that toggles seg-init does not bit-shift the policy's
        exploration trajectory. First-call bootstrap derives from the
        policy noise key via a single ``jax.random.fold_in`` so a fixed
        PPO seed still produces a deterministic reset stream.
        """
        import jax

        if hasattr(self, "_jit_reset_key"):
            key = self._jit_reset_key
        else:
            # Fold a distinct domain tag into the policy noise key. The
            # raw seed is reused only as the bootstrap source; once
            # _jit_reset_key is stored we split it independently from
            # _jit_scan_key on every subsequent rollout.
            key = jax.random.fold_in(self.policy.noise_key, 0xFEEDBEEF)
        key, sub = jax.random.split(key)
        self._jit_reset_key = key
        return sub
