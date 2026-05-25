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

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import torch as th
from sbx import PPO

from lsy_drone_racing.control.rl_sbx.rollout import (
    RLSBXScanResult,
    make_static_config,
    scan_rollout,
)
from lsy_drone_racing.envs.race_core import obs as race_core_obs

if TYPE_CHECKING:
    from stable_baselines3.common.buffers import RolloutBuffer
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecEnv

# rl_sbx env wrapper has a single drone per vec env; mirrors
# :data:`rl_sbx.rollout.SINGLE_DRONE_INDEX`.
SINGLE_DRONE_INDEX: int = 0


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
        )

        # ``_last_episode_starts`` lives on SBX as a float32 / bool numpy
        # array. The scan needs a JAX bool of shape (n_envs,) — convert
        # explicitly so a stale dtype on the SBX side doesn't propagate.
        next_done_jax = jnp.asarray(self._last_episode_starts, dtype=jnp.bool_)

        # Advance the wrapper's RNG once per rollout. The scan does its
        # own per-step splits internally; we only need one fresh key
        # per dispatch.
        rng_key = self._next_rollout_rng_key()

        scan_result: RLSBXScanResult = scan_rollout(
            env.jax_env.data,
            self.policy.actor_state.params,
            self.policy.vf_state.params,
            env.actor_normalizer,
            env.critic_normalizer,
            env._prev_action,
            rng_key,
            next_done_jax,
            env.jax_env._step,
            env.jax_env._reset,
            static_cfg,
        )

        # Round-trip env state. The wrapper's ``_prev_env_obs`` is only
        # consumed by ``step_wait``; refreshing it here keeps the wrapper
        # in a valid state if external code calls ``step_wait`` between
        # iterations (e.g. a manual eval interleaved with training).
        env.jax_env.data = scan_result.env_data
        env._prev_action = scan_result.prev_action_env_4vec
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

        callback.on_rollout_end()
        return True

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
