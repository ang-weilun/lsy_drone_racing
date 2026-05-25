"""JAX-scanned PPO rollout collection for the rl_sbx stack.

Compiled-once collector that runs the env + actor + critic forward
``n_steps`` times inside a single :func:`jax.lax.scan`. Eliminates SBX's
per-step host-loop dispatch overhead (~75k env-steps/s with stock
``sbx.PPO.collect_rollouts``) and restores rl_song-class throughput
(~250k-500k+ env-steps/s).

Architecture parity with :mod:`lsy_drone_racing.control.rl_song.rollout`:

* same race-core ``_step`` / ``_reset`` calls,
* same actor sampling + critic evaluation contract,
* same masked-geometry ``step_reward`` (v85 fix — no ``true_*`` kwargs).

Architecture diff vs rl_song:

* **flat-concat observations** — the env wrapper packs
  ``[actor (ACTOR_OBS_DIM) | critic (ACTOR_OBS_DIM)]`` into a single
  ``(n_envs, 2*ACTOR_OBS_DIM)`` array so the SBX PPO loss code, which
  reads observations as a single tensor, works unmodified. The scan
  body builds both halves with ``obs_encoding.vmap_build_*`` and
  ``jnp.concatenate``-s along the feature axis.
* **tfd-returning Actor** — the SBX-style :class:`Actor` returns a
  :class:`tfd.MultivariateNormalDiag`. We call ``dist.sample`` /
  ``dist.log_prob`` here instead of rl_song's
  ``sample_and_log_prob(actor_params, ...)`` helper.
* **no seg-init / no Phase 2 / no per-source tracking** — milestone-1
  scope is the throughput win on the baseline reward. Done envs are
  reset via ``env_reset_fn(data, seed=None, mask=done)`` only. The
  ``_reset_done_worlds`` hook from rl_song is left for a follow-up
  commit; see the spec at ``docs/specs/2026-05-24-sbx-migration-design.md``.
* **no in-scan timeout bootstrap** — SBX's stock collector mutates
  ``rewards[idx] += gamma * V(s_terminal)`` per truncated env in
  Python. We skip that for milestone-1 (small bias on truncating
  episodes); see :class:`~lsy_drone_racing.control.rl_sbx.jit_scan_ppo.JitScanPPO`
  for the rationale.

References:
----------
Song, Y. et al. (2023). Reaching the limit in autonomous racing.
    *Science Robotics* 8, eadg1462.
Stable Baselines Jax (SBX), https://github.com/araffin/sbx.
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from lsy_drone_racing.control.rl_sbx.policy import LOG_STD_INIT, NET_ARCH, Actor, Critic
from lsy_drone_racing.control.rl_song import obs as obs_encoding
from lsy_drone_racing.control.rl_song.config import (
    ACTOR_OBS_DIM,
    ENV_ACTION_DIM,
    RAW_ACTION_DIM,
    TANGENT_ALPHA_MAX_RAD,
    RewardConfig,
)
from lsy_drone_racing.control.rl_song.policy import raw_to_env_action
from lsy_drone_racing.control.rl_song.reward import step_reward
from lsy_drone_racing.envs.race_core import EnvData
from lsy_drone_racing.envs.race_core import obs as race_core_obs

# Single-drone simulation; the racing env's drone axis is size 1. The vec axis
# is the per-env axis. Mirrors :data:`rl_song.rollout.SINGLE_DRONE_INDEX`.
SINGLE_DRONE_INDEX: int = 0

# Total flat-concat obs dim. Matches :data:`rl_sbx.policy.FLAT_CONCAT_OBS_DIM`
# but re-derived locally to avoid an extra import dependency for what is a
# trivial constant.
FLAT_CONCAT_OBS_DIM: int = 2 * ACTOR_OBS_DIM


EnvStepFn = Any  # Callable[[EnvData, Array], tuple[EnvData, tuple[...]]]
EnvResetFn = Any  # Callable[[EnvData, int | None, Array | None], tuple[EnvData, ...]]


class RLSBXRolloutStaticConfig(NamedTuple):
    """Static configuration for a compiled rollout scan.

    Carried via :func:`functools.partial`'s ``static_argnames`` so the scan
    is re-traced only when one of these fields changes. ``RewardConfig`` is
    a frozen dataclass — hashable and JAX-safe.

    Parameters
    ----------
    n_steps : int
        Number of environment steps collected per PPO rollout.
    n_envs : int
        Number of vectorized racing worlds.
    thrust_min, thrust_max : float
        Total-thrust bounds in newtons used by the raw-to-env action
        projection.
    tangent_alpha_max_rad : float
        Per-step rotation budget on ``‖τ_scaled‖`` (rad). One source of
        truth for the env-action projection inside :func:`scan_rollout`.
    reward_cfg : RewardConfig
        Reward weights for :func:`reward.step_reward`.
    """

    n_steps: int
    n_envs: int
    thrust_min: float
    thrust_max: float
    tangent_alpha_max_rad: float
    reward_cfg: RewardConfig


class RLSBXRolloutOutputs(NamedTuple):
    """Stacked transitions from one scan rollout.

    Shapes follow SB3's :class:`stable_baselines3.common.buffers.RolloutBuffer`
    convention: the leading axis is time, then env, then the per-feature axis
    (if any). Fields map one-for-one to the buffer's ``self.observations``,
    ``self.actions``, ``self.rewards``, ``self.episode_starts``,
    ``self.values``, and ``self.log_probs``; see
    :meth:`~lsy_drone_racing.control.rl_sbx.jit_scan_ppo.JitScanPPO.collect_rollouts`
    for the bulk write.

    Note:
    ----
    SB3's :meth:`RolloutBuffer.compute_returns_and_advantage` consumes
    ``episode_starts`` (NOT ``dones``) for the GAE recursion's
    ``next_non_terminal`` mask. ``episode_starts[t]`` is the previous
    step's done flag — i.e. ``True`` iff the env was reset at the end of
    step ``t-1`` and therefore starts a fresh episode at step ``t``. The
    value at ``t=0`` comes from the caller's ``next_done`` carried across
    rollout boundaries.
    """

    observations: Array  # (n_steps, n_envs, 2*ACTOR_OBS_DIM)
    actions: Array  # (n_steps, n_envs, RAW_ACTION_DIM)
    rewards: Array  # (n_steps, n_envs)
    episode_starts: Array  # (n_steps, n_envs)  float32
    values: Array  # (n_steps, n_envs)
    log_probs: Array  # (n_steps, n_envs)


class RLSBXScanResult(NamedTuple):
    """Output of one compiled rollout scan.

    Fields
    ------
    env_data : EnvData
        Race-core state after the in-scan autoresets, ready to feed the
        next rollout.
    actor_normalizer, critic_normalizer : NormalizerState
        Unchanged by the scan; passed through so the call site has a
        single return value with everything it needs to round-trip env
        state. The normalizers are updated *after* the rollout by the
        :class:`~lsy_drone_racing.control.rl_sbx.callbacks.NormalizerUpdateCallback`,
        which reads the post-write rollout buffer.
    prev_action_env_4vec : Array, shape (n_envs, 4)
        Previous env-boundary action ``[roll, pitch, yaw, thrust]``.
        Zeroed on done envs inside the scan.
    rng_key : Array, shape (2,)
        Updated policy-sampling PRNG key.
    next_done : Array, shape (n_envs,), bool
        Done flag at the end of the rollout. Becomes the next rollout's
        ``episode_starts[0]`` (after the float cast).
    last_values : Array, shape (n_envs,)
        ``V(s_{T+1})`` for the GAE bootstrap term in
        :meth:`RolloutBuffer.compute_returns_and_advantage`.
    outputs : RLSBXRolloutOutputs
        Stacked transition tensors for the PPO buffer.
    """

    env_data: EnvData
    actor_normalizer: obs_encoding.NormalizerState
    critic_normalizer: obs_encoding.NormalizerState
    prev_action_env_4vec: Array
    rng_key: Array
    next_done: Array
    last_values: Array
    outputs: RLSBXRolloutOutputs


class _ScanCarry(NamedTuple):
    """Mutable scan carry. Kept minimal — no per-source / phase2 state."""

    env_data: EnvData
    prev_action: Array  # env-action 4-vec, (n_envs, ENV_ACTION_DIM)
    rng_key: Array
    next_done: Array  # bool, (n_envs,) — done at end of previous step


def _build_flat_obs(
    env_obs: dict[str, Array],
    prev_action: Array,
    env_data: EnvData,
    actor_normalizer: obs_encoding.NormalizerState,
    critic_normalizer: obs_encoding.NormalizerState,
) -> Array:
    """Encode the flat-concat ``[actor | critic]`` observation.

    Mirrors :meth:`RLSBXVecEnv._build_obs` but expressed in pure JAX so the
    scan can trace through it. The actor half is normalized by
    ``actor_normalizer``; the critic half is normalized by
    ``critic_normalizer``; the two halves are concatenated along the
    feature axis.

    Parameters
    ----------
    env_obs : dict[str, Array]
        Drone-axis-squeezed race-core obs (each leaf is ``(n_envs, ...)``).
    prev_action : Array, shape (n_envs, ENV_ACTION_DIM)
        Previous env-boundary action.
    env_data : EnvData
        Race-core state. Used to source unmasked true gate / obstacle
        poses for the critic half (asymmetric AC privilege).
    actor_normalizer, critic_normalizer : NormalizerState
        Running statistics for each half.

    Returns:
    -------
    Array, shape (n_envs, 2*ACTOR_OBS_DIM)
        Flat-concat observation.
    """
    actor_half = obs_encoding.vmap_build_actor_obs(env_obs, prev_action, actor_normalizer)
    critic_half = obs_encoding.vmap_build_critic_obs(
        env_obs,
        prev_action,
        critic_normalizer,
        true_gates_pos=env_data.gates_pos,
        true_gates_quat=env_data.gates_quat,
        true_obstacles_pos=env_data.obstacles_pos,
    )
    return jnp.concatenate([actor_half, critic_half], axis=-1)


def _single_drone_obs(env_obs: dict[str, Array]) -> dict[str, Array]:
    """Squeeze the single-drone axis from a race-core obs dict."""
    return {key: value[:, SINGLE_DRONE_INDEX] for key, value in env_obs.items()}


@partial(jax.jit, static_argnames=("env_step_fn", "env_reset_fn", "static_cfg"))
def scan_rollout(
    env_data: EnvData,
    actor_params: Any,
    vf_params: Any,
    actor_normalizer: obs_encoding.NormalizerState,
    critic_normalizer: obs_encoding.NormalizerState,
    prev_action_env_4vec: Array,
    rng_key: Array,
    next_done: Array,
    env_step_fn: EnvStepFn,
    env_reset_fn: EnvResetFn,
    static_cfg: RLSBXRolloutStaticConfig,
) -> RLSBXScanResult:
    """Collect one PPO rollout inside a single JAX dispatch.

    Parameters
    ----------
    env_data : EnvData
        Current JAX-pure race-core state.
    actor_params, vf_params : pytree
        Flax parameter pytrees for the SBX-style :class:`Actor` and
        :class:`Critic`.
    actor_normalizer, critic_normalizer : NormalizerState
        Running observation normalizers. Unchanged by the scan; updates
        happen post-rollout from the populated buffer via
        :class:`NormalizerUpdateCallback`.
    prev_action_env_4vec : Array, shape (n_envs, ENV_ACTION_DIM)
        Previous env-boundary action ``[roll, pitch, yaw, thrust]``.
    rng_key : Array, shape (2,)
        PRNG key used for raw-action sampling.
    next_done : Array, shape (n_envs,), bool
        Done flag carried from the previous rollout. Becomes the
        ``episode_starts[0]`` of this rollout.
    env_step_fn : callable
        JIT-compatible ``RaceCoreEnv._step``.
    env_reset_fn : callable
        JIT-compatible ``RaceCoreEnv._reset``.
    static_cfg : RLSBXRolloutStaticConfig
        Static rollout, reward, and action-projection configuration.

    Returns:
    -------
    RLSBXScanResult
        Updated env state, RNG key, stacked rollout buffers, and the
        bootstrap value for GAE.

    Notes:
    -----
    Done envs are autoreset inside the scan via
    ``env_reset_fn(stepped_data, seed=None, mask=done_bool)``. No
    perturbation / Phase-1 seg-init / Phase-2 replay is applied — that
    hook is reserved for a follow-up commit (see module docstring).
    """
    _validate_inputs(prev_action_env_4vec, next_done, static_cfg)

    actor = Actor(
        action_dim=RAW_ACTION_DIM, net_arch=NET_ARCH, log_std_init=LOG_STD_INIT, ortho_init=False
    )
    critic = Critic(net_arch=NET_ARCH)

    def scan_step(carry: _ScanCarry, _: None) -> tuple[_ScanCarry, RLSBXRolloutOutputs]:
        env_obs = _single_drone_obs(race_core_obs(carry.env_data))
        flat_obs = _build_flat_obs(
            env_obs, carry.prev_action, carry.env_data, actor_normalizer, critic_normalizer
        )

        rng_key, sample_key = jax.random.split(carry.rng_key)
        dist = actor.apply(actor_params, flat_obs)
        raw_action = dist.sample(seed=sample_key)
        log_prob = dist.log_prob(raw_action)
        # ``Critic.__call__`` returns shape (..., 1); squeeze the trailing
        # singleton so the stacked buffer matches SB3's (n_steps, n_envs).
        value = critic.apply(vf_params, flat_obs).squeeze(-1)

        env_action = raw_to_env_action(
            raw_action,
            env_obs["quat"],
            static_cfg.thrust_min,
            static_cfg.thrust_max,
            alpha_max=static_cfg.tangent_alpha_max_rad,
        )

        stepped_data, (next_obs_full, _, term_full, trunc_full, _) = env_step_fn(
            carry.env_data, env_action
        )
        next_env_obs = _single_drone_obs(next_obs_full)
        terminated = term_full[:, SINGLE_DRONE_INDEX].astype(jnp.bool_)
        truncated = trunc_full[:, SINGLE_DRONE_INDEX].astype(jnp.bool_)

        # ``finished`` is the lap-complete signal (target_gate sentinel ==
        # -1). ``gate_just_passed`` is the per-step gate-bonus mask used
        # by ``step_reward``. Mirrors rl_song's scan body.
        current_target = next_env_obs["target_gate"]
        previous_target = env_obs["target_gate"]
        finished = current_target < 0
        terminated = terminated | finished
        gate_just_passed = ((current_target > previous_target) & (previous_target >= 0)) | (
            finished & (previous_target >= 0)
        )

        # Masked-geometry reward (v85): no ``true_*`` kwargs. The reward
        # sees only the actor's observation so gradients can't exploit
        # randomization the policy is denied at deploy.
        reward, _components = step_reward(
            next_env_obs,
            env_obs,
            terminated,
            truncated,
            finished,
            gate_just_passed,
            static_cfg.reward_cfg,
        )

        done_bool = terminated | truncated

        # Autoreset done envs. ``env_reset_fn`` is wrapped so a no-op
        # mask leaves the data untouched; the unconditional call keeps
        # the scan trace shape-stable.
        reset_data, _ = env_reset_fn(stepped_data, None, done_bool)

        # Zero ``prev_action`` on done envs so the next episode starts
        # with the same "no prior command" condition as a true reset.
        next_prev_action = jnp.where(done_bool[:, None], jnp.zeros_like(env_action), env_action)

        transition = RLSBXRolloutOutputs(
            observations=flat_obs,
            actions=raw_action,
            rewards=reward,
            # SB3 GAE uses ``episode_starts`` not ``dones`` (see
            # ``buffers.py`` ``compute_returns_and_advantage``). The flag
            # at step t is the previous-step done bool: True iff the env
            # was reset between step t-1 and step t.
            episode_starts=carry.next_done.astype(jnp.float32),
            values=value,
            log_probs=log_prob,
        )
        next_carry = _ScanCarry(
            env_data=reset_data, prev_action=next_prev_action, rng_key=rng_key, next_done=done_bool
        )
        return next_carry, transition

    initial_carry = _ScanCarry(
        env_data=env_data, prev_action=prev_action_env_4vec, rng_key=rng_key, next_done=next_done
    )
    final_carry, stacked_outputs = jax.lax.scan(
        scan_step, initial_carry, None, length=static_cfg.n_steps
    )

    # Bootstrap value V(s_{T+1}) for GAE on the post-reset env state.
    # SB3's ``compute_returns_and_advantage`` uses this as ``last_values``
    # alongside ``next_done`` (the rollout's final done flag) to seed
    # the GAE recursion at step ``buffer_size - 1``.
    final_env_obs = _single_drone_obs(race_core_obs(final_carry.env_data))
    final_flat_obs = _build_flat_obs(
        final_env_obs,
        final_carry.prev_action,
        final_carry.env_data,
        actor_normalizer,
        critic_normalizer,
    )
    last_values = critic.apply(vf_params, final_flat_obs).squeeze(-1)

    return RLSBXScanResult(
        env_data=final_carry.env_data,
        actor_normalizer=actor_normalizer,
        critic_normalizer=critic_normalizer,
        prev_action_env_4vec=final_carry.prev_action,
        rng_key=final_carry.rng_key,
        next_done=final_carry.next_done,
        last_values=last_values,
        outputs=stacked_outputs,
    )


def make_static_config(
    *,
    n_steps: int,
    n_envs: int,
    thrust_min: float,
    thrust_max: float,
    reward_cfg: RewardConfig,
    tangent_alpha_max_rad: float = TANGENT_ALPHA_MAX_RAD,
) -> RLSBXRolloutStaticConfig:
    """Construct a :class:`RLSBXRolloutStaticConfig` with the project defaults.

    Thin convenience constructor — keeps call sites in ``train.py`` /
    ``jit_scan_ppo.py`` from re-stating the default ``tangent_alpha_max_rad``
    every time.
    """
    return RLSBXRolloutStaticConfig(
        n_steps=int(n_steps),
        n_envs=int(n_envs),
        thrust_min=float(thrust_min),
        thrust_max=float(thrust_max),
        tangent_alpha_max_rad=float(tangent_alpha_max_rad),
        reward_cfg=reward_cfg,
    )


def _validate_inputs(
    prev_action_env_4vec: Array, next_done: Array, static_cfg: RLSBXRolloutStaticConfig
) -> None:
    """Validate static rollout input shapes before tracing the scan."""
    n_envs = static_cfg.n_envs
    if prev_action_env_4vec.shape != (n_envs, ENV_ACTION_DIM):
        raise ValueError(
            "prev_action_env_4vec must have shape "
            f"{(n_envs, ENV_ACTION_DIM)}; got {prev_action_env_4vec.shape}"
        )
    if next_done.shape != (n_envs,):
        raise ValueError(f"next_done must have shape {(n_envs,)}; got {next_done.shape}")
