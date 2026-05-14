"""JAX-scanned PPO rollout collection for the Song-2023 controller."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from lsy_drone_racing.control.rl_song import obs as obs_encoding
from lsy_drone_racing.control.rl_song.config import (
    ENV_ACTION_DIM,
    RewardConfig,
)
from lsy_drone_racing.control.rl_song.policy import (
    Critic,
    raw_to_env_action,
    sample_and_log_prob,
)
from lsy_drone_racing.control.rl_song.reward import step_reward
from lsy_drone_racing.envs.race_core import (
    EnvData,
    _reset_env_data,
    obs as race_core_obs,
)

SINGLE_DRONE_INDEX: int = 0
RESET_RNG_SPLITS: int = 4
YAW_TO_HALF_ANGLE: float = 0.5  # quaternion half-angle factor

EnvStepFn = Callable[
    [EnvData, Array],
    tuple[EnvData, tuple[dict[str, Array], Array, Array, Array, dict]],
]
EnvResetFn = Callable[
    [EnvData, int | None, Array | None],
    tuple[EnvData, tuple[dict[str, Array], dict]],
]


@dataclass(frozen=True)
class RolloutStaticConfig:
    """Static configuration for a compiled rollout scan.

    Parameters
    ----------
    n_steps : int
        Number of environment steps collected per PPO rollout.
    n_envs : int
        Number of vectorized racing worlds.
    thrust_min, thrust_max : float
        Total-thrust bounds in newtons for the env action projection.
    max_episode_steps : int
        Episode timeout in environment steps. The race-core step function
        already applies this; the value is kept here to document the compiled
        rollout contract.
    reward_cfg : RewardConfig
        Reward weights for :func:`reward.step_reward`.
    reset_pos_perturb_m : float
        Uniform reset-position perturbation half-width in meters.
    reset_vel_perturb_mps : float
        Uniform reset-velocity perturbation half-width in meters per second.
    reset_yaw_perturb_rad : float
        Uniform reset-yaw perturbation half-width in radians.
    """

    n_steps: int
    n_envs: int
    thrust_min: float
    thrust_max: float
    max_episode_steps: int
    reward_cfg: RewardConfig
    reset_pos_perturb_m: float = 0.0
    reset_vel_perturb_mps: float = 0.0
    reset_yaw_perturb_rad: float = 0.0

    @property
    def reset_perturbation_enabled(self) -> bool:
        """Return whether curriculum reset perturbations are nonzero."""
        return any(
            value > 0.0
            for value in (
                self.reset_pos_perturb_m,
                self.reset_vel_perturb_mps,
                self.reset_yaw_perturb_rad,
            )
        )


class RolloutScanOutputs(NamedTuple):
    """Stacked transition tensors emitted by the rollout scan.

    Fields
    ------
    actor_obs, critic_obs : Array, shape (n_steps, n_envs, obs_dim)
    raw_actions : Array, shape (n_steps, n_envs, 7)
    logprobs, rewards, dones, values : Array, shape (n_steps, n_envs)
    reward_components : dict[str, Array]
        Per-component reward arrays, each shaped ``(n_steps, n_envs)``.
    target_gate_progress : Array, shape (n_steps, n_envs)
    crash, finished : Array, shape (n_steps, n_envs)
    """

    actor_obs: Array
    critic_obs: Array
    raw_actions: Array
    logprobs: Array
    rewards: Array
    dones: Array
    values: Array
    reward_components: dict[str, Array]
    target_gate_progress: Array
    crash: Array
    finished: Array


class RolloutMetricSums(NamedTuple):
    """Episode aggregates accumulated inside the rollout scan.

    Fields
    ------
    completed_return_sum, completed_length_sum, completed_count : Array
        Scalar sums over episodes completed during the rollout.
    """

    completed_return_sum: Array
    completed_length_sum: Array
    completed_count: Array


class RolloutScanResult(NamedTuple):
    """Result of one compiled PPO rollout scan.

    Fields
    ------
    env_data : EnvData
        Race-core state after all in-scan autoresets.
    prev_action_env_4vec : Array, shape (n_envs, 4)
        Previous env action after zeroing done worlds.
    rng_key, reset_rng_key : Array, shape (2,)
        Updated policy-sampling and reset-perturbation keys.
    next_done, episode_returns, episode_lengths : Array, shape (n_envs,)
    next_env_obs : dict[str, Array]
        Squeezed race-core observation for the final state.
    next_obs : dict[str, Array]
        Final normalized actor and critic observations.
    outputs : RolloutScanOutputs
        Stacked PPO buffers.
    metrics : RolloutMetricSums
        Completed-episode scalar sums.
    """

    env_data: EnvData
    prev_action_env_4vec: Array
    rng_key: Array
    reset_rng_key: Array
    next_done: Array
    episode_returns: Array
    episode_lengths: Array
    next_env_obs: dict[str, Array]
    next_obs: dict[str, Array]
    outputs: RolloutScanOutputs
    metrics: RolloutMetricSums


class _ScanCarry(NamedTuple):
    """Mutable scan carry for rollout collection."""

    env_data: EnvData
    prev_action_env_4vec: Array
    rng_key: Array
    reset_rng_key: Array
    next_done: Array
    episode_returns: Array
    episode_lengths: Array
    completed_return_sum: Array
    completed_length_sum: Array
    completed_count: Array


@partial(
    jax.jit,
    static_argnames=("env_step_fn", "env_reset_fn", "static_cfg"),
)
def scan_rollout(
    env_data: EnvData,
    actor_params: dict,
    critic_params: dict,
    normalizer: obs_encoding.NormalizerState,
    prev_action_env_4vec: Array,
    rng_key: Array,
    reset_rng_key: Array,
    next_done: Array,
    episode_returns: Array,
    episode_lengths: Array,
    env_step_fn: EnvStepFn,
    env_reset_fn: EnvResetFn,
    static_cfg: RolloutStaticConfig,
) -> RolloutScanResult:
    """Collect one PPO rollout inside a single JAX dispatch.

    Parameters
    ----------
    env_data : EnvData
        Current JAX-pure race-core state.
    actor_params, critic_params : dict
        Separate Flax parameter PyTrees for actor and critic.
    normalizer : NormalizerState
        Running observation normalizer used for every observation in the scan.
    prev_action_env_4vec : Array, shape (n_envs, 4)
        Previous env-boundary action ``[roll, pitch, yaw, thrust]``.
    rng_key : Array, shape (2,)
        PRNG key used for raw 7-vector policy sampling.
    reset_rng_key : Array, shape (2,)
        PRNG key used for curriculum reset perturbations.
    next_done : Array, shape (n_envs,)
        Done flags carried from the previous rollout.
    episode_returns, episode_lengths : Array, shape (n_envs,)
        Partial episode statistics carried across rollout boundaries.
    env_step_fn : callable
        JIT-compatible ``RaceCoreEnv._step`` function.
    env_reset_fn : callable
        JIT-compatible ``RaceCoreEnv._reset`` function.
    static_cfg : RolloutStaticConfig
        Static rollout, reward, action, and reset configuration.

    Returns
    -------
    RolloutScanResult
        New env state, RNG keys, final observations, stacked rollout buffers,
        and completed-episode metric sums.

    Notes
    -----
    The scan calls the race-core step function directly, then computes the
    Song reward from the pre-reset post-step observation and unmasked true gate
    and obstacle positions. Done worlds are reset in-scan using
    ``RaceCoreEnv._reset``. Curriculum reset perturbations are also applied in
    pure JAX; this mirrors ``RLSongVecEnv._apply_reset_perturbation`` without
    calling SciPy inside the trace.
    """
    _validate_scan_inputs(
        prev_action_env_4vec,
        next_done,
        episode_returns,
        episode_lengths,
        static_cfg,
    )
    zero_scalar = jnp.asarray(0.0, dtype=jnp.float32)
    initial_carry = _ScanCarry(
        env_data=env_data,
        prev_action_env_4vec=prev_action_env_4vec,
        rng_key=rng_key,
        reset_rng_key=reset_rng_key,
        next_done=next_done,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        completed_return_sum=zero_scalar,
        completed_length_sum=zero_scalar,
        completed_count=zero_scalar,
    )

    def scan_step(
        carry: _ScanCarry, _: None
    ) -> tuple[_ScanCarry, RolloutScanOutputs]:
        env_obs = _single_drone_obs(race_core_obs(carry.env_data))
        actor_obs = obs_encoding.vmap_build_actor_obs(
            env_obs,
            carry.prev_action_env_4vec,
            normalizer,
        )
        critic_obs = obs_encoding.vmap_build_critic_obs(
            env_obs,
            carry.prev_action_env_4vec,
            normalizer,
        )

        rng_key, action_key = jax.random.split(carry.rng_key)
        raw_action, logprob = sample_and_log_prob(
            actor_params,
            actor_obs,
            action_key,
        )
        value = Critic().apply({"params": critic_params}, critic_obs)
        env_action = raw_to_env_action(
            raw_action,
            static_cfg.thrust_min,
            static_cfg.thrust_max,
        )

        stepped_data, (next_obs_full, _, terminated_full, truncated_full, _) = (
            env_step_fn(carry.env_data, env_action)
        )
        next_env_obs = _single_drone_obs(next_obs_full)
        terminated = terminated_full[:, SINGLE_DRONE_INDEX].astype(jnp.bool_)
        truncated = truncated_full[:, SINGLE_DRONE_INDEX].astype(jnp.bool_)

        current_target = next_env_obs["target_gate"]
        previous_target = env_obs["target_gate"]
        finished = current_target < 0
        terminated = terminated | finished
        gate_just_passed = (
            (current_target > previous_target) & (previous_target >= 0)
        ) | (finished & (previous_target >= 0))

        reward, components = step_reward(
            next_env_obs,
            env_obs,
            terminated,
            truncated,
            finished,
            gate_just_passed,
            static_cfg.reward_cfg,
            true_gates_pos=stepped_data.gates_pos,
            true_obstacles_pos=stepped_data.obstacles_pos,
        )

        done_bool = terminated | truncated
        done = done_bool.astype(jnp.float32)
        episode_returns = carry.episode_returns + reward
        episode_lengths = carry.episode_lengths + 1.0
        completed_return_sum = carry.completed_return_sum + jnp.sum(
            jnp.where(done_bool, episode_returns, 0.0)
        )
        completed_length_sum = carry.completed_length_sum + jnp.sum(
            jnp.where(done_bool, episode_lengths, 0.0)
        )
        completed_count = carry.completed_count + jnp.sum(done)

        reset_data, reset_rng_key = _reset_done_worlds(
            stepped_data,
            done_bool,
            carry.reset_rng_key,
            env_reset_fn,
            static_cfg,
        )
        next_prev_action = jnp.where(
            done_bool[:, None],
            jnp.zeros_like(env_action),
            env_action,
        )
        next_episode_returns = jnp.where(done_bool, 0.0, episode_returns)
        next_episode_lengths = jnp.where(done_bool, 0.0, episode_lengths)

        n_gates = env_obs["gates_pos"].shape[1]
        target_gate_progress = jnp.where(
            finished,
            n_gates,
            current_target,
        ).astype(jnp.float32)
        crash = terminated & ~finished

        next_carry = _ScanCarry(
            env_data=reset_data,
            prev_action_env_4vec=next_prev_action,
            rng_key=rng_key,
            reset_rng_key=reset_rng_key,
            next_done=done,
            episode_returns=next_episode_returns,
            episode_lengths=next_episode_lengths,
            completed_return_sum=completed_return_sum,
            completed_length_sum=completed_length_sum,
            completed_count=completed_count,
        )
        transition = RolloutScanOutputs(
            actor_obs=actor_obs,
            critic_obs=critic_obs,
            raw_actions=raw_action,
            logprobs=logprob,
            rewards=reward,
            dones=carry.next_done,
            values=value,
            reward_components=components,
            target_gate_progress=target_gate_progress,
            crash=crash,
            finished=finished,
        )
        return next_carry, transition

    final_carry, outputs = jax.lax.scan(
        scan_step,
        initial_carry,
        None,
        length=static_cfg.n_steps,
    )
    next_env_obs = _single_drone_obs(race_core_obs(final_carry.env_data))
    next_actor_obs = obs_encoding.vmap_build_actor_obs(
        next_env_obs,
        final_carry.prev_action_env_4vec,
        normalizer,
    )
    next_critic_obs = obs_encoding.vmap_build_critic_obs(
        next_env_obs,
        final_carry.prev_action_env_4vec,
        normalizer,
    )
    metric_sums = RolloutMetricSums(
        completed_return_sum=final_carry.completed_return_sum,
        completed_length_sum=final_carry.completed_length_sum,
        completed_count=final_carry.completed_count,
    )
    return RolloutScanResult(
        env_data=final_carry.env_data,
        prev_action_env_4vec=final_carry.prev_action_env_4vec,
        rng_key=final_carry.rng_key,
        reset_rng_key=final_carry.reset_rng_key,
        next_done=final_carry.next_done,
        episode_returns=final_carry.episode_returns,
        episode_lengths=final_carry.episode_lengths,
        next_env_obs=next_env_obs,
        next_obs={
            "actor_obs": next_actor_obs,
            "critic_obs": next_critic_obs,
        },
        outputs=outputs,
        metrics=metric_sums,
    )


def _validate_scan_inputs(
    prev_action_env_4vec: Array,
    next_done: Array,
    episode_returns: Array,
    episode_lengths: Array,
    static_cfg: RolloutStaticConfig,
) -> None:
    """Validate static rollout input shapes before tracing the scan."""
    n_envs = static_cfg.n_envs
    if prev_action_env_4vec.shape != (n_envs, ENV_ACTION_DIM):
        raise ValueError(
            "prev_action_env_4vec must have shape "
            f"{(n_envs, ENV_ACTION_DIM)}; got {prev_action_env_4vec.shape}"
        )
    for name, value in (
        ("next_done", next_done),
        ("episode_returns", episode_returns),
        ("episode_lengths", episode_lengths),
    ):
        if value.shape != (n_envs,):
            raise ValueError(f"{name} must have shape {(n_envs,)}; got {value.shape}")


def _reset_done_worlds(
    env_data: EnvData,
    done: Array,
    reset_rng_key: Array,
    env_reset_fn: EnvResetFn,
    static_cfg: RolloutStaticConfig,
) -> tuple[EnvData, Array]:
    """Reset completed worlds and apply curriculum perturbations."""

    def reset_branch(data: EnvData) -> tuple[EnvData, Array]:
        reset_data, _ = env_reset_fn(data, None, done)
        return _apply_reset_perturbation(
            reset_data,
            done,
            reset_rng_key,
            static_cfg,
        )

    return jax.lax.cond(
        jnp.any(done),
        reset_branch,
        lambda data: (data, reset_rng_key),
        env_data,
    )


def _apply_reset_perturbation(
    env_data: EnvData,
    mask: Array,
    rng_key: Array,
    static_cfg: RolloutStaticConfig,
) -> tuple[EnvData, Array]:
    """Apply the curriculum reset perturbation in pure JAX."""
    if not static_cfg.reset_perturbation_enabled:
        return env_data, rng_key

    rng_key, pos_key, vel_key, yaw_key = jax.random.split(
        rng_key,
        RESET_RNG_SPLITS,
    )
    states = env_data.sim_data.states
    pos_delta = jax.random.uniform(
        pos_key,
        shape=states.pos.shape,
        minval=-static_cfg.reset_pos_perturb_m,
        maxval=static_cfg.reset_pos_perturb_m,
    )
    vel = jax.random.uniform(
        vel_key,
        shape=states.vel.shape,
        minval=-static_cfg.reset_vel_perturb_mps,
        maxval=static_cfg.reset_vel_perturb_mps,
    )
    yaw_delta = jax.random.uniform(
        yaw_key,
        shape=states.quat.shape[:-1],
        minval=-static_cfg.reset_yaw_perturb_rad,
        maxval=static_cfg.reset_yaw_perturb_rad,
    )

    mask_broadcast = mask[:, None, None]
    pos = jnp.clip(
        states.pos + pos_delta,
        env_data.pos_limit_low,
        env_data.pos_limit_high,
    )
    quat = _apply_yaw_delta(states.quat, yaw_delta, mask_broadcast)
    states = states.replace(
        pos=jnp.where(mask_broadcast, pos, states.pos),
        vel=jnp.where(mask_broadcast, vel, states.vel),
        quat=quat,
    )
    sim_data = env_data.sim_data.replace(states=states)
    env_data = env_data.replace(sim_data=sim_data)
    return _reset_env_data(env_data, mask), rng_key


def _apply_yaw_delta(quat: Array, yaw_delta: Array, mask: Array) -> Array:
    """Apply a world-frame yaw delta to xyzw quaternions in pure JAX.

    Notes
    -----
    The Python wrapper uses SciPy for this reset perturbation. The scanned
    rollout needs the same operation to be JAX-traceable inside ``lax.scan``.
    """
    yaw_quat = _yaw_quat_xyzw(yaw_delta)
    perturbed = _quat_multiply_xyzw(yaw_quat, quat)
    return jnp.where(mask, perturbed, quat)


def _yaw_quat_xyzw(yaw: Array) -> Array:
    """Return xyzw quaternions for pure-z yaw rotations."""
    half_yaw = YAW_TO_HALF_ANGLE * yaw
    zeros = jnp.zeros_like(yaw)
    return jnp.stack(
        [
            zeros,
            zeros,
            jnp.sin(half_yaw),
            jnp.cos(half_yaw),
        ],
        axis=-1,
    )


def _quat_multiply_xyzw(left: Array, right: Array) -> Array:
    """Multiply two xyzw quaternion arrays."""
    left_x, left_y, left_z, left_w = jnp.moveaxis(left, -1, 0)
    right_x, right_y, right_z, right_w = jnp.moveaxis(right, -1, 0)
    x = left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y
    y = left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x
    z = left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w
    w = left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z
    return jnp.stack([x, y, z, w], axis=-1)


def _single_drone_obs(env_obs: dict[str, Array]) -> dict[str, Array]:
    """Squeeze the single-drone axis from race-core observations."""
    return {
        key: value[:, SINGLE_DRONE_INDEX]
        for key, value in env_obs.items()
    }
