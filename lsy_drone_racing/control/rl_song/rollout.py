"""JAX-scanned PPO rollout collection for the Song-2023 controller."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from lsy_drone_racing.control.rl_song import obs as obs_encoding
from lsy_drone_racing.control.rl_song.config import ENV_ACTION_DIM, RewardConfig
from lsy_drone_racing.control.rl_song.policy import Critic, raw_to_env_action, sample_and_log_prob
from lsy_drone_racing.control.rl_song.reward import step_reward
from lsy_drone_racing.envs.race_core import EnvData, _reset_env_data, obs as race_core_obs

SINGLE_DRONE_INDEX: int = 0
RESET_RNG_SPLITS: int = 4
YAW_TO_HALF_ANGLE: float = 0.5  # quaternion half-angle factor


def _patch_env_obs_with_placed(
    env_obs: dict[str, Array],
    placed_gates_pos: Array,
    placed_gates_quat: Array,
    placed_obstacles_pos: Array,
    track_perturbation_enabled: bool,
) -> dict[str, Array]:
    """Replace toml-nominal pose entries with the per-env Layer-1 placement.

    The framework's ``race_core.obs`` masks ``gates_pos`` / ``gates_quat`` /
    ``obstacles_pos`` between ``data.<field>`` (visited) and
    ``data.nominal_<field>`` (not visited). On level-3 with
    ``track.randomize=true`` the nominal fields stay at the toml's
    ``(0, 0, z)`` placeholders, so the un-visited branch leaks dead info
    instead of the actual placement. This helper rebuilds the masking using
    the per-env ``placed_*`` snapshots (Layer-1 placement, pre-wobble) for
    the un-visited branch.

    No-op when ``track_perturbation_enabled`` is False (level 1 / 2 still
    have informative nominal fields from the toml).

    Parameters
    ----------
    env_obs : dict[str, Array]
        Single-drone env observation from ``race_core.obs`` after
        ``_single_drone_obs``. Mutated in-place is *not* required; a fresh
        dict is returned.
    placed_gates_pos : Array, shape (n_envs, n_gates, 3)
    placed_gates_quat : Array, shape (n_envs, n_gates, 4)
    placed_obstacles_pos : Array, shape (n_envs, n_obstacles, 3)
    track_perturbation_enabled : bool
        Static flag set by the wrapper. When False, returns ``env_obs``
        unchanged.

    Returns
    -------
    dict[str, Array]
        Patched env observation. ``env_obs`` is not mutated.
    """
    if not track_perturbation_enabled:
        return env_obs
    patched = dict(env_obs)
    gates_visited = env_obs["gates_visited"].astype(jnp.bool_)[..., None]
    patched["gates_pos"] = jnp.where(gates_visited, env_obs["gates_pos"], placed_gates_pos)
    patched["gates_quat"] = jnp.where(gates_visited, env_obs["gates_quat"], placed_gates_quat)
    obstacles_visited = env_obs["obstacles_visited"].astype(jnp.bool_)[..., None]
    patched["obstacles_pos"] = jnp.where(
        obstacles_visited, env_obs["obstacles_pos"], placed_obstacles_pos
    )
    return patched


EnvStepFn = Callable[
    [EnvData, Array], tuple[EnvData, tuple[dict[str, Array], Array, Array, Array, dict]]
]
EnvResetFn = Callable[
    [EnvData, int | None, Array | None], tuple[EnvData, tuple[dict[str, Array], dict]]
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
    # Level-3 track perturbation. When enabled, each reset snaps
    # ``nominal_*`` fields to the just-placed layout (so the controller's
    # pre-visit observation matches placement) and then adds per-axis
    # uniform wobble to the post-randomization ``gates_pos`` / ``obstacles_pos``
    # within the half-widths below. See ``env_wrapper.track_perturbation_bounds``.
    track_perturbation_enabled: bool = False
    gate_pos_perturb_max: tuple[float, float, float] = (0.0, 0.0, 0.0)
    obstacle_pos_perturb_max: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # v9 (Song 2023 §III-B Phase 1) segment initialization. With probability
    # ``segment_init_prob``, an env that just reset is re-spawned hovering
    # at the midpoint of a uniformly-random path segment with target_gate
    # advanced to match. Disabled (0.0) for stages where it is unused.
    segment_init_prob: float = 0.0
    segment_init_perturb_m: float = 0.10
    # v29: speed in m/s applied to seg-init re-spawns. When >0, the drone
    # is given velocity ``segment_init_vel_mps * unit(next_gate - prev_anchor)``
    # at re-spawn instead of zero. See ``CurriculumStage.segment_init_vel_mps``.
    segment_init_vel_mps: float = 0.0

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
    completed_max_gate_sum : Array
        Scalar sum of the per-episode maximum target-gate index reached, in
        the same ``target_gate_progress`` scale used elsewhere (``n_gates``
        means the lap was finished). Dividing by ``completed_count`` gives
        the average best-gate-reached per finished episode and is robust to
        the episode-length bias that distorts ``target_gate_mean`` for fast
        policies.
    true_start_completed_count, true_start_finished_count : Array
        Scalar counters restricted to episodes that did *not* start with
        Song §III-B seg-init (i.e. true ground-spawn starts). Their ratio
        is the unbiased ``finish_rate_true_start`` metric — it reports the
        per-episode finish rate the controller would see when deployed on
        a real ground spawn, regardless of how aggressively seg-init is
        applied during training. When seg-init is disabled
        (``segment_init_prob = 0``) both counters equal their unrestricted
        counterparts ``completed_count`` and the finished-subset thereof.
    """

    completed_return_sum: Array
    completed_length_sum: Array
    completed_count: Array
    completed_max_gate_sum: Array
    true_start_completed_count: Array
    true_start_finished_count: Array


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
    # Per-env Layer-1 placement (pre-wobble) after the rollout's resets.
    # Plumbed back into the wrapper so the next rollout starts with
    # ``placed_*`` consistent with the env data carried forward.
    placed_gates_pos: Array
    placed_gates_quat: Array
    placed_obstacles_pos: Array


class _ScanCarry(NamedTuple):
    """Mutable scan carry for rollout collection."""

    env_data: EnvData
    prev_action_env_4vec: Array
    rng_key: Array
    reset_rng_key: Array
    next_done: Array
    episode_returns: Array
    episode_lengths: Array
    episode_max_gate: Array
    completed_return_sum: Array
    completed_length_sum: Array
    completed_count: Array
    completed_max_gate_sum: Array
    # Per-env flag: True if the currently-running episode started via Song
    # §III-B seg-init (re-spawned at a segment midpoint), False if it
    # started from the true reset state (toml start position + reset
    # perturbation). Updated on each ``done`` event using the ``do_seg``
    # mask returned by ``_reset_done_worlds``. Drives the
    # ``finish_rate_true_start`` metric: the dying episode's stats are
    # tallied into ``true_start_*`` counters *only* if this flag was
    # False, so the metric reports performance on the deployment
    # state distribution regardless of seg-init aggressiveness.
    is_seg_init: Array
    true_start_completed_count: Array
    true_start_finished_count: Array
    # Per-env Layer-1 placement (pre-wobble), used to patch ``env_obs`` for
    # non-visited gates / obstacles so the actor sees the placement instead
    # of the framework's broken ``(0, 0, z)`` toml nominal. See
    # ``_apply_reset_perturbation``.
    placed_gates_pos: Array
    placed_gates_quat: Array
    placed_obstacles_pos: Array


@partial(jax.jit, static_argnames=("env_step_fn", "env_reset_fn", "static_cfg"))
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
    placed_gates_pos: Array,
    placed_gates_quat: Array,
    placed_obstacles_pos: Array,
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
        prev_action_env_4vec, next_done, episode_returns, episode_lengths, static_cfg
    )
    zero_scalar = jnp.asarray(0.0, dtype=jnp.float32)
    episode_max_gate = jnp.zeros_like(episode_returns)
    # Initialize is_seg_init to all False (assume the first episode in each
    # env is a true-start). The eager wrapper's initial reset may apply
    # seg-init, so the very first batch of completed episodes can be
    # mis-classified — but this washes out after one episode-length per env
    # (~3-10 s of training), negligible against the 60+ min runs we do.
    initial_is_seg_init = jnp.zeros_like(next_done, dtype=jnp.bool_)
    initial_carry = _ScanCarry(
        env_data=env_data,
        prev_action_env_4vec=prev_action_env_4vec,
        rng_key=rng_key,
        reset_rng_key=reset_rng_key,
        next_done=next_done,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        episode_max_gate=episode_max_gate,
        completed_return_sum=zero_scalar,
        completed_length_sum=zero_scalar,
        completed_count=zero_scalar,
        completed_max_gate_sum=zero_scalar,
        is_seg_init=initial_is_seg_init,
        true_start_completed_count=zero_scalar,
        true_start_finished_count=zero_scalar,
        placed_gates_pos=placed_gates_pos,
        placed_gates_quat=placed_gates_quat,
        placed_obstacles_pos=placed_obstacles_pos,
    )

    def scan_step(carry: _ScanCarry, _: None) -> tuple[_ScanCarry, RolloutScanOutputs]:
        env_obs = _patch_env_obs_with_placed(
            _single_drone_obs(race_core_obs(carry.env_data)),
            carry.placed_gates_pos,
            carry.placed_gates_quat,
            carry.placed_obstacles_pos,
            static_cfg.track_perturbation_enabled,
        )
        actor_obs = obs_encoding.vmap_build_actor_obs(
            env_obs, carry.prev_action_env_4vec, normalizer
        )
        critic_obs = obs_encoding.vmap_build_critic_obs(
            env_obs,
            carry.prev_action_env_4vec,
            normalizer,
            true_gates_pos=carry.env_data.gates_pos,
            true_gates_quat=carry.env_data.gates_quat,
            true_obstacles_pos=carry.env_data.obstacles_pos,
        )

        rng_key, action_key = jax.random.split(carry.rng_key)
        raw_action, logprob = sample_and_log_prob(actor_params, actor_obs, action_key)
        value = Critic().apply({"params": critic_params}, critic_obs)
        env_action = raw_to_env_action(raw_action, static_cfg.thrust_min, static_cfg.thrust_max)

        stepped_data, (next_obs_full, _, terminated_full, truncated_full, _) = env_step_fn(
            carry.env_data, env_action
        )
        next_env_obs = _single_drone_obs(next_obs_full)
        terminated = terminated_full[:, SINGLE_DRONE_INDEX].astype(jnp.bool_)
        truncated = truncated_full[:, SINGLE_DRONE_INDEX].astype(jnp.bool_)

        current_target = next_env_obs["target_gate"]
        previous_target = env_obs["target_gate"]
        finished = current_target < 0
        terminated = terminated | finished
        gate_just_passed = ((current_target > previous_target) & (previous_target >= 0)) | (
            finished & (previous_target >= 0)
        )

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

        n_gates = env_obs["gates_pos"].shape[1]
        target_gate_progress = jnp.where(finished, n_gates, current_target).astype(jnp.float32)
        episode_max_gate = jnp.maximum(carry.episode_max_gate, target_gate_progress)

        completed_return_sum = carry.completed_return_sum + jnp.sum(
            jnp.where(done_bool, episode_returns, 0.0)
        )
        completed_length_sum = carry.completed_length_sum + jnp.sum(
            jnp.where(done_bool, episode_lengths, 0.0)
        )
        completed_max_gate_sum = carry.completed_max_gate_sum + jnp.sum(
            jnp.where(done_bool, episode_max_gate, 0.0)
        )
        completed_count = carry.completed_count + jnp.sum(done)

        # finish_rate_true_start tally: a dying episode counts here iff it
        # started from a true reset (carry.is_seg_init is False). Use the
        # current is_seg_init *before* updating it for the just-reset envs
        # below — the flag belongs to the episode that just ended.
        true_start_done = done_bool & ~carry.is_seg_init
        true_start_completed_count = carry.true_start_completed_count + jnp.sum(
            true_start_done.astype(jnp.float32)
        )
        true_start_finished_count = carry.true_start_finished_count + jnp.sum(
            (true_start_done & finished).astype(jnp.float32)
        )

        (
            reset_data,
            reset_rng_key,
            next_placed_gates_pos,
            next_placed_gates_quat,
            next_placed_obstacles_pos,
            do_seg,
        ) = _reset_done_worlds(
            stepped_data,
            done_bool,
            carry.reset_rng_key,
            carry.placed_gates_pos,
            carry.placed_gates_quat,
            carry.placed_obstacles_pos,
            env_reset_fn,
            static_cfg,
        )
        # On done events, the new episode's is_seg_init is set from the
        # do_seg mask returned by the reset path. Non-reset envs carry
        # their existing flag forward.
        next_is_seg_init = jnp.where(done_bool, do_seg, carry.is_seg_init)
        next_prev_action = jnp.where(done_bool[:, None], jnp.zeros_like(env_action), env_action)
        next_episode_returns = jnp.where(done_bool, 0.0, episode_returns)
        next_episode_lengths = jnp.where(done_bool, 0.0, episode_lengths)
        next_episode_max_gate = jnp.where(done_bool, 0.0, episode_max_gate)
        crash = terminated & ~finished

        next_carry = _ScanCarry(
            env_data=reset_data,
            prev_action_env_4vec=next_prev_action,
            rng_key=rng_key,
            reset_rng_key=reset_rng_key,
            next_done=done,
            episode_returns=next_episode_returns,
            episode_lengths=next_episode_lengths,
            episode_max_gate=next_episode_max_gate,
            completed_return_sum=completed_return_sum,
            completed_length_sum=completed_length_sum,
            completed_count=completed_count,
            completed_max_gate_sum=completed_max_gate_sum,
            is_seg_init=next_is_seg_init,
            true_start_completed_count=true_start_completed_count,
            true_start_finished_count=true_start_finished_count,
            placed_gates_pos=next_placed_gates_pos,
            placed_gates_quat=next_placed_gates_quat,
            placed_obstacles_pos=next_placed_obstacles_pos,
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

    final_carry, outputs = jax.lax.scan(scan_step, initial_carry, None, length=static_cfg.n_steps)
    next_env_obs = _patch_env_obs_with_placed(
        _single_drone_obs(race_core_obs(final_carry.env_data)),
        final_carry.placed_gates_pos,
        final_carry.placed_gates_quat,
        final_carry.placed_obstacles_pos,
        static_cfg.track_perturbation_enabled,
    )
    next_actor_obs = obs_encoding.vmap_build_actor_obs(
        next_env_obs, final_carry.prev_action_env_4vec, normalizer
    )
    next_critic_obs = obs_encoding.vmap_build_critic_obs(
        next_env_obs,
        final_carry.prev_action_env_4vec,
        normalizer,
        true_gates_pos=final_carry.env_data.gates_pos,
        true_gates_quat=final_carry.env_data.gates_quat,
        true_obstacles_pos=final_carry.env_data.obstacles_pos,
    )
    metric_sums = RolloutMetricSums(
        completed_return_sum=final_carry.completed_return_sum,
        completed_length_sum=final_carry.completed_length_sum,
        completed_count=final_carry.completed_count,
        completed_max_gate_sum=final_carry.completed_max_gate_sum,
        true_start_completed_count=final_carry.true_start_completed_count,
        true_start_finished_count=final_carry.true_start_finished_count,
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
        next_obs={"actor_obs": next_actor_obs, "critic_obs": next_critic_obs},
        outputs=outputs,
        metrics=metric_sums,
        placed_gates_pos=final_carry.placed_gates_pos,
        placed_gates_quat=final_carry.placed_gates_quat,
        placed_obstacles_pos=final_carry.placed_obstacles_pos,
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
    placed_gates_pos: Array,
    placed_gates_quat: Array,
    placed_obstacles_pos: Array,
    env_reset_fn: EnvResetFn,
    static_cfg: RolloutStaticConfig,
) -> tuple[EnvData, Array, Array, Array, Array, Array]:
    """Reset completed worlds and apply curriculum perturbations.

    Returns ``(env_data, rng_key, placed_gates_pos, placed_gates_quat,
    placed_obstacles_pos, do_seg)``. The placed snapshots reflect the
    Layer-1 placement (pre-wobble) for envs that were just reset and the
    previous values for envs that were not. ``do_seg`` is a per-env bool
    mask indicating which envs had their state replaced by seg-init this
    call; envs not reset have ``do_seg = False``.
    """

    def reset_branch(data: EnvData) -> tuple[EnvData, Array, Array, Array, Array, Array]:
        reset_data, _ = env_reset_fn(data, None, done)
        return _apply_reset_perturbation(
            reset_data,
            done,
            reset_rng_key,
            placed_gates_pos,
            placed_gates_quat,
            placed_obstacles_pos,
            static_cfg,
        )

    return jax.lax.cond(
        jnp.any(done),
        reset_branch,
        lambda data: (
            data,
            reset_rng_key,
            placed_gates_pos,
            placed_gates_quat,
            placed_obstacles_pos,
            jnp.zeros_like(done, dtype=jnp.bool_),
        ),
        env_data,
    )


def _apply_reset_perturbation(
    env_data: EnvData,
    mask: Array,
    rng_key: Array,
    placed_gates_pos: Array,
    placed_gates_quat: Array,
    placed_obstacles_pos: Array,
    static_cfg: RolloutStaticConfig,
) -> tuple[EnvData, Array, Array, Array, Array, Array]:
    """Apply curriculum drone-state and track perturbations in pure JAX.

    Returns the updated env data, RNG key, per-env ``placed_*`` snapshots
    (Layer-1 layout, *before* the Layer-2 wobble is added), and the
    ``do_seg`` mask identifying which envs had their state replaced by
    seg-init this call. Callers plumb ``placed_*`` through their carry /
    instance state to patch ``env_obs["gates_pos"]`` for non-visited
    entries (the framework's ``nominal_*`` fields can't be repurposed
    because ``race_core.obs`` assumes they have shape ``(n_gates, 3)``)
    and use ``do_seg`` to update the per-env ``is_seg_init`` flag that
    drives the ``finish_rate_true_start`` metric.

    Order of operations:

    1. Drone state perturbation (position, velocity, yaw) when
       ``reset_perturbation_enabled``.
    2. Track perturbation: snapshot ``env_data.gates_pos`` / quat /
       ``obstacles_pos`` into ``placed_*`` (Layer-1 placement, before wobble),
       then add per-axis uniform wobble to the env-data positions
       (Layer-2 ±max ``static_cfg.gate_pos_perturb_max``). Only when
       ``track_perturbation_enabled``.
    3. ``_reset_env_data`` to recompute ``gates_visited`` etc. with the
       final (Layer-1+2) positions.
    """
    # Snapshot the toml start position before any perturbation is applied.
    # ``_apply_segment_init`` consumes this as the segment-0 anchor.
    start_pos = env_data.sim_data.states.pos

    if static_cfg.reset_perturbation_enabled:
        rng_key, pos_key, vel_key, yaw_key = jax.random.split(rng_key, RESET_RNG_SPLITS)
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
        pos = jnp.clip(states.pos + pos_delta, env_data.pos_limit_low, env_data.pos_limit_high)
        quat = _apply_yaw_delta(states.quat, yaw_delta, mask_broadcast)
        states = states.replace(
            pos=jnp.where(mask_broadcast, pos, states.pos),
            vel=jnp.where(mask_broadcast, vel, states.vel),
            quat=quat,
        )
        sim_data = env_data.sim_data.replace(states=states)
        env_data = env_data.replace(sim_data=sim_data)

    if static_cfg.track_perturbation_enabled:
        rng_key, gate_key, obs_key = jax.random.split(rng_key, 3)
        gate_pos_max = jnp.asarray(static_cfg.gate_pos_perturb_max, dtype=jnp.float32)
        obs_pos_max = jnp.asarray(static_cfg.obstacle_pos_perturb_max, dtype=jnp.float32)
        gate_delta = jax.random.uniform(
            gate_key, shape=env_data.gates_pos.shape, minval=-gate_pos_max, maxval=gate_pos_max
        )
        obs_delta = jax.random.uniform(
            obs_key, shape=env_data.obstacles_pos.shape, minval=-obs_pos_max, maxval=obs_pos_max
        )
        mask_b = mask[:, None, None]
        # Snapshot Layer-1 placement (before wobble) for envs being reset.
        placed_gates_pos = jnp.where(mask_b, env_data.gates_pos, placed_gates_pos)
        placed_gates_quat = jnp.where(mask_b, env_data.gates_quat, placed_gates_quat)
        placed_obstacles_pos = jnp.where(mask_b, env_data.obstacles_pos, placed_obstacles_pos)
        # Apply Layer-2 wobble.
        env_data = env_data.replace(
            gates_pos=jnp.where(mask_b, env_data.gates_pos + gate_delta, env_data.gates_pos),
            obstacles_pos=jnp.where(
                mask_b, env_data.obstacles_pos + obs_delta, env_data.obstacles_pos
            ),
        )

    env_data = _reset_env_data(env_data, mask)
    if static_cfg.segment_init_prob > 0.0:
        env_data, rng_key, do_seg = _apply_segment_init(
            env_data, mask, rng_key, placed_gates_pos, start_pos, static_cfg
        )
    else:
        do_seg = jnp.zeros_like(mask, dtype=jnp.bool_)
    return env_data, rng_key, placed_gates_pos, placed_gates_quat, placed_obstacles_pos, do_seg


def _apply_segment_init(
    env_data: EnvData,
    mask: Array,
    rng_key: Array,
    placed_gates_pos: Array,
    start_pos: Array,
    static_cfg: RolloutStaticConfig,
) -> tuple[EnvData, Array, Array]:
    """Re-spawn a Bernoulli-selected subset of envs at random segment centers.

    Pure-JAX counterpart of ``RLSongVecEnv._apply_segment_init`` used inside
    the scanned rollout path. Both branches must produce the same
    state-distribution semantics for the policy. Implements Song 2023
    §III-B Phase 1 (state-coverage initial-state distribution).

    v29: when ``static_cfg.segment_init_vel_mps > 0`` the seg-init re-spawn
    velocity is set to ``segment_init_vel_mps * unit(next_gate -
    prev_anchor)`` instead of zero. Removes the trivially-exploitable
    "spawned hovering at a convenient midpoint" state distribution that
    caused v24's lucky-zone collapse, while still putting the policy at
    later-gate approach poses.

    Parameters
    ----------
    env_data : EnvData
        Post-``_reset_env_data`` env state.
    mask : Array, shape (n_envs,)
        Envs eligible for segment init (i.e., those that just reset).
    rng_key : Array
        PRNG key; consumed and returned.
    placed_gates_pos : Array, shape (n_envs, n_gates, 3)
        Layer-1 placed gate positions (pre-wobble) used to anchor segment
        midpoints.
    start_pos : Array, shape (n_envs, n_drones, 3)
        Pre-perturbation drone position used as the segment-0 anchor.
    static_cfg : RolloutStaticConfig
        Provides ``segment_init_prob``, ``segment_init_perturb_m`` and
        ``segment_init_vel_mps``.

    Returns
    -------
    env_data : EnvData
        Env state with ``sim_data.states.pos / vel / quat`` and ``target_gate``
        overridden for the selected envs.
    rng_key : Array
        Advanced PRNG key.
    do_seg : Array, shape (n_envs,)
        Boolean mask of envs whose state was overridden by seg-init this
        call. Plumbed back to ``scan_rollout`` so the per-env
        ``is_seg_init`` flag (used by the ``finish_rate_true_start``
        metric) can be updated.
    """
    rng_key, bern_key, seg_key, jit_key = jax.random.split(rng_key, 4)
    n_envs = env_data.gates_pos.shape[0]
    n_gates = env_data.gates_pos.shape[1]
    do_seg = jax.random.bernoulli(bern_key, p=static_cfg.segment_init_prob, shape=(n_envs,)) & mask
    segment_idx = jax.random.randint(seg_key, shape=(n_envs,), minval=0, maxval=n_gates)
    env_arange = jnp.arange(n_envs)
    prev_idx = jnp.clip(segment_idx - 1, 0, n_gates - 1)
    prev_gate = placed_gates_pos[env_arange, prev_idx]
    prev_anchor = jnp.where((segment_idx == 0)[:, None], start_pos[:, 0, :], prev_gate)
    next_gate = placed_gates_pos[env_arange, segment_idx]
    midpoint = 0.5 * (prev_anchor + next_gate)
    jitter = jax.random.uniform(
        jit_key,
        shape=(n_envs, 3),
        minval=-static_cfg.segment_init_perturb_m,
        maxval=static_cfg.segment_init_perturb_m,
    )
    new_pos = jnp.clip(midpoint + jitter, env_data.pos_limit_low, env_data.pos_limit_high)

    # v29: velocity-aware seg-init. Compute unit direction from prev anchor
    # to next gate; scale by segment_init_vel_mps. Falls back to zero
    # velocity when the speed is 0 (original Song §III-B behavior).
    direction = next_gate - prev_anchor
    direction_norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
    unit_direction = direction / jnp.maximum(direction_norm, 1e-6)
    seg_vel = static_cfg.segment_init_vel_mps * unit_direction  # (n_envs, 3)

    states = env_data.sim_data.states
    mask_b3 = do_seg[:, None, None]
    new_pos_b = new_pos[:, None, :]
    new_vel_b = seg_vel[:, None, :]
    identity_quat = jnp.zeros_like(states.quat).at[..., 3].set(1.0)
    new_states = states.replace(
        pos=jnp.where(mask_b3, new_pos_b, states.pos),
        vel=jnp.where(mask_b3, new_vel_b, states.vel),
        quat=jnp.where(mask_b3, identity_quat, states.quat),
    )

    new_target = jnp.where(
        do_seg[:, None],
        segment_idx[:, None].astype(env_data.target_gate.dtype),
        env_data.target_gate,
    )

    sim_data = env_data.sim_data.replace(states=new_states)
    env_data = env_data.replace(sim_data=sim_data, target_gate=new_target)
    return env_data, rng_key, do_seg


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
    return jnp.stack([zeros, zeros, jnp.sin(half_yaw), jnp.cos(half_yaw)], axis=-1)


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
    return {key: value[:, SINGLE_DRONE_INDEX] for key, value in env_obs.items()}
