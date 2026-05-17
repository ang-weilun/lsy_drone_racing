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

# Per-env episode-source enum stored in ``_ScanCarry.source``. Drives the
# per-source ``finish_rate_*`` metrics. Width int8 keeps the carry small.
SRC_TRUE_START: int = 0
SRC_PHASE1_SEG: int = 1
SRC_PHASE2_REPLAY: int = 2


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
    # Song 2023 §III-B Phase 2 successful-state buffer. ``phase2_prob`` is
    # the *effective* probability inside the scan — the caller (train.py)
    # zeroes it during the warm-up window so the rollout is traced with
    # ``phase2_prob=0.0`` until the buffer has populated, then re-traced
    # once with the configured value. ``phase2_capacity_per_gate`` is the
    # per-gate ring-buffer capacity and is therefore part of the static
    # shape contract.
    phase2_prob: float = 0.0
    phase2_capacity_per_gate: int = 4096

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


class Phase2Buffer(NamedTuple):
    """Per-gate stratified ring buffer of successful gate-pass states.

    Implements Song 2023 §III-B Phase 2 — a buffer of past states the
    policy has reached, from which selected envs are re-spawned at reset
    so the policy practices later-gate approaches without having to
    survive the early gates first. Stratification by target gate keeps
    the buffer from being dominated by gate-0 → gate-1 transitions
    (which are far more common than gate-(N-1) → finish transitions).

    States are stored in the **previous gate's local frame**: the position
    offset rotated by the gate's orientation, the velocity rotated to that
    frame, and the orientation expressed relative to the gate. This makes
    the buffer valid across level-3 track-randomization layouts:
    reconstruction at replay applies the current layout's gate transform
    to the stored local-frame entry.

    Fields
    ------
    data : Array, shape (n_gates, capacity, state_dim), dtype float32
        Ring-buffered entries. Slot index ``g`` holds states whose
        ``target_gate == g``. Slot 0 is unused (every true-start episode
        already approaches gate 0). Slot ``n_gates`` is never written
        because finished states (``target_gate == -1``) are deliberately
        filtered out.
    ptr : Array, shape (n_gates,), dtype int32
        Per-slot write head. Next write goes to ``data[g, ptr[g] % capacity]``.
    fill : Array, shape (n_gates,), dtype int32
        Per-slot count of valid entries, clipped to ``capacity``. Read
        sampling masks slots with ``fill[g] == 0`` so empty slots aren't
        sampled before being populated.
    """

    data: Array
    ptr: Array
    fill: Array


# State-tuple layout for ``Phase2Buffer.data`` entries (v31 layout-restoring).
# Each entry packs the absolute drone state plus the full layout the
# state came from, so replay can override BOTH the drone state and the
# env's layout fields to make the respawn geometrically self-consistent
# with what the drone observes (the v30 gate-frame transform was warped
# by independently-randomized next-gate positions; v31 sidesteps that
# by replaying into the *same* layout that generated the entry).
#
# Slice layout (with n_g = n_gates, n_o = n_obstacles), see helper
# :func:`_phase2_offsets`:
#
#   [0:3]                  pos_world (absolute)
#   [3:6]                  vel_world
#   [6:10]                 quat_world (xyzw)
#   [10:13]                ang_vel (body frame)
#   [13:17]                prev_action (env-action 4-vec)
#   [17 : 17+n_o]          obstacles_visited (bool stored as float)
#   gates_pos        (Layer-2 post-wobble),  3 * n_g floats
#   gates_quat       (no wobble; shared),    4 * n_g floats
#   obstacles_pos    (Layer-2 post-wobble),  3 * n_o floats
#   placed_gates_pos (Layer-1 pre-wobble),   3 * n_g floats
#   placed_obstacles_pos (Layer-1),          3 * n_o floats
#
# Total = PHASE2_DRONE_DIM + 7 * n_obstacles + 10 * n_gates.
PHASE2_DRONE_DIM: int = 3 + 3 + 4 + 3 + 4  # pos, vel, quat, ang_vel, prev_action = 17


def phase2_state_dim(n_obstacles: int, n_gates: int) -> int:
    """Return the per-entry state dim of a v31 layout-restoring Phase 2 buffer."""
    return (
        PHASE2_DRONE_DIM
        + n_obstacles  # obstacles_visited
        + 3 * n_gates  # gates_pos (Layer-2)
        + 4 * n_gates  # gates_quat (no wobble; shared with placed)
        + 3 * n_obstacles  # obstacles_pos (Layer-2)
        + 3 * n_gates  # placed_gates_pos (Layer-1)
        + 3 * n_obstacles  # placed_obstacles_pos (Layer-1)
    )


def _phase2_offsets(n_obstacles: int, n_gates: int) -> dict[str, int]:
    """Return start offsets per field for the Phase 2 entry layout."""
    offs = {"pos": 0, "vel": 3, "quat": 6, "ang_vel": 10, "prev_action": 13}
    cursor = PHASE2_DRONE_DIM
    offs["obstacles_visited"] = cursor
    cursor += n_obstacles
    offs["gates_pos"] = cursor
    cursor += 3 * n_gates
    offs["gates_quat"] = cursor
    cursor += 4 * n_gates
    offs["obstacles_pos"] = cursor
    cursor += 3 * n_obstacles
    offs["placed_gates_pos"] = cursor
    cursor += 3 * n_gates
    offs["placed_obstacles_pos"] = cursor
    cursor += 3 * n_obstacles
    offs["_end"] = cursor
    return offs


def empty_phase2_buffer(n_gates: int, capacity: int, state_dim: int) -> Phase2Buffer:
    """Allocate an empty (all-zero) Phase 2 buffer.

    Called once at training startup and again on each curriculum
    promotion (so per-stage capacity changes drop the previous buffer
    instead of trying to copy it).
    """
    return Phase2Buffer(
        data=jnp.zeros((n_gates, capacity, state_dim), dtype=jnp.float32),
        ptr=jnp.zeros((n_gates,), dtype=jnp.int32),
        fill=jnp.zeros((n_gates,), dtype=jnp.int32),
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
    p2_event_valid : Array, shape (n_steps, n_envs), bool
    p2_event_slot  : Array, shape (n_steps, n_envs), int32
    p2_event_data  : Array, shape (n_steps, n_envs, state_dim), float32
        Per-step Phase 2 buffer-write candidates. Folded into the buffer
        once after the scan in ``_apply_phase2_writes``; the in-scan body
        only emits them so the write itself doesn't run ``n_steps`` times.
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
    p2_event_valid: Array
    p2_event_slot: Array
    p2_event_data: Array


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
    (true_start|phase1_seg|phase2_replay)_(completed|finished)_count : Array
        Scalar counters restricted to episodes that started from each of
        the three sources (see ``SRC_*`` constants). Their per-source
        ratios are the ``finish_rate_<source>`` metrics. The
        ``finish_rate_true_start`` ratio is the unbiased deployment metric
        — the per-episode finish rate the controller would see from a real
        ground spawn, regardless of how aggressively the curriculum's
        Phase 1 / Phase 2 respawns are applied during training.
    """

    completed_return_sum: Array
    completed_length_sum: Array
    completed_count: Array
    completed_max_gate_sum: Array
    true_start_completed_count: Array
    true_start_finished_count: Array
    phase1_seg_completed_count: Array
    phase1_seg_finished_count: Array
    phase2_replay_completed_count: Array
    phase2_replay_finished_count: Array


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
    source : Array, shape (n_envs,), dtype int8
        Per-env episode-source code at the end of the rollout (see
        ``SRC_*`` constants). Plumbed back through the train loop so the
        classification persists across rollout boundaries — episodes
        longer than ``n_steps`` would otherwise be reclassified mid-flight,
        biasing the per-source finish-rate metrics.
    phase2_buffer : Phase2Buffer
        Per-gate stratified buffer of successful gate-pass states after
        this rollout's writes have been folded in. Plumbed back so the
        next rollout reads from the latest buffer.
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
    source: Array
    phase2_buffer: Phase2Buffer
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
    # Per-env episode-source code (int8, see ``SRC_*`` constants). Tags
    # how the currently-running episode started: from the true reset
    # state, from a Song §III-B Phase 1 seg-init midpoint, or replayed
    # from the Phase 2 successful-state buffer. Updated on each ``done``
    # event using the source returned by ``_reset_done_worlds``. Drives
    # the per-source ``finish_rate_*`` metrics: the dying episode's
    # stats are tallied into the counter pair matching its source, so
    # each ratio reports performance on its own start distribution.
    source: Array
    true_start_completed_count: Array
    true_start_finished_count: Array
    phase1_seg_completed_count: Array
    phase1_seg_finished_count: Array
    phase2_replay_completed_count: Array
    phase2_replay_finished_count: Array
    # Per-env Layer-1 placement (pre-wobble), used to patch ``env_obs`` for
    # non-visited gates / obstacles so the actor sees the placement instead
    # of the framework's broken ``(0, 0, z)`` toml nominal. See
    # ``_apply_reset_perturbation``.
    placed_gates_pos: Array
    placed_gates_quat: Array
    placed_obstacles_pos: Array
    # Phase 2 successful-state buffer. Read by the reset path to sample
    # replay states; written once after the scan (in ``scan_rollout``,
    # not inside the scan body) using a masked-scatter pattern. Carrying
    # it through the scan keeps the read path consistent across steps
    # of the same rollout (writes from earlier steps are not visible to
    # later same-rollout reads — by design, to avoid in-rollout feedback).
    phase2_buffer: Phase2Buffer


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
    source: Array,
    placed_gates_pos: Array,
    placed_gates_quat: Array,
    placed_obstacles_pos: Array,
    phase2_buffer: Phase2Buffer,
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
    source : Array, shape (n_envs,), dtype int8
        Per-env episode-source code carried from the previous rollout
        (see ``SRC_*`` constants). The caller must persist this across
        rollout boundaries: episodes longer than ``n_steps`` would
        otherwise be reclassified as ``SRC_TRUE_START`` mid-flight,
        biasing every per-source ``finish_rate_*`` metric.
    phase2_buffer : Phase2Buffer
        Per-gate stratified buffer of past successful gate-pass states.
        Read by the in-scan reset path for Phase 2 replay (B3 — not yet
        wired) and written once after the scan (B2 — not yet wired).
        Threaded through unchanged by the B1 plumbing.
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
        prev_action_env_4vec, next_done, episode_returns, episode_lengths, source, static_cfg
    )
    zero_scalar = jnp.asarray(0.0, dtype=jnp.float32)
    episode_max_gate = jnp.zeros_like(episode_returns)
    # ``source`` is passed in by the caller (train.py) and threaded back
    # out via ``RolloutScanResult.source`` so the per-env source code
    # persists across rollout boundaries. Initializing here to all
    # ``SRC_TRUE_START`` would reclassify episodes longer than
    # ``n_steps`` mid-flight, inflating ``finish_rate_true_start`` and
    # deflating the Phase 1 / Phase 2 ratios. The eager wrapper's first
    # reset may apply Phase 1 seg-init without tagging it, so the first
    # batch of completed episodes after process start can still be
    # mis-classified; that washes out after one episode-length per env.
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
        source=source,
        true_start_completed_count=zero_scalar,
        true_start_finished_count=zero_scalar,
        phase1_seg_completed_count=zero_scalar,
        phase1_seg_finished_count=zero_scalar,
        phase2_replay_completed_count=zero_scalar,
        phase2_replay_finished_count=zero_scalar,
        placed_gates_pos=placed_gates_pos,
        placed_gates_quat=placed_gates_quat,
        placed_obstacles_pos=placed_obstacles_pos,
        phase2_buffer=phase2_buffer,
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

        # Per-source finish-rate tally: a dying episode counts toward the
        # counter pair matching its source code. Use ``carry.source``
        # (the *pre-reset* value), so the dying episode's classification
        # belongs to the episode that just ended.
        true_start_done = done_bool & (carry.source == SRC_TRUE_START)
        phase1_seg_done = done_bool & (carry.source == SRC_PHASE1_SEG)
        phase2_replay_done = done_bool & (carry.source == SRC_PHASE2_REPLAY)
        true_start_completed_count = carry.true_start_completed_count + jnp.sum(
            true_start_done.astype(jnp.float32)
        )
        true_start_finished_count = carry.true_start_finished_count + jnp.sum(
            (true_start_done & finished).astype(jnp.float32)
        )
        phase1_seg_completed_count = carry.phase1_seg_completed_count + jnp.sum(
            phase1_seg_done.astype(jnp.float32)
        )
        phase1_seg_finished_count = carry.phase1_seg_finished_count + jnp.sum(
            (phase1_seg_done & finished).astype(jnp.float32)
        )
        phase2_replay_completed_count = carry.phase2_replay_completed_count + jnp.sum(
            phase2_replay_done.astype(jnp.float32)
        )
        phase2_replay_finished_count = carry.phase2_replay_finished_count + jnp.sum(
            (phase2_replay_done & finished).astype(jnp.float32)
        )

        (
            reset_data,
            reset_rng_key,
            next_placed_gates_pos,
            next_placed_gates_quat,
            next_placed_obstacles_pos,
            do_seg,
            do_phase2,
            replay_prev_action,
        ) = _reset_done_worlds(
            stepped_data,
            done_bool,
            carry.reset_rng_key,
            carry.placed_gates_pos,
            carry.placed_gates_quat,
            carry.placed_obstacles_pos,
            carry.phase2_buffer,
            env_reset_fn,
            static_cfg,
        )
        # On done events, derive the new episode's source code from the
        # reset path's two masks. Phase 2 wins over Phase 1 (they're
        # disjoint by construction so the priority order doesn't actually
        # matter, but the conditional keeps the intent explicit).
        # Non-reset envs carry their existing source code forward.
        reset_source = jnp.where(
            do_phase2,
            jnp.asarray(SRC_PHASE2_REPLAY, dtype=carry.source.dtype),
            jnp.where(
                do_seg,
                jnp.asarray(SRC_PHASE1_SEG, dtype=carry.source.dtype),
                jnp.asarray(SRC_TRUE_START, dtype=carry.source.dtype),
            ),
        )
        next_source = jnp.where(done_bool, reset_source, carry.source)
        # Default ``prev_action`` zeroing on done; Phase 2 replay
        # overrides with the action stored alongside the replayed state
        # so the policy's autoregressive input is consistent with the
        # respawned pose.
        next_prev_action = jnp.where(done_bool[:, None], jnp.zeros_like(env_action), env_action)
        next_prev_action = jnp.where(do_phase2[:, None], replay_prev_action, next_prev_action)
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
            source=next_source,
            true_start_completed_count=true_start_completed_count,
            true_start_finished_count=true_start_finished_count,
            phase1_seg_completed_count=phase1_seg_completed_count,
            phase1_seg_finished_count=phase1_seg_finished_count,
            phase2_replay_completed_count=phase2_replay_completed_count,
            phase2_replay_finished_count=phase2_replay_finished_count,
            placed_gates_pos=next_placed_gates_pos,
            placed_gates_quat=next_placed_gates_quat,
            placed_obstacles_pos=next_placed_obstacles_pos,
            # Buffer round-trips unchanged in B1 — writes land after the
            # scan in B2. In-scan reads for Phase 2 replay land in B3.
            phase2_buffer=carry.phase2_buffer,
        )
        p2_event_valid, p2_event_slot, p2_event_data = _compute_phase2_event(
            stepped_data=stepped_data,
            next_env_obs=next_env_obs,
            env_action=env_action,
            placed_gates_pos=carry.placed_gates_pos,
            placed_obstacles_pos=carry.placed_obstacles_pos,
            current_target=current_target,
            gate_just_passed=gate_just_passed,
            done_bool=done_bool,
            source=carry.source,
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
            p2_event_valid=p2_event_valid,
            p2_event_slot=p2_event_slot,
            p2_event_data=p2_event_data,
        )
        return next_carry, transition

    final_carry, outputs = jax.lax.scan(scan_step, initial_carry, None, length=static_cfg.n_steps)
    # Fold this rollout's gate-pass events into the Phase 2 buffer once
    # (instead of n_steps times inside the scan body). ``n_gates`` is a
    # Python int known at trace time, so the per-slot loop unrolls.
    n_gates_for_writes = int(env_data.gates_pos.shape[1])
    updated_phase2_buffer = _apply_phase2_writes(
        final_carry.phase2_buffer,
        outputs.p2_event_valid,
        outputs.p2_event_slot,
        outputs.p2_event_data,
        n_gates_for_writes,
    )
    final_carry = final_carry._replace(phase2_buffer=updated_phase2_buffer)
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
        phase1_seg_completed_count=final_carry.phase1_seg_completed_count,
        phase1_seg_finished_count=final_carry.phase1_seg_finished_count,
        phase2_replay_completed_count=final_carry.phase2_replay_completed_count,
        phase2_replay_finished_count=final_carry.phase2_replay_finished_count,
    )
    return RolloutScanResult(
        env_data=final_carry.env_data,
        prev_action_env_4vec=final_carry.prev_action_env_4vec,
        rng_key=final_carry.rng_key,
        reset_rng_key=final_carry.reset_rng_key,
        next_done=final_carry.next_done,
        episode_returns=final_carry.episode_returns,
        episode_lengths=final_carry.episode_lengths,
        source=final_carry.source,
        phase2_buffer=final_carry.phase2_buffer,
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
    source: Array,
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
        ("source", source),
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
    phase2_buffer: Phase2Buffer,
    env_reset_fn: EnvResetFn,
    static_cfg: RolloutStaticConfig,
) -> tuple[EnvData, Array, Array, Array, Array, Array, Array, Array]:
    """Reset completed worlds and apply curriculum perturbations.

    Returns ``(env_data, rng_key, placed_gates_pos, placed_gates_quat,
    placed_obstacles_pos, do_seg, do_phase2, replay_prev_action)``. The
    placed snapshots reflect the Layer-1 placement (pre-wobble) for envs
    that were just reset and the previous values for envs that were not.
    ``do_seg`` / ``do_phase2`` are per-env bool masks identifying which
    envs had their state replaced by each respawn source. ``replay_prev_action``
    is the env-action 4-vec from each Phase 2-replayed entry — the scan
    caller uses it to overwrite ``prev_action_env_4vec`` for those envs.
    Envs not reset have all-False masks and zero replay_prev_action.
    """
    n_envs_local = done.shape[0]

    def reset_branch(
        data: EnvData,
    ) -> tuple[EnvData, Array, Array, Array, Array, Array, Array, Array]:
        reset_data, _ = env_reset_fn(data, None, done)
        return _apply_reset_perturbation(
            reset_data,
            done,
            reset_rng_key,
            placed_gates_pos,
            placed_gates_quat,
            placed_obstacles_pos,
            phase2_buffer,
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
            jnp.zeros_like(done, dtype=jnp.bool_),
            jnp.zeros((n_envs_local, ENV_ACTION_DIM), dtype=jnp.float32),
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
    phase2_buffer: Phase2Buffer,
    static_cfg: RolloutStaticConfig,
) -> tuple[EnvData, Array, Array, Array, Array, Array, Array, Array]:
    """Apply curriculum drone-state and track perturbations in pure JAX.

    Returns the updated env data, RNG key, per-env ``placed_*`` snapshots
    (Layer-1 layout, *before* the Layer-2 wobble is added), and the two
    masks ``do_seg`` / ``do_phase2`` identifying which envs were
    respawned by which path, plus ``replay_prev_action`` (the env-action
    4-vec from each Phase 2-replayed entry, used to overwrite
    ``prev_action_env_4vec`` on respawn so the policy's autoregressive
    input is consistent with the replayed state).

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
    4. Three-way categorical: assign each just-reset env to Phase 2
       replay, Phase 1 seg-init, or true start. Apply each branch on
       its mask.
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

    # Three-way categorical reset partition. A single uniform draw
    # assigns each just-reset env to Phase 2 replay (u < p_p2), Phase 1
    # seg-init (p_p2 <= u < p_p2 + p_p1), or true start (rest). Phase 2
    # and Phase 1 are mutually exclusive by construction. When
    # ``p_phase2 == 0`` (warm-up) Phase 2 is skipped entirely; when
    # ``segment_init_prob == 0`` Phase 1 is skipped entirely.
    n_envs = mask.shape[0]
    if static_cfg.phase2_prob > 0.0 or static_cfg.segment_init_prob > 0.0:
        rng_key, cat_key = jax.random.split(rng_key)
        u = jax.random.uniform(cat_key, shape=(n_envs,))
    else:
        u = jnp.zeros((n_envs,), dtype=jnp.float32)
    do_phase2_desired = mask & (u < static_cfg.phase2_prob)
    do_seg_desired = (
        mask
        & (u >= static_cfg.phase2_prob)
        & (u < static_cfg.phase2_prob + static_cfg.segment_init_prob)
    )

    # Phase 2: replay from the successful-state buffer. ``do_phase2`` may
    # be a strict subset of ``do_phase2_desired`` if the buffer has no
    # non-empty slots yet (the envs that wanted replay fall back to
    # true-start, not to Phase 1, to keep the categorical interpretable).
    # v31: replay also restores the layout the entry came from, returning
    # updated placed_* snapshots that the caller threads back into the
    # carry.
    if static_cfg.phase2_prob > 0.0:
        (
            env_data,
            rng_key,
            do_phase2,
            replay_prev_action,
            placed_gates_pos,
            placed_gates_quat,
            placed_obstacles_pos,
        ) = _apply_phase2_replay(
            env_data,
            do_phase2_desired,
            rng_key,
            phase2_buffer,
            placed_gates_pos,
            placed_gates_quat,
            placed_obstacles_pos,
        )
    else:
        do_phase2 = jnp.zeros_like(mask, dtype=jnp.bool_)
        replay_prev_action = jnp.zeros((n_envs, ENV_ACTION_DIM), dtype=jnp.float32)

    # Phase 1: seg-init on the precomputed mask (independent of buffer state).
    if static_cfg.segment_init_prob > 0.0:
        env_data, rng_key = _apply_segment_init(
            env_data, do_seg_desired, rng_key, placed_gates_pos, start_pos, static_cfg
        )
        do_seg = do_seg_desired
    else:
        do_seg = jnp.zeros_like(mask, dtype=jnp.bool_)
    return (
        env_data,
        rng_key,
        placed_gates_pos,
        placed_gates_quat,
        placed_obstacles_pos,
        do_seg,
        do_phase2,
        replay_prev_action,
    )


def _apply_phase2_replay(
    env_data: EnvData,
    do_phase2_desired: Array,
    rng_key: Array,
    buffer: Phase2Buffer,
    placed_gates_pos: Array,
    placed_gates_quat: Array,
    placed_obstacles_pos: Array,
) -> tuple[EnvData, Array, Array, Array, Array, Array, Array]:
    """Re-spawn selected envs from the Phase 2 successful-state buffer.

    v31 layout-restoring design: each buffer entry packs the absolute
    drone state AND the full layout (Layer-1 placed + Layer-2 wobbled)
    the state came from. On replay we override BOTH the drone state and
    the env layout so the respawn is geometrically self-consistent with
    everything the actor observes.

    Steps per replayed env:
    1. Sample a slot ``g`` uniformly over non-empty slots (slot 0 is
       unused). If every slot is empty, the effective mask is all-False
       and the caller's categorical leaves these envs at true-start.
    2. Sample an entry index uniformly in ``[0, fill[g])``.
    3. Unpack the absolute drone state and stored layout from the entry.
    4. Override ``sim_data.states.pos / vel / quat / ang_vel``,
       ``target_gate``, and ``env_data.gates_pos / gates_quat /
       obstacles_pos`` for the selected envs; refresh aux fields via
       :func:`_refresh_aux_fields_after_respawn`.

    Returns
    -------
    env_data : EnvData
        Env state with the Phase 2 overrides (drone state + layout) applied.
    rng_key : Array
        Advanced PRNG key.
    do_phase2_effective : Array, shape (n_envs,), bool
        Mask of envs whose state was actually replaced.
    replay_prev_action : Array, shape (n_envs, ENV_ACTION_DIM), float32
        Per-env env-action 4-vec from the replayed entry. Caller uses
        it to override ``prev_action_env_4vec`` so the policy's
        autoregressive input matches the replayed state.
    placed_gates_pos, placed_gates_quat, placed_obstacles_pos : Array
        Updated Layer-1 placement snapshots (replayed envs get the
        stored Layer-1; non-replayed envs keep what was passed in).
    """
    n_envs = env_data.gates_pos.shape[0]
    n_gates = env_data.gates_pos.shape[1]
    n_obstacles = env_data.obstacles_pos.shape[1]
    offs = _phase2_offsets(n_obstacles, n_gates)

    rng_key, slot_key, idx_key = jax.random.split(rng_key, 3)

    # Sample slot uniformly over non-empty gates. Slot 0 is unused
    # (drones already approach gate 0 from every true-start episode);
    # mask it out. If all sampleable slots are empty, fall back to
    # do_phase2_effective = False — the caller leaves these envs as
    # true-start, no override applied.
    sampleable = (buffer.fill > 0).at[0].set(False)
    any_sampleable = jnp.any(sampleable)
    logits = jnp.where(sampleable, 0.0, -jnp.inf)
    g_raw = jax.random.categorical(slot_key, logits, shape=(n_envs,))
    # Guard for the all-empty case where categorical's output is
    # undefined; the where-mask below discards the result anyway, but
    # we clamp the index so the gathers don't read garbage.
    g_safe = jnp.clip(g_raw, 1, n_gates - 1)

    slot_fill = buffer.fill[g_safe]
    entry_idx = jax.random.randint(
        idx_key, shape=(n_envs,), minval=0, maxval=jnp.maximum(slot_fill, 1)
    )
    entry = buffer.data[g_safe, entry_idx]  # (n_envs, state_dim)

    # Unpack the absolute drone state + stored layout (matches
    # ``_compute_phase2_event``'s concatenation order).
    pos_world = entry[..., offs["pos"] : offs["pos"] + 3]
    vel_world = entry[..., offs["vel"] : offs["vel"] + 3]
    quat_world = entry[..., offs["quat"] : offs["quat"] + 4]
    ang_vel = entry[..., offs["ang_vel"] : offs["ang_vel"] + 3]
    prev_action = entry[..., offs["prev_action"] : offs["prev_action"] + 4]
    obstacles_visited_f = entry[
        ..., offs["obstacles_visited"] : offs["obstacles_visited"] + n_obstacles
    ]
    stored_gates_pos = entry[..., offs["gates_pos"] : offs["gates_pos"] + 3 * n_gates].reshape(
        n_envs, n_gates, 3
    )
    stored_gates_quat = entry[..., offs["gates_quat"] : offs["gates_quat"] + 4 * n_gates].reshape(
        n_envs, n_gates, 4
    )
    stored_obstacles_pos = entry[
        ..., offs["obstacles_pos"] : offs["obstacles_pos"] + 3 * n_obstacles
    ].reshape(n_envs, n_obstacles, 3)
    stored_placed_gates_pos = entry[
        ..., offs["placed_gates_pos"] : offs["placed_gates_pos"] + 3 * n_gates
    ].reshape(n_envs, n_gates, 3)
    stored_placed_obstacles_pos = entry[
        ..., offs["placed_obstacles_pos"] : offs["placed_obstacles_pos"] + 3 * n_obstacles
    ].reshape(n_envs, n_obstacles, 3)

    # Effective mask: desired AND buffer has at least one usable entry.
    do_phase2_effective = do_phase2_desired & any_sampleable

    # Override env state (absolute coords — no rotmat math).
    mask_b3 = do_phase2_effective[:, None, None]
    states = env_data.sim_data.states
    pos_world_clipped = jnp.clip(pos_world, env_data.pos_limit_low, env_data.pos_limit_high)
    new_states = states.replace(
        pos=jnp.where(mask_b3, pos_world_clipped[:, None, :], states.pos),
        vel=jnp.where(mask_b3, vel_world[:, None, :], states.vel),
        quat=jnp.where(mask_b3, quat_world[:, None, :], states.quat),
        ang_vel=jnp.where(mask_b3, ang_vel[:, None, :], states.ang_vel),
    )
    new_target = jnp.where(
        do_phase2_effective[:, None],
        g_safe[:, None].astype(env_data.target_gate.dtype),
        env_data.target_gate,
    )

    # Override env layout for replayed envs. Both Layer-2 wobbled fields
    # (env_data.{gates_pos,gates_quat,obstacles_pos}) and the carry's
    # Layer-1 placed_* snapshots are restored so the actor obs (which
    # mixes both layers via ``_patch_env_obs_with_placed``) sees the
    # exact layout the entry came from.
    new_gates_pos = jnp.where(mask_b3, stored_gates_pos, env_data.gates_pos)
    new_gates_quat = jnp.where(mask_b3, stored_gates_quat, env_data.gates_quat)
    new_obstacles_pos = jnp.where(mask_b3, stored_obstacles_pos, env_data.obstacles_pos)
    sim_data = env_data.sim_data.replace(states=new_states)
    env_data = env_data.replace(
        sim_data=sim_data,
        target_gate=new_target,
        gates_pos=new_gates_pos,
        gates_quat=new_gates_quat,
        obstacles_pos=new_obstacles_pos,
    )
    env_data = _refresh_aux_fields_after_respawn(
        env_data, do_phase2_effective, pos_world_clipped, g_safe
    )

    # Override obstacles_visited with the stored entry (the helper
    # defaulted to all-True; v31 has the original visited mask available).
    replayed_obs_visited = obstacles_visited_f > 0.5
    new_obstacles_visited = jnp.where(
        do_phase2_effective[:, None, None],
        replayed_obs_visited[:, None, :],
        env_data.obstacles_visited,
    )
    env_data = env_data.replace(obstacles_visited=new_obstacles_visited)

    # Update Layer-1 placed_* snapshots so the carry stays consistent.
    new_placed_gates_pos = jnp.where(mask_b3, stored_placed_gates_pos, placed_gates_pos)
    new_placed_gates_quat = jnp.where(mask_b3, stored_gates_quat, placed_gates_quat)
    new_placed_obstacles_pos = jnp.where(mask_b3, stored_placed_obstacles_pos, placed_obstacles_pos)

    # Mask prev_action to zero for envs that did NOT replay (the gather
    # produced arbitrary bytes for those lanes).
    replay_prev_action = jnp.where(do_phase2_effective[:, None], prev_action, 0.0)

    return (
        env_data,
        rng_key,
        do_phase2_effective,
        replay_prev_action,
        new_placed_gates_pos,
        new_placed_gates_quat,
        new_placed_obstacles_pos,
    )


def _apply_segment_init(
    env_data: EnvData,
    do_seg: Array,
    rng_key: Array,
    placed_gates_pos: Array,
    start_pos: Array,
    static_cfg: RolloutStaticConfig,
) -> tuple[EnvData, Array]:
    """Re-spawn the envs identified by ``do_seg`` at random segment centers.

    Pure-JAX counterpart of ``RLSongVecEnv._apply_segment_init`` used inside
    the scanned rollout path. Both branches must produce the same
    state-distribution semantics for the policy. Implements Song 2023
    §III-B Phase 1 (state-coverage initial-state distribution).

    Selection is now done by the caller (``_apply_reset_perturbation``)
    via a three-way categorical that also dispatches Phase 2 replay; this
    function only applies the override on the envs the caller pre-selected.

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
    do_seg : Array, shape (n_envs,), dtype bool
        Pre-computed mask of envs to apply seg-init to. Caller's
        responsibility to ensure these envs are also marked as
        ``SRC_PHASE1_SEG`` in the source enum.
    rng_key : Array
        PRNG key; consumed and returned.
    placed_gates_pos : Array, shape (n_envs, n_gates, 3)
        Layer-1 placed gate positions (pre-wobble) used to anchor segment
        midpoints.
    start_pos : Array, shape (n_envs, n_drones, 3)
        Pre-perturbation drone position used as the segment-0 anchor.
    static_cfg : RolloutStaticConfig
        Provides ``segment_init_perturb_m`` and ``segment_init_vel_mps``.

    Returns
    -------
    env_data : EnvData
        Env state with ``sim_data.states.pos / vel / quat`` and ``target_gate``
        overridden for the selected envs.
    rng_key : Array
        Advanced PRNG key.
    """
    rng_key, seg_key, jit_key = jax.random.split(rng_key, 3)
    n_envs = env_data.gates_pos.shape[0]
    n_gates = env_data.gates_pos.shape[1]
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
    env_data = _refresh_aux_fields_after_respawn(env_data, do_seg, new_pos, segment_idx)
    return env_data, rng_key


def _refresh_aux_fields_after_respawn(
    env_data: EnvData, mask: Array, new_pos: Array, new_target_gate: Array
) -> EnvData:
    """Recompute ``EnvData`` aux fields after a seg-init / Phase 2 respawn.

    ``_apply_segment_init`` and the future Phase 2 replay override
    ``sim_data.states.pos / vel / quat`` and ``target_gate`` but leave
    ``last_drone_pos``, ``takeoff_pos``, ``gates_visited`` and
    ``obstacles_visited`` stale. Stale ``last_drone_pos`` corrupts the
    next step's ``check_gate_pass`` line-crossing test; stale
    ``takeoff_pos`` corrupts the platform-departure crash logic in
    ``race_core.py:check_done``; stale ``gates_visited`` mis-masks the
    actor observation.

    Parameters
    ----------
    env_data : EnvData
        Env state after the respawn override has been applied to
        ``sim_data.states`` and ``target_gate``.
    mask : Array, shape (n_envs,)
        Per-env mask identifying which envs were respawned this call.
        Non-masked envs are left untouched.
    new_pos : Array, shape (n_envs, 3)
        Post-respawn world-frame drone position. Used for both
        ``last_drone_pos`` (line-crossing reference) and ``takeoff_pos``
        (platform reference). Setting ``takeoff_pos`` to ``new_pos``
        means the platform-check is satisfied trivially on the first
        post-respawn step — a respawned drone hovers off-platform.
    new_target_gate : Array, shape (n_envs,)
        New target-gate index. ``gates_visited`` is reconstructed
        deterministically as ``[i < new_target_gate for i in range(n_gates)]``.

    Returns
    -------
    EnvData
        Env state with aux fields consistent with the respawned position
        and target. Single-drone assumption (drone axis size 1).
    """
    mask_drone3 = mask[:, None, None]  # (n_envs, 1, 1) for (n_envs, n_drones, 3)
    new_pos_drones = jnp.broadcast_to(new_pos[:, None, :], env_data.last_drone_pos.shape)
    n_gates = env_data.gates_visited.shape[-1]
    gate_indices = jnp.arange(n_gates)
    new_gates_visited = jnp.broadcast_to(
        gate_indices[None, None, :] < new_target_gate[:, None, None], env_data.gates_visited.shape
    )
    # Default-True for obstacles: a respawned drone is assumed to have
    # already seen the course's obstacles. This avoids spurious
    # sensor-bonus rewards on the first post-respawn step and matches
    # the typical mid-course state distribution.
    new_obstacles_visited = jnp.ones_like(env_data.obstacles_visited)
    return env_data.replace(
        last_drone_pos=jnp.where(mask_drone3, new_pos_drones, env_data.last_drone_pos),
        takeoff_pos=jnp.where(mask_drone3, new_pos_drones, env_data.takeoff_pos),
        gates_visited=jnp.where(mask_drone3, new_gates_visited, env_data.gates_visited),
        obstacles_visited=jnp.where(mask_drone3, new_obstacles_visited, env_data.obstacles_visited),
    )


def _compute_phase2_event(
    stepped_data: EnvData,
    next_env_obs: dict[str, Array],
    env_action: Array,
    placed_gates_pos: Array,
    placed_obstacles_pos: Array,
    current_target: Array,
    gate_just_passed: Array,
    done_bool: Array,
    source: Array,
) -> tuple[Array, Array, Array]:
    """Build per-env Phase 2 event tensors for one scan step.

    For every env that just passed a gate (and meets the storage filters),
    packs the post-step drone state (absolute world coords) and the
    full layout the state came from (Layer-1 ``placed_*`` and Layer-2
    wobbled ``env_data.*``) into the per-gate buffer state layout. The
    layout-restoring design (v31) replaces v30's gate-frame transform:
    on replay we restore the exact layout the entry came from, so the
    drone state stays geometrically consistent with everything it
    observes.

    Filters (codex): valid iff
    * ``gate_just_passed`` this step,
    * the new ``current_target`` indexes a writable slot
      (``1 <= current_target < n_gates``); slot 0 is unused and
      ``current_target == -1`` means finished,
    * the episode did not crash / truncate on the pass step
      (``~done_bool``), and
    * the dying episode's source was not itself a Phase 2 replay (avoid
      buffer feeding itself).

    Returns
    -------
    event_valid : Array, shape (n_envs,), bool
    event_slot  : Array, shape (n_envs,), int32 (the new target gate index)
    event_data  : Array, shape (n_envs, state_dim)
    """
    n_envs = stepped_data.gates_pos.shape[0]
    n_gates = stepped_data.gates_pos.shape[1]

    drone_pos = stepped_data.sim_data.states.pos[:, SINGLE_DRONE_INDEX]
    drone_vel = stepped_data.sim_data.states.vel[:, SINGLE_DRONE_INDEX]
    drone_quat = stepped_data.sim_data.states.quat[:, SINGLE_DRONE_INDEX]
    drone_ang_vel = stepped_data.sim_data.states.ang_vel[:, SINGLE_DRONE_INDEX]

    obstacles_visited = next_env_obs["obstacles_visited"].astype(jnp.float32)
    # Flatten per-env layout arrays into 1-D state slots. The unflatten
    # at replay (in ``_apply_phase2_replay``) reshapes them back.
    gates_pos_flat = stepped_data.gates_pos.reshape(n_envs, -1)
    gates_quat_flat = stepped_data.gates_quat.reshape(n_envs, -1)
    obstacles_pos_flat = stepped_data.obstacles_pos.reshape(n_envs, -1)
    placed_gates_pos_flat = placed_gates_pos.reshape(n_envs, -1)
    placed_obstacles_pos_flat = placed_obstacles_pos.reshape(n_envs, -1)

    event_data = jnp.concatenate(
        [
            drone_pos,
            drone_vel,
            drone_quat,
            drone_ang_vel,
            env_action,
            obstacles_visited,
            gates_pos_flat,
            gates_quat_flat,
            obstacles_pos_flat,
            placed_gates_pos_flat,
            placed_obstacles_pos_flat,
        ],
        axis=-1,
    )
    event_valid = (
        gate_just_passed
        & (current_target >= 1)
        & (current_target < n_gates)
        & ~done_bool
        & (source != SRC_PHASE2_REPLAY)
    )
    event_slot = current_target.astype(jnp.int32)
    return event_valid, event_slot, event_data


def _apply_phase2_writes(
    buffer: Phase2Buffer, event_valid: Array, event_slot: Array, event_data: Array, n_gates: int
) -> Phase2Buffer:
    """Fold one rollout's gate-pass events into the per-gate ring buffer.

    Applies the cumsum-rank scatter pattern (codex) once per writable
    slot. ``unique_indices`` would be hard to prove without the
    capacity guard, so we use ``mode="drop"`` with OOB indices to safely
    discard the "no event" lanes — both more readable than scatter and
    no slower at this scale (n_steps * n_envs ~= a few thousand entries
    per slot per rollout, capacity 4096).

    Parameters
    ----------
    buffer : Phase2Buffer
        Buffer state at the start of this rollout.
    event_valid : Array, shape (n_steps, n_envs), bool
        Gate-pass events to write. Flattened internally.
    event_slot : Array, shape (n_steps, n_envs), int32
        Target-gate index per event (== which buffer slot).
    event_data : Array, shape (n_steps, n_envs, state_dim), float32
        Per-event state-tuple in gate-frame (see ``_compute_phase2_event``).
    n_gates : int
        Number of gates on the track. Loop bound (Python int, unrolled
        by the JIT trace).

    Returns
    -------
    Phase2Buffer
        Updated buffer with ``ptr`` and ``fill`` advanced per slot.
    """
    capacity = buffer.data.shape[1]
    state_dim = buffer.data.shape[-1]
    valid_flat = event_valid.reshape(-1)
    slot_flat = event_slot.reshape(-1)
    data_flat = event_data.reshape(-1, state_dim)

    new_data = buffer.data
    new_ptr = buffer.ptr
    new_fill = buffer.fill
    # Slot 0 is unused (a drone "approaching gate 0" is the true-start
    # condition every env already trains on). Skip it to keep the buffer
    # densely packed on the slots we actually replay from.
    for g in range(1, n_gates):
        mask_g = valid_flat & (slot_flat == g)
        # Per-event rank in this slot's write batch. ``rank`` is 0-based
        # for valid entries, irrelevant for invalid ones (masked below).
        rank = jnp.cumsum(mask_g.astype(jnp.int32)) - 1
        idx = (buffer.ptr[g] + rank) % capacity
        # Map invalid entries to OOB (= ``capacity``) so the ``mode="drop"``
        # scatter discards them. This avoids a separate gather + boolean
        # mask round-trip.
        idx_masked = jnp.where(mask_g, idx, capacity)
        new_data_g = buffer.data[g].at[idx_masked].set(data_flat, mode="drop")
        n_added = jnp.sum(mask_g.astype(jnp.int32))
        new_ptr_g = (buffer.ptr[g] + n_added) % capacity
        new_fill_g = jnp.minimum(capacity, buffer.fill[g] + n_added)
        new_data = new_data.at[g].set(new_data_g)
        new_ptr = new_ptr.at[g].set(new_ptr_g)
        new_fill = new_fill.at[g].set(new_fill_g)
    return Phase2Buffer(data=new_data, ptr=new_ptr, fill=new_fill)


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
