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
* **pure-jax Phase 1 + Phase 2 reset curriculum** — the scan body calls
  rl_song's reset-perturbation adapter for Song 2023 §III-B Phase-1
  segment-init and Phase-2 successful-state replay. Per-env episode
  source codes are threaded across rollout boundaries so Phase-2 writes
  can avoid replay-buffer self-feeding.
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
from lsy_drone_racing.control.rl_song.policy import raw_to_env_action, raw_to_physical_action
from lsy_drone_racing.control.rl_song.reward import step_reward
from lsy_drone_racing.control.rl_song.rollout import (
    SRC_PHASE1_SEG,
    SRC_PHASE2_REPLAY,
    SRC_TRUE_START,
    Phase2Buffer,
    _apply_phase2_writes,
    _apply_reset_perturbation,
    _compute_phase2_event,
)
from lsy_drone_racing.control.rl_song.rollout import (
    RolloutStaticConfig as RLSongRolloutStaticConfig,
)
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
    reset_pos_perturb_m, reset_vel_perturb_mps, reset_yaw_perturb_rad : float
        Uniform-jitter half-widths applied to ``sim_data.states.{pos,vel,
        quat}`` of just-reset envs. All-zero disables the perturbation
        branch.
    segment_init_prob : float
        Bernoulli probability per just-reset env that the env is
        re-spawned at a random gate's entry waypoint (Song 2023 §III-B
        Phase 1). ``0.0`` disables seg-init entirely.
    segment_init_perturb_m, segment_init_vel_mps : float
        Spawn-jitter half-width (m) and traversal-axis speed (m/s) for
        seg-init re-spawns. Match the rl_song stage values so the
        respawn distribution is identical across stacks.
    """

    n_steps: int
    n_envs: int
    thrust_min: float
    thrust_max: float
    tangent_alpha_max_rad: float
    reward_cfg: RewardConfig
    reset_pos_perturb_m: float = 0.0
    reset_vel_perturb_mps: float = 0.0
    reset_yaw_perturb_rad: float = 0.0
    segment_init_prob: float = 0.0
    segment_init_perturb_m: float = 0.10
    segment_init_vel_mps: float = 0.0
    phase2_prob: float = 0.0
    phase2_warmup_steps: int = 0
    phase2_capacity_per_gate: int = 1


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
    # v125+: diagnostic outputs. Not consumed by SB3's rollout buffer or
    # GAE machinery — purely for wandb logging via the SB3 logger record
    # path. Aggregated in JitScanPPO.collect_rollouts and pushed to
    # ``reward/*`` and ``env/*`` namespaces in wandb.
    target_gate: Array  # (n_steps, n_envs) int, current_target after the step
    terminated: Array  # (n_steps, n_envs) bool
    finished: Array  # (n_steps, n_envs) bool — race-complete signal
    truncated: Array  # (n_steps, n_envs) bool
    reward_components: dict[str, Array]  # name -> (n_steps, n_envs) per-term reward
    p2_event_valid: Array  # (n_steps, n_envs) bool
    p2_event_slot: Array  # (n_steps, n_envs) int32
    p2_event_data: Array  # (n_steps, n_envs, phase2_state_dim)
    # Per-step episode-source code (int8, see SRC_* constants in
    # rl_song.rollout). Holds the source of the active episode AT this
    # step; on done steps it is still the dying episode's source (the
    # carry update to ``next_source`` happens after this output is built).
    # Used by JitScanPPO.collect_rollouts to break finish_rate down by
    # source so Phase-2-replay finishes don't mask true-start performance.
    source: Array  # (n_steps, n_envs) int8


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
    phase2_buffer: Phase2Buffer
    source: Array
    outputs: RLSBXRolloutOutputs


class _ScanCarry(NamedTuple):
    """Mutable scan carry for rollout collection."""

    env_data: EnvData
    prev_action: Array  # env-action 4-vec, (n_envs, ENV_ACTION_DIM)
    prev_physical_action: Array  # [tau_scaled, thrust], (n_envs, RAW_ACTION_DIM)
    rng_key: Array  # policy-sampling key
    # Independent PRNG stream for seg-init / drone-state perturbation.
    # Keeping it separate from the action key means a stage that
    # toggles seg-init does not shift the action-sampling sequence,
    # which would otherwise change every rollout's trajectory bit-for-bit
    # and invalidate run-to-run comparability.
    reset_rng_key: Array
    next_done: Array  # bool, (n_envs,) — done at end of previous step
    phase2_buffer: Phase2Buffer
    source: Array  # int8, (n_envs,)


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


def _to_rl_song_static_cfg(static_cfg: RLSBXRolloutStaticConfig) -> RLSongRolloutStaticConfig:
    """Build the rl_song :class:`RolloutStaticConfig` slice ``_apply_reset_perturbation`` reads.

    ``_apply_reset_perturbation`` only references reset-perturbation,
    seg-init, and ``reward_cfg.lookahead_entry_offset_m`` fields. The
    other rl_song fields (``n_steps``, ``n_envs``, ``thrust_*``, etc.)
    are required by the dataclass constructor but never read inside the
    reset path. We forward what overlaps and stub the rest with the
    rl_sbx values — the function never observes them.

    ``phase2_prob`` is the configured stage probability. The per-iteration
    warmup gate is applied by masking the buffer's ``fill`` at runtime in
    ``_seg_init_pure_jax``; this keeps the static config stable across the
    warmup boundary.
    """
    return RLSongRolloutStaticConfig(
        n_steps=static_cfg.n_steps,
        n_envs=static_cfg.n_envs,
        thrust_min=static_cfg.thrust_min,
        thrust_max=static_cfg.thrust_max,
        max_episode_steps=0,  # unused inside _apply_reset_perturbation
        reward_cfg=static_cfg.reward_cfg,
        reset_pos_perturb_m=static_cfg.reset_pos_perturb_m,
        reset_vel_perturb_mps=static_cfg.reset_vel_perturb_mps,
        reset_yaw_perturb_rad=static_cfg.reset_yaw_perturb_rad,
        tangent_alpha_max_rad=static_cfg.tangent_alpha_max_rad,
        segment_init_prob=static_cfg.segment_init_prob,
        segment_init_perturb_m=static_cfg.segment_init_perturb_m,
        segment_init_vel_mps=static_cfg.segment_init_vel_mps,
        phase2_prob=static_cfg.phase2_prob,
        phase2_capacity_per_gate=static_cfg.phase2_capacity_per_gate,
    )


def _seg_init_pure_jax(
    env_data: EnvData,
    done_mask: Array,
    rng_key: Array,
    phase2_buffer: Phase2Buffer,
    effective_phase2_prob: Array,
    static_cfg: RLSBXRolloutStaticConfig,
) -> tuple[EnvData, Array, Array, Array, Array]:
    """Apply drone-state perturbation + reset curriculum to just-reset envs.

    Thin adapter around :func:`rl_song.rollout._apply_reset_perturbation`
    that keeps all reset-source masks so the SBX scan can mirror rl_song's
    source tracking and Phase-2 prev-action restoration.

    Parameters
    ----------
    env_data : EnvData
        Post-autoreset env state. Worlds that did not just reset are
        left untouched by the underlying perturbation routine.
    done_mask : Array, shape (n_envs,), bool
        Mask of envs that completed on the last step.
    rng_key : Array, shape (2,)
        PRNG key for the perturbation + seg-init draws; consumed and
        returned advanced.
    phase2_buffer : Phase2Buffer
        Successful-state replay buffer carried from the previous rollout.
    effective_phase2_prob : Array, shape ()
        Runtime warmup-gated Phase-2 probability. A zero value masks the
        buffer empty without changing static JIT cache keys.
    static_cfg : RLSBXRolloutStaticConfig
        Sources seg-init / perturbation knobs.

    Returns:
    -------
    env_data : EnvData
        Env state with the perturbation, seg-init, and Phase-2 overrides
        applied.
    rng_key : Array
        Advanced PRNG key.

    Notes:
    -----
    rl_song's helper branches on the static configured probability. To
    avoid recompiling when the warmup gate opens, SBX keeps that branch
    compiled and masks ``buffer.fill`` to zero while the effective runtime
    probability is zero.
    """
    rl_song_cfg = _to_rl_song_static_cfg(static_cfg)
    replay_enabled = effective_phase2_prob > 0.0
    gated_buffer = Phase2Buffer(
        data=phase2_buffer.data,
        ptr=phase2_buffer.ptr,
        fill=jnp.where(replay_enabled, phase2_buffer.fill, jnp.zeros_like(phase2_buffer.fill)),
    )
    env_data, rng_key, do_seg, do_phase2, replay_prev_action = _apply_reset_perturbation(
        env_data, done_mask, rng_key, gated_buffer, rl_song_cfg
    )
    return env_data, rng_key, do_seg, do_phase2, replay_prev_action


@partial(jax.jit, static_argnames=("env_step_fn", "env_reset_fn", "static_cfg"))
def scan_rollout(
    env_data: EnvData,
    actor_params: Any,
    vf_params: Any,
    actor_normalizer: obs_encoding.NormalizerState,
    critic_normalizer: obs_encoding.NormalizerState,
    prev_action_env_4vec: Array,
    rng_key: Array,
    reset_rng_key: Array,
    next_done: Array,
    phase2_buffer: Phase2Buffer,
    source: Array,
    effective_phase2_prob: Array,
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
    reset_rng_key : Array, shape (2,)
        Independent PRNG key used for the in-scan seg-init / drone-state
        perturbation. Separated from ``rng_key`` so toggling seg-init
        does not perturb the policy's action sequence.
    next_done : Array, shape (n_envs,), bool
        Done flag carried from the previous rollout. Becomes the
        ``episode_starts[0]`` of this rollout.
    phase2_buffer : Phase2Buffer
        Phase-2 successful-state buffer carried by the env wrapper.
    source : Array, shape (n_envs,), int8
        Per-env episode-source code carried across rollout boundaries.
    effective_phase2_prob : Array, shape ()
        Runtime warmup-gated Phase-2 probability.
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
    ``env_reset_fn(stepped_data, seed=None, mask=done_bool)`` and then
    fed through :func:`_seg_init_pure_jax` for the Song 2023 §III-B
    Phase-1 seg-init re-spawn + drone-state perturbation. Phase-2
    successful-state replay is read during resets and written after the
    scan body, matching rl_song's no in-rollout feedback pattern.
    """
    _validate_inputs(prev_action_env_4vec, next_done, reset_rng_key, source, static_cfg)

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
        physical_action = raw_to_physical_action(
            raw_action,
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
        reward, components = step_reward(
            next_env_obs,
            env_obs,
            terminated,
            truncated,
            finished,
            gate_just_passed,
            static_cfg.reward_cfg,
            physical_action=physical_action,
            prev_physical_action=carry.prev_physical_action,
        )

        done_bool = terminated | truncated

        # Autoreset done envs. ``env_reset_fn`` is wrapped so a no-op
        # mask leaves the data untouched; the unconditional call keeps
        # the scan trace shape-stable.
        reset_data, _ = env_reset_fn(stepped_data, None, done_bool)

        # Phase-1 seg-init + drone-state perturbation on the just-reset
        # envs. ``_seg_init_pure_jax`` is a no-op on envs with
        # ``done_bool=False`` and a fast static no-op when both seg-init
        # and the perturbation knobs are disabled in ``static_cfg``.
        (reset_data, reset_rng_key, do_seg, do_phase2, replay_prev_action) = _seg_init_pure_jax(
            reset_data,
            done_bool,
            carry.reset_rng_key,
            carry.phase2_buffer,
            effective_phase2_prob,
            static_cfg,
        )

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

        p2_event_valid, p2_event_slot, p2_event_data = _compute_phase2_event(
            stepped_data=stepped_data,
            next_env_obs=next_env_obs,
            env_action=env_action,
            current_target=current_target,
            gate_just_passed=gate_just_passed,
            done_bool=done_bool,
            source=carry.source,
        )

        # True-start / Phase-1 resets begin with no prior command; Phase-2
        # replays restore the action stored alongside the replayed state so
        # the autoregressive input matches the respawned pose.
        next_prev_action = jnp.where(done_bool[:, None], jnp.zeros_like(env_action), env_action)
        next_prev_action = jnp.where(do_phase2[:, None], replay_prev_action, next_prev_action)
        next_prev_physical_action = jnp.where(
            done_bool[:, None], jnp.zeros_like(physical_action), physical_action
        )

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
            target_gate=current_target,
            terminated=terminated,
            finished=finished,
            truncated=truncated,
            reward_components=components,
            source=carry.source,
            p2_event_valid=p2_event_valid,
            p2_event_slot=p2_event_slot,
            p2_event_data=p2_event_data,
        )
        next_carry = _ScanCarry(
            env_data=reset_data,
            prev_action=next_prev_action,
            prev_physical_action=next_prev_physical_action,
            rng_key=rng_key,
            reset_rng_key=reset_rng_key,
            next_done=done_bool,
            phase2_buffer=carry.phase2_buffer,
            source=next_source,
        )
        return next_carry, transition

    initial_carry = _ScanCarry(
        env_data=env_data,
        prev_action=prev_action_env_4vec,
        prev_physical_action=jnp.zeros((static_cfg.n_envs, RAW_ACTION_DIM), dtype=jnp.float32),
        rng_key=rng_key,
        reset_rng_key=reset_rng_key,
        next_done=next_done,
        phase2_buffer=phase2_buffer,
        source=source,
    )
    final_carry, stacked_outputs = jax.lax.scan(
        scan_step, initial_carry, None, length=static_cfg.n_steps
    )
    n_gates_python_int = int(env_data.gates_pos.shape[1])
    updated_buffer = _apply_phase2_writes(
        final_carry.phase2_buffer,
        stacked_outputs.p2_event_valid,
        stacked_outputs.p2_event_slot,
        stacked_outputs.p2_event_data,
        n_gates_python_int,
    )
    final_carry = final_carry._replace(phase2_buffer=updated_buffer)

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
        phase2_buffer=final_carry.phase2_buffer,
        source=final_carry.source,
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
    reset_pos_perturb_m: float = 0.0,
    reset_vel_perturb_mps: float = 0.0,
    reset_yaw_perturb_rad: float = 0.0,
    segment_init_prob: float = 0.0,
    segment_init_perturb_m: float = 0.10,
    segment_init_vel_mps: float = 0.0,
    phase2_prob: float = 0.0,
    phase2_warmup_steps: int = 0,
    phase2_capacity_per_gate: int = 1,
) -> RLSBXRolloutStaticConfig:
    """Construct a :class:`RLSBXRolloutStaticConfig` with the project defaults.

    Thin convenience constructor — keeps call sites in ``train.py`` /
    ``jit_scan_ppo.py`` from re-stating the default ``tangent_alpha_max_rad``
    every time. Seg-init / perturbation knobs default to disabled so a
    caller that does not pass them gets the milestone-1 behaviour.
    """
    return RLSBXRolloutStaticConfig(
        n_steps=int(n_steps),
        n_envs=int(n_envs),
        thrust_min=float(thrust_min),
        thrust_max=float(thrust_max),
        tangent_alpha_max_rad=float(tangent_alpha_max_rad),
        reward_cfg=reward_cfg,
        reset_pos_perturb_m=float(reset_pos_perturb_m),
        reset_vel_perturb_mps=float(reset_vel_perturb_mps),
        reset_yaw_perturb_rad=float(reset_yaw_perturb_rad),
        segment_init_prob=float(segment_init_prob),
        segment_init_perturb_m=float(segment_init_perturb_m),
        segment_init_vel_mps=float(segment_init_vel_mps),
        phase2_prob=float(phase2_prob),
        phase2_warmup_steps=int(phase2_warmup_steps),
        phase2_capacity_per_gate=int(phase2_capacity_per_gate),
    )


def _validate_inputs(
    prev_action_env_4vec: Array,
    next_done: Array,
    reset_rng_key: Array,
    source: Array,
    static_cfg: RLSBXRolloutStaticConfig,
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
    if source.shape != (n_envs,):
        raise ValueError(f"source must have shape {(n_envs,)}; got {source.shape}")
    if source.dtype != jnp.int8:
        raise ValueError(f"source must have dtype int8; got {source.dtype}")
    if reset_rng_key.shape != (2,):
        raise ValueError(f"reset_rng_key must have shape (2,); got {reset_rng_key.shape}")
