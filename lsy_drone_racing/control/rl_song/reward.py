"""Song-2023 dense reward for the RL drone-racing controller.

The function in this module replaces the sparse reward emitted by
``VecDroneRaceEnv.step``. It consumes the post-step observation, the previous
observation, and terminal flags after ``VecDroneRaceEnv`` has squeezed the
single-drone axis, so every array has a leading ``n_envs`` dimension.

Notes
-----
``env_obs["gates_pos"]`` is intentionally insufficient for the progress term:
the racing env reports nominal gate poses until a gate is revealed by the
sensor-range mask. The wrapper must therefore pass unmasked true gate positions
through ``true_gates_pos``. If this argument is omitted, the function falls back
to ``env_obs["gates_pos"]`` for callers that intentionally use masked poses.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from lsy_drone_racing.control.rl_song.config import RewardConfig
from lsy_drone_racing.control.rl_song.obs import GATE_HALF_SIZE_M, _quat_to_matrix

# Squared-meter tolerance for detecting positions on the gate guidance axis.
GUIDANCE_AXIS_EPS_M2: float = 1e-8
# Dimensionless denominator floor for the aperture-normalized guidance radius.
GUIDANCE_DENOM_EPS: float = 1e-8
# Squared-length floor for the parametric projection in
# :func:`_gate_frame_edge_dist_sq` (guards against degenerate zero-length
# edges; gate corners are well-separated so this only fires on numerical noise).
SEGMENT_AB_SQ_EPS: float = 1e-12

# Gate opening corners in gate-local coords (x_through = 0, ±h_y, ±h_z).
# Same ordering as ``obs._GATE_CORNERS_LOCAL`` so downstream edge pairs
# stay consistent. Corner indices:
#   0: (+h_y, +h_z) — top-right     1: (+h_y, -h_z) — bottom-right
#   2: (-h_y, +h_z) — top-left      3: (-h_y, -h_z) — bottom-left
_GATE_HALF_Y, _GATE_HALF_Z = GATE_HALF_SIZE_M
_GATE_FRAME_CORNERS_LOCAL: Array = jnp.asarray(
    [
        [0.0, +_GATE_HALF_Y, +_GATE_HALF_Z],
        [0.0, +_GATE_HALF_Y, -_GATE_HALF_Z],
        [0.0, -_GATE_HALF_Y, +_GATE_HALF_Z],
        [0.0, -_GATE_HALF_Y, -_GATE_HALF_Z],
    ],
    dtype=jnp.float32,
)
# Edge endpoint index pairs (4 sides of the square opening): right
# vertical (0-1), left vertical (2-3), top horizontal (0-2), bottom
# horizontal (1-3).
_GATE_FRAME_EDGE_INDICES: Array = jnp.asarray([[0, 1], [2, 3], [0, 2], [1, 3]], dtype=jnp.int32)


def _gate_frame_edge_dist_sq(pos: Array, gates_pos: Array, gates_quat: Array) -> Array:
    """Return squared distance from drone position to each gate frame edge.

    For each (env, gate) pair, computes the four opening corners in the
    world frame using the gate's xyzw quaternion, builds the four edge
    line segments, and returns the per-edge min squared distance from
    the drone position to that segment (point-to-segment projection,
    clamped to the segment endpoints).

    Parameters
    ----------
    pos : Array, shape (n_envs, 3)
        World-frame drone position.
    gates_pos : Array, shape (n_envs, n_gates, 3)
    gates_quat : Array, shape (n_envs, n_gates, 4)
        xyzw quaternions.

    Returns
    -------
    Array, shape (n_envs, n_gates, 4)
        Squared distance per edge.
    """
    # Rotate the four canonical corners into each gate's world frame:
    # corners_world[e, g, c] = R(quat[e, g]) @ corner_local[c] + gate_pos[e, g].
    rot = jax.vmap(jax.vmap(_quat_to_matrix))(gates_quat)  # (n_envs, n_gates, 3, 3)
    corners_world = jnp.einsum("egij,cj->egci", rot, _GATE_FRAME_CORNERS_LOCAL)
    corners_world = corners_world + gates_pos[..., None, :]  # (n_envs, n_gates, 4, 3)

    # Edge endpoints: a = corners[indices[:, 0]], b = corners[indices[:, 1]].
    a = corners_world[:, :, _GATE_FRAME_EDGE_INDICES[:, 0], :]  # (n_envs, n_gates, 4, 3)
    b = corners_world[:, :, _GATE_FRAME_EDGE_INDICES[:, 1], :]
    ab = b - a
    ap = pos[:, None, None, :] - a  # (n_envs, n_gates, 4, 3)
    ab_sq = jnp.sum(ab * ab, axis=-1)  # (n_envs, n_gates, 4)
    t = jnp.sum(ap * ab, axis=-1) / jnp.maximum(ab_sq, SEGMENT_AB_SQ_EPS)
    t_clamped = jnp.clip(t, 0.0, 1.0)
    closest = a + t_clamped[..., None] * ab
    diff = pos[:, None, None, :] - closest
    return jnp.sum(diff * diff, axis=-1)  # (n_envs, n_gates, 4)


def _gate_phi(pos: Array, gate_pos: Array, gate_quat: Array, reward_cfg: RewardConfig) -> Array:
    """Compute the Δ-potential gate guidance scalar Φ(pos | gate).

    Φ = aperture_score(y,z) · sigmoid(-x / guide_kx)
    in the target gate's local frame (x = forward through gate, y/z =
    aperture coordinates). Monotonic front-to-back along the gate normal,
    so ΔΦ over a perfectly centered pass integrates to ~1 and hovering
    integrates to ~0.

    Parameters
    ----------
    pos : Array, shape (n_envs, 3)
        World-frame position.
    gate_pos : Array, shape (n_envs, 3)
        World-frame target-gate position.
    gate_quat : Array, shape (n_envs, 4)
        Target-gate xyzw quaternion.
    reward_cfg : RewardConfig
        Source of ``guide_k2`` (aperture spread base) and ``guide_kx``
        (traversal sigmoid scale).

    Returns
    -------
    phi : Array, shape (n_envs,)
        Scalar potential in ``[0, 1]``.
    """
    rot_gw = _quat_to_matrix(gate_quat)
    pos_local = jnp.einsum("nji,nj->ni", rot_gw, pos - gate_pos)
    x = pos_local[..., 0]
    y = pos_local[..., 1]
    z = pos_local[..., 2]
    yz_sq = jnp.square(y) + jnp.square(z)
    h_wp, w_wp = GATE_HALF_SIZE_M
    denom = jnp.square(z / h_wp) + jnp.square(y / w_wp)
    spread = jnp.where(
        yz_sq > GUIDANCE_AXIS_EPS_M2,
        reward_cfg.guide_k2 * jnp.sqrt(yz_sq / jnp.maximum(denom, GUIDANCE_DENOM_EPS)),
        reward_cfg.guide_k2,
    )
    aperture = jnp.exp(-yz_sq / (2.0 * spread))
    traversal = jax.nn.sigmoid(-x / reward_cfg.guide_kx)
    return aperture * traversal


def step_reward(
    env_obs: dict[str, Array],
    prev_env_obs: dict[str, Array],
    terminated: Array,
    truncated: Array,
    finished: Array,
    gate_just_passed: Array,
    reward_cfg: RewardConfig,
    *,
    true_gates_pos: Array | None = None,
    true_gates_quat: Array | None = None,
    true_obstacles_pos: Array | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Compute one vectorized Song-style reward step.

    Parameters
    ----------
    env_obs : dict[str, Array]
        Current observation after ``VecDroneRaceEnv.step``. Values have a
        leading ``n_envs`` axis; ``pos`` is shape ``(n_envs, 3)``.
    prev_env_obs : dict[str, Array]
        Previous-step observation with the same structure as ``env_obs``.
    terminated : Array, shape (n_envs,)
        Environment termination flags after the step.
    truncated : Array, shape (n_envs,)
        Timeout flags after the step. Timeouts receive no terminal bonus or
        crash penalty.
    finished : Array, shape (n_envs,)
        Boolean mask indicating that the race finished on this transition.
    gate_just_passed : Array, shape (n_envs,)
        Boolean mask indicating that ``target_gate`` advanced on this
        transition.
    reward_cfg : RewardConfig
        Reward weights. In particular ``omega_coef`` is the 50 Hz coefficient.
    true_gates_pos : Array, shape (n_envs, n_gates, 3), optional
        Unmasked true gate positions. The wrapper should pass this from
        ``env.data.gates_pos``.
    true_gates_quat : Array, shape (n_envs, n_gates, 4), optional
        Unmasked true gate xyzw quaternions. The wrapper should pass this
        from ``env.data.gates_quat``. If omitted, ``env_obs["gates_quat"]``
        is used.
    true_obstacles_pos : Array, shape (n_envs, n_obstacles, 3), optional
        Unmasked obstacle positions. If omitted, ``env_obs["obstacles_pos"]``
        is used.

    Returns
    -------
    reward : Array, shape (n_envs,)
        Total replacement reward.
    components : dict[str, Array]
        Per-component rewards with keys ``r_prog``, ``r_omega``, ``r_obs``,
        ``r_gate_bonus``, ``r_exit_vel``, ``r_terminal``, ``r_time``,
        ``r_vel``, and ``r_guid``. Every value has shape ``(n_envs,)``.

    Notes
    -----
    ``r_obs`` and ``r_gate_frame`` use the already-masked ``env_obs``
    fields (post upstream PR #91, ``race_core.obs`` masks gate and
    obstacle positions between the wobbled physics state and the
    Layer-1 nominal snapshot via the visited flags), so the actor is
    graded against the same poses it observes.
    """
    _ = truncated, true_obstacles_pos
    gates_pos = env_obs["gates_pos"] if true_gates_pos is None else true_gates_pos
    gates_quat = env_obs["gates_quat"] if true_gates_quat is None else true_gates_quat

    safety_gates_pos = env_obs["gates_pos"]
    safety_gates_quat = env_obs["gates_quat"]
    safety_obstacles_pos = env_obs["obstacles_pos"]

    prev_target = prev_env_obs["target_gate"]
    current_target = env_obs["target_gate"]
    target_idx = jnp.where(prev_target >= 0, prev_target, jnp.maximum(current_target, 0))
    env_idx = jnp.arange(gates_pos.shape[0])
    gate_pos = gates_pos[env_idx, target_idx]
    gate_quat = gates_quat[env_idx, target_idx]

    prev_pos = prev_env_obs["pos"]
    pos = env_obs["pos"]
    # v20: Song-2023 / Kaufmann-2023 velocity-projection progress, gated
    # by ``use_velocity_progress``. The drone enters each gate's local
    # frame from x_local < 0, passes the aperture at x_local ≈ 0, and
    # exits at x_local > 0, so the gate-frame x-component of the world-
    # frame displacement is the signed progress along the gate normal.
    # Reward direction is fixed in gate frame (the gate's traversal
    # axis), rather than rotating with the line from drone to gate as
    # in the legacy distance-delta formulation. The two are equal in
    # the limit of small displacements along that line; they diverge
    # when motion has a lateral component, where velocity-projection
    # only credits the traversal-aligned part — encouraging cleaner
    # nose-first passes.
    if reward_cfg.use_velocity_progress:
        rot_gw = _quat_to_matrix(gate_quat)
        delta_local = jnp.einsum("nji,nj->ni", rot_gw, pos - prev_pos)
        r_prog = reward_cfg.progress_coef * delta_local[..., 0]
    else:
        r_prog = jnp.linalg.norm(gate_pos - prev_pos, axis=-1)
        r_prog = reward_cfg.progress_coef * (r_prog - jnp.linalg.norm(gate_pos - pos, axis=-1))

    r_omega = -reward_cfg.omega_coef * jnp.linalg.norm(env_obs["ang_vel"], ord=1, axis=-1)

    # v10: forward-flight bias (Liu eq. 8). Penalize lateral and backward
    # body-frame velocity. Symmetric ``r_prog`` cannot distinguish a drone
    # that slides sideways toward a gate from one that flies through nose-
    # first; this term breaks that symmetry without touching ``r_prog``.
    rot_wb = _quat_to_matrix(env_obs["quat"])
    vel_body = jnp.einsum("nji,nj->ni", rot_wb, env_obs["vel"])
    vel_lat = vel_body[..., 1]
    vel_back = jnp.minimum(vel_body[..., 0], 0.0)
    r_vel = jnp.where(
        jnp.asarray(reward_cfg.use_vel_shaping, dtype=bool),
        reward_cfg.vel_lat_coef * jnp.square(vel_lat)
        + reward_cfg.vel_back_coef * jnp.square(vel_back),
        jnp.zeros_like(r_prog),
    )

    # Gate guidance. Two formulations live behind this branch:
    #
    # v10/v11/v13A — static Liu eqs. 6-7 field, ``r_guid <= 0``. The
    # original comment described front-side shaping as "attractive
    # toward the centerline", but as written the on-axis penalty peaks
    # on the approach line; mechanically the term is a *localized loiter
    # penalty* that pushes the policy to pass through fast or step away.
    # v13B — Δ-potential shaping, ``r_guid = guide_coef · (Φ_t − Φ_{t-1})``
    # with Φ monotonic front-to-back. Pays only on transitions that move
    # the drone toward aperture-centered traversal; hovering produces
    # zero r_guid. Selected by ``reward_cfg.use_guide_delta_phi``.
    if reward_cfg.use_guide_delta_phi:
        phi_curr = _gate_phi(pos, gate_pos, gate_quat, reward_cfg)
        phi_prev = _gate_phi(prev_pos, gate_pos, gate_quat, reward_cfg)
        r_guid_raw = reward_cfg.guide_coef * (phi_curr - phi_prev)
    else:
        rot_gw = _quat_to_matrix(gate_quat)
        pos_local = jnp.einsum("nji,nj->ni", rot_gw, pos - gate_pos)
        x, y, z = pos_local[..., 0], pos_local[..., 1], pos_local[..., 2]
        guide_window = jnp.maximum(1.0 - jnp.sign(x) * x / reward_cfg.guide_k0, 0.0)
        h_wp, w_wp = GATE_HALF_SIZE_M
        denom = jnp.square(z / h_wp) + jnp.square(y / w_wp)
        yz_sq = jnp.square(y) + jnp.square(z)
        guide_spread_axis = reward_cfg.guide_k2 * (1.0 + jnp.square(guide_window))
        guide_spread = jnp.where(
            yz_sq > GUIDANCE_AXIS_EPS_M2,
            guide_spread_axis * jnp.sqrt(yz_sq / jnp.maximum(denom, GUIDANCE_DENOM_EPS)),
            guide_spread_axis,
        )
        guide_front = reward_cfg.guide_k1 * jnp.exp(-yz_sq / (2.0 * guide_spread))
        guide_back = 1.0 - jnp.exp(-yz_sq / (2.0 * guide_spread))
        guide_field = jnp.where(x > 0.0, guide_front, guide_back)
        r_guid_raw = -reward_cfg.guide_coef * jnp.square(guide_window) * guide_field
    r_guid = jnp.where(
        jnp.asarray(reward_cfg.use_guide, dtype=bool), r_guid_raw, jnp.zeros_like(r_prog)
    )

    # v32a: XY-only distance. ``obstacles_pos`` stores the top marker of
    # a vertical capsule (see ``config/level3.toml`` and
    # ``envs/assets/obstacle.xml``) at z ≈ 1.55, but the capsule
    # extends downward through the drone's flight altitude (~0.7 m).
    # Full 3D distance from the drone to that top marker is dominated
    # by the ~0.85 m vertical offset, giving a Gaussian barrier of
    # ~exp(-8) ≈ 3e-4 even when the drone is right next to the
    # capsule horizontally — which is exactly why v32's r_obs stayed
    # near zero despite dropping the visited mask. Dropping z treats
    # obstacles as infinite vertical poles, matching the actual capsule
    # geometry well over the racing altitude range.
    # v33: distance computed against ``safety_obstacles_pos`` (true for
    # visited, placed for unvisited) so the actor's per-step gradient is
    # against the obstacle it can actually observe. v32a used
    # ``obstacles_pos`` unconditionally, which graded the policy on the
    # post-wobble true location even when the actor obs was showing the
    # pre-wobble placed location (up to 0.15 m off) — that mismatch is
    # roughly the obstacle_sigma half-width, so the avoidance gradient
    # was being optimized against a feature distribution shifted from
    # what the policy could see.
    obstacle_delta_xy = pos[:, None, :2] - safety_obstacles_pos[:, :, :2]
    obstacle_dist_sq = jnp.sum(jnp.square(obstacle_delta_xy), axis=-1)
    # v32: drop the ``obstacle_active = 1 - obstacles_visited`` mask. The
    # old mask zeroed the penalty exactly when ``obstacles_visited``
    # flipped True (i.e. when the drone entered sensor range and was
    # finally close enough for the small-sigma barrier to matter), so
    # the policy got no gradient toward obstacle avoidance. v32
    # evaluates the barrier unconditionally so the drone is rewarded
    # for keeping its distance whether or not the obstacle has been
    # "discovered" by the sensor.
    obstacle_barrier = jnp.exp(-obstacle_dist_sq / jnp.square(reward_cfg.obstacle_sigma))
    r_obs = -reward_cfg.obstacle_weight * jnp.sum(obstacle_barrier, axis=-1)

    # v32: Gate-frame soft barrier. Each gate's 4 opening corners (from
    # ``obs.GATE_HALF_SIZE_M``) form 4 line-segment edges in world frame.
    # We compute the per-edge min squared distance from the drone to the
    # segment, apply a Gaussian barrier, and sum over edges and gates.
    # v33: applied with two changes vs v32a:
    #   * Distance is to ``safety_gates_pos`` / ``safety_gates_quat``
    #     (true for visited, placed for unvisited), so the actor's
    #     per-step avoidance gradient is against the frame location it
    #     can actually observe. v32a graded against true post-wobble
    #     pose, which moved by up to 0.15 m XY (and 0.20 rad yaw) from
    #     the actor's observed pose for unvisited gates — comparable
    #     to one ``gate_frame_sigma`` step and enough to invert the
    #     gradient at close approach.
    #   * Masked to the gates in ``{target_idx - 1, target_idx,
    #     target_idx + 1}``. v32a summed over all 4 gates, including
    #     gates outside the actor's ``N_FUTURE_GATES=2`` observation
    #     window. Those far-gate contributions were near zero on
    #     average but their gradient (when occasionally aligned with
    #     the racing line) pulled the actor off the natural path. The
    #     window keeps the just-passed exit frame (target_idx - 1) for
    #     post-pass clearance, the current target frame for the
    #     active approach, and the next target frame for look-ahead.
    # With ``gate_frame_sigma=0.08 m`` and passage half-width 0.20 m the
    # barrier at passage center is exp(-(0.20/0.08)^2) ≈ 0.002
    # (negligible) and at 0.10 m from the rim is exp(-(0.10/0.08)^2)
    # ≈ 0.21 (steep avoidance gradient).
    edge_dist_sq = _gate_frame_edge_dist_sq(
        pos, safety_gates_pos, safety_gates_quat
    )  # (n_envs, n_gates, 4)
    gate_frame_barrier = jnp.exp(-edge_dist_sq / jnp.square(reward_cfg.gate_frame_sigma))
    n_gates_total = safety_gates_pos.shape[1]
    gate_indices = jnp.arange(n_gates_total)
    # Window mask: gate g is active iff ``|g - target_idx| <= 1`` AND
    # ``0 <= g <= n_gates - 1``. Edge cases:
    #   * target_idx == 0   -> window = {0, 1}        (no previous gate)
    #   * target_idx == N-1 -> window = {N-2, N-1}    (no next gate)
    #   * finished step -> ``prev_target`` was the last gate ``N-1``,
    #     so ``target_idx = N-1`` and the window is ``{N-2, N-1}``.
    #     Dense terms are *not* zeroed on finish steps (only on crash
    #     steps via ``crash_mask = terminated & ~finished`` below), so
    #     r_gate_frame is a legitimate per-step penalty on the finish
    #     step itself — but the drone is geometrically past the last
    #     frame's plane and the penalty is negligible.
    window_lo = jnp.maximum(target_idx - 1, 0)[:, None]
    window_hi = jnp.minimum(target_idx + 1, n_gates_total - 1)[:, None]
    gate_window = (gate_indices[None, :] >= window_lo) & (gate_indices[None, :] <= window_hi)
    gate_frame_barrier = gate_frame_barrier * gate_window[..., None].astype(
        gate_frame_barrier.dtype
    )
    r_gate_frame = -reward_cfg.gate_frame_weight * jnp.sum(gate_frame_barrier, axis=(-1, -2))

    # Per-gate jackpot scaling (v7). At a crossing, ``target_idx`` is still the
    # pre-step target index because of the ``prev_target >= 0`` branch above,
    # which is the index of the gate just passed. Scaling by ``(target_idx +
    # 1)`` makes gate 1 worth 1x, gate 2 worth 2x, ..., gate 4 worth 4x of the
    # base ``gate_pass_bonus``. The intent is to pull the policy through the
    # harder late-gate transitions where v5/v6 plateaued.
    gate_bonus_weight = jnp.asarray(reward_cfg.gate_pass_bonus, dtype=pos.dtype)
    gate_bonus_scale = jnp.where(
        jnp.asarray(reward_cfg.scale_gate_bonus_by_index, dtype=bool),
        target_idx.astype(pos.dtype) + 1.0,
        jnp.ones_like(r_prog),
    )
    gate_bonus_enabled = jnp.asarray(reward_cfg.use_gate_pass_bonus, dtype=bool)
    r_gate_bonus = jnp.where(
        gate_bonus_enabled & gate_just_passed,
        gate_bonus_weight * gate_bonus_scale,
        jnp.zeros_like(r_prog),
    )

    # v28: exit-velocity bonus at gate-pass. See
    # ``RewardConfig.use_exit_vel_bonus`` for motivation. Computes the
    # signed projection of world-frame velocity onto the unit vector from
    # the drone's current position to the new target gate (i.e. the gate
    # that comes *after* the one just passed). Disabled on the finish step
    # (``current_target == -1``) where there is no "next" gate.
    new_target = current_target
    n_gates = gates_pos.shape[1]
    new_target_clamped = jnp.clip(new_target, 0, n_gates - 1)
    next_gate_pos_for_exit = gates_pos[env_idx, new_target_clamped]
    diff_to_next = next_gate_pos_for_exit - pos
    norm_to_next = jnp.linalg.norm(diff_to_next, axis=-1, keepdims=True)
    direction_to_next = diff_to_next / jnp.maximum(norm_to_next, 1e-6)
    v_to_next = jnp.sum(env_obs["vel"] * direction_to_next, axis=-1)
    # v33: clip ``v_to_next`` to ``±exit_vel_clip_mps``. With v33's
    # ``exit_vel_coef=10.0`` an unclipped outlier (e.g. ~12 m/s
    # transient on a hard pitch) would mint a +120 one-shot reward at
    # gate pass, which is comparable to a full gate's jackpot and would
    # spike value-function targets. Clipping at 5 m/s leaves a clean
    # racing-speed exit (3-5 m/s aligned) paying +30 to +50, while
    # capping outliers at the same +50 ceiling.
    v_to_next_clipped = jnp.clip(
        v_to_next, -reward_cfg.exit_vel_clip_mps, reward_cfg.exit_vel_clip_mps
    )
    exit_vel_active = (
        jnp.asarray(reward_cfg.use_exit_vel_bonus, dtype=bool)
        & gate_just_passed
        & (new_target >= 0)
    )
    r_exit_vel = jnp.where(
        exit_vel_active, reward_cfg.exit_vel_coef * v_to_next_clipped, jnp.zeros_like(r_prog)
    )

    r_finish = jnp.where(finished, reward_cfg.finish_bonus, jnp.zeros_like(r_prog))
    r_crash = jnp.where(terminated & ~finished, -reward_cfg.crash_penalty, jnp.zeros_like(r_prog))
    r_terminal = r_finish + r_crash

    # v8: per-step time penalty to break the hover Q=0 attractor. Without it,
    # a random-init policy on a randomized track collects 0 progress reward
    # on average and times out at the episode cap, leaving "do nothing" as
    # the best policy under the entropy bonus. See ``RewardConfig.time_penalty``.
    r_time = jnp.full_like(r_prog, -reward_cfg.time_penalty)

    # v32a: zero the position-dependent dense terms on a crash step.
    # ``race_core`` warps a disabled drone to ``[-1, -1, -1]`` *before*
    # producing the post-step observation, so ``pos`` on the crash step
    # is the warp location, not the collision location. r_prog computed
    # against that warp can give a large spurious positive value (drone
    # "fell back" near gate 0) that PPO would otherwise treat as a
    # reward for crashing. The Gaussian-barrier terms (r_obs,
    # r_gate_frame, r_guid) all give ~0 at the warp because it's far
    # from any track feature, but we zero them too for consistency and
    # to avoid the dependency on warp-location numerics. r_terminal
    # (including r_crash) is preserved — that's the signal we want on
    # crash. r_omega / r_time / r_vel are independent of position and
    # are also preserved.
    crash_mask = terminated & ~finished
    not_crash = ~crash_mask
    r_prog = jnp.where(not_crash, r_prog, jnp.zeros_like(r_prog))
    r_obs = jnp.where(not_crash, r_obs, jnp.zeros_like(r_obs))
    r_gate_frame = jnp.where(not_crash, r_gate_frame, jnp.zeros_like(r_gate_frame))
    r_guid = jnp.where(not_crash, r_guid, jnp.zeros_like(r_guid))

    components = {
        "r_prog": r_prog,
        "r_omega": r_omega,
        "r_obs": r_obs,
        "r_gate_frame": r_gate_frame,
        "r_gate_bonus": r_gate_bonus,
        "r_exit_vel": r_exit_vel,
        "r_terminal": r_terminal,
        "r_time": r_time,
        "r_vel": r_vel,
        "r_guid": r_guid,
    }
    reward = (
        r_prog
        + r_omega
        + r_obs
        + r_gate_frame
        + r_gate_bonus
        + r_exit_vel
        + r_terminal
        + r_time
        + r_vel
        + r_guid
    )
    return reward, components
