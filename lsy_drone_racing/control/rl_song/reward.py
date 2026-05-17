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
    """
    _ = truncated
    gates_pos = env_obs["gates_pos"] if true_gates_pos is None else true_gates_pos
    gates_quat = env_obs["gates_quat"] if true_gates_quat is None else true_gates_quat
    obstacles_pos = env_obs["obstacles_pos"] if true_obstacles_pos is None else true_obstacles_pos

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

    obstacle_delta = pos[:, None, :] - obstacles_pos
    obstacle_dist_sq = jnp.sum(jnp.square(obstacle_delta), axis=-1)
    obstacle_active = 1.0 - env_obs["obstacles_visited"].astype(jnp.float32)
    obstacle_barrier = jnp.exp(-obstacle_dist_sq / jnp.square(reward_cfg.obstacle_sigma))
    r_obs = -reward_cfg.obstacle_weight * jnp.sum(obstacle_barrier * obstacle_active, axis=-1)

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
    exit_vel_active = (
        jnp.asarray(reward_cfg.use_exit_vel_bonus, dtype=bool)
        & gate_just_passed
        & (new_target >= 0)
    )
    r_exit_vel = jnp.where(
        exit_vel_active, reward_cfg.exit_vel_coef * v_to_next, jnp.zeros_like(r_prog)
    )

    r_finish = jnp.where(finished, reward_cfg.finish_bonus, jnp.zeros_like(r_prog))
    r_crash = jnp.where(terminated & ~finished, -reward_cfg.crash_penalty, jnp.zeros_like(r_prog))
    r_terminal = r_finish + r_crash

    # v8: per-step time penalty to break the hover Q=0 attractor. Without it,
    # a random-init policy on a randomized track collects 0 progress reward
    # on average and times out at the episode cap, leaving "do nothing" as
    # the best policy under the entropy bonus. See ``RewardConfig.time_penalty``.
    r_time = jnp.full_like(r_prog, -reward_cfg.time_penalty)

    components = {
        "r_prog": r_prog,
        "r_omega": r_omega,
        "r_obs": r_obs,
        "r_gate_bonus": r_gate_bonus,
        "r_exit_vel": r_exit_vel,
        "r_terminal": r_terminal,
        "r_time": r_time,
        "r_vel": r_vel,
        "r_guid": r_guid,
    }
    reward = (
        r_prog + r_omega + r_obs + r_gate_bonus + r_exit_vel + r_terminal + r_time + r_vel + r_guid
    )
    return reward, components
