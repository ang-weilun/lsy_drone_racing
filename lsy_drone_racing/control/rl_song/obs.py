"""Observation encoding for the Song-2023 RL prototype.

Builds the 61-dimensional actor observation and the (Week-1-equivalent)
critic observation from the racing env's observation dict. Pure JAX, designed
to be ``jit``- and ``vmap``-friendly.

Layout
------
The actor observation is
``[drone | gates | visited | prev_action | obstacles | proximity]``:

* drone (13): 6D rotation rep (first two columns of ``R_wb``), body-frame
  linear velocity (3), body-frame angular velocity (3), drone z (1).
* gates (24): the next ``N_FUTURE_GATES = 2`` gates' four opening corners. The
  target-gate corners are expressed in the **drone body frame**; the next
  gate's corners are expressed in the **target gate's frame** (Song 2021's
  recursive trick, used unchanged by Song 2023 / Romero 2024 / Wang 2025).
* visited (2): float flags for whether each of the two future gates has been
  revealed.
* prev_action (4): the previous env-action 4-vec ``[roll, pitch, yaw, thrust]``.
* obstacles (16): four obstacles' body-frame relative positions (3 each) and
  visited flags (1 each).
* proximity (2): scalar danger features — XY clearance to the nearest
  obstacle, and the signed closing speed along the direction to that
  obstacle (positive when approaching). Pre-computes the cross-channel
  interaction that v33b / v34 evaluations showed the policy was failing
  to learn from the raw obstacle channel alone.

Cyclic shift: the gate list is rotated so that ``gates[0]`` is always the
current target. The target-gate index is therefore not encoded explicitly.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_DIM, ENV_ACTION_DIM

# Number of future gates encoded in the actor observation (target + 1).
N_FUTURE_GATES: int = 2
# Number of obstacles encoded. Hard-coded to the level-1 / level-3 layout
# (4 obstacles); raises at runtime if the env exposes a different count.
N_OBSTACLES: int = 4
# Half extents of the gate opening in the gate's local (y, z) plane. The
# track configs specify a 0.4 m x 0.4 m opening; see e.g. ``config/level1.toml``
# header comment "Gates are square. Gates are 0.72m wide ... with a 0.4m wide
# opening." Gate local +x is the through direction.
GATE_HALF_SIZE_M: tuple[float, float] = (0.20, 0.20)
# Clip range for the per-feature normalized observation; matches CleanRL.
NORM_CLIP: float = 10.0
# Tiny constant to keep the running variance positive even at step 0.
NORM_VAR_EPS: float = 1e-4


class NormalizerState(NamedTuple):
    """Welford running-statistics state for the observation normalizer.

    Fields are sized to the observation dimension and updated batch-wise from
    rollouts. Frozen at deploy time.
    """

    mean: Array
    var: Array
    count: Array  # scalar float; fractional epsilon supported for warm start


def init_normalizer(obs_dim: int = ACTOR_OBS_DIM) -> NormalizerState:
    """Return a fresh normalizer in the standard CleanRL warm-start state.

    Parameters
    ----------
    obs_dim : int, optional
        Dimensionality of the observation vector.

    Returns
    -------
    NormalizerState
        ``mean=0``, ``var=1``, ``count=NORM_VAR_EPS``.
    """
    return NormalizerState(
        mean=jnp.zeros((obs_dim,), dtype=jnp.float32),
        var=jnp.ones((obs_dim,), dtype=jnp.float32),
        count=jnp.asarray(NORM_VAR_EPS, dtype=jnp.float32),
    )


def update_normalizer(state: NormalizerState, batch: Array) -> NormalizerState:
    """Parallel Welford update from a batch of observations.

    Parameters
    ----------
    state : NormalizerState
        Current running statistics.
    batch : Array, shape (n, obs_dim)
        New raw (un-normalized) observations.

    Returns
    -------
    NormalizerState
        Updated statistics. Numerically stable parallel Welford (Chan et al.).
    """
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = jnp.asarray(batch.shape[0], dtype=state.count.dtype)

    delta = batch_mean - state.mean
    total = state.count + batch_count
    new_mean = state.mean + delta * batch_count / total
    m_a = state.var * state.count
    m_b = batch_var * batch_count
    new_m2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / total
    new_var = new_m2 / total
    return NormalizerState(mean=new_mean, var=new_var, count=total)


def apply_normalizer(state: NormalizerState, x: Array) -> Array:
    """Normalize ``x`` with current statistics and clip to ``±NORM_CLIP``."""
    std = jnp.sqrt(state.var + NORM_VAR_EPS)
    return jnp.clip((x - state.mean) / std, -NORM_CLIP, NORM_CLIP)


def _quat_to_matrix(quat_xyzw: Array) -> Array:
    """Convert an xyzw quaternion to a ``3x3`` rotation matrix in pure JAX.

    Parameters
    ----------
    quat_xyzw : Array, shape (..., 4)
        Drone or gate orientation as an xyzw quaternion (env convention).
        Assumed unit-norm (sim and env guarantee this).

    Returns
    -------
    Array, shape (..., 3, 3)
        Rotation matrix ``R`` so that ``v_world = R @ v_local``.

    Notes
    -----
    Hand-rolled rather than calling
    ``jax.scipy.spatial.transform.Rotation.from_quat`` then ``.as_matrix()``
    because
    that wrapper goes through ``np.vectorize`` and adds ~30 ms of Python-side
    overhead per call. With three quat-to-matrix conversions per env step and
    ``n_envs=4096``, the wrapper alone consumed 35 % of training wall time
    while the GPU sat at 1-5 % utilization. Per CLAUDE.md, the ecosystem
    library would normally be the default; the real-time training budget
    forces the exemption.
    """
    x = quat_xyzw[..., 0]
    y = quat_xyzw[..., 1]
    z = quat_xyzw[..., 2]
    w = quat_xyzw[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    row0 = jnp.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)], axis=-1)
    row1 = jnp.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)], axis=-1)
    row2 = jnp.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


# Gate-local corners of the opening: (x_through=0, ±h_y, ±h_z).
_HALF_Y, _HALF_Z = GATE_HALF_SIZE_M
_GATE_CORNERS_LOCAL: Array = jnp.asarray(
    [
        [0.0, +_HALF_Y, +_HALF_Z],
        [0.0, +_HALF_Y, -_HALF_Z],
        [0.0, -_HALF_Y, +_HALF_Z],
        [0.0, -_HALF_Y, -_HALF_Z],
    ],
    dtype=jnp.float32,
)


def _gate_corners_world(gate_pos: Array, gate_quat: Array) -> Array:
    """Return the four opening corners of a gate in world coordinates.

    Parameters
    ----------
    gate_pos : Array, shape (3,)
    gate_quat : Array, shape (4,)
        xyzw quaternion.

    Returns
    -------
    Array, shape (4, 3)
        Corners stacked as rows.
    """
    rot = _quat_to_matrix(gate_quat)
    return (_GATE_CORNERS_LOCAL @ rot.T) + gate_pos


def build_actor_obs(
    env_obs: dict[str, Array], prev_action: Array, normalizer: NormalizerState
) -> Array:
    """Encode one (un-batched) env observation as the 61-d actor tensor.

    Parameters
    ----------
    env_obs : dict[str, Array]
        Single-sample (un-batched) racing-env observation. Keys: ``pos``,
        ``quat``, ``vel``, ``ang_vel``, ``target_gate``, ``gates_pos``,
        ``gates_quat``, ``gates_visited``, ``obstacles_pos``,
        ``obstacles_visited``.
    prev_action : Array, shape (4,)
        Previously commanded env action ``[roll, pitch, yaw, thrust]``.
    normalizer : NormalizerState
        Running mean/std applied to the assembled raw observation.

    Returns
    -------
    Array, shape (ACTOR_OBS_DIM,)
        Normalized actor observation.

    Notes
    -----
    Pure JAX. To use across ``n_envs`` rollouts, ``jax.vmap`` this function
    over the leading dimension of every input.
    """
    pos = env_obs["pos"]
    quat = env_obs["quat"]
    vel = env_obs["vel"]
    ang_vel = env_obs["ang_vel"]
    target = env_obs["target_gate"]
    gates_pos = env_obs["gates_pos"]
    gates_quat = env_obs["gates_quat"]
    gates_visited = env_obs["gates_visited"]
    obstacles_pos = env_obs["obstacles_pos"]
    obstacles_visited = env_obs["obstacles_visited"]

    n_gates = gates_pos.shape[0]
    # ``target == -1`` means race finished. Episode is terminating; clamp so
    # downstream gather stays in-bounds. The actor output on a terminal step
    # is discarded by the rollout buffer.
    target_idx = jnp.where(target < 0, 0, target)
    gate_indices = (target_idx + jnp.arange(N_FUTURE_GATES)) % n_gates

    # Drone channel.
    rot_wb = _quat_to_matrix(quat)
    rot_6d = rot_wb[:, :2].reshape(6)  # first two columns
    rot_bw = rot_wb.T
    vel_body = rot_bw @ vel
    z = pos[2:3]
    drone_chan = jnp.concatenate([rot_6d, vel_body, ang_vel, z])

    # Target gate corners in drone body frame.
    g_target_pos = gates_pos[gate_indices[0]]
    g_target_quat = gates_quat[gate_indices[0]]
    g_target_corners_w = _gate_corners_world(g_target_pos, g_target_quat)
    target_corners_body = (g_target_corners_w - pos) @ rot_bw.T  # (4, 3)

    # Next gate corners expressed in the target gate's frame (Song 2021 trick).
    g_next_pos = gates_pos[gate_indices[1]]
    g_next_quat = gates_quat[gate_indices[1]]
    g_next_corners_w = _gate_corners_world(g_next_pos, g_next_quat)
    rot_target_world = _quat_to_matrix(g_target_quat)
    next_corners_in_target = (g_next_corners_w - g_target_pos) @ rot_target_world
    gate_chan = jnp.concatenate(
        [target_corners_body.reshape(-1), next_corners_in_target.reshape(-1)]
    )

    visited_chan = gates_visited[gate_indices].astype(jnp.float32)

    prev_action_chan = jnp.asarray(prev_action, dtype=jnp.float32).reshape(ENV_ACTION_DIM)

    # Obstacle channel: body-frame relative position + visited flag, per
    # obstacle.
    #
    # v33: project the obstacle's world-frame XY onto the drone's altitude
    # plane (replace ``obs_z`` with ``pos_z``) before rotating into the
    # body frame. Obstacles are vertical capsules from the floor to
    # z≈1.55 (see ``envs/assets/obstacle.xml`` and the level-3 toml
    # header); the relevant collision surface is at the drone's altitude,
    # not at the top marker. ``reward.step_reward`` already uses
    # XY-only distance for ``r_obs`` (treating obstacles as infinite
    # vertical poles); pre-v33 the actor saw the body-frame vector to
    # the top marker (z≈1.55) instead, so the geometry the policy was
    # graded on disagreed with the geometry it could observe. Same
    # 3 floats per obstacle, so the obs dimensionality (and the
    # observation normalizer it interacts with) is unchanged.
    obstacles_at_alt = obstacles_pos.at[:, 2].set(pos[2])
    obstacles_rel_body = (obstacles_at_alt - pos) @ rot_bw.T  # (N_OBSTACLES, 3)
    obstacle_chan = jnp.concatenate(
        [obstacles_rel_body, obstacles_visited.astype(jnp.float32)[..., None]], axis=-1
    ).reshape(-1)

    # v35: scalar proximity features. The raw obstacle channel already
    # contains the body-frame relative positions, so in principle the
    # network could infer ``min over obstacles of |Δxy|`` and the closing
    # speed itself. v33b / v34 evaluations showed that does not happen:
    # the policy generates near-zero roll commands as it approaches an
    # obstacle on the spawn→gate-0 line. Pre-computing two scalars gives
    # PPO a direct gradient on the danger signal instead of requiring it
    # to learn the multiplicative cross-channel interaction
    # (self_velocity · obstacle_direction) from sparse reward.
    obstacle_delta_xy = obstacles_pos[:, :2] - pos[:2]  # (N_OBSTACLES, 2), world XY
    obstacle_dist_xy = jnp.linalg.norm(obstacle_delta_xy, axis=-1)  # (N_OBSTACLES,)
    min_clearance_xy = jnp.min(obstacle_dist_xy)
    nearest_idx = jnp.argmin(obstacle_dist_xy)
    # Unit vector from drone XY to nearest obstacle, in world frame. Guarded
    # for the (numerically negligible) co-located case so the dot product
    # below is well-defined.
    dir_to_nearest = obstacle_delta_xy[nearest_idx]
    dir_norm = jnp.linalg.norm(dir_to_nearest)
    safe_norm = jnp.maximum(dir_norm, 1e-6)
    unit_to_nearest = dir_to_nearest / safe_norm
    # Closing speed: positive when drone XY velocity has a component pointing
    # toward the nearest obstacle. Built from world-frame velocity, not body
    # frame, since the unit vector is in world frame.
    closing_speed = jnp.dot(vel[:2], unit_to_nearest)
    proximity_chan = jnp.stack(
        [min_clearance_xy.astype(jnp.float32), closing_speed.astype(jnp.float32)]
    )

    raw = jnp.concatenate(
        [drone_chan, gate_chan, visited_chan, prev_action_chan, obstacle_chan, proximity_chan]
    )
    return apply_normalizer(normalizer, raw)


def build_critic_obs(
    env_obs: dict[str, Array],
    prev_action: Array,
    normalizer: NormalizerState,
    true_gates_pos: Array | None = None,
    true_gates_quat: Array | None = None,
    true_obstacles_pos: Array | None = None,
) -> Array:
    """Encode one observation as the critic tensor.

    Asymmetric actor/critic: when ``true_*`` arguments are provided, the
    critic sees the unmasked (post-randomization, pre-visited-mask) gate and
    obstacle poses instead of the partially-observed values in ``env_obs``.
    Layout and dimensionality are identical to :func:`build_actor_obs`; only
    the *values* of the gate/obstacle channels change.

    Parameters
    ----------
    env_obs, prev_action, normalizer
        See :func:`build_actor_obs`.
    true_gates_pos : Array, shape (n_gates, 3), optional
        Unmasked true gate positions from ``env.data.gates_pos``. When
        ``None``, the masked value from ``env_obs`` is used.
    true_gates_quat : Array, shape (n_gates, 4), optional
        Unmasked true gate orientations from ``env.data.gates_quat``.
    true_obstacles_pos : Array, shape (n_obstacles, 3), optional
        Unmasked true obstacle positions from ``env.data.obstacles_pos``.

    Returns
    -------
    Array, shape (ACTOR_OBS_DIM,)
        Normalized critic observation. Same dimensionality as the actor obs;
        the affine normalizer (calibrated on actor obs statistics) is reused.
    """
    privileged_obs = dict(env_obs)
    if true_gates_pos is not None:
        privileged_obs["gates_pos"] = true_gates_pos
    if true_gates_quat is not None:
        privileged_obs["gates_quat"] = true_gates_quat
    if true_obstacles_pos is not None:
        privileged_obs["obstacles_pos"] = true_obstacles_pos
    return build_actor_obs(privileged_obs, prev_action, normalizer)


def vmap_build_actor_obs(
    env_obs: dict[str, Array], prev_action: Array, normalizer: NormalizerState
) -> Array:
    """Batched variant of :func:`build_actor_obs` over the leading axis.

    Parameters
    ----------
    env_obs : dict[str, Array]
        Each value has a leading ``n_envs`` axis.
    prev_action : Array, shape (n_envs, 4)
    normalizer : NormalizerState
        Broadcast over the batch.

    Returns
    -------
    Array, shape (n_envs, ACTOR_OBS_DIM)
    """
    # Normalizer is shared across the batch (broadcasted), so ``in_axes=None``.
    return jax.vmap(build_actor_obs, in_axes=({k: 0 for k in env_obs}, 0, None))(
        env_obs, prev_action, normalizer
    )


def vmap_build_critic_obs(
    env_obs: dict[str, Array],
    prev_action: Array,
    normalizer: NormalizerState,
    true_gates_pos: Array | None = None,
    true_gates_quat: Array | None = None,
    true_obstacles_pos: Array | None = None,
) -> Array:
    """Batched variant of :func:`build_critic_obs` over the leading axis.

    Parameters
    ----------
    env_obs : dict[str, Array]
        Each value has a leading ``n_envs`` axis.
    prev_action : Array, shape (n_envs, 4)
    normalizer : NormalizerState
        Broadcast over the batch.
    true_gates_pos : Array, shape (n_envs, n_gates, 3), optional
    true_gates_quat : Array, shape (n_envs, n_gates, 4), optional
    true_obstacles_pos : Array, shape (n_envs, n_obstacles, 3), optional
        Privileged unmasked poses. When all three are ``None`` the critic
        falls back to the actor encoding (no information advantage).

    Returns
    -------
    Array, shape (n_envs, ACTOR_OBS_DIM)
        Normalized critic observations.
    """
    privileged_obs = dict(env_obs)
    if true_gates_pos is not None:
        privileged_obs["gates_pos"] = true_gates_pos
    if true_gates_quat is not None:
        privileged_obs["gates_quat"] = true_gates_quat
    if true_obstacles_pos is not None:
        privileged_obs["obstacles_pos"] = true_obstacles_pos
    return vmap_build_actor_obs(privileged_obs, prev_action, normalizer)
