"""Observation encoding for the Song-2023 RL prototype.

Builds a 52-dim actor obs `[drone | gates | obstacles]` and a privileged-pose
critic obs of the same shape. Pure JAX, jit/vmap-friendly. Gate list is
cyclically shifted so `gates[0]` is always the target. Obstacle channel is
the K nearest in body-frame XY in permutation-stable slots with an identity
one-hot so the network sees rank-flips as labels, not motion.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_DIM, N_NEAREST_OBSTACLES, N_OBSTACLES

N_FUTURE_GATES: int = 2
GATE_HALF_SIZE_M: tuple[float, float] = (0.20, 0.20)
NORM_CLIP: float = 10.0
NORM_VAR_EPS: float = 1e-4


class NormalizerState(NamedTuple):
    """Welford running mean/var/count. Frozen at deploy."""

    mean: Array
    var: Array
    count: Array


def init_normalizer(obs_dim: int = ACTOR_OBS_DIM) -> NormalizerState:
    """Fresh normalizer: mean=0, var=1, count=eps."""
    return NormalizerState(
        mean=jnp.zeros((obs_dim,), dtype=jnp.float32),
        var=jnp.ones((obs_dim,), dtype=jnp.float32),
        count=jnp.asarray(NORM_VAR_EPS, dtype=jnp.float32),
    )


def update_normalizer(state: NormalizerState, batch: Array) -> NormalizerState:
    """Parallel Welford update (Chan et al.)."""
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
    """Normalize and clip to `±NORM_CLIP`."""
    std = jnp.sqrt(state.var + NORM_VAR_EPS)
    return jnp.clip((x - state.mean) / std, -NORM_CLIP, NORM_CLIP)


def _quat_to_matrix(quat_xyzw: Array) -> Array:
    """Convert xyzw quat to a 3x3 rotation matrix.

    Hand-rolled because scipy's wrapper adds ~30 ms/call via np.vectorize.
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
    """Four opening corners of a gate in world coordinates."""
    rot = _quat_to_matrix(gate_quat)
    return (_GATE_CORNERS_LOCAL @ rot.T) + gate_pos


def build_actor_obs(
    env_obs: dict[str, Array], prev_action: Array, normalizer: NormalizerState
) -> Array:
    """Encode one env obs as the 52-d actor tensor.

    `prev_action` is accepted for API compatibility with rollout but is unused.
    """
    pos = env_obs["pos"]
    quat = env_obs["quat"]
    vel = env_obs["vel"]
    target = env_obs["target_gate"]
    gates_pos = env_obs["gates_pos"]
    gates_quat = env_obs["gates_quat"]
    obstacles_pos = env_obs["obstacles_pos"]
    obstacles_visited = env_obs["obstacles_visited"]

    n_gates = gates_pos.shape[0]
    # target == -1 means race finished; clamp so the gather stays in-bounds
    # (the rollout buffer discards terminal-step actor outputs).
    target_idx = jnp.where(target < 0, 0, target)
    gate_indices = (target_idx + jnp.arange(N_FUTURE_GATES)) % n_gates

    rot_wb = _quat_to_matrix(quat)
    rot_9d = rot_wb.reshape(9)
    rot_bw = rot_wb.T
    vel_body = rot_bw @ vel
    drone_chan = jnp.concatenate([rot_9d, vel_body])

    g_target_pos = gates_pos[gate_indices[0]]
    g_target_quat = gates_quat[gate_indices[0]]
    g_target_corners_w = _gate_corners_world(g_target_pos, g_target_quat)
    target_corners_body = (g_target_corners_w - pos) @ rot_bw.T

    g_next_pos = gates_pos[gate_indices[1]]
    g_next_quat = gates_quat[gate_indices[1]]
    g_next_corners_w = _gate_corners_world(g_next_pos, g_next_quat)
    inter_gate_delta_body = (g_next_corners_w - g_target_corners_w) @ rot_bw.T

    gate_chan = jnp.concatenate(
        [target_corners_body.reshape(-1), inter_gate_delta_body.reshape(-1)]
    )

    _ = prev_action

    # Obstacles are vertical poles; project onto the drone's altitude plane
    # before ranking by body-frame XY distance.
    obstacles_at_alt = obstacles_pos.at[:, 2].set(pos[2])
    obstacles_rel_body = (obstacles_at_alt - pos) @ rot_bw.T
    obstacles_xy_body = obstacles_rel_body[:, :2]
    obstacles_dist_xy = jnp.linalg.norm(obstacles_xy_body, axis=-1)

    nearest_indices = jnp.argsort(obstacles_dist_xy)[:N_NEAREST_OBSTACLES]
    nearest_xy_body = obstacles_xy_body[nearest_indices]
    nearest_dist = obstacles_dist_xy[nearest_indices]
    nearest_visited = obstacles_visited[nearest_indices].astype(jnp.float32)

    safe_norm = jnp.maximum(nearest_dist, 1e-6)[:, None]
    unit_to_obstacle = nearest_xy_body / safe_norm
    vel_xy_body = vel_body[:2]
    vel_proj = unit_to_obstacle @ vel_xy_body

    identity_onehot = jax.nn.one_hot(nearest_indices, num_classes=N_OBSTACLES, dtype=jnp.float32)

    obstacle_chan = jnp.concatenate(
        [nearest_xy_body, vel_proj[:, None], identity_onehot, nearest_visited[:, None]], axis=-1
    ).reshape(-1)

    raw = jnp.concatenate([drone_chan, gate_chan, obstacle_chan])
    return apply_normalizer(normalizer, raw)


def build_critic_obs(
    env_obs: dict[str, Array],
    prev_action: Array,
    normalizer: NormalizerState,
    true_gates_pos: Array | None = None,
    true_gates_quat: Array | None = None,
    true_obstacles_pos: Array | None = None,
) -> Array:
    """Same layout as `build_actor_obs`, but swap in unmasked poses when provided."""
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
    """Batched `build_actor_obs` over the leading axis."""
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
    """Batched `build_critic_obs` over the leading axis."""
    privileged_obs = dict(env_obs)
    if true_gates_pos is not None:
        privileged_obs["gates_pos"] = true_gates_pos
    if true_gates_quat is not None:
        privileged_obs["gates_quat"] = true_gates_quat
    if true_obstacles_pos is not None:
        privileged_obs["obstacles_pos"] = true_obstacles_pos
    return vmap_build_actor_obs(privileged_obs, prev_action, normalizer)
