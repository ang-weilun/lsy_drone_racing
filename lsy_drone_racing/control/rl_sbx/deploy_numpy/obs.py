"""Numpy mirror of the SBX actor observation encoder."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.rl_sbx.deploy_numpy.constants import read_rl_song_obs_constant
from lsy_drone_racing.control.rl_sbx.deploy_numpy.normalizer import (
    NormalizerState,
    apply_normalizer,
)
from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_ANG_VEL_DIM, ACTOR_OBS_DIM

# Number of future gates encoded in actor observations.
N_FUTURE_GATES: int = int(read_rl_song_obs_constant("N_FUTURE_GATES"))

# Number of obstacles encoded in actor observations.
N_OBSTACLES: int = int(read_rl_song_obs_constant("N_OBSTACLES"))

# Number of nearest obstacles packed into actor-observation slots.
N_NEAREST_OBSTACLES: int = int(read_rl_song_obs_constant("N_NEAREST_OBSTACLES"))

# Gate opening half extents in meters in local (y, z).
GATE_HALF_SIZE_M: tuple[float, float] = tuple(read_rl_song_obs_constant("GATE_HALF_SIZE_M"))

# Gate-local opening corners in meters, with local +x as through direction.
_GATE_CORNERS_LOCAL: npt.NDArray[np.float32] = np.asarray(
    [
        [0.0, +GATE_HALF_SIZE_M[0], +GATE_HALF_SIZE_M[1]],
        [0.0, +GATE_HALF_SIZE_M[0], -GATE_HALF_SIZE_M[1]],
        [0.0, -GATE_HALF_SIZE_M[0], +GATE_HALF_SIZE_M[1]],
        [0.0, -GATE_HALF_SIZE_M[0], -GATE_HALF_SIZE_M[1]],
    ],
    dtype=np.float32,
)

# Minimum norm in meters for a stable obstacle-direction unit vector.
_SAFE_DIRECTION_NORM_M: float = 1e-6


def gate_corners_local() -> npt.NDArray[np.float32]:
    """Return a copy of the gate-local opening corners in meters."""
    return _GATE_CORNERS_LOCAL.copy()


def build_actor_obs(
    env_obs: dict[str, npt.NDArray[np.floating]],
    prev_action: npt.NDArray[np.floating],
    normalizer: NormalizerState,
    gate_corners_local: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.float32]:
    """Encode one unbatched env observation as a normalized actor observation.

    Parameters
    ----------
    env_obs : dict[str, ndarray]
        Observation with keys ``pos``, ``quat``, ``vel``, ``ang_vel``,
        ``target_gate``, ``gates_pos``, ``gates_quat``, ``obstacles_pos``,
        and ``obstacles_visited``.
    prev_action : ndarray, shape (4,)
        Accepted for API compatibility. The actor observation no longer
        includes a previous-action channel.
    normalizer : NormalizerState
        Frozen actor-observation normalizer.
    gate_corners_local : ndarray, shape (4, 3), optional
        Precomputed gate-local opening corners in meters.

    Returns:
    -------
    actor_obs : ndarray, shape (ACTOR_OBS_DIM,)
        Normalized actor observation.
    """
    corners_local = _GATE_CORNERS_LOCAL if gate_corners_local is None else gate_corners_local
    pos = np.asarray(env_obs["pos"], dtype=np.float32)
    quat = np.asarray(env_obs["quat"], dtype=np.float32)
    vel = np.asarray(env_obs["vel"], dtype=np.float32)
    target = int(np.asarray(env_obs["target_gate"]))
    gates_pos = np.asarray(env_obs["gates_pos"], dtype=np.float32)
    gates_quat = np.asarray(env_obs["gates_quat"], dtype=np.float32)
    obstacles_pos = np.asarray(env_obs["obstacles_pos"], dtype=np.float32)
    obstacles_visited = np.asarray(env_obs["obstacles_visited"])

    n_gates = gates_pos.shape[0]
    target_idx = 0 if target < 0 else target
    gate_indices = (target_idx + np.arange(N_FUTURE_GATES, dtype=np.int64)) % n_gates

    rot_wb = Rotation.from_quat(quat).as_matrix().astype(np.float32)
    rot_9d = rot_wb.reshape(9)
    rot_bw = rot_wb.T
    vel_body = rot_bw @ vel
    drone_parts = [rot_9d, vel_body]
    if ACTOR_OBS_ANG_VEL_DIM:
        # Body-frame body rates, appended raw (mirror of rl_song.obs).
        drone_parts.append(np.asarray(env_obs["ang_vel"], dtype=np.float32))
    drone_chan = np.concatenate(drone_parts)

    g_target_pos = gates_pos[gate_indices[0]]
    g_target_quat = gates_quat[gate_indices[0]]
    g_target_corners_w = _gate_corners_world(g_target_pos, g_target_quat, corners_local)
    target_corners_body = (g_target_corners_w - pos) @ rot_bw.T

    g_next_pos = gates_pos[gate_indices[1]]
    g_next_quat = gates_quat[gate_indices[1]]
    g_next_corners_w = _gate_corners_world(g_next_pos, g_next_quat, corners_local)
    inter_gate_delta_body = (g_next_corners_w - g_target_corners_w) @ rot_bw.T
    gate_chan = np.concatenate([target_corners_body.reshape(-1), inter_gate_delta_body.reshape(-1)])

    _ = prev_action

    obstacles_at_alt = obstacles_pos.copy()
    obstacles_at_alt[:, 2] = pos[2]
    obstacles_rel_body = (obstacles_at_alt - pos) @ rot_bw.T
    obstacles_xy_body = obstacles_rel_body[:, :2]
    obstacles_dist_xy = np.linalg.norm(obstacles_xy_body, axis=-1)
    nearest_indices = np.argsort(obstacles_dist_xy)[:N_NEAREST_OBSTACLES]
    nearest_xy_body = obstacles_xy_body[nearest_indices]
    nearest_dist = obstacles_dist_xy[nearest_indices]
    nearest_visited = obstacles_visited[nearest_indices].astype(np.float32)

    safe_norm = np.maximum(nearest_dist, _SAFE_DIRECTION_NORM_M)[:, None]
    unit_to_obstacle = nearest_xy_body / safe_norm
    vel_proj = unit_to_obstacle @ vel_body[:2]
    identity_onehot = np.eye(N_OBSTACLES, dtype=np.float32)[nearest_indices]

    obstacle_chan = np.concatenate(
        [nearest_xy_body, vel_proj[:, None], identity_onehot, nearest_visited[:, None]], axis=-1
    ).reshape(-1)

    raw = np.concatenate([drone_chan, gate_chan, obstacle_chan]).astype(np.float32, copy=False)
    _validate_shape(raw, (ACTOR_OBS_DIM,), "raw actor observation")
    return apply_normalizer(normalizer, raw)


def _gate_corners_world(
    gate_pos: npt.NDArray[np.floating],
    gate_quat: npt.NDArray[np.floating],
    gate_corners_local: npt.NDArray[np.floating],
) -> npt.NDArray[np.float32]:
    """Return four gate opening corners in world coordinates."""
    rot = Rotation.from_quat(gate_quat).as_matrix().astype(np.float32)
    return ((gate_corners_local @ rot.T) + gate_pos).astype(np.float32, copy=False)


def _validate_shape(array: npt.NDArray[np.floating], shape: tuple[int, ...], name: str) -> None:
    """Raise when ``array`` does not match ``shape``."""
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
