"""Observation encoding for the Song-2023 RL prototype.

Builds the 59-dimensional actor observation and the (Week-1-equivalent)
critic observation from the racing env's observation dict. Pure JAX, designed
to be ``jit``- and ``vmap``-friendly.

Layout
------
The actor observation is ``[drone | gates | visited | prev_action | obstacles]``:

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

Cyclic shift: the gate list is rotated so that ``gates[0]`` is always the
current target. The target-gate index is therefore not encoded explicitly.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.spatial.transform import Rotation

from lsy_drone_racing.control.rl_song.config import (
    ACTOR_OBS_DIM,
    ENV_ACTION_DIM,
)

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
    """Convert an xyzw quaternion to a ``3x3`` rotation matrix in JAX.

    Parameters
    ----------
    quat_xyzw : Array, shape (4,)
        Drone or gate orientation as an xyzw quaternion (env convention).

    Returns
    -------
    Array, shape (3, 3)
        Rotation matrix ``R`` such that ``v_world = R @ v_local``.
    """
    return Rotation.from_quat(quat_xyzw).as_matrix()


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
    """Encode one (un-batched) env observation as the 59-d actor tensor.

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

    # Obstacle channel: body-frame relative position + visited flag, per obstacle.
    obstacles_rel_body = (obstacles_pos - pos) @ rot_bw.T  # (N_OBSTACLES, 3)
    obstacle_chan = jnp.concatenate(
        [obstacles_rel_body, obstacles_visited.astype(jnp.float32)[..., None]], axis=-1
    ).reshape(-1)

    raw = jnp.concatenate(
        [drone_chan, gate_chan, visited_chan, prev_action_chan, obstacle_chan]
    )
    return apply_normalizer(normalizer, raw)


def build_critic_obs(
    env_obs: dict[str, Array], prev_action: Array, normalizer: NormalizerState
) -> Array:
    """Encode one observation as the critic tensor.

    Week 1 behavior: identical to :func:`build_actor_obs`. The asymmetric
    actor/critic seam is wired through so that stage-3+ can substitute
    privileged information (unmasked true gate corners + DR realization) here
    without touching the actor or the training loop.

    Parameters
    ----------
    env_obs, prev_action, normalizer
        See :func:`build_actor_obs`.

    Returns
    -------
    Array, shape (ACTOR_OBS_DIM,)
        Currently equal to the actor obs. Will diverge at stage 3+.
    """
    # TODO(stage3): substitute true gate corners (ignoring visited mask),
    # true (un-noised) drone state, and the current DR realization. Requires
    # reaching into the env's pre-masking ``sim_data`` from the wrapper.
    return build_actor_obs(env_obs, prev_action, normalizer)


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
