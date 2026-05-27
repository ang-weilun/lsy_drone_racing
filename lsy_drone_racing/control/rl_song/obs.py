"""Observation encoding for the Song-2023 RL prototype.

Builds the 52-dimensional actor observation and the (Week-1-equivalent)
critic observation from the racing env's observation dict. Pure JAX, designed
to be ``jit``- and ``vmap``-friendly.

Layout
------
The actor observation is ``[drone | gates | obstacles]``:

* drone (12): full 9D rotation matrix and body-frame linear velocity (3).
* gates (24): target gate corners in body frame (12), then the next-gate
  minus target-gate corner deltas in body frame (12).
* obstacles (16): the ``N_NEAREST_OBSTACLES = 2`` nearest obstacles in
  body-frame XY distance, packed into permutation-stable slots. Per slot:
  body-frame XY relative position (2), body-frame velocity projected onto
  the unit vector toward the obstacle (1, positive when closing in), a
  ``N_OBSTACLES``-wide identity one-hot for which physical obstacle this
  slot points to (4), and the visited flag (1). Replaces the v33-v124
  per-obstacle 16-float block + 2-float global proximity pair; rationale
  in ``build_actor_obs``.

Cyclic shift: the gate list is rotated so that ``gates[0]`` is always the
current target.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_DIM, N_NEAREST_OBSTACLES, N_OBSTACLES

# Number of future gates encoded in the actor observation (target + 1).
N_FUTURE_GATES: int = 2
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

    Returns:
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

    Returns:
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

    Returns:
    -------
    Array, shape (..., 3, 3)
        Rotation matrix ``R`` so that ``v_world = R @ v_local``.

    Notes:
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

    Returns:
    -------
    Array, shape (4, 3)
        Corners stacked as rows.
    """
    rot = _quat_to_matrix(gate_quat)
    return (_GATE_CORNERS_LOCAL @ rot.T) + gate_pos


def build_actor_obs(
    env_obs: dict[str, Array], prev_action: Array, normalizer: NormalizerState
) -> Array:
    """Encode one (un-batched) env observation as the ``ACTOR_OBS_DIM``-d actor tensor.

    Parameters
    ----------
    env_obs : dict[str, Array]
        Single-sample (un-batched) racing-env observation. Keys: ``pos``,
        ``quat``, ``vel``, ``ang_vel``, ``target_gate``, ``gates_pos``,
        ``gates_quat``, ``gates_visited``, ``obstacles_pos``,
        ``obstacles_visited``.
    prev_action : Array, shape (4,)
        Accepted for API compatibility with rollout and controller call sites.
        The actor observation no longer includes a previous-action channel.
    normalizer : NormalizerState
        Running mean/std applied to the assembled raw observation.

    Returns:
    -------
    Array, shape (ACTOR_OBS_DIM,)
        Normalized actor observation.

    Notes:
    -----
    Pure JAX. To use across ``n_envs`` rollouts, ``jax.vmap`` this function
    over the leading dimension of every input.
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
    # ``target == -1`` means race finished. Episode is terminating; clamp so
    # downstream gather stays in-bounds. The actor output on a terminal step
    # is discarded by the rollout buffer.
    target_idx = jnp.where(target < 0, 0, target)
    gate_indices = (target_idx + jnp.arange(N_FUTURE_GATES)) % n_gates

    # Drone channel.
    rot_wb = _quat_to_matrix(quat)
    rot_9d = rot_wb.reshape(9)
    rot_bw = rot_wb.T
    vel_body = rot_bw @ vel
    drone_chan = jnp.concatenate([rot_9d, vel_body])

    # Song's recursive gate channel keeps both 12-float blocks in body
    # frame so the target and inter-gate geometry share one rotation basis.
    g_target_pos = gates_pos[gate_indices[0]]
    g_target_quat = gates_quat[gate_indices[0]]
    g_target_corners_w = _gate_corners_world(g_target_pos, g_target_quat)
    # Target-gate corners expressed in drone body frame.
    target_corners_body = (g_target_corners_w - pos) @ rot_bw.T  # (4, 3)

    g_next_pos = gates_pos[gate_indices[1]]
    g_next_quat = gates_quat[gate_indices[1]]
    g_next_corners_w = _gate_corners_world(g_next_pos, g_next_quat)
    inter_gate_delta_body = (g_next_corners_w - g_target_corners_w) @ rot_bw.T

    gate_chan = jnp.concatenate(
        [target_corners_body.reshape(-1), inter_gate_delta_body.reshape(-1)]
    )

    _ = prev_action

    # Obstacle channel: the ``N_NEAREST_OBSTACLES`` nearest obstacles in
    # body-frame XY distance, packed into permutation-stable slots. Per slot
    # the actor sees [xy_body (2), vel_proj (1), identity_onehot (N_OBSTACLES),
    # visited (1)].
    #
    # v33: project the obstacle's world-frame XY onto the drone's altitude
    # plane before rotating into the body frame. Obstacles are vertical
    # capsules from the floor to z≈1.55 (see ``envs/assets/obstacle.xml`` and
    # the level-3 toml header); the relevant collision surface is at the
    # drone's altitude, not at the top marker. ``reward.step_reward`` uses
    # XY-only distance for ``r_obs`` (infinite vertical poles), so the z
    # dimension carries no information the policy is graded on and is
    # dropped entirely in this layout.
    #
    # v??? (slot layout): replaces the v33-v124 16-float per-obstacle block
    # + 2-float global proximity pair (total 18). The raw block was 16/59 ≈
    # 27% of the actor obs but encoded the cross-channel "am I approaching
    # a hazard" signal indirectly — v33b / v34 evaluations showed the
    # policy failed to learn it from the block alone, which v35 patched
    # with hand-rolled proximity scalars. The per-slot velocity projection
    # makes that signal explicit for each of the K tracked obstacles
    # (subsumes the v35 scalars), and the identity one-hot lets the
    # network handle the rank-flip discontinuity: when the drone passes
    # obstacle A and B becomes the new nearest, the xy / vel-proj values
    # in slot 0 jump; without an identity label the policy reads the jump
    # as a physics event and the observation normalizer treats it as
    # variance, inflating σ on those channels. The visited flag per slot
    # guards against ranking an unobserved obstacle (at its possibly-stale
    # nominal position) into a slot.
    obstacles_at_alt = obstacles_pos.at[:, 2].set(pos[2])
    obstacles_rel_body = (obstacles_at_alt - pos) @ rot_bw.T  # (N_OBSTACLES, 3)
    obstacles_xy_body = obstacles_rel_body[:, :2]  # (N_OBSTACLES, 2)
    obstacles_dist_xy = jnp.linalg.norm(obstacles_xy_body, axis=-1)  # (N_OBSTACLES,)

    # Sort by body-frame XY distance and keep the K nearest. ``argsort`` is
    # O(N log N) on a 4-element vector — negligible inside the jit graph.
    nearest_indices = jnp.argsort(obstacles_dist_xy)[:N_NEAREST_OBSTACLES]
    nearest_xy_body = obstacles_xy_body[nearest_indices]  # (K, 2)
    nearest_dist = obstacles_dist_xy[nearest_indices]  # (K,)
    nearest_visited = obstacles_visited[nearest_indices].astype(jnp.float32)

    # Body-frame velocity projected onto unit-to-obstacle in the same frame.
    # Positive when the drone is closing in on that slot's obstacle. Guard
    # the unit vector against the (numerically negligible) co-located case
    # so the projection stays finite.
    safe_norm = jnp.maximum(nearest_dist, 1e-6)[:, None]
    unit_to_obstacle = nearest_xy_body / safe_norm  # (K, 2)
    vel_xy_body = vel_body[:2]
    vel_proj = unit_to_obstacle @ vel_xy_body  # (K,)

    # Identity one-hot among the N_OBSTACLES physical obstacles. Obstacle i
    # is the same physical capsule across all episodes, so the network can
    # learn "when the one-hot in this slot flips, the xy/vel-proj jump is
    # a slot reassignment, not motion."
    identity_onehot = jax.nn.one_hot(
        nearest_indices, num_classes=N_OBSTACLES, dtype=jnp.float32
    )  # (K, N_OBSTACLES)

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

    Returns:
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

    Returns:
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

    Returns:
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
