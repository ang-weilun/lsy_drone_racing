"""One-off ablation controller for the gate-4 lookahead hypothesis.

Subclasses :class:`RLSBXController` and mutates dims 25-36 of the actor
observation (the next-gate corners in target-gate frame) before the actor
sees them. The mutation mode is read from the ``ABLATE_MODE`` environment
variable so a single controller file can support all variants without
breaking the one-controller-per-file rule.

Modes
-----
``baseline``
    No mutation; identical to :class:`RLSBXController`.
``zero``
    Set dims 25-36 to zero (post-normalization mean).
``clamp``
    Recompute the next-gate slot under ``next_idx = min(target+1, n_gates-1)``
    semantics instead of the production ``(target + arange) % n_gates``
    wraparound. At the last gate this collapses the next-gate channel to
    constant per-gate-geometry corners (``GATE_CORNERS_LOCAL`` rotated into
    the target-gate frame, which is the identity transform when next == target).
``randpost3``
    Keep the production wrap for ``target < n_gates - 1``; replace dims
    25-36 with samples from a unit Gaussian only on steps where the drone
    is heading toward the final gate. Tests whether the policy's terminal
    behavior depends on the lookahead channel at all.

Not for training. Not for deployment. Only used by the gate-4 lookahead
ablation experiment dispatched 2026-05-26.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from lsy_drone_racing.control.rl_sbx.controller import RLSBXController
from lsy_drone_racing.control.rl_song import obs as obs_encoding
from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_DIM
from lsy_drone_racing.control.rl_song.policy import raw_to_env_action

if TYPE_CHECKING:
    import numpy.typing as npt

# Slice for the next-gate corners channel in the post-normalization actor obs.
# See lsy_drone_racing/control/rl_song/obs.py:283-285 for the concat order:
# drone(13) | target_corners(12) | next_corners(12) | prev_action(4) | obstacles(16) | prox(2).
LOOKAHEAD_START: int = 25
LOOKAHEAD_END: int = 37
LOOKAHEAD_DIM: int = LOOKAHEAD_END - LOOKAHEAD_START

_ALLOWED_MODES: frozenset[str] = frozenset({"baseline", "zero", "clamp", "randpost3"})


def _resolve_mode() -> str:
    """Read and validate the ``ABLATE_MODE`` env var; default ``baseline``."""
    mode = os.environ.get("ABLATE_MODE", "baseline").strip().lower()
    if mode not in _ALLOWED_MODES:
        raise ValueError(
            f"ABLATE_MODE={mode!r} not in {sorted(_ALLOWED_MODES)}"
        )
    return mode


def _build_actor_obs_clamp(
    env_obs: dict[str, jax.Array],
    prev_action: jax.Array,
    normalizer: obs_encoding.NormalizerState,
) -> jax.Array:
    """Mirror :func:`obs.build_actor_obs` but clamp ``gate_indices[1]``.

    Differs from the production encoder only in the wraparound behavior of
    the next-gate lookup: where production uses ``% n_gates``, this uses
    ``jnp.minimum(target_idx + arange, n_gates - 1)``. All other channels
    (drone, prev_action, obstacles, proximity) and the normalizer are
    identical, so the only delta is what the actor sees in dims 25-36 when
    the drone is approaching the last gate.
    """
    pos = env_obs["pos"]
    quat = env_obs["quat"]
    vel = env_obs["vel"]
    ang_vel = env_obs["ang_vel"]
    target = env_obs["target_gate"]
    gates_pos = env_obs["gates_pos"]
    gates_quat = env_obs["gates_quat"]
    obstacles_pos = env_obs["obstacles_pos"]
    obstacles_visited = env_obs["obstacles_visited"]

    n_gates = gates_pos.shape[0]
    target_idx = jnp.where(target < 0, 0, target)
    # The single line that differs from obs.build_actor_obs: clamp instead of wrap.
    gate_indices = jnp.minimum(
        target_idx + jnp.arange(obs_encoding.N_FUTURE_GATES), n_gates - 1
    )

    rot_wb = obs_encoding._quat_to_matrix(quat)
    rot_6d = rot_wb[:, :2].reshape(6)
    rot_bw = rot_wb.T
    vel_body = rot_bw @ vel
    z = pos[2:3]
    drone_chan = jnp.concatenate([rot_6d, vel_body, ang_vel, z])

    g_target_pos = gates_pos[gate_indices[0]]
    g_target_quat = gates_quat[gate_indices[0]]
    g_target_corners_w = obs_encoding._gate_corners_world(g_target_pos, g_target_quat)
    target_corners_body = (g_target_corners_w - pos) @ rot_bw.T

    g_next_pos = gates_pos[gate_indices[1]]
    g_next_quat = gates_quat[gate_indices[1]]
    g_next_corners_w = obs_encoding._gate_corners_world(g_next_pos, g_next_quat)
    rot_target_world = obs_encoding._quat_to_matrix(g_target_quat)
    next_corners_in_target = (g_next_corners_w - g_target_pos) @ rot_target_world

    gate_chan = jnp.concatenate(
        [target_corners_body.reshape(-1), next_corners_in_target.reshape(-1)]
    )
    prev_action_chan = jnp.asarray(prev_action, dtype=jnp.float32).reshape(4)

    obstacles_at_alt = obstacles_pos.at[:, 2].set(pos[2])
    obstacles_rel_body = (obstacles_at_alt - pos) @ rot_bw.T
    obstacle_chan = jnp.concatenate(
        [obstacles_rel_body, obstacles_visited.astype(jnp.float32)[..., None]],
        axis=-1,
    ).reshape(-1)

    obstacle_delta_xy = obstacles_pos[:, :2] - pos[:2]
    obstacle_dist_xy = jnp.linalg.norm(obstacle_delta_xy, axis=-1)
    min_clearance_xy = jnp.min(obstacle_dist_xy)
    nearest_idx = jnp.argmin(obstacle_dist_xy)
    dir_to_nearest = obstacle_delta_xy[nearest_idx]
    dir_norm = jnp.linalg.norm(dir_to_nearest)
    safe_norm = jnp.maximum(dir_norm, 1e-6)
    unit_to_nearest = dir_to_nearest / safe_norm
    closing_speed = jnp.dot(vel[:2], unit_to_nearest)
    proximity_chan = jnp.stack(
        [min_clearance_xy.astype(jnp.float32), closing_speed.astype(jnp.float32)]
    )

    raw = jnp.concatenate(
        [drone_chan, gate_chan, prev_action_chan, obstacle_chan, proximity_chan]
    )
    return obs_encoding.apply_normalizer(normalizer, raw)


class RLSBXAblateController(RLSBXController):
    """Ablation variant of :class:`RLSBXController` controlled by ``ABLATE_MODE``."""

    def __init__(
        self,
        obs: dict[str, npt.NDArray[np.floating]],
        info: dict,
        config: dict,
    ) -> None:
        """Construct the deploy actor and seed the ablation RNG."""
        super().__init__(obs, info, config)
        self._ablate_mode: str = _resolve_mode()
        self._ablate_rng: jax.Array = jax.random.PRNGKey(0)
        print(f"[ABLATE] mode={self._ablate_mode}")

    def compute_control(
        self,
        obs: dict[str, npt.NDArray[np.floating]],
        info: dict | None = None,
    ) -> npt.NDArray[np.floating]:
        """Run the actor on a possibly-mutated obs and return a 4-d env action."""
        del info
        target_gate_int = int(np.asarray(obs["target_gate"]).item())
        n_gates = int(np.asarray(obs["gates_pos"]).shape[0])
        jax_obs = {key: jnp.asarray(value) for key, value in obs.items()}

        if self._ablate_mode == "clamp":
            actor_obs = _build_actor_obs_clamp(
                jax_obs, self.prev_action_env_4vec, self.actor_normalizer
            )
        else:
            actor_obs = obs_encoding.build_actor_obs(
                jax_obs, self.prev_action_env_4vec, self.actor_normalizer
            )

        if self._ablate_mode == "zero":
            actor_obs = actor_obs.at[LOOKAHEAD_START:LOOKAHEAD_END].set(0.0)
        elif self._ablate_mode == "randpost3":
            # Mutate only when the drone has passed the second-to-last gate,
            # i.e. is heading into the final gate. ``target_gate == -1`` is the
            # post-finish step; skip the mutation there to avoid divergent RNG
            # consumption between modes.
            if target_gate_int == n_gates - 1:
                self._ablate_rng, sub = jax.random.split(self._ablate_rng)
                noise = jax.random.normal(sub, (LOOKAHEAD_DIM,), dtype=actor_obs.dtype)
                actor_obs = actor_obs.at[LOOKAHEAD_START:LOOKAHEAD_END].set(noise)

        flat_obs = jnp.concatenate(
            [actor_obs, jnp.zeros((ACTOR_OBS_DIM,), dtype=actor_obs.dtype)], axis=-1
        )
        dist = self._actor.apply(self.actor_params, flat_obs[None, :])
        raw_action = dist.mean()[0]
        env_action = raw_to_env_action(
            raw_action,
            jax_obs["quat"],
            self.thrust_min,
            self.thrust_max,
            alpha_max=self.alpha_max_rad,
        )
        self.prev_action_env_4vec = env_action
        return np.asarray(env_action, dtype=np.float32)
