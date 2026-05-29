"""Diagnostic deploy controller — overlays commanded action on the render.

Subclasses :class:`RLSBXController` to capture per-step diagnostic state
(thrust, tau_scaled, env_action, next-gate corners) and draw it into the
MuJoCo offscreen render via crazyflow's :func:`draw_points` and
:func:`draw_line` helpers. Used for visual diagnosis of policy behavior,
not for evaluation or deployment.

The overlay components, all in world frame:

* **Thrust vector** — blue line from drone along its body +z axis, length
  scaled by ``thrust / thrust_max`` (0 m at min thrust, 0.5 m at max).
* **τ_scaled vector** — magenta line from drone along the body-frame
  per-step rotation axis, length scaled by ``‖τ_scaled‖ / alpha_max``
  (0 m at zero command, 0.3 m at the alpha_max budget). Direction shows
  which way the policy is trying to rotate this step.
* **Body axis triad** — short red/green/blue lines along the drone's
  current body x / y / z, length 0.1 m each. Anchors all the other arrows.
* **Target-gate corner vectors** — 4 saturated lines from the drone's
  position to each of the target gate's aperture corners, plus a sphere
  at each endpoint. Visualizes the body-frame target-corners channel
  (dims 13-24 of the actor obs) directly: as the drone yaws, the vectors
  yaw with it.
* **Lookahead-gate corner vectors** — 4 dimmer lines from the target
  gate to each of the lookahead gate's aperture corners. Visualizes the
  target-gate-frame next-corners channel (dims 25-36 of the actor obs):
  if the target gate is randomization-perturbed, these vectors anchor on
  the perturbed pose.
* **Trajectory trail** — fading red polyline of the last ~50 positions.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from crazyflow.sim.visualize import draw_line, draw_points

from lsy_drone_racing.control.rl_sbx.controller import RLSBXController
from lsy_drone_racing.control.rl_song import obs as obs_encoding
from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_DIM
from lsy_drone_racing.control.rl_song.policy import (
    THRUST_RAW_DIM,
    raw_to_env_action,
    scale_tangent,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from crazyflow.sim.sim import Sim
    from jax import Array

# Visual scales (m).
THRUST_ARROW_MAX_M: float = 0.5
TAU_ARROW_MAX_M: float = 0.3
BODY_AXIS_LEN_M: float = 0.1
CORNER_MARKER_SIZE_M: float = 0.04
TRAIL_LENGTH: int = 50

# Gate corner local positions — same convention as rl_song/obs.py.
GATE_HALF_Y: float = 0.20
GATE_HALF_Z: float = 0.20
_GATE_CORNERS_LOCAL: np.ndarray = np.asarray(
    [
        [0.0, +GATE_HALF_Y, +GATE_HALF_Z],
        [0.0, +GATE_HALF_Y, -GATE_HALF_Z],
        [0.0, -GATE_HALF_Y, +GATE_HALF_Z],
        [0.0, -GATE_HALF_Y, -GATE_HALF_Z],
    ],
    dtype=np.float32,
)
_CORNER_COLORS_TARGET: np.ndarray = np.asarray(
    [
        [1.0, 0.2, 0.2, 1.0],  # +y +z : red
        [1.0, 0.8, 0.2, 1.0],  # +y -z : yellow
        [0.2, 1.0, 0.2, 1.0],  # -y +z : green
        [0.2, 0.6, 1.0, 1.0],  # -y -z : blue
    ],
    dtype=np.float64,
)
# Lookahead corners drawn dimmer so they're visually distinguishable from
# the saturated target-gate corners.
_CORNER_COLORS_LOOKAHEAD: np.ndarray = _CORNER_COLORS_TARGET.copy()
_CORNER_COLORS_LOOKAHEAD[:, :3] *= 0.45
_CORNER_COLORS_LOOKAHEAD[:, 3] = 0.6
_LOOKAHEAD_MARKER_SIZE_M: float = 0.025


def _quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert an xyzw quaternion to a 3x3 rotation matrix using scipy."""
    from scipy.spatial.transform import Rotation as R

    return R.from_quat(np.asarray(quat, dtype=np.float64)).as_matrix()


class RLSBXDiagController(RLSBXController):
    """Diagnostic SBX deploy controller with action overlays."""

    def __init__(
        self,
        obs: dict[str, npt.NDArray[np.floating]],
        info: dict,
        config: dict,
    ) -> None:
        """Construct the parent controller and initialize diagnostic buffers."""
        super().__init__(obs, info, config)
        self._diag_pos: np.ndarray = np.zeros(3, dtype=np.float64)
        self._diag_quat: np.ndarray = np.zeros(4, dtype=np.float64)
        self._diag_tau_scaled: np.ndarray = np.zeros(3, dtype=np.float64)
        self._diag_thrust_norm: float = 0.0
        self._diag_target_corners_world: np.ndarray = np.zeros((4, 3), dtype=np.float64)
        self._diag_lookahead_corners_world: np.ndarray = np.zeros((4, 3), dtype=np.float64)
        self._diag_target_pos_world: np.ndarray = np.zeros(3, dtype=np.float64)
        self._diag_have_corners: bool = False
        self._trail: deque[np.ndarray] = deque(maxlen=TRAIL_LENGTH)

    def compute_control(
        self,
        obs: dict[str, npt.NDArray[np.floating]],
        info: dict | None = None,
    ) -> npt.NDArray[np.floating]:
        """Run the actor and capture per-step state for the render overlay."""
        del info
        jax_obs = {key: jnp.asarray(value) for key, value in obs.items()}
        actor_obs = obs_encoding.build_actor_obs(
            jax_obs, self.prev_action_env_4vec, self.actor_normalizer
        )
        flat_obs = jnp.concatenate(
            [actor_obs, jnp.zeros((ACTOR_OBS_DIM,), dtype=actor_obs.dtype)], axis=-1
        )
        dist = self._actor.apply(self.actor_params, flat_obs[None, :])
        raw_action: Array = dist.mean()[0]

        tau_raw = raw_action[THRUST_RAW_DIM:]
        tau_scaled = scale_tangent(tau_raw, self.alpha_max_rad)

        env_action = raw_to_env_action(
            raw_action,
            jax_obs["quat"],
            self.thrust_min,
            self.thrust_max,
            alpha_max=self.alpha_max_rad,
        )
        self.prev_action_env_4vec = env_action

        # Cache diagnostic state for render_callback.
        self._diag_pos = np.asarray(obs["pos"], dtype=np.float64)
        self._diag_quat = np.asarray(obs["quat"], dtype=np.float64)
        self._diag_tau_scaled = np.asarray(tau_scaled, dtype=np.float64)
        thrust = float(np.asarray(env_action)[3])
        self._diag_thrust_norm = (thrust - self.thrust_min) / max(
            self.thrust_max - self.thrust_min, 1e-9
        )

        target_idx = int(np.asarray(obs["target_gate"]).item())
        n_gates = int(np.asarray(obs["gates_pos"]).shape[0])
        if 0 <= target_idx < n_gates:
            lookahead_idx = (target_idx + 1) % n_gates
            target_pos = np.asarray(obs["gates_pos"][target_idx], dtype=np.float64)
            target_quat = np.asarray(obs["gates_quat"][target_idx], dtype=np.float64)
            target_rot = _quat_xyzw_to_matrix(target_quat)
            self._diag_target_corners_world = (
                _GATE_CORNERS_LOCAL @ target_rot.T
            ) + target_pos
            self._diag_target_pos_world = target_pos

            look_pos = np.asarray(obs["gates_pos"][lookahead_idx], dtype=np.float64)
            look_quat = np.asarray(obs["gates_quat"][lookahead_idx], dtype=np.float64)
            look_rot = _quat_xyzw_to_matrix(look_quat)
            self._diag_lookahead_corners_world = (
                _GATE_CORNERS_LOCAL @ look_rot.T
            ) + look_pos
            self._diag_have_corners = True
        else:
            self._diag_have_corners = False

        self._trail.append(self._diag_pos.copy())

        return np.asarray(env_action, dtype=np.float32)

    def render_callback(self, sim: "Sim") -> None:
        """Draw the cached action + perception overlays into the sim's scene."""
        if sim.viewer is None:
            return

        rot = _quat_xyzw_to_matrix(self._diag_quat)
        body_x = rot[:, 0]
        body_y = rot[:, 1]
        body_z = rot[:, 2]

        # Body axis triad (red, green, blue).
        draw_line(
            sim,
            np.stack([self._diag_pos, self._diag_pos + BODY_AXIS_LEN_M * body_x]),
            rgba=np.array([1.0, 0.0, 0.0, 1.0]),
        )
        draw_line(
            sim,
            np.stack([self._diag_pos, self._diag_pos + BODY_AXIS_LEN_M * body_y]),
            rgba=np.array([0.0, 1.0, 0.0, 1.0]),
        )
        draw_line(
            sim,
            np.stack([self._diag_pos, self._diag_pos + BODY_AXIS_LEN_M * body_z]),
            rgba=np.array([0.0, 0.0, 1.0, 1.0]),
        )

        # Thrust arrow along body +z, scaled by thrust fraction.
        thrust_len = THRUST_ARROW_MAX_M * float(np.clip(self._diag_thrust_norm, 0.0, 1.0))
        if thrust_len > 1e-3:
            draw_line(
                sim,
                np.stack([self._diag_pos, self._diag_pos + thrust_len * body_z]),
                rgba=np.array([0.2, 0.4, 1.0, 1.0]),
                start_size=6.0,
                end_size=6.0,
            )

        # τ_scaled arrow: body-frame rotation command this step.
        tau_world = rot @ self._diag_tau_scaled
        tau_norm = float(np.linalg.norm(tau_world))
        if tau_norm > 1e-4:
            scale = TAU_ARROW_MAX_M * tau_norm / self.alpha_max_rad
            tau_dir = tau_world / tau_norm
            draw_line(
                sim,
                np.stack([self._diag_pos, self._diag_pos + scale * tau_dir]),
                rgba=np.array([1.0, 0.0, 1.0, 1.0]),
                start_size=4.0,
                end_size=4.0,
            )

        # Target-gate corner vectors from the drone (body-frame obs channel)
        # and lookahead corner vectors from the target gate (target-frame
        # obs channel). Each ends with a sphere at the corner's world position.
        if self._diag_have_corners:
            for corner, rgba in zip(
                self._diag_target_corners_world, _CORNER_COLORS_TARGET
            ):
                draw_line(
                    sim,
                    np.stack([self._diag_pos, corner]),
                    rgba=rgba,
                    start_size=2.5,
                    end_size=2.5,
                )
                draw_points(
                    sim,
                    corner.reshape(1, 3),
                    rgba=rgba,
                    size=CORNER_MARKER_SIZE_M,
                )
            for corner, rgba in zip(
                self._diag_lookahead_corners_world, _CORNER_COLORS_LOOKAHEAD
            ):
                draw_line(
                    sim,
                    np.stack([self._diag_target_pos_world, corner]),
                    rgba=rgba,
                    start_size=1.5,
                    end_size=1.5,
                )
                draw_points(
                    sim,
                    corner.reshape(1, 3),
                    rgba=rgba,
                    size=_LOOKAHEAD_MARKER_SIZE_M,
                )

        # Trajectory trail.
        if len(self._trail) >= 2:
            trail_arr = np.asarray(self._trail, dtype=np.float64)
            draw_line(
                sim,
                trail_arr,
                rgba=np.array([1.0, 0.5, 0.0, 0.9]),
                start_size=2.0,
                end_size=2.0,
            )
