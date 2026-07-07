"""Pure SFC trajectory planner — extracted from sfc_controller.py.

Provides the path-skeleton builder, capsule obstacle model, flight-corridor
builder, and B-spline optimization. Consumed by both `sfc_controller.py`
(state-mode tracker) and `sfc_attitude_controller.py` (attitude-mode tracker).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import BSpline

from lsy_drone_racing.control.planner.geometry import (
    generate_flight_corridors,
    get_all_obstacle_capsules,
)
from lsy_drone_racing.control.planner.path_optimizer import CasadiPlanner
from lsy_drone_racing.control.planner.skeleton import calculate_anchors
from lsy_drone_racing.control.sfc_planner_mpc_config import PlannerConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SfcCorridorPlanner:
    """Pure SFC trajectory planner. Build once, update each tick, evaluate at any time."""

    def __init__(
        self, obs: dict[str, NDArray], freq: int, config: PlannerConfig | None = None
    ) -> None:
        """Initialize the SfcCorridorPlanner.

        Args:
            obs: Initial observation dict containing drone state and target gate.
            freq: The control frequency.
            config: Optional PlannerConfig instance.
        """
        self._freq = freq
        self.config = config or PlannerConfig()

        self.gates_pos = obs["gates_pos"].copy()
        self.gates_quat = obs["gates_quat"].copy()
        self.obstacles_pos = obs.get("obstacles_pos", np.array([])).copy()
        self.target_gate_idx = 0
        self._tick = 0
        self._last_replan_tick = -self.config.REPLAN_DEBOUNCE_TICKS

        self.replan_events: list[dict] = []
        self.last_replan_event: dict | None = None
        self._traj_history = []

        self._casadi_planner = CasadiPlanner(self.config)

        initial_vel = obs.get("vel", np.zeros(3))
        self._build_spline(obs["pos"], initial_vel)
        self._record_replan_event(reason="init")

    def update(self, obs: dict[str, NDArray]) -> bool:
        """Update the planner state and recompute the path if necessary.

        Args:
            obs: The current observation dict.

        Returns:
            True if the path was replanned, False otherwise.
        """
        self._tick += 1

        env_target = int(obs.get("target_gate", self.target_gate_idx))
        gate_changed = False
        if env_target == -1:
            if self.target_gate_idx != len(self.gates_pos):
                gate_changed = True
            self.target_gate_idx = len(self.gates_pos)
        else:
            if self.target_gate_idx != env_target:
                gate_changed = True
            self.target_gate_idx = env_target

        moved, reason = self._check_objects_moved(obs)

        if not moved and not gate_changed:
            return False
        if self._tick - self._last_replan_tick < self.config.REPLAN_DEBOUNCE_TICKS:
            return False
        if self.target_gate_idx >= len(self.gates_pos) and not gate_changed:
            return False

        if not moved and gate_changed:
            reason = "gate_passed"

        if moved:
            self.gates_pos = obs["gates_pos"].copy()
            self.gates_quat = obs["gates_quat"].copy()
            self.obstacles_pos = obs.get("obstacles_pos", np.array([])).copy()

        self._build_spline(obs["pos"], obs.get("vel", np.zeros(3)))
        self._last_replan_tick = self._tick
        self._record_replan_event(reason=reason)
        return True

    def evaluate_spatial(self, u: float) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Evaluate the B-spline path at a spatial parameter u in [0, 1].

        Args:
            u: The spatial parameter (normalized arc length approximation).

        Returns:
            A tuple of (position, velocity, acceleration, jerk) arrays.
        """
        if not hasattr(self, "_des_pos_spline") or self._des_pos_spline is None:
            cp_last = np.asarray(self._control_points[-1], dtype=np.float64)
            return cp_last, np.zeros(3), np.zeros(3), np.zeros(3)

        u_clamped = float(np.clip(u, 0.0, 1.0))

        pos = np.asarray(self._des_pos_spline(u_clamped), dtype=np.float64)
        dpos = np.asarray(self._des_pos_spline.derivative(nu=1)(u_clamped), dtype=np.float64)
        ddpos = np.asarray(self._des_pos_spline.derivative(nu=2)(u_clamped), dtype=np.float64)
        dddpos = np.asarray(self._des_pos_spline.derivative(nu=3)(u_clamped), dtype=np.float64)

        return pos, dpos, ddpos, dddpos

    def evaluate_corridor_spatial(self, u: float) -> tuple[NDArray, NDArray] | None:
        """Get the flight corridor hyperplanes for a spatial parameter u.

        Args:
            u: The spatial parameter in [0, 1].

        Returns:
            A tuple of (A, b) defining the safe region, or None if no corridors exist.
        """
        if not hasattr(self, "corridors") or self.corridors is None or len(self.corridors) == 0:
            return None

        u_clamped = float(np.clip(u, 0.0, 1.0))
        n_segments = len(self.corridors)
        seg_idx = int(np.floor(u_clamped * n_segments))
        seg_idx = min(seg_idx, n_segments - 1)

        A = np.array(self.corridors[seg_idx].A)
        b = np.array(self.corridors[seg_idx].b)
        return A, b

    @property
    def des_pos_spline(self) -> BSpline:
        """Get the desired position B-spline."""
        return self._des_pos_spline

    @property
    def control_points(self) -> NDArray:
        """Get the optimized B-spline control points."""
        return self._control_points

    def episode_reset(self) -> None:
        """Reset the planner state for a new episode."""
        self.target_gate_idx = 0
        self._tick = 0
        self._last_replan_tick = -self.config.REPLAN_DEBOUNCE_TICKS
        self.replan_events = []
        self.last_replan_event = None
        self._traj_history = []

    def _build_spline(self, current_pos: NDArray, current_vel: NDArray) -> None:
        skeleton_path = calculate_anchors(
            current_pos[:3],
            current_vel,
            self.obstacles_pos,
            self.gates_pos,
            self.gates_quat,
            self.target_gate_idx,
            self.config,
        )
        self.skeleton_path = skeleton_path
        self._current_pos_for_spline = current_pos[:3].copy()
        capsules = get_all_obstacle_capsules(
            self.obstacles_pos, self.gates_pos, self.gates_quat, self.config
        )
        self.capsules = capsules
        corridors = generate_flight_corridors(
            skeleton_path, capsules, self.gates_pos, self.gates_quat, self.config
        )
        self.corridors = corridors

        control_points = self._casadi_planner.optimize_control_points(
            skeleton_path, corridors, current_vel, self._current_pos_for_spline
        )
        self._control_points = control_points

        k = 3
        n_ctrl = len(control_points)

        cp_dists = np.maximum(np.linalg.norm(np.diff(control_points, axis=0), axis=1), 1e-4)
        u_params = np.concatenate(([0], np.cumsum(cp_dists)))
        if u_params[-1] > 0:
            u_params /= u_params[-1]

        knots = np.zeros(n_ctrl + k + 1)
        knots[: k + 1] = 0.0
        knots[-k - 1 :] = 1.0

        for i in range(1, n_ctrl - k):
            knots[i + k] = np.mean(u_params[i : i + k])

        self._des_pos_spline = BSpline(knots, control_points, k)

    def _check_objects_moved(self, obs: dict[str, NDArray]) -> tuple[bool, str]:
        gate_moved = False
        obs_moved = False
        new_gates_pos = obs["gates_pos"]
        jitter_th = self.config.JITTER_THRESHOLD
        if (
            len(self.gates_pos) > 0
            and np.max(np.linalg.norm(new_gates_pos - self.gates_pos, axis=1)) > jitter_th
        ):
            gate_moved = True

        new_obs_pos = obs.get("obstacles_pos", np.array([]))
        if len(new_obs_pos) != len(self.obstacles_pos) or (
            len(new_obs_pos) > 0
            and np.max(np.linalg.norm(new_obs_pos - self.obstacles_pos, axis=1)) > jitter_th
        ):
            obs_moved = True

        if gate_moved and obs_moved:
            reason = "gate+obstacle_jitter"
        elif gate_moved:
            reason = "gate_jitter"
        elif obs_moved:
            reason = "obstacle_jitter"
        else:
            reason = ""
        return gate_moved or obs_moved, reason

    def current_spline_snapshot(self) -> dict:
        """Get a snapshot of the current spline state for tracing/debugging."""
        return {
            "knots": np.asarray(self._des_pos_spline.t, dtype=np.float64).copy(),
            "control_points": np.asarray(self._control_points, dtype=np.float64).copy(),
            "k": int(self._des_pos_spline.k),
            "target_gate_idx": int(self.target_gate_idx),
        }

    def _record_replan_event(self, reason: str) -> None:
        evt = {
            "tick": int(self._tick),
            "reason": reason,
            "snapshot": self.current_spline_snapshot(),
        }
        self.replan_events.append(evt)
        self.last_replan_event = evt

    def add_trajectory_point(self, pos: NDArray) -> None:
        """Add a position to the trajectory history.

        Args:
            pos: The drone position to record.
        """
        self._traj_history.append(pos.copy())

    def get_trajectory_history(self) -> NDArray:
        """Get the recorded trajectory history as an array.

        Returns:
            An array of recorded drone positions.
        """
        if len(self._traj_history) == 0:
            return np.empty((0, 3))
        return np.array(self._traj_history)
