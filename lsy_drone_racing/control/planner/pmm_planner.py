import logging
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control.planner_utils.environment import (
    Capsule,
    get_obstacle_capsules,
    get_gate_capsules,
)
from lsy_drone_racing.control.planner_utils.environment_config import EnvironmentConfig
from lsy_drone_racing.control.planner.pmm_planner_config import PmmPlannerConfig
from lsy_drone_racing.control.planner.pmm_utils import pmm_cone_refocusing, evaluate_1d_trajectory
from lsy_drone_racing.control.planner.base_planner import BasePlanner

logger = logging.getLogger(__name__)


class PmmPlanner(BasePlanner):
    """A Point-Mass Model planner that implements time-optimal trajectory generation via Cone Refocusing."""

    def __init__(self, obs: dict[str, np.ndarray], freq: int, config: PmmPlannerConfig) -> None:
        self.env_config = EnvironmentConfig()
        super().__init__(obs, freq, config)

        self.capsules: list[Capsule] = []
        self.corridors: list[Any] = []
        
        self.capsules = get_obstacle_capsules(self.obstacles_pos, self.env_config)
        self.capsules.extend(get_gate_capsules(self.gates_pos, self.gates_quat, self.env_config))

        self._build_trajectory(obs["pos"], obs.get("vel", np.zeros(3)))

    def _build_trajectory(self, current_pos: np.ndarray, current_vel: np.ndarray) -> None:
        self._last_replan_tick = self._tick
        self.last_replan_event = {"reason": "init"}

        if self.target_gate_idx >= len(self.gates_pos):
            # No more gates, just hover or hold position
            self._spline = CubicSpline([0.0, 1.0], [current_pos, current_pos], bc_type="natural")
            return

        # Get the next Hg gates
        end_idx = min(len(self.gates_pos), self.target_gate_idx + self.config.gate_horizon)
        p_waypoints = self.gates_pos[self.target_gate_idx:end_idx]

        # Call Cone Refocusing
        sols, vs = pmm_cone_refocusing(current_pos, current_vel, p_waypoints, self.config)

        if sols is None or len(sols) == 0:
            logger.warning("PMM Cone Refocusing failed to find a valid trajectory.")
            self._spline = CubicSpline([0.0, 1.0], [current_pos, current_pos], bc_type="natural")
            return

        # Build dense points from piecewise quadratic trajectories
        pts = []
        p_curr = current_pos.copy()
        v_curr = current_vel.copy()
        
        for i, sol in enumerate(sols):
            # sol is a list of 3 tuples: [(t1x, t2x, u1x, u2x), (t1y, ...), (t1z, ...)]
            # the total time is the same for x, y, z (approx)
            T_segment = sol[0][0] + sol[0][1]
            t_evals = np.linspace(0, T_segment, max(10, int(T_segment / 0.05)))
            
            for t in t_evals:
                px, vx = evaluate_1d_trajectory(p_curr[0], v_curr[0], sol[0][0], sol[0][1], sol[0][2], sol[0][3], t)
                py, vy = evaluate_1d_trajectory(p_curr[1], v_curr[1], sol[1][0], sol[1][1], sol[1][2], sol[1][3], t)
                pz, vz = evaluate_1d_trajectory(p_curr[2], v_curr[2], sol[2][0], sol[2][1], sol[2][2], sol[2][3], t)
                pts.append([px, py, pz])
                
            # Update curr state for next segment
            px_f, vx_f = evaluate_1d_trajectory(p_curr[0], v_curr[0], sol[0][0], sol[0][1], sol[0][2], sol[0][3], T_segment)
            py_f, vy_f = evaluate_1d_trajectory(p_curr[1], v_curr[1], sol[1][0], sol[1][1], sol[1][2], sol[1][3], T_segment)
            pz_f, vz_f = evaluate_1d_trajectory(p_curr[2], v_curr[2], sol[2][0], sol[2][1], sol[2][2], sol[2][3], T_segment)
            
            p_curr = np.array([px_f, py_f, pz_f])
            v_curr = np.array([vx_f, vy_f, vz_f])

        # Remove consecutive duplicates
        pts_array = np.array(pts)
        unique_pts = [pts_array[0]]
        for i in range(1, len(pts_array)):
            if np.linalg.norm(pts_array[i] - unique_pts[-1]) > 1e-4:
                unique_pts.append(pts_array[i])
        unique_pts = np.array(unique_pts)

        if len(unique_pts) < 2:
            self._spline = CubicSpline([0.0, 1.0], [current_pos, current_pos], bc_type="natural")
            return

        # Fit cubic spline parametrized by normalized chord length
        diffs = np.diff(unique_pts, axis=0)
        chords = np.linalg.norm(diffs, axis=1)
        s = np.concatenate(([0.0], np.cumsum(chords)))
        if s[-1] > 0:
            s_norm = s / s[-1]
        else:
            s_norm = s

        k_spline = min(len(s_norm) - 1, 3)
        self._spline = CubicSpline(
            s_norm, unique_pts, bc_type="natural" if k_spline == 3 else "not-a-knot"
        )

    def update(self, obs: dict[str, np.ndarray]) -> bool:
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

        if gate_changed:
            self._build_trajectory(obs["pos"], obs.get("vel", np.zeros(3)))
            return True

        return False

    def des_pos_spline(self, u: np.ndarray | float) -> np.ndarray:
        return self._spline(u)
