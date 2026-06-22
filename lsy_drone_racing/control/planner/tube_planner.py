import logging
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.planner.sfc_planner_mpc import SkeletonPoint
from lsy_drone_racing.control.planner_utils.environment import (
    Capsule,
    get_obstacle_capsules,
    get_gate_capsules,
)
from lsy_drone_racing.control.planner_utils.environment_config import EnvironmentConfig

logger = logging.getLogger(__name__)


class TubePlanner:
    """A lightweight planner that connects gates via spline and outputs tube obstacles."""

    def __init__(self, obs: dict[str, np.ndarray], freq: int, config: Any) -> None:
        self.config = config
        self.env_config = EnvironmentConfig()
        self.freq = freq
        self._tick = 0
        self._last_replan_tick = -1000
        self.last_replan_event: dict[str, Any] | None = None

        self.gates_pos = obs.get("gates_pos", np.array([]))
        self.gates_quat = obs.get("gates_quat", np.array([]))
        self.obstacles_pos = obs.get("obstacles_pos", np.array([]))

        self.target_gate_idx = int(obs.get("target_gate", 0))
        if self.target_gate_idx == -1:
            self.target_gate_idx = len(self.gates_pos)

        self.capsules: list[Capsule] = []
        self.corridors: list[Any] = []  # Not used by the tube constraint
        self.skeleton_path: list[SkeletonPoint] = []
        self.control_points: np.ndarray | None = None

        self._build_spline(obs["pos"], obs.get("vel", np.zeros(3)))

    def evaluate_corridor_spatial(self, u: float) -> tuple[Any, float]:
        """Return dynamic tube constraints.
        Returns (None, radius) where radius shrinks near gates.
        """
        if callable(self._spline):
            pos = self._spline(u)
        else:
            pos = self.des_pos_spline(u)

        # 1m at max, reducing to 40cm gate opening size (20cm radius)
        default_radius = 1.0
        gate_radius = 0.2

        radius = default_radius
        if len(self.gates_pos) > 0:
            dists = np.linalg.norm(self.gates_pos - pos, axis=1)
            min_dist = np.min(dists)
            if min_dist < 2.0:
                # Smooth reduction all the way to the gate center
                t = min_dist / 2.0
                radius = gate_radius + t * (default_radius - gate_radius)

        return None, radius

    def _build_spline(self, current_pos: np.ndarray, current_vel: np.ndarray) -> None:
        self._last_replan_tick = self._tick
        self.last_replan_event = {"reason": "init"}

        pts = [current_pos]
        weights = [10.0]

        # Build smart skeleton path to gates
        for i in range(self.target_gate_idx, len(self.gates_pos)):
            pos = self.gates_pos[i]
            rot = R.from_quat(self.gates_quat[i])
            normal = rot.apply([1.0, 0.0, 0.0])
            right = rot.apply([0.0, 1.0, 0.0])

            # Anchor points to ensure straight passage through the gate opening
            anchor_dist = 0.5

            # If the distance to the gate is small, scale down the anchor distance
            # to prevent self-intersecting loops or excessive wiggles.
            dist_to_gate = np.linalg.norm(pos - pts[-1])
            eff_anchor = min(anchor_dist, max(0.1, dist_to_gate / 3.0))

            p_pre = pos - normal * eff_anchor
            p_post = pos + normal * eff_anchor

            # Smart logic for sharp turns (U-Turns, opposing gates, off-axis approaches)
            flow = p_pre - pts[-1]
            flow_dist = np.linalg.norm(flow)

            if flow_dist > 0.1:
                flow_dir = flow / flow_dist
                # If dot product is < 0.5 (angle > 60 deg), we are approaching from a bad angle
                if np.dot(flow_dir, normal) < 0.5:
                    lat_dist = np.dot(pts[-1] - pos, right)
                    # Swing to the side the drone is already on, to create a wider arc
                    swing_dir = right if lat_dist > 0 else -right

                    swing_dist = min(1.5, max(0.5, dist_to_gate * 0.4))
                    # Move outward laterally and slightly backward from the gate
                    swing_pos = pos + swing_dir * swing_dist - normal * (swing_dist * 0.5)

                    pts.append(swing_pos)
                    weights.append(2.0)  # lower weight so it acts as a soft guide

            pts.append(p_pre)
            weights.append(10.0)
            pts.append(pos)
            weights.append(100.0)
            pts.append(p_post)
            weights.append(10.0)

            self.skeleton_path.append(SkeletonPoint(pos, True, normal, right, None, gate_idx=i))

        # Fast obstacle avoidance: push skeleton path away from obstacles
        req_dist = getattr(self.env_config, "pole_radius", 0.2) + getattr(
            self.env_config, "safety_margin", 0.1
        )

        for _ in range(5):
            new_pts = [pts[0]]
            new_weights = [weights[0]]
            collision_found = False
            for i in range(len(pts) - 1):
                A = pts[i]
                B = pts[i + 1]
                w_B = weights[i + 1]

                closest_obs = None
                min_dist = float("inf")
                closest_proj = None

                for obs in self.obstacles_pos:
                    A2 = A[:2]
                    B2 = B[:2]
                    O2 = obs[:2]

                    AB = B2 - A2
                    len_AB = np.linalg.norm(AB)
                    if len_AB < 1e-3:
                        continue
                    dir_AB = AB / len_AB

                    AP = O2 - A2
                    t = np.dot(AP, dir_AB)
                    t_clamped = np.clip(t, 0, len_AB)
                    proj2 = A2 + t_clamped * dir_AB

                    dist = np.linalg.norm(O2 - proj2)
                    if dist < req_dist:
                        if dist < min_dist:
                            min_dist = dist
                            closest_obs = obs
                            t_3d = t_clamped / len_AB
                            closest_proj = A + t_3d * (B - A)

                if closest_obs is not None:
                    collision_found = True
                    O2 = closest_obs[:2]
                    P2 = closest_proj[:2]
                    vec = P2 - O2
                    if np.linalg.norm(vec) < 1e-5:
                        dir_AB = B[:2] - A[:2]
                        dir_AB = dir_AB / np.linalg.norm(dir_AB)
                        vec = np.array([-dir_AB[1], dir_AB[0]])
                    vec = vec / np.linalg.norm(vec)

                    safe_pt2 = O2 + vec * (req_dist + 0.1)
                    safe_pt = closest_proj.copy()
                    safe_pt[:2] = safe_pt2
                    new_pts.append(safe_pt)
                    new_weights.append(5.0)

                new_pts.append(B)
                new_weights.append(w_B)

            pts = new_pts
            weights = new_weights
            if not collision_found:
                break

        pts_array_raw = np.array(pts)
        w_array_raw = np.array(weights)

        # Remove 3D duplicates and enforce minimum Z
        unique_pts = [pts_array_raw[0]]
        unique_weights = [w_array_raw[0]]
        for i in range(1, len(pts_array_raw)):
            pt = pts_array_raw[i].copy()
            pt[2] = max(0.15, pt[2])  # Ensure Z is at least 0.15 to avoid ground collisions
            if np.linalg.norm(pt - unique_pts[-1]) > 1e-2:
                unique_pts.append(pt)
                unique_weights.append(w_array_raw[i])

        pts_array = np.array(unique_pts)
        w_array = np.array(unique_weights)

        # Fit B-spline with weights. s=1.0 allows smoothing.
        k = min(len(pts_array) - 1, 3)
        if k < 1:
            self.control_points = pts_array
            return

        tck, u = splprep([pts_array[:, 0], pts_array[:, 1], pts_array[:, 2]], w=w_array, s=1.0, k=k)

        # Sample smoothed points densely
        u_fine = np.linspace(0, 1, max(100, len(pts_array) * 10))
        smooth_pts = np.vstack(splev(u_fine, tck)).T

        self.control_points = smooth_pts

        # Compute chord lengths for parameterization
        diffs = np.diff(smooth_pts, axis=0)
        chords = np.linalg.norm(diffs, axis=1)
        s = np.concatenate(([0.0], np.cumsum(chords)))

        # Create the spline, normalized to domain [0, 1] as expected by sfc_mpcc
        s_norm = s / s[-1] if s[-1] > 0 else s

        if len(s_norm) < 4:
            k_spline = min(len(s_norm) - 1, 3)
            self._spline = CubicSpline(
                s_norm, smooth_pts, bc_type="natural" if k_spline == 3 else "not-a-knot"
            )
        else:
            self._spline = CubicSpline(s_norm, smooth_pts, bc_type="natural")

        self._s_total = float(s[-1])

        # 2. Build Capsules for obstacle avoidance (Gate posts)
        # We don't model the corridor as capsules here, because the tube constraint
        # will be strictly enforced in the MPCC via a nonlinear constraint on e_c.
        self.capsules = get_obstacle_capsules(self.obstacles_pos, self.env_config)
        self.capsules.extend(get_gate_capsules(self.gates_pos, self.gates_quat, self.env_config))

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
            self._build_spline(obs["pos"], obs.get("vel", np.zeros(3)))
            return True

        return False

    def des_pos_spline(self, u: np.ndarray | float) -> np.ndarray:
        return self._spline(u)

    def episode_reset(self) -> None:
        self._tick = 0
        self._last_replan_tick = -1000

    def add_trajectory_point(self, pos: np.ndarray) -> None:
        pass

    def get_trajectory_history(self) -> list:
        return []
