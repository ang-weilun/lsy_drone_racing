"""Adaptive controller that follows a dynamically generated, obstacle-avoiding trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
The controller adaptively recomputes the trajectory when gate poses update or new obstacles
are detected. It uses exact Safe Flight Corridors (SFC) for hard safety margins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import cvxpy as cp
import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class SkeletonPoint(NamedTuple):
    """Represents a skeleton point in the planned path with gate information."""

    pos: NDArray
    is_gate: bool
    gate_normal: NDArray | None
    gate_right: NDArray | None
    gate_up: NDArray | None


class Capsule(NamedTuple):
    """Represents a capsule obstacle (cylinder with spherical ends)."""

    p1: NDArray
    p2: NDArray
    radius: float
    is_gate: bool


class FlightCorridor:
    """Represents a convex polyhedron (flight corridor) defined by half-spaces."""

    def __init__(self, p1: NDArray, p2: NDArray) -> None:
        """Initialize a flight corridor between two waypoints.

        Args:
            p1: Start point of the corridor.
            p2: End point of the corridor.
        """
        self.A = []
        self.b = []
        self.p1 = p1
        self.p2 = p2

        # Bounding box (Room limits)
        self.add_halfspace(np.array([0, 0, 1]), np.array([0, 0, 3.0]))
        self.add_halfspace(np.array([0, 0, -1]), np.array([0, 0, -0.2]))
        self.add_halfspace(np.array([1, 0, 0]), np.array([15.0, 0, 0]))
        self.add_halfspace(np.array([-1, 0, 0]), np.array([-15.0, 0, 0]))
        self.add_halfspace(np.array([0, 1, 0]), np.array([0, 15.0, 0]))
        self.add_halfspace(np.array([0, -1, 0]), np.array([0, -15.0, 0]))

    def add_halfspace(self, n: NDArray, p: NDArray):
        """Adds a constraint (x - p) * n <= 0, where n is the OUTWARD normal."""
        self.A.append(n)
        self.b.append(np.dot(n, p))


def closest_points_segments(
    p1: NDArray, q1: NDArray, p2: NDArray, q2: NDArray
) -> tuple[NDArray, NDArray]:
    """Finds the closest points c1 on segment p1-q1 and c2 on segment p2-q2."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)

    if a <= 1e-6 and e <= 1e-6:
        return p1, p2
    if a <= 1e-6:
        s = np.clip(f / e, 0.0, 1.0)
        return p1, p2 + s * d2

    c = np.dot(d1, r)
    if e <= 1e-6:
        t = np.clip(-c / a, 0.0, 1.0)
        return p1 + t * d1, p2

    b = np.dot(d1, d2)
    denom = a * e - b * b

    if denom != 0.0:
        t = np.clip((b * f - c * e) / denom, 0.0, 1.0)
    else:
        t = 0.0

    s = (b * t + f) / e
    if s < 0.0:
        s = 0.0
        t = np.clip(-c / a, 0.0, 1.0)
    elif s > 1.0:
        s = 1.0
        t = np.clip((b - c) / a, 0.0, 1.0)

    return p1 + t * d1, p2 + s * d2


class StateController(Controller):
    """State controller following a Safe Flight Corridor optimized B-Spline trajectory."""

    W_VEL = 2.0
    W_ACC = 6.0
    W_JERK = 10.0
    W_CENTER = 0.01

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict) -> None:
        """Initialize the Safe Flight Corridor controller.

        Args:
            obs: Dictionary of observations from the environment.
            info: Dictionary of environment info.
            config: Configuration dictionary.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._tick = 0
        self._spline_tick = 0
        self._finished = False

        self.anchor_gap = 0.5
        self.base_speed = 1.0
        self.points_per_segment = 4

        self.gate_outer = 0.72
        self.gate_inner = 0.40
        self.gate_thickness = (self.gate_outer - self.gate_inner) / 2.0
        self.gate_depth = 0.10

        self.pole_radius = 0.03 / 2.0
        self.pole_height = 1.52

        self.safety_margin = 0.15

        self.gates_pos = obs["gates_pos"].copy()
        self.gates_quat = obs["gates_quat"].copy()
        self.obstacles_pos = obs.get("obstacles_pos", np.array([]))
        self.target_gate_idx = 0
        self._prev_pos = None

        initial_vel = obs.get("vel", np.zeros(3))
        self.generate_spline(obs["pos"], initial_vel)

    def generate_spline(self, current_pos: NDArray, current_vel: NDArray) -> None:
        """Generate a B-spline trajectory through safe flight corridors.

        Args:
            current_pos: Current position of the drone.
            current_vel: Current velocity of the drone.
        """
        skeleton_path = self._calculate_anchors(current_pos[:3])
        self.skeleton_path = skeleton_path
        self._current_pos_for_spline = current_pos[:3].copy()
        capsules = self._get_all_obstacle_capsules()
        corridors = self._generate_flight_corridors(skeleton_path, capsules)

        control_points = self._optimize_control_points(skeleton_path, corridors, current_vel)
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
        self._t_total = np.sum(cp_dists) / self.base_speed
        self._spline_tick = 0

    def _get_all_obstacle_capsules(self) -> list[Capsule]:
        """Converts all exact trimesh track boundaries into 3D capsule obstacles."""
        capsules = []
        margin = self.safety_margin

        # Poles
        for p in self.obstacles_pos:
            capsules.append(
                Capsule(
                    np.array([p[0], p[1], 0.0]),
                    np.array([p[0], p[1], self.pole_height]),
                    self.pole_radius + margin,
                    False,
                )
            )

        # Gates (Stands & 4-bar inner frames)
        for pos, quat in zip(self.gates_pos, self.gates_quat):
            rot = R.from_quat(quat)
            up = rot.apply([0, 0, 1])
            right = rot.apply([0, 1, 0])

            # Stands
            stand_h = pos[2] - self.gate_outer / 2.0
            if stand_h > 0:
                capsules.append(
                    Capsule(
                        pos - up * (self.gate_outer / 2.0),
                        pos - up * (self.gate_outer / 2.0 + stand_h),
                        0.05 + margin,
                        True,
                    )
                )

            # Gate Frame Bars
            bar_dist = 0.28
            bar_radius = 0.08 + margin

            capsules.append(
                Capsule(
                    pos + up * bar_dist - right * 0.36,
                    pos + up * bar_dist + right * 0.36,
                    bar_radius,
                    True,
                )
            )
            capsules.append(
                Capsule(
                    pos - up * bar_dist - right * 0.36,
                    pos - up * bar_dist + right * 0.36,
                    bar_radius,
                    True,
                )
            )
            capsules.append(
                Capsule(
                    pos - right * bar_dist + up * 0.36,
                    pos - right * bar_dist - up * 0.36,
                    bar_radius,
                    True,
                )
            )
            capsules.append(
                Capsule(
                    pos + right * bar_dist + up * 0.36,
                    pos + right * bar_dist - up * 0.36,
                    bar_radius,
                    True,
                )
            )

        return capsules

    def _generate_flight_corridors(
        self, skeleton_path: list[SkeletonPoint], capsules: list[Capsule]
    ) -> list[FlightCorridor]:
        """Constructs a convex polyhedron for each segment via separating planes."""
        corridors = []
        for i in range(len(skeleton_path) - 1):
            pt1 = skeleton_path[i]
            pt2 = skeleton_path[i + 1]
            corr = FlightCorridor(pt1.pos, pt2.pos)

            # Add separating half-spaces for all capsules
            for cap in capsules:
                # Do not apply collision capsules from the gate we are currently
                # routing through or approaching. If either endpoint of the segment
                # is within 1.0m of the capsule, assume it's part of the gate
                # structure and the skeleton path is already safely routed through.
                if cap.is_gate and (
                    np.linalg.norm(cap.p1 - pt1.pos) < 1.0 or np.linalg.norm(cap.p1 - pt2.pos) < 1.0
                ):
                    continue

                c1, c2 = closest_points_segments(pt1.pos, pt2.pos, cap.p1, cap.p2)
                vec = c1 - c2  # Points from obstacle towards the segment
                dist = np.linalg.norm(vec)

                if dist > 1e-5:
                    n = vec / dist
                else:
                    d1 = pt2.pos - pt1.pos
                    perp = (
                        np.array([-d1[1], d1[0], 0])
                        if np.linalg.norm(d1[:2]) > 1e-5
                        else np.array([1.0, 0.0, 0.0])
                    )
                    n = perp / np.linalg.norm(perp)

                effective_radius = min(cap.radius, dist - 0.005)
                plane_p = c2 + n * effective_radius
                corr.add_halfspace(-n, plane_p)

            corridors.append(corr)
        return corridors

    def _optimize_control_points(
        self,
        skeleton_path: list[SkeletonPoint],
        corridors: list[FlightCorridor],
        current_vel: NDArray,
    ) -> NDArray:
        """Solves a QP to find optimal control points strictly within the Safe Corridors."""
        n_segments = len(corridors)
        pts_per_seg = self.points_per_segment

        # Determine first segment points based on distance to next waypoint
        if len(skeleton_path) > 1:
            dist_to_next = np.linalg.norm(skeleton_path[1].pos - skeleton_path[0].pos)
            if dist_to_next < 0.25:
                pts_first_seg = 1
            elif dist_to_next < 0.50:
                pts_first_seg = 2
            elif dist_to_next < 0.75:
                pts_first_seg = 3
            else:
                pts_first_seg = 4
        else:
            pts_first_seg = 1

        pts_rest_seg = pts_per_seg
        n_ctrl = pts_first_seg + (n_segments - 1) * pts_rest_seg

        P = cp.Variable((n_ctrl, 3))
        constraints = []

        # Build reference points with variable points per segment
        reference_points_list = []
        for i in range(n_segments):
            n_pts = pts_first_seg if i == 0 else pts_rest_seg
            for j in range(n_pts):
                pt = corridors[i].p1 + (j / n_pts) * (corridors[i].p2 - corridors[i].p1)
                reference_points_list.append(pt)
        reference_points = np.array(reference_points_list)

        # Apply corridor constraints with variable points per segment
        idx = 0
        for seg_idx, corr in enumerate(corridors):
            A = np.array(corr.A)
            b = np.array(corr.b)
            n_pts = pts_first_seg if seg_idx == 0 else pts_rest_seg
            for _ in range(n_pts):
                constraints.append(A @ P[idx] <= b)
                idx += 1

        constraints.extend([P[-1] == skeleton_path[-1].pos])

        # Build a mapping from skeleton path indices to control point indices
        cp_idx_map = [0]  # skeleton_path[0] maps to control point 0
        idx = pts_first_seg
        for seg_idx in range(1, n_segments):
            cp_idx_map.append(idx)
            idx += pts_rest_seg
        cp_idx_map.append(n_ctrl - 1)  # Last skeleton point maps to last control point

        for i in range(1, len(skeleton_path) - 1):
            if skeleton_path[i].is_gate:
                gate_cp_idx = cp_idx_map[i]
                normal = skeleton_path[i].gate_normal
                constraints.append(P[gate_cp_idx] == skeleton_path[i].pos)

        cost = (
            self.W_VEL * cp.sum_squares(cp.diff(P, axis=0))
            + self.W_ACC * cp.sum_squares(cp.diff(P, k=2, axis=0))
            + self.W_JERK * cp.sum_squares(cp.diff(P, k=3, axis=0))
            + self.W_CENTER * cp.sum_squares(P - reference_points)
        )

        # Initial position and velocity continuity: anchor spline to current drone state
        current_pos = self._current_pos_for_spline
        cost += 10.0 * cp.sum_squares(P[0] - current_pos)  # Soft anchor P[0] near current pos

        # C1 Continuity (Initial Velocity Matching)
        speed = np.linalg.norm(current_vel)
        if speed > 0.1:
            # Predict where the drone will be in the next 0.05 seconds
            dt = 0.05
            p_expected = current_pos + current_vel * dt
            cost += 50.0 * cp.sum_squares(P[1] - p_expected)  # Strong velocity matching
        else:
            # If stationary, just keep next point close
            cost += 10.0 * cp.sum_squares(P[1] - current_pos)

        for i in range(1, len(skeleton_path) - 1):
            if skeleton_path[i].is_gate:
                gate_cp_idx = cp_idx_map[i]
                normal = skeleton_path[i].gate_normal

                # Enforce symmetry around the gate for smooth straight passage
                if gate_cp_idx - 1 >= 0 and gate_cp_idx + 1 < n_ctrl:
                    constraints.append(
                        P[gate_cp_idx - 1] + P[gate_cp_idx + 1] == 2 * P[gate_cp_idx]
                    )

                # Softly penalize deviation from the normal line
                if gate_cp_idx - 1 >= 0:
                    dp = P[gate_cp_idx - 1] - skeleton_path[i].pos
                    proj = cp.reshape(dp @ normal, (1,), order="C") * normal
                    cost += 100.0 * cp.sum_squares(dp - proj)
                if gate_cp_idx + 1 < n_ctrl:
                    dp = P[gate_cp_idx + 1] - skeleton_path[i].pos
                    proj = cp.reshape(dp @ normal, (1,), order="C") * normal
                    cost += 100.0 * cp.sum_squares(dp - proj)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            pass

        if P.value is None:
            print("Warning: SFC QP infeasible. Relaxing constraints.")
            return reference_points[:n_ctrl]

        return P.value

    def _calculate_anchors(self, current_pos: NDArray) -> list[SkeletonPoint]:
        gate_normals = R.from_quat(self.gates_quat).apply([1.0, 0.0, 0.0])
        raw_path = [SkeletonPoint(current_pos, False, None, None, None)]

        for i in range(self.target_gate_idx, len(self.gates_pos)):
            pos = self.gates_pos[i]
            normal = gate_normals[i].copy()
            rot = R.from_quat(self.gates_quat[i])
            right = rot.apply([0, 1, 0])
            up = rot.apply([0, 0, 1])

            pre_pos = pos - normal * self.anchor_gap
            post_pos = pos + normal * self.anchor_gap

            flow_dir = pos - raw_path[-1].pos

            # ENTRY SWING (U-turn approach logic)
            if np.dot(flow_dir, normal) < -0.1:
                if np.dot(raw_path[-1].pos - pos, right) > 0:
                    swing_pos = pos + right * 0.5
                else:
                    swing_pos = pos - right * 0.5
                raw_path.append(SkeletonPoint(swing_pos, False, None, None, None))

            if np.dot(pos - raw_path[-1].pos, normal) > 0.05:
                raw_path.append(SkeletonPoint(pre_pos, False, None, None, None))

            raw_path.append(SkeletonPoint(pos, True, normal, right, up))
            raw_path.append(SkeletonPoint(post_pos, False, None, None, None))

            # EXIT SWING (Hairpin / Reversal Logic)
            if i + 1 < len(self.gates_pos):
                next_pos = self.gates_pos[i + 1]
                exit_vector = next_pos - post_pos

                # If the next gate is behind us (requires a sharp turn > 90 degrees)
                if np.dot(exit_vector, normal) < -0.2:
                    clearance_pos = post_pos + normal * 1.0
                    raw_path.append(SkeletonPoint(clearance_pos, False, None, None, None))

                    if np.dot(exit_vector, right) > 0:
                        exit_swing = clearance_pos + right * 1.0 - normal * 0.7
                    else:
                        exit_swing = clearance_pos - right * 1.0 - normal * 0.7

                    raw_path.append(SkeletonPoint(exit_swing, False, None, None, None))

        obs_circles = []
        for p in self.obstacles_pos:
            obs_circles.append((p[:2], self.pole_radius + 0.15))
        for j, (p, q) in enumerate(zip(self.gates_pos, self.gates_quat)):
            rot = R.from_quat(q)
            right = rot.apply([0, 1, 0])
            bar_dist = 0.28
            obs_radius = 0.08 + 0.10
            obs_circles.append(((p - right * bar_dist)[:2], obs_radius))
            obs_circles.append(((p + right * bar_dist)[:2], obs_radius))

        path = raw_path
        for _ in range(3):
            new_path = [path[0]]
            for i in range(1, len(path)):
                prev_pt = new_path[-1].pos
                curr_pt = path[i].pos

                AB = curr_pt[:2] - prev_pt[:2]
                len_sq = np.dot(AB, AB)

                if len_sq > 1e-6:
                    first_t = 1.0
                    avoid_pt = None

                    for C, safe_radius in obs_circles:
                        t = max(0.0, min(1.0, np.dot(C - prev_pt[:2], AB) / len_sq))
                        projection = prev_pt[:2] + t * AB
                        dist = np.linalg.norm(projection - C)

                        if dist < safe_radius and t < first_t:
                            first_t = t
                            push_dir = (
                                (projection - C) / dist
                                if dist > 1e-6
                                else np.array([-AB[1], AB[0]]) / np.linalg.norm(AB)
                            )

                            avoidance_pt_2d = C + push_dir * (safe_radius + 0.20)
                            avoidance_z = prev_pt[2] + t * (curr_pt[2] - prev_pt[2])

                            proposed_pos = np.array(
                                [avoidance_pt_2d[0], avoidance_pt_2d[1], avoidance_z]
                            )

                            if (
                                np.linalg.norm(proposed_pos - prev_pt) > 0.3
                                and np.linalg.norm(proposed_pos - curr_pt) > 0.3
                            ):
                                avoid_pt = SkeletonPoint(proposed_pos, False, None, None, None)

                    if avoid_pt is not None:
                        new_path.append(avoid_pt)

                new_path.append(path[i])
            path = new_path

        return path

    def _check_environment_updates(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Check and handle environment updates (moved objects, crossed gates).

        Args:
            obs: Dictionary of observations from the environment.
        """
        pos, vel = obs["pos"], obs.get("vel", np.zeros(3))
        if self._prev_pos is None:
            self._prev_pos = pos.copy()

        self._check_gate_crossed(pos)
        objects_moved = self._check_objects_moved(obs)

        if objects_moved:
            if self.target_gate_idx < len(self.gates_pos):
                self.generate_spline(pos, vel)

        self._prev_pos = pos.copy()

    def _check_gate_crossed(self, current_pos: NDArray) -> bool:
        if self.target_gate_idx >= len(self.gates_pos):
            return False

        gate_pos = self.gates_pos[self.target_gate_idx]
        normal = R.from_quat(self.gates_quat[self.target_gate_idx]).apply([1, 0, 0])

        d_prev = np.dot(self._prev_pos - gate_pos, normal)
        d_curr = np.dot(current_pos - gate_pos, normal)

        if d_prev <= 0.0 < d_curr:
            intersection = self._prev_pos + (d_prev / (d_prev - d_curr)) * (
                current_pos - self._prev_pos
            )
            if np.linalg.norm(intersection - gate_pos) < (self.gate_outer / 2.0 + 0.40):
                self.target_gate_idx += 1
                return True
        return False

    def _check_objects_moved(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        moved = False
        new_gates_pos = obs["gates_pos"]
        if (
            len(self.gates_pos) > 0
            and np.max(np.linalg.norm(new_gates_pos - self.gates_pos, axis=1)) > 0.05
        ):
            self.gates_pos, self.gates_quat = new_gates_pos.copy(), obs["gates_quat"].copy()
            moved = True

        new_obs_pos = obs.get("obstacles_pos", np.array([]))
        if len(new_obs_pos) != len(self.obstacles_pos) or (
            len(new_obs_pos) > 0
            and np.max(np.linalg.norm(new_obs_pos - self.obstacles_pos, axis=1)) > 0.05
        ):
            self.obstacles_pos = new_obs_pos.copy()
            moved = True

        return moved

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute control outputs given current observations.

        Args:
            obs: Dictionary of observations from the environment.
            info: Optional dictionary of environment info.

        Returns:
            Array containing desired position, velocity, acceleration, yaw, and zeros.
        """
        self._check_environment_updates(obs)

        t = min(self._spline_tick / self._freq, self._t_total)
        if t >= self._t_total and self.target_gate_idx >= len(self.gates_pos):
            self._finished = True

        u = t / self._t_total if self._t_total > 0 else 0.0
        du_dt = 1.0 / self._t_total if self._t_total > 0 else 0.0

        des_pos = self._des_pos_spline(u)
        des_vel = self._des_pos_spline.derivative(nu=1)(u) * du_dt
        des_acc = self._des_pos_spline.derivative(nu=2)(u) * (du_dt**2)

        yaw = np.arctan2(des_vel[1], des_vel[0]) if np.linalg.norm(des_vel[:2]) > 0.1 else 0.0

        return np.concatenate((des_pos, des_vel, des_acc, [yaw], np.zeros(3)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Callback executed after each environment step.

        Args:
            action: Action taken in the environment.
            obs: Dictionary of observations.
            reward: Reward received.
            terminated: Whether episode terminated.
            truncated: Whether episode was truncated.
            info: Dictionary of environment info.

        Returns:
            Whether the controller has finished executing its plan.
        """
        self._tick += 1
        self._spline_tick += 1
        return self._finished

    def episode_callback(self) -> None:
        """Reset controller state at the start of a new episode."""
        self._tick, self._spline_tick, self._finished, self.target_gate_idx = 0, 0, False, 0
        self._prev_pos = None

    def render_callback(self, sim: Sim) -> None:
        """Render visualization of the trajectory and waypoints.

        Args:
            sim: The simulator instance to draw on.
        """
        if not hasattr(self, "_des_pos_spline"):
            return
        u = (
            min(self._spline_tick / self._freq, self._t_total) / self._t_total
            if self._t_total > 0
            else 0.0
        )
        draw_points(
            sim, self._des_pos_spline(u).reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.04
        )
        draw_line(sim, self._des_pos_spline(np.linspace(0.0, 1.0, 100)), rgba=(0.0, 1.0, 0.0, 1.0))
        if hasattr(self, "_control_points") and len(self._control_points) > 0:
            draw_points(sim, self._control_points, rgba=(0.0, 0.0, 1.0, 1.0), size=0.02)
        if hasattr(self, "skeleton_path") and len(self.skeleton_path) > 0:
            skel_pts = np.array([p.pos for p in self.skeleton_path])
            draw_line(sim, skel_pts, rgba=(1.0, 1.0, 0.0, 1.0))
