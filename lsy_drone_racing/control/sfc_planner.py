"""Pure SFC trajectory planner — extracted from sfc_controller.py.

Provides the path-skeleton builder, capsule obstacle model, flight-corridor
builder, and B-spline optimization. Consumed by both `sfc_controller.py`
(state-mode tracker) and `sfc_attitude_controller.py` (attitude-mode tracker).
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline, CubicSpline
from scipy.spatial.transform import Rotation as R

import lsy_drone_racing.control.sfc_config as cfg

logger = logging.getLogger(__name__)


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
    gate_idx: int | None = None


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


class SfcPlanner:
    """Pure SFC trajectory planner. Build once, update each tick, evaluate at any time."""

    W_VEL = 2.0
    W_ACC = 6.0
    W_JERK = 10.0
    W_CENTER = 0.01
    REPLAN_DEBOUNCE_TICKS = 5

    # --- TOPP (variable-speed schedule) tunables ---
    V_MAX_GLOBAL = cfg.V_MAX_GLOBAL
    TILT_LIMIT_PLANNER = cfg.TILT_LIMIT_PLANNER
    A_LONG_MAX_FACTOR = cfg.A_LONG_MAX_FACTOR
    V_FLOOR = cfg.V_FLOOR
    N_TOPP_SAMPLES = cfg.N_TOPP_SAMPLES

    def __init__(self, obs: dict[str, NDArray], freq: int) -> None:
        self._freq = freq
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
        self.obstacles_pos = obs.get("obstacles_pos", np.array([])).copy()
        self.target_gate_idx = 0
        self._tick = 0
        self._last_replan_tick = -self.REPLAN_DEBOUNCE_TICKS  # allow first move-triggered replan

        self._t_to_u: CubicSpline | None = None
        initial_vel = obs.get("vel", np.zeros(3))
        self._build_spline(obs["pos"], initial_vel)

    def update(self, obs: dict[str, NDArray]) -> bool:
        """Sync target_gate_idx from obs and replan if any object moved (debounced)."""
        self._tick += 1

        # 1. Sync gate counter — does NOT trigger replan
        env_target = int(obs.get("target_gate", self.target_gate_idx))
        if env_target == -1:
            self.target_gate_idx = len(self.gates_pos)
        else:
            self.target_gate_idx = env_target

        # 2. Detect movement
        moved = self._check_objects_moved(obs)
        if not moved:
            return False
        if self._tick - self._last_replan_tick < self.REPLAN_DEBOUNCE_TICKS:
            return False
        if self.target_gate_idx >= len(self.gates_pos):
            return False

        self._build_spline(obs["pos"], obs.get("vel", np.zeros(3)))
        self._last_replan_tick = self._tick
        return True

    def evaluate(self, t: float) -> tuple[NDArray, NDArray, NDArray]:
        """Return (pos, vel, acc) at *seconds* into the current spline.

        Uses the TOPP-computed t→u cubic map (built in _build_spline) to convert
        wall-clock time to spline parameter, then applies the full chain rule:
          vel = r'(u) · du/dt
          acc = r''(u) · (du/dt)² + r'(u) · d²u/dt²
        Returns SI units (m, m/s, m/s²). Falls back to the legacy uniform
        mapping when _t_to_u is None (TOPP failure path).
        """
        if self._t_total <= 0:
            cp_last = np.asarray(self._control_points[-1], dtype=np.float64)
            return cp_last, np.zeros(3), np.zeros(3)

        t_clamped = float(np.clip(t, 0.0, self._t_total))

        if self._t_to_u is not None:
            u = float(self._t_to_u(t_clamped))
            du_dt = float(self._t_to_u(t_clamped, 1))
            d2u_dt2 = float(self._t_to_u(t_clamped, 2))
        else:
            # TOPP fallback: uniform schedule.
            u = t_clamped / self._t_total
            du_dt = 1.0 / self._t_total
            d2u_dt2 = 0.0

        u = float(np.clip(u, 0.0, 1.0))

        r1 = np.asarray(self._des_pos_spline.derivative(nu=1)(u), dtype=np.float64)
        r2 = np.asarray(self._des_pos_spline.derivative(nu=2)(u), dtype=np.float64)
        pos = np.asarray(self._des_pos_spline(u), dtype=np.float64)
        vel = r1 * du_dt
        acc = r2 * (du_dt**2) + r1 * d2u_dt2
        return pos, vel, acc

    @property
    def t_total(self) -> float:
        return self._t_total

    @property
    def des_pos_spline(self) -> BSpline:
        return self._des_pos_spline

    @property
    def control_points(self) -> NDArray:
        return self._control_points

    def episode_reset(self) -> None:
        """Reset internal counters. Spline retained — callers should rebuild via update()."""
        self.target_gate_idx = 0
        self._tick = 0
        self._last_replan_tick = -self.REPLAN_DEBOUNCE_TICKS

    def _build_spline(self, current_pos: NDArray, current_vel: NDArray) -> None:
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

        v_start = float(np.linalg.norm(current_vel))
        try:
            self._t_to_u, self._t_total = self._compute_time_schedule(
                self._des_pos_spline, v_start
            )
        except Exception as exc:  # noqa: BLE001 — fallback path
            logger.warning("TOPP scheduling failed (%s); falling back to uniform schedule.", exc)
            self._t_to_u = None
            self._t_total = float(np.sum(cp_dists) / self.base_speed)

    def _compute_time_schedule(
        self, spline: BSpline, v_start: float
    ) -> tuple[CubicSpline, float]:
        """TOPP-style time parameterization: build a t→u cubic spline.

        Given a fixed-geometry BSpline over u ∈ [0, 1] and the drone's current
        speed, compute v(u) respecting:
          - Lateral accel:  v² · κ(u) ≤ a_lat_max
          - Longitudinal accel:  |dv/dt| ≤ a_long_max  (forward + backward sweeps)
          - Global cap:  v ≤ V_MAX_GLOBAL
          - Floor:  v ≥ V_FLOOR

        Returns:
            t_to_u: CubicSpline mapping wall-clock time → spline parameter u.
            t_total: Total schedule duration in seconds.
        """
        g = 9.81
        a_lat_max = g * np.tan(self.TILT_LIMIT_PLANNER)
        a_long_max = self.A_LONG_MAX_FACTOR * a_lat_max
        eps = 1e-6

        N = self.N_TOPP_SAMPLES
        u_k = np.linspace(0.0, 1.0, N)

        # --- Step 1: sample geometry at each u_k ---
        d1 = spline.derivative(nu=1)(u_k)        # shape (N, 3)
        d2 = spline.derivative(nu=2)(u_k)        # shape (N, 3)
        ds_du = np.linalg.norm(d1, axis=1)       # shape (N,)
        cross = np.cross(d1, d2)                 # shape (N, 3)
        kappa = np.linalg.norm(cross, axis=1) / np.maximum(ds_du**3, eps)

        # --- Step 2: lateral-accel envelope + global cap ---
        v_curve = np.sqrt(a_lat_max / np.maximum(kappa, eps))
        v = np.minimum(v_curve, self.V_MAX_GLOBAL)

        # --- Step 3: forward sweep (longitudinal accel from v_start) ---
        v[0] = min(v[0], v_start)
        for k in range(1, N):
            ds = 0.5 * (ds_du[k] + ds_du[k - 1]) * (u_k[k] - u_k[k - 1])
            v_max_fwd = np.sqrt(v[k - 1] ** 2 + 2.0 * a_long_max * ds)
            v[k] = min(v[k], v_max_fwd)

        # --- Step 4: backward sweep (must brake in time for upcoming curves) ---
        for k in range(N - 2, -1, -1):
            ds = 0.5 * (ds_du[k + 1] + ds_du[k]) * (u_k[k + 1] - u_k[k])
            v_max_bwd = np.sqrt(v[k + 1] ** 2 + 2.0 * a_long_max * ds)
            v[k] = min(v[k], v_max_bwd)

        # --- Step 5: floor ---
        v = np.maximum(v, self.V_FLOOR)

        # --- Step 6: integrate t(u) = ∫ ds / v  via trapezoid on (1/v_avg)·ds ---
        t = np.zeros(N)
        for k in range(1, N):
            ds = 0.5 * (ds_du[k] + ds_du[k - 1]) * (u_k[k] - u_k[k - 1])
            v_avg = 0.5 * (v[k] + v[k - 1])
            t[k] = t[k - 1] + ds / max(v_avg, self.V_FLOOR)

        t_to_u = CubicSpline(t, u_k)  # not-a-knot (default) is more appropriate than "natural"
        return t_to_u, float(t[-1])

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
        for gate_i, (pos, quat) in enumerate(zip(self.gates_pos, self.gates_quat)):
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
                        gate_i,
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
                    gate_i,
                )
            )
            capsules.append(
                Capsule(
                    pos - up * bar_dist - right * 0.36,
                    pos - up * bar_dist + right * 0.36,
                    bar_radius,
                    True,
                    gate_i,
                )
            )
            capsules.append(
                Capsule(
                    pos - right * bar_dist + up * 0.36,
                    pos - right * bar_dist - up * 0.36,
                    bar_radius,
                    True,
                    gate_i,
                )
            )
            capsules.append(
                Capsule(
                    pos + right * bar_dist + up * 0.36,
                    pos + right * bar_dist - up * 0.36,
                    bar_radius,
                    True,
                    gate_i,
                )
            )

        return capsules

    def _generate_flight_corridors(
        self, skeleton_path: list[SkeletonPoint], capsules: list[Capsule]
    ) -> list[FlightCorridor]:
        """Constructs a convex polyhedron for each segment via separating planes."""
        # Precompute per-gate normal in world frame; used to scope the gate-skip rule.
        gate_normals = R.from_quat(self.gates_quat).apply([1.0, 0.0, 0.0])

        corridors = []
        for i in range(len(skeleton_path) - 1):
            pt1 = skeleton_path[i]
            pt2 = skeleton_path[i + 1]
            corr = FlightCorridor(pt1.pos, pt2.pos)

            # Add separating half-spaces for all capsules
            for cap in capsules:
                # Skip a gate's frame capsules only for segments that pass
                # through the gate's opening. The segment must (a) cross the
                # gate's normal plane and (b) cross it within the frame's
                # radius - otherwise it's just going around the gate, and
                # the corridor must keep the frame as a real obstacle.
                if cap.is_gate and cap.gate_idx is not None:
                    g_pos = self.gates_pos[cap.gate_idx]
                    g_normal = gate_normals[cap.gate_idx]
                    d1 = float(np.dot(pt1.pos - g_pos, g_normal))
                    d2 = float(np.dot(pt2.pos - g_pos, g_normal))
                    # Frame outer radius is 0.36m; allow a small slack so the
                    # pre/post anchors (which sit on the normal axis) still
                    # qualify as a through-segment.
                    near_radius = self.gate_outer / 2.0 + 0.10
                    if d1 * d2 < 0.0:
                        t = -d1 / (d2 - d1)
                        crossing = pt1.pos + t * (pt2.pos - pt1.pos)
                        if np.linalg.norm(crossing - g_pos) < near_radius:
                            continue
                    elif d1 == 0.0 and np.linalg.norm(pt1.pos - g_pos) < near_radius:
                        continue
                    elif d2 == 0.0 and np.linalg.norm(pt2.pos - g_pos) < near_radius:
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

        # Preserve the just-passed gate's clearance anchor when a replan fires
        # mid-exit. Without this, the new skeleton goes straight from the
        # drone (still inside the previous gate's exit zone) to the next
        # gate's pre_pos, ignoring the forward-along-normal commitment the
        # drone is currently flying out on. The drone ends up flying with
        # momentum along the old route while the spline pulls it onto a path
        # that demands an instantaneous direction change. Re-emitting only
        # the clearance anchor (along the prev gate's normal) keeps the
        # tangent aligned with the drone's current heading without forcing
        # the perpendicular exit_swing detour, which over-commits when the
        # drone has already started its turn.
        prev_gate_idx = self.target_gate_idx - 1
        if 0 <= prev_gate_idx < len(self.gates_pos) and self.target_gate_idx < len(
            self.gates_pos
        ):
            prev_pos = self.gates_pos[prev_gate_idx]
            prev_normal = gate_normals[prev_gate_idx]
            d_post = float(np.dot(current_pos - prev_pos, prev_normal))
            if 0.0 < d_post < 1.0:
                next_pos = self.gates_pos[self.target_gate_idx]
                prev_post_pos = prev_pos + prev_normal * self.anchor_gap
                exit_vector = next_pos - prev_post_pos
                if float(np.dot(exit_vector, prev_normal)) < -0.2:
                    clearance_pos = prev_post_pos + prev_normal * 1.0
                    raw_path.append(SkeletonPoint(clearance_pos, False, None, None, None))

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
            # If approaching the gate from the wrong side, we need a U-turn maneuver.
            if np.dot(flow_dir, normal) < -0.1:
                is_next_to = False
                faces_different = False
                if i > 0:
                    prev_pos = self.gates_pos[i - 1]
                    prev_normal = gate_normals[i - 1]
                    dp = pos - prev_pos
                    
                    # Calculate lateral (side-to-side) vs longitudinal (front-to-back) offset
                    # relative to the CURRENT gate's orientation.
                    dist_lat = abs(float(np.dot(dp, right)))
                    dist_long = abs(float(np.dot(dp, normal)))
                    
                    # If the lateral offset is larger, the gates are placed side-by-side.
                    # If the longitudinal offset is larger, they are placed in-line (back-to-back).
                    is_next_to = dist_lat > dist_long
                    
                    # Check if the previous gate and current gate face opposite directions.
                    faces_different = np.dot(normal, prev_normal) < 0.0

                # Determine the type of U-turn needed:
                # Vertical/Drop-down U-turn is used when gates are in-line (back-to-back)
                # and facing opposite directions.
                if faces_different and not is_next_to:
                    # Vertical U-turn: Drop down into the gate from above
                    swing_pos = pre_pos + up * 0.8
                else:
                    # Lateral U-turn: For side-by-side gates or gates facing the same general direction.
                    # Choose left or right approach based on the drone's incoming position.
                    dot_r = np.dot(raw_path[-1].pos - pos, right)
                    lat_dir = right if dot_r > 0 else -right
                    swing_pos = pos + lat_dir * 0.5
                raw_path.append(SkeletonPoint(swing_pos, False, None, None, None))

            if np.dot(pos - raw_path[-1].pos, normal) > 0.05:
                raw_path.append(SkeletonPoint(pre_pos, False, None, None, None))

            raw_path.append(SkeletonPoint(pos, True, normal, right, up))
            raw_path.append(SkeletonPoint(post_pos, False, None, None, None))

            # EXIT SWING (Hairpin / Reversal Logic)
            if i + 1 < len(self.gates_pos):
                next_pos = self.gates_pos[i + 1]
                next_normal = gate_normals[i + 1]
                exit_vector = next_pos - post_pos

                # If the next gate is behind us (requires a sharp turn > 90 degrees)
                if np.dot(exit_vector, normal) < -0.2:
                    clearance_pos = post_pos + normal * 1.0
                    raw_path.append(SkeletonPoint(clearance_pos, False, None, None, None))

                    # Calculate lateral vs longitudinal offset to the NEXT gate
                    # relative to the CURRENT gate's orientation.
                    dp = next_pos - pos
                    dist_lat = abs(float(np.dot(dp, right)))
                    dist_long = abs(float(np.dot(dp, normal)))
                    
                    # If the lateral offset is larger, the next gate is placed side-by-side.
                    # If the longitudinal offset is larger, it is in-line (back-to-back).
                    is_next_to = dist_lat > dist_long
                    
                    # Check if the current gate and the next gate face opposite directions.
                    faces_different = np.dot(normal, next_normal) < 0.0

                    # Determine the type of U-turn needed to exit the current gate
                    # and prepare for the next gate.
                    if faces_different and not is_next_to:
                        # Over-the-top U-turn: Used for in-line (back-to-back) gates facing opposite ways.
                        # The drone reverses direction by flying up and directly over the current gate.
                        exit_swing = pos + up * 1.2
                    else:
                        # Lateral U-turn: Used for side-by-side gates.
                        # The drone swings out laterally to the left or right to re-orient.
                        dot_r = np.dot(exit_vector, right)
                        lat_dir = right if dot_r > 0 else -right
                        exit_swing = clearance_pos + lat_dir * 1.0 - normal * 0.7

                    raw_path.append(SkeletonPoint(exit_swing, False, None, None, None))
                else:
                    # Smarter intermediate waypoint handling for non-U-turn setups
                    next_pre_pos = next_pos - next_normal * self.anchor_gap
                    dp = (next_pre_pos - post_pos)[:2]
                    d1 = normal[:2]
                    d2 = -next_normal[:2]
                    
                    det = d1[0] * d2[1] - d1[1] * d2[0]
                    if abs(det) > 1e-3:
                        t1 = (dp[0] * d2[1] - dp[1] * d2[0]) / det
                        t2 = (d1[0] * dp[1] - d1[1] * dp[0]) / det
                        
                        if t1 > 0.2 and t2 > 0.2:
                            intersect1 = post_pos + normal * t1
                            intersect2 = next_pre_pos - next_normal * t2
                            midpoint = (intersect1 + intersect2) / 2.0
                            
                            max_dist = np.linalg.norm(next_pre_pos - post_pos)
                            if np.linalg.norm(midpoint - post_pos) < max_dist * 1.5:
                                raw_path.append(SkeletonPoint(midpoint, False, None, None, None))

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

    def _check_objects_moved(self, obs: dict[str, NDArray]) -> bool:
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
