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


class SfcCorridorPlanner:
    """Pure SFC trajectory planner. Build once, update each tick, evaluate at any time."""

    W_VEL = 2.0
    W_ACC = 6.0
    W_JERK = 10.0
    W_CENTER = 0.01
    W_GATE_ALIGN = 30.0  # Soft cost weight on P[i±1] lateral offset from gate-normal axis.
    GATE_TUBE_RADIUS = 0.18  # m. Inscribed lateral fence on P[i±1] from gate-normal axis.
    GATE_TUBE_HALF_LENGTH = 0.5  # m. Axial fence on P[i±1] from gate centre (max).
    GATE_TUBE_AXIAL_MIN = (
        0.0  # m. Min axial distance, signed by side. 0 = no min (just sign convention).
    )
    GATE_TUBE_N_FACETS = 8  # Polyhedral facets approximating the lateral cylinder.
    REPLAN_DEBOUNCE_TICKS = 5

    # --- TOPP (variable-speed schedule) tunables ---
    V_MAX_GLOBAL = 1.5  # m/s. Speed ceiling on straights.
    TILT_LIMIT_PLANNER = 0.5  # rad. Mirrors controller TILT_LIMIT. Drives a_lat_max.
    A_LONG_MAX_FACTOR = (
        0.53  # a_long_max = factor * a_lat_max. Vertical thrust eats some accel budget.
    )
    V_FLOOR = (
        0.2  # m/s. Floor on scheduled speed (avoid divide-by-near-zero in pathological curvature).
    )
    N_TOPP_SAMPLES = 200  # Number of points to sample u ∈ [0, 1] when building the schedule.

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
        self.replan_events: list[dict] = []
        self.last_replan_event: dict | None = None
        self._traj_history = []
        self._max_history_len = 200
        self._current_t = 0.0
        initial_vel = obs.get("vel", np.zeros(3))
        self._build_spline(obs["pos"], initial_vel)
        self._record_replan_event(reason="init")

    def update(self, obs: dict[str, NDArray]) -> bool:
        """Sync target_gate_idx from obs and replan if any object moved or gate passed."""
        self._tick += 1

        # 1. Sync gate counter — triggers replan if we only look ahead a few gates
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

        # Update current projection on spline
        self._current_t = self._find_closest_t(obs["pos"])

        # 2. Detect movement or gate change
        moved, reason = self._check_objects_moved(obs)
        if gate_changed:
            moved = True
            reason = "gate_passed"

        if not moved:
            return False
        if self._tick - self._last_replan_tick < self.REPLAN_DEBOUNCE_TICKS:
            return False
        if self.target_gate_idx >= len(self.gates_pos) and not gate_changed:
            return False

        self._build_spline(obs["pos"], obs.get("vel", np.zeros(3)))
        self._last_replan_tick = self._tick
        self._record_replan_event(reason=reason)
        return True

    def evaluate(self, t_offset: float) -> tuple[NDArray, NDArray, NDArray]:
        t_eval = self._current_t + t_offset
        return self._evaluate_absolute(t_eval)

    def evaluate_corridor(self, t_offset: float) -> tuple[NDArray, NDArray] | None:
        """Returns the (A, b) matrices for the flight corridor at time t_offset.
        
        Args:
            t_offset: Time in seconds into the future.
            
        Returns:
            Tuple (A, b) representing A*x <= b, or None if no corridor is active.
        """
        if not hasattr(self, "corridors") or self.corridors is None or len(self.corridors) == 0:
            return None

        t_eval = self._current_t + t_offset
        if self._t_total <= 0:
            u = 1.0
        else:
            t_clamped = float(np.clip(t_eval, 0.0, self._t_total))
            if self._t_to_u is not None:
                u = float(self._t_to_u(t_clamped))
            else:
                u = t_clamped / self._t_total
        u = float(np.clip(u, 0.0, 1.0))

        # Determine segment index. u spans [0, 1].
        # There are n_segments = len(corridors).
        # We need to map u back to the segment index.
        # The B-spline knots define the parameterization.
        # A simple approximation: u roughly maps linearly to segments.
        # A better approach: use the control point parametrization mapping from optimize_control_points.
        # For now, approximate linearly over segments since control points are evenly spaced.
        n_segments = len(self.corridors)
        seg_idx = int(np.floor(u * n_segments))
        seg_idx = min(seg_idx, n_segments - 1)
        
        A = np.array(self.corridors[seg_idx].A)
        b = np.array(self.corridors[seg_idx].b)
        return A, b

    def _find_closest_t(self, pos: NDArray) -> float:
        if not hasattr(self, "_des_pos_spline") or self._des_pos_spline is None:
            return 0.0
        u_samples = np.linspace(0, 1, 100)
        pts = self._des_pos_spline(u_samples)
        dists = np.linalg.norm(pts - pos, axis=1)
        u_closest = float(u_samples[np.argmin(dists)])

        if self._t_to_u is not None:
            roots = self._t_to_u.solve(y=u_closest, extrapolate=False)
            if len(roots) > 0:
                # Find the root that is closest to our expected range (usually only one root anyway)
                return float(roots[0])
            return 0.0
        else:
            return u_closest * self._t_total

    def _evaluate_absolute(self, t: float) -> tuple[NDArray, NDArray, NDArray]:
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
        self.replan_events = []
        self.last_replan_event = None
        self._current_t = 0.0
        self._traj_history = []

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
        self.capsules = capsules
        corridors = self._generate_flight_corridors(skeleton_path, capsules)
        self.corridors = corridors

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
            self._t_to_u, self._t_total = self._compute_time_schedule(self._des_pos_spline, v_start)
        except Exception as exc:  # noqa: BLE001 — fallback path
            logger.warning("TOPP scheduling failed (%s); falling back to uniform schedule.", exc)
            self._t_to_u = None
            self._t_total = float(np.sum(cp_dists) / self.base_speed)

    def _compute_time_schedule(self, spline: BSpline, v_start: float) -> tuple[CubicSpline, float]:
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
        d1 = spline.derivative(nu=1)(u_k)  # shape (N, 3)
        d2 = spline.derivative(nu=2)(u_k)  # shape (N, 3)
        ds_du = np.linalg.norm(d1, axis=1)  # shape (N,)
        cross = np.cross(d1, d2)  # shape (N, 3)
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


    def _init_casadi_planner(self) -> None:
        """Initializes the fixed-size parametric CasADi optimizer."""
        import casadi as ca
        self.MAX_CTRL = 80
        self.MAX_PLANES = 25

        self.opti = ca.Opti()
        self.P_ca = self.opti.variable(self.MAX_CTRL, 3)

        self.mask_ca = self.opti.parameter(self.MAX_CTRL)
        self.ref_pts_ca = self.opti.parameter(self.MAX_CTRL, 3)
        
        self.A_corr_ca = self.opti.parameter(self.MAX_CTRL * self.MAX_PLANES, 3)
        self.b_corr_ca = self.opti.parameter(self.MAX_CTRL * self.MAX_PLANES)
        
        self.is_gate_ca = self.opti.parameter(self.MAX_CTRL)
        self.gate_pos_ca = self.opti.parameter(self.MAX_CTRL, 3)
        
        self.tube_mask_ca = self.opti.parameter(self.MAX_CTRL)
        self.tube_gate_pos_ca = self.opti.parameter(self.MAX_CTRL, 3)
        self.tube_normal_ca = self.opti.parameter(self.MAX_CTRL, 3)
        self.tube_sign_ca = self.opti.parameter(self.MAX_CTRL)
        self.tube_facets_ca = self.opti.parameter(self.MAX_CTRL * 8, 3)
        
        self.align_mask_ca = self.opti.parameter(self.MAX_CTRL)
        
        self.P0_ref_ca = self.opti.parameter(3)
        self.P1_ref_ca = self.opti.parameter(3)
        self.P1_weight_ca = self.opti.parameter(1)
        
        self.end_mask_ca = self.opti.parameter(self.MAX_CTRL)
        self.end_pos_ca = self.opti.parameter(3)

        cost = 1e-6 * ca.sumsqr(self.P_ca)

        for i in range(self.MAX_CTRL - 1):
            diff = self.P_ca[i+1, :] - self.P_ca[i, :]
            cost += self.W_VEL * self.mask_ca[i] * self.mask_ca[i+1] * ca.sumsqr(diff)
            
        for i in range(self.MAX_CTRL - 2):
            diff = self.P_ca[i+2, :] - 2*self.P_ca[i+1, :] + self.P_ca[i, :]
            cost += self.W_ACC * self.mask_ca[i] * self.mask_ca[i+1] * self.mask_ca[i+2] * ca.sumsqr(diff)

        for i in range(self.MAX_CTRL - 3):
            diff = self.P_ca[i+3, :] - 3*self.P_ca[i+2, :] + 3*self.P_ca[i+1, :] - self.P_ca[i, :]
            cost += self.W_JERK * self.mask_ca[i] * self.mask_ca[i+1] * self.mask_ca[i+2] * self.mask_ca[i+3] * ca.sumsqr(diff)

        for i in range(self.MAX_CTRL):
            cost += self.W_CENTER * self.mask_ca[i] * ca.sumsqr(self.P_ca[i, :] - self.ref_pts_ca[i, :])
            cost += 1e5 * self.is_gate_ca[i] * ca.sumsqr(self.P_ca[i, :].T - self.gate_pos_ca[i, :].T)
            cost += 1e5 * self.end_mask_ca[i] * ca.sumsqr(self.P_ca[i, :].T - self.end_pos_ca)
            
        cost += 10.0 * ca.sumsqr(self.P_ca[0, :].T - self.P0_ref_ca)
        cost += self.P1_weight_ca * ca.sumsqr(self.P_ca[1, :].T - self.P1_ref_ca)

        for i in range(self.MAX_CTRL):
            dp = self.P_ca[i, :] - self.tube_gate_pos_ca[i, :]
            normal = self.tube_normal_ca[i, :]
            proj = ca.dot(dp, normal) * normal
            cost += self.W_GATE_ALIGN * self.align_mask_ca[i] * ca.sumsqr(dp - proj)

        self.opti.minimize(cost)

        for i in range(self.MAX_CTRL):
            A_i = self.A_corr_ca[i*self.MAX_PLANES : (i+1)*self.MAX_PLANES, :]
            b_i = self.b_corr_ca[i*self.MAX_PLANES : (i+1)*self.MAX_PLANES]
            self.opti.subject_to( ca.mtimes(A_i, self.P_ca[i, :].T) <= b_i )

            dp = self.P_ca[i, :] - self.tube_gate_pos_ca[i, :]
            for f in range(8):
                facet = self.tube_facets_ca[i*8 + f, :]
                val = self.tube_mask_ca[i] * ca.dot(dp, facet)
                bound = self.tube_mask_ca[i] * self.GATE_TUBE_RADIUS + (1 - self.tube_mask_ca[i]) * 1000.0
                self.opti.subject_to( val <= bound )
                
            proj_n = self.tube_mask_ca[i] * self.tube_sign_ca[i] * ca.dot(dp, self.tube_normal_ca[i, :])
            min_bound = self.tube_mask_ca[i] * self.GATE_TUBE_AXIAL_MIN - (1 - self.tube_mask_ca[i]) * 1000.0
            max_bound = self.tube_mask_ca[i] * self.GATE_TUBE_HALF_LENGTH + (1 - self.tube_mask_ca[i]) * 1000.0
            self.opti.subject_to( self.opti.bounded(min_bound, proj_n, max_bound) )

        p_opts = {"expand": True}
        s_opts = {"max_iter": 100, "print_level": 0, "tol": 1e-4, "acceptable_tol": 1e-3, "sb": "yes"}
        self.opti.solver('ipopt', p_opts, s_opts)
        self._casadi_initialized = True
        self._last_P = None

    def _optimize_control_points(
        self,
        skeleton_path: list[SkeletonPoint],
        corridors: list[FlightCorridor],
        current_vel: NDArray,
    ) -> NDArray:
        """Solves a fixed-size parametric CasADi QP to find optimal control points."""
        if getattr(self, "_casadi_initialized", False) is False:
            self._init_casadi_planner()

        n_segments = len(corridors)
        pts_per_seg = self.points_per_segment

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

        if n_ctrl < 4:
            pts_first_seg = 4 - (n_segments - 1) * pts_rest_seg
            n_ctrl = pts_first_seg + (n_segments - 1) * pts_rest_seg

        n_ctrl = min(n_ctrl, self.MAX_CTRL)

        v_mask = np.zeros(self.MAX_CTRL)
        v_ref = np.zeros((self.MAX_CTRL, 3))
        v_A = np.zeros((self.MAX_CTRL * self.MAX_PLANES, 3))
        v_b = np.ones(self.MAX_CTRL * self.MAX_PLANES) * 1000.0
        v_is_gate = np.zeros(self.MAX_CTRL)
        v_gate_pos = np.zeros((self.MAX_CTRL, 3))
        v_tube_mask = np.zeros(self.MAX_CTRL)
        v_tube_gate = np.zeros((self.MAX_CTRL, 3))
        v_tube_norm = np.zeros((self.MAX_CTRL, 3))
        v_tube_sign = np.zeros(self.MAX_CTRL)
        v_tube_facets = np.zeros((self.MAX_CTRL * 8, 3))
        v_align_mask = np.zeros(self.MAX_CTRL)
        v_end_mask = np.zeros(self.MAX_CTRL)

        idx = 0
        for seg_idx, corr in enumerate(corridors):
            n_pts = pts_first_seg if seg_idx == 0 else pts_rest_seg
            A = np.array(corr.A)
            b = np.array(corr.b)
            n_planes = min(len(b), self.MAX_PLANES)
            
            for j in range(n_pts):
                if idx >= self.MAX_CTRL:
                    break
                v_mask[idx] = 1.0
                pt = corr.p1 + (j / n_pts) * (corr.p2 - corr.p1)
                v_ref[idx] = pt
                
                v_A[idx * self.MAX_PLANES : idx * self.MAX_PLANES + n_planes] = A[:n_planes]
                v_b[idx * self.MAX_PLANES : idx * self.MAX_PLANES + n_planes] = b[:n_planes]
                idx += 1

        if idx == 0:
            return np.array([self._current_pos_for_spline]*4)
            
        n_ctrl = idx 

        cp_idx_map = [0]
        curr_idx = pts_first_seg
        for seg_idx in range(1, n_segments):
            cp_idx_map.append(curr_idx)
            curr_idx += pts_rest_seg
        cp_idx_map.append(n_ctrl - 1)

        for i in range(1, len(skeleton_path) - 1):
            if skeleton_path[i].is_gate:
                gate_cp_idx = cp_idx_map[i]
                if gate_cp_idx >= self.MAX_CTRL:
                    continue
                    
                gate_pos = skeleton_path[i].pos
                normal = skeleton_path[i].gate_normal
                right = skeleton_path[i].gate_right
                up = skeleton_path[i].gate_up

                v_is_gate[gate_cp_idx] = 1.0
                v_gate_pos[gate_cp_idx] = gate_pos

                facet_dirs = []
                for k in range(self.GATE_TUBE_N_FACETS):
                    theta = 2.0 * np.pi * k / self.GATE_TUBE_N_FACETS
                    facet_dirs.append(np.cos(theta) * right + np.sin(theta) * up)
                facet_dirs = np.array(facet_dirs)

                if gate_cp_idx - 1 >= 0:
                    v_ref[gate_cp_idx - 1] = gate_pos - normal * self.anchor_gap
                    v_tube_mask[gate_cp_idx - 1] = 1.0
                    v_tube_gate[gate_cp_idx - 1] = gate_pos
                    v_tube_norm[gate_cp_idx - 1] = normal
                    v_tube_sign[gate_cp_idx - 1] = -1.0
                    v_tube_facets[(gate_cp_idx - 1)*8 : gate_cp_idx*8] = facet_dirs
                    v_align_mask[gate_cp_idx - 1] = 1.0

                if gate_cp_idx + 1 < n_ctrl:
                    v_ref[gate_cp_idx + 1] = gate_pos + normal * self.anchor_gap
                    v_tube_mask[gate_cp_idx + 1] = 1.0
                    v_tube_gate[gate_cp_idx + 1] = gate_pos
                    v_tube_norm[gate_cp_idx + 1] = normal
                    v_tube_sign[gate_cp_idx + 1] = 1.0
                    v_tube_facets[(gate_cp_idx + 1)*8 : (gate_cp_idx + 2)*8] = facet_dirs
                    v_align_mask[gate_cp_idx + 1] = 1.0

        v_end_mask[n_ctrl - 1] = 1.0
        v_end_pos = skeleton_path[-1].pos

        v_P0_ref = self._current_pos_for_spline
        speed = np.linalg.norm(current_vel)
        if speed > 0.1:
            v_P1_ref = v_P0_ref + current_vel * 0.05
            v_P1_weight = 50.0
        else:
            v_P1_ref = v_P0_ref
            v_P1_weight = 10.0

        self.opti.set_value(self.mask_ca, v_mask)
        self.opti.set_value(self.ref_pts_ca, v_ref)
        self.opti.set_value(self.A_corr_ca, v_A)
        self.opti.set_value(self.b_corr_ca, v_b)
        self.opti.set_value(self.is_gate_ca, v_is_gate)
        self.opti.set_value(self.gate_pos_ca, v_gate_pos)
        self.opti.set_value(self.tube_mask_ca, v_tube_mask)
        self.opti.set_value(self.tube_gate_pos_ca, v_tube_gate)
        self.opti.set_value(self.tube_normal_ca, v_tube_norm)
        self.opti.set_value(self.tube_sign_ca, v_tube_sign)
        self.opti.set_value(self.tube_facets_ca, v_tube_facets)
        self.opti.set_value(self.align_mask_ca, v_align_mask)
        self.opti.set_value(self.end_mask_ca, v_end_mask)
        self.opti.set_value(self.end_pos_ca, v_end_pos)
        self.opti.set_value(self.P0_ref_ca, v_P0_ref)
        self.opti.set_value(self.P1_ref_ca, v_P1_ref)
        self.opti.set_value(self.P1_weight_ca, v_P1_weight)

        if self._last_P is not None:
            self.opti.set_initial(self.P_ca, self._last_P)
        else:
            self.opti.set_initial(self.P_ca, v_ref)

        try:
            sol = self.opti.solve()
            P_opt = sol.value(self.P_ca)
            self._last_P = P_opt
            return P_opt[:n_ctrl]
        except Exception as e:
            logger.warning(f"SFC CasADi QP failed: {e}. Relaxing constraints.")
            return v_ref[:n_ctrl]
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
        if 0 <= prev_gate_idx < len(self.gates_pos) and self.target_gate_idx < len(self.gates_pos):
            prev_pos = self.gates_pos[prev_gate_idx]
            prev_normal = gate_normals[prev_gate_idx]
            d_post = float(np.dot(current_pos - prev_pos, prev_normal))
            if 0.0 < d_post < 1.0:
                next_pos = self.gates_pos[self.target_gate_idx]
                # Test against prev_pos + prev_normal * anchor_gap, the canonical
                # post-gate exit point (no post_pos anchor in skeleton anymore).
                exit_vector = next_pos - (prev_pos + prev_normal * self.anchor_gap)
                if float(np.dot(exit_vector, prev_normal)) < -0.2:
                    clearance_pos = prev_pos + prev_normal * (self.anchor_gap + 1.0)
                    raw_path.append(SkeletonPoint(clearance_pos, False, None, None, None))

        for i in range(self.target_gate_idx, len(self.gates_pos)):
            pos = self.gates_pos[i]
            normal = gate_normals[i].copy()
            rot = R.from_quat(self.gates_quat[i])
            right = rot.apply([0, 1, 0])
            up = rot.apply([0, 0, 1])

            flow_dir = pos - raw_path[-1].pos

            # ENTRY SWING (U-turn approach logic). Computed against gate centre
            # rather than a pre_pos anchor — same dot-product test, ~0.5 m
            # offset along normal does not flip the U-turn detection.
            if np.dot(flow_dir, normal) < -0.1:
                if np.dot(raw_path[-1].pos - pos, right) > 0:
                    swing_pos = pos + right * 0.5
                else:
                    swing_pos = pos - right * 0.5
                raw_path.append(SkeletonPoint(swing_pos, False, None, None, None))

            raw_path.append(SkeletonPoint(pos, True, normal, right, up))

            # EXIT SWING (Hairpin / Reversal Logic)
            if i + 1 < len(self.gates_pos):
                next_pos = self.gates_pos[i + 1]
                # Test against pos + normal * anchor_gap, the canonical post-gate
                # exit point, even though no post_pos anchor is in the skeleton.
                exit_vector = next_pos - (pos + normal * self.anchor_gap)

                if np.dot(exit_vector, normal) < -0.2:
                    clearance_pos = pos + normal * (self.anchor_gap + 1.0)
                    raw_path.append(SkeletonPoint(clearance_pos, False, None, None, None))

                    if np.dot(exit_vector, right) > 0:
                        exit_swing = clearance_pos + right * 1.0 - normal * 0.7
                    else:
                        exit_swing = clearance_pos - right * 1.0 - normal * 0.7

                    raw_path.append(SkeletonPoint(exit_swing, False, None, None, None))

        # Add an additional waypoint after the final gate to maintain speed through the finish line
        if len(self.gates_pos) > 0 and self.target_gate_idx <= len(self.gates_pos):
            last_gate_idx = len(self.gates_pos) - 1
            last_pos = self.gates_pos[last_gate_idx]
            last_normal = gate_normals[last_gate_idx]
            finish_pos = last_pos + last_normal * 0.75
            raw_path.append(SkeletonPoint(finish_pos, False, None, None, None))

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

    def _check_objects_moved(self, obs: dict[str, NDArray]) -> tuple[bool, str]:
        gate_moved = False
        obs_moved = False
        new_gates_pos = obs["gates_pos"]
        if (
            len(self.gates_pos) > 0
            and np.max(np.linalg.norm(new_gates_pos - self.gates_pos, axis=1)) > 0.05
        ):
            self.gates_pos, self.gates_quat = new_gates_pos.copy(), obs["gates_quat"].copy()
            gate_moved = True

        new_obs_pos = obs.get("obstacles_pos", np.array([]))
        if len(new_obs_pos) != len(self.obstacles_pos) or (
            len(new_obs_pos) > 0
            and np.max(np.linalg.norm(new_obs_pos - self.obstacles_pos, axis=1)) > 0.05
        ):
            self.obstacles_pos = new_obs_pos.copy()
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
        """Return a dict fully describing the current spline (for trace dumps)."""
        return {
            "t_total": float(self._t_total),
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

    @property
    def current_pos_ref(self) -> NDArray:
        if hasattr(self, "_t_total") and self._t_total > 0:
            pos_ref, _, _ = self.evaluate(0.0)
            return pos_ref
        return np.array([0.0, 0.0, 0.0])

    @property
    def current_vel_ref(self) -> NDArray:
        if hasattr(self, "_t_total") and self._t_total > 0:
            _, vel_ref, _ = self.evaluate(0.0)
            return vel_ref
        return np.array([0.0, 0.0, 0.0])

    def get_mpc_horizon_trajectory(self, N: int, dt: float) -> NDArray:
        if not hasattr(self, "_t_total") or self._t_total <= 0:
            return np.zeros((N + 1, 3))

        traj = []
        for k in range(N + 1):
            t_k = k * dt
            pos_ref, _, _ = self.evaluate(t_k)
            traj.append(pos_ref)

        return np.array(traj)

    def add_trajectory_point(self, pos: NDArray) -> None:
        if len(self._traj_history) >= self._max_history_len:
            self._traj_history.pop(0)
        self._traj_history.append(pos.copy())

    def get_trajectory_history(self) -> NDArray:
        if len(self._traj_history) == 0:
            return np.empty((0, 3))
        return np.array(self._traj_history)
