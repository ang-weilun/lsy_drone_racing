"""Pure SFC trajectory planner — extracted from sfc_controller.py.

Provides the path-skeleton builder, capsule obstacle model, flight-corridor
builder, and B-spline optimization. Consumed by both `sfc_controller.py`
(state-mode tracker) and `sfc_attitude_controller.py` (attitude-mode tracker).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.sfc_planner_mpc_config import PlannerConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SkeletonPoint(NamedTuple):
    """Represents a skeleton point in the planned path with gate information."""

    pos: NDArray
    is_gate: bool
    gate_normal: NDArray | None
    gate_right: NDArray | None
    gate_up: NDArray | None
    gate_idx: int | None = None


class Capsule(NamedTuple):
    """Represents a capsule obstacle (cylinder with spherical ends)."""

    p1: NDArray
    p2: NDArray
    radius: float
    is_gate: bool
    gate_idx: int | None = None


class FlightCorridor:
    """Represents a convex polyhedron (flight corridor) defined by half-spaces."""

    def __init__(self, p1: NDArray, p2: NDArray, limit_low: NDArray, limit_high: NDArray) -> None:
        """Initialize a flight corridor between two waypoints.

        Args:
            p1: Start point of the corridor.
            p2: End point of the corridor.
            limit_low: Bounding box lower limits.
            limit_high: Bounding box upper limits.
        """
        self.A = []
        self.b = []
        self.p1 = p1
        self.p2 = p2

        # Bounding box (Room limits)
        self.add_halfspace(np.array([0, 0, 1]), np.array([0, 0, limit_high[2]]))
        self.add_halfspace(np.array([0, 0, -1]), np.array([0, 0, limit_low[2]]))
        self.add_halfspace(np.array([1, 0, 0]), np.array([limit_high[0], 0, 0]))
        self.add_halfspace(np.array([-1, 0, 0]), np.array([limit_low[0], 0, 0]))
        self.add_halfspace(np.array([0, 1, 0]), np.array([0, limit_high[1], 0]))
        self.add_halfspace(np.array([0, -1, 0]), np.array([0, limit_low[1], 0]))

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

    def __init__(
        self, obs: dict[str, NDArray], freq: int, config: PlannerConfig | None = None
    ) -> None:
        """Initialize the SfcCorridorPlanner.

        Args:
            obs: Initial observation dict containing gate and obstacle positions.
            freq: Controller operating frequency (Hz).
            config: Path planner configuration dataclass.
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
        initial_vel = obs.get("vel", np.zeros(3))
        self._build_spline(obs["pos"], initial_vel)
        self._record_replan_event(reason="init")

    def update(self, obs: dict[str, NDArray]) -> bool:
        """Sync target_gate_idx from obs and replan if any object moved or gate passed."""
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
        """Evaluate the path B-spline and its derivatives at normalized arc-length parameter u.

        Args:
            u: Normalized arc-length parameter in [0, 1].

        Returns:
            A tuple containing:
                - Position vector [x, y, z] (m).
                - First derivative (velocity vector).
                - Second derivative (acceleration vector).
                - Third derivative (jerk vector).
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
        """Get the flight corridor half-space constraints (A, b) at normalized parameter u.

        Args:
            u: Normalized arc-length parameter in [0, 1].

        Returns:
            A tuple (A, b) defining the half-spaces A * x <= b, or None if no corridors exist.
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
        """Get the optimized B-spline representing the drone's target path."""
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

    def _get_all_obstacle_capsules(self) -> list[Capsule]:
        capsules = []
        margin = self.config.safety_margin

        for p in self.obstacles_pos:
            capsules.append(
                Capsule(
                    np.array([p[0], p[1], 0.0]),
                    np.array([p[0], p[1], self.config.pole_height]),
                    self.config.pole_radius + margin,
                    False,
                )
            )

        for gate_i, (pos, quat) in enumerate(zip(self.gates_pos, self.gates_quat)):
            rot = R.from_quat(quat)
            up = rot.apply([0, 0, 1])
            right = rot.apply([0, 1, 0])

            stand_h = pos[2] - self.config.gate_outer / 2.0
            if stand_h > 0:
                capsules.append(
                    Capsule(
                        pos - up * (self.config.gate_outer / 2.0),
                        pos - up * (self.config.gate_outer / 2.0 + stand_h),
                        self.config.gate_stand_radius + margin,
                        True,
                        gate_i,
                    )
                )

            bar_dist = self.config.gate_bar_dist
            bar_radius = self.config.gate_bar_radius + margin
            half_outer = self.config.gate_outer / 2.0

            capsules.append(
                Capsule(
                    pos + up * bar_dist - right * half_outer,
                    pos + up * bar_dist + right * half_outer,
                    bar_radius,
                    True,
                    gate_i,
                )
            )
            capsules.append(
                Capsule(
                    pos - up * bar_dist - right * half_outer,
                    pos - up * bar_dist + right * half_outer,
                    bar_radius,
                    True,
                    gate_i,
                )
            )
            capsules.append(
                Capsule(
                    pos - right * bar_dist + up * half_outer,
                    pos - right * bar_dist - up * half_outer,
                    bar_radius,
                    True,
                    gate_i,
                )
            )
            capsules.append(
                Capsule(
                    pos + right * bar_dist + up * half_outer,
                    pos + right * bar_dist - up * half_outer,
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
            corr = FlightCorridor(
                pt1.pos,
                pt2.pos,
                limit_low=np.array(self.config.ROOM_LIMIT_LOW),
                limit_high=np.array(self.config.ROOM_LIMIT_HIGH),
            )

            # Add separating half-spaces for all capsules
            for cap in capsules:
                # Skip a gate's frame capsules only for segments that pass
                # through the gate's opening. The segment must (a) cross the
                # gate's normal plane and (b) cross it within the frame's
                # radius - otherwise it's just going around the gate, and
                # the corridor must keep the frame as a real obstacle.
                if cap.is_gate and cap.gate_idx is not None:
                    # ONLY skip if the segment is connecting to/from this gate
                    if (
                        getattr(pt1, "gate_idx", None) == cap.gate_idx
                        or getattr(pt2, "gate_idx", None) == cap.gate_idx
                    ):
                        g_pos = self.gates_pos[cap.gate_idx]
                        g_normal = gate_normals[cap.gate_idx]
                        d1 = float(np.dot(pt1.pos - g_pos, g_normal))
                        d2 = float(np.dot(pt2.pos - g_pos, g_normal))
                        # Frame outer radius is 0.36m; allow a small slack so the
                        # pre/post anchors (which sit on the normal axis) still
                        # qualify as a through-segment.
                        near_radius = self.config.gate_outer / 2.0 + 0.10
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
                    d1_vec = pt2.pos - pt1.pos
                    perp = (
                        np.array([-d1_vec[1], d1_vec[0], 0])
                        if np.linalg.norm(d1_vec[:2]) > 1e-5
                        else np.array([1.0, 0.0, 0.0])
                    )
                    n = perp / np.linalg.norm(perp)

                effective_radius = min(cap.radius, dist - 0.005)
                plane_p = c2 + n * effective_radius
                corr.add_halfspace(-n, plane_p)

            corridors.append(corr)
        return corridors

    def _init_casadi_planner(self) -> None:
        import casadi as ca

        self.MAX_CTRL = self.config.MAX_CTRL
        self.MAX_PLANES = self.config.MAX_PLANES

        self.opti = ca.Opti("conic")
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
            diff = self.P_ca[i + 1, :] - self.P_ca[i, :]
            cost += self.config.W_VEL * self.mask_ca[i] * self.mask_ca[i + 1] * ca.sumsqr(diff)

        for i in range(self.MAX_CTRL - 2):
            diff = self.P_ca[i + 2, :] - 2 * self.P_ca[i + 1, :] + self.P_ca[i, :]
            cost += (
                self.config.W_ACC
                * self.mask_ca[i]
                * self.mask_ca[i + 1]
                * self.mask_ca[i + 2]
                * ca.sumsqr(diff)
            )

        for i in range(self.MAX_CTRL - 3):
            diff = (
                self.P_ca[i + 3, :]
                - 3 * self.P_ca[i + 2, :]
                + 3 * self.P_ca[i + 1, :]
                - self.P_ca[i, :]
            )
            cost += (
                self.config.W_JERK
                * self.mask_ca[i]
                * self.mask_ca[i + 1]
                * self.mask_ca[i + 2]
                * self.mask_ca[i + 3]
                * ca.sumsqr(diff)
            )

        for i in range(self.MAX_CTRL):
            cost += (
                self.config.W_CENTER
                * self.mask_ca[i]
                * ca.sumsqr(self.P_ca[i, :] - self.ref_pts_ca[i, :])
            )
            cost += (
                self.config.W_GATE_HARD
                * self.is_gate_ca[i]
                * ca.sumsqr(self.P_ca[i, :].T - self.gate_pos_ca[i, :].T)
            )
            cost += (
                self.config.W_GATE_HARD
                * self.end_mask_ca[i]
                * ca.sumsqr(self.P_ca[i, :].T - self.end_pos_ca)
            )

        cost += self.config.W_P0_REF * ca.sumsqr(self.P_ca[0, :].T - self.P0_ref_ca)
        cost += self.P1_weight_ca * ca.sumsqr(self.P_ca[1, :].T - self.P1_ref_ca)

        for i in range(self.MAX_CTRL):
            dp = self.P_ca[i, :] - self.tube_gate_pos_ca[i, :]
            normal = self.tube_normal_ca[i, :]
            proj = ca.dot(dp, normal) * normal
            cost += self.config.W_GATE_ALIGN * self.align_mask_ca[i] * ca.sumsqr(dp - proj)

        self.opti.minimize(cost)

        for i in range(self.MAX_CTRL):
            A_i = self.A_corr_ca[i * self.MAX_PLANES : (i + 1) * self.MAX_PLANES, :]
            b_i = self.b_corr_ca[i * self.MAX_PLANES : (i + 1) * self.MAX_PLANES]
            self.opti.subject_to(ca.mtimes(A_i, self.P_ca[i, :].T) <= b_i)

            dp = self.P_ca[i, :] - self.tube_gate_pos_ca[i, :]
            for f in range(8):
                facet = self.tube_facets_ca[i * 8 + f, :]
                val = self.tube_mask_ca[i] * ca.dot(dp, facet)
                bound = (
                    self.tube_mask_ca[i] * self.config.GATE_TUBE_RADIUS
                    + (1 - self.tube_mask_ca[i]) * 1000.0
                )
                self.opti.subject_to(val <= bound)

            proj_n = (
                self.tube_mask_ca[i] * self.tube_sign_ca[i] * ca.dot(dp, self.tube_normal_ca[i, :])
            )
            min_bound = (
                self.tube_mask_ca[i] * self.config.GATE_TUBE_AXIAL_MIN
                - (1 - self.tube_mask_ca[i]) * 1000.0
            )
            max_bound = (
                self.tube_mask_ca[i] * self.config.GATE_TUBE_HALF_LENGTH
                + (1 - self.tube_mask_ca[i]) * 1000.0
            )
            self.opti.subject_to(self.opti.bounded(min_bound, proj_n, max_bound))

        self.opti.solver("qpoases", {"printLevel": "none"})
        self._casadi_initialized = True
        self._last_P = None

    def _prepare_corridor_constraints(
        self, corridors: list[FlightCorridor], n_pts_first: int, n_pts_rest: int
    ) -> tuple[int, dict]:
        v_mask = np.zeros(self.MAX_CTRL)
        v_ref = np.zeros((self.MAX_CTRL, 3))
        v_A = np.zeros((self.MAX_CTRL * self.MAX_PLANES, 3))
        v_b = np.ones(self.MAX_CTRL * self.MAX_PLANES) * 1000.0

        idx = 0
        for seg_idx, corr in enumerate(corridors):
            n_pts = n_pts_first if seg_idx == 0 else n_pts_rest
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

        return idx, {"mask": v_mask, "ref": v_ref, "A": v_A, "b": v_b}

    def _prepare_gate_constraints(
        self,
        skeleton_path: list[SkeletonPoint],
        n_segments: int,
        n_ctrl: int,
        n_pts_first: int,
        n_pts_rest: int,
    ) -> dict:
        v_is_gate = np.zeros(self.MAX_CTRL)
        v_gate_pos = np.zeros((self.MAX_CTRL, 3))
        v_tube_mask = np.zeros(self.MAX_CTRL)
        v_tube_gate = np.zeros((self.MAX_CTRL, 3))
        v_tube_norm = np.zeros((self.MAX_CTRL, 3))
        v_tube_sign = np.zeros(self.MAX_CTRL)
        v_tube_facets = np.zeros((self.MAX_CTRL * 8, 3))
        v_align_mask = np.zeros(self.MAX_CTRL)

        cp_idx_map = [0]
        curr_idx = n_pts_first
        for seg_idx in range(1, n_segments):
            cp_idx_map.append(curr_idx)
            curr_idx += n_pts_rest
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
                for k in range(self.config.GATE_TUBE_N_FACETS):
                    theta = 2.0 * np.pi * k / self.config.GATE_TUBE_N_FACETS
                    facet_dirs.append(np.cos(theta) * right + np.sin(theta) * up)
                facet_dirs = np.array(facet_dirs)

                if gate_cp_idx - 1 >= 0:
                    v_tube_mask[gate_cp_idx - 1] = 1.0
                    v_tube_gate[gate_cp_idx - 1] = gate_pos
                    v_tube_norm[gate_cp_idx - 1] = normal
                    v_tube_sign[gate_cp_idx - 1] = -1.0
                    v_tube_facets[(gate_cp_idx - 1) * 8 : gate_cp_idx * 8] = facet_dirs
                    v_align_mask[gate_cp_idx - 1] = 1.0

                if gate_cp_idx + 1 < n_ctrl:
                    v_tube_mask[gate_cp_idx + 1] = 1.0
                    v_tube_gate[gate_cp_idx + 1] = gate_pos
                    v_tube_norm[gate_cp_idx + 1] = normal
                    v_tube_sign[gate_cp_idx + 1] = 1.0
                    v_tube_facets[(gate_cp_idx + 1) * 8 : (gate_cp_idx + 2) * 8] = facet_dirs
                    v_align_mask[gate_cp_idx + 1] = 1.0

        return {
            "is_gate": v_is_gate,
            "gate_pos": v_gate_pos,
            "tube_mask": v_tube_mask,
            "tube_gate": v_tube_gate,
            "tube_norm": v_tube_norm,
            "tube_sign": v_tube_sign,
            "tube_facets": v_tube_facets,
            "align_mask": v_align_mask,
        }

    def _optimize_control_points(
        self,
        skeleton_path: list[SkeletonPoint],
        corridors: list[FlightCorridor],
        current_vel: NDArray,
    ) -> NDArray:
        if getattr(self, "_casadi_initialized", False) is False:
            self._init_casadi_planner()

        n_segments = len(corridors)
        pts_per_seg = self.config.points_per_segment

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

        idx, corridor_params = self._prepare_corridor_constraints(
            corridors, pts_first_seg, pts_rest_seg
        )
        if idx == 0:
            return np.array([self._current_pos_for_spline] * 4)

        n_ctrl = idx
        gate_params = self._prepare_gate_constraints(
            skeleton_path, n_segments, n_ctrl, pts_first_seg, pts_rest_seg
        )
        v_ref = corridor_params["ref"]

        # Override pre/post gate references to align with normals
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
                if gate_cp_idx - 1 >= 0:
                    v_ref[gate_cp_idx - 1] = gate_pos - normal * self.config.anchor_gap
                if gate_cp_idx + 1 < n_ctrl:
                    v_ref[gate_cp_idx + 1] = gate_pos + normal * self.config.anchor_gap

        if not self.config.use_optimization:
            return v_ref[:n_ctrl]

        v_end_mask = np.zeros(self.MAX_CTRL)
        v_end_mask[n_ctrl - 1] = 1.0
        v_end_pos = skeleton_path[-1].pos

        v_P0_ref = self._current_pos_for_spline
        speed = np.linalg.norm(current_vel)
        if speed > 0.1:
            v_P1_ref = v_P0_ref + current_vel * 0.05
            v_P1_weight = self.config.W_P1_REF_HIGH
        else:
            v_P1_ref = v_P0_ref
            v_P1_weight = self.config.W_P1_REF_LOW

        self.opti.set_value(self.mask_ca, corridor_params["mask"])
        self.opti.set_value(self.ref_pts_ca, v_ref)
        self.opti.set_value(self.A_corr_ca, corridor_params["A"])
        self.opti.set_value(self.b_corr_ca, corridor_params["b"])
        self.opti.set_value(self.is_gate_ca, gate_params["is_gate"])
        self.opti.set_value(self.gate_pos_ca, gate_params["gate_pos"])
        self.opti.set_value(self.tube_mask_ca, gate_params["tube_mask"])
        self.opti.set_value(self.tube_gate_pos_ca, gate_params["tube_gate"])
        self.opti.set_value(self.tube_normal_ca, gate_params["tube_norm"])
        self.opti.set_value(self.tube_sign_ca, gate_params["tube_sign"])
        self.opti.set_value(self.tube_facets_ca, gate_params["tube_facets"])
        self.opti.set_value(self.align_mask_ca, gate_params["align_mask"])
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

    def _build_raw_skeleton(self, current_pos: NDArray) -> list[SkeletonPoint]:
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
        is_far_from_prev_gate = True
        prev_gate_idx = self.target_gate_idx - 1
        if 0 <= prev_gate_idx < len(self.gates_pos) and self.target_gate_idx < len(self.gates_pos):
            prev_pos = self.gates_pos[prev_gate_idx]
            prev_normal = gate_normals[prev_gate_idx]
            d_post = float(np.dot(current_pos - prev_pos, prev_normal))
            if d_post <= self.config.PREV_GATE_EXIT_THRESHOLD:
                is_far_from_prev_gate = False

            if 0.0 < d_post < self.config.PREV_GATE_EXIT_THRESHOLD:
                next_pos = self.gates_pos[self.target_gate_idx]
                # Test against prev_pos + prev_normal * anchor_gap, the canonical
                # post-gate exit point (no post_pos anchor in skeleton anymore).
                exit_vector = next_pos - (prev_pos + prev_normal * self.config.anchor_gap)
                if float(np.dot(exit_vector, prev_normal)) < self.config.EXIT_SWING_THRESHOLD:
                    clearance_dist = self.config.anchor_gap + self.config.EXIT_SWING_CLEARANCE
                    if d_post < clearance_dist:
                        clearance_pos = prev_pos + prev_normal * clearance_dist
                        raw_path.append(
                            SkeletonPoint(
                                clearance_pos, False, None, None, None, gate_idx=prev_gate_idx
                            )
                        )

        for i in range(self.target_gate_idx, len(self.gates_pos)):
            pos = self.gates_pos[i].copy()
            # pos[2] += 0.00  # Add upward bias to counter altitude drop
            normal = gate_normals[i].copy()
            rot = R.from_quat(self.gates_quat[i])
            right = rot.apply([0, 1, 0])
            up = rot.apply([0, 0, 1])

            flow_dir = pos - raw_path[-1].pos

            # ENTRY SWING (U-turn approach logic). Computed against gate centre
            # rather than a pre_pos anchor — same dot-product test, ~0.5 m
            if np.dot(flow_dir, normal) < self.config.ENTRY_SWING_THRESHOLD:
                skip_entry_swing = (i == self.target_gate_idx) and is_far_from_prev_gate
                
                if not skip_entry_swing:
                    if np.dot(raw_path[-1].pos - pos, right) > 0:
                        swing_pos = pos + right * self.config.ENTRY_SWING_OFFSET
                    else:
                        swing_pos = pos - right * self.config.ENTRY_SWING_OFFSET
                    
                    raw_path.append(SkeletonPoint(swing_pos, False, None, None, None, gate_idx=i))

            raw_path.append(SkeletonPoint(pos, True, normal, right, up, gate_idx=i))

            # EXIT SWING (Hairpin / Reversal Logic)
            if i + 1 < len(self.gates_pos):
                next_pos = self.gates_pos[i + 1]
                # Test against pos + normal * anchor_gap, the canonical post-gate
                # exit point, even though no post_pos anchor is in the skeleton.
                exit_vector = next_pos - (pos + normal * self.config.anchor_gap)
                if np.dot(exit_vector, normal) < self.config.EXIT_SWING_THRESHOLD:
                    clearance_dist = self.config.anchor_gap + self.config.EXIT_SWING_CLEARANCE
                    clearance_pos = pos + normal * clearance_dist
                    raw_path.append(
                        SkeletonPoint(clearance_pos, False, None, None, None, gate_idx=i)
                    )
                    side_offset = self.config.EXIT_SWING_SIDE_OFFSET
                    back_offset = self.config.EXIT_SWING_BACK_OFFSET
                    if np.dot(exit_vector, right) > 0:
                        exit_swing = clearance_pos + right * side_offset - normal * back_offset
                    else:
                        exit_swing = clearance_pos - right * side_offset - normal * back_offset
                    raw_path.append(SkeletonPoint(exit_swing, False, None, None, None, gate_idx=i))

        # Add an additional waypoint after the final gate to maintain speed through the finish line
        if len(self.gates_pos) > 0 and self.target_gate_idx <= len(self.gates_pos):
            last_gate_idx = len(self.gates_pos) - 1
            last_pos = self.gates_pos[last_gate_idx]
            last_normal = gate_normals[last_gate_idx]
            finish_pos = last_pos + last_normal * self.config.FINISH_LINE_EXT_DIST
            raw_path.append(
                SkeletonPoint(finish_pos, False, None, None, None, gate_idx=last_gate_idx)
            )

        return raw_path

    def _apply_obstacle_avoidance(self, raw_path: list[SkeletonPoint]) -> list[SkeletonPoint]:
        obs_circles = []
        margin = self.config.OBSTACLE_AVOIDANCE_MARGIN
        for p in self.obstacles_pos:
            obs_circles.append((p[:2], self.config.pole_radius + margin))
        for j, (p, q) in enumerate(zip(self.gates_pos, self.gates_quat)):
            rot = R.from_quat(q)
            right = rot.apply([0, 1, 0])
            bar_dist = self.config.gate_bar_dist
            obs_radius = self.config.gate_bar_radius + self.config.OBSTACLE_AVOIDANCE_MARGIN
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
                        dot_val = np.dot(C - prev_pt[:2], AB)
                        t = np.clip(dot_val / len_sq, 0.0, 1.0)
                        projection = prev_pt[:2] + t * AB
                        dist = np.linalg.norm(projection - C)

                        if dist < safe_radius and t < first_t:
                            first_t = t
                            push_dir = (
                                (projection - C) / dist
                                if dist > 1e-6
                                else np.array([-AB[1], AB[0]]) / np.linalg.norm(AB)
                            )

                            push_extra = self.config.OBSTACLE_AVOIDANCE_PUSH_EXTRA
                            avoidance_pt_2d = C + push_dir * (safe_radius + push_extra)
                            avoidance_z = prev_pt[2] + t * (curr_pt[2] - prev_pt[2])

                            proposed_pos = np.array(
                                [avoidance_pt_2d[0], avoidance_pt_2d[1], avoidance_z]
                            )

                            min_dist = self.config.OBSTACLE_AVOIDANCE_MIN_DIST
                            if (
                                np.linalg.norm(proposed_pos - prev_pt) > min_dist
                                and np.linalg.norm(proposed_pos - curr_pt) > min_dist
                            ):
                                avoid_pt = SkeletonPoint(proposed_pos, False, None, None, None)

                    if avoid_pt is not None:
                        new_path.append(avoid_pt)

                new_path.append(path[i])
            path = new_path
        return path

    def _calculate_anchors(self, current_pos: NDArray) -> list[SkeletonPoint]:
        raw_path = self._build_raw_skeleton(current_pos)
        path = self._apply_obstacle_avoidance(raw_path)

        low = np.array(self.config.CORRIDOR_LIMIT_LOW) + self.config.CORRIDOR_BUFFER
        high = np.array(self.config.CORRIDOR_LIMIT_HIGH) - self.config.CORRIDOR_BUFFER
        clipped_path = []
        for pt in path:
            clipped_pos = np.clip(pt.pos, low, high)
            clipped_path.append(
                SkeletonPoint(
                    clipped_pos,
                    pt.is_gate,
                    pt.gate_normal,
                    pt.gate_right,
                    pt.gate_up,
                    gate_idx=getattr(pt, "gate_idx", None),
                )
            )

        return clipped_path

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
        """Capture a snapshot of the current optimized B-spline state.

        Returns:
            A dictionary containing the spline's knots, control points,
            degree k, and target gate index.
        """
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
        """Add a position point to the flown trajectory history buffer.

        Args:
            pos: 3D position vector [x, y, z] to record.
        """
        self._traj_history.append(pos.copy())

    def get_trajectory_history(self) -> NDArray:
        """Retrieve the recorded flown trajectory history.

        Returns:
            An Nx3 array of historical position coordinates.
        """
        if len(self._traj_history) == 0:
            return np.empty((0, 3))
        return np.array(self._traj_history)
