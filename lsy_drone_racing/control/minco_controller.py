"""Minimum-jerk piecewise-quintic receding-horizon controller (L1.5).

Each replan builds an M-segment quintic polynomial trajectory through the
remaining gates. Per segment the 6 quintic coefficients are given in closed
form by the 6 boundary states ``(pos, vel, acc)`` at both ends — there is no
outer optimisation, so the replan is cheap. Intermediate waypoint velocities
and accelerations are set heuristically (v = gate-axis × V_NOMINAL, a = 0);
start boundary is the drone's current ``(pos, vel, acc)`` so the trajectory
stays attached to the real vehicle. A feasibility pass re-scales any segment
whose sampled peak velocity or acceleration exceeds V_MAX / A_MAX. Replans
fire only when observed gate/obstacle geometry or target-gate state changes,
with ``REPLAN_EVERY_TICKS`` as a cooldown.

Obstacle via-points: before committing the waypoint list, any consecutive
pair whose chord passes within R_SAFE of an obstacle gets an extra
intermediate waypoint at R_SAFE perpendicular distance from the obstacle.
The quintic absorbs it smoothly — there is no dogleg the drone has to
whiplash around.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


def quintic_bvp(
    p0: NDArray[np.floating],
    v0: NDArray[np.floating],
    a0: NDArray[np.floating],
    p1: NDArray[np.floating],
    v1: NDArray[np.floating],
    a1: NDArray[np.floating],
    T: float,
) -> NDArray[np.floating]:
    """Solve the two-point BVP for a single minimum-jerk quintic per axis.

    p(τ) = c0 + c1 τ + c2 τ² + c3 τ³ + c4 τ⁴ + c5 τ⁵
    BCs:  p(0), ṗ(0), p̈(0) = p0, v0, a0
          p(T), ṗ(T), p̈(T) = p1, v1, a1

    (c0, c1, c2) are trivial; (c3, c4, c5) come from a 3×3 solve per axis.
    Returns coefficient array of shape (6, 3) where row i is cᵢ.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    a0 = np.asarray(a0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    # Matrix is the same for every axis, so solve once as a batched RHS.
    T2, T3, T4, T5 = T * T, T ** 3, T ** 4, T ** 5
    A = np.array(
        [
            [T3, T4, T5],
            [3 * T2, 4 * T3, 5 * T4],
            [6 * T, 12 * T2, 20 * T3],
        ],
        dtype=np.float64,
    )
    rhs = np.stack(
        [
            p1 - p0 - v0 * T - 0.5 * a0 * T2,
            v1 - v0 - a0 * T,
            a1 - a0,
        ],
        axis=0,
    )  # shape (3, 3); one column per xyz axis.
    c345 = np.linalg.solve(A, rhs)  # (3, 3)
    coeffs = np.zeros((6, 3), dtype=np.float64)
    coeffs[0] = p0
    coeffs[1] = v0
    coeffs[2] = 0.5 * a0
    coeffs[3:6] = c345
    return coeffs


def _eval_poly(
    coeffs: NDArray[np.floating], tau: float, derivative: int = 0
) -> NDArray[np.floating]:
    """Evaluate a single quintic segment (or its k-th derivative) at τ.

    ``coeffs`` is (6, 3); ``derivative`` ∈ {0, 1, 2}. For derivative=1 the
    coefficients shift to represent ṗ (a quartic in τ); same for =2.
    """
    if derivative == 0:
        powers = np.array(
            [1.0, tau, tau ** 2, tau ** 3, tau ** 4, tau ** 5], dtype=np.float64
        )
        return powers @ coeffs
    if derivative == 1:
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        powers = np.array(
            [1.0, tau, tau ** 2, tau ** 3, tau ** 4], dtype=np.float64
        )
        return (d * powers) @ coeffs[1:6]
    if derivative == 2:
        d = np.array([2.0, 6.0, 12.0, 20.0], dtype=np.float64)
        powers = np.array([1.0, tau, tau ** 2, tau ** 3], dtype=np.float64)
        return (d * powers) @ coeffs[2:6]
    raise ValueError(f"unsupported derivative order: {derivative}")


def _segment_peak_v_a(coeffs: NDArray[np.floating], T: float, n: int = 20) -> tuple[float, float]:
    """Sample ``n`` points on a segment and return peak ||v||, peak ||a||."""
    taus = np.linspace(0.0, T, n)
    d1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    d2 = np.array([2.0, 6.0, 12.0, 20.0], dtype=np.float64)
    p1 = np.stack([taus ** i for i in range(5)], axis=1)  # (n,5)
    p2 = np.stack([taus ** i for i in range(4)], axis=1)  # (n,4)
    vel = (p1 * d1) @ coeffs[1:6]  # (n, 3)
    acc = (p2 * d2) @ coeffs[2:6]  # (n, 3)
    v_peak = float(np.linalg.norm(vel, axis=1).max())
    a_peak = float(np.linalg.norm(acc, axis=1).max())
    return v_peak, a_peak


class MincoController(Controller):
    """MINCO L1.5: heuristic piecewise quintic with feasibility time-scaling."""

    # Target cruise speed at intermediate gate waypoints. Quintic segments
    # interpolate smoothly between the drone's current vel and this value;
    # higher is faster but also triggers feasibility rescaling sooner.
    V_NOMINAL: float = 0.75

    # Feasibility caps. cf21B_500 can do ~3 m/s cruise and ~15 m/s² peak
    # acceleration under Mellinger cascade; stay conservative so the
    # cascade doesn't saturate during tracking.
    V_MAX: float = 2.5
    A_MAX: float = 8.0

    # Minimum time between geometry-triggered replans. The controller no
    # longer rebuilds the trajectory on a blind cadence; replanning without
    # new gate/obstacle information was the main source of reference churn.
    REPLAN_EVERY_TICKS: int = 12

    # Fallback lead time for time-domain evaluation. Normal tracking uses
    # LOOKAHEAD_DIST on the sampled path, which is more stable across replans.
    LEAD_TIME: float = 0.20
    LOOKAHEAD_DIST: float = 0.17
    FEEDFORWARD_SCALE: float = 0.32
    PROGRESS_RATE_LIMIT: float = 1.08
    TRACK_CHANGE_EPS: float = 1e-4

    # Gate waypoint layout. Each gate contributes a pre-knot + centre +
    # exit-knot, D_OFFSET along the gate axis on each side. The pre/exit
    # knots keep the trajectory locally axial so the drone flies straight
    # through the opening instead of clipping a frame bar.
    D_OFFSET: float = 0.25
    GATE_FRAME_HALF_OUTER: float = 0.36
    GATE_FRAME_MARGIN: float = 0.08
    GATE_FRAME_AXIAL_CLEARANCE: float = 0.42
    GATE_KNOT_OBSTACLE_CLEARANCE: float = 0.20

    # Obstacle avoidance. When a chord between two waypoints passes within
    # R_SAFE of an obstacle, insert an extra waypoint at R_SAFE from the
    # obstacle centre perpendicular to the chord. R_SAFE must exceed the
    # physical clearance (DRONE + OBS + margin ≈ 0.13 m) by enough to
    # absorb pose jitter (up to 0.15 m per axis) before the next replan.
    DRONE_RADIUS: float = 0.086
    OBSTACLE_RADIUS: float = 0.015
    R_SAFE: float = 0.35
    VIA_ENDPOINT_MARGIN: float = 0.20
    VIA_ENDPOINT_MARGIN_AFTER_GATE: float = 0.10

    # Minimum segment duration. Prevents division blow-ups for near-zero
    # chord lengths (e.g. pre→centre at D_OFFSET=0.25 m with V_NOMINAL=1.8
    # already gives ~0.14s; floor stops edge cases dropping below).
    T_MIN_SEG: float = 0.08
    EXIT_CLEARANCE_MARGIN: float = 0.05

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize planner state and build the first trajectory."""
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._tick = 0
        self._finished = False
        self._last_plan_tick = -self.REPLAN_EVERY_TICKS  # force an initial plan
        # Segments: list of (coeffs (6,3), T). _t_local is retained for
        # fallback evaluation and debug dumps; normal command generation
        # uses spatial projection onto _traj_samples.
        self._segments: list[tuple[NDArray[np.floating], float]] = []
        self._total_T = 0.0
        self._t_local = 0.0
        self._last_track_vector = self._track_vector(obs)
        self._exit_cleared = np.zeros(len(obs["gates_pos"]), dtype=bool)
        self._last_target_gate = int(obs["target_gate"])
        self._path_progress_s = 0.0
        # Visualisation buffers (render_callback reads these).
        self._traj_samples = np.zeros((0, 3), dtype=np.float64)
        self._traj_arc = np.zeros(0, dtype=np.float64)
        self._cmd_pos = np.zeros(3, dtype=np.float64)

        # Instrumentation: when MINCO_DUMP_PATH is set, every replan and
        # every per-tick state is recorded and pickled to that path on
        # episode_callback. Zero overhead when the env var is unset.
        self._dump_path: str | None = os.environ.get("MINCO_DUMP_PATH")
        self._dump_replans: list[dict] = []
        self._dump_ticks: list[dict] = []

        initial_start_pos = np.asarray(obs["pos"], dtype=np.float64)
        initial_start_vel = np.asarray(obs["vel"], dtype=np.float64)
        initial_start_acc = np.zeros(3, dtype=np.float64)
        self._replan(
            pos=initial_start_pos,
            vel=initial_start_vel,
            acc=initial_start_acc,
            gates_pos=np.asarray(obs["gates_pos"], dtype=np.float64),
            gates_quat=np.asarray(obs["gates_quat"], dtype=np.float64),
            obstacles_pos=np.asarray(obs["obstacles_pos"], dtype=np.float64),
            target_gate=int(obs["target_gate"]),
        )
        if self._dump_path is not None:
            self._record_replan(
                tick=0,
                t_local_pre=0.0,
                virtual_pos=initial_start_pos,
                virtual_vel=initial_start_vel,
                virtual_acc=initial_start_acc,
                obs=obs,
            )
        self._last_plan_tick = 0

    # ------------------------------------------------------------------
    # Plan building
    # ------------------------------------------------------------------

    def _build_waypoints(
        self,
        start_pos: NDArray[np.floating],
        start_vel: NDArray[np.floating],
        start_acc: NDArray[np.floating],
        gates_pos: NDArray[np.floating],
        gates_quat: NDArray[np.floating],
        obstacles_pos: NDArray[np.floating],
        avoidance_gates_pos: NDArray[np.floating] | None = None,
        avoidance_gates_quat: NDArray[np.floating] | None = None,
        filter_gate_count: int = 1,
    ) -> list[dict]:
        """Return an ordered list of waypoint dicts ``{pos, vel, acc}``.

        Two-pass build. Pass 1 collects positions only: start → optional
        obstacle via-points → gate pre/centre/exit knots (filtered to those
        still ahead along each gate's axis). Pass 2 assigns interior knot
        velocities as the *chord tangent* through each knot: for knot ``i``,
        ``vel = normalize(p[i+1] − p[i−1]) × V_NOMINAL``. This replaces the
        old rigid ``axis × V_NOMINAL`` assignment, which forced a quintic
        segment entering a gate off-axis to rotate velocity direction over
        a short T — the quintic that satisfies a large (v₀, v₁) angle with
        zero boundary accs has large mid-segment curvature, producing the
        "tight loops" we saw visually. Chord-tangent interior velocities
        keep the velocity direction consistent with the path through each
        knot, so each segment's boundary data is internally consistent and
        no loop is forced. By symmetry, when the chord prev→centre and
        centre→exit are both aligned with ``axis`` (the usual case with no
        obstacle via-point right before the gate), the chord tangent at
        centre is still axial — so we don't sacrifice gate-normal crossing.
        """
        # --- Pass 1: positions, plus per-position axial override for
        # centre knots. ``axial_vels[i]`` is either a 3-D unit axis vector
        # (the gate's axis) for centre knots, or None for everything else.
        # Centre knots get their velocity forced to axis × V_NOMINAL in
        # pass 2, regardless of whether the pre or exit knot of the same
        # gate survived the passed-knot filter — when pre is dropped, the
        # chord tangent at centre is no longer colinear with the gate
        # axis, and the drone ends up crossing the gate plane off-axis.
        # Forcing the axial velocity at centre restores gate-normal
        # crossing unconditionally.
        positions: list[NDArray[np.floating]] = [np.asarray(start_pos, dtype=np.float64)]
        axial_vels: list[NDArray[np.floating] | None] = [None]
        prev_dir_xy: NDArray[np.floating] | None = None
        speed_start = float(np.linalg.norm(start_vel[:2]))
        if speed_start > 1e-3:
            prev_dir_xy = start_vel[:2] / speed_start

        obs_xy = np.asarray(obstacles_pos, dtype=np.float64)[:, :2]
        avoidance_gate_frames: list[tuple[NDArray[np.floating], NDArray[np.floating]]] = []
        endpoint_margin = (
            self.VIA_ENDPOINT_MARGIN_AFTER_GATE
            if avoidance_gates_pos is not None and len(avoidance_gates_pos) > 0
            else self.VIA_ENDPOINT_MARGIN
        )
        if avoidance_gates_pos is not None and avoidance_gates_quat is not None:
            avoidance_gate_frames = [
                (np.asarray(gp, dtype=np.float64), np.asarray(gq, dtype=np.float64))
                for gp, gq in zip(avoidance_gates_pos, avoidance_gates_quat)
            ]
        for gate_idx, (gp, gq) in enumerate(zip(gates_pos, gates_quat)):
            axis = Rotation.from_quat(gq).apply(np.array([1.0, 0.0, 0.0]))
            axis_xy = axis[:2]
            axis_xy_norm = float(np.linalg.norm(axis_xy))
            gp_f = np.asarray(gp, dtype=np.float64)
            axis_unit = axis / max(float(np.linalg.norm(axis)), 1e-9)
            pre = self._gate_axis_knot(
                gp_f, axis_unit, -1.0, obs_xy, self.GATE_KNOT_OBSTACLE_CLEARANCE
            )
            centre = gp_f
            exit_pt = self._gate_axis_knot(
                gp_f, axis_unit, 1.0, obs_xy, self.GATE_KNOT_OBSTACLE_CLEARANCE
            )

            # Passed-knot filter: drop knots behind the drone along the
            # gate axis, otherwise the quintic plans drone → behind-knot →
            # ahead-knot and produces a hairpin before the gate plane.
            # The filter is meaningful only for gates the drone is actively
            # crossing. Future gates are always included wholesale; projecting
            # the current drone position onto a future gate's axis would say
            # more about that gate's yaw than about actual race progress.
            # Each gate knot carries an optional axial override: centre
            # knots get ``axis_unit`` (forces axial velocity in pass 2);
            # pre and exit knots get None (chord-tangent).
            if gate_idx < filter_gate_count:
                drone_along = float(np.dot(start_pos - gp_f, axis))
                gate_knots: list[tuple[NDArray[np.floating], NDArray[np.floating] | None]] = []
                if drone_along <= -self.D_OFFSET:
                    gate_knots.append((pre, None))
                if drone_along <= 0.0:
                    gate_knots.append((centre, axis_unit))
                if drone_along <= self.D_OFFSET:
                    gate_knots.append((exit_pt, None))
            else:
                gate_knots = [(pre, None), (centre, axis_unit), (exit_pt, None)]
            if not gate_knots:
                continue

            # Obstacle via-point on the chord from the last position laid
            # down so far to the first gate knot still ahead.
            first_target = gate_knots[0][0]
            for avoid_gp, avoid_gq in reversed(avoidance_gate_frames):
                frame_vias = self._gate_frame_via_points(
                    positions[-1],
                    first_target,
                    avoid_gp,
                    avoid_gq,
                    obstacles_pos,
                )
                if not frame_vias:
                    continue
                for frame_via in frame_vias:
                    obstacle_via = self._obstacle_via_point(
                        positions[-1],
                        frame_via,
                        obs_xy,
                        prev_dir_xy,
                        endpoint_margin=endpoint_margin,
                    )
                    if obstacle_via is not None:
                        positions.append(obstacle_via)
                        axial_vels.append(None)
                    positions.append(frame_via)
                    axial_vels.append(None)
                break
            via = self._obstacle_via_point(
                positions[-1],
                first_target,
                obs_xy,
                prev_dir_xy,
                endpoint_margin=endpoint_margin,
            )
            if via is not None:
                positions.append(via)
                axial_vels.append(None)
            for pt, ax_override in gate_knots:
                positions.append(pt)
                axial_vels.append(ax_override)
            avoidance_gate_frames.append((gp_f, np.asarray(gq, dtype=np.float64)))
            if axis_xy_norm > 1e-6:
                prev_dir_xy = axis_xy / axis_xy_norm

        # --- Pass 2: velocities. Start knot keeps the world-given
        # (vel, acc); terminal knot is zero-vel / zero-acc; interior knots
        # get chord-tangent velocity by default, with one exception —
        # centre knots (``axial_vels[i]`` not None) are forced to axial
        # velocity at V_NOMINAL magnitude to guarantee gate-normal
        # crossing even when the pre knot has been dropped.
        start_vel_plan = self._start_velocity_for_plan(start_vel, positions)
        wps: list[dict] = [
            {
                "pos": positions[0],
                "vel": start_vel_plan,
                "acc": np.asarray(start_acc, dtype=np.float64),
            }
        ]
        for i in range(1, len(positions) - 1):
            axial_override = axial_vels[i]
            if axial_override is not None:
                vel = axial_override * self.V_NOMINAL
            else:
                tangent = positions[i + 1] - positions[i - 1]
                t_len = float(np.linalg.norm(tangent))
                vel = (
                    (tangent / t_len) * self.V_NOMINAL
                    if t_len > 1e-6
                    else np.zeros(3, dtype=np.float64)
                )
            wps.append({"pos": positions[i], "vel": vel, "acc": np.zeros(3)})
        if len(positions) >= 2:
            wps.append(
                {"pos": positions[-1], "vel": np.zeros(3), "acc": np.zeros(3)}
            )
        return wps

    def _start_velocity_for_plan(
        self,
        start_vel: NDArray[np.floating],
        positions: list[NDArray[np.floating]],
    ) -> NDArray[np.floating]:
        """Use only the component of measured velocity that helps the new plan.

        The measured velocity is a useful hint when it already points along
        the first planned leg. When the drone is off the path, though, the
        sideways component is tracking error, not a trajectory boundary
        condition. Forcing the quintic to preserve that component at every
        replan is what made short first segments curl away from the course.
        """
        if len(positions) < 2:
            return np.zeros(3, dtype=np.float64)
        first_leg = np.asarray(positions[1] - positions[0], dtype=np.float64)
        first_leg_len = float(np.linalg.norm(first_leg))
        if first_leg_len < 1e-6:
            return np.zeros(3, dtype=np.float64)
        direction = first_leg / first_leg_len
        speed_along = float(np.dot(np.asarray(start_vel, dtype=np.float64), direction))
        speed = float(np.clip(speed_along, 0.0, self.V_NOMINAL))
        return direction * speed

    @staticmethod
    def _obstacle_clearance_xy(
        point: NDArray[np.floating],
        obs_xy: NDArray[np.floating],
    ) -> float:
        """Return the closest XY distance from ``point`` to any obstacle."""
        if obs_xy.shape[0] == 0:
            return np.inf
        return float(np.min(np.linalg.norm(obs_xy - point[:2], axis=1)))

    def _gate_axis_knot(
        self,
        centre: NDArray[np.floating],
        axis_unit: NDArray[np.floating],
        sign: float,
        obs_xy: NDArray[np.floating],
        required_clearance: float,
    ) -> NDArray[np.floating]:
        """Place a pre/exit gate knot, shortening it if an obstacle is too close."""
        target_offset = self.D_OFFSET
        point = centre + sign * target_offset * axis_unit
        if (
            obs_xy.shape[0] == 0
            or self._obstacle_clearance_xy(point, obs_xy)
            >= required_clearance
        ):
            return point
        if self._obstacle_clearance_xy(centre, obs_xy) < required_clearance:
            return point

        lo, hi = 0.0, target_offset
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            candidate = centre + sign * mid * axis_unit
            if self._obstacle_clearance_xy(candidate, obs_xy) >= required_clearance:
                lo = mid
            else:
                hi = mid
        return centre + sign * lo * axis_unit

    def _obstacle_via_point(
        self,
        a: NDArray[np.floating],
        b: NDArray[np.floating],
        obs_xy: NDArray[np.floating],
        prev_dir_xy: NDArray[np.floating] | None,
        endpoint_margin: float | None = None,
    ) -> NDArray[np.floating] | None:
        """Return an obstacle-clearance via-point for chord ``a→b``.

        If the chord passes within R_SAFE of any obstacle, return a 3-D
        via-point at R_SAFE perpendicular from the obstacle centre; else
        return None. Push direction biased toward ``prev_dir_xy`` so the
        trajectory wraps the obstacle on the forward-continuing side.
        Via-point z is linearly interpolated between ``a.z`` and ``b.z``.
        """
        a_xy = a[:2]
        b_xy = b[:2]
        ab = b_xy - a_xy
        ab_len = float(np.linalg.norm(ab))
        if ab_len < 1e-5 or obs_xy.shape[0] == 0:
            return None
        endpoint_margin = self.VIA_ENDPOINT_MARGIN if endpoint_margin is None else endpoint_margin
        ab_unit = ab / ab_len
        best_d = self.R_SAFE
        best_hit: tuple[float, NDArray[np.floating], NDArray[np.floating]] | None = None
        for o_xy in obs_xy:
            t = max(0.0, min(ab_len, float(np.dot(o_xy - a_xy, ab_unit))))
            closest = a_xy + t * ab_unit
            d = float(np.linalg.norm(o_xy - closest))
            dist_a = float(np.linalg.norm(o_xy - a_xy))
            dist_b = float(np.linalg.norm(o_xy - b_xy))
            near_start = t < endpoint_margin and dist_b >= dist_a
            near_end = (ab_len - t) < endpoint_margin and dist_a >= dist_b
            if near_start or near_end:
                continue
            if d < best_d:
                best_d = d
                best_hit = (t, closest, o_xy)
        if best_hit is None:
            return None
        t_close, closest, o_xy = best_hit
        perp1 = np.array([-ab_unit[1], ab_unit[0]])
        perp2 = -perp1
        # Start from the obstacle→closest vector; if that's near zero
        # (chord passes through obstacle centre), fall back to perp1.
        offset = closest - o_xy
        offset_len = float(np.linalg.norm(offset))
        push_dir = offset / offset_len if offset_len > 1e-6 else perp1
        if prev_dir_xy is not None:
            pd = np.asarray(prev_dir_xy, dtype=np.float64)
            pd_len = float(np.linalg.norm(pd))
            if pd_len > 1e-6:
                pd_unit = pd / pd_len
                s1 = float(np.dot(perp1, pd_unit))
                s2 = float(np.dot(perp2, pd_unit))
                if abs(s1 - s2) > 0.1:
                    push_dir = perp1 if s1 >= s2 else perp2
        via_xy = o_xy + self.R_SAFE * push_dir
        z_frac = t_close / ab_len
        via_z = a[2] + z_frac * (b[2] - a[2])
        return np.array([via_xy[0], via_xy[1], via_z], dtype=np.float64)

    def _gate_collision_boxes(
        self,
    ) -> tuple[tuple[NDArray[np.floating], NDArray[np.floating]], ...]:
        """Collision boxes from ``envs/assets/gate.xml`` in gate-local coordinates."""
        return (
            (np.array([0.0, 0.0, 0.28]), np.array([0.01, 0.36, 0.08])),
            (np.array([0.0, 0.0, -0.28]), np.array([0.01, 0.36, 0.08])),
            (np.array([0.0, -0.28, 0.0]), np.array([0.01, 0.08, 0.36])),
            (np.array([0.0, 0.28, 0.0]), np.array([0.01, 0.08, 0.36])),
        )

    @staticmethod
    def _box_signed_distance(
        p_local: NDArray[np.floating],
        center: NDArray[np.floating],
        half_size: NDArray[np.floating],
    ) -> float:
        """Signed distance from a point to an axis-aligned box."""
        q = np.abs(p_local - center) - half_size
        outside = float(np.linalg.norm(np.maximum(q, 0.0)))
        inside = float(min(max(q[0], max(q[1], q[2])), 0.0))
        return outside + inside

    def _gate_frame_distance_local(self, p_local: NDArray[np.floating]) -> float:
        """Minimum signed distance from ``p_local`` to the gate collision frame."""
        return min(
            self._box_signed_distance(p_local, center, half_size)
            for center, half_size in self._gate_collision_boxes()
        )

    def _chord_gate_frame_distance(
        self,
        a_local: NDArray[np.floating],
        b_local: NDArray[np.floating],
        n: int = 40,
    ) -> float:
        """Sample a local-frame chord and return its closest gate-frame distance."""
        return min(
            self._gate_frame_distance_local(a_local + u * (b_local - a_local))
            for u in np.linspace(0.0, 1.0, n)
        )

    def _polyline_obstacle_clearance(
        self,
        points: list[NDArray[np.floating]],
        obstacles_pos: NDArray[np.floating],
    ) -> float:
        """Return the minimum XY clearance from a candidate polyline to obstacles."""
        obstacles_pos = np.asarray(obstacles_pos, dtype=np.float64)
        if obstacles_pos.shape[0] == 0:
            return np.inf
        best = np.inf
        obs_xy = obstacles_pos[:, :2]
        for p0, p1 in zip(points[:-1], points[1:]):
            a_xy = p0[:2]
            b_xy = p1[:2]
            ab = b_xy - a_xy
            ab_len2 = float(np.dot(ab, ab))
            for o_xy in obs_xy:
                if ab_len2 < 1e-10:
                    closest = a_xy
                else:
                    u = float(np.clip(np.dot(o_xy - a_xy, ab) / ab_len2, 0.0, 1.0))
                    closest = a_xy + u * ab
                best = min(best, float(np.linalg.norm(o_xy - closest)))
        return best

    def _gate_frame_via_points(
        self,
        a: NDArray[np.floating],
        b: NDArray[np.floating],
        gate_pos: NDArray[np.floating],
        gate_quat: NDArray[np.floating],
        obstacles_pos: NDArray[np.floating],
    ) -> list[NDArray[np.floating]]:
        """Return side-detour points if chord ``a→b`` cuts an already-passed gate frame."""
        rot = Rotation.from_quat(gate_quat)
        a_local = rot.apply(np.asarray(a, dtype=np.float64) - gate_pos, inverse=True)
        b_local = rot.apply(np.asarray(b, dtype=np.float64) - gate_pos, inverse=True)
        min_dist = self._chord_gate_frame_distance(a_local, b_local)
        required_clearance = self.DRONE_RADIUS + self.GATE_FRAME_MARGIN
        if min_dist >= required_clearance:
            return []

        side_offset = self.GATE_FRAME_HALF_OUTER + required_clearance
        midpoint_y = 0.5 * (a_local[1] + b_local[1])
        preferred = 1.0 if (b_local[1] >= 0.0 or midpoint_y >= 0.0) else -1.0
        signs = (preferred, -preferred)
        candidates: list[tuple[float, list[NDArray[np.floating]]]] = []
        min_axial_clearance = required_clearance + 0.02
        for sign in signs:
            y_side = sign * side_offset
            via1_x = (
                a_local[0]
                if abs(a_local[0]) >= min_axial_clearance
                else np.copysign(self.GATE_FRAME_AXIAL_CLEARANCE, a_local[0] or 1.0)
            )
            via2_x = (
                b_local[0]
                if abs(b_local[0]) >= min_axial_clearance
                else np.copysign(self.GATE_FRAME_AXIAL_CLEARANCE, b_local[0] or 1.0)
            )
            via1_local = np.array(
                [
                    via1_x,
                    y_side,
                    a_local[2],
                ],
                dtype=np.float64,
            )
            via2_local = np.array(
                [
                    via2_x,
                    y_side,
                    b_local[2],
                ],
                dtype=np.float64,
            )
            via_points = [
                rot.apply(via1_local) + gate_pos,
                rot.apply(via2_local) + gate_pos,
            ]
            polyline = [
                np.asarray(a, dtype=np.float64),
                *via_points,
                np.asarray(b, dtype=np.float64),
            ]
            length = sum(
                float(np.linalg.norm(p1 - p0)) for p0, p1 in zip(polyline[:-1], polyline[1:])
            )
            obstacle_clearance = self._polyline_obstacle_clearance(polyline, obstacles_pos)
            obstacle_penalty = max(0.0, self.R_SAFE - obstacle_clearance) * 5.0
            candidates.append((length + obstacle_penalty, via_points))

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _solve_segment(self, wp0: dict, wp1: dict, T0: float) -> tuple[NDArray[np.floating], float]:
        """Solve one quintic segment with feasibility time-scaling.

        Starts from duration ``T0``; if sampled peak velocity or accel
        exceeds the feasibility caps, re-scale T and re-solve. One or two
        iterations are usually enough because scaling is near-linear:
        doubling T halves peak velocity and quarters peak accel. Caps
        iterations at 4 to avoid pathological blow-ups.
        """
        T = max(float(T0), self.T_MIN_SEG)
        coeffs = quintic_bvp(
            wp0["pos"], wp0["vel"], wp0["acc"],
            wp1["pos"], wp1["vel"], wp1["acc"],
            T,
        )
        for _ in range(4):
            v_peak, a_peak = _segment_peak_v_a(coeffs, T)
            v_ratio = v_peak / self.V_MAX if self.V_MAX > 0 else 1.0
            a_ratio = np.sqrt(a_peak / self.A_MAX) if self.A_MAX > 0 else 1.0
            scale = float(max(1.0, v_ratio, a_ratio))
            if scale <= 1.001:
                return coeffs, T
            T = T * scale
            coeffs = quintic_bvp(
                wp0["pos"], wp0["vel"], wp0["acc"],
                wp1["pos"], wp1["vel"], wp1["acc"],
                T,
            )
        return coeffs, T

    def _replan(
        self,
        pos: NDArray[np.floating],
        vel: NDArray[np.floating],
        acc: NDArray[np.floating],
        gates_pos: NDArray[np.floating],
        gates_quat: NDArray[np.floating],
        obstacles_pos: NDArray[np.floating],
        target_gate: int,
    ) -> None:
        """Build and install a new feasible piecewise-quintic trajectory.

        Waypoints are built from the current/previous gate context, each
        segment is solved and time-scaled, and path samples are cached for
        spatial-lookahead tracking.
        """
        if target_gate < 0:
            # No gates remaining — stop replanning; let the current plan
            # coast to its terminal state.
            return
        wps = self._build_waypoints(
            start_pos=pos,
            start_vel=vel,
            start_acc=acc,
            gates_pos=gates_pos[target_gate:],
            gates_quat=gates_quat[target_gate:],
            obstacles_pos=obstacles_pos,
            avoidance_gates_pos=gates_pos[:target_gate],
            avoidance_gates_quat=gates_quat[:target_gate],
            filter_gate_count=1,
        )
        segments: list[tuple[NDArray[np.floating], float]] = []
        for wp0, wp1 in zip(wps[:-1], wps[1:]):
            dist = float(np.linalg.norm(wp1["pos"] - wp0["pos"]))
            T0 = max(dist / self.V_NOMINAL, self.T_MIN_SEG)
            coeffs, T = self._solve_segment(wp0, wp1, T0)
            segments.append((coeffs, T))
        self._segments = segments
        self._total_T = float(sum(T for _, T in segments))
        self._t_local = 0.0
        self._path_progress_s = 0.0

        # Cache trajectory samples for viz; also used for debug reads.
        if segments:
            ss = np.linspace(0.0, self._total_T, 120)
            self._traj_samples = np.stack(
                [self._eval(s, derivative=0) for s in ss], axis=0
            )
            ds = np.linalg.norm(np.diff(self._traj_samples, axis=0), axis=1)
            self._traj_arc = np.concatenate(([0.0], np.cumsum(ds)))
        else:
            self._traj_samples = np.zeros((0, 3), dtype=np.float64)
            self._traj_arc = np.zeros(0, dtype=np.float64)

    # ------------------------------------------------------------------
    # Instrumentation
    # ------------------------------------------------------------------

    def _track_vector(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Flatten observed track geometry for cheap change detection."""
        return np.concatenate(
            [
                np.asarray(obs["gates_pos"], dtype=np.float64).ravel(),
                np.asarray(obs["gates_quat"], dtype=np.float64).ravel(),
                np.asarray(obs["obstacles_pos"], dtype=np.float64).ravel(),
            ]
        )

    def _track_changed(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        track_vector = self._track_vector(obs)
        if track_vector.shape != self._last_track_vector.shape:
            return True
        return bool(np.max(np.abs(track_vector - self._last_track_vector)) > self.TRACK_CHANGE_EPS)

    def _update_exit_clearance(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Latch passed-gate exits so replans cannot resurrect old gates."""
        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
        gates_quat = np.asarray(obs["gates_quat"], dtype=np.float64)
        if self._exit_cleared.shape[0] != gates_pos.shape[0]:
            self._exit_cleared = np.zeros(gates_pos.shape[0], dtype=bool)

        target_gate = int(obs["target_gate"])
        if target_gate <= 0:
            return

        pos = np.asarray(obs["pos"], dtype=np.float64)
        for gate_idx in range(min(target_gate, gates_pos.shape[0])):
            if self._exit_cleared[gate_idx]:
                continue
            axis = Rotation.from_quat(gates_quat[gate_idx]).apply(np.array([1.0, 0.0, 0.0]))
            drone_along = float(np.dot(pos - gates_pos[gate_idx], axis))
            if drone_along > self.D_OFFSET + self.EXIT_CLEARANCE_MARGIN:
                self._exit_cleared[gate_idx] = True

    def _in_gate_handoff(self, target_gate: int) -> bool:
        """Return True while the previous gate's exit corridor is not clear."""
        return (
            target_gate > 0
            and target_gate - 1 < self._exit_cleared.shape[0]
            and not self._exit_cleared[target_gate - 1]
        )

    def _record_replan(
        self,
        tick: int,
        t_local_pre: float,
        virtual_pos: NDArray[np.floating],
        virtual_vel: NDArray[np.floating],
        virtual_acc: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
    ) -> None:
        """Append a full snapshot of the just-built plan to the dump buffer."""
        self._dump_replans.append(
            {
                "tick": int(tick),
                "t_wall": float(tick) / float(self._freq),
                "t_local_pre": float(t_local_pre),
                "virtual_start_pos": np.asarray(virtual_pos, dtype=np.float64).copy(),
                "virtual_start_vel": np.asarray(virtual_vel, dtype=np.float64).copy(),
                "virtual_start_acc": np.asarray(virtual_acc, dtype=np.float64).copy(),
                "obs_pos": np.asarray(obs["pos"], dtype=np.float64).copy(),
                "obs_vel": np.asarray(obs["vel"], dtype=np.float64).copy(),
                "target_gate": int(obs["target_gate"]),
                "gates_pos": np.asarray(obs["gates_pos"], dtype=np.float64).copy(),
                "gates_quat": np.asarray(obs["gates_quat"], dtype=np.float64).copy(),
                "obstacles_pos": np.asarray(obs["obstacles_pos"], dtype=np.float64).copy(),
                "segments": [(c.copy(), float(T)) for c, T in self._segments],
                "total_T": float(self._total_T),
            }
        )

    # ------------------------------------------------------------------
    # Trajectory evaluation
    # ------------------------------------------------------------------

    def _eval(self, t: float, derivative: int = 0) -> NDArray[np.floating]:
        """Evaluate the piecewise trajectory at global time ``t``.

        Walks the segment list to find which segment contains ``t``, then
        evaluates the local quintic at the relative τ inside that segment.
        Times past the end hold at the terminal state; times before the
        start hold at the initial state. Because we build with zero-vel /
        zero-acc terminal BCs, holding-at-end is the rest state.
        """
        if not self._segments:
            return np.zeros(3, dtype=np.float64)
        if t <= 0.0:
            c0, _ = self._segments[0]
            return _eval_poly(c0, 0.0, derivative)
        acc_T = 0.0
        for coeffs, T in self._segments:
            if t <= acc_T + T or T <= 0.0:
                tau = max(0.0, min(T, t - acc_T))
                return _eval_poly(coeffs, tau, derivative)
            acc_T += T
        coeffs, T = self._segments[-1]
        return _eval_poly(coeffs, T, derivative)

    def _path_setpoint(
        self, pos: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Return a spatial-lookahead point and tangent on the sampled path."""
        if self._traj_samples.shape[0] < 2 or self._traj_arc.shape[0] < 2:
            return self._eval(self._t_local + self.LEAD_TIME, 0), np.zeros(3)

        best_dist = np.inf
        best_s = 0.0
        best_tangent = np.zeros(3, dtype=np.float64)
        for i, (a, b) in enumerate(zip(self._traj_samples[:-1], self._traj_samples[1:])):
            ab = b - a
            seg_len2 = float(np.dot(ab, ab))
            if seg_len2 < 1e-10:
                continue
            u = float(np.clip(np.dot(pos - a, ab) / seg_len2, 0.0, 1.0))
            closest = a + u * ab
            dist = float(np.linalg.norm(pos - closest))
            if dist < best_dist:
                seg_len = float(np.sqrt(seg_len2))
                best_dist = dist
                best_s = float(self._traj_arc[i] + u * seg_len)
                best_tangent = ab / seg_len

        best_s = max(best_s, self._path_progress_s)
        max_step = self.PROGRESS_RATE_LIMIT / float(self._freq)
        best_s = min(best_s, self._path_progress_s + max_step)
        self._path_progress_s = best_s
        target_s = min(best_s + self.LOOKAHEAD_DIST, float(self._traj_arc[-1]))
        idx = int(np.searchsorted(self._traj_arc, target_s, side="right") - 1)
        idx = max(0, min(idx, self._traj_samples.shape[0] - 2))
        s0 = float(self._traj_arc[idx])
        s1 = float(self._traj_arc[idx + 1])
        a = self._traj_samples[idx]
        b = self._traj_samples[idx + 1]
        if s1 <= s0:
            return a.copy(), best_tangent
        u = (target_s - s0) / (s1 - s0)
        tangent = b - a
        tangent_len = float(np.linalg.norm(tangent))
        if tangent_len > 1e-9:
            tangent = tangent / tangent_len
        else:
            tangent = best_tangent
        return (a + u * (b - a)).astype(np.float64), tangent.astype(np.float64)

    # ------------------------------------------------------------------
    # Controller API
    # ------------------------------------------------------------------

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next state-command action."""
        pos = np.asarray(obs["pos"], dtype=np.float64)
        vel = np.asarray(obs["vel"], dtype=np.float64)
        self._update_exit_clearance(obs)
        target_gate = int(obs["target_gate"])
        target_changed = target_gate != self._last_target_gate
        in_handoff = self._in_gate_handoff(target_gate)

        # Geometry-triggered replan. Start BC = drone's actual (pos, vel) with
        # start_acc = 0. Using zero acc keeps the three BCs internally
        # consistent (no Frankenstein mix of true state and commanded
        # state) while sidestepping the noise-spike problem of numerical-
        # differentiated acc that the old obs-based approach had — zeros
        # has no noise.
        #
        # The fully virtual-state start BC we tried previously (p, v, a
        # all from old-plan _eval at _t_local) gave C² seam continuity but
        # silently diverged from the drone's real state whenever tracking
        # error opened up: by replan 5 of a dump capture, the virtual-
        # state velocity pointed 117° away from the drone's actual
        # velocity, and the planner was solving trajectories for a ghost
        # drone. Grounding the BC in the drone's actual (pos, vel) keeps
        # the plan attached to reality; the "jerk" at the seam from the
        # lost C² continuity is bounded by the drone's current tracking
        # error, which is much smaller than the alternative failure mode.
        can_replan = self._tick - self._last_plan_tick >= self.REPLAN_EVERY_TICKS
        should_replan = (not self._segments) or (
            can_replan and (target_changed or self._track_changed(obs))
        )
        if in_handoff and self._segments:
            should_replan = False
        if should_replan:
            t_local_pre = float(self._t_local)
            start_pos = pos
            start_vel = vel
            start_acc = np.zeros(3, dtype=np.float64)
            self._replan(
                pos=start_pos,
                vel=start_vel,
                acc=start_acc,
                gates_pos=np.asarray(obs["gates_pos"], dtype=np.float64),
                gates_quat=np.asarray(obs["gates_quat"], dtype=np.float64),
                obstacles_pos=np.asarray(obs["obstacles_pos"], dtype=np.float64),
                target_gate=target_gate,
            )
            self._last_plan_tick = self._tick
            self._last_track_vector = self._track_vector(obs)
            if self._dump_path is not None:
                self._record_replan(
                    tick=self._tick,
                    t_local_pre=t_local_pre,
                    virtual_pos=start_pos,
                    virtual_vel=start_vel,
                    virtual_acc=start_acc,
                    obs=obs,
                )
        if should_replan or not target_changed:
            self._last_target_gate = target_gate

        dt = 1.0 / self._freq
        self._t_local += dt
        des_pos, tangent = self._path_setpoint(pos)
        des_vel = tangent * (self.V_NOMINAL * self.FEEDFORWARD_SCALE)
        des_acc = np.zeros(3, dtype=np.float64)
        vx, vy = float(des_vel[0]), float(des_vel[1])
        yaw = float(np.arctan2(vy, vx)) if (vx * vx + vy * vy) > 1e-6 else 0.0

        self._cmd_pos = des_pos.copy()
        if self._dump_path is not None:
            self._dump_ticks.append(
                {
                    "tick": int(self._tick),
                    "t_wall": float(self._tick) / float(self._freq),
                    "t_local": float(self._t_local),
                    "obs_pos": pos.copy(),
                    "obs_vel": vel.copy(),
                    "cmd_pos": des_pos.astype(np.float64).copy(),
                    "cmd_vel": des_vel.astype(np.float64).copy(),
                    "cmd_acc": des_acc.astype(np.float64).copy(),
                    "cmd_yaw": float(yaw),
                    "target_gate": int(obs["target_gate"]),
                }
            )

        action = np.zeros(13, dtype=np.float32)
        action[0:3] = des_pos
        action[3:6] = des_vel
        action[6:9] = des_acc
        action[9] = yaw
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance controller bookkeeping after an environment step."""
        self._tick += 1
        if int(obs["target_gate"]) == -1:
            self._finished = True
        return self._finished

    def episode_callback(self) -> None:
        """Flush optional debug data and reset episode-local state."""
        # Flush the dump before resetting per-episode state. If multiple
        # episodes run under the same dump path, append a numeric suffix
        # to avoid clobbering. File is written only when there's something
        # to save and when instrumentation is enabled.
        if self._dump_path is not None and (self._dump_replans or self._dump_ticks):
            path = Path(self._dump_path)
            if path.exists():
                i = 1
                while path.with_suffix(f".{i}{path.suffix}").exists():
                    i += 1
                path = path.with_suffix(f".{i}{path.suffix}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(
                    {"replans": self._dump_replans, "ticks": self._dump_ticks},
                    f,
                )
            self._dump_replans = []
            self._dump_ticks = []

        self._tick = 0
        self._finished = False
        self._segments = []
        self._total_T = 0.0
        self._t_local = 0.0
        self._last_plan_tick = -self.REPLAN_EVERY_TICKS
        self._last_track_vector = np.zeros(0, dtype=np.float64)
        self._exit_cleared = np.zeros(0, dtype=bool)
        self._last_target_gate = 0
        self._path_progress_s = 0.0

    def render_callback(self, sim: Sim) -> None:
        """Draw the planned trajectory (green) and the commanded setpoint (red)."""
        if self._traj_samples.shape[0] >= 2:
            draw_line(sim, self._traj_samples, rgba=(0.0, 1.0, 0.0, 1.0))
        draw_points(
            sim, self._cmd_pos.reshape(1, -1),
            rgba=(1.0, 0.0, 0.0, 1.0), size=0.03,
        )
