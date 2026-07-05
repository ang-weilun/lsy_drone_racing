"""CasADi optimizer for B-spline control points."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.planner.util_types import FlightCorridor, SkeletonPoint
    from lsy_drone_racing.control.sfc_planner_mpc_config import PlannerConfig


class CasadiPlanner:
    """CasADi optimizer for B-spline control points."""

    def __init__(self, config: PlannerConfig) -> None:
        """Initialize the CasadiPlanner.

        Args:
            config: The planner configuration.
        """
        self.config = config
        self._casadi_initialized = False
        self._last_P = None

    def init_casadi_planner(self) -> None:
        """Initialize CasADi variables, parameters, and constraints.

        Builds the optimization problem for B-spline control point generation.
        """
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

        self.is_waypoint_ca = self.opti.parameter(self.MAX_CTRL)
        self.waypoint_pos_ca = self.opti.parameter(self.MAX_CTRL, 3)

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
            cost += (
                self.config.W_WAYPOINT_SOFT
                * self.is_waypoint_ca[i]
                * ca.sumsqr(self.P_ca[i, :].T - self.waypoint_pos_ca[i, :].T)
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

    def prepare_corridor_constraints(
        self, corridors: list[FlightCorridor], n_pts_first: int, n_pts_rest: int
    ) -> tuple[int, dict]:
        """Convert corridor bounding boxes into matrices for CasADi.

        Args:
            corridors: List of FlightCorridor objects.
            n_pts_first: Number of control points in the first segment.
            n_pts_rest: Number of control points in subsequent segments.

        Returns:
            A tuple of (number of control points used, constraint parameter dict).
        """
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

    def prepare_gate_constraints(
        self,
        skeleton_path: list[SkeletonPoint],
        n_segments: int,
        n_ctrl: int,
        n_pts_first: int,
        n_pts_rest: int,
    ) -> dict:
        """Set up constraints for flying through gates.

        Args:
            skeleton_path: The sequence of skeleton points.
            n_segments: Number of corridor segments.
            n_ctrl: Total number of active control points.
            n_pts_first: Control points for the first segment.
            n_pts_rest: Control points for subsequent segments.

        Returns:
            A dictionary of CasADi constraint parameters for the gates.
        """
        v_is_gate = np.zeros(self.MAX_CTRL)
        v_gate_pos = np.zeros((self.MAX_CTRL, 3))
        v_is_waypoint = np.zeros(self.MAX_CTRL)
        v_waypoint_pos = np.zeros((self.MAX_CTRL, 3))
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

        for i in range(1, len(skeleton_path) - 1):
            cp_idx = cp_idx_map[i]
            if cp_idx >= self.MAX_CTRL:
                continue
            if getattr(skeleton_path[i], "is_in_tube", False):
                if v_align_mask[cp_idx] == 0.0:
                    v_align_mask[cp_idx] = 1.0
                    v_tube_gate[cp_idx] = skeleton_path[i].pos
                    v_tube_norm[cp_idx] = skeleton_path[i].gate_normal
            if getattr(skeleton_path[i], "is_waypoint", False):
                v_is_waypoint[cp_idx] = 1.0
                v_waypoint_pos[cp_idx] = skeleton_path[i].pos

        return {
            "is_gate": v_is_gate,
            "gate_pos": v_gate_pos,
            "is_waypoint": v_is_waypoint,
            "waypoint_pos": v_waypoint_pos,
            "tube_mask": v_tube_mask,
            "tube_gate": v_tube_gate,
            "tube_norm": v_tube_norm,
            "tube_sign": v_tube_sign,
            "tube_facets": v_tube_facets,
            "align_mask": v_align_mask,
        }

    def optimize_control_points(
        self,
        skeleton_path: list[SkeletonPoint],
        corridors: list[FlightCorridor],
        current_vel: NDArray,
        current_pos_for_spline: NDArray,
    ) -> NDArray:
        """Run the CasADi optimization to find B-spline control points.

        Args:
            skeleton_path: The sequence of skeleton points.
            corridors: The list of safe flight corridors.
            current_vel: The current drone velocity.
            current_pos_for_spline: The current drone position.

        Returns:
            An array of optimized control points.
        """
        if not self._casadi_initialized:
            self.init_casadi_planner()

        n_segments = len(corridors)
        pts_per_seg = max(
            1, self.config.points_per_segment // self.config.HERMITE_SAMPLES_PER_SEGMENT
        )

        if len(skeleton_path) > 1:
            dist_to_next = np.linalg.norm(skeleton_path[1].pos - skeleton_path[0].pos)
            if dist_to_next < 0.15:
                pts_first_seg = 1
            elif dist_to_next < 0.3:
                pts_first_seg = 2
            elif dist_to_next < 0.5:
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

        idx, corridor_params = self.prepare_corridor_constraints(
            corridors, pts_first_seg, pts_rest_seg
        )
        if idx == 0:
            return np.array([current_pos_for_spline] * 4)

        n_ctrl = idx
        gate_params = self.prepare_gate_constraints(
            skeleton_path, n_segments, n_ctrl, pts_first_seg, pts_rest_seg
        )
        v_ref = corridor_params["ref"]

        for i in range(1, len(skeleton_path) - 1):
            if skeleton_path[i].is_gate:
                cp_idx_map = [0]
                curr_idx = pts_first_seg
                for seg_idx in range(1, n_segments):
                    cp_idx_map.append(curr_idx)
                    curr_idx += pts_rest_seg
                cp_idx_map.append(n_ctrl - 1)
                gate_cp_idx = cp_idx_map[i]

                if gate_cp_idx >= self.MAX_CTRL:
                    continue
                gate_pos = skeleton_path[i].pos
                normal = skeleton_path[i].gate_normal
                if gate_cp_idx - 1 >= 0:
                    v_ref[gate_cp_idx - 1] = gate_pos - normal * self.config.anchor_gap
                if gate_cp_idx + 1 < n_ctrl:
                    v_ref[gate_cp_idx + 1] = gate_pos + normal * self.config.anchor_gap

        v_end_mask = np.zeros(self.MAX_CTRL)
        v_end_mask[n_ctrl - 1] = 1.0
        v_end_pos = skeleton_path[-1].pos

        v_P0_ref = current_pos_for_spline
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
        self.opti.set_value(self.is_waypoint_ca, gate_params["is_waypoint"])
        self.opti.set_value(self.waypoint_pos_ca, gate_params["waypoint_pos"])
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
