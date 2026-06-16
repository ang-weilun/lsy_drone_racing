"""This module implements a Model Predictive Contouring Controller (MPCC) for a quadrotor.

It generates an arc-length parameterized cubic spline path from waypoints, augments the drone's
state vector with path progress and virtual velocity, and enforces obstacle avoidance constraints.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crazyflow.sim.visualize import draw_capsule, draw_line, draw_points
from drone_models.core import load_params
from drone_models.so_rpy_rotor_drag import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller, mpcc_trace
from lsy_drone_racing.control.sfc_mpcc_config import MPCCConfig
from lsy_drone_racing.control.sfc_planner_mpc import SfcCorridorPlanner
from lsy_drone_racing.control.sfc_planner_mpc_config import PlannerConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def create_acados_model(parameters: dict) -> AcadosModel:
    """Create the Acados model for the MPCC.

    Args:
        parameters: Dictionary containing the drone parameters.

    Returns:
        AcadosModel: The constructed Acados model.
    """
    # model_rotor_vel=False: 12-state base + drag; we add our own scalar thrust lag below.
    X_dot, X, U, _ = symbolic_dynamics_euler(
        model_rotor_vel=False,
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        thrust_time_coef=parameters["thrust_time_coef"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
        drag_matrix=parameters["drag_matrix"],
    )

    # Promote both commands to internal states so the cost can penalize command
    # curvature and the OCP can't plan instantaneous thrust steps:
    #   rpy command -> double-integrator chain (c_rpy -> vc_rpy <- a_rpy input)
    #   thrust command -> scalar lag state f_thrust (thrust = cmd_f_coef * f_thrust, N)
    c_rpy = cs.MX.sym("c_rpy", 3)  # commanded attitude (state), fed to the plant
    vc_rpy = cs.MX.sym("vc_rpy", 3)  # command rate (state)
    a_rpy = cs.MX.sym("a_rpy", 3)  # command acceleration (input, penalized)
    f_thrust = cs.MX.sym("f_thrust")  # collective thrust state (N), lagged
    cmd_thrust = cs.MX.sym("cmd_thrust")  # commanded collective thrust (input)
    X_dot = cs.substitute(X_dot, U, cs.vertcat(c_rpy, f_thrust))
    tau_thrust = float(parameters["thrust_time_coef"])
    f_thrust_dot = (cmd_thrust - f_thrust) / tau_thrust

    theta = cs.MX.sym("theta")
    v_theta = cs.MX.sym("v_theta")
    delta_v_theta = cs.MX.sym("delta_v_theta")

    X_aug = cs.vertcat(X, f_thrust, theta, v_theta, c_rpy, vc_rpy)
    U_aug = cs.vertcat(a_rpy, cmd_thrust, delta_v_theta)
    X_dot_aug = cs.vertcat(X_dot, f_thrust_dot, v_theta, delta_v_theta, vc_rpy, a_rpy)

    p_theta_i = cs.MX.sym("theta_i")
    p_c_x = cs.MX.sym("c_x", 4)
    p_c_y = cs.MX.sym("c_y", 4)
    p_c_z = cs.MX.sym("c_z", 4)
    p_capsules = cs.MX.sym("capsules", 24 * 7)

    p = cs.vertcat(p_theta_i, p_c_x, p_c_y, p_c_z, p_capsules)

    t_val = theta - p_theta_i
    p_d_x = p_c_x[0] * t_val**3 + p_c_x[1] * t_val**2 + p_c_x[2] * t_val + p_c_x[3]
    p_d_y = p_c_y[0] * t_val**3 + p_c_y[1] * t_val**2 + p_c_y[2] * t_val + p_c_y[3]
    p_d_z = p_c_z[0] * t_val**3 + p_c_z[1] * t_val**2 + p_c_z[2] * t_val + p_c_z[3]
    p_d = cs.vertcat(p_d_x, p_d_y, p_d_z)

    t_x = 3 * p_c_x[0] * t_val**2 + 2 * p_c_x[1] * t_val + p_c_x[2]
    t_y = 3 * p_c_y[0] * t_val**2 + 2 * p_c_y[1] * t_val + p_c_y[2]
    t_z = 3 * p_c_z[0] * t_val**2 + 2 * p_c_z[1] * t_val + p_c_z[2]
    t_vec = cs.vertcat(t_x, t_y, t_z)

    pos = X[0:3]
    e = pos - p_d

    t_norm = cs.norm_2(t_vec) + 1e-6
    e_l = cs.dot(t_vec, e) / t_norm
    e_c_vec = e - e_l * (t_vec / t_norm)

    y_obs = _build_obstacle_barrier(p_capsules, pos)

    y_expr = cs.vertcat(
        e_c_vec,  # 3
        e_l,  # 1
        v_theta,  # 1
        X[3:6],  # 3 (rpy)
        X[6:9],  # 3 (vel)
        X[9:12],  # 3 (drpy)
        a_rpy,  # 3 (rpy command curvature)
        cmd_thrust,  # 1 (thrust command)
        delta_v_theta,  # 1
        y_obs,  # 24 (obstacle avoidance)
    )

    y_expr_e = cs.vertcat(e_c_vec, e_l, v_theta, X[3:6], X[6:9], X[9:12], y_obs)

    model = AcadosModel()
    model.name = "sfc_mpcc_model"
    model.f_expl_expr = X_dot_aug
    model.f_impl_expr = None
    model.x = X_aug
    model.u = U_aug
    model.p = p
    model.cost_y_expr = y_expr
    model.cost_y_expr_e = y_expr_e

    return model


def _build_obstacle_barrier(p_capsules: cs.MX, pos: cs.MX) -> cs.MX:
    """Build the C1-continuous barrier function for obstacle avoidance."""
    y_obs = cs.MX.zeros(24)
    for i in range(24):
        p1 = p_capsules[i * 7 + 0 : i * 7 + 3]
        p2 = p_capsules[i * 7 + 3 : i * 7 + 6]
        r = p_capsules[i * 7 + 6]

        v = p2 - p1
        w = pos - p1

        v_dot_v = cs.dot(v, v)
        v_dot_v_safe = cs.if_else(v_dot_v > 1e-6, v_dot_v, 1e-6)

        t = cs.dot(w, v) / v_dot_v_safe
        t = cs.fmax(0.0, cs.fmin(1.0, t))

        closest_pt = p1 + t * v
        diff = pos - closest_pt

        d2 = cs.dot(diff, diff)
        y_obs[i] = cs.fmax(0.0, 1.0 - d2 / (r**2 + 1e-6)) ** 2
    return y_obs


def _get_cost_weights(config: MPCCConfig) -> tuple[np.ndarray, np.ndarray]:
    """Get the cost weight matrices for the MPCC."""
    W_diag = [
        config.Q_c,
        config.Q_c,
        config.Q_c_z,  # e_c (3)
        config.Q_l,  # e_l (1)
        config.W_v_theta,  # v_theta (1)
        config.Q_rpy,
        config.Q_rpy,
        config.Q_rpy,  # rpy (3)
        config.Q_vel,
        config.Q_vel,
        config.Q_vel,  # vel (3)
        config.Q_drpy,
        config.Q_drpy,
        config.Q_drpy,  # drpy (3)
        config.R_curv_rpy,
        config.R_curv_rpy,
        config.R_curv_rpy,  # a_rpy (3) rpy command curvature
        config.R_cmd_thrust,  # thrust (1)
        0.5,  # delta_v_theta (1)
    ] + [config.obstacle_penalty] * 24

    W_e_diag = [
        config.Q_c,
        config.Q_c,
        config.Q_c_z,  # e_c (3)
        config.Q_l,  # e_l (1)
        config.W_v_theta,  # v_theta (1)
        config.Q_rpy,
        config.Q_rpy,
        config.Q_rpy,  # rpy (3)
        config.Q_vel,
        config.Q_vel,
        config.Q_vel,  # vel (3)
        config.Q_drpy,
        config.Q_drpy,
        config.Q_drpy,  # drpy (3)
    ] + [config.obstacle_penalty] * 24

    return np.diag(W_diag), np.diag(W_e_diag)


def create_ocp_solver(
    time_steps: np.ndarray, parameters: dict, config: MPCCConfig, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Create the Acados OCP solver for the MPCC."""
    ocp = AcadosOcp()
    ocp.model = create_acados_model(parameters)

    nx = ocp.model.x.rows()
    ny = ocp.model.cost_y_expr.rows()
    ny_e = ocp.model.cost_y_expr_e.rows()
    np_dim = ocp.model.p.rows()

    N = len(time_steps)
    ocp.solver_options.N_horizon = N

    shooting_nodes = np.zeros(N + 1)
    shooting_nodes[1:] = np.cumsum(time_steps)
    ocp.solver_options.shooting_nodes = shooting_nodes

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    ocp.cost.W, ocp.cost.W_e = _get_cost_weights(config)

    v_ref = config.mu / config.W_v_theta

    yref = np.zeros(ny)
    yref[_YREF_V_THETA] = v_ref
    yref[_YREF_THRUST] = parameters["mass"] * -parameters["gravity_vec"][-1]

    yref_e = np.zeros(ny_e)
    yref_e[_YREF_V_THETA] = v_ref

    ocp.cost.yref = yref
    ocp.cost.yref_e = yref_e

    thrust_min_coll = parameters["thrust_min"] * 4
    thrust_max_coll = parameters["thrust_max"] * 4
    ocp.constraints.lbx = np.array(
        [
            -config.MAX_ROLL_PITCH,
            -config.MAX_ROLL_PITCH,
            -config.MAX_YAW,
            thrust_min_coll,
            0.0, # theta
            config.MIN_V_THETA,
            -config.MAX_RPY_RATES,
            -config.MAX_RPY_RATES,
            -config.MAX_RPY_RATES,
        ]
    )
    ocp.constraints.ubx = np.array(
        [
            config.MAX_ROLL_PITCH,
            config.MAX_ROLL_PITCH,
            config.MAX_YAW,
            thrust_max_coll,
            1000.0, # theta
            config.MAX_V_THETA,
            config.MAX_RPY_RATES,
            config.MAX_RPY_RATES,
            config.MAX_RPY_RATES,
        ]
    )
    ocp.constraints.idxbx = np.array([
        _X_RPY.start, _X_RPY.start + 1, _X_RPY.start + 2,
        _X_F_THRUST, _X_THETA, _X_V_THETA,
        _X_C_RPY.start, _X_C_RPY.start + 1, _X_C_RPY.start + 2
    ])

    ocp.constraints.lbu = np.array(
        [
            -config.MAX_CMD_RPY_ACC,
            -config.MAX_CMD_RPY_ACC,
            -config.MAX_CMD_RPY_ACC,
            parameters["thrust_min"] * 4,
            -config.MAX_DELTA_V_THETA,
        ]
    )
    ocp.constraints.ubu = np.array(
        [
            config.MAX_CMD_RPY_ACC,
            config.MAX_CMD_RPY_ACC,
            config.MAX_CMD_RPY_ACC,
            parameters["thrust_max"] * 4,
            config.MAX_DELTA_V_THETA,
        ]
    )
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    ocp.constraints.x0 = np.zeros(nx)
    ocp.parameter_values = np.zeros(np_dim)

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tol = config.SOLVER_TOL
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = config.QP_SOLVER_ITER_MAX
    ocp.solver_options.nlp_solver_max_iter = config.NLP_SOLVER_MAX_ITER
    ocp.solver_options.tf = float(np.sum(time_steps))

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_sfc_mpcc.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp


# ---------------------------------------------------------------------------
# Index constants for the augmented state, input, and cost vectors.
#
# Augmented state x (21):  [pos(3), rpy(3), vel(3), drpy(3), f_thrust, theta, v_theta, c_rpy(3), vc_rpy(3)]
# Augmented input u (5):   [a_rpy(3), cmd_thrust, delta_v_theta]
# ---------------------------------------------------------------------------

# -- State indices --
_X_POS = slice(0, 3)
_X_RPY = slice(3, 6)
_X_VEL = slice(6, 9)
_X_DRPY = slice(9, 12)
_X_F_THRUST = 12
_X_THETA = 13
_X_V_THETA = 14
_X_C_RPY = slice(15, 18)
_X_VC_RPY = slice(18, 21)

# -- Input indices --
_U_A_RPY = slice(0, 3)
_U_THRUST = 3
_U_DELTA_V_THETA = 4

# -- Cost reference (yref) indices --
_YREF_V_THETA = 4     # virtual velocity reference
_YREF_THRUST = 17     # hover thrust reference (stage cost only)

# -- Bounded-state (ubx) index mapping --
# idxbx = [3, 4, 5, 12, 13, 14, 15, 16, 17] => ubx positions 0..8
_UBX_ROLL = 0
_UBX_PITCH = 1
_UBX_YAW = 2
_UBX_F_THRUST = 3
_UBX_THETA = 4
_UBX_V_THETA = 5
_UBX_C_ROLL = 6
_UBX_C_PITCH = 7
_UBX_C_YAW = 8


class AttitudeMPC(Controller):
    """Example of a MPCC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the AttitudeMPC controller.

        Args:
            obs: Initial observation dict containing drone state and target gate.
            info: Environment information dictionary.
            config: General controller configuration dictionary.
        """
        super().__init__(obs, info, config)
        self.mpcc_config = MPCCConfig()
        self.planner_config = PlannerConfig()

        self._time_steps = np.linspace(
            self.mpcc_config.dt_min, self.mpcc_config.dt_max, self.mpcc_config.N
        )
        self._N = len(self._time_steps)
        self._shooting_nodes = np.concatenate(([0.0], np.cumsum(self._time_steps)))

        self.planner = SfcCorridorPlanner(obs, config.env.freq, self.planner_config)
        self._update_spline()

        self.drone_params = load_params("so_rpy_rotor_drag", config.sim.drone_model)
        self.drone_params["pos_limit_low"] = np.array(
            config.env.track.safety_limits.get("pos_limit_low", [-2.5, -1.5, 0.0])
        )
        self.drone_params["pos_limit_high"] = np.array(
            config.env.track.safety_limits.get("pos_limit_high", [2.5, 1.5, 2.0])
        )

        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._time_steps, self.drone_params, self.mpcc_config
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()

        self._current_theta = 0.0
        self._current_v_theta = 0.5
        # Command states carried across ticks (RTI shift), pinned at node 0 to link
        # consecutive commands. Seed from current attitude, zero rate.
        self._current_cmd_rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        self._current_vc_rpy = np.zeros(3)
        # Thrust-lag state, carried open-loop and seeded at hover (F = m*g / cmd_f_coef).
        hover_force = float(self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1])
        self._hover_thrust = hover_force / float(self.drone_params["cmd_f_coef"])
        self._current_thrust = self._hover_thrust

        self._tick = 0
        self._config = config
        self._finished = False

        self._trace = mpcc_trace.make_recorder_if_enabled(self._N)
        if self._trace is not None:
            self._trace.set_meta(
                seed=int(config.env.seed), freq=int(config.env.freq), n_gates=len(obs["gates_pos"])
            )
            self._record_replan_trace()

    def _update_spline(self) -> None:
        """Update the reference B-spline from the planner's control points."""
        u_samples = np.linspace(0, 1, 100)
        pts = self.planner.des_pos_spline(u_samples)

        diffs = np.diff(pts, axis=0)
        chords = np.linalg.norm(diffs, axis=1)
        s = np.concatenate(([0], np.cumsum(chords)))

        self._des_pos_spline = CubicSpline(s, pts)
        self._s_total = float(s[-1])

    def _update_current_theta(self, pos: np.ndarray) -> None:
        """Update the path progress parameter `theta` based on current drone position.
        
        When replanning, we find the closest point on the new path ahead of the drone,
        but bounded by the next target gate to avoid skipping gates.
        """
        gates_pos = getattr(self.planner, "gates_pos", None)
        target_gate_idx = getattr(self.planner, "target_gate_idx", -1)
        theta_target_gate = self._s_total

        if gates_pos is not None and 0 <= target_gate_idx < len(gates_pos):
            target_gate_pos = gates_pos[target_gate_idx]
            s_eval_gate = np.linspace(0, self._s_total, max(100, int(self._s_total * 10)))
            dists_gate = np.linalg.norm(self._des_pos_spline(s_eval_gate) - target_gate_pos, axis=1)
            theta_target_gate = float(s_eval_gate[np.argmin(dists_gate)])

        s_eval = np.linspace(
            0.0, theta_target_gate, max(20, int(theta_target_gate * 40))
        )
        dists = np.linalg.norm(self._des_pos_spline(s_eval) - pos, axis=1)
        self._current_theta = float(s_eval[np.argmin(dists)])

    def _record_replan_trace(self) -> None:
        """Snapshot the freshly built plan (spline, capsules, corridors) into the trace."""
        s_dense = np.linspace(0.0, self._s_total, mpcc_trace.SPLINE_SAMPLES)
        event = self.planner.last_replan_event
        capsules = self.planner.capsules or []
        # packed per capsule as [p1(3), p2(3), radius, is_gate] — layout consumed by trace_autopsy
        capsule_arr = (
            np.array(
                [[*c.p1, *c.p2, c.radius, float(c.is_gate)] for c in capsules], dtype=np.float32
            )
            if capsules
            else np.zeros((0, 8), np.float32)
        )
        a_blocks = [np.asarray(c.A, dtype=np.float32) for c in self.planner.corridors]
        b_blocks = [np.asarray(c.b, dtype=np.float32) for c in self.planner.corridors]
        offsets = np.concatenate(([0], np.cumsum([len(b) for b in b_blocks]))).astype(np.int32)
        self._trace.record_replan(
            tick=self._tick,
            reason=str(event["reason"]) if event is not None else "init",
            spline=self._des_pos_spline(s_dense).astype(np.float32),
            capsules=capsule_arr,
            corridor_A=np.concatenate(a_blocks) if a_blocks else np.zeros((0, 3), np.float32),
            corridor_b=np.concatenate(b_blocks) if b_blocks else np.zeros(0, np.float32),
            corridor_offsets=offsets,
            gates_pos=self.planner.gates_pos.astype(np.float32),
            gates_quat=self.planner.gates_quat.astype(np.float32),
            obstacles_pos=self.planner.obstacles_pos.astype(np.float32),
        )

    def _extract_closest_obstacles(
        self, ref_points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Extract the closest obstacle capsules and gate flags for the MPC horizon."""
        capsule_params = np.zeros(168)
        for i in range(24):
            capsule_params[i * 7 : i * 7 + 6] = 1000.0
        is_gate_flags = np.zeros(24, dtype=bool)
        min_obs_dist = -1.0

        if hasattr(self.planner, "capsules") and self.planner.capsules is not None:
            capsules = self.planner.capsules
            if len(capsules) > 0:
                midpoints = np.array([(cap.p1 + cap.p2) / 2.0 for cap in capsules])
                min_obs_dist = np.min(np.linalg.norm(midpoints - ref_points[0], axis=1))

                diffs = midpoints[:, None, :] - ref_points[None, :, :]
                dists_to_path = np.linalg.norm(diffs, axis=2)
                min_dists_to_path = np.min(dists_to_path, axis=1)
                closest_idx = np.argsort(min_dists_to_path)[:24]

                for i, idx in enumerate(closest_idx):
                    cap = capsules[idx]
                    capsule_params[i * 7 : i * 7 + 3] = cap.p1
                    capsule_params[i * 7 + 3 : i * 7 + 6] = cap.p2
                    capsule_params[i * 7 + 6] = cap.radius
                    is_gate_flags[i] = getattr(cap, "is_gate", False)

        return capsule_params, is_gate_flags, min_obs_dist

    def _warm_start_solver(self, x0: np.ndarray, replanned: bool, obs: dict) -> None:
        """Warm-start the OCP solver with the initial state and reference."""
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        if self._tick == 0 or replanned:
            v_start_guess = max(0.5, self._current_v_theta)
            for j in range(self._N + 1):
                if j == 0:
                    x_guess = x0
                else:
                    theta_guess = self._current_theta + self._shooting_nodes[j] * v_start_guess
                    theta_guess = np.clip(theta_guess, 0, self._s_total)
                    pos_guess = self._des_pos_spline(theta_guess)

                    alpha = min(1.0, j / 5.0)
                    blended_pos = (1 - alpha) * obs["pos"] + alpha * pos_guess

                    x_guess = np.concatenate(
                        (
                            blended_pos,
                            obs["rpy"] * (1 - alpha),
                            obs["vel"] * (1 - alpha),
                            obs["drpy"] * (1 - alpha),
                            [self._hover_thrust],
                            [theta_guess],
                            [v_start_guess],
                            obs["rpy"] * (1 - alpha),
                            np.zeros(3),
                        )
                    )
                self._acados_ocp_solver.set(j, "x", x_guess)
        else:
            for j in range(self._N):
                self._acados_ocp_solver.set(j, "x", self._acados_ocp_solver.get(j + 1, "x"))
            for j in range(self._N - 1):
                self._acados_ocp_solver.set(j, "u", self._acados_ocp_solver.get(j + 1, "u"))
            self._acados_ocp_solver.set(0, "x", x0)

    def _check_horizon_gate_pass(self) -> tuple[bool, float]:
        """Check if the MPC horizon physically passes the target gate and find its arc-length."""
        gates_pos = getattr(self.planner, "gates_pos", None)
        target_gate_idx = getattr(self.planner, "target_gate_idx", -1)
        gates_quat = getattr(self.planner, "gates_quat", None)

        horizon_passes_gate = False
        theta_target_gate = self._s_total

        if gates_pos is not None and gates_quat is not None and 0 <= target_gate_idx < len(gates_pos):
            target_gate_pos = gates_pos[target_gate_idx]
            gate_normal = R.from_quat(gates_quat[target_gate_idx]).apply([1.0, 0.0, 0.0])

            s_eval = np.linspace(
                max(0, self._current_theta - 0.5), min(self._s_total, self._current_theta + 10.0), 500
            )
            dists = np.linalg.norm(self._des_pos_spline(s_eval) - target_gate_pos, axis=1)
            theta_target_gate = s_eval[np.argmin(dists)] + 0.2

            for j in range(self._N + 1):
                x_j = self._acados_ocp_solver.get(j, "x")
                pos_j = x_j[_X_POS]
                if (
                    np.dot(pos_j - target_gate_pos, gate_normal) > -0.1
                    and np.linalg.norm(pos_j - target_gate_pos) < 0.5
                ):
                    horizon_passes_gate = True
                    break

        return horizon_passes_gate, theta_target_gate

    def _update_horizon_progress_bounds(self, horizon_passes_gate: bool, theta_target_gate: float) -> None:
        """Update the upper bounds of the progress variable theta along the horizon."""
        for j in range(1, self._N):
            x_j = self._acados_ocp_solver.get(j, "x")
            ubx_j = self._acados_ocp_solver.constraints_get(j, "ubx")
            ubx_theta = 1000.0 if horizon_passes_gate else theta_target_gate
            
            ubx_j[_UBX_THETA] = ubx_theta
            self._acados_ocp_solver.constraints_set(j, "ubx", ubx_j)
            
            if float(x_j[_X_THETA]) > ubx_theta:
                x_j[_X_THETA] = ubx_theta
                x_j[_X_V_THETA] = 0.0
                self._acados_ocp_solver.set(j, "x", x_j)

    def _set_mpc_horizon_parameters(self, capsule_params: np.ndarray, obs: dict[str, np.ndarray]) -> None:
        """Set the reference parameters and cost weights along the MPC horizon."""
        horizon_passes_gate, theta_target_gate = self._check_horizon_gate_pass()
        self._update_horizon_progress_bounds(horizon_passes_gate, theta_target_gate)

        gates_pos = getattr(self.planner, "gates_pos", None)
        target_gate_idx = getattr(self.planner, "target_gate_idx", -1)
        target_gate_pos = gates_pos[target_gate_idx] if gates_pos is not None and 0 <= target_gate_idx < len(gates_pos) else None

        for j in range(self._N + 1):
            x_j = self._acados_ocp_solver.get(j, "x")
            theta_j = float(x_j[_X_THETA])
            theta_j = np.clip(theta_j, 0, self._s_total)
            p_j_pos = self._des_pos_spline(theta_j)

            segment = np.searchsorted(self._des_pos_spline.x, theta_j, side="right") - 1
            segment = np.clip(segment, 0, len(self._des_pos_spline.x) - 2)

            theta_i = self._des_pos_spline.x[segment]
            c_x = self._des_pos_spline.c[:, segment, 0]
            c_y = self._des_pos_spline.c[:, segment, 1]
            c_z = self._des_pos_spline.c[:, segment, 2]

            p_j = np.concatenate(([theta_i], c_x, c_y, c_z, capsule_params))
            self._acados_ocp_solver.set(j, "p", p_j)

            Q_c_dynamic = self.mpcc_config.Q_c
            if target_gate_pos is not None:
                dist_to_target_gate = np.linalg.norm(target_gate_pos - p_j_pos)
                dynamic_addition = self.mpcc_config.dynamic_addition * np.exp(
                    -(dist_to_target_gate**2) / (2 * self.mpcc_config.dynamic_sigma**2)
                )
                Q_c_dynamic += dynamic_addition

            W = self._acados_ocp_solver.cost_get(j, "W")
            W[0, 0] = Q_c_dynamic
            W[1, 1] = Q_c_dynamic
            W[2, 2] = max(Q_c_dynamic, self.mpcc_config.Q_c_z)
            self._acados_ocp_solver.cost_set(j, "W", W)

            target_v = self.mpcc_config.mu / self.mpcc_config.W_v_theta
            
            if "gates_visited" in obs and gates_pos is not None:
                for g_idx, is_visited in enumerate(obs["gates_visited"]):
                    if not is_visited:
                        dist = np.linalg.norm(gates_pos[g_idx] - p_j_pos)
                        if dist < self.mpcc_config.unobserved_dist_threshold:
                            target_v = min(target_v, self.mpcc_config.unobserved_velocity_cap)
            
            if "obstacles_visited" in obs and "obstacles_pos" in obs:
                for obs_idx, is_visited in enumerate(obs["obstacles_visited"]):
                    if not is_visited:
                        dist = np.linalg.norm(obs["obstacles_pos"][obs_idx] - p_j_pos)
                        # Obstacles are avoided, so the path won't get as close as to a gate
                        if dist < self.mpcc_config.unobserved_dist_threshold + 0.8:
                            target_v = min(target_v, self.mpcc_config.unobserved_velocity_cap)

            if j < self._N:
                yref_j = self._acados_ocp_solver.cost_get(j, "yref")
                yref_j[_YREF_V_THETA] = target_v
                self._acados_ocp_solver.cost_set(j, "yref", yref_j)
            else:
                yref_e = self._acados_ocp_solver.cost_get(self._N, "yref")
                yref_e[_YREF_V_THETA] = target_v
                self._acados_ocp_solver.cost_set(self._N, "yref", yref_e)

    def _compute_hover_control(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Compute hover fallback control if MPCC fails."""
        if not hasattr(self, "_hover_pos"):
            self._hover_pos = obs["pos"].copy()

        pos_error = self._hover_pos - obs["pos"]
        vel_error = -obs["vel"]

        target_thrust = (
            self.mpcc_config.hover_kp * pos_error + self.mpcc_config.hover_kd * vel_error
        )
        target_thrust[2] += self.drone_params["mass"] * abs(self.drone_params["gravity_vec"][-1])

        z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]
        thrust_desired = np.dot(target_thrust, z_axis)

        z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
        des_yaw = obs["rpy"][2]
        x_c_des = np.array([np.cos(des_yaw), np.sin(des_yaw), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired)
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)

        return np.concatenate([euler_desired, [thrust_desired]], dtype=np.float32)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the control input (desired attitude and thrust) for the current tick.

        Args:
            obs: The current observation dict from the environment.
            info: Optional environment info dict.

        Returns:
            The control command [desired_roll, desired_pitch, desired_yaw, thrust]
            as a float32 array.
        """
        replanned = self.planner.update(obs)
        if self._tick > 0:
            x_prev = self._acados_ocp_solver.get(1, "x")
            self._current_thrust = float(x_prev[_X_F_THRUST])
            self._current_v_theta = max(0.0, float(x_prev[_X_V_THETA]))
            self._current_cmd_rpy = np.asarray(x_prev[_X_C_RPY])
            self._current_vc_rpy = np.asarray(x_prev[_X_VC_RPY])
            if not replanned:
                self._current_theta = float(x_prev[_X_THETA])
        if replanned:
            self._update_spline()
            self._current_theta = 0.0
            self._update_current_theta(obs["pos"])

        if self._current_theta >= self._s_total - 0.1:
            self._finished = True

        ref_points = np.array(
            [
                self._des_pos_spline(
                    np.clip(self._current_theta + t_node * self._current_v_theta, 0, self._s_total)
                )
                for t_node in self._shooting_nodes
            ]
        )

        capsule_params, _, min_obs_dist = self._extract_closest_obstacles(ref_points)

        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])

        x0 = np.concatenate(
            (
                obs["pos"],
                obs["rpy"],
                obs["vel"],
                obs["drpy"],
                [self._current_thrust],
                [self._current_theta],
                [self._current_v_theta],
                self._current_cmd_rpy,
                self._current_vc_rpy,
            )
        )

        if not hasattr(self, "_prev_capsule_params"):
            self._prev_capsule_params = capsule_params

        param_diff = np.linalg.norm(capsule_params - self._prev_capsule_params)
        self._prev_capsule_params = capsule_params.copy()

        self._warm_start_solver(x0, replanned, obs)
        self._set_mpc_horizon_parameters(capsule_params, obs)

        dist_to_target_gate = -1.0
        gates_pos = getattr(self.planner, "gates_pos", None)
        target_gate_idx = getattr(self.planner, "target_gate_idx", -1)
        if gates_pos is not None and 0 <= target_gate_idx < len(gates_pos):
            dist_to_target_gate = np.linalg.norm(gates_pos[target_gate_idx] - obs["pos"])

        num_iters = 1
        if replanned or param_diff > 0.5:
            num_iters = self.mpcc_config.NLP_SOLVER_MAX_ITER

        t_solve_start = time.perf_counter()
        for _ in range(num_iters):
            self._acados_ocp_solver.options_set("rti_phase", 1)
            self._acados_ocp_solver.solve()

            self._acados_ocp_solver.options_set("rti_phase", 2)
            status = self._acados_ocp_solver.solve()
        solve_time = time.perf_counter() - t_solve_start

        is_hovering = status not in [0, 2] or hasattr(self, "_hover_pos")

        if self._tick % 10 == 0:
            dist_gate_str = f"{dist_to_target_gate:.2f}m" if dist_to_target_gate >= 0 else "N/A"
            dist_obs_str = f"{min_obs_dist:.2f}m" if min_obs_dist >= 0 else "N/A"
            hover_str = "HOVER" if is_hovering else "FLY"
            logger.info(
                f"[Tick {self._tick:04d}] Pos: ["
                f"{obs['pos'][0]:.2f}, {obs['pos'][1]:.2f}, {obs['pos'][2]:.2f}] | "
                f"Progress: {self._current_theta:.2f}/{self._s_total:.2f} | "
                f"Speed: {self._current_v_theta:.2f} | "
                f"Gate: {target_gate_idx} ({dist_gate_str}) | "
                f"Obs Dist: {dist_obs_str} | "
                f"Status: {status} ({hover_str})"
            )

        # Record flown trajectory
        self.planner.add_trajectory_point(obs["pos"])

        fallback = status not in [0, 2]
        if fallback:
            logger.warning(f"MPCC solver failed with status {status}. Entering hover mode.")
            cmd = self._compute_hover_control(obs)
        else:
            if hasattr(self, "_hover_pos"):
                del self._hover_pos
            # rpy = command state at node 1 (applied this tick); thrust = input at node 0.
            cmd_rpy = self._acados_ocp_solver.get(1, "x")[_X_C_RPY]
            cmd_thrust = self._acados_ocp_solver.get(0, "u")[_U_THRUST]
            cmd = np.concatenate([cmd_rpy, [cmd_thrust]]).astype(np.float32)

        if self._trace is not None:
            if replanned:
                self._record_replan_trace()
            horizon = np.array([self._acados_ocp_solver.get(j, "x") for j in range(self._N + 1)])
            e_c, e_l = mpcc_trace.contour_lag_errors(
                obs["pos"], self._current_theta, self._des_pos_spline
            )
            self._trace.record_tick(
                obs=obs,
                action=cmd,
                status=int(status),
                solve_time=solve_time,
                fallback=fallback,
                theta=self._current_theta,
                v_theta=self._current_v_theta,
                e_contour=e_c,
                e_lag=e_l,
                horizon_pos=horizon[:, 0:3],
                horizon_theta=horizon[:, _X_THETA],
            )
        return cmd

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Callback executed at each simulation step to increment ticks.

        Args:
            action: The action executed in the step.
            obs: The observation resulting from the step.
            reward: The reward received.
            terminated: Whether the episode terminated.
            truncated: Whether the episode was truncated.
            info: Extra info dictionary.

        Returns:
            bool: True if the drone has completed the track/finished the episode.
        """
        if self._trace is not None:
            self._trace.record_step_result(
                int(obs["target_gate"]), bool(terminated), bool(truncated)
            )
        self._tick += 1
        return self._finished

    def episode_callback(self) -> None:
        """Reset the controller state and variables at the start of a new episode."""
        if self._trace is not None:
            path = self._trace.save()
            logger.info(f"Saved MPCC trace to {path}")
            self._trace = None
        self._tick = 0
        self._current_theta = 0.0
        self._current_v_theta = 0.5
        self._current_cmd_rpy = np.zeros(3)
        self._current_vc_rpy = np.zeros(3)
        self._current_thrust = self._hover_thrust
        self._finished = False
        if hasattr(self, "_hover_pos"):
            del self._hover_pos
        self.planner.episode_reset()

    def render_callback(self, sim: object) -> None:
        """Render debug visualizations in the PyBullet simulator.

        Args:
            sim: The simulator environment instance.
        """
        if hasattr(self.planner, "capsules") and self.planner.capsules is not None:
            safety_margin = getattr(self.planner.config, "safety_margin", 0.0)
            for cap in self.planner.capsules:
                rgba_phys = (
                    np.array([1.0, 0.0, 0.0, 0.3])
                    if cap.is_gate
                    else np.array([0.5, 0.5, 0.5, 0.5])
                )
                draw_capsule(
                    sim,
                    cap.p1,
                    cap.p2,
                    radius=max(0.01, cap.radius - safety_margin),
                    rgba=rgba_phys,
                )

                if safety_margin > 0.0:
                    rgba_margin = (
                        np.array([1.0, 0.0, 0.0, 0.1])
                        if cap.is_gate
                        else np.array([0.5, 0.5, 0.5, 0.2])
                    )
                    draw_capsule(sim, cap.p1, cap.p2, radius=cap.radius, rgba=rgba_margin)

        if hasattr(self.planner, "skeleton_path") and self.planner.skeleton_path is not None:
            skeleton_pts = np.array([pt.pos for pt in self.planner.skeleton_path])
            if len(skeleton_pts) > 1:
                draw_line(
                    sim,
                    skeleton_pts,
                    rgba=np.array([0.0, 1.0, 1.0, 0.5]),
                    start_size=0.005,
                    end_size=0.005,
                )
                draw_points(sim, skeleton_pts, rgba=np.array([0.0, 1.0, 1.0, 0.8]), size=0.01)

        if hasattr(self.planner, "control_points") and self.planner.control_points is not None:
            if len(self.planner.control_points) > 0:
                draw_points(
                    sim, self.planner.control_points, rgba=np.array([1.0, 0.0, 1.0, 0.8]), size=0.02
                )
                draw_line(
                    sim,
                    self.planner.control_points,
                    rgba=np.array([1.0, 0.0, 1.0, 0.3]),
                    start_size=0.002,
                    end_size=0.002,
                )

        predicted_horizon = []
        for j in range(self._N + 1):
            x_j = self._acados_ocp_solver.get(j, "x")
            predicted_horizon.append(x_j[:3])

        full_spline_path = []
        if self._s_total > 0:
            for s in np.linspace(0, self._s_total, 100):
                full_spline_path.append(self._des_pos_spline(s))

        if len(predicted_horizon) > 1:
            draw_line(
                sim,
                np.array(predicted_horizon),
                rgba=np.array([0.0, 1.0, 0.0, 0.9]),
                start_size=0.008,
                end_size=0.008,
            )

        if len(full_spline_path) > 1:
            draw_line(
                sim,
                np.array(full_spline_path),
                rgba=np.array([0.0, 0.5, 1.0, 0.5]),
                start_size=0.005,
                end_size=0.005,
            )

        traj_history = self.planner.get_trajectory_history()[-100:]
        if len(traj_history) > 1:
            draw_line(
                sim,
                traj_history,
                rgba=np.array([1.0, 0.5, 0.0, 0.8]),  # Orange for flown trajectory
                start_size=0.004,
                end_size=0.004,
            )

        if self.planner.gates_pos is not None and len(self.planner.gates_pos) > 0:
            draw_points(sim, self.planner.gates_pos, rgba=np.array([0.0, 0.0, 1.0, 0.8]), size=0.04)
