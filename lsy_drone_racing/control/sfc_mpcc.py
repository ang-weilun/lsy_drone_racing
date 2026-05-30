"""This module implements a Model Predictive Contouring Controller (MPCC) for a quadrotor.

It generates an arc-length parameterized cubic spline path from waypoints, augments the drone's
state vector with path progress and virtual velocity, and enforces obstacle avoidance constraints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
import casadi as cs

from crazyflow.sim.visualize import draw_capsule, draw_line, draw_points
from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.sfc_planner_mpc import SfcCorridorPlanner

if TYPE_CHECKING:
    from numpy.typing import NDArray

def create_acados_model(parameters: dict) -> AcadosModel:
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    theta = cs.MX.sym("theta")
    v_theta = cs.MX.sym("v_theta")
    X_aug = cs.vertcat(X, theta, v_theta)

    delta_v_theta = cs.MX.sym("delta_v_theta")
    U_aug = cs.vertcat(U, delta_v_theta)

    X_dot_aug = cs.vertcat(X_dot, v_theta, delta_v_theta)

    p_theta_i = cs.MX.sym("theta_i")
    p_c_x = cs.MX.sym("c_x", 4)
    p_c_y = cs.MX.sym("c_y", 4)
    p_c_z = cs.MX.sym("c_z", 4)
    p_capsules = cs.MX.sym("capsules", 10 * 7)
    
    p = cs.vertcat(p_theta_i, p_c_x, p_c_y, p_c_z, p_capsules)

    t_val = theta - p_theta_i
    p_d_x = p_c_x[0]*t_val**3 + p_c_x[1]*t_val**2 + p_c_x[2]*t_val + p_c_x[3]
    p_d_y = p_c_y[0]*t_val**3 + p_c_y[1]*t_val**2 + p_c_y[2]*t_val + p_c_y[3]
    p_d_z = p_c_z[0]*t_val**3 + p_c_z[1]*t_val**2 + p_c_z[2]*t_val + p_c_z[3]
    p_d = cs.vertcat(p_d_x, p_d_y, p_d_z)

    t_x = 3*p_c_x[0]*t_val**2 + 2*p_c_x[1]*t_val + p_c_x[2]
    t_y = 3*p_c_y[0]*t_val**2 + 2*p_c_y[1]*t_val + p_c_y[2]
    t_z = 3*p_c_z[0]*t_val**2 + 2*p_c_z[1]*t_val + p_c_z[2]
    t_vec = cs.vertcat(t_x, t_y, t_z)

    pos = X[0:3]
    e = pos - p_d

    t_norm = cs.norm_2(t_vec) + 1e-6
    e_l = cs.dot(t_vec, e) / t_norm
    e_c_vec = e - e_l * (t_vec / t_norm)

    y_expr = cs.vertcat(
        e_c_vec,      # 3
        e_l,          # 1
        v_theta,      # 1
        X[3:6],       # 3 (rpy)
        X[6:9],       # 3 (vel)
        X[9:12],      # 3 (drpy)
        U_aug         # 5 (r_des, p_des, y_des, thrust_des, delta_v_theta)
    )

    y_expr_e = cs.vertcat(
        e_c_vec,
        e_l,
        v_theta,
        X[3:6],
        X[6:9],
        X[9:12]
    )

    h_expr = cs.MX.zeros(10)
    for i in range(10):
        p1 = p_capsules[i*7 + 0 : i*7 + 3]
        p2 = p_capsules[i*7 + 3 : i*7 + 6]
        r  = p_capsules[i*7 + 6]
        
        v = p2 - p1
        w = pos - p1
        
        v_dot_v = cs.dot(v, v)
        v_dot_v_safe = cs.if_else(v_dot_v > 1e-6, v_dot_v, 1e-6)
        
        t = cs.dot(w, v) / v_dot_v_safe
        t = cs.fmax(0.0, cs.fmin(1.0, t))
        
        closest_pt = p1 + t * v
        diff = pos - closest_pt
        
        h_expr[i] = cs.dot(diff, diff) - r**2

    model = AcadosModel()
    model.name = "sfc_mpcc_model"
    model.f_expl_expr = X_dot_aug
    model.f_impl_expr = None
    model.x = X_aug
    model.u = U_aug
    model.p = p
    model.cost_y_expr = y_expr
    model.cost_y_expr_e = y_expr_e
    model.con_h_expr = h_expr
    model.con_h_expr_e = h_expr

    return model

def create_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    ocp = AcadosOcp()
    ocp.model = create_acados_model(parameters)

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = ocp.model.cost_y_expr.rows()
    ny_e = ocp.model.cost_y_expr_e.rows()
    np_dim = ocp.model.p.rows()

    ocp.solver_options.N_horizon = N

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    Q_c = 150.0
    Q_l = 150.0
    W_v_theta = 5.0
    
    W = np.diag([
        Q_c, Q_c, Q_c, # e_c (3)
        Q_l,           # e_l (1)
        W_v_theta,     # v_theta (1)
        1.0, 1.0, 1.0, # rpy (3)
        1.0, 1.0, 1.0, # vel (3)
        5.0, 5.0, 5.0, # drpy (3)
        1.0, 1.0, 1.0, 50.0, # u (4)
        0.5            # delta_v_theta (1)
    ])

    W_e = np.diag([
        Q_c, Q_c, Q_c,
        Q_l,
        W_v_theta,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        5.0, 5.0, 5.0
    ])

    ocp.cost.W = W
    ocp.cost.W_e = W_e

    mu = 20.0
    v_ref = mu / W_v_theta

    yref = np.zeros(ny)
    yref[4] = v_ref
    yref[17] = parameters["mass"] * -parameters["gravity_vec"][-1]

    yref_e = np.zeros(ny_e)
    yref_e[4] = v_ref

    ocp.cost.yref = yref
    ocp.cost.yref_e = yref_e

    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4, -5.0])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4, 5.0])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    ocp.constraints.x0 = np.zeros(nx)

    ocp.constraints.lh = np.zeros(10)
    ocp.constraints.uh = 1e6 * np.ones(10)
    ocp.constraints.lh_e = np.zeros(10)
    ocp.constraints.uh_e = 1e6 * np.ones(10)

    # Configure slack variables for collision avoidance
    ocp.constraints.idxsh = np.arange(10)
    ocp.constraints.idxsh_e = np.arange(10)

    # Slack variable penalties
    ocp.cost.Zl = 200.0 * np.ones(10)
    ocp.cost.Zu = 200.0 * np.ones(10)
    ocp.cost.zl = 2000.0 * np.ones(10)
    ocp.cost.zu = 2000.0 * np.ones(10)

    ocp.cost.Zl_e = 200.0 * np.ones(10)
    ocp.cost.Zu_e = 200.0 * np.ones(10)
    ocp.cost.zl_e = 2000.0 * np.ones(10)
    ocp.cost.zu_e = 2000.0 * np.ones(10)

    ocp.parameter_values = np.zeros(np_dim)

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_sfc_mpcc.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp

class AttitudeMPC(Controller):
    """Example of a MPCC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        self.planner = SfcCorridorPlanner(obs, config.env.freq)
        self._update_spline()

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()

        self._current_theta = 0.0
        self._current_v_theta = 0.5

        self._tick = 0
        self._config = config
        self._finished = False

    def _update_spline(self):
        u_samples = np.linspace(0, 1, 100)
        pts = self.planner._des_pos_spline(u_samples)
        
        diffs = np.diff(pts, axis=0)
        chords = np.linalg.norm(diffs, axis=1)
        s = np.concatenate(([0], np.cumsum(chords)))
        
        self._des_pos_spline = CubicSpline(s, pts)
        self._s_total = float(s[-1])

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        replanned = self.planner.update(obs)
        if replanned:
            self._update_spline()
            self._current_theta = 0.0

        if self._current_theta >= self._s_total - 0.1:
            self._finished = True

        s_eval = np.linspace(max(0, self._current_theta - 0.5), min(self._s_total, self._current_theta + 0.5), 20)
        pos = obs["pos"]
        dists = np.linalg.norm(self._des_pos_spline(s_eval) - pos, axis=1)
        self._current_theta = float(s_eval[np.argmin(dists)])

        capsule_params = np.zeros(70)
        is_gate_flags = np.zeros(10, dtype=bool)
        if hasattr(self.planner, "capsules") and self.planner.capsules is not None:
            capsules = self.planner.capsules
            if len(capsules) > 0:
                midpoints = np.array([(cap.p1 + cap.p2) / 2.0 for cap in capsules])
                dists_to_caps = np.linalg.norm(midpoints - pos, axis=1)
                closest_idx = np.argsort(dists_to_caps)[:10]
                for i, idx in enumerate(closest_idx):
                    cap = capsules[idx]
                    capsule_params[i*7 : i*7+3] = cap.p1
                    capsule_params[i*7+3 : i*7+6] = cap.p2
                    capsule_params[i*7+6] = cap.radius
                    is_gate_flags[i] = getattr(cap, "is_gate", False)

        Zl = 200.0 * np.ones(10)
        Zu = 200.0 * np.ones(10)
        zl = 2000.0 * np.ones(10)
        zu = 2000.0 * np.ones(10)

        for i in range(10):
            if is_gate_flags[i]:
                Zl[i] = 10.0
                Zu[i] = 10.0
                zl[i] = 100.0
                zu[i] = 100.0

        for j in range(1, self._N + 1):
            self._acados_ocp_solver.cost_set(j, "Zl", Zl)
            self._acados_ocp_solver.cost_set(j, "Zu", Zu)
            self._acados_ocp_solver.cost_set(j, "zl", zl)
            self._acados_ocp_solver.cost_set(j, "zu", zu)

        if self._tick > 0 and not replanned:
            x_prev = self._acados_ocp_solver.get(1, "x")
            self._current_theta = float(x_prev[12])
            self._current_v_theta = float(x_prev[13])

        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        
        x0 = np.concatenate((
            obs["pos"], 
            obs["rpy"], 
            obs["vel"], 
            obs["drpy"], 
            [self._current_theta], 
            [self._current_v_theta]
        ))
        
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        gates_pos = getattr(self.planner, "gates_pos", None)

        for j in range(self._N + 1):
            theta_j = self._current_theta + j * self._dt * self._current_v_theta
            theta_j = np.clip(theta_j, 0, self._s_total)
            
            p_j_pos = self._des_pos_spline(theta_j)

            segment = np.searchsorted(self._des_pos_spline.x, theta_j, side="right") - 1
            segment = np.clip(segment, 0, len(self._des_pos_spline.x) - 2)
            
            theta_i = self._des_pos_spline.x[segment]
            c_x = self._des_pos_spline.c[:, segment, 0]
            c_y = self._des_pos_spline.c[:, segment, 1]
            c_z = self._des_pos_spline.c[:, segment, 2]
            
            p_j = np.concatenate((
                [theta_i],
                c_x, c_y, c_z,
                capsule_params
            ))
            
            self._acados_ocp_solver.set(j, "p", p_j)

            Q_c_base = 150.0
            Q_c_dynamic = Q_c_base
            if gates_pos is not None and len(gates_pos) > 0:
                dists_to_gates = np.linalg.norm(gates_pos - p_j_pos, axis=1)
                min_dist = np.min(dists_to_gates)
                # Increase the contouring error penalty as the predicted position gets closer
                # to any gate, with a Gaussian-shaped increase
                # Starts increasing significantly when within ~0.2m of a gate,
                # and maxes out at 3000 when very close
                Q_c_dynamic += 3000.0 * np.exp(-(min_dist**2) / (2 * 0.2**2))
            
            W = self._acados_ocp_solver.cost_get(j, "W")
            W[0, 0] = Q_c_dynamic
            W[1, 1] = Q_c_dynamic
            W[2, 2] = Q_c_dynamic
            self._acados_ocp_solver.cost_set(j, "W", W)

        status = self._acados_ocp_solver.solve()
        if status != 0:
            pass
            
        u0 = self._acados_ocp_solver.get(0, "u")
        return u0[0:4]

    def step_callback(self, action: NDArray[np.floating], obs: dict[str, NDArray[np.floating]], reward: float, terminated: bool, truncated: bool, info: dict) -> bool:
        self._tick += 1
        return self._finished

    def episode_callback(self):
        self._tick = 0
        self._current_theta = 0.0
        self._current_v_theta = 0.5
        self.planner.episode_reset()

    def render_callback(self, sim: object) -> None:
        if hasattr(self.planner, "capsules") and self.planner.capsules is not None:
            safety_margin = getattr(self.planner, "safety_margin", 0.0)
            for cap in self.planner.capsules:
                rgba = (
                    np.array([1.0, 0.0, 0.0, 0.3])
                    if cap.is_gate
                    else np.array([0.5, 0.5, 0.5, 0.5])
                )
                draw_capsule(
                    sim, cap.p1, cap.p2, radius=max(0.01, cap.radius - safety_margin), rgba=rgba
                )

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
            theta_j = self._current_theta + j * self._dt * self._current_v_theta
            theta_j = np.clip(theta_j, 0, self._s_total)
            predicted_horizon.append(self._des_pos_spline(theta_j))
        
        if len(predicted_horizon) > 1:
            draw_line(
                sim,
                np.array(predicted_horizon),
                rgba=np.array([0.0, 1.0, 0.0, 0.9]),
                start_size=0.008,
                end_size=0.008,
            )

        if self.planner.gates_pos is not None and len(self.planner.gates_pos) > 0:
            draw_points(sim, self.planner.gates_pos, rgba=np.array([0.0, 0.0, 1.0, 0.8]), size=0.04)
