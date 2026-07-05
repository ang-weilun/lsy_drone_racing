"""Setup utilities for the MPCC CasADi/Acados model and solver."""
from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.so_rpy_rotor_drag import symbolic_dynamics_euler

if TYPE_CHECKING:
    from lsy_drone_racing.control.sfc_mpcc_config import MPCCConfig


def _build_obstacle_barrier(p_capsules: cs.MX, pos: cs.MX) -> cs.MX:
    """Build the C1-continuous barrier function for obstacle avoidance.

    Args:
        p_capsules: The symbolic representation of obstacle capsules.
        pos: The symbolic representation of the drone position.

    Returns:
        The barrier penalty expression for 24 obstacles.
    """
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

def create_acados_model(parameters: dict) -> AcadosModel:
    """Create the Acados model for the MPCC.

    Args:
        parameters: Dictionary containing the drone parameters.

    Returns:
        AcadosModel: The constructed Acados model.
    """
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

    c_rpy = cs.MX.sym("c_rpy", 3)
    vc_rpy = cs.MX.sym("vc_rpy", 3)
    a_rpy = cs.MX.sym("a_rpy", 3)
    f_thrust = cs.MX.sym("f_thrust")
    cmd_thrust = cs.MX.sym("cmd_thrust")
    
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
        e_c_vec,
        e_l,
        v_theta,
        X[3:6],
        X[6:9],
        X[9:12],
        a_rpy,
        cmd_thrust,
        delta_v_theta,
        y_obs,
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

def _get_cost_weights(config: MPCCConfig) -> tuple[np.ndarray, np.ndarray]:
    """Get the cost weight matrices for the MPCC.

    Args:
        config: The MPCC configuration.

    Returns:
        A tuple of (W, W_e) weight matrices.
    """
    W_diag = [
        config.Q_c,
        config.Q_c,
        config.Q_c_z,
        config.Q_l,
        config.W_v_theta,
        config.Q_rpy,
        config.Q_rpy,
        config.Q_rpy,
        config.Q_vel,
        config.Q_vel,
        config.Q_vel,
        config.Q_drpy,
        config.Q_drpy,
        config.Q_drpy,
        config.R_curv_rpy,
        config.R_curv_rpy,
        config.R_curv_rpy,
        config.R_cmd_thrust,
        0.5,
    ] + [config.obstacle_penalty] * 24

    W_e_diag = [
        config.Q_c,
        config.Q_c,
        config.Q_c_z,
        config.Q_l,
        config.W_v_theta,
        config.Q_rpy,
        config.Q_rpy,
        config.Q_rpy,
        config.Q_vel,
        config.Q_vel,
        config.Q_vel,
        config.Q_drpy,
        config.Q_drpy,
        config.Q_drpy,
    ] + [config.obstacle_penalty] * 24

    return np.diag(W_diag), np.diag(W_e_diag)

def create_ocp_solver(
    time_steps: np.ndarray, parameters: dict, config: MPCCConfig, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Create the Acados OCP solver for the MPCC.

    Args:
        time_steps: The time step schedule for the MPC horizon.
        parameters: The drone parameters dictionary.
        config: The MPCC configuration.
        verbose: Whether to print Acados solver output.

    Returns:
        A tuple of the configured AcadosOcpSolver and AcadosOcp instance.
    """
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
    yref[4] = v_ref
    yref[17] = parameters["mass"] * -parameters["gravity_vec"][-1]

    yref_e = np.zeros(ny_e)
    yref_e[4] = v_ref

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
            config.MAX_V_THETA,
            config.MAX_RPY_RATES,
            config.MAX_RPY_RATES,
            config.MAX_RPY_RATES,
        ]
    )
    ocp.constraints.idxbx = np.array([3, 4, 5, 12, 14, 15, 16, 17])

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
