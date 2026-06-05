"""This module implements a Model Predictive Contouring Controller (MPCC) for a quadrotor.

It generates an arc-length parameterized cubic spline path from waypoints, augments the drone's
state vector with path progress and virtual velocity, and enforces obstacle avoidance constraints.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import casadi as cs
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crazyflow.sim.visualize import draw_capsule, draw_line, draw_points
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
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
        U_aug,  # 5 (r_des, p_des, y_des, thrust_des, delta_v_theta)
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
        1.0,
        1.0,
        1.0,  # rpy (3)
        1.0,
        1.0,
        1.0,  # vel (3)
        5.0,
        5.0,
        5.0,  # drpy (3)
        1.0,
        1.0,
        1.0,
        10.0,  # u (4)
        0.5,  # delta_v_theta (1)
    ] + [config.obstacle_penalty] * 24

    W_e_diag = [
        config.Q_c,
        config.Q_c,
        config.Q_c_z,  # e_c (3)
        config.Q_l,  # e_l (1)
        config.W_v_theta,  # v_theta (1)
        1.0,
        1.0,
        1.0,  # rpy (3)
        1.0,
        1.0,
        1.0,  # vel (3)
        5.0,
        5.0,
        5.0,  # drpy (3)
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
    yref[4] = v_ref
    yref[17] = parameters["mass"] * -parameters["gravity_vec"][-1]

    yref_e = np.zeros(ny_e)
    yref_e[4] = v_ref

    ocp.cost.yref = yref
    ocp.cost.yref_e = yref_e

    ocp.constraints.lbx = np.array([
        -config.MAX_ROLL_PITCH,
        -config.MAX_ROLL_PITCH,
        -config.MAX_YAW,
        config.MIN_V_THETA,
    ])
    ocp.constraints.ubx = np.array([
        config.MAX_ROLL_PITCH,
        config.MAX_ROLL_PITCH,
        config.MAX_YAW,
        config.MAX_V_THETA,
    ])
    ocp.constraints.idxbx = np.array([3, 4, 5, 13])

    ocp.constraints.lbu = np.array([
        -config.MAX_RPY_RATES,
        -config.MAX_RPY_RATES,
        -config.MAX_RPY_RATES,
        parameters["thrust_min"] * 4,
        -config.MAX_DELTA_V_THETA,
    ])
    ocp.constraints.ubu = np.array([
        config.MAX_RPY_RATES,
        config.MAX_RPY_RATES,
        config.MAX_RPY_RATES,
        parameters["thrust_max"] * 4,
        config.MAX_DELTA_V_THETA,
    ])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

    ocp.constraints.x0 = np.zeros(nx)
    ocp.parameter_values = np.zeros(np_dim)

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
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

        dt_fine = 1 / config.env.freq
        dt_coarse = self.mpcc_config.dt_coarse
        self._time_steps = np.concatenate(
            [
                np.full(self.mpcc_config.N_fine, dt_fine),
                np.full(self.mpcc_config.N_coarse, dt_coarse),
            ]
        )
        self._N = len(self._time_steps)
        self._shooting_nodes = np.concatenate(([0.0], np.cumsum(self._time_steps)))

        self.planner = SfcCorridorPlanner(obs, config.env.freq, self.planner_config)
        self._update_spline()

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
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

        self._tick = 0
        self._config = config
        self._finished = False

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
        """Update the path progress parameter `theta` based on current drone position."""
        s_eval = np.linspace(
            max(0, self._current_theta - 0.5), min(self._s_total, self._current_theta + 0.5), 20
        )
        dists = np.linalg.norm(self._des_pos_spline(s_eval) - pos, axis=1)
        self._current_theta = float(s_eval[np.argmin(dists)])

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
                            [theta_guess],
                            [v_start_guess],
                        )
                    )
                self._acados_ocp_solver.set(j, "x", x_guess)

    def _set_mpc_horizon_parameters(self, capsule_params: np.ndarray) -> None:
        """Set the reference parameters and cost weights along the MPC horizon."""
        gates_pos = getattr(self.planner, "gates_pos", None)
        target_gate_idx = getattr(self.planner, "target_gate_idx", -1)
        for j in range(self._N + 1):
            x_j = self._acados_ocp_solver.get(j, "x")
            theta_j = float(x_j[12])
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
            if gates_pos is not None and 0 <= target_gate_idx < len(gates_pos):
                target_gate_pos = gates_pos[target_gate_idx]
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
        if replanned:
            self._update_spline()
            self._current_theta = 0.0

        if self._current_theta >= self._s_total - 0.1:
            self._finished = True

        self._update_current_theta(obs["pos"])

        ref_points = np.array(
            [
                self._des_pos_spline(
                    np.clip(self._current_theta + t_node * self._current_v_theta, 0, self._s_total)
                )
                for t_node in self._shooting_nodes
            ]
        )

        capsule_params, _, min_obs_dist = self._extract_closest_obstacles(ref_points)

        if self._tick > 0 and not replanned:
            x_prev = self._acados_ocp_solver.get(1, "x")
            self._current_theta = float(x_prev[12])
            self._current_v_theta = max(0.0, float(x_prev[13]))

        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])

        x0 = np.concatenate(
            (
                obs["pos"],
                obs["rpy"],
                obs["vel"],
                obs["drpy"],
                [self._current_theta],
                [self._current_v_theta],
            )
        )

        self._warm_start_solver(x0, replanned, obs)
        self._set_mpc_horizon_parameters(capsule_params)

        dist_to_target_gate = -1.0
        gates_pos = getattr(self.planner, "gates_pos", None)
        target_gate_idx = getattr(self.planner, "target_gate_idx", -1)
        if gates_pos is not None and 0 <= target_gate_idx < len(gates_pos):
            dist_to_target_gate = np.linalg.norm(gates_pos[target_gate_idx] - obs["pos"])

        status = self._acados_ocp_solver.solve()
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

        if status not in [0, 2]:
            logger.warning(f"MPCC solver failed with status {status}. Entering hover mode.")
            return self._compute_hover_control(obs)
        else:
            if hasattr(self, "_hover_pos"):
                del self._hover_pos

        u0 = self._acados_ocp_solver.get(0, "u")
        return u0[0:4]

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
        self._tick += 1
        return self._finished

    def episode_callback(self) -> None:
        """Reset the controller state and variables at the start of a new episode."""
        self._tick = 0
        self._current_theta = 0.0
        self._current_v_theta = 0.5
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

        traj_history = self.planner.get_trajectory_history()
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
