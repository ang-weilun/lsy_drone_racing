"""SFC MPC Corridor Controller.

Uses SfcCorridorPlanner to generate safe flight corridors and reference trajectories.
Tracks references using acados MPC with state (position/velocity/acceleration) control.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.sfc_planner_mpc import SfcCorridorPlanner

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def create_acados_model(parameters: dict) -> AcadosModel:
    """Create symbolic acados model from drone dynamics."""
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

    model = AcadosModel()
    model.name = "sfc_corridor_mpc"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U

    return model


def create_corridor_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Create acados OCP and solver for corridor tracking."""
    ocp = AcadosOcp()

    ocp.model = create_acados_model(parameters)

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.solver_options.N_horizon = N

    # Linear least squares cost on states and inputs
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # State weights: prioritize position and velocity tracking (scaled down)
    Q = np.diag(
        [
            50.0,  # pos_x
            50.0,  # pos_y
            200.0,  # pos_z
            0.2,    # roll
            0.2,    # pitch
            0.2,    # yaw
            3.0,   # vel_x
            3.0,   # vel_y
            3.0,   # vel_z
            1.0,    # roll_rate
            1.0,    # pitch_rate
            1.0,    # yaw_rate
        ]
    )

    # Input weights: smooth attitude and thrust changes (increased for smoothness)
    R = np.diag(
        [
            1.0,   # roll_cmd
            1.0,   # pitch_cmd
            1.0,   # yaw_cmd
            10.0,  # thrust_cmd
        ]
    )

    Q_e = Q.copy()
    ocp.cost.W = np.eye(ny)
    ocp.cost.W[:nx, :nx] = Q
    ocp.cost.W[nx:, nx:] = R
    ocp.cost.W_e = np.eye(ny_e) * Q_e

    Vx = np.eye(ny, nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx:, :] = np.eye(nu)
    ocp.cost.Vu = Vu

    Vx_e = np.eye(ny_e, nx)
    ocp.cost.Vx_e = Vx_e

    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # Attitude constraints: ±30 degrees roll/pitch
    ocp.constraints.lbx = np.array([-0.8, -0.8, -0.5])
    ocp.constraints.ubx = np.array([0.8, 0.8, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    # Soften attitude constraints to prevent infeasibility
    ocp.constraints.idxsbx = np.array([0, 1, 2])
    ns = 3
    ocp.cost.Zl = 1000.0 * np.ones(ns)
    ocp.cost.Zu = 1000.0 * np.ones(ns)
    ocp.cost.zl = 100.0 * np.ones(ns)
    ocp.cost.zu = 100.0 * np.ones(ns)

    # Thrust constraints
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # Initial state
    ocp.constraints.x0 = np.zeros(nx)

    # Solver options: improved numerical stability
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.tol = 1e-3
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_tol_stat = 1e-2  # Relaxed
    ocp.solver_options.qp_solver_tol_eq = 1e-2  # Relaxed
    ocp.solver_options.qp_solver_tol_ineq = 1e-2  # Relaxed
    ocp.solver_options.tf = Tf

    solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/sfc_corridor_mpc.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return solver, ocp


class SfcMpcCorridorController(Controller):
    """SFC + Acados MPC controller with corridor-based trajectory planning.

    Pipeline:
    1. SfcCorridorPlanner builds safe flight corridors through gates
    2. Planner generates smooth reference trajectory (position, velocity)
    3. Acados MPC tracks the reference trajectory
    4. Infrastructure ready for later velocity optimization and obstacle avoidance
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict) -> None:
        """Initialize the MPC corridor controller.

        Args:
            obs: Initial observation.
            info: Environment info.
            config: Configuration.
        """
        super().__init__(obs, info, config)

        self._dt = 1.0 / config.env.freq
        self._N = 25
        self._T_horizon = self._N * self._dt

        # Initialize planner
        self.planner = SfcCorridorPlanner(obs, config.env.freq)

        # Load drone parameters and create MPC solver
        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self.solver, self.ocp = create_corridor_ocp_solver(
            self._T_horizon, self._N, self.drone_params, verbose=False
        )

        self._nx = self.ocp.model.x.rows()
        self._nu = self.ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        self._finished = False
        self._tick = 0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute MPC control command.

        Args:
            obs: Current observation.
            info: Optional info.

        Returns:
            Attitude command [roll_des, pitch_des, yaw_des, thrust_des].
        """
        # Update corridor planner
        self.planner.update(obs)

        # Get current drone state
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], rpy, obs["vel"], drpy))

        # Set initial state constraint
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        # Set reference trajectory from planner
        for k in range(self._N):
            t_ref = k * self._dt
            pos_ref, vel_ref, acc_ref = self.planner.evaluate(t_ref)

            yref = np.zeros(self._ny)
            yref[0:3] = pos_ref
            yref[6:9] = vel_ref
            yref[15] = self._hover_thrust

            self.solver.set(k, "yref", yref)

        # Set terminal reference
        t_ref = self._N * self._dt
        pos_ref, vel_ref, acc_ref = self.planner.evaluate(t_ref)
        yref_e = np.zeros(self._ny_e)
        yref_e[0:3] = pos_ref
        yref_e[6:9] = vel_ref
        self.solver.set(self._N, "y_ref", yref_e)

        # Solve OCP with error handling
        try:
            status = self.solver.solve()
            if status != 0:
                logger.warning(f"Solver status {status}, falling back to previous solution")
        except Exception as e:
            logger.warning(f"Solver failed: {e}, using fallback control")
            return np.array([0.0, 0.0, 0.0, self._hover_thrust])

        u0 = self.solver.get(0, "u")
        return u0.flatten()

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Callback after each simulation step.

        Args:
            action: Applied action.
            obs: Latest observation.
            reward: Latest reward.
            terminated: Episode termination flag.
            truncated: Episode truncation flag.
            info: Additional info.

        Returns:
            Whether controller has finished.
        """
        self._tick += 1
        self.planner.add_trajectory_point(obs["pos"])
        return self._finished

    def episode_callback(self):
        """Reset controller state at episode end."""
        self._tick = 0
        self._finished = False

    def render_callback(self, sim: object) -> None:
        """Render visualization: trajectory, reference, gates, velocity vectors.

        Args:
            sim: The simulator object.
        """
        from crazyflow.sim.visualize import draw_line, draw_points

        # Draw full planned trajectory (100 steps into future = ~2 seconds at 50 Hz)
        full_horizon = 100
        full_traj = self.planner.get_mpc_horizon_trajectory(full_horizon, self._dt)
        if len(full_traj) > 1:
            draw_line(
                sim,
                full_traj,
                rgba=np.array([0.2, 0.8, 0.2, 0.6]),
                start_size=0.005,
                end_size=0.005,
            )

        # Draw MPC horizon trajectory (next N steps in brighter green)
        mpc_horizon = self.planner.get_mpc_horizon_trajectory(self._N, self._dt)
        if len(mpc_horizon) > 1:
            draw_line(
                sim,
                mpc_horizon,
                rgba=np.array([0.0, 1.0, 0.0, 0.9]),
                start_size=0.008,
                end_size=0.008,
            )

        # Draw current reference point
        pos_ref = self.planner.current_pos_ref
        draw_points(sim, np.array([pos_ref]), rgba=np.array([1.0, 1.0, 0.0, 1.0]), size=0.015)

        # Draw drone trajectory history
        traj_history = self.planner.get_trajectory_history()
        if len(traj_history) > 1:
            draw_line(
                sim,
                traj_history,
                rgba=np.array([0.5, 0.5, 1.0, 0.6]),
                start_size=0.004,
                end_size=0.004,
            )

        # Draw velocity reference vector
        vel_ref = self.planner.current_vel_ref
        vel_magnitude = np.linalg.norm(vel_ref)
        if vel_magnitude > 0.01:
            vel_start = pos_ref
            vel_end = pos_ref + vel_ref
            draw_line(
                sim,
                np.array([vel_start, vel_end]),
                rgba=np.array([1.0, 0.5, 0.0, 1.0]),
                start_size=0.01,
                end_size=0.01,
            )

        # Draw gates
        if self.planner.gates_pos is not None and len(self.planner.gates_pos) > 0:
            draw_points(sim, self.planner.gates_pos, rgba=np.array([0.0, 0.0, 1.0, 0.8]), size=0.04)
