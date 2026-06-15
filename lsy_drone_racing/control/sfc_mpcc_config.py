"""Configuration settings for the Model Predictive Contouring Controller (MPCC)."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MPCCConfig:
    """Configuration for the Model Predictive Contouring Controller (MPCC)."""

    # --- Horizon Parameters ---
    N: int = 40
    """Number of time steps in the horizon."""

    dt_min: float = 0.02
    """Time step duration for the first horizon step (s)."""

    dt_max: float = 0.08
    """Time step duration for the last horizon step (s)."""

    # --- Cost Weights ---
    Q_c: float = 150.0
    """Contouring error penalty (lateral/longitudinal).
    Higher values force the drone to stay strictly on path.
    """

    Q_c_z: float = 400.0
    """Vertical contouring error penalty. Often higher than Q_c to prevent altitude drops."""

    Q_l: float = 150.0
    """Lag error penalty.
    Penalizes falling behind or rushing ahead of the virtual reference point.
    """

    W_v_theta: float = 5.0
    """Penalty on virtual velocity. Higher values penalize high virtual velocities."""

    obstacle_penalty: float = 5000.0
    """Penalty for entering the obstacle barrier."""

    # --- Reference Parameters ---
    mu: float = 10.0
    """Speed scaling parameter. Reference virtual velocity is v_ref = mu / W_v_theta."""

    # --- Smoothness Penalties ---
    Q_rpy: float = 50.0
    """Penalty on roll, pitch, yaw attitude."""

    Q_drpy: float = 20.0
    """Penalty on roll, pitch, yaw rates (angular velocity). Increase for smoother flight."""

    R_cmd_rpy: float = 50.0
    """Penalty on desired roll, pitch, yaw commands. Increase to limit aggressive control inputs."""

    R_cmd_thrust: float = 100.0
    """Penalty on desired thrust command."""

    Q_vel: float = 1.0
    """Penalty on linear velocity."""

    # --- Dynamic Tuning ---
    dynamic_addition: float = 300.0
    """Additional contouring penalty weight added near target gates."""

    dynamic_sigma: float = 0.3
    """Standard deviation (m) for the Gaussian dynamic weight addition near gates."""

    # --- Hover Controller Gains ---
    hover_kp: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.4, 1.25]))
    """Proportional gains for the hover fallback controller [x, y, z]."""

    hover_kd: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.4]))
    """Derivative gains for the hover fallback controller [x, y, z]."""

    # --- MPC State & Input Constraint Boundaries ---
    MAX_ROLL_PITCH: float = 1.2
    """Maximum roll and pitch attitude angle (rad) allowed inside MPC horizon."""

    MAX_YAW: float = np.pi
    """Maximum yaw attitude angle (rad) allowed inside MPC horizon."""

    MIN_V_THETA: float = 0.1
    """Minimum virtual speed (m/s) allowed along the spline path."""

    MAX_V_THETA: float = 3.0
    """Maximum virtual speed (m/s) allowed along the spline path."""

    unobserved_gate_velocity_cap: float = 0.7
    """Velocity cap when approaching an unobserved gate."""

    unobserved_gate_dist_threshold: float = 0.5
    """Distance threshold to start capping velocity for an unobserved gate."""

    MAX_RPY_RATES: float = 0.5
    """Maximum desired roll, pitch, and yaw rates (rad/s)."""

    MAX_DELTA_V_THETA: float = 2.0
    """Maximum rate of change of the virtual progress speed (m/s^2) / acceleration input."""

    # --- Acados OCP Solver Settings ---
    SOLVER_TOL: float = 1e-3
    """Tolerance for convergence in the ACADOS OCP solver."""

    QP_SOLVER_ITER_MAX: int = 10
    """Maximum iterations allowed for the condensed QP solver in ACADOS."""

    NLP_SOLVER_MAX_ITER: int = 1
    """Maximum iterations allowed for the nonlinear SQP solver in ACADOS."""
