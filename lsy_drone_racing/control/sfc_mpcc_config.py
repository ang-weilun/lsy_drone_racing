"""Configuration settings for the Model Predictive Contouring Controller (MPCC)."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MPCCConfig:
    """Configuration for the Model Predictive Contouring Controller (MPCC)."""

    # --- Planner Configuration ---
    planner_type: str = "pmm"
    """Which planner to use:
    'sfc' for CasADi B-spline SFC,
    'tube' for lightweight Tube SFC,
    'pmm' for Point-Mass Model."""

    TUBE_RADIUS: float = 1.0
    gate_tube_radius: float = 0.4
    """Radius (m) of the flight corridor tube around the path. Used when planner_type is 'tube'."""

    use_soft_tube_constraint: bool = False
    """If True, enforces a soft penalty for leaving the TUBE_RADIUS cylinder around the path."""

    # --- Horizon Parameters ---
    N: int = 50
    """Number of time steps in the horizon."""

    dt_min: float = 0.02
    """Time step duration for the first horizon step (s)."""

    dt_max: float = 0.1
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

    obstacle_penalty: float = 10000.0
    """Penalty for entering the obstacle barrier."""

    gate_margin_reduction: float = 0.12
    """Amount (m) by which to reduce the gate capsule radius in the obstacle penalty. 
    Prevents aggressive braking at gates by shrinking their effective avoidance margin."""

    # --- Soft Constraint Penalties ---
    TUBE_SOFT_PENALTY_L1: float = 10000.0
    """L1 penalty weight for soft tube constraint."""

    TUBE_SOFT_PENALTY_L2: float = 10000.0
    """L2 penalty weight for soft tube constraint."""

    STATE_BOUND_SOFT_PENALTY_L1: float = 10000.0
    """L1 penalty weight for soft state bounds (altitude, attitude)."""

    STATE_BOUND_SOFT_PENALTY_L2: float = 10000.0
    """L2 penalty weight for soft state bounds (altitude, attitude)."""

    # --- Reference Parameters ---
    mu: float = 10.0
    """Speed scaling parameter. Reference virtual velocity is v_ref = mu / W_v_theta."""

    # --- Smoothness Penalties ---
    Q_rpy: float = 10.0
    """Penalty on roll, pitch, yaw attitude."""

    Q_drpy: float = 10.0
    """Penalty on roll, pitch, yaw rates (angular velocity). Increase for smoother flight."""

    R_curv_rpy: float = 1e-4
    """Penalty on desired roll, pitch, yaw commands. Increase to limit aggressive control inputs."""

    R_cmd_thrust: float = 100.0
    """Penalty on desired thrust command."""

    Q_vel: float = 0.0
    """Penalty on linear velocity."""

    # --- Dynamic Tuning ---
    dynamic_addition: float = 1200.0
    """Additional contouring penalty weight added near target gates."""

    dynamic_sigma: float = 0.4
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

    unobserved_velocity_cap: float = 0.4
    """Velocity cap when approaching an unobserved gate or obstacle."""

    unobserved_dist_threshold: float = 0.7
    """Distance threshold to start capping velocity for an unobserved gate or obstacle."""

    MAX_RPY_RATES: float = 0.9
    """Maximum desired roll, pitch, and yaw rates (rad/s)."""

    MAX_CMD_RPY_ACC: float = 3000.0
    """Bound on |rpy command acceleration| (rad/s^2), the command double-integrator input."""

    MAX_DELTA_V_THETA: float = 2.0
    """Maximum rate of change of the virtual progress speed (m/s^2) / acceleration input."""

    # --- Acados OCP Solver Settings ---
    SOLVER_TOL: float = 1e-3
    """Tolerance for convergence in the ACADOS OCP solver."""

    QP_SOLVER_ITER_MAX: int = 10
    """Maximum iterations allowed for the condensed QP solver in ACADOS."""

    NLP_SOLVER_MAX_ITER: int = 1
    """Maximum iterations allowed for the nonlinear SQP solver in ACADOS."""
