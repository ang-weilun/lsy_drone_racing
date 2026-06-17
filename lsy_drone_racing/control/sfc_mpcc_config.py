"""Configuration settings for the Model Predictive Contouring Controller (MPCC)."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MPCCConfig:
    """Configuration for the Model Predictive Contouring Controller (MPCC)."""

    # --- Horizon Parameters ---
    N_fine: int = 40
    """Number of fine time steps at the beginning of the horizon."""

    N_coarse: int = 0
    """Coarse steps at the end of the horizon (0 = uniform grid; beat the ramp in A/B)."""

    dt_fine: float = 0.02
    """Fine step duration (s). Must equal the control period 1/env.freq (RTI shift assumption)."""

    dt_coarse: float = 0.05
    """Coarse step duration (s); largest step under the "linear" schedule."""

    horizon_schedule: str = "two_block"
    """Step layout: "two_block" (N_fine x dt_fine then N_coarse x dt_coarse) or "linear" (ramp)."""

    # --- Cost Weights ---
    Q_c: float = 150.0
    """Contouring error penalty (lateral/longitudinal)."""

    Q_c_z: float = 400.0
    """Vertical contouring error penalty. Higher than Q_c to prevent altitude drops."""

    Q_l: float = 150.0
    """Lag error penalty (falling behind / rushing ahead of the virtual reference point)."""

    W_v_theta: float = 5.0
    """Penalty on virtual velocity."""

    obstacle_penalty: float = 5000.0
    """Penalty for entering the obstacle barrier."""

    # --- Reference Parameters ---
    mu: float = 20.0
    """Speed scaling; reference virtual velocity v_ref = mu / W_v_theta."""

    # --- Smoothness Penalties ---
    Q_rpy: float = 10.0
    """Attitude (roll, pitch, yaw) penalty. Kept low so it doesn't fight cornering bank."""

    Q_drpy: float = 20.0
    """Penalty on roll, pitch, yaw rates. Increase for smoother flight."""

    R_curv_rpy: float = 1e-4
    """Penalty on rpy command curvature (2nd derivative): taxes high-frequency
    chatter, not magnitude or slope. The per-step attitude command sits in a near-flat
    QP valley (slow attitude dynamics barely move over one 20 ms tick), so a single RTI
    step wanders tick-to-tick -> ~8-11 Hz command chatter (seen in sim and on the real
    drone). This penalty adds curvature to that flat direction so the step is pinned.
    A sweep showed jitter is NON-monotonic here: the old 1e-7 sat ON a resonance hump
    (jitterier than ~zero); >=1e-4 collapses it (command HF -86%, dominant freq 8-11 Hz
    -> ~cornering). Raising further (1e-3) is smoother still but stiffens the command
    enough to cost gate-reveal re-track authority on the randomized levels (level2 SR
    6/12 at 1e-4 vs 4/12 at 1e-3); 1e-4 is the knee. The L2 SR cost is a late-reveal
    artifact and does not apply on mocap hardware. See memory mpcc-jitter-curvature-tuning.
    """

    R_cmd_thrust: float = 100.0
    """Penalty on desired thrust command."""

    Q_vel: float = 0.0
    """Linear-velocity penalty. Zero: cost references vel=0, so >0 rewards flying slow."""

    # --- Dynamic Tuning ---
    dynamic_addition: float = 1200.0
    """Extra contouring weight near target gates. Sized for the 0.7 m sensor reveal
    (gate true pose snaps up to ~0.2 m): 300->1200 recovered L2 SR 5->14/20.
    """

    dynamic_sigma: float = 0.8
    """Std (m) of the Gaussian near-gate weight. Widened to bite at the 0.7 m reveal radius."""

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

    MAX_RPY_RATES: float = 0.9
    """Bound on the commanded attitude angles u[0:3] (rad). Misnomer kept for diff hygiene."""

    MAX_CMD_RPY_ACC: float = 3000.0
    """Bound on |rpy command acceleration| (rad/s^2), the command double-integrator
    input. Loose -- blocks single-step jumps; R_curv_rpy shapes smoothness.
    """

    MAX_DELTA_V_THETA: float = 2.0
    """Maximum rate of change of the virtual progress speed (m/s^2) / acceleration input."""

    # --- Acados OCP Solver Settings ---
    SOLVER_TOL: float = 1e-3
    """Tolerance for convergence in the ACADOS OCP solver."""

    QP_SOLVER_ITER_MAX: int = 10
    """Maximum iterations allowed for the condensed QP solver in ACADOS."""

    NLP_SOLVER_MAX_ITER: int = 1
    """Maximum iterations allowed for the nonlinear SQP solver in ACADOS."""
