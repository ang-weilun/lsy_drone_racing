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

    mu_progress: float = 20.0
    """Linear progress reward: the cost includes ``-mu_progress * v_theta`` so the
    solver maximizes path progress directly (classic MPCC), instead of tracking an
    unreachable virtual-speed setpoint. The marginal value of speed is constant (the
    old least-squares pull weakened as v_theta approached the setpoint). Top speed is
    bounded by ``MAX_V_THETA``, not by this weight. The achieved speed is traded off
    against contour/lag cost online; sweep this to move along that tradeoff.
    """

    obstacle_penalty: float = 5000.0
    """Penalty for entering the obstacle barrier."""

    # --- Smoothness Penalties ---
    Q_rpy: float = 10.0
    """Attitude (roll, pitch, yaw) penalty. Kept low so it doesn't fight cornering bank."""

    Q_drpy: float = 5.0
    """Penalty on roll/pitch/yaw body rates. Lowered 20 -> 5: a time-optimal-oracle cost
    diagnosis showed body-rate damping was the dominant speed limiter -- at 20 its per-tick
    penalty exceeded the entire progress reward, so the controller would not maneuver. 5 is
    the L3 optimum from a paired so_rpy_rotor_drag seed=42 sweep (L3 48->50% SR @ 6.38->5.58s,
    L2 75->83% @ 5.16->4.90s); below 5 the extra agility clips randomized L3 gates (44% at 2,
    38% at 1), 20 over-damps. No jitter cost (R_curv_rpy handles command chatter)."""

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

    R_delta_v_theta: float = 0.5
    """Penalty on the virtual-progress acceleration input (smooths v_theta)."""

    Q_vel: float = 0.0
    """Linear-velocity penalty. Zero: cost references vel=0, so >0 rewards flying slow."""

    # --- Dynamic Tuning ---
    dynamic_addition: float = 1200.0
    """Extra contouring weight near target gates. Sized for the 0.7 m sensor reveal
    (gate true pose snaps up to ~0.2 m): 300->1200 recovered L2 SR 5->14/20.
    """

    dynamic_sigma: float = 0.8
    """Std (m) of the Gaussian near-gate weight. Widened to bite at the 0.7 m reveal radius."""

    # --- Contour tunnel (lateral corridor for emergent width) ---
    tunnel_w_gate: float = 0.18
    """Tunnel half-width (m) at a gate: pinched to ~aperture/2 to keep the crossing centered."""

    tunnel_w_wide: float = 0.6
    """Tunnel half-width (m) between gates: opened so a wider/faster line can emerge."""

    tunnel_sigma: float = 0.55
    """Arc-length std (m) of the pinch: how sharply the tunnel narrows toward a gate."""

    TUNNEL_SLACK_LIN: float = 20000.0
    """Linear slack penalty on leaving the tunnel (one-sided, lower bound)."""

    TUNNEL_SLACK_QUAD: float = 80000.0
    """Quadratic slack penalty on leaving the tunnel."""

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

    MAX_V_THETA: float = 3.5
    """Maximum virtual speed (m/s) allowed along the spline path. Raised 3.0 -> 3.5:
    with the contour tunnel centering the line at gates, the higher speed cap no longer
    collapses success rate the way it did on the bare controller (a cap x tunnel sweep
    gave L2 11/12 at 3.5 vs the old SR loss at >=3.5 without the tunnel). The tunnel
    keeps the crossing centered; the cap lets the drone carry more speed between gates.
    """

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
