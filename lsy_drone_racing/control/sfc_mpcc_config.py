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

    mu_progress: float = 50.0
    """Linear progress reward weight: cost includes ``-mu_progress * v_theta`` so the
    solver maximizes path progress directly (classic MPCC). Speed is capped by
    ``MAX_V_THETA``, not this weight; trades off against contour/lag cost."""

    obstacle_penalty: float = 15000.0
    """Penalty for entering the obstacle barrier."""

    # --- Smoothness Penalties ---
    Q_rpy: float = 10.0
    """Attitude (roll, pitch, yaw) penalty. Kept low so it doesn't fight cornering bank."""

    Q_drpy: float = 5.0
    """Penalty on roll/pitch/yaw body rates. Keep low: too high overdamps and the
    controller won't maneuver hard enough to hit gates; too low lets agility clip
    randomized gates. No jitter cost here (``R_curv_rpy`` handles command chatter)."""

    R_curv_rpy: float = 1e-4
    """Penalty on rpy command curvature (2nd derivative); taxes high-frequency
    chatter, not magnitude or slope. Needed because the per-step attitude command
    sits in a near-flat QP valley, letting a single RTI step wander tick-to-tick into
    ~8-11 Hz command chatter. Too high stiffens the command and costs gate-reveal
    re-track authority; 1e-4 is the knee between smoothness and authority."""

    R_cmd_thrust: float = 100.0
    """Penalty on desired thrust command."""

    R_delta_v_theta: float = 0.5
    """Penalty on the virtual-progress acceleration input (smooths v_theta)."""

    Q_vel: float = 0.0
    """Linear-velocity penalty. Zero: cost references vel=0, so >0 rewards flying slow."""

    # --- Dynamic Tuning ---
    dynamic_addition: float = 1200.0
    """Extra contouring weight near target gates. Sized for the 0.7 m sensor reveal
    radius (gate true pose can snap up to ~0.2 m)."""

    dynamic_sigma: float = 0.8
    """Std (m) of the Gaussian near-gate weight. Widened to bite at the 0.7 m reveal radius."""

    # --- Contour tunnel (lateral corridor for emergent width) ---
    tunnel_w_gate: float = 0.12
    """Tunnel half-width (m) at a gate: pinched below aperture/2 to force the body onto
    the gate center and avoid frame clips. Too tight over-constrains and reintroduces
    clips from the opposite direction."""

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
    """Maximum virtual speed (m/s) allowed along the spline path. Relies on the
    contour tunnel to keep gate crossings centered at this speed; without it, success
    rate collapses at higher caps."""

    MAX_RPY_RATES: float = 0.9
    """Bound on the commanded attitude angles u[0:3] (rad). Misnomer kept for diff hygiene."""

    MAX_CMD_RPY_ACC: float = 3000.0
    """Bound on |rpy command acceleration| (rad/s^2), the command double-integrator
    input. Loose -- blocks single-step jumps; R_curv_rpy shapes smoothness.
    """

    MAX_DELTA_V_THETA: float = 2.0
    """Maximum rate of change of the virtual progress speed (m/s^2) / acceleration input."""

    # --- Curvature-coupled v_theta cap (default off) ---
    v_theta_curv_cap: bool = False
    """Slow the virtual progress through tight reference curves. ``v_theta`` is
    otherwise driven to ``MAX_V_THETA`` regardless of local curvature, which can
    demand lateral accel the drone can't hold on sharp turns. When enabled, each
    horizon stage's ``v_theta`` upper bound is set to
    ``clip(sqrt(v_theta_a_lat / kappa), v_theta_curv_floor, MAX_V_THETA)`` from a
    coarse curvature profile, so speed drops only through curves, not on straights."""

    v_theta_a_lat: float = 12.0
    """Lateral-acceleration budget (m/s^2) for the curvature-coupled v_theta cap."""

    v_theta_curv_floor: float = 1.2
    """Floor (m/s) on the curvature-coupled v_theta cap so the drone never crawls to a
    stall at a sharp gate turn (the cusp-stall failure mode)."""

    # --- Gate-proximity v_theta ease (default OFF: the speed-first submission) ---
    v_theta_gate_ease: bool = False
    """Ease the virtual progress speed as the reference approaches each gate.

    Distinct from the curvature cap: targets the virtual reference point racing
    through an oblique gate crossing faster than the body can track, so the body
    arrives still off-center and clips the frame or misses the pass. When enabled,
    each horizon stage's ``v_theta`` upper bound is eased toward ``v_theta_gate_min``
    by a Gaussian in arc-length distance to the nearest upcoming gate (width
    ``v_theta_gate_sigma``), reopening to ``MAX_V_THETA`` between gates. Composes
    with ``v_theta_curv_cap`` (per-stage min)."""

    v_theta_gate_min: float = 1.0
    """Eased ``v_theta`` upper bound (m/s) at a gate under ``v_theta_gate_ease``."""

    v_theta_gate_sigma: float = 0.6
    """Arc-length std (m) of the gate-approach v_theta ease Gaussian."""

    # --- Reference curvature smoothing (gate-pinned, L3 default ON) ---
    ref_smooth: bool = True
    """Smooth the tracked reference to distribute sharp gate turns over a longer arc.

    The PMM-fit reference concentrates each gate turn into a near-cusp at the gate
    waypoint, so tracking it at racing speed demands infeasible lateral accel. A
    FITPACK weighted smoothing spline (scipy ``splprep``) refits the tracked spline
    with high weight pinned within ``ref_gate_pin_dist`` of each gate center (so the
    crossing is preserved) and a smoothing target scaled by ``ref_smooth_tol``
    elsewhere, relaxing the turn cusp to a feasible-curvature arc."""

    ref_smooth_tol: float = 0.08
    """Allowed RMS deviation (m) of non-gate reference samples from the raw PMM path;
    larger => fewer spline knots => smoother => lower peak curvature."""

    ref_gate_pin_dist: float = 0.12
    """Arc-length half-window (m) around each gate center pinned at high weight so the
    smoothing never pulls the reference out of the gate aperture."""

    # --- Acados OCP Solver Settings ---
    SOLVER_TOL: float = 1e-3
    """Tolerance for convergence in the ACADOS OCP solver."""

    QP_SOLVER_ITER_MAX: int = 10
    """Maximum iterations allowed for the condensed QP solver in ACADOS."""

    NLP_SOLVER_MAX_ITER: int = 1
    """Maximum iterations allowed for the nonlinear SQP solver in ACADOS."""

    sqp_iters_per_step: int = 1
    """Real-time-iteration (SQP_RTI) steps run per control tick. Default 1 = classic
    single RTI (~2 ms of a 20 ms tick budget). Running more per tick converges the
    horizon further before committing the command, at the cost of solve time."""
