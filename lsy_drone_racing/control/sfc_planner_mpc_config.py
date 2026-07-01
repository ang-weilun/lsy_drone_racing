"""Configuration settings for the SFC B-Spline Path Planner."""

from dataclasses import dataclass, field


@dataclass
class PlannerConfig:
    """Configuration for the SFC B-Spline Path Planner."""

    # --- Optimizer Weights ---
    W_VEL: float = 2.0
    """Penalty on velocity (first derivative of control points)."""

    W_ACC: float = 6.0
    """Penalty on acceleration (second derivative of control points). Smooths the path."""

    W_JERK: float = 10.0
    """Penalty on jerk (third derivative of control points).
    Minimizes sudden changes in curvature.
    """

    W_CENTER: float = 0.01
    """Soft penalty pulling control points towards their initial heuristic reference."""

    W_GATE_ALIGN: float = 30.0
    """Soft penalty enforcing lateral alignment with the gate normal axis for entry/exit."""

    # --- Gate Tube Enforcement ---
    GATE_TUBE_RADIUS: float = 0.18
    """Inscribed lateral fence (m) on control points from gate-normal axis."""

    GATE_TUBE_HALF_LENGTH: float = 0.5
    """Axial fence (m) on control points from the gate centre (max length)."""

    GATE_TUBE_AXIAL_MIN: float = 0.0
    """Minimum axial distance (m), signed by side. 0 = no min (just sign convention)."""

    GATE_TUBE_N_FACETS: int = 8
    """Number of polyhedral facets approximating the lateral cylinder."""

    # --- Replanning ---
    REPLAN_DEBOUNCE_TICKS: int = 5
    """Minimum number of ticks to wait before allowing another replan."""

    # --- Point-Mass (PMM) Reference Generator ---
    pmm_enabled: bool = True
    """Enable the point-mass time-optimal reference generator.

    When True, ``_build_spline`` replaces the skeleton/QP pipeline with a path
    fitted to :func:`plan_pmm_path`. When False, falls back to the legacy
    skeleton + CasADi gate-pin behavior. Overridable at runtime via the
    ``MPCC_PMM_ENABLED`` env var.
    """

    a_max: float = 14.0
    """Point-mass acceleration bound (m/s^2). Must stay >= ~7.6 for trackability."""

    gate_horizon: int = 4
    """Number of upcoming gates the PMM plans over per replan."""

    pmm_n_dir: int = 5
    """Number of cone directions sampled for the gate velocity search."""

    pmm_n_mag: int = 3
    """Number of speed magnitudes sampled for the gate velocity search."""

    pmm_v_lo: float = 1.0
    """Lower bound (m/s) of the sampled gate-velocity speed range."""

    pmm_v_hi: float = 5.5
    """Upper bound (m/s) of the sampled gate-velocity speed range."""

    pmm_half_angle: float = 0.5
    """Cone half-angle (rad) for the gate velocity samples."""

    pmm_n_per_seg: int = 30
    """Number of samples per PMM segment used to build the fitted path."""

    pmm_pole_margin: float = 0.10
    """Clearance margin (m) for the PMM reference placement and corridor poles.

    Smaller than the load-bearing ``safety_margin`` (0.15) on purpose: the PMM
    reference only needs to fit a perfectly-tracked point-mass body past a pole
    (pole radius 0.015 + this margin). The tracking-error buffer is supplied
    separately by the MPCC obstacle barrier, which keeps the full
    ``safety_margin`` on its capsules (``self.capsules``).
    """

    pmm_gate_inset: float = 0.10
    """Inset (m) from the gate inner edge for the aperture waypoint.

    Keeps the aperture-waypoint search within
    ``+/-(gate_inner/2 - pmm_gate_inset)`` of the gate center so the drone
    body half-width clears the gate-frame inner edge.
    """

    pmm_gate_anchor_gap: float = 0.0
    """Axial half-separation (m) of pre/post-gate crossing anchors along the
    gate normal (0.0 = off, legacy single-waypoint crossing).

    When > 0, each gate's aperture waypoint expands into three colinear
    waypoints along the gate normal (entry, center, exit) to force a robust
    one-way crossing with real +normal velocity. Mutually exclusive with
    ``pmm_cross_v_n_min``; kept below ``anchor_gap`` (0.9) to avoid fouling a
    pole behind/ahead of the gate.
    """

    pmm_cross_normal_weight: float = 0.75
    """Blend of the crossing-velocity cone axis between the next-gate direction
    and the gate normal, in [0, 1]. 0.0 reproduces the legacy equal-blend
    behavior (``normalize(next_dir + normal)``); 1.0 crosses perpendicular to
    the gate. Only active when ``pmm_cross_v_n_min > 0``.
    """

    pmm_cross_v_n_min: float = 2.0
    """Minimum gate-normal velocity component (m/s) required of the PMM
    crossing velocity at each gate (0.0 = unconstrained legacy crossing).

    Structural alternative to ``pmm_gate_anchor_gap``: constrains the sampled
    velocity at each gate node, ``dot(v, gate_normal) >= pmm_cross_v_n_min``,
    so the body carries enough +normal momentum for the env's crossing check
    to register under tracking error. Mutually exclusive with
    ``pmm_gate_anchor_gap`` (set that to 0 when this is > 0).
    """

    pmm_finish_ext_dist: float = 0.3
    """Finish-line extension (m) past the final gate along its normal (0.0 = off).

    The PMM waypoint list otherwise terminates at the last gate's aperture
    point, so the tracked body can stall short of the plane and the env's
    gate-pass crossing check never fires. When > 0, one extra waypoint is
    appended past the last gate so the reference drives the body through,
    mirroring the legacy SFC ``FINISH_LINE_EXT_DIST``.
    """

    pmm_takeoff_alt: float = 0.6
    """Climb-to altitude (m) for the takeoff waypoint.

    A ground start below an elevated gate 0 makes the point-mass min-time path
    skim the floor before climbing. Inserting a near-vertical climb waypoint at
    this altitude (clamped to the first gate's altitude) gains clearance first.
    """

    pmm_takeoff_eps: float = 0.15
    """Ground-start threshold (m) for inserting the takeoff climb waypoint.

    The climb waypoint is inserted only when the start altitude is at least this
    far below ``climb_z`` (the clamped takeoff altitude), so once airborne,
    mid-flight replans do not add it.
    """

    # --- Geometry / Spline Builder ---
    anchor_gap: float = 0.9
    """Distance (m) from gate to place entry/exit anchors along the normal."""

    exit_tangent_blend: float = 0.0
    """Lateral-turn factor ``beta >= 0`` for the post-gate skeleton waypoint.

    ``0.0`` places the post-gate waypoint straight out along the gate normal
    (perpendicular exit). ``beta > 0`` adds a lateral turn toward the next
    gate center while preserving the forward component, so the exit never
    bends back across the gate plane. The immediate gate crossing itself
    stays perpendicular (hard center pin + tube). The last gate is
    unaffected (it heads to the finish line).
    """

    points_per_segment: int = 4
    """Number of B-spline control points per inter-gate segment."""

    safety_margin: float = 0.15
    """Extra collision margin (m) added to obstacle and gate radii."""

    # --- Drone/Track Physical Dimensions ---
    gate_outer: float = 0.72
    """Outer width (m) of the gate frame."""

    gate_inner: float = 0.40
    """Inner width (m) of the gate frame opening."""

    gate_depth: float = 0.10
    """Depth (m) of the gate frame."""

    pole_radius: float = 0.015
    """Radius (m) of pole obstacles."""

    pole_height: float = 1.52
    """Height (m) of pole obstacles."""

    # --- Corridor Limits & Buffer ---
    CORRIDOR_LIMIT_LOW: list[float] = field(default_factory=lambda: [-3.5, -3.5, 0.0])
    """Lower boundaries [x, y, z] (m) for the path clipping bounding box."""

    CORRIDOR_LIMIT_HIGH: list[float] = field(default_factory=lambda: [3.5, 3.5, 3.0])
    """Upper boundaries [x, y, z] (m) for the path clipping bounding box."""

    ROOM_LIMIT_LOW: list[float] = field(default_factory=lambda: [-15.0, -15.0, -0.2])
    """Lower boundaries [x, y, z] (m) for the convex polyhedron corridor constraints."""

    ROOM_LIMIT_HIGH: list[float] = field(default_factory=lambda: [15.0, 15.0, 3.0])
    """Upper boundaries [x, y, z] (m) for the convex polyhedron corridor constraints."""

    CORRIDOR_BUFFER: float = 0.10
    """Buffer distance (m) subtracted/added to bounds when clipping path points."""

    # --- CasADi Optimization Solver Constants ---
    MAX_CTRL: int = 80
    """Maximum number of B-spline control points to optimize in CasADi QP solver."""

    MAX_PLANES: int = 25
    """Maximum number of linear inequality halfspaces /
    separating planes allowed per control point.
    """

    IPOPT_MAX_ITER: int = 100
    """Maximum iterations allowed for the IPOPT solver."""

    IPOPT_TOL: float = 1e-4
    """Tolerance for convergence in IPOPT solver."""

    IPOPT_ACCEPTABLE_TOL: float = 1e-3
    """Acceptable tolerance for IPOPT solver before termination under slow progress."""

    # --- Optimization Objective Weights ---
    W_P0_REF: float = 10.0
    """Penalty on initial control point's offset from the current drone position."""

    W_P1_REF_HIGH: float = 50.0
    """Velocity matching penalty weight on the second control point when drone speed > 0.1 m/s."""

    W_P1_REF_LOW: float = 10.0
    """Velocity matching penalty weight on the second control point when drone speed <= 0.1 m/s."""

    W_GATE_HARD: float = 1e5
    """Constraint penalty multiplier forcing gate-control points to match gate centers exactly."""

    # --- Heuristics, Swing Waypoints & Exit Mechanics ---
    PREV_GATE_EXIT_THRESHOLD: float = 0.4
    """Maximum distance (m) along gate normal within which the
    previous gate's clearance is maintained.
    """

    FINISH_LINE_EXT_DIST: float = 0.75
    """Extension distance (m) along the final gate normal to place the finish line waypoint."""

    HERMITE_TANGENT_SCALE_GATE: float = 1.0
    """Scaling factor for the gate normal vector when used as a tangent for Hermite splines."""

    HERMITE_TANGENT_SCALE_DRONE: float = 0.5
    """Scaling factor for the drone velocity vector when used as a tangent for Hermite splines."""

    HERMITE_SAMPLES_PER_SEGMENT: int = 5
    """Number of intermediate samples to take along the Hermite curve between anchors."""

    # --- Obstacle Avoidance Heuristics ---
    OBSTACLE_AVOIDANCE_MARGIN: float = 0.15
    """Buffer radius (m) added to obstacles (poles and gate rims)
    in 2D collision avoidance checks.
    """

    OBSTACLE_AVOIDANCE_PUSH_EXTRA: float = 0.20
    """Push offset distance (m) beyond obstacle radius to place the generated avoidance waypoint."""

    OBSTACLE_AVOIDANCE_MIN_DIST: float = 0.2
    """Minimum displacement (m) from previous and current path
    points to accept a new avoidance waypoint.
    """

    JITTER_THRESHOLD: float = 0.001
    """Distance shift (m) above which moving obstacles or gates trigger a path replan."""

    # --- Gate Frame Details ---
    gate_stand_radius: float = 0.05
    """Radius (m) of the vertical stands holding the gate frame."""

    gate_bar_dist: float = 0.28
    """Distance (m) from gate center to the vertical/horizontal
    border bars of the frame.
    """

    gate_bar_radius: float = 0.08
    """Radius (m) of the gate border bars."""
