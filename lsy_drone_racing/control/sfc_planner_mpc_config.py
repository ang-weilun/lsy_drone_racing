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

    # --- Geometry / Spline Builder ---
    anchor_gap: float = 0.5
    """Distance (m) from gate to place entry/exit anchors along the normal."""

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
    PREV_GATE_EXIT_THRESHOLD: float = 1.0
    """Maximum distance (m) along gate normal within which the
    previous gate's clearance is maintained.
    """

    ENTRY_SWING_THRESHOLD: float = -0.1
    """Threshold on velocity-normal alignment below which the
    drone triggers entry swing (U-turn approach).
    """

    ENTRY_SWING_OFFSET: float = 0.5
    """Lateral offset distance (m) added to entry swing waypoints to widen approach turns."""

    EXIT_SWING_THRESHOLD: float = -0.2
    """Threshold on exit velocity-normal alignment below which
    exit swing (hairpin turnaround) is triggered.
    """

    EXIT_SWING_CLEARANCE: float = 1.0
    """Additional exit clearance distance (m) along the gate normal for turnaround anchors."""

    EXIT_SWING_SIDE_OFFSET: float = 1.0
    """Lateral offset distance (m) added to exit swing waypoints to define turnaround radius."""

    EXIT_SWING_BACK_OFFSET: float = 0.7
    """Axial backtracking offset distance (m) against the
    normal axis to stabilize hairpin exit turns.
    """

    FINISH_LINE_EXT_DIST: float = 0.75
    """Extension distance (m) along the final gate normal to place the finish line waypoint."""

    # --- Obstacle Avoidance Heuristics ---
    OBSTACLE_AVOIDANCE_MARGIN: float = 0.15
    """Buffer radius (m) added to obstacles (poles and gate rims)
    in 2D collision avoidance checks.
    """

    OBSTACLE_AVOIDANCE_PUSH_EXTRA: float = 0.20
    """Push offset distance (m) beyond obstacle radius to place the generated avoidance waypoint."""

    OBSTACLE_AVOIDANCE_MIN_DIST: float = 0.3
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
