from dataclasses import dataclass


@dataclass
class PlannerConfig:
    """Configuration for the SFC B-Spline Path Planner."""

    # --- Optimizer Weights ---
    W_VEL: float = 2.0
    """Penalty on velocity (first derivative of control points)."""

    W_ACC: float = 6.0
    """Penalty on acceleration (second derivative of control points). Smooths the path."""

    W_JERK: float = 10.0
    """Penalty on jerk (third derivative of control points). Minimizes sudden changes in curvature."""

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

    # --- TOPP (Variable Speed Schedule) Tunables ---
    V_MAX_GLOBAL: float = 1.5
    """Speed ceiling (m/s) on straight segments."""

    TILT_LIMIT_PLANNER: float = 0.5
    """Max tilt angle (rad) mirroring controller limits. Drives lateral acceleration max."""

    A_LONG_MAX_FACTOR: float = 0.53
    """Factor determining max longitudinal acceleration from max lateral acceleration."""

    V_FLOOR: float = 0.2
    """Floor on scheduled speed (m/s) to avoid division by zero in pathological curvature."""

    N_TOPP_SAMPLES: int = 200
    """Number of points to sample when building the TOPP velocity schedule."""

    # --- Geometry / Spline Builder ---
    anchor_gap: float = 0.5
    """Distance (m) from gate to place entry/exit anchors along the normal."""

    base_speed: float = 1.0
    """Fallback constant speed (m/s) if TOPP fails."""

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
