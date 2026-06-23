"""Configuration settings for the Tube Path Planner."""

from dataclasses import dataclass


@dataclass
class TubePlannerConfig:
    """Configuration for the Tube Path Planner."""

    # --- Tube Planner Parameters ---
    tube_radius: float = 1.0
    """Maximum radius (m) of the flight corridor tube around the path."""

    gate_tube_radius: float = 0.4
    """Radius (m) of the flight corridor tube exactly at the gate center."""

    gate_radius_transition_dist: float = 2.0
    """Distance (m) from the gate to start reducing the tube radius to gate_tube_radius."""

    base_anchor_dist: float = 0.5
    """Base distance (m) to place anchor points before and after a gate."""

    min_anchor_dist: float = 0.1
    """Minimum distance (m) to place anchor points."""

    anchor_dist_scaling: float = 3.0
    """Scaling factor for anchor distance based on distance to the gate."""

    max_swing_dist: float = 1.5
    """Maximum lateral swing distance (m) for U-turns."""

    min_swing_dist: float = 0.5
    """Minimum lateral swing distance (m) for U-turns."""

    swing_dist_scaling: float = 0.4
    """Scaling factor for lateral swing distance based on distance to the gate."""

    obstacle_avoidance_iterations: int = 5
    """Number of iterations for the fast obstacle avoidance heuristic."""

    obstacle_clearance_margin: float = 0.1
    """Extra margin (m) to add on top of the required obstacle avoidance distance."""

    min_z_height: float = 0.15
    """Minimum allowed Z height (m) to prevent ground collisions."""

    spline_samples_multiplier: int = 10
    """Multiplier for the number of spline samples per control point."""

    min_spline_samples: int = 100
    """Minimum number of samples for the dense spline evaluation."""

    visualization_downsample_factor: int = 5
    """Downsample factor for drawing the control points in MuJoCo to avoid geom limits."""
