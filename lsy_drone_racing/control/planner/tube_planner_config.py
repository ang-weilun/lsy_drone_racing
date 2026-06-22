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
