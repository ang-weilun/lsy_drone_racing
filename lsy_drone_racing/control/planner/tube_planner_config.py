"""Configuration settings for the Tube Path Planner."""

from dataclasses import dataclass


@dataclass
class TubePlannerConfig:
    """Configuration for the Tube Path Planner."""

    # --- Tube Planner Parameters ---
    tube_radius: float = 1.0
    """Maximum radius (m) of the flight corridor tube around the path."""

    gate_tube_radius: float = 0.2
    """Radius (m) of the flight corridor tube exactly at the gate center (40cm opening -> 20cm radius)."""
