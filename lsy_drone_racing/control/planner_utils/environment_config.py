"""Configuration settings for the Environment."""

from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    """Configuration for the Environment (gate and obstacle sizes)."""

    # --- Geometry & Collision Margins ---
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

    # --- Gate Frame Details ---
    gate_stand_radius: float = 0.05
    """Radius (m) of the vertical stands holding the gate frame."""

    gate_bar_dist: float = 0.28
    """Distance (m) from gate center to the vertical/horizontal
    border bars of the frame.
    """

    gate_bar_radius: float = 0.08
    """Radius (m) of the gate border bars."""
