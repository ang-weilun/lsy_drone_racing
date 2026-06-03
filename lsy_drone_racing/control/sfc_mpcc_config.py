from dataclasses import dataclass, field

import numpy as np


@dataclass
class MPCCConfig:
    """Configuration for the Model Predictive Contouring Controller (MPCC)."""

    # --- Horizon Parameters ---
    N_fine: int = 20
    """Number of fine time steps at the beginning of the horizon."""

    N_coarse: int = 15
    """Number of coarse time steps at the end of the horizon."""

    dt_coarse: float = 0.05
    """Time step duration for the coarse horizon steps (s)."""

    # --- Cost Weights ---
    Q_c: float = 150.0
    """Contouring error penalty (lateral/longitudinal). Higher values force the drone to stay strictly on path."""

    Q_c_z: float = 400.0
    """Vertical contouring error penalty. Often higher than Q_c to prevent altitude drops."""

    Q_l: float = 150.0
    """Lag error penalty. Penalizes falling behind or rushing ahead of the virtual reference point."""

    W_v_theta: float = 5.0
    """Penalty on virtual velocity. Higher values penalize high virtual velocities."""

    obstacle_penalty: float = 100000.0
    """Penalty for entering the obstacle barrier."""

    # --- Reference Parameters ---
    mu: float = 15.0
    """Speed scaling parameter. Reference virtual velocity is v_ref = mu / W_v_theta."""

    # --- Dynamic Tuning ---
    dynamic_addition: float = 300.0
    """Additional contouring penalty weight added near target gates."""

    dynamic_sigma: float = 0.3
    """Standard deviation (m) for the Gaussian dynamic weight addition near gates."""

    # --- Hover Controller Gains ---
    hover_kp: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.4, 1.25]))
    """Proportional gains for the hover fallback controller [x, y, z]."""

    hover_kd: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.2, 0.4]))
    """Derivative gains for the hover fallback controller [x, y, z]."""
