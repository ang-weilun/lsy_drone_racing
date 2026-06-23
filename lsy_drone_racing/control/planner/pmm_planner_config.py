from dataclasses import dataclass


@dataclass
class PmmPlannerConfig:
    """Configuration for the Point-Mass Model (PMM) Planner."""

    # Velocity Search Parameters
    s: int = 3
    """Number of bins for velocity norm, pitch, and yaw."""

    K: int = 2
    """Number of refocusing iterations."""

    epsilon: float = 1.01
    """Convergence criteria for time improvement."""

    gate_horizon: int = 4
    """Number of gates to look ahead for the Dijkstra search (Hg in the paper)."""

    # Base velocity sampling bounds
    v_min: float = 0.1
    v_max: float = 10.0

    # Point-mass acceleration limits (approximating drone thrust limits)
    # E.g., for a drone with T/W = 3.0, max accel is ~20 m/s^2.
    u_min: float = -10.0
    u_max: float = 10.0
