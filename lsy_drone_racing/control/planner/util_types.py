"""Common types used by the trajectory planner."""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SkeletonPoint(NamedTuple):
    """Represents a skeleton point in the planned path with gate information."""
    pos: NDArray
    is_gate: bool
    gate_normal: NDArray | None
    gate_right: NDArray | None
    gate_up: NDArray | None
    gate_idx: int | None = None
    is_in_tube: bool = False
    is_waypoint: bool = False

class Capsule(NamedTuple):
    """Represents a capsule obstacle (cylinder with spherical ends)."""
    p1: NDArray
    p2: NDArray
    radius: float
    is_gate: bool
    gate_idx: int | None = None

class FlightCorridor:
    """Represents a convex polyhedron (flight corridor) defined by half-spaces."""
    def __init__(self, p1: NDArray, p2: NDArray, limit_low: NDArray, limit_high: NDArray) -> None:
        """Initialize the FlightCorridor with bounding box limits.

        Args:
            p1: The start point of the corridor segment.
            p2: The end point of the corridor segment.
            limit_low: The lower bounds of the environment.
            limit_high: The upper bounds of the environment.
        """
        self.A = []
        self.b = []
        self.p1 = p1
        self.p2 = p2

        self.add_halfspace(np.array([0, 0, 1]), np.array([0, 0, limit_high[2]]))
        self.add_halfspace(np.array([0, 0, -1]), np.array([0, 0, limit_low[2]]))
        self.add_halfspace(np.array([1, 0, 0]), np.array([limit_high[0], 0, 0]))
        self.add_halfspace(np.array([-1, 0, 0]), np.array([limit_low[0], 0, 0]))
        self.add_halfspace(np.array([0, 1, 0]), np.array([0, limit_high[1], 0]))
        self.add_halfspace(np.array([0, -1, 0]), np.array([0, limit_low[1], 0]))

    def add_halfspace(self, n: NDArray, p: NDArray) -> None:
        """Add a bounding halfspace to the corridor.

        Args:
            n: The normal vector of the halfspace.
            p: A point on the halfspace boundary.
        """
        self.A.append(n)
        self.b.append(np.dot(n, p))
