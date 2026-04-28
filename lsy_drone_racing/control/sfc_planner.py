"""Pure SFC trajectory planner — extracted from sfc_controller.py.

Provides the path-skeleton builder, capsule obstacle model, flight-corridor
builder, and B-spline optimization. Consumed by both `sfc_controller.py`
(state-mode tracker) and `sfc_attitude_controller.py` (attitude-mode tracker).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class SkeletonPoint(NamedTuple):
    """Represents a skeleton point in the planned path with gate information."""

    pos: NDArray
    is_gate: bool
    gate_normal: NDArray | None
    gate_right: NDArray | None
    gate_up: NDArray | None


class Capsule(NamedTuple):
    """Represents a capsule obstacle (cylinder with spherical ends)."""

    p1: NDArray
    p2: NDArray
    radius: float
    is_gate: bool
    gate_idx: int | None = None


class FlightCorridor:
    """Represents a convex polyhedron (flight corridor) defined by half-spaces."""

    def __init__(self, p1: NDArray, p2: NDArray) -> None:
        """Initialize a flight corridor between two waypoints.

        Args:
            p1: Start point of the corridor.
            p2: End point of the corridor.
        """
        self.A = []
        self.b = []
        self.p1 = p1
        self.p2 = p2

        # Bounding box (Room limits)
        self.add_halfspace(np.array([0, 0, 1]), np.array([0, 0, 3.0]))
        self.add_halfspace(np.array([0, 0, -1]), np.array([0, 0, -0.2]))
        self.add_halfspace(np.array([1, 0, 0]), np.array([15.0, 0, 0]))
        self.add_halfspace(np.array([-1, 0, 0]), np.array([-15.0, 0, 0]))
        self.add_halfspace(np.array([0, 1, 0]), np.array([0, 15.0, 0]))
        self.add_halfspace(np.array([0, -1, 0]), np.array([0, -15.0, 0]))

    def add_halfspace(self, n: NDArray, p: NDArray):
        """Adds a constraint (x - p) * n <= 0, where n is the OUTWARD normal."""
        self.A.append(n)
        self.b.append(np.dot(n, p))


def closest_points_segments(
    p1: NDArray, q1: NDArray, p2: NDArray, q2: NDArray
) -> tuple[NDArray, NDArray]:
    """Finds the closest points c1 on segment p1-q1 and c2 on segment p2-q2."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)

    if a <= 1e-6 and e <= 1e-6:
        return p1, p2
    if a <= 1e-6:
        s = np.clip(f / e, 0.0, 1.0)
        return p1, p2 + s * d2

    c = np.dot(d1, r)
    if e <= 1e-6:
        t = np.clip(-c / a, 0.0, 1.0)
        return p1 + t * d1, p2

    b = np.dot(d1, d2)
    denom = a * e - b * b

    if denom != 0.0:
        t = np.clip((b * f - c * e) / denom, 0.0, 1.0)
    else:
        t = 0.0

    s = (b * t + f) / e
    if s < 0.0:
        s = 0.0
        t = np.clip(-c / a, 0.0, 1.0)
    elif s > 1.0:
        s = 1.0
        t = np.clip((b - c) / a, 0.0, 1.0)

    return p1 + t * d1, p2 + s * d2
