"""Defines the skeleton point used in trajectory planning."""

from typing import NamedTuple
from numpy.typing import NDArray

class SkeletonPoint(NamedTuple):
    """Represents a skeleton point in the planned path with gate information."""

    pos: NDArray
    is_gate: bool
    gate_normal: NDArray | None
    gate_right: NDArray | None
    gate_up: NDArray | None
    gate_idx: int | None = None
