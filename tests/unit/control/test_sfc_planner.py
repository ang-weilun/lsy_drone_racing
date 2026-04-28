"""Tests for the extracted SFC planner module."""

from __future__ import annotations

import numpy as np

from lsy_drone_racing.control.sfc_planner import (
    Capsule,
    FlightCorridor,
    SkeletonPoint,
    closest_points_segments,
)


def test_closest_points_segments_simple():
    # Two parallel segments along x at y=0 and y=1; closest points are
    # endpoints of the perpendicular between them at (0, 0, 0) and (0, 1, 0).
    p1 = np.array([0.0, 0.0, 0.0])
    q1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    q2 = np.array([1.0, 1.0, 0.0])
    c1, c2 = closest_points_segments(p1, q1, p2, q2)
    assert np.allclose(c2 - c1, [0.0, 1.0, 0.0])


def test_skeleton_point_is_namedtuple():
    pt = SkeletonPoint(np.array([0.0, 0.0, 0.0]), False, None, None, None)
    assert pt.is_gate is False
    assert pt.gate_normal is None


def test_capsule_default_gate_idx():
    cap = Capsule(np.array([0.0, 0, 0]), np.array([0.0, 0, 1]), 0.05, False)
    assert cap.gate_idx is None


def test_flight_corridor_initializes_room_bounds():
    corr = FlightCorridor(np.array([0.0, 0, 1.0]), np.array([1.0, 0, 1.0]))
    # Room limits add 6 half-spaces (top/bottom + 4 walls)
    assert len(corr.A) == 6
    assert len(corr.b) == 6
