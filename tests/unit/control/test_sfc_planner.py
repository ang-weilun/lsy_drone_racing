"""Tests for the extracted SFC planner module."""

from __future__ import annotations

import numpy as np
import pytest

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


def _minimal_obs():
    """Obs fixture matching what level 0 emits at t=0: 4 gates, 4 obstacles, drone at start."""
    return {
        "pos": np.array([-1.5, 0.75, 0.05]),
        "vel": np.array([0.0, 0.0, 0.0]),
        "quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "gates_pos": np.array(
            [
                [0.5, 0.25, 0.7],
                [1.05, 0.75, 1.2],
                [-1.0, -0.25, 0.7],
                [0.0, -0.75, 1.2],
            ]
        ),
        "gates_quat": np.array(
            [
                [0.0, 0.0, np.sin(-0.78 / 2), np.cos(-0.78 / 2)],
                [0.0, 0.0, np.sin(2.35 / 2), np.cos(2.35 / 2)],
                [0.0, 0.0, np.sin(3.14 / 2), np.cos(3.14 / 2)],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        "obstacles_pos": np.array(
            [[0.0, 0.75, 1.55], [1.0, 0.25, 1.55], [-1.5, -0.25, 1.55], [-0.5, -0.75, 1.55]]
        ),
        "target_gate": 0,
    }


def test_sfc_planner_constructs_initial_spline():
    from lsy_drone_racing.control.sfc_planner import SfcPlanner

    planner = SfcPlanner(_minimal_obs(), freq=50)
    assert planner.t_total > 0
    assert planner.target_gate_idx == 0


def test_sfc_planner_evaluate_returns_time_scaled_derivatives():
    """vel and acc must be in m/s and m/s², not in BSpline parameter units."""
    from lsy_drone_racing.control.sfc_planner import SfcPlanner

    planner = SfcPlanner(_minimal_obs(), freq=50)
    pos0, vel0, acc0 = planner.evaluate(0.0)
    pos1, _, _ = planner.evaluate(0.1)

    # Forward-difference vel estimate vs evaluate's vel: should agree to first order
    fd_vel = (pos1 - pos0) / 0.1
    assert np.linalg.norm(vel0 - fd_vel) < 1.0  # loose — just confirms units


def test_sfc_planner_update_returns_false_when_nothing_moves():
    from lsy_drone_racing.control.sfc_planner import SfcPlanner

    obs = _minimal_obs()
    planner = SfcPlanner(obs, freq=50)
    assert planner.update(obs) is False


def test_sfc_planner_update_replans_when_obstacle_moves_above_threshold():
    from lsy_drone_racing.control.sfc_planner import SfcPlanner

    obs = _minimal_obs()
    planner = SfcPlanner(obs, freq=50)
    # Tick past the debounce window
    for _ in range(6):
        planner.update(obs)

    obs2 = {**obs, "obstacles_pos": obs["obstacles_pos"].copy()}
    obs2["obstacles_pos"][0] = obs["obstacles_pos"][0] + np.array([0.10, 0.0, 0.0])  # 10 cm shift
    assert planner.update(obs2) is True


def test_sfc_planner_update_debounces_consecutive_replans():
    from lsy_drone_racing.control.sfc_planner import SfcPlanner

    obs = _minimal_obs()
    planner = SfcPlanner(obs, freq=50)
    obs2 = {**obs, "obstacles_pos": obs["obstacles_pos"].copy()}
    obs2["obstacles_pos"][0] = obs["obstacles_pos"][0] + np.array([0.10, 0.0, 0.0])

    # First post-init update with movement → replans
    assert planner.update(obs2) is True
    # Immediately again → debounced even with another movement
    obs3 = {**obs2, "obstacles_pos": obs2["obstacles_pos"].copy()}
    obs3["obstacles_pos"][1] = obs["obstacles_pos"][1] + np.array([0.10, 0.0, 0.0])
    assert planner.update(obs3) is False


def test_sfc_planner_update_syncs_target_gate_from_obs():
    from lsy_drone_racing.control.sfc_planner import SfcPlanner

    obs = _minimal_obs()
    planner = SfcPlanner(obs, freq=50)
    obs_g1 = {**obs, "target_gate": 1}
    planner.update(obs_g1)
    assert planner.target_gate_idx == 1


def test_sfc_planner_episode_reset_zeroes_counters():
    from lsy_drone_racing.control.sfc_planner import SfcPlanner

    obs = _minimal_obs()
    planner = SfcPlanner(obs, freq=50)
    obs_g2 = {**obs, "target_gate": 2}
    planner.update(obs_g2)
    assert planner.target_gate_idx == 2

    planner.episode_reset()
    assert planner.target_gate_idx == 0


def test_sfc_planner_records_init_replan_event():
    from lsy_drone_racing.control.sfc_planner import SfcPlanner

    planner = SfcPlanner(_minimal_obs(), freq=50)
    assert len(planner.replan_events) == 1
    evt = planner.replan_events[0]
    assert evt["reason"] == "init"
    assert evt["tick"] == 0
    snap = evt["snapshot"]
    for key in ("t_total", "knots", "control_points", "k", "target_gate_idx"):
        assert key in snap
    assert snap["t_total"] > 0
    assert snap["k"] == 3


def test_sfc_planner_records_replan_event_on_obstacle_move():
    from lsy_drone_racing.control.sfc_planner import SfcPlanner

    obs = _minimal_obs()
    planner = SfcPlanner(obs, freq=50)
    for _ in range(6):
        planner.update(obs)
    obs2 = {**obs, "obstacles_pos": obs["obstacles_pos"].copy()}
    obs2["obstacles_pos"][0] = obs["obstacles_pos"][0] + np.array([0.10, 0.0, 0.0])
    assert planner.update(obs2) is True
    # init + this replan
    assert len(planner.replan_events) == 2
    assert planner.replan_events[-1]["reason"] == "obstacle_jitter"
    assert planner.last_replan_event is planner.replan_events[-1]


def test_sfc_planner_episode_reset_clears_replan_events():
    from lsy_drone_racing.control.sfc_planner import SfcPlanner

    planner = SfcPlanner(_minimal_obs(), freq=50)
    assert len(planner.replan_events) == 1
    planner.episode_reset()
    assert planner.replan_events == []
    assert planner.last_replan_event is None
