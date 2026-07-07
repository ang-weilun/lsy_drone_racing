"""Tests for the extracted SFC planner module."""

from __future__ import annotations

import numpy as np

from lsy_drone_racing.control.sfc_planner_relaxed import (
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


def _minimal_obs():  # noqa: ANN202
    """Obs fixture matching what level 0 emits at t=0: 4 gates, 4 obstacles, drone at start."""
    return {
        "pos": np.array([-1.5, 0.75, 0.05]),
        "vel": np.array([0.0, 0.0, 0.0]),
        "quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "gates_pos": np.array(
            [[0.5, 0.25, 0.7], [1.05, 0.75, 1.2], [-1.0, -0.25, 0.7], [0.0, -0.75, 1.2]]
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
    from lsy_drone_racing.control.sfc_planner_relaxed import SfcPlanner

    planner = SfcPlanner(_minimal_obs(), freq=50)
    assert planner.t_total > 0
    assert planner.target_gate_idx == 0


def test_sfc_planner_evaluate_returns_time_scaled_derivatives():
    """Vel and acc must be in m/s and m/s², not in BSpline parameter units."""
    from lsy_drone_racing.control.sfc_planner_relaxed import SfcPlanner

    planner = SfcPlanner(_minimal_obs(), freq=50)
    pos0, vel0, acc0 = planner.evaluate(0.0)
    pos1, _, _ = planner.evaluate(0.1)

    # Forward-difference vel estimate vs evaluate's vel: should agree to first order
    fd_vel = (pos1 - pos0) / 0.1
    assert np.linalg.norm(vel0 - fd_vel) < 1.0  # loose — just confirms units


def test_sfc_planner_update_returns_false_when_nothing_moves():
    from lsy_drone_racing.control.sfc_planner_relaxed import SfcPlanner

    obs = _minimal_obs()
    planner = SfcPlanner(obs, freq=50)
    assert planner.update(obs) is False


def test_sfc_planner_update_replans_when_obstacle_moves_above_threshold():
    from lsy_drone_racing.control.sfc_planner_relaxed import SfcPlanner

    obs = _minimal_obs()
    planner = SfcPlanner(obs, freq=50)
    # Tick past the debounce window
    for _ in range(6):
        planner.update(obs)

    obs2 = {**obs, "obstacles_pos": obs["obstacles_pos"].copy()}
    obs2["obstacles_pos"][0] = obs["obstacles_pos"][0] + np.array([0.10, 0.0, 0.0])  # 10 cm shift
    assert planner.update(obs2) is True


def test_sfc_planner_update_debounces_consecutive_replans():
    from lsy_drone_racing.control.sfc_planner_relaxed import SfcPlanner

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
    from lsy_drone_racing.control.sfc_planner_relaxed import SfcPlanner

    obs = _minimal_obs()
    planner = SfcPlanner(obs, freq=50)
    obs_g1 = {**obs, "target_gate": 1}
    planner.update(obs_g1)
    assert planner.target_gate_idx == 1


def test_sfc_planner_episode_reset_zeroes_counters():
    from lsy_drone_racing.control.sfc_planner_relaxed import SfcPlanner

    obs = _minimal_obs()
    planner = SfcPlanner(obs, freq=50)
    obs_g2 = {**obs, "target_gate": 2}
    planner.update(obs_g2)
    assert planner.target_gate_idx == 2

    planner.episode_reset()
    assert planner.target_gate_idx == 0


def test_sfc_planner_records_init_replan_event():
    from lsy_drone_racing.control.sfc_planner_relaxed import SfcPlanner

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
    from lsy_drone_racing.control.sfc_planner_relaxed import SfcPlanner

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
    from lsy_drone_racing.control.sfc_planner_relaxed import SfcPlanner

    planner = SfcPlanner(_minimal_obs(), freq=50)
    assert len(planner.replan_events) == 1
    planner.episode_reset()
    assert planner.replan_events == []
    assert planner.last_replan_event is None


def test_tube_fence_holds_at_gate_planes():
    """Spline lateral offset at each gate's plane must stay inside the tube."""
    from scipy.spatial.transform import Rotation as R

    from lsy_drone_racing.control.sfc_planner_relaxed import SfcPlanner

    planner = SfcPlanner(_minimal_obs(), freq=50)
    spline = planner.des_pos_spline

    u_samples = np.linspace(0.0, 1.0, 2000)
    pts = spline(u_samples)  # (N, 3)

    gate_normals = R.from_quat(planner.gates_quat).apply([1.0, 0.0, 0.0])
    for g_pos, g_normal in zip(planner.gates_pos, gate_normals):
        signed = (pts - g_pos) @ g_normal  # (N,)
        # Find indices where the spline crosses the gate plane.
        crossings = np.where(np.diff(np.sign(signed)) != 0)[0]
        for c in crossings:
            # Linear interp of the crossing point.
            t = signed[c] / (signed[c] - signed[c + 1])
            crossing_pt = pts[c] + t * (pts[c + 1] - pts[c])
            offset = crossing_pt - g_pos
            lateral = offset - (offset @ g_normal) * g_normal
            # Only check crossings near the gate centre (within 0.5 m laterally);
            # the spline may cross the infinite gate plane elsewhere on the track.
            if np.linalg.norm(lateral) > 0.5:
                continue
            assert np.linalg.norm(lateral) < planner.GATE_TUBE_RADIUS + 0.02, (
                f"Gate at {g_pos}: lateral offset {np.linalg.norm(lateral):.3f} "
                f"exceeds tube radius {planner.GATE_TUBE_RADIUS}"
            )


def test_diagonal_gate_entry_when_prior_anchor_is_offset():
    """When the previous skeleton anchor is offset laterally from the gate normal,
    the spline should pass through the gate centre with a non-trivial cross angle —
    confirming the QP is using its diagonal freedom rather than forcing on-normal entry.
    """  # noqa: D205
    from lsy_drone_racing.control.sfc_planner_relaxed import SfcPlanner

    # Build an obs where the drone starts well off the normal axis of gate 0.
    # Gate 0 sits at (0, 0, 1) with yaw=0 → normal = +x. We start the drone
    # at (-1.0, 1.0, 1.0) — 1.0m left of the normal axis.
    obs = {
        "pos": np.array([-1.0, 1.0, 1.0]),
        "vel": np.array([0.0, 0.0, 0.0]),
        "quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "gates_pos": np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]]),
        "gates_quat": np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        "obstacles_pos": np.array([]).reshape(0, 3),
        "target_gate": 0,
    }

    planner = SfcPlanner(obs, freq=50)
    spline = planner.des_pos_spline

    # Find spline parameter where it crosses gate 0's plane (x = 0).
    u_samples = np.linspace(0.0, 1.0, 5000)
    pts = spline(u_samples)
    signed = pts[:, 0]  # gate normal is +x → signed distance is x-coord
    crossings = np.where(np.diff(np.sign(signed)) != 0)[0]
    assert len(crossings) >= 1, "spline must cross gate 0 plane"
    c = crossings[0]
    t = signed[c] / (signed[c] - signed[c + 1])
    crossing_pt = pts[c] + t * (pts[c + 1] - pts[c])

    # Position assertion: crossing point near gate centre (within 0.05 m).
    assert np.linalg.norm(crossing_pt - np.array([0.0, 0.0, 1.0])) < 0.05

    # Velocity-direction assertion: cross angle should be non-trivial.
    # Spline's tangent at crossing parameter:
    u_cross = u_samples[c] + t * (u_samples[c + 1] - u_samples[c])
    tangent = np.asarray(spline.derivative(nu=1)(u_cross))
    tangent /= np.linalg.norm(tangent)
    cos_angle = abs(tangent[0])  # |dot(tangent, gate_normal)|

    # Pre-relaxation, the hard symmetry equality forced cos_angle = 1.0
    # exactly. After relaxation (drop pre/post anchors, no symmetry, soft
    # alignment cost), the QP has freedom to deviate. Threshold 0.99 is
    # intentionally generous: gate-1 crash diagnosis showed strong diagonal
    # entry causes frame-bar clips, so W_GATE_ALIGN was bumped to firmly
    # suppress lateral offset. The relaxation that actually delivers laptime
    # gain is the corridor-topology change (pre/post off the segmentation),
    # not the entry angle. This test still confirms the QP is no longer
    # rigidly forcing cos_angle = 1.0.
    assert cos_angle < 0.99, (
        f"Cross-angle cosine {cos_angle:.3f} — QP appears to be exactly "
        "forcing on-normal entry; symmetry equality may be back."
    )
