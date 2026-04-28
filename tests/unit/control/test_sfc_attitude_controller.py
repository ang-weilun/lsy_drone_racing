"""Unit tests for the SFC attitude controller's position controller helper."""

from __future__ import annotations

import numpy as np
import pytest

from lsy_drone_racing.control.sfc_attitude_controller import (
    KP,
    KD,
    KI_RANGE,
    G,
    TILT_LIMIT,
    TILT_RATE_LIMIT,
    compute_attitude_command,
)


def test_hover_equilibrium_emits_level_attitude_and_hover_thrust():
    """At hover (zero error, zero ref accel), command should be (0,0,0,m·g)."""
    p = np.array([0.0, 0.0, 1.0])
    v = np.array([0.0, 0.0, 0.0])
    p_ref = p.copy()
    v_ref = v.copy()
    a_ref = np.zeros(3)
    quat = np.array([0.0, 0.0, 0.0, 1.0])  # identity rotation
    mass = 0.043
    integrator = np.zeros(3)
    thrust_min, thrust_max = 0.0, 1.0
    yaw_prev = 0.0
    y_b_prev = None
    rpy_prev = np.zeros(3)

    action, new_int, new_yaw, new_y_b, new_rpy = compute_attitude_command(
        p, v, p_ref, v_ref, a_ref, quat, mass,
        integrator, thrust_min, thrust_max, yaw_prev, y_b_prev, rpy_prev,
    )

    assert action.shape == (4,)
    assert action.dtype == np.float32
    # Action layout: [roll, pitch, yaw, thrust]
    assert abs(float(action[0])) < 1e-3, f"roll should be ~0, got {action[0]}"
    assert abs(float(action[1])) < 1e-3, f"pitch should be ~0, got {action[1]}"
    assert abs(float(action[2])) < 1e-3, f"yaw should be ~0, got {action[2]}"
    assert abs(float(action[3]) - mass * G) < 1e-3, \
        f"thrust should be ~m*g={mass * G:.4f}, got {action[3]}"


def test_pure_pd_no_acc_ff_drives_thrust_above_hover_when_below_setpoint():
    """If the drone is 0.1 m below the setpoint, thrust > hover."""
    p = np.array([0.0, 0.0, 1.0])
    v = np.array([0.0, 0.0, 0.0])
    p_ref = np.array([0.0, 0.0, 1.1])  # 0.1 m above current
    v_ref = np.zeros(3)
    a_ref = np.zeros(3)
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    mass = 0.043

    action, *_ = compute_attitude_command(
        p, v, p_ref, v_ref, a_ref, quat, mass,
        np.zeros(3), 0.0, 1.0, 0.0, None, np.zeros(3),
    )
    assert float(action[3]) > mass * G + KP[2] * 0.1 * 0.99  # PD term contributes


def test_thrust_clamps_to_thrust_max():
    """If F_des projection exceeds thrust_max, thrust is clipped."""
    # Big positive z-error → big z-thrust demand → exceeds thrust_max=0.5 N
    p = np.array([0.0, 0.0, 0.5])
    p_ref = np.array([0.0, 0.0, 5.0])
    action, *_ = compute_attitude_command(
        p, np.zeros(3), p_ref, np.zeros(3), np.zeros(3),
        np.array([0.0, 0.0, 0.0, 1.0]), 0.043,
        np.zeros(3), 0.0, 0.5, 0.0, None, np.zeros(3),
    )
    assert abs(float(action[3]) - 0.5) < 1e-6, f"thrust should clip to 0.5, got {action[3]}"


def test_integrator_frozen_on_saturation():
    """When thrust saturates, the integrator is NOT updated this tick."""
    p = np.array([0.0, 0.0, 0.5])
    p_ref = np.array([0.0, 0.0, 5.0])
    integrator = np.array([0.0, 0.0, 0.5])
    _, new_int, *_ = compute_attitude_command(
        p, np.zeros(3), p_ref, np.zeros(3), np.zeros(3),
        np.array([0.0, 0.0, 0.0, 1.0]), 0.043,
        integrator, 0.0, 0.5, 0.0, None, np.zeros(3),
    )
    assert np.allclose(new_int, integrator), "integrator should be frozen when thrust saturates"


def test_integrator_advances_when_unsaturated():
    """When thrust is comfortably within bounds, integrator updates by e_p * dt."""
    p = np.array([0.0, 0.0, 1.0])
    p_ref = np.array([0.05, 0.0, 1.0])
    integrator = np.zeros(3)
    _, new_int, *_ = compute_attitude_command(
        p, np.zeros(3), p_ref, np.zeros(3), np.zeros(3),
        np.array([0.0, 0.0, 0.0, 1.0]), 0.043,
        integrator, 0.0, 1.0, 0.0, None, np.zeros(3),
    )
    assert new_int[0] > 0.0, f"integrator x should advance positive, got {new_int}"
    assert new_int[0] < KI_RANGE[0]  # within clamp


def test_integrator_clamped_to_ki_range():
    """Integrator can never exceed ±KI_RANGE."""
    p = np.array([0.0, 0.0, 1.0])
    p_ref = np.array([100.0, 0.0, 1.0])  # absurd error
    integrator = KI_RANGE.copy() - 0.001  # near-clamp
    _, new_int, *_ = compute_attitude_command(
        p, np.zeros(3), p_ref, np.zeros(3), np.zeros(3),
        np.array([0.0, 0.0, 0.0, 1.0]), 0.043,
        integrator, 0.0, 100.0, 0.0, None, np.zeros(3),
    )
    assert new_int[0] <= KI_RANGE[0] + 1e-9


def test_yaw_holds_previous_when_desired_speed_below_threshold():
    """At low desired horizontal speed, yaw should stay at yaw_prev (not snap to 0)."""
    yaw_prev = 1.234  # arbitrary held heading
    _, _, new_yaw, *_ = compute_attitude_command(
        np.zeros(3), np.zeros(3),
        np.zeros(3), np.array([0.05, 0.0, 0.0]),  # speed_xy = 0.05 < threshold 0.1
        np.zeros(3),
        np.array([0.0, 0.0, 0.0, 1.0]), 0.043,
        np.zeros(3), 0.0, 1.0, yaw_prev, None, np.zeros(3),
    )
    assert abs(new_yaw - yaw_prev) < 1e-9, f"yaw should hold {yaw_prev}, got {new_yaw}"


def test_yaw_tracks_des_vel_when_above_threshold():
    """At high desired horizontal speed, yaw aligns with des_vel direction."""
    _, _, new_yaw, *_ = compute_attitude_command(
        np.zeros(3), np.zeros(3),
        np.zeros(3), np.array([1.0, 1.0, 0.0]),  # 45° heading
        np.zeros(3),
        np.array([0.0, 0.0, 0.0, 1.0]), 0.043,
        np.zeros(3), 0.0, 1.0, 0.0, None, np.zeros(3),
    )
    assert abs(new_yaw - np.pi / 4) < 1e-6


def test_singularity_guard_returns_finite_unit_vector():
    """When z_b_des aligns with x_c, the cross-product norm goes to ~0; guard kicks in."""
    p = np.zeros(3); p_ref = np.array([100.0, 0.0, 0.0])
    y_b_prev = np.array([0.5, 0.5, 0.5]) / np.sqrt(0.75)  # arbitrary previous unit vector
    _, _, _, new_y_b, _ = compute_attitude_command(
        p, np.zeros(3), p_ref, np.zeros(3), np.zeros(3),
        np.array([0.0, 0.0, 0.0, 1.0]), 0.0001,  # tiny mass → tiny m·g; F_des dominated by KP·e_p
        np.zeros(3), 0.0, 1e9, 0.0, y_b_prev, np.zeros(3),
    )
    # Either the guard fires (returns y_b_prev) or it doesn't (returns a normalized cross).
    # Just assert the result is a finite unit vector (not NaN from divide-by-zero).
    assert np.all(np.isfinite(new_y_b))
    assert abs(np.linalg.norm(new_y_b) - 1.0) < 1e-6


def test_tilt_capped_to_TILT_LIMIT():
    """Even with huge horizontal error, roll/pitch never exceed TILT_LIMIT (~28°)."""
    p = np.zeros(3); p_ref = np.array([100.0, 100.0, 1.0])
    # Allow rate limit to relax: rpy_prev set so the slew limit isn't the tighter bound
    rpy_prev = np.array([0.5, 0.5, 0.0])
    action, *_ = compute_attitude_command(
        p, np.zeros(3), p_ref, np.zeros(3), np.zeros(3),
        np.array([0.0, 0.0, 0.0, 1.0]), 0.043,
        np.zeros(3), 0.0, 1.0, 0.0, None, rpy_prev,
    )
    assert abs(float(action[0])) <= TILT_LIMIT + 1e-6
    assert abs(float(action[1])) <= TILT_LIMIT + 1e-6


def test_slew_rate_limited_per_tick():
    """Roll/pitch can change by at most TILT_RATE_LIMIT in one tick."""
    p = np.zeros(3); p_ref = np.array([100.0, 0.0, 1.0])
    rpy_prev = np.zeros(3)  # last tick was level
    action, *_ = compute_attitude_command(
        p, np.zeros(3), p_ref, np.zeros(3), np.zeros(3),
        np.array([0.0, 0.0, 0.0, 1.0]), 0.043,
        np.zeros(3), 0.0, 1.0, 0.0, None, rpy_prev,
    )
    # Roll moved at most TILT_RATE_LIMIT per axis from rpy_prev=0
    assert abs(float(action[0])) <= TILT_RATE_LIMIT + 1e-6
    assert abs(float(action[1])) <= TILT_RATE_LIMIT + 1e-6
