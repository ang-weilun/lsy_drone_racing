"""Unit tests for the SFC attitude controller's position controller helper."""

from __future__ import annotations

import numpy as np
import pytest

from lsy_drone_racing.control.sfc_attitude_controller import (
    KP,
    KD,
    KI_RANGE,
    G,
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
