"""SFC tracker in attitude mode.

Wraps SfcPlanner and emits a 4D attitude command [roll, pitch, yaw, thrust]
using a PID + acceleration-feedforward position controller (Mellinger-Kumar /
Handout eq. 17). See docs/superpowers/specs/2026-04-28-sfc-attitude-controller-design.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Position-controller gains (Newtons / metre, ported from attitude_controller.py)
KP = np.array([0.4, 0.4, 1.25])
KI = np.array([0.05, 0.05, 0.05])
KD = np.array([0.2, 0.2, 0.4])
KI_RANGE = np.array([2.0, 2.0, 2.0])         # symmetric integrator clamp
G = 9.81

# Saturation / smoothing
TILT_LIMIT = 0.5                              # rad (~28°)
TILT_RATE_LIMIT = 0.3                         # rad per 50 Hz tick
YAW_SPEED_THRESHOLD = 0.1                     # m/s
Y_CROSS_EPS = 1e-3                            # singularity guard for cross(z_b_des, x_c)

# Replan handling (used by the controller class, not the helper)
REPLAN_I_RESET_THRESHOLD = 0.10               # m, horizontal-I reset gate


def compute_attitude_command(
    p: NDArray,
    v: NDArray,
    p_ref: NDArray,
    v_ref: NDArray,
    a_ref: NDArray,
    quat: NDArray,
    mass: float,
    integrator: NDArray,
    thrust_min: float,
    thrust_max: float,
    yaw_prev: float,
    y_b_prev: NDArray | None,
    rpy_prev: NDArray,
) -> tuple[NDArray, NDArray, float, NDArray, NDArray]:
    """Mellinger-Kumar / Handout eq. 17 position controller.

    Returns:
        action: [roll, pitch, yaw, thrust] as float32, in radians and Newtons.
        integrator_next: updated integrator state (frozen if thrust saturated).
        yaw_next: yaw used this tick (held from yaw_prev when speed too low).
        y_b_next: new y-body axis (or previous if singularity guard fired).
        rpy_next: rpy after slew-rate limiter (also stored as prev for next tick).
    """
    e_p = p_ref - p
    e_v = v_ref - v

    # Tentative integral update, clamped per-axis
    dt = 1.0 / 50.0  # outer-loop tick assumption (env.freq = 50 Hz)
    integrator_tentative = np.clip(integrator + e_p * dt, -KI_RANGE, KI_RANGE)

    # Mellinger-Kumar: F_des = m·a_ref + Kp·e_p + Ki·∫e_p + Kd·e_v + m·g·ẑ
    F_des = (
        mass * a_ref
        + KP * e_p + KI * integrator_tentative + KD * e_v
    )
    F_des[2] += mass * G

    # Scalar collective thrust = projection on current body z, clamped
    z_b_curr = R.from_quat(quat).as_matrix()[:, 2]
    thrust_unclipped = float(F_des @ z_b_curr)
    thrust = float(np.clip(thrust_unclipped, thrust_min, thrust_max))
    # Anti-windup: freeze integrator on saturation
    integrator_next = integrator if thrust != thrust_unclipped else integrator_tentative

    # Yaw: hold previous when desired horizontal speed too low
    speed_xy = float(np.linalg.norm(v_ref[:2]))
    yaw_next = (
        float(np.arctan2(v_ref[1], v_ref[0])) if speed_xy > YAW_SPEED_THRESHOLD else float(yaw_prev)
    )

    # Build R_des from F_des direction + yaw, with singularity guard
    F_norm = float(np.linalg.norm(F_des))
    z_b_des = F_des / F_norm if F_norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    x_c = np.array([np.cos(yaw_next), np.sin(yaw_next), 0.0])
    y_b_unnorm = np.cross(z_b_des, x_c)
    y_b_norm = float(np.linalg.norm(y_b_unnorm))
    if y_b_norm < Y_CROSS_EPS:
        y_b_next = y_b_prev if y_b_prev is not None else np.array([0.0, 1.0, 0.0])
    else:
        y_b_next = y_b_unnorm / y_b_norm
    x_b_des = np.cross(y_b_next, z_b_des)
    R_des = np.column_stack([x_b_des, y_b_next, z_b_des])
    rpy_des = np.array(R.from_matrix(R_des).as_euler("xyz"))

    # Tilt cap (per-axis), then slew-rate limit
    rpy_des[:2] = np.clip(rpy_des[:2], -TILT_LIMIT, TILT_LIMIT)
    rpy_next = rpy_prev + np.clip(rpy_des - rpy_prev, -TILT_RATE_LIMIT, TILT_RATE_LIMIT)

    action = np.array([rpy_next[0], rpy_next[1], rpy_next[2], thrust], dtype=np.float32)
    return action, integrator_next, yaw_next, y_b_next, rpy_next
