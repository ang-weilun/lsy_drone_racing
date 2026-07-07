"""SFC tracker in attitude mode.

Wraps SfcPlanner and emits a 4D attitude command [roll, pitch, yaw, thrust]
using a PID + acceleration-feedforward position controller (Mellinger-Kumar /
Handout eq. 17).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

from drone_models.core import load_params

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.sfc_planner_relaxed import SfcPlanner

# Position-controller gains (Newtons / metre, ported from attitude_controller.py)
KP = np.array([0.8, 0.8, 1.25])
KI = np.array([0.05, 0.05, 0.05])
KD = np.array([0.283, 0.283, 0.4])
KI_RANGE = np.array([2.0, 2.0, 2.0])  # symmetric integrator clamp
G = 9.81

# Saturation / smoothing
TILT_LIMIT = 0.7  # rad (~40°)
TILT_RATE_LIMIT = 0.3  # rad per 50 Hz tick
YAW_SPEED_THRESHOLD = 0.1  # m/s
Y_CROSS_EPS = 1e-3  # singularity guard for cross(z_b_des, x_c)

# Replan handling (used by the controller class, not the helper)
REPLAN_I_RESET_THRESHOLD = 0.10  # m, horizontal-I reset gate

# Acceleration-FF lookahead. Evaluates the planner's a_ref at t + ACC_FF_LOOKAHEAD
# (instead of t) when feeding mass·a_ref into F_des. Compensates for crazyflie
# attitude-loop reaction lag (~10–20 ms): by the time the firmware tracks the
# commanded rpy, the drone is at the right attitude for the *current* path
# point rather than one tick behind. Set to 0.0 to disable. p_ref / v_ref are
# still evaluated at t so the position/velocity error terms refer to the actual
# current schedule point.
ACC_FF_LOOKAHEAD = 0.10  # s (5 outer-loop ticks at 50 Hz)


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
        yaw_next: yaw used this tick (always 0 — see body for why).
        y_b_next: new y-body axis (or previous if singularity guard fired).
        rpy_next: rpy after slew-rate limiter (also stored as prev for next tick).
    """
    e_p = p_ref - p
    e_v = v_ref - v

    # Tentative integral update, clamped per-axis
    dt = 1.0 / 50.0  # outer-loop tick assumption (env.freq = 50 Hz)
    integrator_tentative = np.clip(integrator + e_p * dt, -KI_RANGE, KI_RANGE)

    # Mellinger-Kumar: F_des = m·a_ref + Kp·e_p + Ki·∫e_p + Kd·e_v + m·g·ẑ
    F_des = mass * a_ref + KP * e_p + KI * integrator_tentative + KD * e_v
    F_des[2] += mass * G

    # Scalar collective thrust = projection on current body z, clamped
    z_b_curr = R.from_quat(quat).as_matrix()[:, 2]
    thrust_unclipped = float(F_des @ z_b_curr)
    thrust = float(np.clip(thrust_unclipped, thrust_min, thrust_max))
    # Anti-windup: freeze integrator on saturation
    integrator_next = integrator if thrust != thrust_unclipped else integrator_tentative

    # Yaw: command always 0. The env's attitude action space clips yaw to ±π/2;
    # planner-derived yaw values (e.g., facing +135° at gate 1) get silently
    # clipped to +90°, and the resulting 180° SO(3) rotation errors are
    # degenerate (the vee operator returns zero), causing the drone to drift
    # arbitrarily. Hardcoding yaw=0 matches the working attitude_controller.py
    # and lets our (roll, pitch) decomposition at actual_yaw handle thrust
    # direction correctly regardless of the drone's heading.
    yaw_next = 0.0

    # Build R_des using the drone's ACTUAL yaw (not commanded) so the (roll, pitch)
    # decomposition produces the correct world-frame thrust direction regardless of
    # yaw tracking lag. The Crazyflie firmware caps yaw rate at ~250°/s; when the
    # planner sweeps yaw faster than that, the drone's actual yaw lags by tens of
    # degrees, and a (roll, pitch) decomposition built at yaw_next would rotate
    # the world-frame thrust by the lag angle. Decoupling is the standard
    # geometric-controller fix (Mellinger 2011 §III).
    yaw_actual = float(R.from_quat(quat).as_euler("xyz")[2])
    F_norm = float(np.linalg.norm(F_des))
    z_b_des = F_des / F_norm if F_norm > 1e-9 else np.array([0.0, 0.0, 1.0])
    x_c = np.array([np.cos(yaw_actual), np.sin(yaw_actual), 0.0])
    y_b_unnorm = np.cross(z_b_des, x_c)
    y_b_norm = float(np.linalg.norm(y_b_unnorm))
    if y_b_norm < Y_CROSS_EPS:
        y_b_next = y_b_prev if y_b_prev is not None else np.array([0.0, 1.0, 0.0])
    else:
        y_b_next = y_b_unnorm / y_b_norm
    x_b_des = np.cross(y_b_next, z_b_des)
    R_des = np.column_stack([x_b_des, y_b_next, z_b_des])
    rpy_des = np.array(R.from_matrix(R_des).as_euler("xyz"))

    # Override the extracted yaw with the COMMANDED yaw so the firmware rotates
    # toward the planned heading at its own pace; meanwhile the (roll, pitch)
    # tilts are applied in the drone's current body frame and produce correct
    # world-frame thrust immediately.
    rpy_des[2] = yaw_next

    # Tilt cap (per-axis), then slew-rate limit
    rpy_des[:2] = np.clip(rpy_des[:2], -TILT_LIMIT, TILT_LIMIT)
    delta = rpy_des - rpy_prev
    # Yaw wraps at ±π; use shortest signed angular distance, otherwise the slew
    # limit drives the drone the long way around (e.g. +3.0 → -3.0 rad is
    # +0.28 rad the short way, but naive subtraction gives -6.0 rad → spin).
    delta[2] = ((delta[2] + np.pi) % (2 * np.pi)) - np.pi
    delta_clipped = np.clip(delta, -TILT_RATE_LIMIT, TILT_RATE_LIMIT)
    rpy_next = rpy_prev + delta_clipped
    # Keep stored yaw in [-π, π] so future deltas stay bounded
    rpy_next[2] = ((rpy_next[2] + np.pi) % (2 * np.pi)) - np.pi

    action = np.array([rpy_next[0], rpy_next[1], rpy_next[2], thrust], dtype=np.float32)
    return action, integrator_next, yaw_next, y_b_next, rpy_next


class SfcAttitudeController(Controller):
    """SFC tracker emitting [roll, pitch, yaw, thrust] via PID + acceleration FF."""

    def __init__(self, obs, info, config) -> None:  # noqa: ANN001, D107
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        params = load_params(config.sim.physics, config.sim.drone_model)
        self._mass = float(params["mass"])
        # Per-motor thrust bounds × 4 motors = total collective thrust bounds (Newtons)
        self._thrust_min = float(params["thrust_min"]) * 4.0
        self._thrust_max = float(params["thrust_max"]) * 4.0

        self._tick = 0
        self._spline_tick = 0
        self._finished = False
        self._i_error = np.zeros(3)
        self._yaw_prev = None  # lazy-init from current heading on first tick
        self._rpy_prev = np.zeros(3)
        self._y_b_prev = None
        self._replan_idx = 0
        self._last_diag: dict | None = None
        self.planner = SfcPlanner(obs, self._freq)

    def compute_control(self, obs, info=None):  # noqa: ANN001, ANN201, D102
        if self._yaw_prev is None:
            self._yaw_prev = float(R.from_quat(obs["quat"]).as_euler("xyz")[2])

        replanned = self.planner.update(obs)
        if replanned:
            self._spline_tick = 0
            self._replan_idx += 1
            new_pos0, _, _ = self.planner.evaluate(0.0)
            # Horizontal I reset only on big jump; vertical I always preserved.
            if np.linalg.norm(new_pos0[:2] - obs["pos"][:2]) > REPLAN_I_RESET_THRESHOLD:
                self._i_error[:2] = 0.0

        t = min(self._spline_tick / self._freq, self.planner.t_total)
        des_pos, des_vel, des_acc = self.planner.evaluate(t)

        # Acceleration-FF lookahead: a_ref leads p_ref/v_ref by ACC_FF_LOOKAHEAD
        # to compensate for firmware attitude-loop lag. Spline gives this exactly.
        if ACC_FF_LOOKAHEAD > 0.0:
            t_ff = min(t + ACC_FF_LOOKAHEAD, self.planner.t_total)
            _, _, des_acc_ff = self.planner.evaluate(t_ff)
        else:
            des_acc_ff = des_acc

        if t >= self.planner.t_total and obs.get("target_gate", 0) == -1:
            self._finished = True

        action, self._i_error, self._yaw_prev, self._y_b_prev, self._rpy_prev = (
            compute_attitude_command(
                obs["pos"],
                obs.get("vel", np.zeros(3)),
                des_pos,
                des_vel,
                des_acc_ff,
                obs["quat"],
                self._mass,
                self._i_error,
                self._thrust_min,
                self._thrust_max,
                self._yaw_prev,
                self._y_b_prev,
                self._rpy_prev,
            )
        )

        rpy_act = np.asarray(R.from_quat(obs["quat"]).as_euler("xyz"), dtype=np.float64)
        thrust_cmd = float(action[3])
        eps_t = 1e-4
        self._last_diag = {
            "des_pos": np.asarray(des_pos, dtype=np.float64).copy(),
            "des_vel": np.asarray(des_vel, dtype=np.float64).copy(),
            "des_acc": np.asarray(des_acc, dtype=np.float64).copy(),
            "thrust_cmd": thrust_cmd,
            "at_thrust_min": thrust_cmd <= self._thrust_min + eps_t,
            "at_thrust_max": thrust_cmd >= self._thrust_max - eps_t,
            "i_error": self._i_error.copy(),
            "rpy_cmd": np.asarray(action[:3], dtype=np.float64).copy(),
            "rpy_act": rpy_act,
        }

        return action

    def step_callback(self, action, obs, reward, terminated, truncated, info):  # noqa: ANN001, ANN201, D102
        self._tick += 1
        self._spline_tick += 1
        return self._finished

    def episode_callback(self):  # noqa: D102
        self._tick = 0
        self._spline_tick = 0
        self._finished = False
        self._i_error[:] = 0.0
        self._yaw_prev = None
        self._rpy_prev[:] = 0.0
        self._y_b_prev = None
        self._replan_idx = 0
        self.planner.episode_reset()

    def render_callback(self, sim):  # noqa: ANN001, D102
        if self.planner.t_total <= 0:
            return
        from crazyflow.sim.visualize import draw_line, draw_points

        u = min(self._spline_tick / self._freq, self.planner.t_total) / self.planner.t_total
        draw_points(
            sim, self.planner.des_pos_spline(u).reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.04
        )
        draw_line(
            sim, self.planner.des_pos_spline(np.linspace(0.0, 1.0, 100)), rgba=(0.0, 1.0, 0.0, 1.0)
        )
