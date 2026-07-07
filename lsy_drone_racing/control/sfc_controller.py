"""SFC state-mode tracker. Wraps SfcPlanner and emits a 13D state command.

Baseline tracker: the env's built-in position controller turns the 13D command
into firmware-level (thrust + tilt) commands. See `sfc_attitude_controller.py`
for the attitude-mode replacement.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.sfc_planner import SfcPlanner


def _dump_enabled() -> bool:
    return os.environ.get("SFC_DUMP", "").lower() in ("1", "true", "yes")


def _dump_path() -> str:
    return os.environ.get("SFC_DUMP_PATH", "")


class StateController(Controller):
    """SFC tracker emitting a 13D state command (env's position controller closes the loop)."""

    def __init__(self, obs, info, config) -> None:  # noqa: ANN001, D107
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._tick = 0
        self._spline_tick = 0
        self._finished = False
        self.planner = SfcPlanner(obs, self._freq)

        self._dump_active = _dump_enabled()
        if self._dump_active:
            self._dump_initial_gates_pos = obs["gates_pos"].copy()
            self._dump_initial_gates_quat = obs["gates_quat"].copy()
            self._dump_initial_obstacles_pos = (
                obs.get("obstacles_pos", np.array([])).copy()
                if obs.get("obstacles_pos") is not None
                else np.array([])
            )
            self._dump_tick_rows = []
            self._dump_replans = []
            self._dump_replan_idx = 0  # initial spline counts as replan 0
            self._record_replan_snapshot(obs)
            self._dump_terminated_reason = "running"
            self._dump_episode_idx = 0

    def compute_control(self, obs, info=None):  # noqa: ANN001, ANN201, D102
        replanned = self.planner.update(obs)
        if replanned:
            self._spline_tick = 0
            if self._dump_active:
                self._dump_replan_idx += 1
                self._record_replan_snapshot(obs)

        t = min(self._spline_tick / self._freq, self.planner.t_total)
        if t >= self.planner.t_total and self.planner.target_gate_idx >= len(
            self.planner.gates_pos
        ):
            self._finished = True

        des_pos, des_vel, des_acc = self.planner.evaluate(t)
        yaw = (
            float(np.arctan2(des_vel[1], des_vel[0])) if np.linalg.norm(des_vel[:2]) > 0.1 else 0.0
        )
        action = np.concatenate((des_pos, des_vel, des_acc, [yaw], np.zeros(3)), dtype=np.float32)

        if self._dump_active:
            self._dump_tick_rows.append(
                {
                    "tick": int(self._tick),
                    "spline_tick": int(self._spline_tick),
                    "t": float(t),
                    "u": float(t / self.planner.t_total) if self.planner.t_total > 0 else 0.0,
                    "pos": np.asarray(obs["pos"][:3], dtype=np.float64),
                    "vel": np.asarray(obs.get("vel", np.zeros(3))[:3], dtype=np.float64),
                    "des_pos": np.asarray(des_pos, dtype=np.float64),
                    "des_vel": np.asarray(des_vel, dtype=np.float64),
                    "des_acc": np.asarray(des_acc, dtype=np.float64),
                    "yaw": float(yaw),
                    "target_gate_idx": int(self.planner.target_gate_idx),
                    "replan_idx": int(self._dump_replan_idx),
                }
            )

        return action

    def _record_replan_snapshot(self, obs):  # noqa: ANN001
        skel = self.planner._calculate_anchors(obs["pos"][:3])  # for dump only
        anchors_pos = np.array([p.pos for p in skel], dtype=np.float64)
        anchors_is_gate = np.array([p.is_gate for p in skel], dtype=bool)
        anchors_normal = np.array(
            [p.gate_normal if p.gate_normal is not None else np.full(3, np.nan) for p in skel],
            dtype=np.float64,
        )
        self._dump_replans.append(
            {
                "tick": self._tick,
                "target_gate_idx": int(self.planner.target_gate_idx),
                "anchors_pos": anchors_pos,
                "anchors_is_gate": anchors_is_gate,
                "anchors_normal": anchors_normal,
                "control_points": np.asarray(self.planner.control_points, dtype=np.float64),
                "knots": np.asarray(self.planner.des_pos_spline.t, dtype=np.float64),
                "t_total": float(self.planner.t_total),
                "current_pos": np.asarray(obs["pos"][:3], dtype=np.float64),
                "current_vel": np.asarray(obs.get("vel", np.zeros(3))[:3], dtype=np.float64),
                "gates_pos": self.planner.gates_pos.copy(),
                "gates_quat": self.planner.gates_quat.copy(),
                "obstacles_pos": (
                    self.planner.obstacles_pos.copy()
                    if len(self.planner.obstacles_pos) > 0
                    else np.zeros((0, 3))
                ),
            }
        )

    def step_callback(self, action, obs, reward, terminated, truncated, info):  # noqa: ANN001, ANN201, D102
        self._tick += 1
        self._spline_tick += 1

        if self._dump_active and (terminated or truncated or self._finished):
            if terminated and obs.get("target_gate", 0) != -1:
                self._dump_terminated_reason = "collision"
            elif truncated:
                self._dump_terminated_reason = "timeout"
            else:
                self._dump_terminated_reason = "finished"
            self._dump_final_target_gate = int(obs.get("target_gate", -2))
            self._dump_final_pos = np.asarray(obs["pos"][:3], dtype=np.float64)

        return self._finished

    def episode_callback(self) -> None:  # noqa: D102
        if self._dump_active:
            self._write_dump()
            self._dump_tick_rows = []
            self._dump_replans = []
            self._dump_replan_idx = 0
            self._dump_terminated_reason = "running"
            self._dump_episode_idx += 1
        self._tick = 0
        self._spline_tick = 0
        self._finished = False
        self.planner.episode_reset()

    def render_callback(self, sim) -> None:  # noqa: ANN001, D102
        if self.planner.t_total <= 0:
            return
        u = min(self._spline_tick / self._freq, self.planner.t_total) / self.planner.t_total
        draw_points(
            sim, self.planner.des_pos_spline(u).reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.04
        )
        draw_line(
            sim, self.planner.des_pos_spline(np.linspace(0.0, 1.0, 100)), rgba=(0.0, 1.0, 0.0, 1.0)
        )
        if len(self.planner.control_points) > 0:
            draw_points(sim, self.planner.control_points, rgba=(0.0, 0.0, 1.0, 1.0), size=0.02)

    def _write_dump(self) -> None:
        """Serialize collected dump buffers to an .npz file. Path resolution:

        - SFC_DUMP_PATH unset → ./sfc_dump.npz (or sfc_dump_ep{N}.npz for n_runs>1)
        - SFC_DUMP_PATH ends with /  → directory; episode files inside
        - SFC_DUMP_PATH otherwise → exact file (single-episode runs) or
          stem-suffixed (multi-episode runs, appends `_ep{N}`)
        """  # noqa: D415
        out = _dump_path() or "sfc_dump.npz"
        path = Path(out)
        if str(out).endswith(("/", os.sep)):
            path.mkdir(parents=True, exist_ok=True)
            path = path / f"episode_{self._dump_episode_idx:03d}.npz"
        elif self._dump_episode_idx > 0 or hasattr(self, "_dump_force_suffix"):
            path = path.with_name(
                f"{path.stem}_ep{self._dump_episode_idx:03d}{path.suffix or '.npz'}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._dump_tick_rows:
            tick_keys = list(self._dump_tick_rows[0].keys())
            tick_arrs = {k: np.array([r[k] for r in self._dump_tick_rows]) for k in tick_keys}
        else:
            tick_arrs = {}

        replan_obj_keys = (
            "anchors_pos",
            "anchors_is_gate",
            "anchors_normal",
            "control_points",
            "knots",
            "current_pos",
            "current_vel",
            "gates_pos",
            "gates_quat",
            "obstacles_pos",
        )
        replan_scalar_keys = ("tick", "target_gate_idx", "t_total")
        replan_arrs = {}
        for k in replan_scalar_keys:
            replan_arrs[f"replan_{k}"] = np.array([r[k] for r in self._dump_replans])
        for k in replan_obj_keys:
            replan_arrs[f"replan_{k}"] = np.array([r[k] for r in self._dump_replans], dtype=object)

        meta = {
            "terminated_reason": self._dump_terminated_reason,
            "final_target_gate": getattr(self, "_dump_final_target_gate", -99),
            "final_pos": getattr(self, "_dump_final_pos", np.full(3, np.nan)),
            "initial_gates_pos": self._dump_initial_gates_pos,
            "initial_gates_quat": self._dump_initial_gates_quat,
            "initial_obstacles_pos": self._dump_initial_obstacles_pos,
            "anchor_gap": self.planner.anchor_gap,
            "base_speed": self.planner.base_speed,
            "safety_margin": self.planner.safety_margin,
            "n_replans": len(self._dump_replans),
            "n_ticks": len(self._dump_tick_rows),
        }

        np.savez_compressed(
            path, **tick_arrs, **replan_arrs, **{f"meta_{k}": v for k, v in meta.items()}
        )
