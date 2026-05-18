r"""Sim eval for the Song-2023 RL controller with render-friendly extras.

Mirrors ``scripts/sim.py`` but adds two affordances specific to the RL
Song controller: a video recording path (``--record path.mp4``) that
captures one frame per env step, and a per-gate colour override on the
MuJoCo model so each gate is visually distinguishable in renders. The
underlying env construction is identical to ``scripts/sim.py``; this
file does not patch the observation any more (upstream PR #91 fixed the
level-3 ``nominal_*`` masking that previously made the un-visited branch
return the toml ``(0, 0, z)`` placeholders).

Usage
-----
::

    pixi run -e rl-train python -m lsy_drone_racing.control.rl_song.eval_sim \\
        --config level3.toml \\
        --checkpoint <path-to-step_NNNNNNNNNNNN-or-run-dir> \\
        --control_mode attitude \\
        --record renders/level3_eval.mp4 \\
        --n_runs 8
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

import lsy_drone_racing  # noqa: F401  ensures the env id is registered
from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller


logger = logging.getLogger(__name__)

DEFAULT_RECORD_CAMERA = "track_cam:0"
DEFAULT_RECORD_FPS = 50  # matches the 50 Hz env step → real-time playback
DEFAULT_RECORD_WIDTH = 1280
DEFAULT_RECORD_HEIGHT = 720
DEFAULT_CONTROLLER = "rl_song/controller.py"

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Per-gate colours applied to the frame, ropes, and stand geoms by
# :func:`_color_code_gates` so that gate 0 ... gate N-1 are visually distinct
# in renders. The textured front / back aperture panels are intentionally left
# alone so each gate still reads as a gate. Heat-progression palette
# (red → orange → yellow → green) mirrors gate-index progress through the
# course; longer-than-4-gate courses repeat from the start.
GATE_COLORS = np.array(
    [
        [1.00, 0.20, 0.20, 1.0],  # gate 0 — red
        [1.00, 0.60, 0.00, 1.0],  # gate 1 — orange
        [1.00, 1.00, 0.20, 1.0],  # gate 2 — yellow
        [0.20, 0.90, 0.30, 1.0],  # gate 3 — green
    ],
    dtype=np.float32,
)
# Geom *base* names (pre-suffix) inside a gate body that get the per-gate
# colour override. Matches the names in ``envs/assets/gate.xml``.
GATE_ACCENT_GEOMS = (
    "frame_top",
    "frame_bottom",
    "frame_left",
    "frame_right",
    "rope_top_left",
    "rope_top_right",
    "rope_bottom_left",
    "rope_bottom_right",
    "gate_stand",
)


class _TraceWriter:
    """JSONL trace writer with per-episode files and a header row."""

    SCHEMA_VERSION = 1

    def __init__(self, dump_dir: Path, header_common: dict[str, Any]) -> None:
        self.dump_dir = dump_dir
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self._header_common = header_common
        self._fh: TextIO | None = None
        self._episode_idx = -1

    def open_episode(self, episode_idx: int, episode_header: dict[str, Any]) -> None:
        self._episode_idx = episode_idx
        path = self.dump_dir / f"episode_{episode_idx:03d}.jsonl"
        self._fh = path.open("w", encoding="utf-8", newline="\n")
        header = {
            "_header": True,
            **self._header_common,
            **episode_header,
            "schema_version": self.SCHEMA_VERSION,
        }
        self._fh.write(json.dumps(header, separators=(",", ":")) + "\n")

    def write_row(self, row: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("write_row called before open_episode")
        self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    def close_episode(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write_run_meta(self, run_meta: dict[str, Any]) -> None:
        path = self.dump_dir / "run_meta.json"
        path.write_text(json.dumps(run_meta, indent=2, sort_keys=True), encoding="utf-8")


def _get_git_sha() -> str | None:
    """Return the short git SHA, or None if git is unavailable."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None


def simulate(
    config: str = "level3.toml",
    controller: str | None = None,
    n_runs: int = 1,
    render: bool | None = None,
    record: str | None = None,
    camera: str = DEFAULT_RECORD_CAMERA,
    fps: int = DEFAULT_RECORD_FPS,
    width: int = DEFAULT_RECORD_WIDTH,
    height: int = DEFAULT_RECORD_HEIGHT,
    checkpoint: str | None = None,
    control_mode: str | None = None,
    dump_trace: str | None = None,
    reward_cfg: str | None = None,
) -> list[float | None]:
    """Run the RL Song controller in sim with recording / render extras.

    Mirrors ``scripts.sim.simulate`` so the recording / CLI behaviour is
    identical; the differences are the gate colour override and the
    offscreen video recording path.

    Parameters
    ----------
    config : str
        Race config filename under ``config/``. Defaults to ``level3.toml``
        since that is the config this script is meaningfully different from
        ``scripts/sim.py`` on.
    controller : str, optional
        Override ``config.controller.file``. Defaults to
        :data:`DEFAULT_CONTROLLER` (``rl_song/controller.py``).
    n_runs : int
        Number of episodes to run.
    render : bool, optional
        Live MuJoCo viewer toggle. ``None`` inherits ``config.sim.render``.
        Forced to ``False`` when ``record`` is set.
    record : str, optional
        Path to an mp4 file. If set, captures one frame per env step.
    camera, fps, width, height
        Offscreen render parameters; see :mod:`scripts.sim`.
    checkpoint : str, optional
        Override / inject ``config.controller.checkpoint`` for controllers
        that read it (e.g. the RL Song controller). Resolved by the
        controller, not by this script.
    control_mode : str, optional
        Override ``config.env.control_mode`` (``"state"`` or ``"attitude"``).
        Needed when running an attitude-output controller against a
        state-mode config without editing the toml.
    dump_trace : str, optional
        If set, write per-step JSONL traces under this directory. Header
        row carries metadata; one row per env step. Default: no dump.
    reward_cfg : str, optional
        Override path to ``reward_config.json``. Default: resolved
        relative to the checkpoint.

    Returns
    -------
    ep_times : list[float | None]
        One entry per episode: flight time if the drone finished, ``None``
        otherwise.
    """
    cfg = load_config(_REPO_ROOT / "config" / config)
    if render is None:
        render = cfg.sim.render
    if record is not None:
        render = False
    cfg.sim.render = render

    if control_mode is not None:
        cfg.env.control_mode = control_mode
    if checkpoint is not None:
        cfg.controller.checkpoint = checkpoint

    # This script is dedicated to the RL Song controller; ignore
    # ``cfg.controller.file`` (which is ``state_controller.py`` in the stock
    # config TOMLs and would silently produce 13-d state-mode actions against
    # an attitude-mode env).
    control_path = _REPO_ROOT / "lsy_drone_racing" / "control"
    controller_rel = controller or DEFAULT_CONTROLLER
    controller_cls = load_controller(control_path / controller_rel)

    env = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    )
    _color_code_gates(env)

    trace_writer: _TraceWriter | None = None
    if dump_trace is not None:
        dump_dir = Path(dump_trace)
        if not dump_dir.is_absolute():
            dump_dir = _REPO_ROOT / dump_dir
        trace_writer = _TraceWriter(
            dump_dir=dump_dir,
            header_common={
                "config": config,
                "control_mode": cfg.env.control_mode,
                "freq": int(cfg.env.freq),
                "checkpoint": str(checkpoint) if checkpoint else None,
                "n_gates": len(cfg.env.track.gates),
                "n_obstacles": len(cfg.env.track.obstacles),
            },
        )
        trace_writer.write_run_meta(
            {
                "checkpoint": str(checkpoint) if checkpoint else None,
                "config": config,
                "control_mode": cfg.env.control_mode,
                "n_runs": n_runs,
                "seed": cfg.env.seed,
                "schema_version": _TraceWriter.SCHEMA_VERSION,
                "git_sha": _get_git_sha(),
            }
        )

    env = JaxToNumpy(env)

    video_writer = _open_video_writer(record, fps) if record else None
    ep_times: list[float | None] = []
    try:
        for episode_idx in range(n_runs):
            ep_time = _run_episode(
                env=env,
                controller_cls=controller_cls,
                cfg=cfg,
                video_writer=video_writer,
                camera=camera,
                width=width,
                height=height,
                trace_writer=trace_writer,
                episode_idx=episode_idx,
            )
            ep_times.append(ep_time)
    finally:
        if video_writer is not None and record is not None:
            video_writer.close()
            logger.info("wrote %s", Path(record).resolve())
        env.close()

    return ep_times


def _run_episode(
    env: gymnasium.Env,
    controller_cls: type[Controller],
    cfg: ConfigDict,
    video_writer: Any | None,
    camera: str,
    width: int,
    height: int,
    trace_writer: _TraceWriter | None = None,
    episode_idx: int = 0,
) -> float | None:
    """Run one episode end-to-end. Mirrors ``scripts.sim._run_episode``."""
    obs, info = env.reset()
    if trace_writer is not None:
        trace_writer.open_episode(
            episode_idx, {"spawn_pos": obs["pos"].tolist(), "spawn_quat": obs["quat"].tolist()}
        )
    try:
        controller: Controller = controller_cls(obs, info, cfg)
        fps_live_view = 60  # Hz cadence for the live MuJoCo viewer
        i = 0
        curr_time = 0.0
        prev_obs = obs

        while True:
            action = controller.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            i += 1
            curr_time = i / cfg.env.freq

            prev_tg = int(prev_obs["target_gate"])
            obs_tg = int(obs["target_gate"])
            gate_just_passed = (prev_tg >= 0) and (obs_tg != prev_tg)
            finished = (obs_tg == -1) and (prev_tg != -1)

            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )

            if video_writer is not None:
                frame = _grab_offscreen_frame(env, camera, width, height)
                if frame is not None:
                    video_writer.append_data(frame)
            elif cfg.sim.render:
                if ((i * fps_live_view) % cfg.env.freq) < fps_live_view:
                    controller.render_callback(env.unwrapped.sim)
                    env.render()

            if terminated or truncated or controller_finished or finished:
                break
            prev_obs = obs

        controller.episode_callback()
        _log_episode_stats(obs, cfg, curr_time)
        controller.episode_reset()
    finally:
        if trace_writer is not None:
            trace_writer.close_episode()

    return curr_time if obs["target_gate"] == -1 else None


def _color_code_gates(env: gymnasium.Env) -> None:
    """Override per-gate accent geoms with a distinct colour for each gate.

    Walks the MuJoCo model for geoms named ``<base>:<i>`` for each
    ``<base>`` in :data:`GATE_ACCENT_GEOMS` and each gate index ``i``,
    sets ``geom_matid = -1`` (disable the shared ``frame_mat`` /
    ``rope_mat`` / ``stand_mat`` materials), and writes the per-gate
    colour from :data:`GATE_COLORS` into ``geom_rgba``. The textured
    front and back panels (``front_*:i`` / ``back_*:i``) are left alone
    so the gate aperture is still legible.

    Naming convention comes from
    :meth:`lsy_drone_racing.envs.race_core.RaceCoreEnv._load_track_into_sim`,
    which attaches each gate body with suffix ``:<i>`` and propagates
    that suffix to all child geom names.

    Notes
    -----
    Modifies ``env.unwrapped.sim.mj_model`` in place. MuJoCo's offscreen
    renderer reads ``mj_model.geom_rgba``, so the change is reflected in
    every subsequent render call. The MJX physics path is unaffected
    because it does not consume geom colour.
    """
    sim = env.unwrapped.sim
    mj_model = sim.mj_model
    for gate_idx in range(GATE_COLORS.shape[0]):
        color = GATE_COLORS[gate_idx]
        for base_name in GATE_ACCENT_GEOMS:
            geom_name = f"{base_name}:{gate_idx}"
            try:
                geom = mj_model.geom(geom_name)
            except (KeyError, ValueError):
                continue  # gate <gate_idx> does not exist on this track
            if geom is None:
                continue
            mj_model.geom_matid[geom.id] = -1
            mj_model.geom_rgba[geom.id] = color


def _grab_offscreen_frame(
    env: gymnasium.Env, camera: str, width: int, height: int
) -> np.ndarray | None:
    """Capture one RGB frame from Crazyflow's offscreen MuJoCo renderer."""
    sim_core = env.unwrapped
    if not sim_core.data.sim_data.core.mjx_synced:
        sim_core.data, sim_core.sim.mjx_data = sim_core._render_sync(
            sim_core.data, sim_core.sim.mjx_data
        )
    frame = sim_core.sim.render(mode="rgb_array", camera=camera, width=width, height=height)
    if frame is None:
        return None
    return np.asarray(frame)


def _open_video_writer(path: str, fps: int) -> Any:
    """Open an imageio mp4 writer with H.264 defaults matching scripts/sim.py."""
    import imageio.v2 as imageio  # local import: optional dep

    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[3] / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    return imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8, macro_block_size=1)


def _log_episode_stats(obs: dict, cfg: ConfigDict, curr_time: float) -> None:
    """Log per-episode flight time / finished / gates-passed."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:
        gates_passed = len(cfg.env.track.gates)
    finished = gates_passed == len(cfg.env.track.gates)
    logger.info(
        "Flight time (s): %s\nFinished: %s\nGates passed: %s\n", curr_time, finished, gates_passed
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
