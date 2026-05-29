"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Pass ``--record path/to/clip.mp4`` to capture an offscreen video instead of
opening the live MuJoCo viewer (e.g. for a headless training box). The optional
``--camera`` arg picks the MuJoCo camera; ``track_cam:0`` is the third-person
drone-tracking view and ``fpv_cam:0`` is onboard first-person.

Look for instructions in ``README.md`` and in the official documentation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv


logger = logging.getLogger(__name__)

# Offscreen frame capture defaults. ``Sim`` caches its renderer on the first
# call (including camera id, width, height), so all frames share these values.
DEFAULT_RECORD_CAMERA = "track_cam:0"
DEFAULT_RECORD_FPS = 50  # matches the 50 Hz env step so playback is real-time
DEFAULT_RECORD_WIDTH = 1280
DEFAULT_RECORD_HEIGHT = 720


def simulate(
    config: str = "level0.toml",
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
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Parameters
    ----------
    config : str
        Race config filename under ``config/``. Default ``level0.toml``.
    controller : str, optional
        Override ``config.controller.file``. Path is resolved under
        ``lsy_drone_racing/control/``.
    n_runs : int
        Number of episodes to run.
    render : bool, optional
        Enable / disable the live MuJoCo viewer. ``None`` (default) inherits
        ``config.sim.render``. Forced to ``False`` when ``record`` is set.
    record : str, optional
        Path to an mp4 file. If set, captures one frame per env step from
        Crazyflow's offscreen renderer and writes them to disk instead of
        opening the live viewer. The directory is created if missing.
    camera : str
        MuJoCo camera name used when recording. ``track_cam:0`` follows drone
        0 from a third-person view; ``fpv_cam:0`` is onboard.
    fps : int
        Output video frame rate. The env steps at ``config.env.freq`` (50 Hz
        for the standard configs), so the default ``fps=50`` plays back in
        real time.
    width, height : int
        Offscreen render resolution.
    checkpoint : str, optional
        Override / inject ``config.controller.checkpoint`` for controllers
        that read it (e.g. the RL Song controller). Resolved by the
        controller, not by this script.
    control_mode : str, optional
        Override ``config.env.control_mode`` (``"state"`` or ``"attitude"``).
        Needed when running an attitude-output controller (e.g. an RL policy)
        against a state-mode config without editing the toml.

    Returns
    -------
    ep_times : list[float]
        One entry per episode: flight time if the drone finished, ``None``
        otherwise.
    """
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    if render is None:
        render = cfg.sim.render
    if record is not None:
        render = False  # offscreen capture replaces the live viewer
    cfg.sim.render = render

    if control_mode is not None:
        cfg.env.control_mode = control_mode
    if checkpoint is not None:
        cfg.controller.checkpoint = checkpoint

    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or cfg.controller.file)
    controller_cls = load_controller(controller_path)

    env: DroneRaceEnv = gymnasium.make(
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
    env = JaxToNumpy(env)

    video_writer: Any = _open_video_writer(record, fps) if record else None
    ep_times = []
    try:
        for _ in range(n_runs):
            ep_time, _ = _run_episode(
                env=env,
                controller_cls=controller_cls,
                cfg=cfg,
                video_writer=video_writer,
                camera=camera,
                width=width,
                height=height,
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
) -> tuple[float | None, int]:
    """Run one episode. Capture frames if ``video_writer`` is provided."""
    obs, info = env.reset()
    controller: Controller = controller_cls(obs, info, cfg)
    fps_live_view = 60  # Hz cadence for the live MuJoCo viewer
    i = 0
    curr_time = 0.0
    n_frames = 0

    while True:
        curr_time = i / cfg.env.freq
        action = controller.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        controller_finished = controller.step_callback(
            action, obs, reward, terminated, truncated, info
        )

        if video_writer is not None:
            # First frame's viewer doesn't exist yet (Sim.render lazily creates
            # it), so the draw_* helpers no-op silently and the first frame
            # has no overlay. Subsequent frames overlay correctly.
            controller.render_callback(env.unwrapped.sim)
            frame = _grab_offscreen_frame(env, camera, width, height)
            if frame is not None:
                video_writer.append_data(frame)
                n_frames += 1
        elif cfg.sim.render:
            if ((i * fps_live_view) % cfg.env.freq) < fps_live_view:
                controller.render_callback(env.unwrapped.sim)
                env.render()

        if terminated or truncated or controller_finished:
            break
        i += 1

    controller.episode_callback()
    _log_episode_stats(obs, info, cfg, curr_time)
    controller.episode_reset()
    ep_time = curr_time if obs["target_gate"] == -1 else None
    return ep_time, n_frames


def _grab_offscreen_frame(
    env: gymnasium.Env, camera: str, width: int, height: int
) -> np.ndarray | None:
    """Capture one RGB frame from Crazyflow's offscreen MuJoCo renderer.

    Mirrors ``RaceCoreEnv.render``'s mjx→mj_data sync, but calls
    ``Sim.render`` directly in ``rgb_array`` mode so no viewer window opens.
    """
    sim_core = env.unwrapped
    if not sim_core.data.sim_data.core.mjx_synced:
        sim_core.data, sim_core.sim.mjx_data = sim_core._render_sync(
            sim_core.data, sim_core.sim.mjx_data
        )
    frame = sim_core.sim.render(
        mode="rgb_array", camera=camera, width=width, height=height
    )
    if frame is None:
        return None
    return np.asarray(frame)


def _open_video_writer(path: str, fps: int) -> object:
    """Open an imageio mp4 writer with sensible H.264 defaults."""
    import imageio.v2 as imageio  # local import: optional dep

    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Headless boxes need an offscreen GL backend for MuJoCo.
    os.environ.setdefault("MUJOCO_GL", "egl")
    return imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,  # avoids the "not multiple of 16" warning
    )


def _log_episode_stats(
    obs: dict, info: dict, cfg: ConfigDict, curr_time: float
) -> None:
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:
        gates_passed = len(cfg.env.track.gates)
    finished = gates_passed == len(cfg.env.track.gates)
    logger.info(
        "Flight time (s): %s\nFinished: %s\nGates passed: %s\n",
        curr_time,
        finished,
        gates_passed,
    )


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Public-name shim retained for callers that import ``log_episode_stats``."""
    _log_episode_stats(obs, info, config, curr_time)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
