r"""Sim eval for the Song-2023 RL controller with the level-3 dead-obs patch.

``scripts/sim.py`` is the upstream eval entrypoint, but on level 3 it feeds
the controller an observation where ``obs["gates_pos"]`` /
``obs["obstacles_pos"]`` return the toml-nominal ``(0, 0, z)`` placeholders
for any gate/obstacle not yet within sensor range. The framework's
``build_full_track_randomization_fn`` regenerates ``env.data.gates_pos`` per
episode but leaves ``env.data.nominal_gates_pos`` unchanged, so the
un-visited branch leaks dead info instead of the true placement.

The training wrapper :class:`RLSongVecEnv` substitutes the truth from
``env.data.gates_pos`` for the un-visited branch, so the trained policy
sees the placement and never the placeholder. ``scripts/sim.py`` does not
apply that patch, which is why level-3 sim eval shows the drone reacting
to phantom ``(0, 0, z)`` gates instead of the real layout.

This script wraps the env in :class:`TruePoseObsWrapper` to re-apply the
``gates_visited`` / ``obstacles_visited`` mask with ``env.unwrapped.data``
as the truth source. On real hardware the issue does not arise because
Mocap populates ``nominal_*_pos`` with the true measured positions at
reset (see :mod:`lsy_drone_racing.envs.real_race_env`), so this wrapper
is a no-op there. The base ``envs/``, ``scripts/sim.py``, and config files
are intentionally untouched so the competition code-check sees no diff.

Usage
-----
::

    pixi run -e rl-train python -m lsy_drone_racing.control.rl_song.eval_sim \\
        --config level3.toml \\
        --checkpoint <path-to-step_NNNNNNNNNNNN-or-run-dir> \\
        --control_mode attitude \\
        --record renders/v24_level3_patched.mp4 \\
        --n_runs 8
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


class TruePoseObsWrapper(gymnasium.ObservationWrapper):
    """Patch the un-visited branch of the per-step obs with the placed truth.

    Re-applies the ``gates_visited`` / ``obstacles_visited`` mask using
    ``env.unwrapped.data`` as the truth source for the un-visited branch.

    See module docstring for the bug this works around. Apply *after*
    :class:`JaxToNumpy` so the obs is numpy; the truth is read from
    ``env.unwrapped.data`` (JAX arrays) and converted on the fly.

    Notes
    -----
    ``env.unwrapped.data.gates_pos`` has shape ``(n_drones, n_gates, 3)``.
    The single-drone env this script targets has ``n_drones = 1``, so the
    ``[0]`` slice matches the unbatched per-drone obs layout.
    """

    def observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return an obs dict with the un-visited branch sourced from truth."""
        data = self.unwrapped.data
        truth_gates_pos = np.asarray(data.gates_pos[0])
        truth_gates_quat = np.asarray(data.gates_quat[0])
        truth_obstacles_pos = np.asarray(data.obstacles_pos[0])

        out = dict(observation)
        gates_visited = np.asarray(out["gates_visited"]).astype(bool)
        mask_g = gates_visited[..., None]
        out["gates_pos"] = np.where(mask_g, out["gates_pos"], truth_gates_pos)
        out["gates_quat"] = np.where(mask_g, out["gates_quat"], truth_gates_quat)
        obstacles_visited = np.asarray(out["obstacles_visited"]).astype(bool)
        mask_o = obstacles_visited[..., None]
        out["obstacles_pos"] = np.where(mask_o, out["obstacles_pos"], truth_obstacles_pos)
        return out


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
) -> list[float | None]:
    """Run the RL Song controller in sim with the dead-obs patch.

    Mirrors ``scripts.sim.simulate`` so the recording / CLI behaviour is
    identical; the only functional difference is the
    :class:`TruePoseObsWrapper` inserted after :class:`JaxToNumpy`.

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

    Returns
    -------
    ep_times : list[float | None]
        One entry per episode: flight time if the drone finished, ``None``
        otherwise.
    """
    repo_root = Path(__file__).resolve().parents[3]
    cfg = load_config(repo_root / "config" / config)
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
    control_path = repo_root / "lsy_drone_racing" / "control"
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
    env = JaxToNumpy(env)
    env = TruePoseObsWrapper(env)

    video_writer = _open_video_writer(record, fps) if record else None
    ep_times: list[float | None] = []
    try:
        for _ in range(n_runs):
            ep_time = _run_episode(
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
) -> float | None:
    """Run one episode end-to-end. Mirrors ``scripts.sim._run_episode``."""
    obs, info = env.reset()
    controller: Controller = controller_cls(obs, info, cfg)
    fps_live_view = 60  # Hz cadence for the live MuJoCo viewer
    i = 0
    curr_time = 0.0

    while True:
        curr_time = i / cfg.env.freq
        action = controller.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
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

        if terminated or truncated or controller_finished:
            break
        i += 1

    controller.episode_callback()
    _log_episode_stats(obs, cfg, curr_time)
    controller.episode_reset()
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
