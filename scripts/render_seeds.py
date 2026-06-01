"""Render specific seeded episodes to a video — for inspecting failure cases.

Unlike ``scripts/sim.py`` (which seeds once at construction and resets without a
per-episode seed), this resets each episode with an explicit seed so it
reproduces the exact layouts that ``scripts/eval_l3_seed_matched.py`` scored —
letting you render the specific seeds that failed in eval. Frames are captured
via the same offscreen path as ``sim.py``.

Usage
-----
    pixi run -e rl-train python scripts/render_seeds.py \
        --checkpoint <step_dir> --config level3.toml \
        --controller rl_sbx/controller_numpy.py --control-mode attitude \
        --seeds 2,3,4,0,5,7 --record /tmp/fails.mp4 --camera track_cam:0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gymnasium
import imageio.v2 as imageio
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

logger = logging.getLogger(__name__)

# Match sim.py's offscreen capture defaults.
_FRAME_WIDTH = 1280
_FRAME_HEIGHT = 720


def _grab_frame(env: gymnasium.Env, camera: str) -> np.ndarray | None:
    """Capture one RGB frame from Crazyflow's offscreen renderer (mirrors sim.py)."""
    sim_core = env.unwrapped
    if not sim_core.data.sim_data.core.mjx_synced:
        sim_core.data, sim_core.sim.mjx_data = sim_core._render_sync(
            sim_core.data, sim_core.sim.mjx_data
        )
    frame = sim_core.sim.render(
        mode="rgb_array", camera=camera, width=_FRAME_WIDTH, height=_FRAME_HEIGHT
    )
    return None if frame is None else np.asarray(frame)


def render_seeds(
    checkpoint: str,
    seeds: str,
    record: str,
    config: str = "level3.toml",
    controller: str = "rl_sbx/controller_numpy.py",
    control_mode: str = "attitude",
    camera: str = "track_cam:0",
    fps: int = 50,
) -> None:
    """Render each given seed as one episode, concatenated into a single video.

    Parameters
    ----------
    checkpoint : str
        Checkpoint step directory passed to the controller.
    seeds : str
        Comma-separated seed list, e.g. ``"2,3,4,0,5,7"``.
    record : str
        Output mp4 path.
    config, controller, control_mode, camera, fps
        As in ``scripts/sim.py``.
    """
    # Fire parses ``--seeds 2,3,4`` into a tuple; ``--seeds 2`` into an int; and
    # ``--seeds "2,3,4"`` into a str. Normalize all three to a list[int].
    if isinstance(seeds, (list, tuple)):
        seed_list = [int(s) for s in seeds]
    elif isinstance(seeds, int):
        seed_list = [int(seeds)]
    else:
        seed_list = [int(s) for s in str(seeds).split(",") if s.strip() != ""]
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.controller.checkpoint = checkpoint
    cfg.env.control_mode = control_mode
    cfg.sim.render = False

    control_path = Path(__file__).parents[1] / "lsy_drone_racing" / "control"
    controller_cls = load_controller(control_path / controller)

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
    env = JaxToNumpy(env)
    writer = imageio.get_writer(record, fps=fps, macro_block_size=1)

    try:
        for seed in seed_list:
            obs, info = env.reset(seed=seed)
            ctrl = controller_cls(obs, info, cfg)
            i = 0
            while True:
                action = ctrl.compute_control(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                done = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
                ctrl.render_callback(env.unwrapped.sim)
                frame = _grab_frame(env, camera)
                if frame is not None:
                    writer.append_data(frame)
                if terminated or truncated or done:
                    break
                i += 1
            gate = int(np.asarray(obs["target_gate"]).item())
            finished = gate == -1
            logger.info(
                "seed=%d finished=%s gate=%s steps=%d", seed, finished, gate, i
            )
            ctrl.episode_callback()
            ctrl.episode_reset()
    finally:
        writer.close()
        env.close()
        logger.info("wrote %s", Path(record).resolve())


if __name__ == "__main__":
    import fire

    logging.basicConfig(level=logging.INFO)
    fire.Fire(render_seeds)
