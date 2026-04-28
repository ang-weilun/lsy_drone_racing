"""Sweep N seeds with the SFC planner, dumping per-tick + per-replan state to .npz files.

Usage:
    pixi run python scripts/sweep_dump.py --config level2.toml --n_seeds 30 --out_dir /tmp/sfc_sweep

Set SFC_DUMP=1 is done automatically.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import fire
import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

logger = logging.getLogger(__name__)


def sweep(
    config: str = "level2.toml",
    n_seeds: int = 30,
    out_dir: str = "/tmp/sfc_sweep",
    seed_start: int = 0,
    controller: str | None = None,
) -> None:
    """Run n_seeds episodes with explicit seeds, dumping each to <out_dir>/run_<seed>.npz.

    Args:
        config: TOML config name in config/.
        n_seeds: Number of episodes to run.
        out_dir: Directory for per-episode dumps + sweep summary.
        seed_start: First seed (inclusive). Sweep covers seed_start..seed_start+n_seeds-1.
        controller: Override controller filename. None = use config's default.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    os.environ["SFC_DUMP"] = "1"

    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = False

    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or cfg.controller.file)
    controller_cls = load_controller(controller_path)

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

    summary_lines = ["seed,gates_passed,terminated_reason,episode_time"]
    for k in range(n_seeds):
        seed = seed_start + k
        os.environ["SFC_DUMP_PATH"] = str(out_path / f"run_{seed:04d}.npz")

        obs, info = env.reset(seed=seed)
        controller_inst = controller_cls(obs, info, cfg)

        i = 0
        terminated = truncated = controller_finished = False
        while True:
            action = controller_inst.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = controller_inst.step_callback(
                action, obs, reward, terminated, truncated, info
            )
            if terminated or truncated or controller_finished:
                break
            i += 1
        curr_time = i / cfg.env.freq
        controller_inst.episode_callback()

        gates_passed = obs["target_gate"]
        if gates_passed == -1:
            gates_passed = len(cfg.env.track.gates)
            reason = "finished"
        elif truncated:
            reason = "timeout"
        else:
            reason = "collision"

        summary_lines.append(f"{seed},{int(gates_passed)},{reason},{curr_time:.3f}")
        logger.info(
            "seed=%d gates=%d reason=%s t=%.2f", seed, int(gates_passed), reason, curr_time
        )

        controller_inst.episode_reset()

    env.close()
    (out_path / "summary.csv").write_text("\n".join(summary_lines) + "\n")
    logger.info("wrote summary to %s", out_path / "summary.csv")


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(sweep, serialize=lambda _: None)
