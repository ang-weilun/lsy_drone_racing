"""Sweep controller-tuning configurations on a fixed seed list.

Used to record baseline + tuning-sweep metrics for SfcAttitudeController.
Differs from sweep_dump.py: no per-tick .npz dumps, supports --override
KEY=VAL flags that monkey-patch controller-module constants before runs.

Usage:
    pixi run python scripts/sweep_controller.py \\
        --n_seeds 100 --seed_start 0 \\
        --out_csv /tmp/baseline.csv

    pixi run python scripts/sweep_controller.py \\
        --n_seeds 100 --seed_start 0 \\
        --overrides "TILT_LIMIT=0.7;TILT_RATE_LIMIT=0.5" \\
        --planner_overrides "TILT_LIMIT_PLANNER=0.5" \\
        --out_csv /tmp/tilt07_rate05.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import ModuleType

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

logger = logging.getLogger(__name__)


def _apply_overrides(target: object, overrides: tuple[str, ...]) -> dict[str, object]:
    """Monkey-patch float / ndarray attributes on a module or class in-place.

    Works for both module globals (controller constants) and class attributes
    (SfcPlanner tunables). Returns {name: value} actually set, for logging.
    """
    if not overrides:
        return {}
    applied: dict[str, object] = {}
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"override must be KEY=VAL, got {kv!r}")
        k, v = kv.split("=", 1)
        if not hasattr(target, k):
            raise AttributeError(f"{target!r} has no attribute {k!r}")
        cur = getattr(target, k)
        if isinstance(cur, np.ndarray):
            new = np.array([float(x) for x in v.split(",")], dtype=cur.dtype)
            if new.shape != cur.shape:
                raise ValueError(f"shape mismatch for {k}: {new.shape} vs {cur.shape}")
        else:
            new = float(v)
        setattr(target, k, new)
        applied[k] = new
    return applied


def sweep(
    config: str = "level2.toml",
    n_seeds: int = 100,
    seed_start: int = 0,
    out_csv: str = "/tmp/sweep.csv",
    controller: str | None = None,
    overrides: str = "",
    planner_overrides: str = "",
) -> None:
    """Run n_seeds deterministic episodes; record pass/time/clearance per seed.

    Args:
        config: TOML config in config/.
        n_seeds: Number of episodes (seeds = seed_start..seed_start+n_seeds-1).
        seed_start: First seed (inclusive).
        out_csv: Output CSV path.
        controller: Override controller filename. None = use config default.
        overrides: Semicolon-separated KEY=VAL list applied to the controller
            module (e.g. --overrides "TILT_LIMIT=0.7;KP=0.8,0.8,1.25"). Vector
            values use comma-separated floats inside the value.
        planner_overrides: Same syntax, but applied as class attributes on
            SfcPlanner (e.g. --planner_overrides "TILT_LIMIT_PLANNER=0.5").
    """
    override_parts = tuple(p for p in overrides.split(";") if p.strip())
    planner_override_parts = tuple(p for p in planner_overrides.split(";") if p.strip())

    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.sim.render = False

    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or cfg.controller.file)
    controller_cls = load_controller(controller_path)

    # load_controller registers the file under sys.modules["controller"], not
    # under its package path — the controller class's globals point at THAT
    # module. Patch it there or overrides are silent no-ops.
    ctrl_mod = sys.modules["controller"]
    applied = _apply_overrides(ctrl_mod, override_parts)
    if applied:
        logger.info("controller overrides applied: %s", applied)

    if planner_override_parts:
        from lsy_drone_racing.control.sfc_planner import SfcPlanner
        applied_p = _apply_overrides(SfcPlanner, planner_override_parts)
        logger.info("planner overrides applied: %s", applied_p)

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

    n_gates_total = len(cfg.env.track.gates)
    rows = ["seed,finished,gates_passed,reason,episode_time,min_clearance"]
    for k in range(n_seeds):
        seed = seed_start + k

        obs, info = env.reset(seed=seed)
        controller_inst = controller_cls(obs, info, cfg)

        # Track horizontal min-clearance to obstacle poles. Obstacles are
        # vertical cylinders, so distance is computed in xy only.
        min_clear = float("inf")

        i = 0
        terminated = truncated = controller_finished = False
        while True:
            action = controller_inst.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)

            obs_pos_xy = np.asarray(obs["pos"])[:2]
            obstacles_xy = np.asarray(obs["obstacles_pos"])[:, :2]
            d = np.linalg.norm(obstacles_xy - obs_pos_xy[None, :], axis=1).min()
            if d < min_clear:
                min_clear = float(d)

            controller_finished = controller_inst.step_callback(
                action, obs, reward, terminated, truncated, info
            )
            if terminated or truncated or controller_finished:
                break
            i += 1
        curr_time = i / cfg.env.freq
        controller_inst.episode_callback()

        gates_passed = int(obs["target_gate"])
        if gates_passed == -1:
            gates_passed = n_gates_total
            reason = "finished"
        elif truncated:
            reason = "timeout"
        else:
            reason = "collision"
        finished = reason == "finished"

        rows.append(
            f"{seed},{int(finished)},{gates_passed},{reason},"
            f"{curr_time:.3f},{min_clear:.4f}"
        )
        logger.info(
            "seed=%d gates=%d reason=%s t=%.2f clear=%.3f",
            seed, gates_passed, reason, curr_time, min_clear,
        )

        controller_inst.episode_reset()

    env.close()

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n")

    finished_times = [
        float(r.split(",")[4]) for r in rows[1:] if r.split(",")[3] == "finished"
    ]
    pass_rate = len(finished_times) / n_seeds
    mean_time = float(np.mean(finished_times)) if finished_times else float("nan")
    logger.info(
        "summary: pass=%.0f%% mean_t=%.2fs n=%d csv=%s",
        pass_rate * 100, mean_time, n_seeds, out_path,
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(sweep, serialize=lambda _: None)
