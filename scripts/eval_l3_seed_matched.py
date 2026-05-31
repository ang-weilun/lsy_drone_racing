"""Seed-matched deterministic full-DR evaluation for the SBX deploy controller.

Unlike ``scripts/sim.py`` / ``rl_song.eval_sim`` — which seed the env once at
construction and then call ``env.reset()`` without a per-episode seed — this
script resets every episode from a fixed seed list (``base_seed + k``). Because
``RaceCoreEnv._reset`` re-seeds the sim RNG (``seed_sim``) and derives the
track-randomization key from it, episode ``k`` reproduces the *same* randomized
layout for any checkpoint, regardless of how many sim steps prior episodes took.
That removes the RNG drift that makes the no-seed harnesses non-comparable
across policies, so the resulting SR / lap-time figures can be ranked directly.

Run as::

    pixi run -e rl-train python scripts/eval_l3_seed_matched.py \
        --checkpoint <run_or_step_dir> --config level3.toml \
        --controller rl_sbx/controller_numpy.py --control-mode attitude \
        --n-runs 100 --base-seed 0 --out results.json

The deploy controller (``rl_sbx/controller_numpy.py``) is deterministic (it runs
the actor mean), so the only source of episode variation is the seeded layout —
exactly what we want held fixed across checkpoints.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Lap-time percentiles reported over finished episodes only.
_LAP_PERCENTILES = (10, 25, 50, 75, 90)


def evaluate(
    checkpoint: str,
    *,
    config: str = "level3.toml",
    controller: str = "rl_sbx/controller_numpy.py",
    n_runs: int = 100,
    base_seed: int = 0,
    control_mode: str = "attitude",
    out: str | None = None,
) -> dict:
    """Run a seed-matched deterministic evaluation of one checkpoint.

    Parameters
    ----------
    checkpoint : str
        Run directory (latest ``step_*`` is auto-selected) or a concrete
        ``step_*`` directory, passed to the controller via
        ``config.controller.checkpoint``.
    config : str
        Config filename under ``config/`` (default ``level3.toml`` = full DR).
    controller : str
        Controller path under ``lsy_drone_racing/control/``.
    n_runs : int
        Number of seed-matched episodes (seeds ``base_seed .. base_seed+n-1``).
    base_seed : int
        First seed; episode ``k`` uses ``base_seed + k``.
    control_mode : str
        Env control mode (the SBX policy outputs attitude commands).
    out : str, optional
        If given, write the full result dict to this JSON path.

    Returns:
    -------
    result : dict
        Aggregate metrics plus the per-episode table.
    """
    cfg = load_config(REPO_ROOT / "config" / config)
    cfg.sim.render = False
    cfg.env.control_mode = control_mode
    cfg.controller.checkpoint = checkpoint

    controller_path = REPO_ROOT / "lsy_drone_racing" / "control" / controller
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

    n_gates = len(cfg.env.track.gates)
    seeds = np.arange(base_seed, base_seed + n_runs, dtype=int)
    flight_times = np.full(n_runs, np.nan, dtype=float)
    gates_passed = np.zeros(n_runs, dtype=int)
    finished = np.zeros(n_runs, dtype=bool)

    try:
        for k, seed in enumerate(seeds):
            ep_time, ep_gates, ep_finished = _run_episode(
                env, controller_cls, cfg, int(seed), n_gates
            )
            gates_passed[k] = ep_gates
            finished[k] = ep_finished
            if ep_finished:
                flight_times[k] = ep_time
            logger.info(
                "seed=%d finished=%s gates=%d/%d time=%s",
                seed,
                ep_finished,
                ep_gates,
                n_gates,
                f"{ep_time:.3f}" if ep_finished else "—",
            )
    finally:
        env.close()

    result = _aggregate(
        checkpoint, config, seeds, flight_times, gates_passed, finished, n_gates
    )
    _print_summary(result)
    if out is not None:
        Path(out).write_text(json.dumps(result, indent=2))
        logger.info("wrote %s", Path(out).resolve())
    return result


def _run_episode(
    env: gymnasium.Env,
    controller_cls: type[Controller],
    cfg: ConfigDict,
    seed: int,
    n_gates: int,
) -> tuple[float, int, bool]:
    """Run one seeded episode; return (flight_time, gates_passed, finished)."""
    obs, info = env.reset(seed=seed)
    controller = controller_cls(obs, info, cfg)
    i = 0
    while True:
        action = controller.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        controller_finished = controller.step_callback(
            action, obs, reward, terminated, truncated, info
        )
        if terminated or truncated or controller_finished:
            break
        i += 1
    controller.episode_callback()
    controller.episode_reset()

    flight_time = i / cfg.env.freq
    target_gate = int(obs["target_gate"])
    finished = target_gate == -1
    gates = n_gates if finished else target_gate
    return flight_time, gates, finished


def _aggregate(
    checkpoint: str,
    config: str,
    seeds: np.ndarray,
    flight_times: np.ndarray,
    gates_passed: np.ndarray,
    finished: np.ndarray,
    n_gates: int,
) -> dict:
    """Build the aggregate metrics dict from per-episode arrays."""
    n_runs = int(seeds.size)
    n_finished = int(finished.sum())
    finished_times = flight_times[finished]
    lap_stats: dict[str, float] = {}
    if n_finished > 0:
        lap_stats = {
            "mean": float(np.mean(finished_times)),
            "std": float(np.std(finished_times)),
            "min": float(np.min(finished_times)),
            "max": float(np.max(finished_times)),
        }
        for pct in _LAP_PERCENTILES:
            lap_stats[f"p{pct}"] = float(np.percentile(finished_times, pct))

    gate_hist = {g: int(np.sum(gates_passed == g)) for g in range(n_gates + 1)}
    return {
        "checkpoint": checkpoint,
        "config": config,
        "n_runs": n_runs,
        "base_seed": int(seeds[0]),
        "n_gates": n_gates,
        "success_rate": n_finished / n_runs,
        "n_finished": n_finished,
        "lap_time_finished_only": lap_stats,
        "gates_passed_hist": gate_hist,
        "per_episode": [
            {
                "seed": int(seeds[k]),
                "finished": bool(finished[k]),
                "gates_passed": int(gates_passed[k]),
                "flight_time": (
                    float(flight_times[k]) if finished[k] else None
                ),
            }
            for k in range(n_runs)
        ],
    }


def _print_summary(result: dict) -> None:
    """Print a compact human-readable summary."""
    lap = result["lap_time_finished_only"]
    lines = [
        "",
        f"=== seed-matched eval: {result['checkpoint']} ===",
        f"config={result['config']}  n_runs={result['n_runs']}  "
        f"base_seed={result['base_seed']}",
        f"SR = {result['success_rate']:.1%} "
        f"({result['n_finished']}/{result['n_runs']})",
    ]
    if lap:
        lines.append(
            "lap (finished only): "
            f"mean={lap['mean']:.3f}s  median={lap['p50']:.3f}s  "
            f"min={lap['min']:.3f}s  p10={lap['p10']:.3f}s  "
            f"p90={lap['p90']:.3f}s  std={lap['std']:.3f}s"
        )
    lines.append(f"gates_passed histogram: {result['gates_passed_hist']}")
    logger.info("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fire.Fire(evaluate)
