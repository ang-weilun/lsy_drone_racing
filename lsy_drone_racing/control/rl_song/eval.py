"""Evaluate a deterministic RL Song checkpoint on one racing rollout."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import numpy as np
import orbax.checkpoint as ocp
import tyro
from jax import Array

from lsy_drone_racing.control.rl_song.config import TrainConfig
from lsy_drone_racing.control.rl_song.env_wrapper import RLSongVecEnv
from lsy_drone_racing.control.rl_song.obs import NormalizerState
from lsy_drone_racing.control.rl_song.policy import deterministic_raw_action

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
CHECKPOINT_PREFIX: str = "step_"
N_EVAL_ENVS: int = 1


@dataclass(frozen=True)
class EvalArgs:
    """Command-line arguments for deterministic evaluation."""

    checkpoint: Path
    stage: int = 1
    seed: int = 0
    no_render: bool = False


def main() -> None:
    """Parse CLI arguments and run one deterministic rollout."""
    args = tyro.cli(EvalArgs)
    evaluate(args)


def evaluate(args: EvalArgs) -> dict[str, float]:
    """Run one deterministic rollout from an Orbax checkpoint.

    Parameters
    ----------
    args : EvalArgs
        Evaluation CLI arguments. ``stage`` is one-indexed.

    Returns
    -------
    metrics : dict[str, float]
        Episode return, length, target gate reached, and reward-component sums.
    """
    if args.stage < 1:
        raise ValueError(f"stage must be one-indexed and positive; got {args.stage}")

    checkpoint = _restore_checkpoint(_resolve_checkpoint_path(args.checkpoint))
    actor_params = checkpoint["actor_params"]
    normalizer = _normalizer_from_checkpoint(checkpoint["normalizer"])
    train_cfg = replace(
        TrainConfig(),
        seed=args.seed,
        initial_stage_index=args.stage - 1,
    )
    env = RLSongVecEnv(
        train_cfg,
        n_envs=N_EVAL_ENVS,
        stage_idx=args.stage - 1,
        seed=args.seed,
        device="gpu",
    )
    env.set_normalizer(normalizer)
    obs, _ = env.reset(seed=args.seed)

    episode_return = 0.0
    episode_length = 0
    target_gate_reached = 0.0
    component_sums: dict[str, float] = {}

    for _ in range(train_cfg.max_episode_steps):
        raw_action = _deterministic_raw_action(actor_params, obs["actor_obs"])
        obs, reward, terminated, truncated, info = env.step(raw_action)
        episode_return += float(np.asarray(reward[0]))
        episode_length += 1
        target_gate_reached = max(
            target_gate_reached,
            float(np.asarray(info["target_gate_progress"][0])),
        )
        for key, value in info["reward_components"].items():
            component_sums[key] = component_sums.get(key, 0.0) + float(
                np.asarray(value[0])
            )
        if not args.no_render:
            env.render()
        if bool(np.asarray((terminated | truncated)[0])):
            break

    env.close()
    metrics = {
        "episode_return": episode_return,
        "episode_length": float(episode_length),
        "target_gate_reached": target_gate_reached,
    }
    metrics.update(component_sums)
    _print_metrics(metrics)
    return metrics


@jax.jit
def _deterministic_raw_action(actor_params: dict[str, Any], actor_obs: Array) -> Array:
    """Return the actor mean raw action for a batched observation."""
    return deterministic_raw_action(actor_params, actor_obs)


def _print_metrics(metrics: dict[str, float]) -> None:
    """Print rollout metrics in a stable order."""
    ordered_keys = [
        "episode_return",
        "episode_length",
        "target_gate_reached",
        "r_prog",
        "r_omega",
        "r_obs",
        "r_gate_bonus",
        "r_terminal",
    ]
    for key in ordered_keys:
        if key in metrics:
            print(f"{key}: {metrics[key]:.6g}")


def _resolve_checkpoint_path(checkpoint: Path) -> Path:
    """Resolve either a checkpoint directory or a run directory."""
    path = checkpoint.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if path.is_dir() and path.name.startswith(CHECKPOINT_PREFIX):
        return path
    latest = _latest_checkpoint_path(path)
    if latest is None:
        raise ValueError(f"No Orbax checkpoint directories found under: {path}")
    return latest


def _latest_checkpoint_path(run_dir: Path) -> Path | None:
    """Return the newest ``step_*`` checkpoint directory under ``run_dir``."""
    if not run_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.iterdir():
        if not path.is_dir() or not path.name.startswith(CHECKPOINT_PREFIX):
            continue
        step_str = path.name.removeprefix(CHECKPOINT_PREFIX)
        if step_str.isdecimal():
            candidates.append((int(step_str), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _restore_checkpoint(path: Path) -> dict[str, Any]:
    """Restore an Orbax checkpoint directory."""
    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(path)


def _normalizer_from_checkpoint(data: dict[str, Array]) -> NormalizerState:
    """Restore a frozen observation normalizer from checkpoint data."""
    return NormalizerState(
        mean=data["mean"],
        var=data["var"],
        count=data["count"],
    )


if __name__ == "__main__":
    main()
