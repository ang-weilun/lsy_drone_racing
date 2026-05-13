"""Resumable RL training loop for spot/preemptible GPU instances.

Lifecycle:

1. Build agent + vec env from ``cfg`` via pluggable builders.
2. Download the latest checkpoint from the remote store (if any) and
   resume from it — same W&B run, same global_step, same RNG.
3. Run the rollout / update loop, saving every ``cfg.save_every`` steps.
4. On SIGTERM/SIGINT, finish the current iteration, save one last
   blocking checkpoint, and exit cleanly so the spot host can yank us.

Run it directly during development; in production it's launched by
docker/bootstrap.sh as PID 1 (via tini) so signals propagate.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import fire

from lsy_drone_racing.rl.builders import build_agent, build_vec_env
from lsy_drone_racing.rl.checkpoint import (
    CfgDriftError,
    CheckpointManager,
    CheckpointPayload,
    cfg_hash,
    restore_rng,
    snapshot_rng,
)
from lsy_drone_racing.rl.remote_sync import RcloneSync
from lsy_drone_racing.rl.signals import SignalGuard

log = logging.getLogger("train_resumable")


@dataclass
class TrainConfig:
    builder: str = "noop_smoke"
    """Builder name registered in ``lsy_drone_racing.rl.builders``."""

    total_steps: int = 100_000
    save_every: int = 10_000
    log_every: int = 1_000

    ckpt_dir: str = "checkpoints"
    """Local ckpt dir. Mounted-out on real instances, ephemeral in tests."""

    remote: str | None = None
    """rclone remote spec, e.g. ``r2:bucket/path``. ``None`` = local-only."""

    keep_last: int = 3
    upload_async: bool = True

    seed: int = 42
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    """Set ``"disabled"`` to skip W&B entirely (smoke tests use this)."""

    builder_kwargs: dict[str, Any] = field(default_factory=dict)
    """Forwarded to the builder so e.g. n_envs / obs_dim are configurable."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LSY_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _init_wandb(cfg: TrainConfig, run_id: str | None) -> tuple[Any | None, str | None]:
    """Initialize wandb; resume the same run if we have an id from ckpt."""
    if cfg.wandb_mode == "disabled" or cfg.wandb_project is None:
        return None, None
    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed — running without tracking.")
        return None, None
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        id=run_id,
        resume="allow",
        mode=cfg.wandb_mode,
        config=cfg.to_dict(),
    )
    return run, run.id


def _build_manager(cfg: TrainConfig) -> CheckpointManager:
    remote = RcloneSync(remote=cfg.remote) if cfg.remote else None
    return CheckpointManager(
        local_dir=Path(cfg.ckpt_dir),
        remote=remote,
        keep_last=cfg.keep_last,
        upload_async=cfg.upload_async,
    )


def _resume(
    manager: CheckpointManager, agent: Any, env: Any, cfg: TrainConfig
) -> tuple[int, str | None]:
    """Try to load the latest ckpt. Returns ``(global_step, wandb_run_id)``."""
    cfg_dict = {k: v for k, v in cfg.to_dict().items() if k not in _DRIFT_IGNORE}
    try:
        payload = manager.load_latest(cfg_dict)
    except CfgDriftError as e:
        log.error(str(e))
        raise

    if payload is None:
        log.info("Starting from scratch (no ckpt found).")
        return 0, None

    agent.load_state_dict(payload.agent_state)
    env.load_state_dict(payload.env_state)
    restore_rng(payload.rng_state)
    log.info(
        "Resumed from step=%d wandb_run_id=%s", payload.global_step, payload.wandb_run_id
    )
    return payload.global_step, payload.wandb_run_id


# Fields that legitimately differ run-to-run and shouldn't trip the
# drift check. Add to this list as your config grows.
_DRIFT_IGNORE: set[str] = {
    "ckpt_dir",
    "remote",
    "upload_async",
    "wandb_mode",
    "wandb_entity",
}


def _snapshot(
    cfg: TrainConfig, agent: Any, env: Any, step: int, wandb_run_id: str | None
) -> CheckpointPayload:
    cfg_dict = {k: v for k, v in cfg.to_dict().items() if k not in _DRIFT_IGNORE}
    return CheckpointPayload(
        global_step=step,
        agent_state=agent.state_dict(),
        env_state=env.state_dict(),
        rng_state=snapshot_rng(),
        wandb_run_id=wandb_run_id,
        cfg=cfg_dict,
        cfg_hash=cfg_hash(cfg_dict),
        replay_buffer=None,  # already inside agent.state_dict() for noop_smoke;
        # off-policy builders should populate this instead.
    )


def main(**overrides: Any) -> int:
    """Entry point. CLI overrides via fire override TrainConfig fields."""
    _setup_logging()
    cfg = TrainConfig(**overrides)
    log.info("Config:\n%s", json.dumps(cfg.to_dict(), indent=2, default=str))

    guard = SignalGuard()
    guard.install()

    # Seed before building anything so env/agent construction is reproducible.
    import random
    import numpy as np
    import torch

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = build_vec_env({"builder": cfg.builder, **cfg.builder_kwargs})
    agent = build_agent({"builder": cfg.builder, **cfg.builder_kwargs}, env)
    manager = _build_manager(cfg)

    global_step, prior_run_id = _resume(manager, agent, env, cfg)
    run, run_id = _init_wandb(cfg, prior_run_id)
    if run_id is None:
        run_id = prior_run_id

    obs = env.reset()
    last_save = global_step
    last_log = global_step
    t0 = time.time()
    log.info("Training start: global_step=%d total_steps=%d", global_step, cfg.total_steps)

    try:
        while global_step < cfg.total_steps and not guard.should_stop:
            action = agent.act(obs)
            obs, _, _, _ = env.step(action)
            metrics = agent.update(batch=None)
            global_step += 1

            if global_step - last_save >= cfg.save_every:
                manager.save(_snapshot(cfg, agent, env, global_step, run_id))
                last_save = global_step

            if global_step - last_log >= cfg.log_every:
                elapsed = time.time() - t0
                sps = (global_step - last_log) / max(elapsed, 1e-9)
                log.info("step=%d sps=%.1f metrics=%s", global_step, sps, metrics)
                if run is not None:
                    run.log({"step": global_step, "sps": sps, **metrics})
                last_log = global_step
                t0 = time.time()

        if guard.should_stop:
            log.warning("Stop requested — performing emergency blocking save.")
            manager.save_blocking(_snapshot(cfg, agent, env, global_step, run_id))
        else:
            log.info("Reached total_steps=%d — final save.", cfg.total_steps)
            manager.save_blocking(_snapshot(cfg, agent, env, global_step, run_id))
    finally:
        if run is not None:
            run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(fire.Fire(main))
