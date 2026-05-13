"""Atomic, resumable checkpoints for RL training on spot instances.

A checkpoint bundles everything needed to resume bit-identically modulo
the unavoidable non-determinism of GPU kernels:

* model weights + optimizer state (torch state_dicts)
* replay buffer (for off-policy algos)
* env normalization running stats
* global_step
* RNG state (numpy, torch, jax)
* wandb run id (so we resume the same run row)
* config snapshot (so we can refuse to resume on cfg drift)

Files are named ``ckpt-{step:012d}.pt`` so lexicographic sort matches
training order. Writes are atomic: ``.tmp`` -> fsync -> rename. We keep
the last ``keep_last`` locally and the same number on the remote.

JAX RNG handling: ``jax.random.key`` instances are PRNGKeyArrays which
torch can't pickle directly. We convert to a uint32 numpy array on save
and rebuild a key on load. If the trainer doesn't use JAX RNG it can
pass ``jax_rng=None``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lsy_drone_racing.rl.remote_sync import RcloneSync

log = logging.getLogger(__name__)

CKPT_GLOB = "ckpt-*.pt"
CKPT_FMT = "ckpt-{step:012d}.pt"


def _ckpt_step(path: Path) -> int:
    return int(path.stem.split("-")[1])


def _hash_cfg(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass
class CheckpointPayload:
    """Everything we serialize. Kept as a dataclass for IDE help."""

    global_step: int
    agent_state: dict[str, Any]
    env_state: dict[str, Any]
    rng_state: dict[str, Any]
    wandb_run_id: str | None
    cfg: dict[str, Any]
    cfg_hash: str
    replay_buffer: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointManager:
    """Save/load/prune. Owns the local ckpt dir and (optionally) a remote."""

    local_dir: Path
    remote: RcloneSync | None = None
    keep_last: int = 3
    upload_async: bool = True

    def __post_init__(self) -> None:
        self.local_dir = Path(self.local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

    def save(self, payload: CheckpointPayload) -> Path:
        """Atomic local save + (optional) remote upload + prune."""
        name = CKPT_FMT.format(step=payload.global_step)
        final = self.local_dir / name
        # Atomic write: write to a sibling .tmp in the same dir so the
        # rename is on the same filesystem and therefore atomic.
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{name}.", suffix=".tmp", dir=str(self.local_dir)
        )
        tmp = Path(tmp_path)
        try:
            with os.fdopen(fd, "wb") as fh:
                torch.save(self._as_dict(payload), fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, final)
            # fsync the directory so the rename is durable on crash.
            self._fsync_dir(self.local_dir)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
            raise

        log.info("Saved checkpoint %s (step=%d)", final.name, payload.global_step)

        if self.remote is not None:
            if self.upload_async:
                self.remote.upload_async(final)
            else:
                self.remote.upload_sync(final)

        self._prune_local()
        if self.remote is not None:
            self.remote.prune_remote(self.keep_last)
        return final

    def save_blocking(self, payload: CheckpointPayload) -> Path:
        """Same as save() but waits for the remote upload before returning.

        Used by the SIGTERM emergency path so we don't lose the ckpt to
        the instance going away.
        """
        original_async = self.upload_async
        self.upload_async = False
        try:
            return self.save(payload)
        finally:
            self.upload_async = original_async

    def load_latest(self, current_cfg: dict[str, Any]) -> CheckpointPayload | None:
        """Pull from remote (if newer) then load the newest local ckpt.

        Refuses to return a payload whose cfg_hash differs from
        ``current_cfg`` — caller decides whether to abort or override.
        """
        if self.remote is not None:
            with contextlib.suppress(Exception):
                self.remote.download_latest(self.local_dir, glob=CKPT_GLOB)
        local = sorted(self.local_dir.glob(CKPT_GLOB), key=_ckpt_step)
        if not local:
            log.info("No local checkpoints in %s", self.local_dir)
            return None
        path = local[-1]
        log.info("Loading checkpoint %s", path)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        payload = self._from_dict(raw)

        expected = _hash_cfg(current_cfg)
        if payload.cfg_hash != expected:
            raise CfgDriftError(
                f"Refusing to resume {path.name}: config hash mismatch.\n"
                f"  ckpt cfg hash:    {payload.cfg_hash}\n"
                f"  current cfg hash: {expected}\n"
                "Either fix the config or move the old ckpts aside."
            )
        return payload

    def _prune_local(self) -> None:
        ckpts = sorted(self.local_dir.glob(CKPT_GLOB), key=_ckpt_step)
        for old in ckpts[: -self.keep_last]:
            log.info("Pruning local ckpt %s", old.name)
            with contextlib.suppress(FileNotFoundError):
                old.unlink()

    @staticmethod
    def _fsync_dir(p: Path) -> None:
        # Directory fsync is a no-op on Windows but required on Linux to
        # make the rename durable across power loss.
        try:
            fd = os.open(str(p), os.O_DIRECTORY)
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _as_dict(p: CheckpointPayload) -> dict[str, Any]:
        return {
            "global_step": p.global_step,
            "agent_state": p.agent_state,
            "env_state": p.env_state,
            "rng_state": p.rng_state,
            "wandb_run_id": p.wandb_run_id,
            "cfg": p.cfg,
            "cfg_hash": p.cfg_hash,
            "replay_buffer": p.replay_buffer,
            "extras": p.extras,
        }

    @staticmethod
    def _from_dict(d: dict[str, Any]) -> CheckpointPayload:
        return CheckpointPayload(
            global_step=d["global_step"],
            agent_state=d["agent_state"],
            env_state=d["env_state"],
            rng_state=d["rng_state"],
            wandb_run_id=d.get("wandb_run_id"),
            cfg=d["cfg"],
            cfg_hash=d["cfg_hash"],
            replay_buffer=d.get("replay_buffer"),
            extras=d.get("extras", {}),
        )


class CfgDriftError(RuntimeError):
    """Raised when a ckpt was produced under a different config."""


def snapshot_rng() -> dict[str, Any]:
    """Capture python/numpy/torch (+ jax if present) RNG state."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import jax

        # Use a fixed-seed key as a fingerprint; if the trainer uses JAX
        # PRNG it should overwrite jax_key via extras with its real key.
        state["jax_key"] = np.asarray(jax.random.key_data(jax.random.key(0)))
    except ImportError:
        pass
    return state


def restore_rng(state: dict[str, Any]) -> None:
    """Inverse of ``snapshot_rng``."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    # JAX key is informational only — trainers using JAX RNG should
    # round-trip their own key via the extras dict.


def cfg_hash(cfg: dict[str, Any]) -> str:
    """Public re-export."""
    return _hash_cfg(cfg)
