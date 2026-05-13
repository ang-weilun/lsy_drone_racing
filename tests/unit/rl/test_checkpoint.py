"""Unit tests for the resumable-training checkpoint helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lsy_drone_racing.rl.checkpoint import (
    CKPT_GLOB,
    CfgDriftError,
    CheckpointManager,
    CheckpointPayload,
    cfg_hash,
    restore_rng,
    snapshot_rng,
)


def _payload(step: int, cfg: dict) -> CheckpointPayload:
    return CheckpointPayload(
        global_step=step,
        agent_state={"w": torch.zeros(3)},
        env_state={"running_mean": np.zeros(2)},
        rng_state=snapshot_rng(),
        wandb_run_id="abc123",
        cfg=cfg,
        cfg_hash=cfg_hash(cfg),
    )


@pytest.mark.unit
def test_save_then_load_round_trip(tmp_path):
    cfg = {"lr": 1e-3, "n_envs": 2}
    mgr = CheckpointManager(local_dir=tmp_path, remote=None, keep_last=3)
    mgr.save(_payload(100, cfg))
    loaded = mgr.load_latest(cfg)
    assert loaded is not None
    assert loaded.global_step == 100
    assert loaded.wandb_run_id == "abc123"


@pytest.mark.unit
def test_cfg_drift_refuses_resume(tmp_path):
    cfg_a = {"lr": 1e-3, "n_envs": 2}
    cfg_b = {"lr": 5e-4, "n_envs": 2}  # different lr -> different hash
    mgr = CheckpointManager(local_dir=tmp_path, remote=None, keep_last=3)
    mgr.save(_payload(50, cfg_a))
    with pytest.raises(CfgDriftError):
        mgr.load_latest(cfg_b)


@pytest.mark.unit
def test_prune_keeps_only_last_n(tmp_path):
    cfg = {"lr": 1e-3}
    mgr = CheckpointManager(local_dir=tmp_path, remote=None, keep_last=2)
    for step in (10, 20, 30, 40):
        mgr.save(_payload(step, cfg))
    remaining = sorted(p.name for p in tmp_path.glob(CKPT_GLOB))
    assert remaining == ["ckpt-000000000030.pt", "ckpt-000000000040.pt"]


@pytest.mark.unit
def test_atomic_write_leaves_no_tmp_files(tmp_path):
    cfg = {"lr": 1e-3}
    mgr = CheckpointManager(local_dir=tmp_path, remote=None, keep_last=3)
    mgr.save(_payload(1, cfg))
    leftover_tmp = list(tmp_path.glob(".*.tmp"))
    assert leftover_tmp == []


@pytest.mark.unit
def test_rng_round_trip_is_deterministic():
    np.random.seed(0)
    torch.manual_seed(0)
    state = snapshot_rng()

    np.random.seed(0)
    expected = np.random.randn(5)
    expected_torch = torch.randn(5)

    # Burn entropy then restore — subsequent draws should match the baseline.
    _ = np.random.randn(10)
    _ = torch.randn(10)
    restore_rng(state)
    assert np.allclose(np.random.randn(5), expected)
    assert torch.allclose(torch.randn(5), expected_torch)
