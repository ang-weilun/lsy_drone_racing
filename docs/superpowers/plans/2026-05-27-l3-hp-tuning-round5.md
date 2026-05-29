# Round 5 — L3 HP-Tuning Sequenced Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a 200M-step sample-efficiency probe with `n_epochs=3 → 6` and unchanged other HPs to test whether the under-utilized PPO trust-region budget can be cashed into a steeper L3 finish-rate slope.

**Architecture:** Single-lever ablation: one-line diff (`--n-epochs 6`) on the existing `run_sbx_redesign_L3stage4dr_round*.sh` launcher template. Warm-start from the round-4 final ckpt (1.4B cumulative steps). Live abort gate on `train/approx_kl`, `train/value_loss`, `train/explained_variance` during the first 5-10 iters. Post-run slope evaluation vs the +10 pp / 400M baseline.

**Tech Stack:** SBX (`sbx-rl >= 0.18.0`) + patched `JitScanPPO` + Flax actor/critic. Training runs on vast.ai (RTX 5090, `192.3.91.246:25482`). Eval on the same box via the fast numpy controller (parity 1.19e-7 vs JAX).

**Spec:** `docs/superpowers/specs/2026-05-27-l3-hp-tuning-design.md`. Read it before starting.

---

## File Structure

| File | Where | Purpose |
|---|---|---|
| `run_sbx_redesign_L3stage4dr_round5.sh` | vast `/root/` | Round-5 launcher (created in Task 1) |
| `training_logs/sbx_redesign_warm1400_L3stage4dr_n6_200M.log` | vast | Run log (auto-created by launcher) |
| `lsy_drone_racing/control/rl_sbx/checkpoints/sbx_redesign_warm1400_L3stage4dr_n6_200M/` | vast | 10 intermediate ckpts + final (auto-created) |
| `eval_round5_multilevel.py` | dev VM `/tmp` → vast `/root/lsy_drone_racing/scripts/` | n=20 deterministic eval per level (Task 6) |
| `/root/renders/round5_*.mp4` | vast | Visual renders (Task 6) |

**Read-only constraint:** Dev VM's `.git` is mounted read-only. All edits to in-tree files stay uncommitted on dev VM; vast files are session-scoped and not versioned. Spec + plan + launcher scripts already in working tree are sufficient artifacts.

---

## Task 1: Write and launch the round-5 launcher

**Files:**
- Create: vast `/root/run_sbx_redesign_L3stage4dr_round5.sh`

- [ ] **Step 1: Verify vast box is alive and idle**

Run:
```
ssh -p 25482 -o StrictHostKeyChecking=no root@192.3.91.246 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; pgrep -af "rl_sbx.train" | head -3'
```

Expected: GPU memory ≤ ~200 MiB used, utilization 0 %, no `rl_sbx.train` process running.
If a training process is running, abort and investigate before proceeding.

- [ ] **Step 2: Confirm the round-4 final ckpt is on vast**

Run:
```
ssh -p 25482 -o StrictHostKeyChecking=no root@192.3.91.246 'ls /root/lsy_drone_racing/lsy_drone_racing/control/rl_sbx/checkpoints/sbx_redesign_warm1000_L3stage4dr_400M/step_000402653184/'
```

Expected: `actor.params.msgpack`, `actor_normalizer.json`, `critic.params.msgpack`, `critic_normalizer.json`, `policy_config.json` all listed.
If missing, the warm-start source is corrupted — STOP and tell the user.

- [ ] **Step 3: Write the launcher script on vast**

Run (HEREDOC into ssh; matches round-4 template exactly except the `--n-epochs 6` line and the run-name / init-from / total-timesteps):

```bash
ssh -p 25482 -o StrictHostKeyChecking=no root@192.3.91.246 'cat > /root/run_sbx_redesign_L3stage4dr_round5.sh << "EOF"
#!/usr/bin/env bash
set -u
export PATH=$HOME/.pixi/bin:$PATH
cd /root/lsy_drone_racing
mkdir -p training_logs
pixi run -e rl-train python -m lsy_drone_racing.control.rl_sbx.train \
    --run-name sbx_redesign_warm1400_L3stage4dr_n6_200M \
    --total-timesteps 200000000 \
    --save-freq-steps 20000000 \
    --ent-coef 0.005 --ent-coef-final 0.001 \
    --alpha-max-rad 0.32 \
    --progress-coef 15.0 \
    --omega-coef 0.01 \
    --time-penalty 0 --guide-coef 0 --gate-pass-bonus 0 \
    --gate-frame-weight 0 --obstacle-weight 0 \
    --wrong-side-coef 0 --dipole-coef 0 \
    --curriculum full --stage-idx 5 \
    --n-epochs 6 \
    --init-from lsy_drone_racing/control/rl_sbx/checkpoints/sbx_redesign_warm1000_L3stage4dr_400M/step_000402653184 \
    > training_logs/sbx_redesign_warm1400_L3stage4dr_n6_200M.log 2>&1
echo TRAIN_EXIT=$? >> training_logs/sbx_redesign_warm1400_L3stage4dr_n6_200M.log
EOF
chmod +x /root/run_sbx_redesign_L3stage4dr_round5.sh
echo wrote'
```

Expected: `wrote` on stdout. No errors.

- [ ] **Step 4: Diff round-5 launcher against round-4 to verify the only change is `--n-epochs 6`**

Run:
```
ssh -p 25482 -o StrictHostKeyChecking=no root@192.3.91.246 'diff /root/run_sbx_redesign_L3stage4dr_round4.sh /root/run_sbx_redesign_L3stage4dr_round5.sh'
```

Expected diff: only the `--run-name`, `--total-timesteps`, `--init-from`, and `--n-epochs 6` lines differ. No other changes.
If unexpected lines differ (e.g., a reward weight changed), STOP and re-write the launcher.

- [ ] **Step 5: Launch the run detached via setsid -f**

Per `[[feedback-rl-sbx-setsid]]` — `nohup` alone is not enough; `setsid -f` is mandatory so the process survives ssh disconnect.

Run:
```
ssh -p 25482 -o StrictHostKeyChecking=no root@192.3.91.246 'setsid -f /root/run_sbx_redesign_L3stage4dr_round5.sh < /dev/null > /dev/null 2>&1; sleep 3; pgrep -af "rl_sbx.train" | head -3'
```

Expected: at least one process matching `pixi run -e rl-train python -m lsy_drone_racing.control.rl_sbx.train --run-name sbx_redesign_warm1400_L3stage4dr_n6_200M ...`.
If no process appears: read the log immediately and diagnose; likely a missing-flag error.

---

## Task 2: First-iter validation

**Files:**
- Read: vast `training_logs/sbx_redesign_warm1400_L3stage4dr_n6_200M.log`

- [ ] **Step 1: Wait ~30 seconds, then tail the log for iter 1 metrics**

Run:
```
ssh -p 25482 -o StrictHostKeyChecking=no root@192.3.91.246 'sleep 30 && tail -80 /root/lsy_drone_racing/training_logs/sbx_redesign_warm1400_L3stage4dr_n6_200M.log'
```

Expected: a wandb init banner, plus PPO hyperparam dump (look for `"n_epochs": 6` in the config printout — confirms the flag took effect), plus the first iter's metric table showing `iterations | 1` and `total_timesteps | 4194304` (one rollout buffer).

If `n_epochs` is logged as anything other than 6: `pkill -f rl_sbx.train` and re-launch with the correct flag — the run is wasted.

- [ ] **Step 2: Confirm reward terms match the redesign 5-term composition**

Visually scan the iter-1 metric table from Step 1 for the `reward/` block. Expected non-zero terms: `r_prog`, `r_omega`, `r_crash`, `r_finish` (and `r_terminal` as the sum). Expected zero terms: `r_caution`, `r_dipole`, `r_exit_vel`, `r_gate_bonus`, `r_gate_frame`, `r_guid`, `r_obs`, `r_smooth`, `r_time`, `r_vel`, `r_wrong_side`.

If any of the "expected zero" terms are non-zero: a reward-weight CLI flag was set incorrectly. Kill, fix the launcher, re-launch.

---

## Task 3: Mid-run health check (early abort gate)

**Files:**
- Read: vast wandb run page (or training log if wandb unavailable)

This is the **load-bearing abort decision point.** If the run is going wrong, catching it at iter 5-10 saves ~40 min of bad compute.

- [ ] **Step 1: Wait until iter ~5 (after ~5 minutes of wall time)**

Iter wall is roughly 50-60 s with `n_epochs=6` per the spec's revised estimate. Wait ~5 min from launch, then check progress.

Run:
```
ssh -p 25482 -o StrictHostKeyChecking=no root@192.3.91.246 'grep -E "iterations|total_timesteps|approx_kl|clip_fraction|explained_variance|value_loss" /root/lsy_drone_racing/training_logs/sbx_redesign_warm1400_L3stage4dr_n6_200M.log | tail -40'
```

- [ ] **Step 2: Apply the abort gate**

Look at the most recent iter's reported values:

| Metric | Healthy | Yellow flag | Abort condition |
|---|---|---|---|
| `approx_kl` | < 0.02 | 0.02-0.05 | > 0.05 sustained for 2+ iters |
| `clip_fraction` | < 0.25 | 0.25-0.40 | > 0.40 sustained |
| `value_loss` | flat or decreasing across iters | flat-ish | monotonically increasing across 3+ iters |
| `explained_variance` | > 0.5 | 0.3-0.5 | < 0.3 sustained |

If any "Abort condition" cell is hit:
```
ssh -p 25482 -o StrictHostKeyChecking=no root@192.3.91.246 'pkill -f rl_sbx.train; sleep 2; pgrep -af rl_sbx.train'
```
Then re-do Task 1 with `--n-epochs 4` (smaller step) as the conservative fallback, OR revert to round-4 final and discuss with user.

If all metrics are Healthy or only one is Yellow (KL alone yellow is the most common, often resolves itself by iter 10): proceed to Task 4.

- [ ] **Step 3: Repeat the gate at iter ~10**

Wait another ~5 min, re-run Step 1's grep, re-apply the gate. If still healthy, the run can be left to complete.

---

## Task 4: Run completion

**Files:**
- Read: vast `training_logs/sbx_redesign_warm1400_L3stage4dr_n6_200M.log`

- [ ] **Step 1: Wait for the run to finish**

Total expected wall: ~50 min from launch per the spec. Use ScheduleWakeup to come back, don't poll aggressively.

If running interactively, periodic check:
```
ssh -p 25482 -o StrictHostKeyChecking=no root@192.3.91.246 'grep TRAIN_EXIT /root/lsy_drone_racing/training_logs/sbx_redesign_warm1400_L3stage4dr_n6_200M.log; tail -3 /root/lsy_drone_racing/training_logs/sbx_redesign_warm1400_L3stage4dr_n6_200M.log'
```

Expected on completion: `TRAIN_EXIT=0`. If non-zero: read the full log tail and diagnose.

- [ ] **Step 2: Verify all 10 intermediate ckpts saved + final ckpt**

Run:
```
ssh -p 25482 -o StrictHostKeyChecking=no root@192.3.91.246 'ls /root/lsy_drone_racing/lsy_drone_racing/control/rl_sbx/checkpoints/sbx_redesign_warm1400_L3stage4dr_n6_200M/'
```

Expected: 10 intermediate `step_*` directories (every 20M steps from 20M to 200M) + a final `step_*` directory matching the actual total step count.

---

## Task 5: Slope evaluation against the gate

**Files:**
- Read: vast log + wandb summary

- [ ] **Step 1: Pull the wandb-style end-of-run summary**

Run:
```
ssh -p 25482 -o StrictHostKeyChecking=no root@192.3.91.246 'tail -80 /root/lsy_drone_racing/training_logs/sbx_redesign_warm1400_L3stage4dr_n6_200M.log | sed -n "/Run summary/,/Synced/p"'
```

Expected: a wandb summary block with `env/finish_rate_true_start`, `env/mean_target_gate`, etc.

Capture the value of `env/finish_rate_true_start`. Reference round-4 ending value was 0.34241.

- [ ] **Step 2: Compute slope vs the baseline gate**

The spec's pass condition: per-200M improvement ≥ +7 pp (1.4× the baseline +10 pp / 400M = +5 pp / 200M).

Baseline (round 4): finish_rate at end of 400M was 0.34241. So a 200M increment under the baseline recipe would be expected to be ~0.34241 + 0.05 ≈ 0.39.

Round 5 gate: `env/finish_rate_true_start` at end of 200M should be ≥ 0.41 (=0.34241 + 0.07) to count as a 1.4× speedup.

| Round 5 end finish_rate_true_start | Verdict |
|---|---|
| ≥ 0.41 | PASS — n_epochs=6 helped; proceed to round 6 (layer LR=5e-4) |
| 0.36 - 0.41 | MARGINAL — slope unchanged or small win; rerun with `n_epochs=10` instead OR proceed to round 6 with both layered carefully |
| < 0.36 | FAIL — n_epochs=6 did not help; revert to `n_epochs=3` and try a different lever (batch_size or LR alone) |

Capture the verdict.

- [ ] **Step 3: Sanity-check the per-iter trajectory for monotonicity**

Run:
```
ssh -p 25482 -o StrictHostKeyChecking=no root@192.3.91.246 'tail -120 /root/lsy_drone_racing/training_logs/sbx_redesign_warm1400_L3stage4dr_n6_200M.log | grep -A1 "env/finish_rate_true_start" | head -3'
```

This pulls the wandb sparkline. Expected: monotonic ▁→█ or close to it. A noisy non-monotonic sparkline + a high end value would mean we got lucky, not that the lever helped — flag for additional analysis before proceeding to round 6.

---

## Task 6: Post-run eval + render

**Files:**
- Create: `/tmp/eval_round5_multilevel.py` (dev VM) then push to vast `/root/lsy_drone_racing/scripts/eval_round5_multilevel.py`
- Output: `/root/renders/round5_L{0,1,2,3}.mp4` (vast)

This mirrors what was done for round 4. The artifacts are for the user to inspect + for memory.

- [ ] **Step 1: Write the eval script on dev VM**

Create `/tmp/eval_round5_multilevel.py`:

```python
"""Headless multi-level eval for the round-5 redesign ckpt."""

from __future__ import annotations

import logging
import statistics

from sim import simulate

CONTROLLER: str = "rl_sbx/controller_numpy.py"
CKPT: str = (
    "lsy_drone_racing/control/rl_sbx/checkpoints/"
    "sbx_redesign_warm1400_L3stage4dr_n6_200M/step_000201326592"
)
N_RUNS: int = 20
LEVELS: tuple[str, ...] = (
    "level0.toml",
    "level1.toml",
    "level2.toml",
    "level3.toml",
)


def main() -> None:
    """Run n=20 deterministic eval per level, print summary."""
    logging.basicConfig(level=logging.INFO)
    print()
    print(f"==== ckpt: {CKPT} ====")
    print(f"==== n_runs={N_RUNS} deterministic, numpy controller ====")
    for cfg in LEVELS:
        ep_times = simulate(
            config=cfg,
            controller=CONTROLLER,
            n_runs=N_RUNS,
            render=False,
            checkpoint=CKPT,
            control_mode="attitude",
        )
        finished = [t for t in ep_times if t is not None]
        pretty_times = [f"{t:.2f}" if t is not None else "CRASH" for t in ep_times]
        print()
        print(f"---- {cfg} ----")
        print(
            f"finished: {len(finished)}/{N_RUNS} "
            f"({100 * len(finished) / N_RUNS:.1f}%)"
        )
        if finished:
            print(
                "lap times (s): "
                f"min={min(finished):.2f}, "
                f"mean={statistics.mean(finished):.2f}, "
                f"median={statistics.median(finished):.2f}, "
                f"max={max(finished):.2f}"
            )
        print(f"per-episode: {pretty_times}")


if __name__ == "__main__":
    main()
```

**Verify the `CKPT` path matches the actual final step directory from Task 4 Step 2.** If the run did not produce exactly `step_000201326592`, edit the `CKPT` constant before running.

- [ ] **Step 2: Push the eval script to vast**

Run:
```
scp -P 25482 -o StrictHostKeyChecking=no /tmp/eval_round5_multilevel.py root@192.3.91.246:/root/lsy_drone_racing/scripts/eval_round5_multilevel.py
```

Expected: one file transferred, no errors.

- [ ] **Step 3: Launch the eval in background**

Run:
```
ssh -p 25482 -o StrictHostKeyChecking=no root@192.3.91.246 'cat > /root/run_eval_round5.sh << "EOF"
#!/usr/bin/env bash
set -u
export PATH=$HOME/.pixi/bin:$PATH
export JAX_PLATFORMS=cpu
export SCIPY_ARRAY_API=1
cd /root/lsy_drone_racing
pixi run -e rl-train python scripts/eval_round5_multilevel.py > /root/eval_round5.log 2>&1
echo EVAL_EXIT=$? >> /root/eval_round5.log
EOF
chmod +x /root/run_eval_round5.sh
setsid -f /root/run_eval_round5.sh < /dev/null > /dev/null 2>&1
sleep 3
pgrep -af eval_round5_multilevel | head -3'
```

Expected: a `pixi run ... eval_round5_multilevel.py` process visible.

- [ ] **Step 4: Launch the render in background (parallel with eval)**

Run:
```
ssh -p 25482 -o StrictHostKeyChecking=no root@192.3.91.246 'cat > /root/run_render_round5.sh << "EOF"
#!/usr/bin/env bash
set -u
export PATH=$HOME/.pixi/bin:$PATH
export JAX_PLATFORMS=cpu
cd /root/lsy_drone_racing
CKPT=lsy_drone_racing/control/rl_sbx/checkpoints/sbx_redesign_warm1400_L3stage4dr_n6_200M/step_000201326592
mkdir -p /root/renders
for LVL in 0 1 2 3; do
  echo "=== rendering level${LVL} ==="
  pixi run -e rl-train python scripts/sim.py \
      --config level${LVL}.toml \
      --controller rl_sbx/controller.py \
      --control_mode attitude \
      --checkpoint $CKPT \
      --record /root/renders/round5_L${LVL}.mp4 \
      --n_runs 2 2>&1 | tail -20
done
echo "=== DONE ==="
ls -lh /root/renders/round5_*.mp4
EOF
chmod +x /root/run_render_round5.sh
setsid -f /root/run_render_round5.sh < /dev/null > /root/render_round5.log 2>&1
sleep 3
pgrep -af "run_render_round5\|sim.py" | head -3'
```

Expected: a `bash /root/run_render_round5.sh` or `pixi run ... sim.py` process. Render will sequentially work through L0 → L1 → L2 → L3.

- [ ] **Step 5: Wait for both eval and render to finish**

Eval typically ~5-15 min (numpy controller); render typically ~5-10 min (4 levels × 2 episodes JAX). Total ~15 min combined.

Periodic check:
```
ssh -p 25482 -o StrictHostKeyChecking=no root@192.3.91.246 'grep EVAL_EXIT /root/eval_round5.log; grep "=== DONE ===" /root/render_round5.log'
```

Expected on completion: both `EVAL_EXIT=0` and `=== DONE ===` present.

- [ ] **Step 6: Pull eval summary**

Run:
```
ssh -p 25482 -o StrictHostKeyChecking=no root@192.3.91.246 'grep -E "^----|finished:|lap times|per-episode|EVAL_EXIT" /root/eval_round5.log'
```

Expected: a 4-level summary like the round-4 table. Capture the L3 finish-rate value — this is the post-hoc n=20 deterministic confirmation of the round-5 outcome.

- [ ] **Step 7: Pull MP4s back to dev VM and send to user**

Run:
```
mkdir -p /tmp/round5_renders && rsync -av -e "ssh -p 25482 -o StrictHostKeyChecking=no" root@192.3.91.246:/root/renders/round5_*.mp4 /tmp/round5_renders/
```

Then use `SendUserFile` to deliver the 4 MP4s with a caption mentioning the L3 finish rate from Step 6. Per `[[reference_gdrive_rclone]]`, delete `/tmp/round5_renders/` after sending.

---

## Task 7: Document the outcome

**Files:**
- Update auto-memory under `/home/exedev/.claude/projects/-home-exedev/memory/`

- [ ] **Step 1: Decide whether the result warrants a new memory file**

If round-5 finish rate ≥ 0.41 (passed the slope gate), write a memory file for the recipe + outcome. If marginal or failed, update existing `[[project-v83s40M-crosslevel-sota]]` / `[[project-v77s155M-l0l1-sota]]`-style memory only if this is a new SOTA.

Memory criteria (from the `auto memory` rules):
- Save only if **surprising or not obvious from code**.
- Do NOT save the launcher diff or HP values — those are recoverable from `git blame` on the launcher.
- DO save: whether `n_epochs=6` actually delivered the predicted speedup, and the verdict that informs round 6.

- [ ] **Step 2: If saving, follow the auto-memory format**

Create `feedback_n_epochs_lever.md` (or `project_v???-n6-result.md`, depending on whether it became SOTA) under the memory dir, with the standard frontmatter (`name`, `description`, `metadata.type: feedback` or `project`), and a body that leads with the rule/fact + `**Why:**` + `**How to apply:**`. Link to `[[project-v83s40M-crosslevel-sota]]` and the spec.

Add the corresponding one-line index entry to `MEMORY.md`.

---

## Self-review

Pass against the spec at `docs/superpowers/specs/2026-05-27-l3-hp-tuning-design.md`:

**1. Spec coverage:**
- Spec §"Round 5 launcher" → Task 1
- Spec §"Verification gate (during the run)" → Tasks 2 + 3
- Spec §"Rollback condition" → Task 3 Step 2 (abort path)
- Spec §"Post-round-5 selection" → Task 5 (slope verdict feeds the round-6 decision; not in this plan because round 6 is conditional)
- Spec §"Out of scope" → respected (no r_smooth, no α_max, no critic-normalizer changes)

All in-scope spec sections covered.

**2. Placeholder scan:** No TBDs, no "implement later", no "add appropriate error handling", no "similar to Task N" without code. Every step has the exact shell command or python code it needs.

**3. Type / name consistency:**
- Run name: `sbx_redesign_warm1400_L3stage4dr_n6_200M` used consistently in launcher (Task 1), log path (Tasks 2-5), ckpt path (Task 6).
- Final ckpt path: `step_000201326592` matches 200M total timesteps (1024 × 4194304 / 200000000 ≈ 48 iters × 4194304 = 201,326,592). Consistent across Tasks 5 + 6.
- Round-4 final ckpt path used as warm-start: `sbx_redesign_warm1000_L3stage4dr_400M/step_000402653184` matches Task-1 Step 2 verify.

Plan complete.
