# L3 HP-tuning: sequenced sample-efficiency probe

**Date:** 2026-05-27
**Branch:** `rl/reward-fix-2026-05-25`
**Status:** approved by user 2026-05-27 ~20:30 UTC; pending spec review

## Goal

Reach **>50 % L3 deterministic finish rate** in fewer training steps than
the baseline +10 pp / 400M slope would deliver. Current state: 30 % L3 det
@ 1.4B cumulative steps (round-4 final ckpt).

## Diagnosis

End-of-round-3 PPO diagnostics show massive trust-region slack —
`approx_kl=0.006` against `clip_range=0.2` and `clip_fraction=7 %`. The
policy is barely using the gradient budget per rollout. Codex verified
(2026-05-27 read-only audit) that `JitScanPPO.train()` performs exactly
`n_epochs × n_minibatches` optimizer steps per iter. At the current
`n_epochs=3`, `n_envs × n_steps / batch_size = 256` minibatches gives
**768 updates per iter**; the SBX/SB3 default of `n_epochs=10` would give
2560 (3.33×). We are leaving an obvious lever on the table.

## Decision

**Sequenced single-lever ablation, codex-aligned.**

| Round | Change | Other HPs | Pass condition to advance |
|---|---|---|---|
| **Round 5** | `n_epochs: 3 → 6` | unchanged | Per-iter `finish_rate_true_start` slope clearly steeper than baseline +10 pp/400M, AND `approx_kl < 0.02` median, AND `explained_variance > 0.5`, AND `value_loss` not increasing monotonically across iters. |
| Round 6 (conditional) | Layer `LR: 3e-4 → 5e-4` on top of `n_epochs=6` | unchanged | Same gates as round 5. |
| (reserve) | `batch_size: 16384 → 8192` | — | Only if round 5 + round 6 do not break the +10 pp/400M ceiling. |

Anti-pattern explicitly avoided: setting `target_kl`. The SBX
`KLAdaptiveLR` callback saturates LR to `max_learning_rate=1e-2` on our
recipe (768 calls/iter); this melted v110 ([[feedback-sbx-target-kl]]).
`train.py:333-348` already pins `target_kl=None`; do not change.

## Round 5 launcher

One-line addition (`--n-epochs 6`) to the existing
`run_sbx_redesign_L3stage4dr_round*.sh` template. Other knobs unchanged:

- `--total-timesteps 200000000` (probe length; cheaper-faster signal than
  400M while still letting the slope assert itself for 50+ iters)
- `--save-freq-steps 20000000` (10 intermediate ckpts, same as rounds 1-4)
- `--ent-coef 0.005 --ent-coef-final 0.001`
- `--alpha-max-rad 0.32`
- `--progress-coef 15.0`
- `--omega-coef 0.01`
- `--time-penalty 0 --guide-coef 0 --gate-pass-bonus 0
  --gate-frame-weight 0 --obstacle-weight 0 --wrong-side-coef 0
  --dipole-coef 0`
- `--curriculum full --stage-idx 5` (stage4_level3_dr)
- `--init-from .../sbx_redesign_warm1000_L3stage4dr_400M/step_000402653184`
- **NEW: `--n-epochs 6`**

Expected wall: ~40-60 min on the 5090. Round-3 observed throughput was
~105k sps with `n_epochs=3` (1905 s for 200M). Doubling `n_epochs` to 6
doubles the gradient-update phase of each iter (rollout phase unchanged),
so end-to-end sps will drop. Conservative estimate ~60-70k sps with
`n_epochs=6` → ~50 min. Cost: ~$0.25.

## Verification gate (during the run)

Watch on wandb live, not post-hoc:

- `train/approx_kl` — must stay < 0.02 median across iters. If it climbs
  past 0.05 sustained, **abort** and revert to `n_epochs=3`. The mean is
  fine but per-minibatch can be more aggressive — codex caveat.
- `train/clip_fraction` — should remain below 25 %.
- `train/value_loss` — should not increase monotonically across iters
  (would indicate the critic falling behind due to `clip_range_vf=0.2`
  cutting off too many of its update steps).
- `train/explained_variance` — should not dip below 0.4 sustained.
- `env/finish_rate_true_start` — slope is the metric of interest.
  Baseline reference: rounds 1-4 hit +10 pp per 400M = ~+5 pp per 200M.
  Round 5 success = ≥ +7 pp per 200M (1.4× speedup) at minimum to justify
  layering the LR bump in round 6.

## Rollback condition

If round 5 regresses the policy (finish rate at end of run < 30 % or
KL/value_loss blow up): keep the round-4 final ckpt as the warm-start
source for any future attempt; do NOT continue from the broken round-5
ckpt. The 1.4B ckpt is intact at
`sbx_redesign_warm1000_L3stage4dr_400M/step_000402653184`.

## Post-round-5 selection

If round 5 succeeds: warm-start round 6 from the best round-5 intermediate
ckpt (post-hoc selection across 10 intermediates), NOT necessarily the
final. Per [[project-v56s163-sota]], the final step of a continuation run
is not always the best.

## Out of scope (for this design)

- `r_smooth_coef` activation (Kaufmann action-delta penalty; wired but
  inactive). Sim2real lever; separate experiment.
- α_max bump (speed lever; v53 SOTA pattern). Speed redesign is a
  different optimization target.
- Critic normalizer separation (codex review §2 of the original
  redesign). Risk mitigation, not sample-efficiency.
- Mixed-curriculum L0/L1/L2 regression recovery. Eval at end of round 4
  showed no regression (L0=100 %, L1=100 %, L2=35 %), so this is not
  needed.

## Commit constraint

Dev VM's `.git` is mounted read-only. This spec lives in the working tree
uncommitted; the human commits from a different host. Round 5's launcher
script lives on vast (`/root/run_sbx_redesign_*.sh`) per session pattern.
