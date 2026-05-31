# Autonomous campaign — obstacle-barrier activation + frontier push (2026-05-31 PM2)

> Autoresearch loop log. Goal: push L3 seed-matched **SR (↑)** and **lap mean (↓)**
> past the converged frontier. Each iteration = one reward/training change →
> warm-start finetune on vast box 38672277 (RTX 5090) → seed-matched eval
> (n=100 select seeds 0–99, n=100 held-out 100–199) → keep/discard vs frontier.

## Verify
```
pixi run -e rl-train python scripts/eval_l3_seed_matched.py \
  --checkpoint <step_dir> --config level3.toml \
  --controller rl_sbx/controller_numpy.py --control-mode attitude \
  --n-runs 100 --base-seed {0|100} --out <json>
```

## Frontier (baseline, from PM convergence-campaign handoff; seed-matched)
| pick | α / tp | SR (n≈200) | lap mean | min | role |
|---|---|---:|---:|---:|---|
| relB | 0.8 / 0.25 | ~77 % | 5.2 s | 3.42 | reliability |
| relBfast ⭐ | 1.1 / 0.30 | ~74 % | 4.94 s | 3.26 | balanced |
| spd8 | 1.4 / 0.40 | ~71 % | 4.74 s | 3.36 | speed |

Targets: leader 3.8 s / 70 %; teammate MPCC 5.5 s / 60 %.

## Root-cause finding (iteration 0)
The active training reward is **only** `r_prog + r_omega + r_smooth + r_crash +
r_finish + r_time + r_gate_frame`. The obstacle barrier `r_obs` is computed but
was **never summed** into `reward` — commit `de3364b` ("add --use-obstacle-barrier
CLI flag") wired only the CLI flag, which flips the gate *inside* the component;
the component itself was never added to the sum line. So every campaign run that
passed `--use-obstacle-barrier=True --obstacle-weight=0.05` trained with a **no-op
obstacle term**, and 0.05 is 12× below the v34-validated `obstacle_weight=0.6`
(`config.py:700`: w=0.6 → ~−0.58/step danger-zone penalty, forces evasion; w=0.8
discourages gate attempts; w=0.4 marginal). Memory `reward-myopia-l3-findings`:
**L3 crashes are obstacle-collision dominated.** `lookahead_*` / `use_velocity_progress`
are dead config fields (not implemented in reward.py). So the obstacle term is the
single highest-EV untried reliability lever.

## Iteration log
| # | change | warm-from | recipe | steps | SR (0–99) | SR (100–199) | lap mean | verdict |
|---|---|---|---|---:|---|---|---|---|
| 1 | sum `+ r_obs`, obstacle_weight=0.6 | relB | a1.1 tp0.30 (=relBfast recipe) | +1B | … | … | … | running |

### Iteration 1 — relBobs: relBfast recipe + working obstacle barrier
Controlled A/B vs relBfast: identical warm-start (relB step_001002438656), identical
α1.1/tp0.30/omega0.005/gate_frame0.5, +1B — the *only* difference is `r_obs` now
enters the reward (obstacle_weight 0.6). If SR rises at ~equal lap → strictly better
balanced pick and confirms the obstacle-collision hypothesis.
