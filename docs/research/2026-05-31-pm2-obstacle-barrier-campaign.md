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
| 1 | sum `+ r_obs`, obstacle_weight=0.6 | relB | a1.1 tp0.30 (=relBfast recipe) | +1B | 69 % | 69 % | 4.83 | **DISCARD** (dominated by spd8) |
| 2 | obstacle_weight=0.3 | relB | a1.1 tp0.30 | +1B | **84 %** | **78 %** | 4.84 | **KEEP — new SOTA ⭐** (step_985) |
| 3 | obstacle_weight=0.3 on speed branch | spd7 | a1.4 tp0.40 | +1B | 73 % | 73 % | **4.62** | **KEEP — speed SOTA** (step_964) |
| 4 | + exit-vel bonus (coef 5) | relB | a1.1 tp0.30 w0.3 | +1B | 70–73 % | 69–77 % | 4.86 | **DISCARD** (SR↓, no lap gain; reverted) |
| 5 | frontier-fill a1.25 tp0.35 w0.3 | relB | a1.25 tp0.35 | +1B | 65–71 % | — | 4.85 | **DISCARD** (dominated by relBobs03) |

### Held-out (100–199) baseline failure structure (for per-seed rescue diff)
- relBfast SR 71 %, fails 29: 104,108,109,112,118,122,125,131,143,145,148,149,152,153,156,157,161,164,168,171,174,175,180,181,182,184,189,192,194
- relB SR 73 %, fails 27; spd8 SR 72 %, fails 28.
- **~16 seeds all three fail** (persistent hard): 104,108,109,112,143,145,148,153,156,157,164,175,180,181,184,192.
- Healthy-run telemetry confirmed `r_obs ≈ −0.238`/step active (was identically 0 pre-fix).

### Iteration 1 — relBobs: relBfast recipe + working obstacle barrier
Controlled A/B vs relBfast: identical warm-start (relB step_001002438656), identical
α1.1/tp0.30/omega0.005/gate_frame0.5, +1B — the *only* difference is `r_obs` now
enters the reward (obstacle_weight 0.6). If SR rises at ~equal lap → strictly better
balanced pick and confirms the obstacle-collision hypothesis.

**Result: DISCARD as a recipe (but the `r_obs`-sum code is RETAINED — it's correct
infra; only the weight is wrong).** All 3 late steps: b0 69–70 %, b100 69–73 %, lap
~4.78–4.83 s. Lands on spd8's profile (≈70 %/4.78) but is *weakly dominated* by spd8
(71–72 %/4.74). The obstacle term made the policy *faster* (rushes through danger
zones to cut integrated `r_obs`) and *less reliable* — opposite of intent.

**Per-seed diff (held-out, relBobs step_1002 vs relBfast):** rescued **11** genuine
failures {122,125,131,145,149,152,156,161,168,171,181}, newly broke **13** (gate-0×3:
147,179,186; rest gate-1/2). Net −2. ⇒ The barrier *does* fix real obstacle failures,
but w=0.6 over-bends trajectories past obstacles near the gate-1/2 line. Right lever,
too strong → iteration 2 halves the weight.

### Iteration 2 — relBobs03: same recipe, obstacle_weight=0.3 → **KEEP, NEW SOTA**
Identical to iteration 1 except `obstacle_weight 0.6 → 0.3`. `r_obs ≈ −0.122`/step (half).

| step | b0 SR | b0 lap | b100 SR | b100 lap | combined (n=200) |
|---|---:|---:|---:|---:|---:|
| 964 | 80 % | 4.87 | 74 % | 4.78 | 77 % |
| **985 ⭐** | **84 %** | 4.86 | **78 %** | 4.81 | **~81 %** |
| 1002 | 84 % | 4.90 | 74 % | 4.81 | 79 % |

**`relBobs03/step_000985661440` is the new balanced+reliability SOTA: ~81 % SR (n=200),
lap ~4.84 s.** Dominates the entire prior frontier — beats relB (77 %/5.2), relBfast
(74 %/4.94), spd8 (71 %/4.74-SR) on both/equal axes. Per-seed diff (step_985 vs relBfast,
held-out): **rescued 10** {118,125,131,145,148,152,161,171,189,194}, **newly broke 3**
{102,188,195} → net **+7**. Halving the weight kept the rescues (10 vs 11) and slashed
the breakage (3 vs 13). Backed up to gdrive. **Keep the `r_obs`-sum code + w=0.3 recipe.**

Takeaway: the obstacle barrier *is* a real reliability lever for converged policies —
the campaign just never had it active. Optimal weight ≈ 0.3 at progress_coef=15 (the
v34 calibration of 0.6 was for the slower progress_coef=10 era).

### Iteration 3 — spdobs03: speed branch + obstacle barrier → **KEEP, speed SOTA**
spd8 recipe (warm spd7, α1.4/tp0.40) + obstacle_weight=0.3. `step_000964689920`:
b0 73 %/4.633, b100 73 %/4.603 → **73 % SR / 4.62 s** (the earliest of the 3 late steps;
b0 trend 73→72→71 % shows mild over-train after). **Beats spd8 (71–72 %/4.74) on both
axes — fastest reliable ckpt of the campaign.** Per-seed (vs spd8, held-out): rescued 9,
broke 8 → net +1 (SR ~wash on the aggressive branch; the win is the −0.12 s lap). Backed
up `step_964` only. **New frontier:**

| pick | α / tp / w_obs | SR (n=200) | lap | role |
|---|---|---:|---:|---|
| **relBobs03** ⭐ | 1.1 / 0.30 / 0.3 | **~81 %** | 4.84 | balanced / reliability SOTA |
| **spdobs03** | 1.4 / 0.40 / 0.3 | ~73 % | **4.62** | speed SOTA |

Both strictly improve on the prior frontier (relBfast 74 %/4.94, spd8 71 %/4.74). The
inert-obstacle-barrier fix lifted *both* branches.

### Iteration 4 — relBobsXV: relBobs03 recipe + exit-velocity bonus (coef 5)
Isolates exit-vel vs relBobs03: warm relB, α1.1/tp0.30/w0.3 + `use_exit_vel_bonus=True
exit_vel_coef=5`. r_exit_vel was another computed-but-unsummed term (now summed, CLI
wired). Hypothesis: rewarding speed carried through gates cuts lap below 4.84 s while the
high-SR base + obstacle barrier hold reliability. Risk: overshoot → SR drop (discard if so).

**Result: DISCARD.** All steps b0 70–73 %, b100 69–77 % (combined ~73 %), lap ~4.86 s —
SR *down* from relBobs03's 81 % and **no lap improvement**. The one-shot gate-cross-speed
bonus induces overshoot/wrong-side (failures spread evenly across gates 0/1/2) and the
finishers burn time recovering. Reverted the r_exit_vel sum + CLI (commit dc1948d); kept
a NOTE in reward.py. Lesson: exit-vel is the wrong speed lever for these converged
policies — unlike r_obs, activating it does not help.

### Iteration 5 — relBmid: frontier-fill at α1.25 (between relBobs03 and spdobs03)
Warm relB, α1.25/tp0.35/w0.3 (proven active levers only, no exit-vel). Aims to fill the
4.62↔4.84 s gap with a balanced-fast pick (~77 %/4.72 expected) for competition selection.

**Result: DISCARD.** b0 65–71 %, lap ~4.85 s — lower SR than relBobs03 (84 %) and **not
faster** (4.85 vs 4.84). The α/SR/lap surface is non-monotonic: the good points are
α1.1 (relBobs03, balanced) and α1.4 (spdobs03, speed); α1.25 lands in an inferior middle
(tp0.35 added time pressure the policy converted to crashes, not speed, at this length).
No useful interpolation — keep the two endpoints.

## Campaign conclusion
**5 iterations, 2 kept, 3 discarded.** The headline is the inert-obstacle-barrier fix:
- **Kept:** relBobs03 (α1.1/w0.3, ~81 %/4.84) and spdobs03 (α1.4/w0.3, ~73 %/4.62) —
  new SOTA on both branches, strictly improving the prior frontier on both axes.
- **Discarded:** w=0.6 (too strong), exit-vel coef 5 (overshoot, reverted), α1.25 (dominated).
- **Levers that work for these converged policies:** obstacle barrier at w≈0.3, and α as
  the speed↔SR dial. **Levers that don't:** exit-velocity bonus, lookahead, guiding-path
  (all either reverted, dead config, or previously net-neutral).
- **Best single competition pick:** relBobs03 if SR-gated; spdobs03 if lap-weighted.
- **Remaining gap:** mean lap 4.62 vs leader 3.8 s — needs a time-optimal/contour reward
  (not exit-vel) or a fundamentally faster policy class; out of scope for this session.
