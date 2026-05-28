# warm2790 tp20 final — L2 speed-tune (4.42 s @ 95% L2)

**Checkpoint:** `step_000201326592` from run
`sbx_redesign_warm2790_L2default_tp20_200M`. Final step of the +200M
time-penalty=0.20 continuation. 12th link in the redesign chain (round 7
→ warm2200 → warm2390 → warm2590 → warm2790).

**Status: L2 speed-tune frontier ckpt** as of 2026-05-28 ~09:29 UTC.

## Headline results (n=20 deterministic, numpy controller)

| Level | Finish rate | Lap time (s, median) | Lap min / max |
|---|---:|---:|---:|
| L0 | 100% (20/20) | **4.32** | 4.30 / 4.34 |
| L1 | 100% (20/20) | **4.33** | 4.26 / 4.40 |
| **L2** | **95% (19/20)** | **4.42** | 4.24 / 4.58 |
| L3 | 0% (0/20) | — | — |

L2 lap **min = 4.24 s ≡ v83 SOTA's median** (`[[project-v83s40M-crosslevel-sota]]`).
Our median 4.42 s is +5 % over v83's median, but at **2.8× v83's SR** (v83 was
34 % @ n=100). Pareto win vs v83 unless the leaderboard scores pure speed.

## Chain progression on L2

| Run | Δ from prev | L2 SR | L2 median |
|---|---|---:|---:|
| round-7 (L3 baseline, 2.2B) | — | 35% (round-4 measure) | — |
| warm2200 (α 0.32 → 0.48, L2 curriculum, tp=0) | speed lever pt 1 | 90% (final) / 100% (step_188M) | 5.71 / 5.76 s |
| **warm2390 (tp 0 → 0.10)** | +0.10 tp | 100% | 5.26 s (-0.45) |
| **warm2590 (tp 0.10 → 0.15)** | +0.05 tp | 100% | 4.73 s (-0.53) |
| **warm2790 (tp 0.15 → 0.20)** | +0.05 tp | **95%** | **4.42 s (-0.31)** |

Marginal slope: tp10→tp15 = -10.6 s per tp unit; **tp15→tp20 = -6.2 s per tp unit**.
Diminishing returns now visible (-41 % marginal slope). L2 SR also regressed
for the first time (100 → 95 %), so the lever is approaching its operating
limit at this α_max + curriculum.

## HPs (deltas vs warm2200 = tp=0)

| HP | warm2200 | warm2790 |
|---|---:|---:|
| `time_penalty` | 0 | **0.20** |
| `tangent_alpha_max_rad` | 0.48 | 0.48 (unchanged) |
| `curriculum` | default (L2 stage1) | default (unchanged) |
| Everything else (lr, n_epochs=3, gamma=0.998, ent 0.005→0.001, progress=15, omega=0.01) | — | unchanged |

200M timesteps, 10 ckpts saved every 20M, ~33 min wall on RTX 5090.

## Observation / reward / action

Identical to round-7 / warm2200 — see `snapshots/round7_ckpt_2.2B/SPEC.md`.
Summary:
- **Obs 52-d**: drone 12 + gates 24 + obstacles 16. See
  `[[reference-actor-obs-layout]]`.
- **Reward**: Song-pure 5-term + `r_time = -0.20·dt`. The only difference
  from round-7 reward is the time penalty.
- **Action**: raw `[T_raw, τ]` → `raw_to_env_action`. `α_max=0.48`.

## Files in this snapshot

| File | Bytes | Purpose |
|---|---:|---|
| `actor.params.msgpack` | 322,488 | Flax actor PyTree |
| `critic.params.msgpack` | 319,365 | Flax critic PyTree |
| `actor_normalizer.json` | 2,203 | Welford running stats (actor) |
| `critic_normalizer.json` | 2,203 | Welford running stats (critic) |
| `policy_config.json` | 35 | `{"tangent_alpha_max_rad": 0.48}` |
| `SPEC.md` | this file | Full provenance |

## Predicted next levers (not yet decided)

1. **tp=0.25** — continued single-lever push. Expected lap ~4.2 s, SR may
   drop to ~90 %. Cleanest extension of the existing chain.
2. **γ=0.998 → 0.995** — codex's "if tp stalls" trigger now applies.
   Trade-off: +22 % finish-bonus gradient but **-26 % tp PV** (would need to
   bump tp to compensate). Critic shock real but recoverable at 200M.
3. **Combine tp=0.25 + γ=0.995** — strongest signal; hardest to interpret.
4. **Add `--use-exit-vel-bonus`** — code edit (3-line CLI plumb), v83 pattern.
5. **Stop here.** 4.42 s @ 95 % L2 is a strong leaderboard entry; further
   chasing may not be worth the risk.

## Predict-lap-time-from-r_prog rule

For this policy family: **`lap_time ≈ 1.9 / r_prog_per_step`** when
`finish_rate_true_start > 0.98`. Verified across warm2200 / tp10 / tp15 /
tp20 within 1-2 % accuracy. Live wandb reading trick for future runs.

## Provenance

- Branch: `rl/reward-fix-2026-05-25` (uncommitted on dev VM).
- Vast: 192.3.91.246:25482, contract 37779019.
- Training log: `/root/lsy_drone_racing/training_logs/sbx_redesign_warm2790_L2default_tp20_200M.log`
- Eval log: `/root/eval_warm2790_tp20.log`
- wandb: `weilun-ang-technical-university-munich/lsy-drone-racing-rl-song/sbx_redesign_warm2790_L2default_tp20_200M`
