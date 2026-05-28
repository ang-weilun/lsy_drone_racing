# Round-7 (2.2B cumulative) checkpoint — current best L3 baseline

**Checkpoint:** `step_000402653184` from run
`sbx_redesign_warm1800_L3stage4dr_baseline_400M`. Baseline +400M continuation
from round-6 final; n_epochs reverted to 3, entropy back to 0.005→0.001
(default since the n_epochs=6 and ent=0.01 experiments were both null —
see `[[feedback-hp-tuning-null]]`).

**Status: current best L3 baseline** as of 2026-05-27 ~23:45 UTC.
Supersedes `snapshots/round6_ckpt_1.8B/` for L3.

## Headline results

| Metric | Value |
|---|---:|
| **L3 stochastic finish_rate_true_start** (wandb training, n=15167 episodes) | **0.504** |
| **L3 deterministic finish (strict-filter, n=50 seeds)** | **0.620 (31/50)** |
| L3 deterministic median lap | 6.88 s (range 5.46–8.50) |
| Strict-filter rejection rate | 5.7 % (3/53 seeds, only truly unflyable) |

The deterministic rate (62 %) exceeds the stochastic (50.4 %) partly
because the eval uses the filtered seed list (removes ~5.7 % unflyable
configurations) and partly because the deterministic policy mean is more
reliable than sampled actions in tight scenarios.

## Chain lineage (8 links)

| # | Run | Cum steps | Recipe note |
|---|---|---:|---|
| 1 | `sbx_redesign_50M` | 50M | Cold-train default_curriculum |
| 2 | `sbx_redesign_warm50_200M` | 200M | Warm +150M L2 |
| 3 | `sbx_redesign_warm200_L3stage4dr_400M` | 600M | Warm +400M L3 stage4_dr |
| 4 | `sbx_redesign_warm400_L3stage4dr_200M` | 800M | Warm +200M L3 |
| 5 | `sbx_redesign_warm1000_L3stage4dr_400M` | 1.4B | Warm +400M L3 |
| 6 | `sbx_redesign_warm1400_L3stage4dr_n6_200M` | 1.6B | HP exp: n_epochs=6 → null |
| 7 | `sbx_redesign_warm1600_L3stage4dr_ent01_200M` | 1.8B | HP exp: entropy 0.01→0.005 → null |
| **8** | **`sbx_redesign_warm1800_L3stage4dr_baseline_400M`** | **2.2B** | **Baseline +400M from round-6** |

## Slope history (L3 stochastic across the chain)

| Cum | finish_rate | Δ vs prev | Per-200M |
|---|---:|---:|---:|
| 600M | 0.042 | — | — |
| 800M | 0.130 | +8.8 pp | +8.8 pp/200M |
| 1.4B | 0.342 | +21.2 pp / 600M | +7.1 pp/200M |
| 1.6B (n_epochs=6) | 0.392 | +5.0 pp | +5.0 pp/200M |
| 1.8B (ent=0.01) | 0.426 | +3.4 pp | +3.4 pp/200M |
| **2.2B (baseline)** | **0.504** | **+7.8 pp / 400M** | **+3.9 pp/200M** |

Slope continues a slow decay: late-chain baseline rounds delivering
~+4 pp/200M (vs ~+5 in early rounds). Compatible with the policy
approaching the natural ceiling of the Song-pure 5-term reward on L3 DR
randomization — further gains likely require a different lever
(reward shaping, α_max, DR curriculum).

## HPs used (identical to rounds 1-4)

| HP | Value |
|---|---:|
| `learning_rate` | 3e-4 |
| `n_epochs` | 3 |
| `n_steps` | 256 |
| `n_envs` | 16384 |
| `batch_size` | 16384 |
| `gamma` | 0.998 |
| `gae_lambda` | 0.97 |
| `clip_range` | 0.2 |
| `clip_range_vf` | 0.2 (patched VF) |
| `target_kl` | None (load-bearing) |
| `ent_coef` | 0.005 → 0.001 |
| `progress_coef` | 15.0 |
| `omega_coef` | 0.01 |
| `r_smooth_coef` | 0.0 (inactive) |
| `crash_penalty` | 15.0 |
| `finish_bonus` | 100.0 |
| `tangent_alpha_max_rad` | **0.32** (the v53-SOTA value) |
| `curriculum` | full, stage 5 (stage4_level3_dr) |

## Observation / reward / action

Identical to round-6 ckpt — see `snapshots/round6_ckpt_1.8B/SPEC.md` for
the full layout. Summary:

- **Obs 52-d**: drone 12 (full R_wb + vel_body) + gates 24 (target
  corners body + inter-gate delta body) + obstacles 16 (2 slots ×
  [xy_body, vel_proj, identity_onehot, visited]). Source:
  `rl_song/obs.py`. See [[reference-actor-obs-layout]].
- **Reward**: Song-pure 5-term (`r_prog · 15.0 + r_omega · 0.01 +
  r_smooth · 0.0 + r_crash · -15 + r_finish · +100`). All other terms
  gated `use_*=False` and zero-coef.
- **Action**: raw `[T_raw, τ_x, τ_y, τ_z]` → `raw_to_env_action` →
  env `[roll, pitch, yaw, thrust]`. `α_max = 0.32` rad/step. Single
  coupled `Dense(4)` head + LeakyReLU trunk.

## Files in this snapshot

| File | Bytes | Purpose |
|---|---:|---|
| `actor.params.msgpack` | 321789 | Flax actor PyTree |
| `critic.params.msgpack` | 318665 | Flax critic PyTree |
| `actor_normalizer.json` | 2203 | Welford running stats (actor) |
| `critic_normalizer.json` | 2203 | Welford running stats (critic) |
| `policy_config.json` | 35 | `{"tangent_alpha_max_rad": 0.32}` |
| `SPEC.md` | this file | All-of-the-above documentation |

## Strict-filter eval methodology (new)

The 62 % deterministic finish rate used the strict L3 seed filter (see
`snapshots/clean_l3_seeds.json` and `[[reference-l3-seed-filter]]`):

- Geometric criterion: for each (gate, obstacle) pair, project obstacle
  into gate's local frame; reject seed if **any** pair has `|local[0]|
  < 0.05 m` (obstacle near gate plane) AND `|local[1]| < 0.05 m`
  (obstacle near gate centerline laterally).
- Rationale: sphere drone radius 0.086 m + obstacle radius 0.015 m gives
  geometric minimum drone-center wiggle of 1.3 cm at gate edge. The
  0.05 m thresholds reject only seeds where the drone has < ~4 cm of
  practical clearance — truly unflyable.
- Effective rejection: 3/53 seeds = 5.7 %. The remaining 50 are flyable
  but include tight cases, which the policy mostly handles (62 %).

## Next-step candidates

- **Speed tune** (priority 1): α_max bump 0.32 → 0.48 per
  [[project-v53-sota]] pattern. Lap times 6.88 s → likely 4.5-5 s.
  Risk: finish-rate regression; mitigated by warm-start from this ckpt.
- **Further baseline grind**: +400M more would project ~55-58 %
  stochastic and probably ~65-68 % filtered det. Diminishing returns;
  not the highest EV.
- **Reward shaping reintroduction**: bring back r_gate_frame or
  r_caution to push beyond the Song-pure ceiling. Higher upside but
  riskier.
