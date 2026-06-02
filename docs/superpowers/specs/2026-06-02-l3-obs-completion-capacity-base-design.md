# L3 base: observation completion + capacity factorial

- **Date:** 2026-06-02
- **Status:** Approved design, pending spec review
- **Branch:** `rl/obs-completion-capacity-2026-06-02`
- **Owner:** ang-weilun

## 1. Motivation

We have been stuck improving L3 metrics for two days. Reading *Environment as
Policy: Learning to Race in Unseen Tracks* (RPG/UZH) prompted a comparison of our
observation head and network against their racing stack. The mapping
(`rl_song/obs.py`, `rl_sbx/policy.py`, `rl_song/reward.py`, `config/level3.toml`)
produced two findings that are correctness/regime issues, not tuning knobs:

1. **We already use their observation head.** Corner-based gate encoding (target
   gate's 4 opening corners + inter-gate corner delta, 2-gate lookahead) is what
   `build_actor_obs` already does, and we feed a full continuous rotation matrix,
   so we never had the quaternion-discontinuity problem the 6D trick exists to
   solve. The obs head is *not* where we differ from the paper.

2. **Two standard channels are missing — and we penalize them while blind to
   them.** The paper's `o_racing` includes angular velocity `ω` and the previous
   action `a_prev`; our 52-d actor obs has neither (`ACTOR_OBS_PREV_ACTION_DIM=0`;
   `prev_action` is accepted then discarded at `obs.py:255`; no `ang_vel` channel).
   Yet `r_omega` penalizes `‖ω‖` and `r_smooth` penalizes `‖Δaction‖²`. The policy
   is being asked to minimize body rate and action jerk it cannot observe. Both
   quantities are trivially available on the real Crazyflie (gyro; own command),
   so their absence is an oversight inherited from the obs port, not a sim2real
   decision.

A third, regime-level observation: **L3 regenerates the entire track every reset**
(gate XY+yaw, obstacle XY, alternating 0.7/1.2 m heights), so we are literally in
the paper's "unseen tracks" regime — the setting in which they argue for a wider
(512×2 vs 256×2) network.

## 2. Goal and success criteria

Establish a corrected, capacity-matched **base policy** that will serve both the
(deferred) speed and reliability arms. This spec covers *only* the base.

- **Primary:** seed-matched L3 success rate (SR) and lap time of the corrected
  base vs the current SOTA reference (`relBobs03`, ~81% SR / 4.84 s).
- **Secondary:** union-of-seeds SR (our reliability headroom has historically
  shown up here), and whether `r_smooth`/`r_omega` stop costing SR once the
  policy can observe the quantities they penalize.
- **Decision rule:** the winning cell (by seed-matched SR, then lap time, with
  union SR as tie-break) becomes the new base. A cell that regresses SR beyond
  seed-matched noise is rejected.

## 3. Design

### 3.1 Observation completion (factor A)

Extend the drone block of the 52-d encoder from 12 → 19 dims:

| Block | Current | After |
|---|---|---|
| rotation matrix (world←body, flattened) | 9 | 9 (unchanged) |
| linear velocity (body frame) | 3 | 3 (unchanged) |
| **angular velocity `ω` (body rates)** | 0 | **3** |
| **previous action `a_prev`** | 0 | **4** |
| gates (corners + inter-gate delta) | 24 | 24 (unchanged) |
| obstacles (2 nearest, 8/slot) | 16 | 16 (unchanged) |
| **total actor obs** | **52** | **59** |

- `ω`: the body-frame angular velocity the env already exposes as
  `env_obs['ang_vel']` and that `r_omega` already consumes. Its frame must be
  confirmed to be body rates during implementation (frame alignment is a primary
  correctness concern, not an afterthought).
- `a_prev`: the previous *executed* raw policy action (the 4-vector in the policy's
  own action space — sampled during training, deterministic at eval/deploy), zero
  at episode reset. Feeding the raw commanded action (not the post-conversion
  physical action) is what makes the closed loop Markov with respect to the
  smoothness penalty.
- Both new channels flow through the existing per-half Welford normalizer
  unchanged (the normalizer shape is driven by `ACTOR_OBS_DIM`).

**Toggleable, not hard-wired.** Add config flags `obs_include_ang_vel` and
`obs_include_prev_action` (default `False`, preserving the reference encoding),
and compute `ACTOR_OBS_DIM` from them rather than asserting `== 52`. This matches
the codebase's existing feature-flag pattern (`use_obstacle_barrier`,
`use_gate_frame_barrier`, …), keeps the factorial a matter of flags, and is
reversible. The new runs set both flags `True`.

**Critic half mirrors the actor layout.** `build_critic_obs` (privileged) gains
the same two channels so the flat-concat stays `[actor(59) | critic(59)] = 118`.

### 3.2 Capacity (factor B)

Actor and critic hidden width 256 → 512, depth unchanged at 2
(`NET_ARCH=(512, 512)`), matching the paper. Width must be a CLI/env knob on
`train.py` (default 256) threaded into `policy_kwargs`, so both 256 and 512 cells
run without per-cell code edits.

### 3.3 Rotation representation: held at 9D

We keep the full 9-d rotation matrix and do **not** make it a factor. Rationale:
Zhou et al.'s 6D continuity result is about rotation *outputs/regression targets*;
as an *input* both 6D and 9D are smooth SO(3) embeddings with no discontinuity.
The redundant third column equals `c1 × c2` and, for a quadrotor, is exactly the
body-z (thrust) axis in world frame — so precomputing it is weakly favorable, not
harmful. The effect is second-order (an MLP learns one cross product trivially),
so changing it now would only add a confound. A 6D-vs-9D one-off is a cheap
follow-up if the base wins.

### 3.4 Reward: held constant

All cells use the current SOTA recipe (`scripts/box_launch_speed.sh`:
`progress_coef=15`, `time_penalty=0.40`, `omega_coef=0.005`,
`obstacle_weight=0.3`+barrier on, `gate_frame_weight=0.5`+barrier on, `alpha=1.4`)
so the factorial measures only factors A and B.

## 4. Experiment matrix

Four cells; three new training runs (the reference already exists). All from
scratch (input dim and width changes preclude warm-start), all parallelizable on
the vast.ai fleet.

| Cell | Width | Obs | Purpose |
|---|---|---|---|
| `ref` | 256 | 52 | current SOTA (`relBobs03`), re-evaluated seed-matched |
| `obsA` | 256 | 59 | isolates observation completion |
| `capB` | 512 | 52 | isolates capacity |
| `both` | 512 | 59 | combined candidate |

`ω` and `a_prev` are bundled into one factor; if `obsA`/`both` win, splitting them
is a cheap follow-up.

## 5. Evaluation protocol

- **Seed-matched** via `scripts/eval_l3_seed_matched.py` — every cell on the same
  seed set, no free-seed comparisons (we have twice mistaken noise for signal).
- Report per-policy **SR** and **lap time**, plus **union-of-seeds SR**.
- Back up only the eval-selected `step_*` checkpoint per run to gdrive.

## 6. Correctness checklist (carry into the plan)

These are the integration points that break silently if missed:

1. **Three obs mirrors updated in lockstep** — `rl_song/obs.py` (canonical),
   `rl_sbx/rollout.py` (JAX-traced rollout), `rl_sbx/deploy_numpy/obs.py` (deploy).
   A divergence here is a sim2real bug.
2. **No hard-coded `52`/`104`** — all actor/critic slicing
   (`rl_sbx/policy.py`, `callbacks.py` `ACTOR_SLICE`/`CRITIC_SLICE`) and the
   `observation_space` (`env_gym.py:157-159`) must derive from `ACTOR_OBS_DIM`.
3. **`ω` frame** confirmed body rates; **`a_prev`** is the raw executed action,
   zeroed at reset.
4. **Normalizer** re-initializes at the new dim (fresh per from-scratch run).

## 7. Out of scope / deferred

- **Speed arm** (non-telescoping speed reward) — deferred by decision
  2026-06-02. The Song concern is resolved in principle: our terminate-on-finish
  single-course episodes lack Song's implicit progress-over-horizon speed
  incentive, justifying an explicit term later; form (velocity-toward-gate
  projection vs raw `‖v‖²`) is undecided.
- **Curriculum / adaptive track distribution** — skipped for now; revisit only if
  reward/selection levers stall. The paper's *learned* environment-policy is a
  heavy, novel-ish build our scope rules discourage; a manual difficulty schedule
  is the lighter fallback.
- **6D-vs-9D rotation** ablation and **ω-vs-a_prev** split — cheap follow-ups
  contingent on the base winning.

## 8. Risks

- **Obs completion is net-neutral or negative.** RL changes are unpredictable;
  the observability argument is principled but not a guarantee. Mitigation: the
  factorial isolates it, seed-matched eval gates the decision.
- **512 width slows training enough to matter.** The throughput work on the
  parent branch offsets this; wall-clock is dominated by the PPO update, and
  width is a modest contributor. Monitor SPS on the first `capB` rollout.
- **Frame/convention mismatch on `ω` or `a_prev`** silently degrades the policy.
  Mitigation: checklist item 3 + a single-rollout sanity check that obs ranges
  are sane before committing compute to full runs.

## 9. References

- Y. Song et al., *Environment as Policy: Learning to Race in Unseen Tracks*
  (RPG/UZH) — obs head (`R̃, v, ω, a_prev, δp1, δp2`), 512×2 MLP, unseen-track
  curriculum.
- Zhou et al., *On the Continuity of Rotation Representations in Neural Networks*,
  CVPR 2019 — 6D rotation representation (output-side continuity).
- Kaufmann et al., *Champion-level drone racing using deep RL*, Nature 2023 —
  corner-based gate observations.
