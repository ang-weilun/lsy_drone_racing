# L2 cold-start screen: angular-velocity channel + 512 saturation diagnostic

- **Date:** 2026-06-02
- **Status:** Approved, ready for implementation plan
- **Branch:** `rl/obs-completion-capacity-2026-06-02`
- **Owner:** ang-weilun

> **Revision history.** v1 proposed a 4-cell obs-completion × capacity factorial on
> L3. Investigation found the **2026-05-27 redesign** already made those choices
> deliberately, matching the *fixed-track* papers (Song/Kaufmann); scope slimmed to
> the two factors that survive scrutiny for L3 (ω, clean 512 retest). v2 then found
> the SOTA is **warm-started into curriculum stage-5**, so changed-shape cells can't
> inherit its weights and a cold L3 run means re-running the whole curriculum. v3
> (this doc) resolves that: **cold-start the cells on L2** — no warm-start, no
> checkpoint surgery, tractable from scratch, uniform across cells.

## 1. Motivation

The current 52-d obs / 256-wide single-head policy is the output of the 2026-05-27
redesign, which matched **Song 2023** and **Kaufmann 2023** — both *fixed/known
track* papers — and deliberately dropped ω and `prev_action`, used full-9D
rotation, kept 2×256, and reverted a 512 + split-head experiment.

The paper that prompted this — *Environment as Policy: Learning to Race in Unseen
Tracks* — is the only one of the three about the **unseen-track regime (= L3)** and
chooses the opposite on every axis (512×2, +ω, +`a_prev`). Two of the redesign's
justifications do not hold:

1. **ω.** The redesign dropped it arguing "the network can infer ω from
   rotation-matrix history." There is **no history** — the policy is a verified
   single-frame memoryless MLP. Song omits ω only because **Song's action *is* body
   rates** (≈ω, directly commanded); **our action is attitude**, so ω is downstream
   and genuinely unobserved. Re-adding ω rests on a concrete technical correction.
2. **512.** The prior 512 result (v131) carries **zero weight**: the obstacle
   barrier `r_obs` was inert until 2026-05-31, the value function was unclipped (the
   clipped-VF patch — "the likely actual fix" — postdates v131), the split-head was
   active, and **nothing was finishing at all** in that era (256 included), so the
   observed `|τ|/α_max` saturation can't be pinned on width.

`prev_action` is deferred: its justification only bites when `r_smooth` is on, and
`r_smooth` is **off** in the SOTA recipe.

## 2. Why L2, and what each cell actually measures

The SOTA warm-starts into stage-5 (full-L3-DR). Changing the obs dim (52→55) or the
width (256→512) changes tensor shapes, so the cells **cannot** load the SOTA
weights. Cold-starting at stage-5 is the exact regime the curriculum exists to
avoid. **Cold-starting on L2** sidesteps both problems: each cell is born at its new
shape, L2 is learnable from scratch in a modest budget, and the protocol is uniform
across cells (no surgery, no warm-start confound). `train.py` reaches this directly:
`curriculum="default"` is a single L2 stage (`config.py:default_curriculum`,
`stage1_level2_phase12`) and omitting `init_from` is a cold start.

There is a deliberate asymmetry in what L2 tells us per factor:

| Cell | Width | Obs | What an L2 cold-start measures |
|---|---|---|---|
| `ref` | 256 | 52 | cold-L2 baseline |
| `omegaA` | 256 | 55 (+ω) | **Benefit test (valid).** ω is proprioceptive control quality; L2 still has pose wobble + disturbances, so a real ω gain should appear. |
| `capB` | 512 | 52 | **Saturation diagnostic (not a benefit test).** The 512 *benefit* is hypothesized for the L3 track *distribution*; on a fixed L2 layout the policy can memorize, so an L2 512-null does **not** disprove the L3 hypothesis. But if 512 trains cleanly here (no `|τ|/α_max` blowup, flies L2) it cheaply **rebuts the v131 "512 breaks SBX" worry** and clears 512 for an L3 test; if it saturates even on the healthy stack, that's a real red flag found cheaply. |

## 3. Design

### 3.1 Angular-velocity channel (factor A)

Drone block 12 → 15 by appending body-frame angular velocity:

| drone sub-block | current | with ω |
|---|---|---|
| rotation matrix (world←body, flat) | 9 | 9 |
| linear velocity (body frame) | 3 | 3 |
| **angular velocity ω (body rates)** | 0 | **3** |

- Source `env_obs["ang_vel"]` — body-frame body rates the env exposes and `r_omega`
  already consumes (`reward.py:400`). Already body-frame → **appended raw**, no
  rotation (unlike linear velocity). Confirm body-frame against `rollout.py`
  (`[10:13] ang_vel (body frame)`).
- Actor obs 52 → 55; critic half mirrors (flat-concat 110).
- **Toggle, default off.** A module-constant `ACTOR_OBS_ANG_VEL_DIM` in `config.py`
  driven by env var `RL_OBS_ANG_VEL` (default `0` → 52-d reference preserved),
  read at import (before tyro parses args) — matching the existing env-var ablation
  pattern. It feeds `ACTOR_OBS_DIM`, so all symbolic consumers update automatically.

### 3.2 Capacity (factor B)

Actor + critic width 256 → 512 (depth 2 unchanged), via a module-constant
`HIDDEN_SIZE` in `rl_sbx/policy.py` driven by env var `RL_HIDDEN_SIZE` (default
256). `NET_ARCH = (HIDDEN_SIZE,) * 2` picks it up; `train.py` passes no `net_arch`
(it's forbidden by the policy), so the env var fully controls width. Single coupled
head and active clipped-VF stay. Log `|τ|/α_max` as the diagnostic.

### 3.3 Held constant

9D rotation; `prev_action` off; single coupled head; clipped-VF; `curriculum=default`
(single L2 stage); cold start (`init_from` omitted); and **one fixed reward recipe
across all cells** so the only varying factor per cell is the toggle under test. The
recipe is the **minimal Song reward** — `r_prog + r_omega + r_crash + r_finish` only
(NO obstacle / gate-frame barriers, NO time penalty) — with `alpha_max=0.36`. This is
the gentle cold-start substrate, **not** the L3 SOTA speed recipe (`alpha=1.4` +
barriers + `time_penalty=0.40`), which over-drives a from-scratch policy. (The first
run, 2026-06-02, mistakenly used the SOTA recipe and produced 0% true-start across all
cells — a recipe artifact, not a factor verdict; corrected here.)

## 4. Evaluation protocol

- **Seed-matched on L2** (`config/level2.toml`): every cell on the same seed set.
  Verify the seed-matched eval harness accepts an L2 target (it may be L3-specific —
  adapt or add a thin L2 variant rather than assume).
- Report per-cell **L2 success rate** and **lap time** + **union-of-seeds SR**.
- For `capB`, additionally log `|τ|/α_max` (the saturation diagnostic).
- **Pre-flight before full runs:** a single-rollout obs sanity check (obs finite,
  `|normalized| ≤ 10`, dims = 55/52 as configured) + the encoder parity check
  (§6.1) — catch toggle/mirror mistakes before burning compute.

## 5. Promotion (follow-up, out of this spec's scope)

If `omegaA` beats `ref` on L2 beyond seed-matched noise, promote it up the
curriculum toward L3 (continue the same run through the stages, or surgery, decided
then). If `capB` is clean on L2, schedule the real L3 capacity test (the regime
where its benefit, if any, can appear). Both promotions are separate work.

## 6. Correctness checklist (carry into the plan)

1. **Two obs encoders in lockstep** — canonical JAX (`rl_song/obs.py`) and the
   deploy numpy mirror (`rl_sbx/deploy_numpy/obs.py`). `rollout.py` and `env_gym.py`
   *call* the canonical encoder, so they need no obs edit (only that they read
   `ACTOR_OBS_DIM` symbolically, which they do).
2. **Encoder parity** — a checkpoint-free check comparing JAX vs numpy actor-obs on
   a fixed fake obs, at both flag settings, with matched (mean 0, var 1) normalizers.
3. **ω body-frame, appended raw** (no rotation); confirmed against `rollout.py`.
4. **No hard-coded `52`/`104`** — slicing (`policy.py`, `callbacks.py`) and
   `observation_space` (`env_gym.py`) derive from `ACTOR_OBS_DIM`.
5. **Normalizer** re-initializes at the new dim (fresh per cold run).
6. **Deploy coupling:** a checkpoint trained with `RL_OBS_ANG_VEL=1` must be deployed
   with the same env var (obs dim must match). Note at the controller load site.

## 7. Out of scope / deferred

- **`prev_action`** — sim2real / smoothness arm.
- **Curriculum redesign** — *Environment as Policy*'s actual contribution; a
  promotion framework already exists in `rl_song/train.py`. Strongest candidate if
  ω/512 stall. (Deferred by user.)
- **Speed reward** (velocity-toward-gate) — deferred.
- **6D-vs-9D**, track-scheme — settled by the redesign; not reopened.
- **L3 promotion** of any winner — §5, separate work.

## 8. Risks

- **ω net-neutral on L2.** Possible; the seed-matched A/B gates it either way.
- **512 saturates on the clean stack.** If `|τ|/α_max` blows up on healthy-stack L2,
  that's a genuine (non-confounded) negative — width interacts badly with SBX
  geometry — and we learned it cheaply.
- **L2→L3 non-transfer.** A factor helping L2 may not help L3 (esp. 512, by design).
  Mitigation: L2 is explicitly a *screen* (ω) / *diagnostic* (512), with L3 the
  separate real test (§5).
- **Toggle / mirror divergence.** Mitigation: checklist 1–4 + pre-flight parity.

## 9. References

- *Environment as Policy: Learning to Race in Unseen Tracks* — 512×2, obs includes
  ω and `a_prev`, unseen-track curriculum.
- Song et al., *Reaching the limit in autonomous racing*, Sci. Robotics 2023 —
  2×256; no ω / no `prev_action`; **body-rate action** (why it can omit ω).
- Kaufmann et al., *Champion-level drone racing*, Nature 2023 — 2×256; obs includes
  `prev_action`; reward includes the smoothness term (consistent pairing).
- Zhou et al., *On the Continuity of Rotation Representations*, CVPR 2019.
- Internal: `docs/superpowers/reviews/2026-05-27-redesign-{brief,implementation-brief}.md`;
  `obstacle-barrier-inert` finding.
