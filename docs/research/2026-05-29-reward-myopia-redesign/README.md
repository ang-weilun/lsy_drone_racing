# Reward redesign for racing-line behaviour on randomised tracks

*Research record and decision log. Started 2026-05-29. Branch `rl/reward-fix-2026-05-25`.*

This directory documents an investigation into a myopic "point-to-point" local
optimum in our RL racing policy: the problem, the literature we reviewed, the
reference implementations we inspected at source level, the candidate fixes, and
the decision trail. It is written to be reusable directly in the end-of-semester
report (problem → literature → method → decision).

## Contents

| File | What it is |
|---|---|
| `README.md` | This synthesis and decision log. |
| `literature-report-1-initial.md` | First deep-research pass (verbatim). Found to be stale w.r.t. our code; kept for the record. |
| `literature-report-2-ranked-redesign.md` | Second pass (verbatim): ranked reward/observation redesign, grounded in our actual implementation. |
| `literature-report-3-chronological.md` | Third pass (verbatim): chronological evolution of the UZH-RPG reward family; explains the literature split. |
| `research-prompt-grounded.md` | The grounded research prompt we authored after the first pass came back stale (documents the method). |
| `liu-2024-guidance-reward-source-extract.md` | Verbatim reward source extracted from Liu 2024's open-source code, with our annotations. |
| `codex-review.md` | Independent adversarial review of the diagnosis and proposed fixes (added after the review completes). |

## 1. The problem we hit

Our L3 policy (fully randomised gate + obstacle layouts) settled into a slow,
**point-to-point** local optimum. Two failure geometries, both present in the
latest L3 checkpoint:

1. **Reverse-out.** When the gate after the current target sits roughly 180°
   behind it, the drone flies slowly *through* the current gate and then
   **reverses back out** toward the next target, instead of carrying momentum
   through and banking around.
2. **Frame collision.** When the next gate is off to the **side**, the drone
   passes through the current gate and then **straight-lines to the next gate's
   centre, clipping the frame of the gate it just passed.**

The behaviour is a stable local optimum, not a transient; it also caps lap time
(the policy never commits to a racing line).

## 2. Root-cause diagnosis

The progress reward is **direction-blind center-distance progress**:

```
r_prog = k · (‖g_center − p_{t−1}‖ − ‖g_center − p_t‖)
```

measured to the **centre of the current target gate**. Gate switching is a
directional **plane-crossing within the aperture**. The moment the drone crosses
gate N's plane, the target advances to N+1 and `r_prog` begins rewarding any
reduction in straight-line distance to **N+1's centre**. If N+1 is behind or
beside N's exit, that gradient points **backward / laterally — straight back
through N's frame**. The plane-crossing switch prevents *re-counting* gate N but
does nothing to remove the backward pull.

**Conclusion: the defect is in the progress-reward geometry**, and the fix
belongs in the reward, not the observation or the switch.

## 3. What we ruled out

- **Observation.** Already multi-gate (current + next gate as four corners each)
  in the drone body frame, with an asymmetric critic. The policy can *see* the
  upcoming geometry; it simply is not *rewarded* for using it anticipatorily.
- **Gate switching.** Already a directional plane-crossing within the aperture
  (`gate_passed` in `lsy_drone_racing/envs/utils.py`), not centre-proximity.
- **Discount / horizon.** γ = 0.998 at 50 Hz gives an effective horizon of
  ≈ 1/(1−γ) = 500 steps ≈ 10 s, far longer than any inter-gate transit. Myopia
  is not coming from the discount.
- **Curriculum exposure.** Segment-init (mid-track respawn with gate-aligned
  velocity), Phase-2 success-state replay, and L2/L3 domain randomisation are all
  in place; the policy is exposed to the hard geometries and still converges to
  the point-to-point optimum.

## 4. Literature reviewed

We reviewed the UZH-RPG drone-racing reward lineage and the recent
randomised/cluttered-track work. The organising insight from the review:

> **The apparent disagreement in the literature is explained by track geometry,
> not by principle.** Papers that fly a *single fixed track* (Swift, Song 2023,
> ACMPC) use bare **centre-distance progress** plus a perception term, because the
> next gate is always essentially "forward". Every paper targeting *randomised or
> cluttered* tracks adds a **path- or gate-frame guidance term** on top of
> progress, specifically to stop side-bypass and wrong-side passage. Our regime
> is the randomised one.

Annotated bibliography (grouped by the insight above):

**Centre-distance-progress family (fixed track):**
- **Kaufmann et al. 2023, "Champion-level drone racing using deep RL", *Nature*.**
  Reward = progress-to-next-gate-centre + perception (keep gate in camera FoV) +
  command + terminal crash. Fixed seven-gate track. → [`wiki/papers/kaufmann-2023-champion-level-rl.md`]
- **Song et al. 2023, "Reaching the Limit in Autonomous Racing: OC vs RL",
  *Science Robotics*.** Gate-progress objective; thesis is that RL wins by
  optimising a *better objective* than trajectory tracking. Two fixed tracks.
  → [`wiki/papers/song-2023-reaching-the-limit.md`]
- **Romero et al. 2024/2025, "Actor-Critic Model Predictive Control" (ICRA/T-RO).**
  Bare centre-distance progress `c·(‖p_{k−1}−g‖ − ‖p_k−g‖) − b‖ω‖` — i.e. exactly
  our current form. Reports that a camera-pointing term creates a *competing*
  objective that trades against speed. → [`wiki/papers/romero-2024-actor-critic-mpc.md`]

**Path-/guidance-term family (randomised / cluttered track):**
- **Song et al. 2021, "Autonomous Drone Racing with Deep RL", IROS.** Two-part
  template: path-projection progress (projection onto the line between gate
  centres) **plus a gate-plane Gaussian "safety reward"** introduced to reduce
  crashes when training on randomised tracks. The ancestor of all later guidance
  terms.
- **Penícka & Scaramuzza 2022, "Learning Minimum-Time Flight in Cluttered
  Environments", RA-L.** Generalises progress to arc length along a **curved
  guiding path**, with an anti-singularity term `+k_s·s(p)` whose stated purpose
  is to counteract "negative-progress singularities … due to the sharp corners of
  the guiding path" — precisely our 180°/lateral geometry. Two-stage slow→fast
  speed curriculum. Coeffs (verbatim in paper): k_p=5.0, k_wp=5.0, k_ω=0.01,
  terminal −10. RL code not public.
- **Liu 2024 (TU Delft), "Learning Generalizable Policy for Obstacle-Aware
  Autonomous Drone Racing", arXiv:2411.04246.** Keeps Swift centre-distance
  progress and adds a **gate-frame guidance reward** that funnels the drone onto
  the gate axis on approach and **rejects wrong-side (behind-gate) approach**.
  Open source (`github.com/ErcBunny/IsaacGymEnvs`) — we extracted it at source
  level; see §5. *(This is the most directly applicable recent recipe for our
  setting.)*
- **Sun et al. 2026 (SJTU), "Curriculum RL for Quadrotor Racing with Random
  Obstacles", arXiv:2602.24030.** Diagnoses that "the gates are also perceived as
  obstacles", making gate-passing and obstacle-avoidance conflicting objectives;
  fixes via multi-stage curriculum + reward balancing. Open source.
- **Pasumarti, Bianchi & Loquercio 2025, "Agile Flight Emerges from Multi-Agent
  Competitive Racing", arXiv:2512.11781.** *Scale-matched (Crazyflie 2.1).*
  Measured dense single-agent centre-progress at 100%/98% success **without**
  obstacles but **0% with** obstacles ("the progress reward discourages the drone
  from moving away from the gate"). Their fix is **sparse outcome-based** rewards,
  argued to transfer more reliably to hardware. The strongest scale-matched
  warning against relying on bare centre-progress.

**Curriculum-as-fix (complementary axis):**
- **Wang/Xing et al. 2024 (RPG), "Environment as Policy", arXiv:2410.22308.**
  A secondary SAC "environment policy" adaptively shapes track layouts to be
  "difficult but achievable"; naive uniform randomisation overburdens the agent.
  → [`wiki/papers/wang-2024-environment-as-policy.md`]
- **Yu et al. 2025, "Mastering Diverse, Unknown, and Cluttered Tracks",
  arXiv:2512.09571.** Two-phase soft→hard collision curriculum to preserve
  high-speed exploration.

**Conceptual ancestor (trajectory optimisation, not RL):**
- **Foehn et al. 2021, "Time-Optimal Planning … Complementary Progress
  Constraints" (CPC), *Science Robotics*.** Binds progress to each waypoint so
  progress advances only in local proximity to that waypoint — the offline
  analogue of "don't credit progress to gate N+1 until N is cleared".

## 5. Source-verified reference implementation (Liu 2024)

A material correction the prose summaries got wrong: **Liu 2024 does not use
path-projection progress.** Reading the actual source
(`/tmp/liu/reward.py`, archived in `liu-2024-guidance-reward-source-extract.md`):

- Progress is **Swift centre-distance**, multiplied by `(~wp_passing)` to **zero
  it on the gate-passing step** (avoids the transition discontinuity — the same
  bug our v85 "r_prog leak" fix addressed).
- The geometric fix is a **separate gate-frame guidance potential**
  `r_guid = k_guidance · (−f²(x)) · g(y,z)`, with `f(x)=clamp(1−|x|/x_thresh,0)`
  a window on the gate-normal axis, an **aperture-aware** in-plane Gaussian that
  funnels the drone onto the gate axis on the approach side, and a **rejection**
  term on the wrong (behind-gate) side. Deployed weight `k_guidance = 1.0`,
  co-equal with progress.

So the two reference approaches are genuinely **different attacks**: Penícka
*replaces* the progress term; Liu *keeps* centre-progress and *adds* a gate-frame
potential. Liu's guidance maps closely onto shaping levers already present (but
disabled) in our `reward.py`: `dipole` (signed front/behind potential),
`wrong_side` penalty, and `vel_shaping` — Liu's is an aperture-aware refinement
of the same idea.

**Scale caveat:** Liu's drone is 0.76 kg with 1.2–3 m gates at 25 Hz; ours is a
~30 g Crazyflie with ~0.45 m gates at 50 Hz. The in-plane geometry auto-scales
via the aperture normalisation, but `guidance_x_thresh` (3 m) and the co-equal
`k_guidance` must be re-tuned down for our scale.

## 6. Candidate fixes

- **Route A — replace progress (Penícka).** Precompute a per-env guiding polyline
  at reset from the true gate poses — `[…, exit_N, entry_{N+1}, center_{N+1}, …]`
  with exit/entry waypoints offset along the gate normals — then reward arc-length
  progress `k_p·(s_t−s_{t−1}) + k_s·s_t` (the `k_s` term removes corner
  singularities). The only route that *structurally* removes the backward
  gradient on a true 180°. Cost: a real `reward.py` rewrite; changes the
  load-bearing term; warm-started critic value-scale will be off initially.
- **Route B — augment (Liu).** Keep centre-distance progress (already zeroed on
  the pass step), add a Liu-style gate-frame guidance funnel + wrong-side
  rejection (upgrade our existing `dipole` to the aperture-aware `−f²·g`), and
  optionally enable `vel_shaping`. Smaller change; composes with warm-start
  chains; strongest on the lateral-clip and wrong-side cases. Progress stays
  centre-distance, so the 180° backward gradient is *counteracted* rather than
  removed.
- **Cheap pre-probe.** Before either, enable the already-plumbed
  `dipole` + `wrong_side` + `vel_shaping` levers (one warm-start run, no code) to
  get fast signal and isolate which symptom (lateral vs 180°) each lever fixes.

## 7. Decision log / status

- **2026-05-29** — Diagnosis and literature review complete. Liu reward extracted
  at source. Two candidate routes + a cheap probe defined. Submitted the
  diagnosis and both routes to an independent adversarial review (Codex);
  `codex-review.md`.
- **2026-05-29 (key)** — Codex's review surfaced, and we verified via git, that
  the trained scalar is `r_prog + r_omega + r_smooth + r_crash + r_finish`
  (reward.py:566), dropping `r_time` and every shaping term that the May-25 `HEAD`
  summed. **This is intentional, not a bug:** the team stripped the accreted patch
  terms back to a pure Song-style reward, which trained markedly better. The
  important consequences:
  - The shaping CLI flags (`dipole`, `wrong_side`, `guid`, `gate_frame`, `obs`,
    `exit_vel`, `vel`, `caution`) and `time_penalty` are **inert** in this tree —
    a term must be *added back to the sum* to take effect; setting its weight
    alone does nothing.
  - The 2026-05-28 L2 "time_penalty chain" therefore trained with `r_time`
    disabled. **wandb confirms the misattribution:** across warm2200→2790,
    `reward/r_time` tracked the tp label exactly (0, −0.10, −0.15, −0.20) but the
    lap-time gains are explained by rising trained `reward/r_prog`
    (0.327→0.369→0.403→0.426); the team's own `lap ≈ 1.9/r_prog` rule predicts the
    measured laps (5.71/5.26/4.73/4.42 s) to ~1–2%. The speed came from continued
    warm-start training optimising the γ-discounted progress + finish — the tp
    bumps were confounded with +200 M more training each. Real speed levers going
    forward: progress weight, γ, α_max, training duration.
  - The policy is effectively trained on **pure centre-distance progress + γ +
    crash/finish** — exactly the myopia-producing configuration the literature
    describes, which sharpens the diagnosis.
  **Revised plan:** keep the pure-Song reward as the baseline; test the geometric
  fix by adding **exactly one** term to the sum (single-lever discipline). Codex
  recommends a hybrid — progress to an **entry waypoint** (`center − d·normal`) +
  a **previous/current gate-frame barrier** (the lateral clip needs the just-passed
  frame priced) + local **wrong-side rejection** — over either pure route, with
  **actor-only warm-start** and a **critic + Phase-2-replay reset** (both encode
  the old reward).
- *(pending)* — Geometric-fix implementation (one term added to the pure-Song
  sum); verification on scripted reverse-out / side-clip trajectories before PPO;
  measured effect on reverse-out / frame-clip and on L3 finish-rate / lap-time.

## 8. Caveats on citations

The bibliography above is what our literature review surfaced. Exact equation
numbers and some arXiv identifiers for the most recent preprints (Sun 2026,
Pasumarti 2025, Yu 2025) and the precise closed forms / coefficients of Song 2021
and Liu 2024 were flagged **not verified verbatim** during the research passes;
Liu 2024's reward we have since confirmed from its open-source code. **Verify each
primary source before citing it in the final report.** Several of these are not
yet ingested into `wiki/papers/` (Penícka 2022, Liu 2024, Sun 2026, Pasumarti
2025, Yu 2025, Foehn/CPC 2021) — candidates for ingestion if they are cited.
