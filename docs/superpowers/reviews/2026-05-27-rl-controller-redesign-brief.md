# RL Controller Redesign — Brief for Independent Review (2026-05-27)

## What this document is

An in-progress brainstorm for a unified redesign of the project's RL controller (reward + observation + network), captured for a red-team review. You (the reviewer) have not seen the conversation; this brief is self-contained.

**What we want from you:** surface things we have **not** considered. Hidden constraints, missing safeguards, coupling risks, untested assumptions, scope creep, false dichotomies, or anything that contradicts what we think we've established. Not asking you to take design decisions for us — asking you to make us aware of what we're missing.

## Project context

**Repository:** `lsy_drone_racing/` (LSY drone-racing autonomous control project; codebase root is this document's parent of parent of parent).

**Task:** A Crazyflie quadrotor races through 4 gates on a track. Two evaluation regimes:
- **L0/L1/L2** = increasing levels of gate / obstacle position randomization at episode start. Static during the episode.
- **L3** = full domain randomization, including obstacle positions revealed only when within sensor range. State of the world differs from the policy's observation until the gate/obstacle enters sensor range.

**Current state:**
- Two RL stacks: `rl_song/` (legacy custom-PPO, where the obs encoder + reward live) and `rl_sbx/` (SBX/JAX PPO, current active branch). The shared `rl_song/obs.py` + `rl_song/reward.py` + `rl_song/config.py` are consumed by both.
- Branch: `rl/reward-fix-2026-05-25` (SBX reward iteration branch, forked from `rl/song-prototype`).
- Current SOTA per memory: v83-step40M = 100/100 L0 @ 4.165 s, 34/100 L2 @ 4.20 s (composite metric 12.35 at n=100). L3 is unresolved (~0% on recent attempts).
- The current `rl_sbx/` line (v112 onward) has not produced a flying policy that finishes — v113h is the only checkpoint that "flies" (passes 2 of 4 gates, 0% finish). Diagnosed root cause is *not* the env or the reward but the SBX framework itself (action saturation, unclipped critic MSE suspected). This is relevant context: the reward redesign is happening on top of a framework that is itself suspect.
- The actor observation was just refactored from 59-d to 57-d (the obstacle channel was rebuilt as a permutation-stable nearest-K-slot layout with identity one-hot + visited flag; proximity scalars removed). This change is uncommitted, on the same branch.

**Stated user objective for the redesign:**
- L3-robust + L2-fast (the hardest of four options offered; weighted blend of L3 finish rate ≥ ~30% and L2 lap time ≤ 4.20s).
- "Build from the ground up" — abandon the v33-v124 accumulated patches; start from a justified base shape.

## Hard constraints

1. **Env action interface is locked.** `race_core.py:213-233` exposes only two control modes: `"state"` (full 13-d state setpoint) and `"attitude"` (4-d `[roll, pitch, yaw, thrust]` Euler+thrust). No body-rate interface. CLAUDE.md hard rule: only `controller.file` and `env.control_mode` can change in `config/levelN.toml` (rest is rejected by online competition's code-check).
2. **Controller class contract.** Controllers in `lsy_drone_racing/control/` must inherit `Controller`; one class per file. Don't modify the base interface.
3. **Asymmetric AC critic** (privileged true gate poses to the critic, masked obs to the actor) is currently load-bearing for L3 per past failure mode where `r_prog` leaked privileged info through the actor reward. The fix is in place.
4. **Tests are out of scope.** Project CLAUDE.md: skip writing tests, validate via training/sim performance.
5. **Code lives in `lsy_drone_racing/control/`** — obs.py, reward.py, config.py shared between rl_song/ and rl_sbx/.

## Decisions made in the brainstorm so far

### Objective
- **L3 robust + L2 fast** (chosen out of four alternative objectives).

### Reward
- Base architecture: **Kaufmann 2023 state-based shape** (verified from paper images), which is essentially Song 2023's distance-delta-to-gate-center plus Kaufmann's action smoothness.
- Components chosen: **5 terms total**
  - `r_prog = λ_1 · (||g − p_{t-1}|| − ||g − p_t||)` — distance-delta to TARGET GATE CENTER. Direction-blind by construction.
  - `r_omega = −b · ||ω||` — body-rate L2.
  - `r_smooth = −λ_5 · ||a_t − a_{t-1}||²` — action smoothness.
  - `r_crash = −C` — terminal collision penalty.
  - `r_finish = +F` — terminal race-finish bonus.
- Components explicitly **rejected**:
  - Time penalty (Song doesn't have it; user opted for strict minimalism; relying on the existing seg-init pipeline to avoid the "do nothing, accumulate zero" attractor at random init).
  - Wrong-side / dipole guard (Song/Kaufmann don't have it; user is accepting direction-blindness of the distance-delta progress).
  - Gaussian barriers (`r_obs`, `r_gate_frame`), aperture shaping (`r_guid`), exit-velocity bonus, jackpot index-scaling, forward-velocity bias (`r_vel`), caution (`r_caution`) — all current-stack components, all dropped.

### Observation
- Total: **52-d** = Song-style vehicle + Song-style track + our obstacle slot layout.
- Vehicle: matches Song's `[v, R]` block where possible (Song 2023 obs is 12-d here; full 9-d rotation matrix). Sub-decisions still open:
  - Drop angular velocity (Song doesn't have it). [Open]
  - Drop z (Song has no position at all). [Open]
  - Swap Zhou 6D rotation → full R(9). [Open — but justification surfaced in conversation: third column of R is the body z-axis = thrust direction in world coordinates; including it explicitly saves the MLP from learning the cross product `r_1 × r_2`; Zhou 2019's continuity argument is about regressing rotations as OUTPUT, not consuming them as input — so doesn't disqualify 9-d input.]
  - Body-frame velocity vs world-frame velocity. [Open]
- Track: Song's recursive corner-deltas scheme (`δp_1` = next gate's 4 corners in vehicle frame; `δp_2` = inter-gate corner delta) vs ours (body-frame target corners + next-gate corners in target-gate frame). Both are "recursive" schemes; specific framing differs. [Open]
- Obstacles: **kept** at 16-d, in the slot layout we just rebuilt (2 nearest obstacles × [body-frame xy (2) + body-frame velocity projection (1) + 4-wide identity one-hot (4) + visited (1)] = 16). Song doesn't have obstacles; our env does.
- Previous action: **dropped** (Song doesn't have it; Kaufmann does). User chose Song's choice.

### Network
- 2 × 256 MLP — matches both Song and Kaufmann.
- Activation: **open**. Kaufmann uses LeakyReLU hidden + tanh output (verified). Song's hidden activation is unspecified in the excerpts seen — only the output tanh is confirmed. Our current `rl_sbx/policy.py` uses tanh everywhere. Switching to LeakyReLU hidden + tanh output would match Kaufmann.
- Action head: currently split thrust/tangent (v132 patch attempting to fix the SBX action-saturation pathology, with separate log-std per head group). [Open whether to revert to single-coupled `Dense(4)` head]. Memory note: the v132 split-head DID NOT actually fix the SBX regression — "Arch was contributor, not full cause." So reverting it doesn't throw away a known-good fix.
- Asymmetric AC stays (load-bearing for L3 per past leak fix).

### Action interface
- **Unchanged.** Stays on the env's attitude mode, with our existing `raw_to_env_action` wrapper that takes `[T_raw, τ_x, τ_y, τ_z]` (thrust + SO(3) tangent vector), squashes T_raw with tanh + rescale into newtons, scales τ by tanh(||τ||) · α_max / ||τ||, exp-maps τ to a ΔR, composes with current orientation, converts to extrinsic xyz Euler. Final env action: `[roll, pitch, yaw, thrust]`.
- This means action smoothness `||a_t − a_{t-1}||²` cannot live on Euler (wraparound). Three candidate locations not yet decided: (p1) raw policy output, (p2) env_action (Euler+thrust, the wraparound hazard), (p3) physical-units intermediate (scaled tangent + squashed thrust in newtons).

## Verified paper details

### Song 2023 (Reaching the limit in autonomous racing, Science Robotics 8 eadg1462)
- Reward Eq. (5): `r(k) = ||g_k − p_{k-1}|| − ||g_k − p_k|| − 0.01 · ||ω_k||`, plus `−10` on collision and `+10` on race finish.
- Obs R^36: vehicle [v (3), R (9)] = 12; track [δp_1 (12) + δp_2 (12)] = 24, N=2 future gates. **No position, no angular velocity, no prev_action**.
- Action R^4: mass-normalized collective thrust + body rates.
- Network: 2×256 MLP, tanh in last layer → output ∈ [-1, 1]. Hidden activation not specified in the excerpts seen.
- Training: "randomly initialized at a new initial state to encourage exploration" = our seg-init.
- Design philosophy: gate progress IS the dense substitute for lap-time-as-reward, providing "a more frequent and informative signal for credit assignments compared to lap time."

### Kaufmann 2023 (Champion-level drone racing using deep RL, Nature)
- Reward: `r_t = r^prog + r^perc + r^cmd − r^crash`, where r^prog is distance-delta (cites Song's earlier work), r^perc = `λ_2 · exp[λ_3 · δ_cam^4]` is camera-axis-at-gate (vision-only, N/A for state-based), r^cmd = `λ_4 · ||a^ω|| + λ_5 · ||a_t − a_{t-1}||²`, r^crash is binary on collision or bounding-box exit.
- Obs R^31: vehicle [p (3), v (3), R (9)] = 15; gate [4 corners in body frame] = 12; prev_action = 4.
- Action R^4: mass-normalized collective thrust + body rates.
- Network: 2×256 MLP, **LeakyReLU hidden + tanh output**. Tanh only on output to bound action in [-1, 1].
- Critic: privileged information (exact position, orientation, velocity of the robot) concatenated into the value-network input.

## Open questions in the brainstorm

1. Vehicle obs final composition (ang_vel? z? Zhou 6D vs full R? body-frame vel vs world-frame vel?).
2. Track obs scheme (Song's recursive vs ours).
3. Network activation (tanh-everywhere vs LeakyReLU-hidden + tanh-output).
4. Network action head (single-coupled vs split thrust/tangent).
5. Action smoothness location (raw policy output / env_action / physical-units intermediate).
6. Reward magnitudes (Song's ±10 vs ours' current ±100 scale).
7. Whether to commit to direction-blind r_prog or hedge with some structural fix for the gate-1→gate-2 U-turn wrong-side attractor (the failure mode that v120 r_wrong_side patched in the legacy reward).

## Codex review request

We've been building this up piecewise via conversation, with multiple paper citation corrections along the way (claims about Kaufmann that turned out to be hallucinated; claims about Song's rotation choice that needed reframing). The user is now asking for an independent review to catch what we have NOT thought about.

Please look at:
1. **Hidden constraints we're ignoring.** What does the env / firmware / sim model actually do that we haven't accounted for?
2. **Coupling between subsystems.** Are reward + obs + network choices actually independent, or are there interactions we're treating as separable when they aren't?
3. **Untested assumptions about paper recipes.** We're matching Song / Kaufmann at the structural level — what specifics of their training pipelines (curriculum, exploration, replay, batch size, learning rate, gradient clipping, KL targeting, GAE, value-loss clipping, normalization) are also load-bearing but absent from the structural match?
4. **Missing safeguards from the legacy code that the redesign drops without justification.** We're cutting 8 reward components — for each one, what failure mode did it patch, and is the new design (or other infrastructure) actually robust against that failure mode? Specifically: the wrong-side attractor, the masked-vs-true gate pose mismatch on randomized levels, the crash-step `[-1,-1,-1]` warp artifact, the random-init exploration coverage.
5. **Scope risks.** Unified controller redesign means one cold-train run with everything changed. If it fails, the ablation surface is large. Should we sequence the changes? If so, in what order, and what's the per-stage validation?
6. **Anything else.** If you spot a class of issue that doesn't fit the above categories — surface it.

Be specific, name code paths or paper sections where relevant. Cite memory references in the repo's `docs/handoffs/` or the project history if helpful. Length: as long as needed; we'd rather have a thorough red team than a tidy summary.
