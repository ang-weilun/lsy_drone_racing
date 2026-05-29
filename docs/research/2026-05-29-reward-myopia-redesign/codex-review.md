# Independent review (Codex) — diagnosis and proposed routes

Reviewer: `codex exec` (codex-cli 0.128.0), read-only over the repo, 2026-05-29.
Brief archived at `/tmp/codex_review_prompt.md`. Codex independently inspected
`rl_song/reward.py`, `rl_song/config.py`, and the `rl_sbx` call sites.

## Headline: a verified reward-accounting defect (the most important finding)

Codex's top finding — which we then verified against git — is that the
**working-tree scalar reward omits `r_time` and every shaping term.**

- `lsy_drone_racing/control/rl_song/reward.py:566` (working tree):
  `reward = r_prog + r_omega + r_smooth + r_crash + r_finish`
- Committed `HEAD` summed the full set:
  `r_prog + r_omega + r_obs + r_gate_frame + r_gate_bonus + r_exit_vel + r_terminal + r_time + r_vel + r_guid + r_caution`.
- The working-tree change is **uncommitted**, written **2026-05-27 10:52 UTC**.
- `rl_sbx` trains on the scalar return of `step_reward`
  (`rollout.py:523`, `env_gym.py:282`); the `components` dict (which contains
  `r_time`, `r_guid`, `r_dipole`, `r_wrong_side`, `r_gate_frame`, `r_obs`,
  `r_exit_vel`, `r_vel`, `r_caution`) is logged but **never re-summed**.

**Consequences:**
1. `time_penalty` and every shaping lever plumbed through `train.py` CLI are
   **no-ops** in the current tree — weights are set, terms are computed and
   logged, but they do not enter the trained scalar.
2. The 2026-05-28 L2 "time_penalty chain" (warm2390→warm2790) ran with this
   working tree (the edit predates it by ~15 h). **`r_time` was not in the
   reward.** The lap-time gains are therefore best explained by γ-discounting of
   the progress + finish rewards (faster laps collect `+finish` and progress
   sooner → higher discounted return) operating over additional warm-start
   training steps — **not** by the `time_penalty` knob. The single-lever
   attribution was confounded: each tp bump was also another +200 M of training.
   This matches the handoff's own progress-based `lap ≈ 1.9 / r_prog_per_step`
   rule and the observed "diminishing returns" (a training curve asymptoting).
3. The policy is effectively trained on **pure centre-distance progress + γ +
   crash/finish** — exactly the configuration the literature says produces
   myopic point-to-point flight. The diagnosis is thus confirmed *and* its cause
   is sharper than "reward geometry": most of the intended reward is silently
   disabled.

This must be reconciled before any reward experiment (probe, Route A, Route B):
decide which terms should be summed and sum them; verify component reward on
scripted trajectories before launching PPO.

## Codex review (verbatim)

> **Top Finding**
> Your premise has a serious code-level hole: in reward.py:566, the returned
> scalar is only `reward = r_prog + r_omega + r_smooth + r_crash + r_finish`.
> So `r_time`, `r_guid`, `r_wrong_side`, `r_dipole`, `r_vel`, `r_gate_frame`,
> `r_obs`, `r_gate_bonus`, and `r_exit_vel` are computed/logged but not trained
> on. SBX also calls this same `step_reward` path. Therefore your "cheap
> pre-probe" will not test anything unless the scalar sum is fixed first. Also,
> `r_prog` is not zeroed on gate-pass in this function; it is computed against
> `prev_target` on the pass step at reward.py:239. Liu zeroes progress on pass;
> your local reward code does not.
>
> **Diagnosis**
> The geometric diagnosis is directionally right but incomplete. Center-distance
> progress absolutely creates the backward/lateral pull after target switch. But
> calling this "purely reward geometry" is too narrow. Other likely contributors:
> - Action authority: `TANGENT_ALPHA_MAX_RAD = 0.16` in config.py:38. If you are
>   actually sweeping 0.32–0.48, that is outside the checked default, but
>   saturation metrics matter. A 180° transition needs enough lateral
>   acceleration and yaw/roll authority; otherwise "reverse out" may be the only
>   controllable solution the policy finds.
> - Reward implementation mismatch: your claimed active time penalty and plumbed
>   levers are not in the scalar reward. This can fully explain why knobs appear
>   ineffective.
> - Gate-frame collision is terminal-only in the actual scalar. The policy gets
>   no smooth cost for clipping the just-passed frame, so it is unsurprising it
>   learns to shave it.
> - Segment-init can bias the value function toward locally valid gate approaches
>   without teaching continuous inter-gate exit dynamics.
>
> **Route A** — I would not implement the proposed Penicka route literally.
> `exit_N → entry_{N+1} → center_{N+1}` does not magically remove the 180°
> singularity. On a true U-turn it places the kink at `exit_N` or between
> `exit_N` and `entry_{N+1}`. Nearest-point projection over the whole polyline
> causes wrong-segment snaps and discontinuous gradients on dense/random layouts;
> monotonic-by-segment needs stateful per-env progress bookkeeping. The
> `+k_s·s(p)` term is dangerous per step — it rewards being at high arc length
> every step, so loitering late in the path can become profitable. Treat it as a
> hack for projection singularities, not a clean fix. Route A only becomes
> credible with monotonic segment index, minimum waypoint spacing, pruning for
> crossed/near-duplicate segments, and a smoothed curve.
>
> **Route B** — Liu-style guidance will likely help lateral target-gate approach
> errors, but it is not sufficient for your two symptoms. For the lateral clip of
> gate N *after* passing it, target-gate guidance on gate N+1 is the wrong
> object — you need an active **previous-gate frame clearance penalty** or an exit
> corridor for gate N; otherwise the old frame is invisible to the scalar except
> via terminal collision. For the 180° reverse-out, Liu guidance only counteracts
> the center-distance pull; it does not remove it. Co-equal `k_guidance ≈
> k_progress` is not scale-safe (your `progress_coef` is 15, gates ~0.4 m, 50 Hz).
>
> **Pick** — Pick a hybrid, but first fix the scalar reward bug.
> 1. Fix `reward = …` so intended terms contribute; make CLI flags set both
>    weights and `use_*` booleans where applicable.
> 2. Add a previous/current/next gate-frame barrier to stop frame clipping
>    (directly targets the lateral symptom).
> 3. Replace center progress with progress to an **entry waypoint**
>    (`center − d·gate_normal`), pass/center only near the aperture — removes much
>    of the wrong-side pull without global-projection complexity.
> 4. Add wrong-side rejection for the target gate, kept local and budgeted.
> 5. Only consider full arc-length path progress after the simpler
>    entry-waypoint/corridor version fails on held-out 180° layouts.
> Not pure Route A (bigger rewrite, new failure modes); not pure Route B (leaves
> the center-distance attractor intact, does not price the just-passed frame).
>
> **Warm start** — Warm-start the actor cautiously; reset or heavily de-emphasize
> the critic after a progress-term change (its values encode the old reward
> geometry). The Phase-2 replay buffer is suspect after a reward redesign: if it
> stores returns/advantages/value-targets tied to the old reward, discard it;
> even if it stores only "successful" states, reduce replay probability initially
> (it over-samples behaviour successful under the old objective).
>
> **Scale transfer** — Liu/Penicka coefficients do not transfer cleanly (smaller
> gate, lighter drone, 50 Hz, tiny aperture margin; a 0.08 m gate-frame sigma is a
> large fraction of a 0.20 m half-aperture). Tune in physical units and integrated
> episode budgets, not raw literature coefficients.
>
> **What you're missing** — The simplest robust fix is not "Penicka vs Liu"; it is
> reward accounting plus corridor constraints: make intended terms enter the
> scalar; add smooth gate-frame and obstacle barriers; use entry/exit corridor
> progress around gates instead of center progress everywhere; verify on scripted
> bad trajectories by plotting component rewards before running PPO. Sparse reward
> is a warning against over-shaped deceptive gradients, but sparse-only on L3 with
> random obstacles is likely sample-inefficient; use sparse outcomes as
> evaluation/regularization, not the first replacement.
>
> **Prioritized verdict**
> 1. Fix the scalar reward bug first. Current experiments are not testing the
>    levers you think they are.
> 2. Do not run the cheap pre-probe until that is fixed.
> 3. Implement hybrid entry-waypoint progress + previous/current gate-frame
>    barrier + wrong-side rejection.
> 4. Warm-start actor, reset critic/replay targets.
> 5. Avoid full Penicka path projection until simpler local geometry fails under
>    controlled trajectory diagnostics.

## Our verification of the headline finding

- `git show HEAD:…/reward.py` → committed sum includes `r_time` and all shaping
  terms (each gated by its weight, defaulting to 0 — so harmless when off but
  *live* when a CLI weight is set).
- Working tree (`git diff HEAD`) replaced that block with the 5-term sum and
  updated the docstring to match — a deliberate-looking edit, but inconsistent
  with the CLI flags plumbed the same week for the now-dropped terms.
- `stat` → reward.py mtime 2026-05-27 10:52 UTC; tp chain ran 2026-05-28
  ~02:00–09:30 UTC. The edit predates the chain.
- `grep rl_sbx` → both reward call sites use the scalar; no separate `r_time`
  application; `components` never re-summed.

**Open items flagged for follow-up:** (a) confirm from wandb that episode return
did *not* respond to `time_penalty` across the chain (decisive cross-check);
(b) confirm whether the v85 "r_prog leak" fix actually zeroes progress on the
pass step or only clamps `target_idx` (Codex says reward.py:239 still computes
against `prev_target` on the pass step); (c) decide whether the 5-term reward
edit was intentional, and which terms to restore.

## Update (post-review, 2026-05-29)

- **The 5-term strip was intentional**, confirmed by the team: the accreted patch
  terms were removed for a pure Song-style reward, which trained markedly better.
  So Codex's "bug" framing is a mis-read of *intent* — but its substantive point
  stands: in this tree, `time_penalty` and the shaping CLI flags are **inert**
  unless the corresponding term is added back into the summed scalar.
- **wandb cross-check done (item a), misattribution confirmed.** Run summaries for
  warm2200→2790: `reward/r_time` = {0, −0.10, −0.15, −0.20} (tracks the tp label,
  but not summed) while trained `reward/r_prog` rose {0.327, 0.369, 0.403, 0.426};
  the team's `lap ≈ 1.9/r_prog` rule predicts the measured laps
  {5.71, 5.26, 4.73, 4.42 s} to ~1–2%. The lap-time gains are progress-optimisation
  over continued warm-start training (+ γ), not the `time_penalty` knob. (No
  trained-total key is logged in wandb, so the airtight proof is the source code;
  wandb corroborates via the progress mechanism.)
- **Revised plan:** keep pure-Song as the baseline; test the geometric fix by
  adding **one** term to the sum (Codex's hybrid: entry-waypoint progress +
  previous/current gate-frame barrier + wrong-side rejection), actor-only
  warm-start, critic + Phase-2-replay reset. Item (b) — the pass-step `r_prog`
  zeroing — remains open to verify before implementing.
