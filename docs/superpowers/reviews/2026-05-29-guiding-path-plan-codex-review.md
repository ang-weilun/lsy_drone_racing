# Codex review — guiding-path progress plan (2026-05-29)

Reviewer: `codex exec` (codex-cli 0.128.0), read-only over the repo, 2026-05-29.
Brief: `/tmp/codex_plan_review_prompt.md`. Target:
`docs/superpowers/plans/2026-05-29-guiding-path-progress.md`.

## Verdict: fix-then-ship

## Findings (verbatim)

> **1. High: the v1 path can still reward the reverse-out geometry it is meant to
> kill.** For a near-180° successor, `exit_P`, `entry_K`, `center_K` are nearly
> collinear, so the Bézier leg folds back across the just-passed gate.
> Closest-segment projection then becomes discontinuous/ambiguous exactly near the
> just-passed frame. Since `r_gate_frame` is not summed (reward.py:578), Stage 1
> can still incentivize passing back through the old aperture after first moving to
> `exit_P`. That is not a structural fix.
>
> **2. Medium: actor-only warm-start should not reset the critic normalizer.** The
> critic normalizer is observation-distribution state, not old reward geometry.
> Load `critic_normalizer`; reset only critic params.
>
> **3. Medium: the diagnostic is too weak for the claimed gate.** Task 5 only
> checks isolated one-step signs. It does not test a multi-step U-turn after
> `exit_P`, projection segment-choice continuity, or the real pass-step handoff
> (`prev_target=K, current_target=K+1, gate_just_passed=True`).

**Confirmed correct by Codex:** gate convention (`exit = center + d·normal` forward,
`entry = center − d·normal` approach; `gate_passed` requires local-x −→+ crossing,
`target_gate_xaxis_world = rot[:, 0]`); helper array math (Bézier broadcast,
`cum_start`, `argmin`, `take_along_axis`); `target_idx` pass-step timing + zero-on-pass.

> **Top 3 changes:** (1) self-overlap defense before PPO — monotonic segment index
> now, or change path construction so a 180° clears the frame, or move
> `r_gate_frame` into Stage 1; (2) expand the diagnostic (multi-step U-turn, real
> pass-step, reverse-through-old-frame); (3) `init_actor_only` resets critic params
> only, still loads `critic_normalizer`.

## Our assessment

- **#2 confirmed against code** (`checkpoint.py:8`: critic normalizer = Welford stats
  over critic *obs*). Apply: keep `critic_normalizer`, reset only critic params.
- **#3 agreed** — same single-step limitation flagged earlier; expand the diagnostic.
- **#1 agreed, and sharper than stated.** Arc-length progress `r = progress_coef·Δs`
  is **telescoping**: `Σ_t r_t = progress_coef·(s_end − s_start)` — route-independent.
  So bank-around vs reverse-back-through-frame (same start/end arc length) earn the
  *same* total progress. Stage-1 geometry can only penalize the **immediate** reverse
  (leading-segment sign, validated); it cannot, alone, prevent the back-through-frame
  route. γ-discounting and `k_s>0` perturb this only weakly. ⇒ the gate-frame barrier
  is **necessary**, not optional, to clear the frame on the 180°/clip.
  - Nuance: on *realistically offset* ~180° layouts the Bézier bows out laterally and
    the path itself routes around the frame (validated: gate1 at (−1.5, 0.5) passed),
    so geometry still helps materially. The collinear case is the worst case.
- **Push back on "monotonic index now":** it fixes projection well-definedness, not
  frame-clearing, so it does not address #1's core. Gate it on the diagnostic
  (report 2's "add robustness only if the corner is still singular").

## Resulting plan revisions (applied 2026-05-29)

1. **Task 4** — `init_actor_only` resets critic *params* only; always load both
   normalizers.
2. **Task 5/6** — expand the diagnostic: (a) per-step immediate-reverse sign (kept),
   (b) **telescoping/route-independence** demo (bank-around vs reverse-through-frame
   integrated `r_prog` ≈ equal ⇒ progress alone can't clear the frame), (c) real
   pass-step handoff (zero-on-pass), (d) reverse-through-old-frame trace.
3. **First experiment bundles geometry + gate-frame barrier** (RANK 1 + RANK 2, which
   report 2 also endorsed). Single-lever purity is relaxed deliberately: the
   telescoping result proves geometry-solo cannot fix the clip, so a geometry-only run
   would be a known-incomplete experiment. Stage-1 success criteria reframed: geometry
   removes the myopic/immediate-reverse gradient; the barrier clears the frame.
   Monotonic segment index remains a documented contingency.
