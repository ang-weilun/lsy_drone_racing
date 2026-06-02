# Non-target gate-frame observation channel — design

**Date:** 2026-06-02
**Branch:** `rl/obs-completion-capacity-2026-06-02`
**Status:** approved (design), pending plan

## Problem

The actor observation encodes only the `{target, target+1}` gate window
(`N_FUTURE_GATES=2`), as aperture corners in body frame. Gates outside that
window are invisible to the policy. A frame-edge-validated 400-episode L3 eval
of the current SOTA (`relBobs03`, 256-wide, `gate_frame_weight=0`) found:

- **~50 % of crashes are gate-frame collisions**, of which **~80 % are
  non-target gates** (26 confirmed strikes, 0 gross mislabels).
- Mechanism: clipping the **just-passed** gate on exit (12) and **folded-track
  re-approaches** of earlier gates (8); gates the policy can't see *ahead* are a
  non-issue (1 downstream strike).
- Non-target frames are the only major hazard with **neither observation nor
  reward signal** — obstacles get both (barrier + 2-nearest in obs) and are
  managed at ~40 %. Raw artifact: `renders/relBobs03_l3_crashdiag/agg.json`.

See memory `l3-crash-diagnosis-nontarget-frames`.

## Design

Encode **all 4 gates** in the actor obs instead of 2. Every level is a 4-gate
track and the window already covers 2, so exactly 2 gates are blind:
`target+2` and `target+3 (≡ target−1, the just-passed gate)`. Add a slot for
each, assigned by **cyclic offset** (not nearest-rank), so the slots are
permutation-stable by construction — no rank-flip discontinuity, no identity
one-hot (unlike the distance-ranked obstacle channel).

Per new slot: the gate's **4 aperture corners as vectors from the drone in body
frame** (12) + a `visited` flag (1) = **13**. Corners are reused verbatim from
the target-gate encoding: they pin down position **and orientation** (a scalar
distance cannot, and a gate is a thin oriented plane), and the solid frame is
the known ~0.16 m band just outside the aperture rectangle — the same implicit
representation the target channel already uses. The two new slots use
**absolute** body-frame corners (drone-relative), not the recursive inter-gate
delta the aiming slots use, because for *avoidance* the useful quantity is
"where is this bar relative to me." `visited` flags when a blind gate's pose is
still the masked nominal estimate (blind gates, especially `target+2`, are the
ones most likely out of sensor range).

### Gate-channel layout

```
current (24):  [ target corners abs (12) | next-gate Δ corners (12) ]
new     (50):  [ target corners abs (12) | next-gate Δ corners (12)
                | gate[target+2] corners abs (12) + visited (1)
                | gate[target+3] = just-passed corners abs (12) + visited (1) ]
```

`ACTOR_OBS_GATE_DIM: 24 → 50`. `ACTOR_OBS_DIM: 52 → 78` (`81` with
`RL_OBS_ANG_VEL=1`). The just-passed gate — the single biggest crash bucket —
gets a fixed, stable slot the policy can key a specific "clear the exit"
response to.

### Rejected alternatives

- **Nearest-1 (+ identity one-hot):** saves ~9 dims but reintroduces the
  rank-flip discontinuity the cyclic scheme avoids for free, drops one of only
  two blind gates (both can be near in folded sections), and scatters the
  just-passed gate across slots. Nearest-K only pays off at large gate counts;
  every level here is 4 gates.
- **center + normal (compact pose):** ~half the dims but forces the network to
  reconstruct the rectangle via a rotation and breaks consistency with the
  corner-based target channel. Dim cost is noise on a 256/512 MLP.
- **More lookahead gates (`target+2/+3` as aiming slots):** the data shows
  downstream-ahead frames cause ~1 strike; not the failure mode.

## Affected components (must stay in lockstep)

- `rl_song/config.py` — `ACTOR_OBS_GATE_DIM`, `ACTOR_OBS_DIM` assertion; add a
  slot-count constant.
- `rl_song/obs.py` — `build_actor_obs` gate channel (authoritative, JAX);
  `build_critic_obs` inherits it automatically (privileged poses).
- `rl_sbx/deploy_numpy/obs.py` — numpy mirror; identical logic.
- `scripts/check_obs_encoder_parity.py` — must pass unchanged (auto-compares the
  two encoders); it is the pre-flight gate the launcher runs.
- `rl_sbx/controller_ablate.py`, `rl_sbx/controller_diag.py` — inline / overlay
  encoders; update or explicitly note divergence (diag is non-flight).

## Out of scope

The reward is unchanged. `r_crash` already supplies the avoidance incentive, and
the v33 note shows a dense gate-frame barrier "pulls the actor off the natural
path" — re-enabling/widening it (now that the frames are observed) is a separate
follow-up experiment, not part of this change.

## Implications & validation

- **Obs dim changes ⇒ no warm-start from 52-d checkpoints.** This is a
  cold-start: cold-train 2×512 on L2 (minimal Song recipe), warm-start onto L3,
  consistent with the existing `box_launch_l2_screen.sh` / `--init-from` flow.
- **Success metric:** re-run the crash diagnostic (`eval_sim --dump_trace` +
  `scripts/aggregate_crash_causes.py`) on the new L3 policy; the non-target
  gate-frame share of crashes should drop materially from its current ~44 %.
- **Risks:** obs growth 52→78 is trivial for the MLP; masked far-gate pose
  staleness is bounded (~0.15 m) and the critic sees true poses; main cost is a
  fresh training run (no warm-start shortcut).
