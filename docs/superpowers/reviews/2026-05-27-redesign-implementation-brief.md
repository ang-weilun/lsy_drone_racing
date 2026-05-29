# Unified RL Controller Redesign — Implementation Brief for Codex (2026-05-27)

## What you are doing

Implement the unified RL controller redesign (reward + observation + network) that the team brainstormed earlier today. You have not seen the conversation; this brief is self-contained. Make the code changes; do NOT run training. After the diff is in, the user reviews and you stop.

**The brainstorm produced verified decisions. Your job is to land them as code.** The dimension-level sub-questions that the brainstorm left open are pinned in this brief — do not re-litigate them, just implement what's specified.

## Project context

- Repository root: this file's parent of parent of parent (i.e., `lsy_drone_racing/` workspace).
- Two RL stacks: `lsy_drone_racing/control/rl_song/` (legacy custom-PPO, shared obs/reward/config) and `lsy_drone_racing/control/rl_sbx/` (SBX/JAX PPO, active branch). The `rl_song/{obs.py, reward.py, config.py}` modules are consumed by both stacks.
- Branch: `rl/reward-fix-2026-05-25`.
- A parallel SBX framework-fix is currently smoke-testing on a remote GPU box. **Do not touch `rl_sbx/jit_scan_ppo.py` or `rl_sbx/train.py`** — those are mid-validation. Any other file under `rl_sbx/` and `rl_song/` is fair game (subject to the hands-off list below).
- Already-uncommitted in the tree: the obstacle channel was rebuilt from a per-obstacle block to a nearest-K slot layout (see `rl_song/obs.py` lines 289-354 + `rl_song/config.py` 40-69). **Do not undo this work.** Your obs changes go on top of it.

## Codebase rules (from project CLAUDE.md — these apply to every file you touch)

- `ruff format` + `ruff check --fix` must pass clean. Line length 88. Run before declaring done.
- PEP 8 naming. Type hints (PEP 484) on all public function signatures. `from __future__ import annotations`; use built-in generics.
- numpydoc-style docstrings (Parameters / Returns / Raises / Notes / References). Document array shapes explicitly.
- No bare `except`. No `assert` for runtime validation in non-test code — raise `ValueError`/`TypeError`.
- No magic numbers — lift to module-level constants with a comment explaining units / source.
- Comments explain WHY, not WHAT.
- **NO AI-assistant branding.** No `Co-Authored-By: Claude`, no AI disclaimers in code/comments. Don't introduce them; don't perpetuate them if you see them elsewhere.
- Don't write tests; don't run pytest.
- JAX-specific: pure functions only inside `jit`/`vmap`. No Python-side state mutation inside transformed functions. Use `jax.numpy` not `numpy` inside transformed functions.

## Hands-off files (do NOT modify)

- `lsy_drone_racing/control/rl_sbx/jit_scan_ppo.py` — SBX clipped-VF patch mid-validation.
- `lsy_drone_racing/control/rl_sbx/train.py` — same.
- `lsy_drone_racing/control/rl_song/obs.py` lines 289-354 (the obstacle-channel slot layout) — keep semantics, you can adjust surrounding obs blocks.
- `lsy_drone_racing/envs/race_core.py` and anything else under `lsy_drone_racing/envs/` — env is locked by the competition's code-check.
- `config/levelN.toml` — only `controller.file` and `env.control_mode` are permitted edits, and you don't need either.

## Decided design

### Reward — five components, distance-delta progress

Keep ONLY these. Delete or gate-off-by-default everything else in `rl_song/reward.py`.

1. `r_prog = λ_1 · (||g - p_{t-1}|| - ||g - p_t||)` — distance-delta to target gate center. Direction-blind by construction; this is the verified Song 2023 / Kaufmann 2023 form. Use the *distance-delta* code path (`reward.py:262-263` today), not the velocity-projection branch. Drop the lookahead branch entirely.
2. `r_omega = -b · ||ω||` — Song's b·||ω||. Already present at `reward.py:349-352`.
3. `r_smooth = -λ_5 · ||a_t - a_{t-1}||²` — NEW component. Action smoothness. See "Smoothness location" below for where this lives.
4. `r_crash` — terminal collision penalty, on `terminated & ~finished`. Already at `reward.py:574-576`.
5. `r_finish` — terminal race-finish bonus. Already at `reward.py:574-576`.

Components to **remove from the reward sum** (the user is committing to Song's minimalism):

- `r_gate_frame`, `r_obs` (Gaussian barriers)
- `r_gate_bonus` (jackpot)
- `r_exit_vel`
- `r_wrong_side`, `r_dipole` (direction-blind r_prog is accepted; the wrong-side attractor reopens)
- `r_caution`
- `r_guid` (both static-field and Δ-Φ variants)
- `r_vel` (forward-flight bias)
- `r_time` (per-step time penalty; seg-init pipeline is expected to break the don't-move attractor)

**Don't delete the dead code unless removal is clean.** If a component lives behind a `use_*` flag in `RewardConfig`, set the default to False and leave the code path so historical reproducibility is preserved. If a component is hardwired, either gate it with a new `use_*` flag (default False) or delete it cleanly with a comment in `reward.py` pointing to git blame.

**Preserve the crash-step zeroing of position-dependent dense terms.** `reward.py:584-608` zeros `r_prog` and friends on the crash step because the env warps the disabled drone to `[-1, -1, -1]` before emitting the post-step obs. The new `r_prog` (distance-delta) is position-dependent and must continue to be zeroed on the crash step — otherwise a crash can mint a fake positive `r_prog`. This is not optional cleanup.

### Smoothness location (where `r_smooth` lives)

Compute `||a_t - a_{t-1}||²` on the **physical-units intermediate** — scaled tangent + thrust in newtons. NOT on raw policy output (T_raw is unbounded pre-tanh), NOT on env_action (Euler angles wrap at ±π and are ill-conditioned).

Specifically: in the reward function, you need access to the *scaled tangent* `τ_scaled` (a 3-vec, bounded by `α_max` rad) and the *squashed thrust* (a scalar in newtons). These are intermediates of `rl_song/policy.py:raw_to_env_action` (`policy.py:262-274`). Two implementation options:

- **(opt-A) Recompute in reward:** `raw_to_env_action` is pure-JAX and cheap; recompute `scale_tangent(τ_raw, α_max)` + `thrust = thrust_min + (thrust_max - thrust_min) * 0.5 * (tanh(T_raw) + 1)` inside the reward. Requires the reward to receive the raw policy output. Currently the reward sees `prev_action` as the env_action (Euler+thrust); you'd need to thread the raw policy output too.
- **(opt-B) Thread the physical intermediate from the rollout loop:** the rollout already computes these to call `env.step`; expose them as a field in the rollout output (`rl_sbx/rollout.py` and the equivalent `rl_song/rollout.py`) and pass them into `step_reward`. Cleaner once done but touches more files.

Recommend (opt-B). Pick what's cleanest given the rollout structure; document the choice in a comment.

The smoothness scalar then is `||δ||²` where `δ = [τ_scaled_t - τ_scaled_{t-1}, thrust_t - thrust_{t-1}]`, a 4-vec in physical units. Mixed units (rad + newtons) is intentional — both components are now bounded (τ in `[-α_max, +α_max]`, thrust in `[thrust_min, thrust_max]`), so the squared norm is dimensionally well-defined as a sum-of-squares once you scalar-normalize.

Add a `r_smooth_coef` (default value: leave it as a config knob, no specific number — let the user tune). Wire it through `RewardConfig` and the reward sum.

### Observation — 52-d Song-style + obstacle slot layout

Total `ACTOR_OBS_DIM = 52`. Composition:

| Block | Dims | Notes |
|---|---|---|
| Vehicle | 12 | Song's `[v, R]` — see "Vehicle obs sub-decisions" below. |
| Track | 24 | Two future gates, recursive frame chain (see "Track obs scheme"). |
| Obstacles | 16 | The slot layout that's already in the code: 2 nearest obstacles × [body-frame xy (2) + body-frame velocity-projection (1) + 4-wide identity one-hot (4) + visited (1)]. **DO NOT TOUCH** this block — it's already implemented at `obs.py:289-354`. Only adjust the surrounding blocks.|
| Prev action | 0 | **DROPPED.** Song doesn't have it; the user committed to dropping it. Remove from obs, remove the channel from the `concatenate`, update the dim arithmetic. |

#### Vehicle obs sub-decisions (pin these)

- **Drop angular velocity (-3).** Song doesn't include ω. We have `r_omega` in the reward and the network can infer ω from rotation-matrix history. Removes 3 floats.
- **Drop z (-1).** Song has no position at all. Drop the z component. Translation-invariant; the policy can infer altitude from gate-corner Y/Z.
- **Switch Zhou 6D → full 9D rotation matrix R (+3).** The user explicitly raised this: the third column of R is the body z-axis (thrust direction in world coordinates). For a quadrotor this is the most physically important direction; including it explicitly saves the MLP from learning the cross-product `r_1 × r_2` (which MLPs are inefficient at because it's a multiplicative cross-channel interaction). Zhou 2019's continuity argument is about regressing rotations as OUTPUT, not consuming them as INPUT — so doesn't disqualify 9D input. Both Song and Kaufmann use 9D for input.
- **Keep body-frame velocity.** Don't swap to world-frame. Body-frame is yaw-equivariant; we already use it; changing this is unnecessary diff surface.

Final vehicle obs: 9 (full R) + 3 (body-frame velocity) = 12-d. No position, no angular velocity. Matches Song's `o^quad ∈ R^12`.

#### Track obs scheme (pin this)

Switch to **Song 2021's recursive corner deltas**. From the Song 2023 obs description (verified by the user from the paper):

> `o^track = [δp_1, ..., δp_N] ∈ R^{12N}` where `δp_i ∈ R^12` denotes the relative position between the vehicle center and the four corners of the next target gate i OR the relative difference in corner distance between two consecutive gates. N = 2.

Implementation:
- `δp_1` = body-frame relative position of the next gate's 4 corners (12 floats; same as today's `target_corners_body` at `obs.py:269`).
- `δp_2` = inter-gate corner delta: `(corners of next-next gate) - (corners of next gate)`, expressed in the same frame (body frame OR target-gate frame — pick one and document; Song's paper doesn't specify but body-frame is the simplest consistent choice). 12 floats.

Total track obs: 24-d. Same dimensionality as today; new semantics. Replace the current `next_corners_in_target` construction with the inter-gate delta.

### Network — Song-style 2x256 MLP, LeakyReLU hidden + tanh output, single-coupled head

- **Trunk:** 2 layers × 256 units. Already matches (`rl_sbx/policy.py:70-72`). Leave the trunk dims alone.
- **Hidden activation:** switch from tanh to **LeakyReLU** (the Kaufmann 2023 confirmed convention; Song doesn't specify in the excerpts we've seen). Default LeakyReLU α (negative slope) = 0.01.
- **Output activation:** tanh on the final layer to bound action in [-1, 1] (matches both papers and our current code).
- **Action head:** revert the v132 split thrust/tangent heads back to a **single coupled `Dense(4)`** head with a single state-independent `log_std` vector (the SBX stock convention). Per memory `[[project-sbx-actor-arch-pathology]]`, the split-head change "was contributor, not full cause" of the SBX saturation regression — reverting it doesn't throw away a known-good fix, and it brings us closer to Song's single-head structure. The active SBX patch (clipped VF) is the likely actual fix; running both at once muddies the ablation.
- **Asymmetric AC:** keep. The critic still receives true (unmasked) gate poses; the actor still receives masked obs. The privileged-info leak fix from `[[project-l3-privileged-info-leak]]` stays applied. Do not collapse the actor and critic obs.
- **Mirror to `rl_song/policy.py`** if the rl_song policy has the same split-head pattern — both stacks should switch together so they stay comparable.
- **Hidden init / log-std init / log-std floor:** leave as they are now (ortho-init optional, log-std init -0.5, log-std floor -2.5). Don't reopen those — separate concerns.

### Action interface — unchanged

The env's `attitude` control mode is locked (per `race_core.py:213-233`). Keep the current `[T_raw, τ_x, τ_y, τ_z]` raw → `raw_to_env_action` → `[roll, pitch, yaw, thrust]` env_action pipeline. Don't touch `rl_song/policy.py:raw_to_env_action` or `TANGENT_ALPHA_MAX_RAD`.

## Commit shape (do this please)

Three separate commits, in this order, each independently buildable + passing `ruff format && ruff check`:

1. **Reward redesign.** Touch `rl_song/reward.py` + `rl_song/config.py` only. Reduce to 5 components, add `r_smooth_coef`, preserve crash-step zeroing, gate-flag the dropped components to defaults False (don't delete unless removal is genuinely clean). Add `r_smooth` (the smoothness scalar) wired through `RewardConfig`. The actual smoothness computation can be a placeholder that consumes a `prev_raw_action` field passed in from the rollout if (opt-A) is chosen; if (opt-B), the threading change comes in commit 3 along with the rollout edits.

2. **Observation redesign.** Touch `rl_song/obs.py` + `rl_song/config.py` only. Drop prev_action from obs, drop ang_vel, drop z, switch to full 9D rotation matrix, switch track scheme to recursive inter-gate deltas. Update `ACTOR_OBS_DIM` (target: 52), update the `_one_hot` / vmap shapes if needed. Don't touch the obstacle slot block (lines 289-354).

3. **Network redesign + smoothness wiring.** Touch `rl_sbx/policy.py` (LeakyReLU hidden + single coupled head); optionally `rl_song/policy.py` for the same; if you chose (opt-B) for the smoothness location, the rollout-side threading for `prev_raw_action` (or whatever intermediate) goes here.

Each commit should have a concise commit message describing the change, no Co-Authored-By trailer.

## Coordination warnings

- The SBX clipped-VF patch is at `rl_sbx/jit_scan_ppo.py` + `rl_sbx/train.py`. Don't touch either file.
- Obs.py currently asserts `ACTOR_OBS_DIM == 57` after my obstacle channel change. Your obs commit (#2) will need to update the assertion to 52 once the prev_action / ang_vel / z drops + rotation upgrade lands.
- Several consumers reference `ACTOR_OBS_DIM` symbolically (`rl_sbx/{policy.py, controller*.py, env_gym.py, rollout.py, callbacks.py}` and `rl_song/{policy.py, env_wrapper.py, train.py, controller.py}`) — they pick up the new dim automatically. The flat-concat `2 * ACTOR_OBS_DIM = 104` in `env_gym.py` likewise picks up the new dim.
- `rl_song/env_wrapper.py:181` still passes true poses into the reward, while `rl_sbx/rollout.py:516` doesn't. This stack divergence is NOT something to fix in this redesign — leave it for a separate change. If you encounter it, note it in a TODO comment in the file but don't touch the divergence itself.
- Memory references in this brief use `[[name]]` link syntax referring to memory files under the user's private store; you don't have access. Treat them as labels naming past project history.

## What "done" looks like

- Three commits land on the current branch (`rl/reward-fix-2026-05-25`).
- `ruff format && ruff check --fix` passes clean on all touched files.
- `python -m py_compile` on every touched file passes.
- A brief commit message per commit (subject + 1-2 sentence body).
- No tests written, no training run.

If anything in this brief contradicts itself or requires a choice I haven't pinned (e.g., the smoothness location opt-A vs opt-B was deferred to you), pick the option that minimizes diff surface and document the choice in a code comment.
