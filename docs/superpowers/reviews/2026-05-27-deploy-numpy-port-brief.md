# Deploy-numpy port — Implementation Brief for Codex (2026-05-27)

## What you are doing

Port `lsy_drone_racing/control/rl_sbx/deploy_numpy/obs.py` from the legacy 59-d
observation layout to the current 52-d layout. The JAX path
(`lsy_drone_racing/control/rl_song/obs.py`) is the source of truth; this is a
pure numpy mirror used by `controller_numpy.py` for fast CPU evaluation
(~17 s/ckpt vs ~2-3 min/ckpt for the JAX controller).

You have not seen the conversation; this brief is self-contained. Make the
code change, run the validation, stop. Do NOT run training.

## Project context

- Repository root: this file's parent of parent of parent.
- Two RL stacks: `lsy_drone_racing/control/rl_song/` (legacy custom-PPO, shared
  obs/reward/config modules) and `lsy_drone_racing/control/rl_sbx/` (SBX/JAX
  PPO, active). The redesigned 52-d obs lives in `rl_song/obs.py` and is
  consumed by both stacks via shared imports.
- Branch: `rl/reward-fix-2026-05-25`.
- A long L3 training run is currently active on a remote GPU box. It does not
  touch any file you touch — work freely.
- `deploy_numpy/` was developed on the GPU box and only just rsync'd to the
  dev VM. Its `obs.py` was last touched when `ACTOR_OBS_DIM == 59`. The JAX
  side has since landed the Song-pure redesign (52-d), so the numpy mirror
  asserts the wrong dim and crashes at first use on redesign checkpoints.

## Codebase rules (from project CLAUDE.md)

- `ruff format` + `ruff check --fix` must pass clean. Line length 88. Run
  before declaring done.
- PEP 8 naming. Type hints (PEP 484) on all public function signatures.
  `from __future__ import annotations`; use built-in generics
  (`list[float]`, not `List[float]`). Use `npt.NDArray` for arrays.
- numpydoc-style docstrings (Parameters / Returns / Raises / Notes /
  References). Document array shapes explicitly.
- No bare `except`. No `assert` for runtime validation in non-test code —
  raise `ValueError`/`TypeError`.
- No magic numbers — lift to module-level constants with a comment
  explaining units / source.
- Comments explain WHY, not WHAT.
- **NO AI-assistant branding.** No `Co-Authored-By: Claude`, no AI
  disclaimers in code/comments.
- Don't write tests under `tests/`. Don't run pytest. A standalone
  validation script in the existing `parity_check.py` shape is fine.
- Prefer the ecosystem: keep using `scipy.spatial.transform.Rotation` for
  quat-to-matrix on the numpy side. (The JAX side hand-rolls
  `_quat_to_matrix` for a documented training-throughput reason; that
  exemption doesn't apply to the numpy deploy path, which runs at 100 Hz
  for ~10 s per episode.)

## Hands-off files (do NOT modify)

- `lsy_drone_racing/control/rl_song/obs.py` — **the source of truth.** Read
  it; mirror it; don't change it.
- `lsy_drone_racing/control/rl_song/config.py` — `ACTOR_OBS_DIM`,
  `N_OBSTACLES`, `N_NEAREST_OBSTACLES`, `ENV_ACTION_DIM` live here. Don't
  change them.
- `lsy_drone_racing/control/rl_sbx/jit_scan_ppo.py`,
  `lsy_drone_racing/control/rl_sbx/train.py` — training code, irrelevant to
  this port.
- `lsy_drone_racing/envs/` — locked by competition code-check.
- `lsy_drone_racing/control/rl_sbx/deploy_numpy/constants.py` — already
  patched on the dev VM to read constants from both `obs.py` and `config.py`.
  Do not undo. You may need to extend its `_SEARCH_PATHS` only if a constant
  you need is missing, which I don't expect.

## In-scope files

You should only need to edit:

1. `lsy_drone_racing/control/rl_sbx/deploy_numpy/obs.py` — the actual port.
2. `lsy_drone_racing/control/rl_sbx/deploy_numpy/parity_check.py` —
   **only if** the existing parity check needs an update because of the
   obs-dim change. Read it first; the current version should still work end
   to end because it tests `RLSBXNumpyController.compute_control` vs
   `RLSBXController.compute_control` (action-level parity), not obs-level
   internals. If untouched, leave untouched.
3. `lsy_drone_racing/control/rl_sbx/deploy_numpy/__init__.py` — only if a
   re-export name changes. Read first; should be untouched.

## Decided design — mirror the 52-d JAX layout

Read `rl_song/obs.py` end to end before editing. The new layout is:

```
ACTOR_OBS_DIM = 52
  drone     (12)  = full 9D rotation matrix R (9, row-major flat) +
                    body-frame linear velocity (3).
                    NO position, NO altitude (z), NO angular velocity.
  gates     (24)  = target-gate four corners in DRONE BODY FRAME (12) +
                    inter-gate corner delta in DRONE BODY FRAME (12),
                    where the inter-gate delta is
                    (next_gate_corners_world - target_gate_corners_world)
                    rotated into the drone body frame via R_bw.
  obstacles (16)  = N_NEAREST_OBSTACLES=2 slots, each
                    [xy_body (2), vel_proj (1), identity_onehot (4),
                     visited (1)] = 8 per slot.
                    NO global proximity pair at the end.
total = 52
```

Key differences from the current `deploy_numpy/obs.py`:

- **Drop the 6D rotation block.** Switch to the full 9-element rotation
  matrix `R_wb` (world-from-body), flattened. Matches `rot_wb.reshape(9)` in
  `rl_song/obs.py:233`.
- **Drop angular velocity from the drone channel.**
- **Drop the altitude column `pos[2:3]`.** No position in the obs.
- **Drop the prev_action concatenation entirely.** The function still takes
  `prev_action` as a parameter for call-site compatibility (matching the
  JAX signature in `rl_song/obs.py:188`), but it is unused. Add `_ = prev_action`
  exactly as the JAX side does to silence linters.
- **Switch the second gate block from "next gate corners in target-gate
  frame" to "inter-gate corner delta in drone body frame."** Specifically:
  `inter_gate_delta_body = (g_next_corners_w - g_target_corners_w) @ rot_bw.T`.
  See `rl_song/obs.py:248-249`.
- **Replace the obstacle block with the slot layout.** Per slot (K=2 slots):
  - `xy_body` (2): body-frame XY of that obstacle, projected onto the
    drone's altitude plane (set obstacle z = drone z before subtracting),
    then rotated into the body frame. Same trick as the current code.
  - `vel_proj` (1): scalar `<vel_body_xy, unit_to_obstacle_xy_body>`.
    Positive when closing. Guard the unit vector with
    `safe_norm = max(dist, 1e-6)`.
  - `identity_onehot` (N_OBSTACLES = 4): a 4-wide one-hot saying which
    physical obstacle (by original index in `obstacles_pos`) currently
    occupies this slot.
  - `visited` (1): `obstacles_visited[nearest_idx]` cast to float32.
  - Slots are ranked by ascending body-frame XY distance. Use
    `np.argsort(...)[:N_NEAREST_OBSTACLES]` to get the slot ordering, same
    as the JAX side's `jnp.argsort` (mergesort-stable behavior is not
    required; the parity check will fly through any standard sort because
    distances are continuous and ties are negligible).
  - Flatten K slots × 8 = 16 floats into the obstacle channel.
- **Drop the trailing `proximity_chan`** (the 2-float pair `[min_clearance_xy,
  closing_speed]`). The slot layout subsumes both signals (slot-0 distance
  ≡ min clearance; slot-0 vel_proj ≡ closing speed toward nearest).

Final concatenation order: `[drone_chan, gate_chan, obstacle_chan]`. Same as
JAX. No prev_action block.

### Things that stay the same

- The function signature `build_actor_obs(env_obs, prev_action, normalizer,
  gate_corners_local=None) -> ndarray`. `controller_numpy.py` calls it with
  exactly these kwargs.
- The `_gate_corners_world(pos, quat, corners_local)` helper. The local
  gate-corner template (`_GATE_CORNERS_LOCAL`, populated from
  `GATE_HALF_SIZE_M`) is unchanged.
- The `apply_normalizer(normalizer, raw)` call at the end.
- The `gate_corners_local()` accessor for the precomputed template (used
  by `controller_numpy.py` for one-time setup).
- All quat-to-matrix conversions stay on
  `scipy.spatial.transform.Rotation.from_quat(...).as_matrix()`. The JAX
  side hand-rolls only because of training throughput; numpy deploy is
  fine on scipy.
- `target_idx` handling for `target_gate == -1` (race finished): clamp to
  0 so downstream gather stays in-bounds, exactly as the JAX side does at
  `rl_song/obs.py:228`.

## Validation (gate; do not skip)

Two checks. Both must pass before you stop.

### 1. obs-level parity (fast, no checkpoint required)

Write a one-shot script (or inline `python -c`) that:

1. Builds a deterministic fake `env_obs` with the same shapes as
   `parity_check.py:_fake_env_obs()` (you can copy it).
2. Calls `lsy_drone_racing.control.rl_song.obs.build_actor_obs(env_obs,
   prev_action, normalizer)` (the JAX path; convert inputs to
   `jnp.asarray`, output to `np.asarray`). Use the freshly-initialized
   normalizer from `init_normalizer(ACTOR_OBS_DIM)`.
3. Calls `lsy_drone_racing.control.rl_sbx.deploy_numpy.obs.build_actor_obs(
   env_obs, prev_action, normalizer_state_numpy)`. The numpy normalizer
   wraps the same `mean=0, var=1, count=NORM_VAR_EPS` warm-start state
   — see `deploy_numpy/normalizer.py:from_jax_state` for the
   conversion helper, or build it directly.
4. Compares the two 52-d vectors with `np.testing.assert_allclose(jax_obs,
   np_obs, atol=1e-5, rtol=1e-5)`. Print `max_abs_diff` and the first 6
   components of each.

This step does NOT need a checkpoint. It tests only the
encoding-and-normalize path, which is what the port changes.

### 2. controller-level parity (existing parity_check.py)

The existing `parity_check.py` already wires this up: load a redesign
checkpoint, build both `RLSBXController` (JAX) and `RLSBXNumpyController`
(numpy), feed the same fake `env_obs`, assert action-level parity with
`atol=1e-5`. Run it once you have a checkpoint to point at:

```
pixi run -e rl-train python -m lsy_drone_racing.control.rl_sbx.deploy_numpy.parity_check \
  <path/to/redesign/step_*/>
```

There is no redesign checkpoint on the dev VM yet. If a checkpoint is not
available at run time, skip step 2 and document that — step 1 is the
load-bearing parity test for the port itself; step 2 is for end-to-end
deploy validation, which requires a checkpoint we'll grab separately. If
you find a redesign checkpoint under `lsy_drone_racing/control/rl_sbx/
checkpoints/` (any directory starting with `sbx_redesign_*`), run step 2
and paste the `max_abs_diff` output into the commit message.

If either check fails:
- First, recheck against `rl_song/obs.py` element by element. The most
  common bug source is a frame mismatch (body vs target-gate vs world)
  on the gate block or a slot-ordering mismatch on obstacles.
- Don't widen tolerances. `1e-5` is the bar; the only legitimate source
  of small numerical drift is float32 round-off in different operation
  orders, and that's already in the budget.

## Commit shape

**Do not run `git commit`.** This dev VM's `.git` is mounted read-only; the
commit will fail with "Read-only file system". Leave the edits uncommitted
in the working tree. Print a final summary to stdout describing:

- Files modified (with line counts).
- The obs-level `max_abs_diff` from step 1 of validation.
- The controller-level `max_abs_diff` from step 2 if a redesign checkpoint
  was found; otherwise note that step 2 was skipped (no ckpt available).
- One paragraph (3-5 sentences) for the human reviewer describing what
  changed and why, in commit-message style. The human will commit from a
  different host.

## What "done" looks like

- `deploy_numpy/obs.py` updated, asserts the right shape `(52,)` on its
  internal validate step (it already calls `_validate_shape(raw,
  (ACTOR_OBS_DIM,), ...)` — that will pick up the new dim automatically).
- `ruff format && ruff check --fix` passes clean on every touched file.
- `python -m py_compile lsy_drone_racing/control/rl_sbx/deploy_numpy/obs.py`
  passes.
- Obs-level parity check (step 1 above) passes with `max_abs_diff < 1e-5`.
- Diff left uncommitted in the working tree with a summary printed.
- No training run launched. No tests under `tests/` written.

If anything in this brief is ambiguous, the answer is "make it match
`rl_song/obs.py` exactly". When in doubt, do the same thing the JAX side
does; the JAX side is the canonical implementation.
