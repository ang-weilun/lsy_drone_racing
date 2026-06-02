# Non-target Gate-Frame Obs Channel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Encode all 4 gates in the actor observation (not just `{target, target+1}`) so the policy can see — and stop clipping — the non-target gate frames that cause ~44% of L3 crashes.

**Architecture:** Additive change to the gate channel. Keep the tuned target (abs corners) and next-gate (delta corners) aiming slots untouched; append two *blind-gate* slots for `target+2` and `target+3` (= just-passed), each as absolute body-frame aperture corners + a `visited` flag, assigned by cyclic offset (permutation-stable, no rank/one-hot). Mirror the edit across the JAX (training) and NumPy (deploy) encoders; the existing `check_obs_encoder_parity.py` is the equivalence gate. Obs dim 52→78.

**Tech stack:** JAX (`rl_song/obs.py`), NumPy/scipy mirror (`rl_sbx/deploy_numpy/obs.py`), pixi `rl-train` env, vast.ai GPU boxes for parity/train/eval (no local Python env on this machine).

**Verification model:** This repo skips the pytest suite (project instruction). The equivalence test for an encoder change is JAX↔NumPy parity + the `ACTOR_OBS_DIM` layout assertion. `scripts/check_obs_encoder_parity.py` already builds both encoders on a fixed fake obs and asserts agreement + finite output + correct dim — that is the red/green gate. It needs the pixi env, so it runs on a bootstrapped box (Phase B step 1), not locally.

---

## Phase A — Encoder change (edit locally, commit, verify on box)

### Task A1: Bump the obs-layout constants (makes the dim assertion the spec)

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/config.py:59` and `:78`

- [ ] **Step 1: Update `ACTOR_OBS_GATE_DIM` 24 → 50**

Replace line 59:
```python
ACTOR_OBS_GATE_DIM: int = 24
```
with:
```python
# 12 (target abs corners) + 12 (next-gate delta corners) + 2 blind-gate slots
# of (12 abs corners + 1 visited) = 50. The blind slots (target+2, target+3 =
# just-passed) encode non-window gate frames for collision avoidance; see
# docs/superpowers/specs/2026-06-02-nontarget-gate-frame-obs-design.md.
ACTOR_OBS_GATE_DIM: int = 50
```

- [ ] **Step 2: Update the layout assertion 52 → 78**

Replace line 78:
```python
assert ACTOR_OBS_DIM == 52 + ACTOR_OBS_ANG_VEL_DIM, "Actor obs layout drift"
```
with:
```python
assert ACTOR_OBS_DIM == 78 + ACTOR_OBS_ANG_VEL_DIM, "Actor obs layout drift"
```

Note: do not commit yet — config alone now expects 78 while the encoders still emit 52. The mismatch is the intended red state, caught by parity in Phase B.

### Task A2: Add the blind-gate slots to the JAX encoder (authoritative)

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/obs.py` (constant near line 45; gate channel ~251–266; module docstring ~14–15)

- [ ] **Step 1: Add the slot-count constant**

After the existing `N_FUTURE_GATES: int = 2` (line 45), add:
```python
# Number of non-window gates encoded purely for collision avoidance: the gates
# outside the {target, target+1} aiming window. For a 4-gate track these are
# target+2 and target+3 (= the just-passed gate). Encoded as absolute body-frame
# aperture corners + a visited flag, NOT as aiming deltas — for avoidance the
# useful quantity is "where is this frame relative to me." Cyclic offset makes
# the slots permutation-stable (no rank-flip), so no identity one-hot is needed.
N_EXTRA_GATE_SLOTS: int = 2
```

- [ ] **Step 2: Replace the gate-channel assembly with the all-4-gates version**

Replace this block (currently ~lines 259–266):
```python
    g_next_pos = gates_pos[gate_indices[1]]
    g_next_quat = gates_quat[gate_indices[1]]
    g_next_corners_w = _gate_corners_world(g_next_pos, g_next_quat)
    inter_gate_delta_body = (g_next_corners_w - g_target_corners_w) @ rot_bw.T

    gate_chan = jnp.concatenate(
        [target_corners_body.reshape(-1), inter_gate_delta_body.reshape(-1)]
    )
```
with:
```python
    g_next_pos = gates_pos[gate_indices[1]]
    g_next_quat = gates_quat[gate_indices[1]]
    g_next_corners_w = _gate_corners_world(g_next_pos, g_next_quat)
    inter_gate_delta_body = (g_next_corners_w - g_target_corners_w) @ rot_bw.T

    # Blind-gate collision channel: gates outside the {target, target+1} window
    # (target+2, target+3 = just-passed). Absolute body-frame corners + visited.
    gates_visited = env_obs["gates_visited"]
    extra_indices = (
        target_idx + jnp.arange(N_FUTURE_GATES, N_FUTURE_GATES + N_EXTRA_GATE_SLOTS)
    ) % n_gates

    def _abs_body_corners(idx: Array) -> Array:
        corners_w = _gate_corners_world(gates_pos[idx], gates_quat[idx])
        return ((corners_w - pos) @ rot_bw.T).reshape(-1)

    extra_corners = jax.vmap(_abs_body_corners)(extra_indices)  # (N_EXTRA, 12)
    extra_visited = gates_visited[extra_indices].astype(jnp.float32)[:, None]  # (N_EXTRA, 1)
    extra_chan = jnp.concatenate([extra_corners, extra_visited], axis=-1).reshape(-1)

    gate_chan = jnp.concatenate(
        [target_corners_body.reshape(-1), inter_gate_delta_body.reshape(-1), extra_chan]
    )
```

- [ ] **Step 3: Update the module docstring gate line**

In the `Layout` block (lines ~14–15), replace:
```python
* gates (24): target gate corners in body frame (12), then the next-gate
  minus target-gate corner deltas in body frame (12).
```
with:
```python
* gates (50): target gate corners in body frame (12), the next-gate minus
  target-gate corner deltas in body frame (12), then the two non-window gates
  (target+2, target+3 = just-passed) as absolute body-frame corners (12) plus a
  visited flag (1) each — the collision-avoidance channel for frames the aiming
  window does not cover.
```

### Task A3: Mirror the slots in the NumPy deploy encoder

**Files:**
- Modify: `lsy_drone_racing/control/rl_sbx/deploy_numpy/obs.py` (constant near line 17; gate channel ~104–108)

- [ ] **Step 1: Read the new constant**

After line 17 (`N_FUTURE_GATES = ...`), add:
```python
# Number of non-window gates encoded for collision avoidance (mirror of
# rl_song.obs.N_EXTRA_GATE_SLOTS).
N_EXTRA_GATE_SLOTS: int = int(read_rl_song_obs_constant("N_EXTRA_GATE_SLOTS"))
```

- [ ] **Step 2: Replace the gate-channel assembly**

Replace lines ~104–108:
```python
    g_next_pos = gates_pos[gate_indices[1]]
    g_next_quat = gates_quat[gate_indices[1]]
    g_next_corners_w = _gate_corners_world(g_next_pos, g_next_quat, corners_local)
    inter_gate_delta_body = (g_next_corners_w - g_target_corners_w) @ rot_bw.T
    gate_chan = np.concatenate([target_corners_body.reshape(-1), inter_gate_delta_body.reshape(-1)])
```
with:
```python
    g_next_pos = gates_pos[gate_indices[1]]
    g_next_quat = gates_quat[gate_indices[1]]
    g_next_corners_w = _gate_corners_world(g_next_pos, g_next_quat, corners_local)
    inter_gate_delta_body = (g_next_corners_w - g_target_corners_w) @ rot_bw.T

    # Blind-gate collision channel (mirror of rl_song.obs): gates outside the
    # {target, target+1} window, as absolute body-frame corners + visited.
    gates_visited = np.asarray(env_obs["gates_visited"])
    extra_indices = (
        target_idx + np.arange(N_FUTURE_GATES, N_FUTURE_GATES + N_EXTRA_GATE_SLOTS, dtype=np.int64)
    ) % n_gates
    extra_parts = []
    for idx in extra_indices:
        corners_w = _gate_corners_world(gates_pos[idx], gates_quat[idx], corners_local)
        corners_body = (corners_w - pos) @ rot_bw.T
        extra_parts.append(
            np.concatenate([corners_body.reshape(-1), [np.float32(gates_visited[idx])]])
        )
    extra_chan = np.concatenate(extra_parts).astype(np.float32)
    gate_chan = np.concatenate(
        [target_corners_body.reshape(-1), inter_gate_delta_body.reshape(-1), extra_chan]
    )
```

### Task A4: Commit the encoder change

- [ ] **Step 1: Sanity-check the edits read cleanly (visual diff)**

Run: `git diff --stat lsy_drone_racing/control/rl_song/obs.py lsy_drone_racing/control/rl_sbx/deploy_numpy/obs.py lsy_drone_racing/control/rl_song/config.py`
Expected: three files modified.

- [ ] **Step 2: Commit (no lint locally — no Python env on this box; ruff runs on the box in Phase B step 1)**

```bash
git add lsy_drone_racing/control/rl_song/obs.py \
        lsy_drone_racing/control/rl_sbx/deploy_numpy/obs.py \
        lsy_drone_racing/control/rl_song/config.py
git -c commit.gpgsign=false commit -m "feat(obs): encode all 4 gates — blind-gate collision channel (52->78)"
git push origin rl/obs-completion-capacity-2026-06-02
```

**Out of scope (note, do not edit):** `rl_sbx/controller_ablate.py` (separate ablation encoder, not used by `box_launch_l2_screen.sh`) and `rl_sbx/controller_diag.py` (offscreen overlay, non-flight) carry their own gate-channel code and will still emit the old 52-d layout. They are NOT exercised by this train/eval run. If an ablation or diag render is later needed on the new obs, apply the same additive edit there.

---

## Phase B — Verify, train, validate (vast box; long-running orchestration)

These steps are orchestration, not 2–5 min code edits. Run from this machine driving a box over SSH (see `/home/exedev/scripts/vast_*.sh`). Headless eval needs `MUJOCO_GL=osmesa` + `apt install libosmesa6 libgl1 libegl1 libgles2 libglew-dev` + `--render=False` (see memory `l3-crash-diagnosis-nontarget-frames`).

### Task B1: Rent + bootstrap a box, lint, run the parity gate

- [ ] **Step 1: Rent + bootstrap** (clones the pushed branch with the new encoders)

```bash
bash /home/exedev/scripts/vast_create_instance.sh drone-ntframe   # prints ssh_host/ssh_port
bash /home/exedev/scripts/vast_bootstrap.sh <ssh_host> <ssh_port> rl/obs-completion-capacity-2026-06-02
```

- [ ] **Step 2: Lint on the box** (the local commit was unlinted)

On box (`cd ~/lsy_drone_racing`):
```bash
export PATH=$HOME/.pixi/bin:$PATH
pixi run -e rl-train ruff format lsy_drone_racing/control/rl_song/obs.py \
  lsy_drone_racing/control/rl_sbx/deploy_numpy/obs.py lsy_drone_racing/control/rl_song/config.py \
  scripts/aggregate_crash_causes.py
pixi run -e rl-train ruff check --fix lsy_drone_racing/control/rl_song/obs.py \
  lsy_drone_racing/control/rl_sbx/deploy_numpy/obs.py scripts/aggregate_crash_causes.py
```
If ruff reformats, rsync the box copy back or re-apply locally and amend the commit. Also `git add scripts/aggregate_crash_causes.py` (the frame-edge crash classifier from the diagnostic) and commit it now that it is lint-clean.

- [ ] **Step 3: Parity gate — RL_OBS_ANG_VEL=0 then =1**

```bash
RL_OBS_ANG_VEL=0 pixi run -e rl-train python scripts/check_obs_encoder_parity.py
RL_OBS_ANG_VEL=1 pixi run -e rl-train python scripts/check_obs_encoder_parity.py
```
Expected: both PASS (JAX≈NumPy, dim 78 / 81, finite). A failure here is the red→green signal — if it reports a shape or mismatch, the two encoders diverged; fix and re-run before training. **Do not start training until both pass.**

### Task B2: Cold-train 2×512 on L2 (new obs)

- [ ] **Step 1: Launch the L2 cold-start (512-wide, ang_vel off)**

```bash
bash scripts/box_launch_l2_screen.sh l2_ntframe_cap512 0 512
```
This runs the minimal Song recipe (prog+omega+crash+finish, alpha_max=0.36) for 300M steps in tmux. The pre-flight parity check inside the launcher must pass (it will, given B1 step 3).

- [ ] **Step 2: Monitor to a stable L2 success rate**

Watch `training_logs/l2_ntframe_cap512.log` for `ep_rew_mean` / gates-passed to plateau (L2 should reach high SR). Confirm it learns (non-trivial finish rate) before committing to the L3 warm-start.

### Task B3: Warm-start onto L3 with the SOTA recipe

- [ ] **Step 1: Warm-start L3 from the L2 checkpoint**

Use the L3 SOTA recipe matched to `relBobs03` (alpha≈1.10, obstacle_weight=0.30, obstacle barrier on) so the comparison is apples-to-apples, warm-starting the actor from L2:
```bash
RL_HIDDEN_SIZE=512 RL_OBS_ANG_VEL=0 pixi run -e rl-train python -m lsy_drone_racing.control.rl_sbx.train \
  --run-name=l3_ntframe_warm_cap512 \
  --init-from=lsy_drone_racing/control/rl_sbx/checkpoints/l2_ntframe_cap512 \
  --curriculum=<L3 stage matching relB campaign> \
  --alpha-max-rad=1.10 --obstacle-weight=0.30 --total-timesteps=<match relB, e.g. 1B>
```
(Resolve `--curriculum` and total-timesteps against the relB launcher / `l3-benchmarks` memory before running; the point is to reproduce the SOTA training conditions with only the obs changed.)

- [ ] **Step 2: Monitor + seed-matched eval**

After training, seed-matched SR/lap:
```bash
bash scripts/box_eval_speed.sh l3_ntframe_warm_cap512 1 100
```
Expect SR in the neighborhood of relBobs03 (~75–81%) or better. A large SR regression means the obs change hurt — investigate before declaring success.

### Task B4: Re-run the crash diagnostic (the actual success criterion)

- [ ] **Step 1: Eval the best L3 checkpoint with trace dump**

```bash
CK=<best step_* dir under l3_ntframe_warm_cap512>
MUJOCO_GL=osmesa pixi run -e rl-train python -m lsy_drone_racing.control.rl_song.eval_sim \
  --config level3.toml --controller rl_sbx/controller_numpy.py --control_mode attitude \
  --render=False --checkpoint $CK --n_runs 400 --dump_trace traces/l3_ntframe
```

- [ ] **Step 2: Aggregate the validated crash breakdown**

```bash
pixi run -e rl-train python scripts/analyze_eval_traces.py traces/l3_ntframe
pixi run -e rl-train python scripts/aggregate_crash_causes.py traces/l3_ntframe
```

- [ ] **Step 3: Compare to the baseline**

Baseline (`relBobs03`): non-target gate-frame ≈44% of crashes; confirmed strikes 26 (21 non-target). **Success = the non-target gate-frame share drops materially** (and overall SR holds or improves). Pull `agg.json` back to `renders/l3_ntframe_crashdiag/` for the record.

- [ ] **Step 4: Tear down the box**

```bash
yes | vastai destroy instance <instance_id>
```

---

## Self-review

- **Spec coverage:** all-4-gates encoding (A2/A3), cyclic blind slots target+2/+3 (A2 step 2), abs body corners + visited (A2/A3), corners-not-center+normal (A2 reuses `_gate_corners_world`), reward untouched (no reward task), parity gate (B1.3), cold-start train L2→L3 (B2/B3), crash-diagnostic validation (B4). Covered.
- **Placeholders:** the only deferred values are the L3 `--curriculum` / `--total-timesteps` in B3 — intentionally resolved against the relB launcher at run time (the recipe must match the SOTA baseline, which lives in campaign config, not this plan). Flagged inline, not a silent TODO.
- **Type/name consistency:** `N_EXTRA_GATE_SLOTS` (obs.py) read by NumPy via `read_rl_song_obs_constant("N_EXTRA_GATE_SLOTS")`; `extra_indices` / `extra_chan` names consistent across A2/A3; `gates_visited` confirmed present in env obs (`race_core.py:694`). `ACTOR_OBS_GATE_DIM=50` ⇒ `ACTOR_OBS_DIM=12+50+16=78`, matching the A1 assertion.
