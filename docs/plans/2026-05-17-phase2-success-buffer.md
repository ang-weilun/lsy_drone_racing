# Phase 2 successful-state buffer Implementation Plan

**Goal:** Add Song 2023 §III-B Phase 2 — a per-gate stratified buffer of
successful gate-pass states that selected envs are re-spawned to on reset,
so the policy practices late-gate approaches without first having to survive
early gates.

**Architecture:** Per-gate ring buffer lives outside the scan (threaded through
`train.py` like `is_seg_init`). Inside the scan, gate-pass events are
recorded as outputs; the buffer is updated *once* after the scan via
masked-scatter (cumsum-rank pattern, codex). Replay at reset samples one
candidate per env and overrides the post-`_reset_env_data` state. States
are stored in the **previous gate's local frame** (pos offset rotated by
gate yaw, quat as relative orientation) so the buffer remains valid under
level-3 track randomization. A source enum `{0=true_start, 1=phase1_seg,
2=phase2_replay}` replaces the existing `is_seg_init` bool and drives
per-source `finish_rate_*` metrics. Phase 2 is gated by a `phase2_warmup_steps`
threshold (two-phase step: 0 before, `phase2_prob` after) — when the
threshold is crossed, `train.py` rebuilds the `RolloutStaticConfig` once,
triggering exactly one JIT retrace.

**Tech Stack:** JAX (pure functional, jit-traced), Orbax for checkpointing,
existing `RLSongVecEnv` + `scan_rollout` infrastructure.

**Out of scope:** No new pytest. Per CLAUDE.md, this RL track validates via
short training runs (5-10M steps, monitor crash-free + metric sanity) and
full-length training (300M, eval = 8-ep patched sim).

---

## Design summary (read before starting)

### Buffer layout

```python
class Phase2Buffer(NamedTuple):
    data:  Array  # (n_gates, per_gate_capacity, STATE_DIM)
    ptr:   Array  # (n_gates,) int32 — write head per gate slot
    fill:  Array  # (n_gates,) int32 — count of valid entries, clipped to capacity
```

- Stratification index = `new_target_gate` after gate-pass (i.e. the gate the
  drone is now heading toward). Slot 0 is **unused** (a drone "approaching
  gate 0" is what every true-start already does). We only ever write to
  slots 1..n_gates−1; slot `n_gates` is impossible because the drone
  finishes the lap and we deliberately don't store finished states (codex).
- `per_gate_capacity = 4096` (configurable). At n_gates=4 this is 12k usable
  entries — fits in carry without recompilation overhead.

### Stored state (per entry)

All quantities below are in the **previous-gate frame** (frame of the gate
the drone just passed). Index is `new_target_gate ∈ {1..n_gates-1}`.

| Field | Shape | Frame | Notes |
|---|---|---|---|
| `pos_local`         | (3,)  | prev-gate | `R(q_prev)^T @ (pos_world − pos_prev_gate)` |
| `vel_local`         | (3,)  | prev-gate | `R(q_prev)^T @ vel_world` |
| `quat_local`        | (4,)  | prev-gate | xyzw, `q_prev^{-1} * q_world` |
| `ang_vel`           | (3,)  | body      | unchanged; body-frame is layout-independent |
| `prev_action`       | (4,)  | env       | env-action 4-vec at the step before gate-pass |
| `obstacles_visited` | (n_obs,) bool | — | preserved; gates_visited is reconstructable from target |

`STATE_DIM = 3 + 3 + 4 + 3 + 4 + n_obs`.  At `n_obs=4`: 21.

`target_gate` is **implicit in the slot index** — we don't need to store it.
`gates_visited` is reconstructable: `gates_visited[i] = (i < target_gate)`.

### Replay reconstruction

Given stored entry from slot `g` (i.e. `target_gate = g`):

```
prev_gate_pos  = current_layout.gates_pos[g - 1]    # post-randomization, post-wobble
prev_gate_quat = current_layout.gates_quat[g - 1]
R_prev         = quat_to_rotmat(prev_gate_quat)

pos_world  = prev_gate_pos + R_prev @ pos_local
vel_world  = R_prev @ vel_local
quat_world = quat_multiply(prev_gate_quat, quat_local)
# ang_vel, prev_action unchanged
target_gate    = g
gates_visited  = [i < g for i in range(n_gates)]
last_drone_pos = pos_world      # codex: fix the existing Phase-1 bug too
takeoff_pos    = pos_world
```

### Source enum

Replace `is_seg_init: bool[n_envs]` with `source: int8[n_envs]`:
- `0 = SRC_TRUE_START` (toml ground spawn)
- `1 = SRC_PHASE1_SEG` (Song Phase 1 midpoint, with or without velocity)
- `2 = SRC_PHASE2_REPLAY` (sampled from buffer)

Per-source metrics:
- `<src>_completed_count`, `<src>_finished_count` summed in `RolloutMetricSums`
- Logged as `finish_rate_true_start`, `finish_rate_phase1_seg`, `finish_rate_phase2_replay`
- Plus `phase2_buffer_fill[g]` per-gate histogram

### Write path (filters per codex)

We write a slot iff **all** hold:
1. `gate_just_passed` is True this step
2. `current_target ∈ [1, n_gates-1]` (not finished, not slot 0)
3. `~done_bool` (no crash/truncate on the gate-pass step)
4. `source != SRC_PHASE2_REPLAY` (avoid the buffer feeding itself)

For each step, build `valid: bool[n_envs]`, `event_data: float[n_envs, STATE_DIM]`,
`event_slot: int[n_envs]` (= `current_target`). Stack across `n_steps`,
then apply **one** masked-scatter per slot after the scan returns. The
in-scan body emits these as additional `RolloutScanOutputs`.

```python
# After scan, per slot g in 1..n_gates-1:
mask_g     = valid_flat & (event_slot_flat == g)
rank       = jnp.cumsum(mask_g.astype(jnp.int32)) - 1
idx        = (buffer.ptr[g] + rank) % per_gate_capacity
idx_masked = jnp.where(mask_g, idx, per_gate_capacity)  # OOB -> drop
new_data   = buffer.data[g].at[idx_masked].set(event_data_flat, mode="drop")
n_added    = jnp.sum(mask_g.astype(jnp.int32))
new_ptr    = (buffer.ptr[g] + n_added) % per_gate_capacity
new_fill   = jnp.minimum(per_gate_capacity, buffer.fill[g] + n_added)
```

### Read path (replay at reset)

For each env being reset, the rollout's reset path already partitions
into `do_seg` (Phase 1). Extend to a three-way categorical draw:

```python
# Inside _reset_done_worlds, replace the bernoulli for do_seg with:
u = jax.random.uniform(cat_key, shape=(n_envs,))
do_phase2 = mask & (u < p_phase2_effective)                            # 0..p_p2
do_seg    = mask & (u >= p_phase2_effective) & (u < p_p1 + p_phase2)   # p_p2..p_p2+p_p1
# everything else is true-start (no override)

# Apply Phase 2 first (it sets pos/vel/quat/target),
# then Phase 1 only where ~do_phase2 (mutually exclusive).
```

`p_phase2_effective = phase2_prob if global_step >= phase2_warmup_steps else 0.0`.
Buffer also gates with `fill[g] > 0` per sampled gate; if the chosen slot
is empty we fall back to Phase 1 (or true-start, if Phase 1 also disabled).

### Schedule: two-phase step

`phase2_warmup_steps` lives in `CurriculumStage`. Pre-warmup, `phase2_prob`
is effectively 0 and the scan is traced with `phase2_prob=0.0` in static
config. At warmup crossing, `train.py` rebuilds the `RolloutStaticConfig`
with `phase2_prob=stage.phase2_prob` — one JIT retrace, no per-iteration
overhead.

### Phase 1 aux-field bug (codex)

`_apply_segment_init` currently overrides `pos/vel/quat/target_gate` but
does not refresh `data.last_drone_pos`, `data.takeoff_pos`, `data.gates_visited`,
`data.obstacles_visited`. After re-spawn these are stale w.r.t. the new
position. Fix as part of this work — Phase 2 reconstruction needs the same
helper, so factor it out:

```python
def _refresh_aux_fields_after_respawn(
    env_data: EnvData,
    mask: Array,
    new_pos: Array,           # (n_envs, 3)
    new_target_gate: Array,   # (n_envs,)
) -> EnvData:
    """Recompute aux fields after a seg-init or phase-2 respawn."""
```

Apply in both `_apply_segment_init` and the new `_apply_phase2_replay`.

---

## Tasks

Each task is a single coherent change. Commit between tasks. Within a task,
the substeps are individually verifiable (parse + ruff format + ruff check
on the touched files, plus a 100k-step training smoke-test on the last
task of each major chunk).

### Task A1: Source enum replaces is_seg_init bool

**Why first:** sets up the per-source metric machinery so subsequent Phase 2
work can plug into existing logging. Pure refactor — no behavior change for
existing stages.

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/rollout.py`
- Modify: `lsy_drone_racing/control/rl_song/train.py`

**Steps:**

1. In `rollout.py` near module constants, define:
   ```python
   SRC_TRUE_START: int = 0
   SRC_PHASE1_SEG: int = 1
   SRC_PHASE2_REPLAY: int = 2
   SOURCE_DTYPE = jnp.int8
   ```
2. In `_ScanCarry`, rename field `is_seg_init: Array` → `source: Array`
   (dtype int8). Update the field's comment.
3. In `RolloutMetricSums`, replace `true_start_completed_count` /
   `true_start_finished_count` with three pairs:
   `(true_start|phase1_seg|phase2_replay)_(completed|finished)_count`
   (six scalars total).
4. In the scan body, the "true-start tally" block becomes a per-source tally:
   ```python
   for src_const, prefix in [(SRC_TRUE_START, "true_start"),
                             (SRC_PHASE1_SEG, "phase1_seg"),
                             (SRC_PHASE2_REPLAY, "phase2_replay")]:
       src_done = done_bool & (carry.source == src_const)
       <prefix>_completed = carry.<prefix>_completed_count + jnp.sum(src_done.astype(jnp.float32))
       <prefix>_finished  = carry.<prefix>_finished_count  + jnp.sum((src_done & finished).astype(jnp.float32))
   ```
   (Unroll the loop manually since carry fields are static names — keep readable.)
5. `_reset_done_worlds` / `_apply_reset_perturbation`: return `new_source`
   instead of `do_seg`. Compute `new_source = jnp.where(do_seg,
   SRC_PHASE1_SEG, SRC_TRUE_START)` for now (no Phase 2 yet).
6. Scan body: `next_source = jnp.where(done_bool, new_source, carry.source)`.
7. `RolloutScanResult`: rename `is_seg_init` → `source`.
8. `_validate_scan_inputs`: rename param, accept dtype int8.
9. `train.py`: rename `is_seg_init` → `source`, initialize as
   `jnp.zeros((n_envs,), dtype=jnp.int8)`, reset on stage promotion.
10. `_rollout_metrics`: emit three finish_rate keys instead of one.
    `_log_iteration`: log all three under `rollout/finish_rate_<src>`.

**Verify:**
- `python -c "import ast; ast.parse(open('.../rollout.py').read())"` parses
- `ruff format` / `ruff check` clean on touched files (modulo pre-existing
  D416 / I001 across the file)
- 100k-step smoke run completes without crash; wandb shows
  `finish_rate_true_start ≈ existing finish_rate_true_start`,
  `finish_rate_phase1_seg ∈ [0, 1]`, `finish_rate_phase2_replay = 0/0 = 0`.

**Commit:** `refactor(rl_song): replace is_seg_init bool with source enum`

---

### Task A2: Aux-field recomputation helper + Phase 1 fix

**Why second:** Phase 2's replay needs to refresh aux fields after a state
override, and Phase 1 has the same latent bug today. Fixing Phase 1 first
keeps the helper in use even before Phase 2 lands, and it surfaces any
regressions early.

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/rollout.py`
- Modify: `lsy_drone_racing/control/rl_song/env_wrapper.py` (eager `_apply_segment_init` — same fix)

**Steps:**

1. Add `_refresh_aux_fields_after_respawn(env_data, mask, new_pos,
   new_target_gate) -> EnvData` in `rollout.py`. Replaces:
   - `data.last_drone_pos[mask] = new_pos`
   - `data.takeoff_pos[mask] = new_pos`
   - `data.gates_visited[mask, i] = (i < new_target_gate)` for each i
   - `data.obstacles_visited[mask, :] = True` (defensible default — drone
     post-respawn is treated as having "seen" all obstacles to avoid
     spurious sensing-bonus rewards on a fresh respawn)
2. Call from `_apply_segment_init` just before the function returns. The
   per-env `mask` is `do_seg`; `new_pos = new_pos`; `new_target_gate =
   segment_idx`.
3. Mirror the call in `env_wrapper._apply_segment_init` (eager path).
4. Confirm `EnvData` has these fields. If `obstacles_visited` isn't carried
   in `EnvData` we drop step 1's fourth line.

**Verify:**
- 100k smoke run completes
- `rollout/r_guid` / other position-derived rewards behave normally
- No crash from invalid aux fields downstream of `race_core.check_gate_pass`

**Commit:** `fix(rl_song): refresh aux fields after seg-init respawn`

---

### Task B1: Phase2Buffer data structure threaded through pipeline

**Why:** All the plumbing, no behavior. Buffer is initialized to zeros and
threaded through `scan_rollout` and `train.py` like `source`.

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/rollout.py`
- Modify: `lsy_drone_racing/control/rl_song/config.py`
- Modify: `lsy_drone_racing/control/rl_song/train.py`

**Steps:**

1. In `config.py` add to `RolloutStaticConfig`:
   ```python
   phase2_capacity_per_gate: int = 4096
   phase2_prob: float = 0.0
   ```
   And to `CurriculumStage`:
   ```python
   phase2_prob: float = 0.0
   phase2_capacity_per_gate: int = 4096
   phase2_warmup_steps: int = 0
   ```
2. In `rollout.py` define:
   ```python
   PHASE2_STATE_DIM = 3 + 3 + 4 + 3 + 4 + N_OBSTACLES  # placeholder; resolve at runtime

   class Phase2Buffer(NamedTuple):
       data: Array  # (n_gates, capacity, STATE_DIM)
       ptr:  Array  # (n_gates,)
       fill: Array  # (n_gates,)
   ```
   `STATE_DIM` is data-dependent (n_obstacles). Build it from
   `env_data.obstacles_pos.shape[1]` at construction time in train.py and
   pass the buffer in as carry data; the rollout consumes it via dynamic
   shapes (still static-shape inside JIT as long as buffer is built once).
3. Add `phase2_buffer: Phase2Buffer` field to `_ScanCarry` and
   `RolloutScanResult`. Plumb through (no read/write yet — buffer just
   round-trips).
4. `train.py`: construct an empty buffer once at startup:
   ```python
   def _empty_phase2_buffer(n_gates, capacity, state_dim) -> Phase2Buffer:
       return Phase2Buffer(
           data=jnp.zeros((n_gates, capacity, state_dim), dtype=jnp.float32),
           ptr=jnp.zeros((n_gates,), dtype=jnp.int32),
           fill=jnp.zeros((n_gates,), dtype=jnp.int32),
       )
   ```
   Thread through `_collect_rollout`. Reset on stage promotion (each stage's
   buffer is independent).
5. `_rollout_static_config`: read `phase2_prob`, `phase2_capacity_per_gate`
   from current stage; emit into `RolloutStaticConfig`. Override `phase2_prob`
   to 0.0 if `global_step < stage.phase2_warmup_steps`.
6. `train.py` main loop: check whether `phase2_prob` in current static_cfg
   should change (warm-up threshold crossed). If yes, force re-collection
   path (next `_collect_rollout` will build a new static_cfg, triggering
   JIT retrace once).

**Verify:**
- 100k smoke completes
- `static_cfg.phase2_prob == 0.0` in current stages → no behavior change
- `RolloutScanResult.phase2_buffer.fill == [0, 0, 0, 0]` (round-trip clean)

**Commit:** `feat(rl_song): add Phase2Buffer plumbing (no read/write yet)`

---

### Task B2: Write path — collect gate-pass events + post-scan scatter

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/rollout.py`

**Steps:**

1. Add to `RolloutScanOutputs`:
   ```python
   p2_event_valid: Array     # (n_steps, n_envs) bool
   p2_event_slot:  Array     # (n_steps, n_envs) int32 — new_target_gate
   p2_event_data:  Array     # (n_steps, n_envs, STATE_DIM)
   ```
2. In the scan body, after computing `gate_just_passed` and the post-step
   state, build the gate-frame event tuple:
   ```python
   prev_gate_idx = jnp.clip(current_target - 1, 0, n_gates - 1)
   prev_gate_pos = stepped_data.gates_pos[env_arange, prev_gate_idx]
   prev_gate_quat = stepped_data.gates_quat[env_arange, prev_gate_idx]
   R_prev_T = _quat_to_rotmat_T(prev_gate_quat)  # transpose of R

   drone_pos = stepped_data.sim_data.states.pos[:, SINGLE_DRONE_INDEX]
   drone_vel = stepped_data.sim_data.states.vel[:, SINGLE_DRONE_INDEX]
   drone_quat = stepped_data.sim_data.states.quat[:, SINGLE_DRONE_INDEX]
   drone_ang_vel = stepped_data.sim_data.states.ang_vel[:, SINGLE_DRONE_INDEX]

   pos_local = jnp.einsum("nij,nj->ni", R_prev_T, drone_pos - prev_gate_pos)
   vel_local = jnp.einsum("nij,nj->ni", R_prev_T, drone_vel)
   quat_local = _quat_multiply_xyzw(_quat_conjugate(prev_gate_quat), drone_quat)

   event_data = jnp.concatenate([
       pos_local, vel_local, quat_local, drone_ang_vel,
       env_action, stepped_obstacles_visited,
   ], axis=-1)
   event_valid = (
       gate_just_passed
       & (current_target >= 1)
       & (current_target < n_gates)
       & ~done_bool
       & (carry.source != SRC_PHASE2_REPLAY)
   )
   event_slot = current_target.astype(jnp.int32)
   ```
3. After the scan, fold gate-pass events into the buffer once per slot:
   ```python
   def _apply_phase2_writes(buffer, outputs):
       valid_flat = outputs.p2_event_valid.reshape(-1)
       slot_flat  = outputs.p2_event_slot.reshape(-1)
       data_flat  = outputs.p2_event_data.reshape(-1, STATE_DIM)
       new_buffer = buffer
       for g in range(1, n_gates):                  # n_gates is python int
           mask_g = valid_flat & (slot_flat == g)
           rank   = jnp.cumsum(mask_g.astype(jnp.int32)) - 1
           idx    = (new_buffer.ptr[g] + rank) % capacity
           idx    = jnp.where(mask_g, idx, capacity)
           new_data_g = new_buffer.data[g].at[idx].set(data_flat, mode="drop")
           n_added    = jnp.sum(mask_g.astype(jnp.int32))
           new_ptr    = (new_buffer.ptr[g] + n_added) % capacity
           new_fill   = jnp.minimum(capacity, new_buffer.fill[g] + n_added)
           new_buffer = new_buffer._replace(
               data=new_buffer.data.at[g].set(new_data_g),
               ptr=new_buffer.ptr.at[g].set(new_ptr),
               fill=new_buffer.fill.at[g].set(new_fill),
           )
       return new_buffer
   ```
   Call from the top-level `scan_rollout` body after the scan returns, before
   building `RolloutScanResult`.
4. Add `_quat_to_rotmat_T` and `_quat_conjugate` helpers in pure JAX (xyzw
   convention to match the existing `_quat_multiply_xyzw`).

**Verify:**
- 200k smoke completes
- `RolloutScanResult.phase2_buffer.fill` is monotone non-decreasing across
  iterations, eventually saturates at `capacity` for early-gate slots and
  smaller for late-gate slots
- per-gate fill histogram logged to wandb (`phase2_buffer_fill_g{1,2,3}`)

**Commit:** `feat(rl_song): Phase 2 buffer write path (gate-frame state, post-scan scatter)`

---

### Task B3: Read path — sample-and-replay at reset

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/rollout.py`

**Steps:**

1. Add `_apply_phase2_replay(env_data, mask, rng_key, buffer, static_cfg)
   -> (EnvData, rng_key, do_phase2: Array)`:
   ```python
   # For each env in mask, sample a slot g ~ Uniform({g : fill[g] > 0}).
   # Then sample an entry idx ~ Uniform({0..fill[g]-1}).
   non_empty = buffer.fill > 0
   # Fallback: if no slot is non-empty (early in training), do_phase2 = False.
   any_non_empty = jnp.any(non_empty)
   slot_logits = jnp.where(non_empty, 0.0, -jnp.inf)
   g = jax.random.categorical(slot_key, slot_logits, shape=(n_envs,))
   entry_idx = jax.random.randint(idx_key, shape=(n_envs,), minval=0, maxval=jnp.maximum(buffer.fill[g], 1))
   entry = buffer.data[g, entry_idx]   # (n_envs, STATE_DIM)
   # Unpack
   pos_local, vel_local, quat_local, ang_vel, prev_action, obs_visited = _unpack(entry)
   # Reconstruct world-frame using *current layout's* prev gate
   prev_gate_pos = env_data.gates_pos[env_arange, g - 1]
   prev_gate_quat = env_data.gates_quat[env_arange, g - 1]
   R_prev = _quat_to_rotmat(prev_gate_quat)
   pos_world = prev_gate_pos + jnp.einsum("nij,nj->ni", R_prev, pos_local)
   vel_world = jnp.einsum("nij,nj->ni", R_prev, vel_local)
   quat_world = _quat_multiply_xyzw(prev_gate_quat, quat_local)
   ```
2. Apply via `do_phase2 = mask & bernoulli(p_phase2_effective) & any_non_empty`.
   Override `env_data.sim_data.states.{pos, vel, quat, ang_vel}` and
   `env_data.target_gate = g` for envs in `do_phase2`.
3. Call `_refresh_aux_fields_after_respawn` with `new_pos=pos_world`,
   `new_target_gate=g`, `mask=do_phase2`.
4. Update `_reset_done_worlds` to call both `_apply_phase2_replay` (first)
   and `_apply_segment_init` (second, on `mask & ~do_phase2`), returning
   `new_source = jnp.select([do_phase2, do_seg],
                            [SRC_PHASE2_REPLAY, SRC_PHASE1_SEG],
                            default=SRC_TRUE_START)`.
5. Restore `prev_action_env_4vec` from the stored entry for the replayed
   envs:
   ```python
   prev_action_env_4vec = jnp.where(do_phase2[:, None], prev_action, prev_action_env_4vec)
   ```
   (Plumb the buffer-derived prev_action out to the scan carry.)

**Verify:**
- 5M smoke run with `phase2_prob=0.3` and `phase2_warmup_steps=500k`:
  - First 500k: no replays (`finish_rate_phase2_replay = 0/0`)
  - 500k–5M: replays happen, `phase2_replay_episodes > 0`,
    `finish_rate_phase2_replay` is non-trivial
  - No NaN reward / drone-flew-into-wall-spam in early replay iterations
- Render a 5-episode sim eval to eyeball that replayed-starts look sensible
  (drone appears mid-air near a gate, not at the floor)

**Commit:** `feat(rl_song): Phase 2 buffer read path (sample-and-replay at reset)`

---

### Task C1: v30 curriculum + launch

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/config.py`

**Steps:**

1. Add stage `level3_v30_phase2` (warm-start from v26 checkpoint):
   ```python
   CurriculumStage(
       level=3,
       reset_pos_perturb_m=0.0,
       reset_vel_perturb_mps=0.0,
       reset_yaw_perturb_rad=0.0,
       gate_rand_scale=1.0,
       segment_init_prob=0.30,        # p_phase1
       segment_init_perturb_m=0.10,
       segment_init_vel_mps=2.5,
       phase2_prob=0.30,              # target — gated by warmup
       phase2_capacity_per_gate=4096,
       phase2_warmup_steps=50_000_000,  # 50M warmup to fill the buffer
       # (true_start probability = 1 - 0.3 - 0.3 = 0.4)
       ...other v26-compatible fields...
   )
   ```
2. Update `docs/handoffs/2026-05-17-rl-v22-v29-and-phase2-design.md`'s
   "Order of work" §4 to point at this curriculum.
3. Launch command (run on RTX 5090 box per CLAUDE.md instructions):
   ```bash
   nohup pixi run -e rl-train train-rl-song \
       --total-timesteps 300000000 \
       --init-from /home/ubuntu/.../level3_warmstart_seed0_v26_v25cont_300M \
       --run-name level3_v30_phase2_seed0_300M \
       > training_logs/v30_phase2.log 2>&1 &
   ```
4. After ~50M (post-warmup), eyeball wandb for:
   - `phase2_buffer_fill_g{1,2,3}` non-zero
   - `finish_rate_phase2_replay` rising
   - `finish_rate_true_start` not collapsing (v29 failure mode)

**Commit:** `train(rl_song): v30 warm-start v26 + Phase 2 (warmup 50M, p=0.3)`

---

## Risk register

| Risk | Mitigation |
|---|---|
| JIT retrace on warmup crossing > 30s pause | One-time cost; document in run log. Could avoid by always tracing with phase2_prob > 0 and using `jnp.where(global_step < warmup, 0.0, phase2_prob)` inside the bernoulli — needs `global_step` traced. Defer unless retrace is > 60 s. |
| Gate-frame transform error → drone spawns inside a wall on replay | Validate on a unit test in a notebook (not committed): pick a known event, write it, read it back from a perturbed layout, confirm pos within bounds. If we see crashes spike right at warmup crossing, this is the cause. |
| Buffer feedback loop (Phase 2 episodes feed Phase 2 buffer) | Filter `source != SRC_PHASE2_REPLAY` at write time — already in plan. |
| Buffer dominated by gate 0 → gate 1 events, late-gate slots starve | Uniform sample over `fill > 0` slots — already in plan. Log per-gate fill so we can verify. |
| `obstacles_visited` mismatch causes spurious sensor-bonus reward | Default-True at respawn — already in plan. |
| Warm-start from v26 forgets takeoff because Phase 2 dominates early episodes | `p_true_start = 0.4` floor — keep practicing takeoff. v29 lesson. |

---

## Execution notes

- Plan is sized for ~8 hours of focused work (Phase 1 fix is small, Phase 2 is bulk).
- Land Task A1 / A2 as separate commits before B1; gives a clean revert path
  if Phase 2 has a buried bug.
- Smoke tests run on the RTX 5090 (`ssh ubuntu@161.184.215.24`). 100k steps
  ≈ 90 s on this box. 5M ≈ 75 min.
- After Task C1 launches, the iteration loop is wandb-watching for ~12 h.
