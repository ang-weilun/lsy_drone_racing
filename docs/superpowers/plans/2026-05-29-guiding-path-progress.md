# Guiding-Path Arc-Length Progress (geometric myopia fix, Path A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace direction-blind centre-distance progress with **Penička-style arc-length progress along a short, per-transition guiding path** (`center_{K−1} → exit_{K−1} → entry_K → center_K`, Bézier-smoothed corner) so that after a plane crossing the rewarded direction is *forward out of the just-passed frame*, structurally removing the reverse-out and lateral-clip local optima on randomised L3 tracks.

**Architecture:** This is **Path A** — report 2's **RANK 1** (Penička guiding-path) + **RANK 2** (soft gate-frame barrier). Chosen over the cheap local "augment" hybrid because (i) the simple "entry-waypoint" variant was numerically verified inert, and (ii) the newest, most scale-matched literature (Penička 2022; Liu 2024; Pasumarti 2025 on Crazyflie 2.1) ranks guiding-path progress first.

**RANK 1 and RANK 2 ship together in the first experiment** (not staged), because arc-length progress `r = progress_coef·Δs` is **telescoping** — `Σ_t r_t = progress_coef·(s_end − s_start)`, i.e. *route-independent*. Geometry-solo therefore only penalizes the *immediate* reverse step (the leading-segment sign); it provably **cannot** prevent the bank-around-vs-reverse-back-through-frame choice on its own (both routes reach gate K at the same arc length). The gate-frame barrier (`r_gate_frame`, already in `reward.py`, windows the just-passed frame) is the term that clears the frame. Report 2 also endorses adding RANK 1+2 concurrently. We relax single-lever purity deliberately and with eyes open: each term's role is known a priori (geometry → myopic/immediate-reverse gradient; barrier → frame clearance), and the expanded diagnostic (Task 5) demonstrates the telescoping result before any GPU. See `docs/superpowers/reviews/2026-05-29-guiding-path-plan-codex-review.md`.

**Tech stack:** JAX (`jax.numpy`, vmap-clean vectorised projection — no Python loops in the hot path), SBX PPO, `fire` CLI, scipy for the diagnostic's gate quaternions. No new dependencies.

**Verification model:** This repo's `CLAUDE.md` overrides TDD — *no pytest, no `tests/`*. Each change is gated by (1) the **scripted-trajectory component-reward diagnostic** (the validated prototype, hardened to call the real `step_reward`) and (2) **sim/eval metrics** (real-lap success-rate + lap time, render).

**Source / references:** Penička & Scaramuzza 2022 (RA-L, Eqs. 7–10: `r_p(t)=k_p·(s(p_t)−s(p_{t−1}))`, anti-singularity `+k_s·s(p)`); Song 2021 safety reward; Liu 2024 (arXiv:2411.04246) gate-frame guidance. Decision trail: `docs/research/2026-05-29-reward-myopia-redesign/` (README §4–7, reports 2 & 3, codex-review.md). Handoff: `docs/handoffs/2026-05-29-session-wrap-geometric-fix-and-L3-plan.md`.

---

## Design (numerically validated 2026-05-29 before writing this plan)

**Guiding path for target gate `K = target_idx`** (the gate being approached during the step):

```
prev gate P = K−1   (just-passed)              target gate K
exit_P  = center_P + d_exit  · normal_P        # beyond P along its through-normal (forward)
entry_K = center_K − d_entry · normal_K        # before K on its approach side
path nodes = [ center_P ,  Bézier_samples(exit_P, entry_K (control), center_K) ]   # M+1 nodes
```

- The **leading `center_P → exit_P` segment** is what makes a backward step right after the plane *decrease* arc length (penalised), and a forward step *increase* it (rewarded). Validated: reverse-out step `r_prog` flips from **+1.0 (baseline rewards reversing)** to **−1.0…−3.0**; forward-to-exit flips to **+2.0**.
- The **quadratic Bézier** `B(t)=(1−t)²·exit_P + 2(1−t)t·entry_K + t²·center_K` rounds the corner and enters `center_K` tangent to `+normal_K` (through-direction). Validated: side-clip lateral step `r_prog` drops from **+1.61 (rewards the clip)** to **0.0**; forward stays **+2.0**; normal approach stays **+2.5**.
- **`K = 0` (first gate from spawn): fall back to centre-distance progress.** No previous gate exists, and the spawn→gate-0 approach has no successor-geometry pathology. (Verified: a gate-0 path starting at `entry_0` gives zero progress while the drone is still behind `entry_0`.)
- **Arc length** `s(p)`: project `p` onto the node polyline (vectorised point-to-segment, pick min-distance segment, accumulate length to that point). `r_prog = progress_coef·(s(pos) − s(prev_pos)) + path_progress_ks·s(pos)`. Both `pos` and `prev_pos` project onto the **same** path (built from the current step's `target_idx`).
- **Anti-singularity term `+k_s·s(p)`** (Penička Eq. 10): default `path_progress_ks = 0.0` (Bézier smoothing already removes most corner singularities); exposed as a knob. Keep small — large `k_s` rewards loitering at high arc length.
- **Liu zero-on-pass:** `r_prog = where(zero_progress_on_pass & gate_just_passed, 0, r_prog)` — the path redefinition at the gate hand-off makes the one-step delta meaningless.
- **Sample count `M`** is a module-level constant (`_PATH_SMOOTH_SAMPLES = 16`), not a config float — it sets JAX array shapes and must be static.

**Telescoping caveat (Codex finding #1, the load-bearing reason RANK 2 ships with RANK 1):** with `path_progress_ks = 0`, `Σ_t r_prog = progress_coef·(s_end − s_start)` — route-independent. So arc-length progress *cannot* distinguish "bank around the just-passed frame" from "reverse back through it" once the drone is past `exit_P`; both reach gate K at the same arc length. On a near-collinear 180° the Bézier even folds back *through* the just-passed aperture, so the back-through-frame route is locally rewarded as forward progress. The leading `center_{K-1}→exit_{K-1}` segment still penalizes the *immediate* reverse (validated), and on realistically-offset ~180° the path bows out laterally and routes around the frame — but the **frame is cleared by `r_gate_frame`, not by the progress term**. Hence RANK 2 is in the first experiment.

**Projection-ambiguity caveat (documented, not v1):** on the return leg of a near-collinear U-turn the folded path can self-overlap, where closest-point projection is ambiguous. The leading segment handles the critical just-passed region without it. If renders show return-leg progress glitches, the fix is a **stateful monotonic segment index** (per-env "furthest segment reached", threaded through the env wrapper) — a follow-on, gated on the diagnostic, not v1.

**Training:** actor-only warm-start + **critic reset** + **Phase-2 replay cleared** (`phase2_prob=0.0`) — the old critic/replay encode the old reward geometry.

---

## File structure

| File | Change | Responsibility |
|---|---|---|
| `lsy_drone_racing/control/rl_song/config.py` | Modify `RewardConfig` (+5 fields) | `use_path_progress`, `path_exit_offset_m`, `path_entry_offset_m`, `path_progress_ks`, `zero_progress_on_pass`. |
| `lsy_drone_racing/control/rl_song/reward.py` | Add `_guiding_path_nodes` + `_path_arclength` helpers; rewrite `r_prog` block (~242–245); sum `r_gate_frame` (~578) | Guiding-path arc-length progress + K=0 fallback + zero-on-pass (RANK 1) and the gated gate-frame barrier in the scalar (RANK 2). |
| `lsy_drone_racing/control/rl_sbx/train.py` | Modify (kwargs + `RewardConfig(...)` + warm-start gate) | Thread the 5 path knobs + `use_gate_frame_barrier` + `init_actor_only` through `fire` CLI. |
| `scripts/diag_path_progress_reward.py` | Create | Scripted-trajectory diagnostic calling the real `step_reward` (the pre-PPO gate). |

---

## Task 1: Add the guiding-path knobs to `RewardConfig`

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/config.py` (insert after `progress_coef: float = 10.0`, ~line 355)

- [ ] **Step 1: Add five fields**

```python
    # 2026-05-29 (geometric fix, Path A / RANK 1): Penička-style arc-length
    # progress along a per-transition guiding path
    #   [center_{K-1}, exit_{K-1}, entry_K, center_K]   (Bézier-smoothed corner)
    # exit_{K-1} = center_{K-1} + path_exit_offset_m * normal_{K-1}  (forward of
    # the just-passed gate); entry_K = center_K - path_entry_offset_m * normal_K
    # (approach side of the target). Arc-length progress makes a backward step
    # right after the plane DECREASE progress (reverse-out penalised) and a
    # forward step increase it. False -> centre-distance (pure-Song) baseline.
    # K=0 always uses centre-distance (no previous gate; spawn approach is
    # pathology-free). See docs/research/2026-05-29-reward-myopia-redesign.
    use_path_progress: bool = False
    # Exit waypoint distance beyond the just-passed gate along its +normal (m).
    # Crazyflie ~0.45 m gates -> 0.3-0.5 m; tuning knob (Penička scale caveat).
    path_exit_offset_m: float = 0.4
    # Entry waypoint distance before the target gate along its -normal (m).
    path_entry_offset_m: float = 0.4
    # Penička anti-singularity term k_s in r = k_p*(s_t - s_{t-1}) + k_s*s_t.
    # 0.0 = pure arc-length delta (Bézier smoothing handles most corner
    # singularities). Keep small: large k_s rewards loitering at high arc length.
    path_progress_ks: float = 0.0
    # 2026-05-29: zero r_prog on the gate-pass step (Liu 2024 ~wp_passing). The
    # guiding path is redefined at the hand-off, so the one-step delta is
    # meaningless there. Default False preserves current behaviour.
    zero_progress_on_pass: bool = False
```

- [ ] **Step 2: Verify import**

Run: `cd /home/exedev/lsy_drone_racing && python -c "from lsy_drone_racing.control.rl_song.config import RewardConfig as R; c=R(); print(c.use_path_progress, c.path_exit_offset_m, c.path_entry_offset_m, c.path_progress_ks, c.zero_progress_on_pass)"`
Expected: `False 0.4 0.4 0.0 False`

- [ ] **Step 3: Lint + commit**

```bash
ruff format lsy_drone_racing/control/rl_song/config.py && ruff check lsy_drone_racing/control/rl_song/config.py
git add lsy_drone_racing/control/rl_song/config.py
git commit -m "rl_song/config: add guiding-path progress knobs (Path A)"
```

---

## Task 2: Add the guiding-path helpers to `reward.py`

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/reward.py` (add module constants near line 38; add two helpers after `_gate_phi`, ~line 143)

- [ ] **Step 1: Add module-level Bézier constants** (after `SEGMENT_AB_SQ_EPS`, ~line 37)

```python
# Quadratic-Bézier sample count for the guiding-path corner. Static (sets JAX
# array shapes); not a runtime config field.
_PATH_SMOOTH_SAMPLES: int = 16
_BEZIER_T: Array = jnp.linspace(0.0, 1.0, _PATH_SMOOTH_SAMPLES)
_BEZIER_W0: Array = jnp.square(1.0 - _BEZIER_T)  # exit_prev weight
_BEZIER_W1: Array = 2.0 * (1.0 - _BEZIER_T) * _BEZIER_T  # entry_tgt (control) weight
_BEZIER_W2: Array = jnp.square(_BEZIER_T)  # center_tgt weight
```

- [ ] **Step 2: Add the path-builder and arc-length helpers** (after `_gate_phi`, before `step_reward`)

```python
def _guiding_path_nodes(
    prev_gate_pos: Array,
    prev_gate_normal: Array,
    gate_pos: Array,
    gate_normal: Array,
    exit_offset: float,
    entry_offset: float,
) -> Array:
    """Build the per-transition guiding-path nodes for each env.

    Path = ``[center_{K-1}] ++ Bézier(exit_{K-1}, entry_K, center_K)`` with a
    quadratic Bézier rounding the corner (control point = entry waypoint).

    Parameters
    ----------
    prev_gate_pos, prev_gate_normal : Array, shape (n_envs, 3)
        Just-passed gate centre and through-normal (gate-local +x in world).
    gate_pos, gate_normal : Array, shape (n_envs, 3)
        Target gate centre and through-normal.
    exit_offset, entry_offset : float
        Waypoint offsets (m) along the respective normals.

    Returns:
    -------
    Array, shape (n_envs, _PATH_SMOOTH_SAMPLES + 1, 3)
        Guiding-path node positions in world frame.
    """
    exit_prev = prev_gate_pos + exit_offset * prev_gate_normal  # (n,3)
    entry_tgt = gate_pos - entry_offset * gate_normal  # (n,3)
    corner = (
        _BEZIER_W0[None, :, None] * exit_prev[:, None, :]
        + _BEZIER_W1[None, :, None] * entry_tgt[:, None, :]
        + _BEZIER_W2[None, :, None] * gate_pos[:, None, :]
    )  # (n, M, 3)
    return jnp.concatenate([prev_gate_pos[:, None, :], corner], axis=1)  # (n, M+1, 3)


def _path_arclength(nodes: Array, pos: Array) -> Array:
    """Arc length of the projection of ``pos`` onto the node polyline.

    Vectorised point-to-segment projection over all segments; selects the
    minimum-distance segment and accumulates path length to the closest point.

    Parameters
    ----------
    nodes : Array, shape (n_envs, n_nodes, 3)
    pos : Array, shape (n_envs, 3)

    Returns:
    -------
    Array, shape (n_envs,)
        Arc length ``s(pos)`` along the polyline.
    """
    a = nodes[:, :-1, :]  # (n, S, 3)
    b = nodes[:, 1:, :]
    ab = b - a
    seg_len = jnp.linalg.norm(ab, axis=-1)  # (n, S)
    ab_sq = jnp.maximum(jnp.sum(ab * ab, axis=-1), SEGMENT_AB_SQ_EPS)  # (n, S)
    ap = pos[:, None, :] - a  # (n, S, 3)
    t = jnp.clip(jnp.sum(ap * ab, axis=-1) / ab_sq, 0.0, 1.0)  # (n, S)
    closest = a + t[..., None] * ab
    dist = jnp.linalg.norm(pos[:, None, :] - closest, axis=-1)  # (n, S)
    # Cumulative length to the START of each segment: [0, L0, L0+L1, ...].
    cum_end = jnp.cumsum(seg_len, axis=-1)
    cum_start = jnp.concatenate(
        [jnp.zeros((seg_len.shape[0], 1), seg_len.dtype), cum_end[:, :-1]], axis=-1
    )
    s_per_seg = cum_start + t * seg_len  # (n, S)
    best = jnp.argmin(dist, axis=-1)  # (n,)
    return jnp.take_along_axis(s_per_seg, best[:, None], axis=-1)[:, 0]  # (n,)
```

- [ ] **Step 3: Lint**

Run: `ruff format lsy_drone_racing/control/rl_song/reward.py && ruff check lsy_drone_racing/control/rl_song/reward.py`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add lsy_drone_racing/control/rl_song/reward.py
git commit -m "rl_song/reward: guiding-path node + arc-length helpers (Path A)"
```

---

## Task 3: Wire the `r_prog` block to use guiding-path progress

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/reward.py:242-245` (the `r_prog` block; `target_gate_xaxis_world` at line 234 and `x_local_target` at line 241 already exist above it)

- [ ] **Step 1: Replace the two `r_prog` lines (244–245)**

Replace:
```python
    r_prog = jnp.linalg.norm(gate_pos - prev_pos, axis=-1)
    r_prog = reward_cfg.progress_coef * (r_prog - jnp.linalg.norm(gate_pos - pos, axis=-1))
```
with:
```python
    # Centre-distance progress (pure-Song baseline; also the K=0 path).
    r_prog_center = reward_cfg.progress_coef * (
        jnp.linalg.norm(gate_pos - prev_pos, axis=-1)
        - jnp.linalg.norm(gate_pos - pos, axis=-1)
    )
    if reward_cfg.use_path_progress:
        # Guiding-path arc-length progress (Path A / RANK 1). The leading
        # center_{K-1} -> exit_{K-1} segment makes a backward step right after
        # the plane decrease arc length (reverse-out penalised); the Bézier
        # corner removes the lateral-clip reward. K=0 has no previous gate, so
        # fall back to centre-distance there. See
        # docs/research/2026-05-29-reward-myopia-redesign.
        prev_idx = jnp.maximum(target_idx - 1, 0)
        prev_gate_pos = gates_pos[env_idx, prev_idx]
        prev_gate_normal = _quat_to_matrix(gates_quat[env_idx, prev_idx])[..., :, 0]
        nodes = _guiding_path_nodes(
            prev_gate_pos,
            prev_gate_normal,
            gate_pos,
            target_gate_xaxis_world,
            reward_cfg.path_exit_offset_m,
            reward_cfg.path_entry_offset_m,
        )
        s_cur = _path_arclength(nodes, pos)
        s_prev = _path_arclength(nodes, prev_pos)
        r_prog_path = (
            reward_cfg.progress_coef * (s_cur - s_prev)
            + reward_cfg.path_progress_ks * s_cur
        )
        r_prog = jnp.where(target_idx > 0, r_prog_path, r_prog_center)
    else:
        r_prog = r_prog_center
    # Liu zero-on-pass: the guiding path is redefined at the gate hand-off
    # (target_idx = prev_target on the pass step), so the one-step delta is
    # meaningless; drop it. Crash-step zeroing happens later (~line 542).
    r_prog = jnp.where(
        jnp.asarray(reward_cfg.zero_progress_on_pass, dtype=bool) & gate_just_passed,
        jnp.zeros_like(r_prog),
        r_prog,
    )
```

- [ ] **Step 2: Confirm baseline equivalence (use_path_progress=False ⇒ unchanged)**

Run:
```bash
cd /home/exedev/lsy_drone_racing && python - <<'PY'
import jax.numpy as jnp
from lsy_drone_racing.control.rl_song.reward import step_reward
from lsy_drone_racing.control.rl_song.config import RewardConfig
n=1
def obs(p,t): return {"pos":jnp.asarray([p],jnp.float32),"vel":jnp.zeros((n,3),jnp.float32),
  "quat":jnp.asarray([[0,0,0,1]],jnp.float32),"ang_vel":jnp.zeros((n,3),jnp.float32),
  "target_gate":jnp.asarray([t],jnp.int32),"gates_pos":jnp.asarray([[[0,0,1.],[0,2,1.]]],jnp.float32),
  "gates_quat":jnp.asarray([[[0,0,0,1],[0,0,0,1]]],jnp.float32),"obstacles_pos":jnp.asarray([[[9,9,9.]]],jnp.float32)}
prev,cur=obs([-1.,0,1.],0),obs([-.5,0,1.],0)
f=dict(terminated=jnp.zeros(n,bool),truncated=jnp.zeros(n,bool),finished=jnp.zeros(n,bool),gate_just_passed=jnp.zeros(n,bool))
_,c=step_reward(cur,prev,**f,reward_cfg=RewardConfig(),true_gates_pos=cur["gates_pos"],true_gates_quat=cur["gates_quat"])
print("baseline r_prog:",float(c["r_prog"][0]))  # 10*(1.0-0.5)=5.0
PY
```
Expected: `baseline r_prog: 5.0`

- [ ] **Step 3: Lint + type-check + commit**

```bash
ruff format lsy_drone_racing/control/rl_song/reward.py && ruff check lsy_drone_racing/control/rl_song/reward.py && mypy lsy_drone_racing/control/rl_song/reward.py
git add lsy_drone_racing/control/rl_song/reward.py
git commit -m "rl_song/reward: guiding-path arc-length progress in r_prog (Path A)"
```

---

## Task 3b: Activate the gate-frame barrier in the summed scalar (RANK 2)

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/reward.py:578` (the `reward = ...` sum)

Bundles RANK 2 with RANK 1 (see Architecture — the telescoping property makes the
barrier necessary to clear the frame). `r_gate_frame` is already computed and is
gated by `use_gate_frame_barrier` (`jnp.where(...)` → 0 when disabled), so adding it
to the sum is backward-compatible: with `use_gate_frame_barrier=False` the scalar is
unchanged from the pure-Song baseline.

- [ ] **Step 1: Add `r_gate_frame` to the summed reward**

Replace (line ~578):
```python
    reward = r_prog + r_omega + r_smooth + r_crash + r_finish + r_time
```
with:
```python
    # r_gate_frame is gated by use_gate_frame_barrier (-> 0 when disabled), so this
    # is identical to the pure-Song baseline unless the barrier is turned on. Bundled
    # with guiding-path progress because arc-length progress is telescoping and cannot
    # clear the just-passed frame on its own — see the plan Architecture / Codex review.
    reward = r_prog + r_omega + r_smooth + r_crash + r_finish + r_time + r_gate_frame
```

- [ ] **Step 2: Update the `step_reward` docstring** (~line 202–208) so the "terms
  summed into `reward`" list includes `r_gate_frame` (it currently lists it as
  diagnostic-only). Change the sentence listing summed terms to add `r_gate_frame`,
  and remove it from the "NOT summed" list.

- [ ] **Step 3: Confirm baseline equivalence (barrier off ⇒ unchanged)**

Run:
```bash
cd /home/exedev/lsy_drone_racing && python - <<'PY'
import jax.numpy as jnp
from lsy_drone_racing.control.rl_song.reward import step_reward
from lsy_drone_racing.control.rl_song.config import RewardConfig
n=1
def obs(p,t): return {"pos":jnp.asarray([p],jnp.float32),"vel":jnp.zeros((n,3),jnp.float32),
  "quat":jnp.asarray([[0,0,0,1]],jnp.float32),"ang_vel":jnp.zeros((n,3),jnp.float32),
  "target_gate":jnp.asarray([t],jnp.int32),"gates_pos":jnp.asarray([[[0,0,1.],[0,2,1.]]],jnp.float32),
  "gates_quat":jnp.asarray([[[0,0,0,1],[0,0,0,1]]],jnp.float32),"obstacles_pos":jnp.asarray([[[9,9,9.]]],jnp.float32)}
prev,cur=obs([-1.,0,1.],0),obs([-.5,0,1.],0)
f=dict(terminated=jnp.zeros(n,bool),truncated=jnp.zeros(n,bool),finished=jnp.zeros(n,bool),gate_just_passed=jnp.zeros(n,bool))
r,_=step_reward(cur,prev,**f,reward_cfg=RewardConfig(),true_gates_pos=cur["gates_pos"],true_gates_quat=cur["gates_quat"])
print("baseline total reward:",float(r[0]))  # barrier off -> r_gate_frame=0; unchanged
PY
```
Expected: a finite scalar with `use_gate_frame_barrier=False` (default) — the barrier contributes 0.

- [ ] **Step 4: Lint + commit**

```bash
ruff format lsy_drone_racing/control/rl_song/reward.py && ruff check lsy_drone_racing/control/rl_song/reward.py
git add lsy_drone_racing/control/rl_song/reward.py
git commit -m "rl_song/reward: sum r_gate_frame (gated) to activate RANK 2 barrier"
```

---

## Task 4: Thread knobs + actor-only warm-start through `train.py`

**Files:**
- Modify: `lsy_drone_racing/control/rl_sbx/train.py` (signature ~131–157; `RewardConfig(...)` ~242; warm-start ~420–423)

- [ ] **Step 1: Add kwargs to `train(...)`** — after `dipole_sigma: float = 0.5,` (line 144). (`gate_frame_weight` is already a kwarg at line 135; only `use_gate_frame_barrier` is missing.)
```python
    use_path_progress: bool = False,
    path_exit_offset_m: float = 0.4,
    path_entry_offset_m: float = 0.4,
    path_progress_ks: float = 0.0,
    zero_progress_on_pass: bool = False,
    use_gate_frame_barrier: bool = False,
```
and after `init_from: str | None = None,` (line 155):
```python
    init_actor_only: bool = False,
```

- [ ] **Step 2: Pass into `RewardConfig(...)`** — after `dipole_sigma=dipole_sigma,` (line 256). (`gate_frame_weight=gate_frame_weight` is already passed at line 247.)
```python
        use_path_progress=use_path_progress,
        path_exit_offset_m=path_exit_offset_m,
        path_entry_offset_m=path_entry_offset_m,
        path_progress_ks=path_progress_ks,
        zero_progress_on_pass=zero_progress_on_pass,
        use_gate_frame_barrier=use_gate_frame_barrier,
```

- [ ] **Step 3: Gate the critic load** — replace lines 420–423:
```python
        model.policy.actor_state = model.policy.actor_state.replace(params=loaded["actor_params"])
        model.policy.vf_state = model.policy.vf_state.replace(params=loaded["critic_params"])
        env.set_actor_normalizer(loaded["actor_normalizer"])
        env.set_critic_normalizer(loaded["critic_normalizer"])
```
with:
```python
        model.policy.actor_state = model.policy.actor_state.replace(params=loaded["actor_params"])
        env.set_actor_normalizer(loaded["actor_normalizer"])
        # The critic normalizer is observation-distribution state (Welford stats
        # over the critic obs; see checkpoint.py), NOT reward geometry, so always
        # load it — a cold normalizer would feed the fresh critic unnormalized
        # inputs (Codex review #2).
        env.set_critic_normalizer(loaded["critic_normalizer"])
        if init_actor_only:
            # Reward geometry changed (guiding-path progress): only the critic
            # PARAMS encode value estimates for the old reward, so leave them
            # freshly initialized and let the critic relearn. The actor and the
            # critic-obs normalizer (loaded above) are kept. See
            # docs/superpowers/reviews/2026-05-29-guiding-path-plan-codex-review.md.
            print("actor-only warm-start: critic params reset, normalizers kept", flush=True)
        else:
            model.policy.vf_state = model.policy.vf_state.replace(
                params=loaded["critic_params"]
            )
```

- [ ] **Step 4: Verify CLI surface**

Run: `cd /home/exedev/lsy_drone_racing && python -c "import inspect; from lsy_drone_racing.control.rl_sbx.train import train; p=inspect.signature(train).parameters; print(all(k in p for k in ['use_path_progress','path_exit_offset_m','path_entry_offset_m','path_progress_ks','zero_progress_on_pass','use_gate_frame_barrier','init_actor_only']))"`
Expected: `True`

- [ ] **Step 5: Lint + commit**

```bash
ruff format lsy_drone_racing/control/rl_sbx/train.py && ruff check lsy_drone_racing/control/rl_sbx/train.py
git add lsy_drone_racing/control/rl_sbx/train.py
git commit -m "rl_sbx/train: guiding-path + gate-frame CLI knobs, --init-actor-only"
```

---

## Task 5: Scripted-trajectory diagnostic (the pre-PPO gate)

**Files:**
- Create: `scripts/diag_path_progress_reward.py` (the validated prototype, hardened to call the real `step_reward`)

- [ ] **Step 1: Create the script**

```python
"""Scripted-trajectory diagnostic for guiding-path progress (Path A / RANK 1).

Builds synthetic observations for the reverse-out, side-clip, normal-approach,
and K=0 geometries and compares per-step ``r_prog`` from the real ``step_reward``
under the pure-Song baseline (use_path_progress=False) vs the guiding-path fix.
Run BEFORE any PPO. Pass criteria encode the validated sign behaviour:
  reverse-out reverse step  : baseline > 0 (rewards reversing); guiding < 0
  side-clip lateral step    : baseline > 0 (rewards clipping);  guiding ~ 0
  forward-to-exit / approach : guiding > 0

Multi-step checks (Codex review #1, #3): a telescoping demonstration (integrated
r_prog is route-independent for shared start/end -> the progress term alone cannot
clear the just-passed frame, which is why the gate-frame barrier ships with it) and
the real gate-pass handoff (r_prog must be zeroed when gate_just_passed=True).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.rl_song.config import RewardConfig
from lsy_drone_racing.control.rl_song.reward import step_reward

Z = 1.0
TOL = 0.15


def yaw_quat(yaw: float) -> list[float]:
    return Rotation.from_euler("z", yaw).as_quat().tolist()


def obs(pos, target, gates_pos, gates_quat):
    n = 1
    return {
        "pos": jnp.asarray([pos], jnp.float32),
        "vel": jnp.zeros((n, 3), jnp.float32),
        "quat": jnp.asarray([[0.0, 0.0, 0.0, 1.0]], jnp.float32),
        "ang_vel": jnp.zeros((n, 3), jnp.float32),
        "target_gate": jnp.asarray([target], jnp.int32),
        "gates_pos": jnp.asarray([gates_pos], jnp.float32),
        "gates_quat": jnp.asarray([gates_quat], jnp.float32),
        "obstacles_pos": jnp.asarray([[[9.0, 9.0, 9.0]]], jnp.float32),
    }


def r_prog(prev_pos, cur_pos, target, gates_pos, gates_quat, cfg):
    n = 1
    cur = obs(cur_pos, target, gates_pos, gates_quat)
    prev = obs(prev_pos, target, gates_pos, gates_quat)
    _, comp = step_reward(
        cur,
        prev,
        terminated=jnp.zeros(n, bool),
        truncated=jnp.zeros(n, bool),
        finished=jnp.zeros(n, bool),
        gate_just_passed=jnp.zeros(n, bool),
        reward_cfg=cfg,
        true_gates_pos=cur["gates_pos"],
        true_gates_quat=cur["gates_quat"],
    )
    return float(comp["r_prog"][0])


def integrated_rprog(positions, target, gates_pos, gates_quat, cfg):
    """Sum r_prog over consecutive positions along a scripted trajectory."""
    return sum(
        r_prog(positions[i], positions[i + 1], target, gates_pos, gates_quat, cfg)
        for i in range(len(positions) - 1)
    )


def r_prog_passstep(prev_pos, cur_pos, prev_target, cur_target, gates_pos, gates_quat, cfg):
    """r_prog on the real gate-pass handoff step (gate_just_passed=True)."""
    n = 1
    cur = obs(cur_pos, cur_target, gates_pos, gates_quat)
    prev = obs(prev_pos, prev_target, gates_pos, gates_quat)
    _, comp = step_reward(
        cur,
        prev,
        terminated=jnp.zeros(n, bool),
        truncated=jnp.zeros(n, bool),
        finished=jnp.zeros(n, bool),
        gate_just_passed=jnp.ones(n, bool),
        reward_cfg=cfg,
        true_gates_pos=cur["gates_pos"],
        true_gates_quat=cur["gates_quat"],
    )
    return float(comp["r_prog"][0])


def main() -> None:
    base = RewardConfig()
    fix = RewardConfig(use_path_progress=True, path_exit_offset_m=0.4, path_entry_offset_m=0.4)
    ok = True
    print(f"{'scenario':<34}{'baseline':>10}{'guiding':>10}{'verdict':>9}")

    def row(name, prev, cur, tgt, gp, gq, want):
        nonlocal ok
        b = r_prog(prev, cur, tgt, gp, gq, base)
        g = r_prog(prev, cur, tgt, gp, gq, fix)
        passed = {
            "neg": g < -TOL,
            "pos": g > TOL,
            "zero": abs(g) <= TOL,
        }[want]
        ok = ok and passed
        print(f"{name:<34}{b:>10.2f}{g:>10.2f}{'PASS' if passed else 'FAIL':>9}")

    # gate0 faces +x at origin; reverse-out has gate1 behind facing -x.
    gp_rev = [[0, 0, Z], [-1.5, 0, Z]]
    gq_rev = [yaw_quat(0.0), yaw_quat(np.pi)]
    row("reverse-out: reverse thru gate0", [0.3, 0, Z], [0.2, 0, Z], 1, gp_rev, gq_rev, "neg")
    row("reverse-out: forward to exit", [0.2, 0, Z], [0.4, 0, Z], 1, gp_rev, gq_rev, "pos")

    # side-clip: gate1 off to +y facing +y.
    gp_side = [[0, 0, Z], [1.5, 2.0, Z]]
    gq_side = [yaw_quat(0.0), yaw_quat(np.pi / 2)]
    row("side-clip: lateral +y", [0.1, 0, Z], [0.1, 0.2, Z], 1, gp_side, gq_side, "zero")
    row("side-clip: forward +x out of gate0", [0.1, 0, Z], [0.3, 0, Z], 1, gp_side, gq_side, "pos")
    row("side-clip: approach gate1", [1.5, 1.0, Z], [1.5, 1.3, Z], 1, gp_side, gq_side, "pos")

    # K=0 (centre-distance fallback): spawn approach to gate0.
    row("K=0: approach gate0 (+x)", [-1.0, 0, Z], [-0.7, 0, Z], 0, gp_rev, gq_rev, "pos")

    # --- Multi-step checks (Codex review #1, #3) ---
    fix_ks0 = RewardConfig(
        use_path_progress=True,
        path_exit_offset_m=0.4,
        path_entry_offset_m=0.4,
        path_progress_ks=0.0,
        zero_progress_on_pass=True,
    )
    # Offset ~180deg: gate1 behind and to the +y side of gate0.
    gp_u = [[0, 0, Z], [-1.5, 0.6, Z]]
    gq_u = [yaw_quat(0.0), yaw_quat(np.pi)]
    start = [0.3, 0.0, Z]
    end = [-1.4, 0.6, Z]  # near gate1 centre
    bank = [start, [0.45, 0.0, Z], [0.5, 0.6, Z], [-0.5, 1.0, Z], [-1.4, 0.8, Z], end]
    rev = [start, [0.1, 0.1, Z], [-0.3, 0.2, Z], [-0.9, 0.4, Z], end]
    s_bank = integrated_rprog(bank, 1, gp_u, gq_u, fix_ks0)
    s_rev = integrated_rprog(rev, 1, gp_u, gq_u, fix_ks0)
    print(
        f"\n[telescoping] integrated r_prog  bank-around={s_bank:+.2f}  "
        f"reverse-through-frame={s_rev:+.2f}  (k_s=0, shared start/end)"
    )
    print(
        "  -> route-independent (telescoping): the progress term alone does NOT "
        "clear the frame; r_gate_frame is required (this is WHY RANK 2 ships now)."
    )

    # Real pass-step handoff: r_prog must be 0 (zero-on-pass).
    rp_pass = r_prog_passstep([0.0, 0, Z], [0.05, 0, Z], 0, 1, gp_u, gq_u, fix_ks0)
    pass_ok = abs(rp_pass) <= 1e-6
    print(
        f"[pass-step] r_prog on gate_just_passed = {rp_pass:+.4f}  "
        f"{'PASS (zeroed)' if pass_ok else 'FAIL (should be 0)'}"
    )
    ok = ok and pass_ok

    print("\nALL PASS" if ok else "\nSOME FAIL — inspect before PPO")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint + commit**

```bash
ruff format scripts/diag_path_progress_reward.py && ruff check scripts/diag_path_progress_reward.py
git add scripts/diag_path_progress_reward.py
git commit -m "scripts: scripted-trajectory diagnostic for guiding-path progress"
```

---

## Task 6: Run the diagnostic — pre-PPO go/no-go gate

**Files:** none (verification only).

- [ ] **Step 1: Run it**

Run: `cd /home/exedev/lsy_drone_racing && JAX_PLATFORMS=cpu python scripts/diag_path_progress_reward.py`
Expected: every single-step row `PASS` and the pass-step `PASS (zeroed)`, `ALL PASS` (exit 0). In particular reverse-out reverse step `guiding < 0` while `baseline > 0`, and side-clip lateral step `guiding ≈ 0` while `baseline > 0`. The `[telescoping]` line should show bank-around ≈ reverse-through-frame (route-independence) — this is the expected confirmation that `r_gate_frame` is needed, not a failure.

- [ ] **Step 2: If any row FAILs — diagnose before GPU**

Do NOT train. Check: sign of the gate normals (`_quat_to_matrix(...)[:, 0]`), the `prev_idx`/`target_idx` selection on the synthetic obs, and whether the exit/entry offsets place the waypoints on the expected sides. Tune `path_exit_offset_m`/`path_entry_offset_m` and re-run. If the *return-leg* singularity shows up (near-collinear), that's the documented stateful-monotonic-index contingency — note it.

- [ ] **Step 3: Record the PASS table** in `docs/research/2026-05-29-reward-myopia-redesign/README.md` §7 decision log (baseline vs guiding per row, offsets used), dated; commit.

---

## Task 7: Training run — actor-only warm-start

**Files:** none (runs on vast box `38410004` or successor; sync the branch first — see handoff "Tooling / gotchas").

- [ ] **Step 1: Pick the warm-start base.** L2-first validation: `snapshots/sbx_spd6_a120_om005_tp20_200M` (robust, 3.47 s @ 96%, carries α_max=1.2 / omega=0.005). For the L3 push, swap to `snapshots/round7_ckpt_2.2B`. Record the choice in the README log.

- [ ] **Step 2: Launch (actor-only warm-start, critic reset, replay cleared)**

```bash
python -m lsy_drone_racing.control.rl_sbx.train \
  --run-name pathprog04_gf05_actoronly_spd6 \
  --init-from snapshots/sbx_spd6_a120_om005_tp20_200M \
  --init-actor-only True \
  --use-path-progress True \
  --path-exit-offset-m 0.4 \
  --path-entry-offset-m 0.4 \
  --zero-progress-on-pass True \
  --use-gate-frame-barrier True \
  --gate-frame-weight 0.5 \
  --alpha-max-rad 1.2 \
  --omega-coef 0.005 \
  --phase2-prob 0.0 \
  --total-timesteps 200000000
```
`gate-frame-weight 0.5` is a small starting value (`gate_frame_sigma=0.08 m` is a
large fraction of the 0.20 m half-aperture — Codex scale note; historical runs used
1.2). Expected at startup: `warm-start from .../step_000201326592` and `actor-only
warm-start: critic params reset, normalizers kept`. Confirm `reward/r_prog` and
`reward/r_gate_frame` are both finite and logged from iteration 1.

- [ ] **Step 3: Sanity-watch ~20M steps.** Confirm no NaNs; `reward/r_prog` recovers after the critic-reset dip; episode length does not collapse to the hover/timeout attractor. If it collapses, stop — try a non-zero `path_progress_ks` (small), a warm critic at low LR, or revisit the offsets.

- [ ] **Step 4: Record the launch** (run name, base ckpt, flags, box id) in the README §7 log.

---

## Task 8: Evaluate — real-lap SR + lap time + reverse-out/clip behaviour

**Files:** none.

- [ ] **Step 1: Real-lap eval L2 (n=50):** `bash /root/eval_l2.sh pathprog04_gf05_actoronly_spd6 50`. Compare vs spd6 (3.47 s @ 96%). The combined fix should hold/raise SR; a small lap-time regression is acceptable if SR holds (the barrier trades a little speed for fewer frame contacts — Song 2021 documents this).
- [ ] **Step 2: Render the reverse-out / side-successor L3 seeds** (numpy controller, `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`); confirm no reverse-out (geometry: drone carries forward through the exit rather than reversing at the plane) and no just-passed-frame clipping (barrier).
- [ ] **Step 3: Keep/discard.** If reverse-out/clip reduced AND L2 SR held → keep: back up the checkpoint to `gdrive:DroneRacing/checkpoints/`. Else → discard, log why, and turn the levers: raise `gate_frame_weight` if frame contact persists, raise `path_progress_ks` (small) or `path_*_offset_m` if the drone still reverses at the plane, or implement the monotonic segment index if renders show return-leg progress glitches.
- [ ] **Step 4: Record** eval table + render notes + verdict in README §7; commit.

---

## Follow-on levers (after the first experiment lands)

- **Attribution ablations** (if you need to separate the two bundled terms): a
  geometry-only run (`use_gate_frame_barrier=False`) and a barrier-only run
  (`use_path_progress=False`), each warm-started, to quantify each term's share. The
  telescoping argument predicts geometry-only fixes the immediate reverse but leaves
  frame clips; barrier-only reduces clips but keeps the myopic progress gradient.
- **Stateful monotonic segment index** — return-leg projection robustness on
  near-collinear U-turns; implement only if renders/diagnostic show return-leg glitches.
- **Wrong-side rejection** (`r_wrong_side`, already in `reward.py`) — add to the sum if
  residual overshoot remains after geometry + barrier.
- **`gate_frame_weight` / offset sweep** — tune the barrier strength and exit/entry
  offsets at Crazyflie scale (literature coefficients are for big quads).

---

## Self-review notes

- **Spec coverage:** path construction + leading segment + Bézier (T2), arc-length + k_s + K=0 fallback + zero-on-pass (T1–T3), gate-frame barrier summed (T3b), CLI + actor-only-warm-start (critic params only) + replay-clear (T4, T7), pre-PPO diagnostic gate incl. telescoping + pass-step (T5–T6), measurement (T8), ablations/monotonic-index/wrong-side (follow-on).
- **No pytest:** intentional (`CLAUDE.md` overrides TDD); verification = diagnostic (T5–T6) + sim eval (T8).
- **Type/name consistency:** `use_path_progress`, `path_exit_offset_m`, `path_entry_offset_m`, `path_progress_ks`, `zero_progress_on_pass`, `use_gate_frame_barrier`, `init_actor_only` identical across `RewardConfig`, `train(...)`, the `RewardConfig(...)` call, and CLI (fire maps `_`↔`-`). Helpers `_guiding_path_nodes`/`_path_arclength` and constants `_PATH_SMOOTH_SAMPLES`/`_BEZIER_W{0,1,2}` referenced exactly as defined.
- **Codex review applied** (`docs/superpowers/reviews/2026-05-29-guiding-path-plan-codex-review.md`): (#1) bundle RANK 2 with RANK 1 because arc-length progress is telescoping → barrier needed to clear the frame; (#2) `init_actor_only` resets critic params only, keeps the critic-obs normalizer; (#3) diagnostic expanded with telescoping demo + real pass-step + reverse-through-frame.
- **Single-lever relaxation (deliberate):** the first experiment changes `r_prog` geometry AND sums the gated `r_gate_frame`. Justified: each term's role is known a priori and the follow-on ablations can separate them; geometry-solo provably can't clear the frame (telescoping).
- **Numerically validated** before writing (sign-flip table in Design); diagnostic (T5) re-checks against the real `step_reward`, incl. the telescoping property.
