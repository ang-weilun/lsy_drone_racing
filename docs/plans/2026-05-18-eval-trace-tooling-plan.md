# Eval Trace Tooling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
>
> **No pytest.** Per `CLAUDE.md` this repo skips unit tests; validation is by running eval/analysis against a known checkpoint and inspecting outputs.
>
> **No AI branding** in commit messages, per `CLAUDE.md`.

**Goal:** Build sim-eval trace dumping + offline analysis tooling so overnight autoresearch can read what happened in eval runs without a human watching the rendered video.

**Architecture:** Opt-in per-step JSONL dump emitted by `eval_sim.py` when `--dump-trace <dir>` is set; a separate `scripts/analyze_eval_traces.py` reads those JSONL files and produces per-episode + run-level summary JSON. The full design (schemas, detectors, edge cases) is in `docs/plans/2026-05-18-eval-trace-tooling-design.md` — **read that first**.

**Tech Stack:** stdlib `json`, NumPy, JAX (for one `step_reward` call per env step), reuse of `_quat_to_matrix` / `_gate_frame_edge_dist_sq` from `lsy_drone_racing/control/rl_song/{obs,reward}.py`.

**Validation checkpoint:** `lsy_drone_racing/control/rl_song/checkpoints/level3_v33b_warmstart_from_v32a_seed0_300M/` (the current SOTA). Used for the smoke runs throughout the plan.

---

## Phase A — Prep work (independent, ~30 min)

### Task A1: Persist `reward_config.json` at training start

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/train.py:144-149` (the early section of `train()`, right after `run_dir = CHECKPOINT_DIR / run_name`)

**Step 1: Add the write with the warm-start guard**

Insert immediately after line 148 (`run_dir = CHECKPOINT_DIR / run_name`):

```python
run_dir.mkdir(parents=True, exist_ok=True)
_write_reward_config(run_dir, train_cfg)
```

Add a helper near the other private helpers in `train.py`:

```python
def _write_reward_config(run_dir: Path, train_cfg: TrainConfig) -> None:
    """Persist ``train_cfg.reward`` as JSON at ``run_dir/reward_config.json``.

    Written once at training start. On warm-start (file already present),
    refuses to overwrite a config that does not match the current one —
    a silent overwrite would mislabel every checkpoint produced by the
    prior run under the stale config.
    """
    import json
    from dataclasses import asdict

    target = run_dir / "reward_config.json"
    current = asdict(train_cfg.reward)
    if target.exists():
        existing = json.loads(target.read_text())
        if existing != current:
            raise RuntimeError(
                f"reward_config.json at {target} disagrees with current "
                f"train_cfg.reward; delete the file to overwrite "
                f"intentionally or fix train_cfg.reward."
            )
        return
    target.write_text(json.dumps(current, indent=2, sort_keys=True))
```

Make sure `Path`, `TrainConfig` are already imported (they are — Path is from pathlib stdlib, TrainConfig from `.config`). `json` and `dataclasses.asdict` go inside the helper to keep top-level imports tidy.

**Step 2: Validate by starting a tiny training run**

```powershell
pixi run -e rl-train python -m lsy_drone_racing.control.rl_song.train --run-name plan-A1-smoke --total-steps 1
```

Expected: `lsy_drone_racing/control/rl_song/checkpoints/plan-A1-smoke/reward_config.json` exists and contains a JSON object with `progress_coef`, `guide_coef`, `obstacle_weight`, `gate_frame_weight`, etc.

**Step 3: Validate warm-start guard**

Re-run the same command. Expected: no error, no overwrite (the existing-and-matches branch is hit). Then hand-edit the JSON (e.g. change `progress_coef` to `999.0`), re-run, expected: `RuntimeError` with the "disagrees with current" message.

Clean up: `Remove-Item -Recurse lsy_drone_racing/control/rl_song/checkpoints/plan-A1-smoke`.

**Step 4: Commit**

```powershell
git add lsy_drone_racing/control/rl_song/train.py
git commit -m "feat(train): persist reward_config.json at run start"
```

---

### Task A2: Surface raw policy mean via controller hook

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/controller.py:56` (the `jax.jit` line that captures `_deterministic_env_action`)
- Modify: `lsy_drone_racing/control/rl_song/controller.py:77-102` (`compute_control`)
- Modify: `lsy_drone_racing/control/rl_song/controller.py:105-110` (`_deterministic_env_action`)

**Step 1: Change `_deterministic_env_action` to return both actions**

Replace lines 105-110:

```python
def _deterministic_env_action(
    actor_params: dict[str, Any], actor_obs: Array, thrust_min: float, thrust_max: float
) -> tuple[Array, Array]:
    """Run deterministic actor inference and raw-to-env projection.

    Returns
    -------
    env_action : Array, shape (4,)
        Projected attitude command ``[roll, pitch, yaw, thrust]``.
    raw_action : Array, shape (7,)
        Policy mean before projection. Surfaced so the controller can
        stash it on ``self._last_policy_mean`` for trace logging.
    """
    raw_action = deterministic_raw_action(actor_params, actor_obs)
    env_action = raw_to_env_action(raw_action, thrust_min, thrust_max)
    return env_action, raw_action
```

**Step 2: Destructure at the call site and stash on `self`**

Replace lines 98-102 in `compute_control`:

```python
        env_action, raw_action = self._deterministic_inference(
            self.actor_params, actor_obs, self.thrust_min, self.thrust_max
        )
        self.prev_action_env_4vec = env_action
        self._last_policy_mean = np.asarray(raw_action, dtype=np.float32)
        return np.asarray(env_action, dtype=np.float32)
```

The jit'd wrapper at line 56 doesn't need changes — `jax.jit` handles pytree returns transparently.

**Step 3: Validate that controller still runs**

```powershell
pixi run -e rl-train python -m lsy_drone_racing.control.rl_song.eval_sim --config level3.toml --checkpoint lsy_drone_racing/control/rl_song/checkpoints/level3_v33b_warmstart_from_v32a_seed0_300M/ --control_mode attitude --n_runs 1
```

Expected: runs to completion, "Flight time" log line printed. No tracebacks. Then verify the attribute is populated by adding a tiny check (one-liner) — actually skip; the eval-side dump in Task B5 will exercise it.

**Step 4: Commit**

```powershell
git add lsy_drone_racing/control/rl_song/controller.py
git commit -m "feat(controller): surface raw policy mean via _last_policy_mean"
```

---

## Phase B — `eval_sim.py` trace dump (~2-3 hours)

Each task in this phase modifies `lsy_drone_racing/control/rl_song/eval_sim.py`. Mostly additive — the existing default behaviour stays unchanged when `--dump-trace` is not passed.

### Task B1: Add `--dump-trace` CLI flag and trace-writer scaffold

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/eval_sim.py`

**Step 1: Add CLI parameter to `simulate(...)`**

Insert into the `simulate` signature (after `control_mode`, before the closing paren around line 95):

```python
    dump_trace: str | None = None,
    reward_cfg: str | None = None,
```

Wire into the docstring's Parameters list:

```
dump_trace : str, optional
    If set, write per-step JSONL traces under this directory. Header
    row carries metadata; one row per env step. Default: no dump.
reward_cfg : str, optional
    Override path to ``reward_config.json``. Default: resolved
    relative to the checkpoint.
```

**Step 2: Add `TraceWriter` helper class**

Above `simulate(...)`:

```python
class _TraceWriter:
    """JSONL trace writer with per-episode files and a header row."""

    SCHEMA_VERSION = 1

    def __init__(self, dump_dir: Path, header_common: dict[str, Any]) -> None:
        self.dump_dir = dump_dir
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self._header_common = header_common
        self._fh: Any = None
        self._episode_idx = -1

    def open_episode(self, episode_idx: int, episode_header: dict[str, Any]) -> None:
        self._episode_idx = episode_idx
        path = self.dump_dir / f"episode_{episode_idx:03d}.jsonl"
        self._fh = path.open("w", encoding="utf-8", newline="\n")
        header = {"_header": True, **self._header_common, **episode_header,
                  "schema_version": self.SCHEMA_VERSION}
        self._fh.write(json.dumps(header, separators=(",", ":")) + "\n")

    def write_row(self, row: dict[str, Any]) -> None:
        self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    def close_episode(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write_run_meta(self, run_meta: dict[str, Any]) -> None:
        path = self.dump_dir / "run_meta.json"
        path.write_text(json.dumps(run_meta, indent=2, sort_keys=True))
```

Add the necessary imports at the top of the file (`import json`, `from typing import Any` already exists, `from pathlib import Path` already exists).

**Step 3: Construct it in `simulate(...)` when `dump_trace` is set**

After `_color_code_gates(env)` (around line 167) and before `env = JaxToNumpy(env)`:

```python
    trace_writer: _TraceWriter | None = None
    if dump_trace is not None:
        dump_dir = Path(dump_trace)
        if not dump_dir.is_absolute():
            dump_dir = Path(__file__).resolve().parents[3] / dump_dir
        trace_writer = _TraceWriter(
            dump_dir=dump_dir,
            header_common={
                "config": config,
                "control_mode": cfg.env.control_mode,
                "freq": int(cfg.env.freq),
                "checkpoint": str(checkpoint) if checkpoint else None,
                "n_gates": len(cfg.env.track.gates),
                "n_obstacles": len(cfg.env.track.obstacles),
            },
        )
        trace_writer.write_run_meta({
            "checkpoint": str(checkpoint) if checkpoint else None,
            "config": config,
            "control_mode": cfg.env.control_mode,
            "n_runs": n_runs,
            "seed": cfg.env.seed,
            "schema_version": _TraceWriter.SCHEMA_VERSION,
            "git_sha": _get_git_sha(),
        })
```

Add a tiny `_get_git_sha()` helper that runs `git rev-parse --short HEAD` via `subprocess.run` and returns `None` on failure. Keep it small — log nothing on failure.

**Step 4: Plumb `trace_writer` through to `_run_episode`**

Add `trace_writer: _TraceWriter | None = None` and `episode_idx: int = 0` parameters to `_run_episode`. Pass from the loop in `simulate`:

```python
        for episode_idx in range(n_runs):
            ep_time = _run_episode(
                env=env,
                controller_cls=controller_cls,
                cfg=cfg,
                video_writer=video_writer,
                camera=camera,
                width=width,
                height=height,
                trace_writer=trace_writer,
                episode_idx=episode_idx,
            )
            ep_times.append(ep_time)
```

In `_run_episode`, after `obs, info = env.reset()`, call `trace_writer.open_episode(...)` if non-None with `{"spawn_pos": obs["pos"].tolist(), "spawn_quat": obs["quat"].tolist()}`. Before `return curr_time ...`, call `trace_writer.close_episode()`.

**Step 5: Smoke validation**

```powershell
pixi run -e rl-train python -m lsy_drone_racing.control.rl_song.eval_sim --config level3.toml --checkpoint lsy_drone_racing/control/rl_song/checkpoints/level3_v33b_warmstart_from_v32a_seed0_300M/ --control_mode attitude --n_runs 2 --dump_trace renders/plan-B1-smoke/trace
```

Expected: `renders/plan-B1-smoke/trace/run_meta.json` exists, plus `episode_000.jsonl` and `episode_001.jsonl` each containing exactly one (header-only) line so far.

**Step 6: Commit**

```powershell
git add lsy_drone_racing/control/rl_song/eval_sim.py
git commit -m "feat(eval_sim): scaffold --dump-trace flag and JSONL writer"
```

---

### Task B2: Fix `curr_time`, stash `prev_obs`, break on finish

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/eval_sim.py:193-233` (`_run_episode`)

**Step 1: Refactor the loop body**

Replace the `while True:` body (currently lines 209-228) with:

```python
    prev_obs = obs
    while True:
        action = controller.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        i += 1
        curr_time = i / cfg.env.freq  # post-step instant; matches trace row's t

        prev_tg = int(prev_obs["target_gate"])
        obs_tg = int(obs["target_gate"])
        gate_just_passed = (prev_tg >= 0) and (obs_tg != prev_tg)
        finished = (obs_tg == -1) and (prev_tg != -1)

        controller_finished = controller.step_callback(
            action, obs, reward, terminated, truncated, info
        )

        # ... (trace write hook goes here, added in B3-B6) ...
        # ... (video frame grab stays where it is) ...

        if video_writer is not None:
            frame = _grab_offscreen_frame(env, camera, width, height)
            if frame is not None:
                video_writer.append_data(frame)
        elif cfg.sim.render:
            if ((i * fps_live_view) % cfg.env.freq) < fps_live_view:
                controller.render_callback(env.unwrapped.sim)
                env.render()

        if terminated or truncated or controller_finished or finished:
            break
        prev_obs = obs
```

Note the changes from the original:
- `i += 1` and `curr_time = i / freq` happen *after* `env.step` (post-step instant).
- `prev_obs` is stashed once before the loop and refreshed at the bottom of each non-terminal iteration.
- `finished` is OR'd into the break condition.

The pre-loop `i = 0` and `curr_time = 0.0` stay; remove the in-loop `curr_time = i / cfg.env.freq` from the *start* of the body.

**Step 2: Smoke validation**

Re-run the B1 command. Expected: still works, "Flight time" log matches what it printed before (within ±1 step due to the post-step shift — that's expected). No tracebacks.

**Step 3: Commit**

```powershell
git add lsy_drone_racing/control/rl_song/eval_sim.py
git commit -m "fix(eval_sim): post-step curr_time, prev_obs stash, finish-break"
```

---

### Task B3: Per-row kinematics, actions, target_gate, termination

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/eval_sim.py:_run_episode` (inside the loop, where the comment placeholder is)

**Step 1: Add `_build_trace_row` helper**

Above `_run_episode`:

```python
def _build_trace_row(
    *,
    i: int,
    t: float,
    obs: dict,
    action_applied: np.ndarray,
    action_policy_mean: np.ndarray | None,
    true_gates_pos: np.ndarray,
    true_gates_quat: np.ndarray,
    true_obstacles_pos: np.ndarray,
    reward_total: float | None,
    reward_terms: dict[str, float] | None,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    """Build one JSONL row. All ndarray values are tolist()'d for json."""
    return {
        "step": i, "t": float(t),
        "pos": obs["pos"].tolist(),
        "vel": obs["vel"].tolist(),
        "quat": obs["quat"].tolist(),
        "ang_vel": obs["ang_vel"].tolist(),
        "action_policy_mean": (
            action_policy_mean.tolist() if action_policy_mean is not None else None
        ),
        "action_applied": action_applied.tolist(),
        "target_gate": int(obs["target_gate"]),
        "gates_pos_true":     true_gates_pos.tolist(),
        "gates_quat_true":    true_gates_quat.tolist(),
        "obstacles_pos_true": true_obstacles_pos.tolist(),
        "gates_pos":     obs["gates_pos"].tolist(),
        "gates_quat":    obs["gates_quat"].tolist(),
        "obstacles_pos": obs["obstacles_pos"].tolist(),
        "gates_visited":     obs["gates_visited"].tolist(),
        "obstacles_visited": obs["obstacles_visited"].tolist(),
        "reward_total": reward_total,
        "reward_terms": reward_terms,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
```

**Step 2: Call it from the loop**

In `_run_episode`, at the trace-write hook position from B2 (after `controller.step_callback(...)`):

```python
        if trace_writer is not None:
            row = _build_trace_row(
                i=i, t=curr_time,
                obs=obs,
                action_applied=np.asarray(action, dtype=np.float32),
                action_policy_mean=getattr(controller, "_last_policy_mean", None),
                true_gates_pos=...,        # filled in B4
                true_gates_quat=...,
                true_obstacles_pos=...,
                reward_total=None,         # filled in B5
                reward_terms=None,
                terminated=terminated,
                truncated=truncated,
            )
            trace_writer.write_row(row)
```

For this commit, use placeholders (`np.zeros((n_gates, 3))`, etc.) for the true poses and `None` for reward. The next two tasks fill them in.

**Step 3: Smoke validation**

Re-run B1's command. Expected: episode JSONL files have one header row + N data rows. Inspect the last row of `episode_000.jsonl` and confirm `pos`, `vel`, `action_applied`, `target_gate` are reasonable values (drone in flight pose, target_gate either 0..n_gates or -1).

**Step 4: Commit**

```powershell
git add lsy_drone_racing/control/rl_song/eval_sim.py
git commit -m "feat(eval_sim): per-row kinematics and action logging"
```

---

### Task B4: True-pose readout from `env.unwrapped.data`

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/eval_sim.py:_run_episode`

**Step 1: Read true poses each step**

Replace the placeholders in the trace-write block with:

```python
            sim_data = env.unwrapped.data
            true_gates_pos    = np.asarray(sim_data.gates_pos)[0]      # squeeze n_envs
            true_gates_quat   = np.asarray(sim_data.gates_quat)[0]
            true_obstacles_pos = np.asarray(sim_data.obstacles_pos)[0]
```

The leading `n_envs` axis is squeezed because `DroneRaceEnv` is non-vec from the caller's perspective; the underlying data is `(1, n_gates, 3)`, so `[0]` gives `(n_gates, 3)`.

**Step 2: Pass them to the row builder**

```python
                true_gates_pos=true_gates_pos,
                true_gates_quat=true_gates_quat,
                true_obstacles_pos=true_obstacles_pos,
```

**Step 3: Smoke validation**

Re-run B1's command on level 3 (where true ≠ policy-view on the unvisited branch).

```powershell
$row = (Get-Content renders/plan-B1-smoke/trace/episode_000.jsonl)[1] | ConvertFrom-Json
$row.gates_pos_true
$row.gates_pos
```

Expected: For an early step on level 3, `gates_pos_true` and `gates_pos` should disagree for gates whose `gates_visited` is `False` (those still show the nominal/randomization-snapshot pose in the policy view).

**Step 4: Commit**

```powershell
git add lsy_drone_racing/control/rl_song/eval_sim.py
git commit -m "feat(eval_sim): log true vs policy-view gate and obstacle poses"
```

---

### Task B5: Reward config load + per-step reward terms

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/eval_sim.py`

**Step 1: Add reward-config loader**

Above `simulate(...)`:

```python
def _load_reward_config(
    checkpoint: Path | None, override: str | None
) -> "RewardConfig | None":
    """Load RewardConfig JSON. Returns None if neither path exists.

    Resolution order:
      1. ``override`` (CLI ``--reward-cfg``), if given.
      2. ``checkpoint/../reward_config.json`` for a step_NNN checkpoint dir.
      3. ``checkpoint/reward_config.json`` for a run-dir checkpoint.
    """
    from lsy_drone_racing.control.rl_song.config import RewardConfig
    candidates: list[Path] = []
    if override is not None:
        candidates.append(Path(override))
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        candidates.append(ckpt_path.parent / "reward_config.json")
        candidates.append(ckpt_path / "reward_config.json")
    for c in candidates:
        if c.exists():
            data = json.loads(c.read_text())
            return RewardConfig(**data), c
    return None, None
```

**Step 2: Load in `simulate(...)`**

After resolving `checkpoint`:

```python
    reward_cfg_obj, reward_cfg_path = _load_reward_config(
        cfg.controller.get("checkpoint", None), reward_cfg
    )
    if reward_cfg_obj is None and dump_trace is not None:
        logger.warning(
            "No reward_config.json found near %s and --reward-cfg not set; "
            "trace will record null reward fields.",
            cfg.controller.get("checkpoint", None),
        )
```

Add `"reward_cfg_path": str(reward_cfg_path) if reward_cfg_path else None` to the `run_meta` dict.

**Step 3: Wrap the reward call**

Above `_run_episode`:

```python
def _compute_reward_terms(
    *,
    prev_obs: dict,
    obs: dict,
    terminated: bool,
    truncated: bool,
    finished: bool,
    gate_just_passed: bool,
    reward_cfg_obj: "RewardConfig",
    true_gates_pos: np.ndarray,
    true_gates_quat: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Call ``step_reward`` with an n_envs=1 leading axis on obs fields only.

    True poses already have the env axis (we read them from
    ``env.unwrapped.data``) — do not re-add it.
    """
    from lsy_drone_racing.control.rl_song.reward import step_reward

    def _batched(d: dict) -> dict:
        return {k: jnp.asarray(v)[None] for k, v in d.items()}

    total, components = step_reward(
        _batched(obs),
        _batched(prev_obs),
        jnp.asarray([terminated]),
        jnp.asarray([truncated]),
        jnp.asarray([finished]),
        jnp.asarray([gate_just_passed]),
        reward_cfg_obj,
        true_gates_pos=jnp.asarray(true_gates_pos)[None],
        true_gates_quat=jnp.asarray(true_gates_quat)[None],
    )
    total_f = float(np.asarray(total).squeeze())
    components_f = {k: float(np.asarray(v).squeeze()) for k, v in components.items()}
    return total_f, components_f
```

Note `true_obstacles_pos` is intentionally not passed — `step_reward` ignores it (codex finding #3, design doc §"What the reward fn actually uses").

**Step 4: Wire into the loop**

In `_run_episode`, after computing `finished` and `gate_just_passed`:

```python
        if trace_writer is not None and reward_cfg_obj is not None:
            reward_total, reward_terms = _compute_reward_terms(
                prev_obs=prev_obs, obs=obs,
                terminated=terminated, truncated=truncated,
                finished=finished, gate_just_passed=gate_just_passed,
                reward_cfg_obj=reward_cfg_obj,
                true_gates_pos=true_gates_pos,
                true_gates_quat=true_gates_quat,
            )
        else:
            reward_total, reward_terms = None, None
```

Pass `reward_cfg_obj` through `_run_episode`'s signature too.

**Step 5: Smoke validation against `level3_v33b_warmstart_from_v32a_seed0_300M`**

This checkpoint pre-dates A1 so won't have `reward_config.json`. Test both back-compat and the explicit override:

```powershell
# Back-compat: warning + null reward fields
pixi run -e rl-train python -m lsy_drone_racing.control.rl_song.eval_sim --config level3.toml --checkpoint lsy_drone_racing/control/rl_song/checkpoints/level3_v33b_warmstart_from_v32a_seed0_300M/ --control_mode attitude --n_runs 1 --dump_trace renders/plan-B5-smoke-nocfg/trace
```

Expected: warning printed, `episode_000.jsonl` rows have `"reward_total": null` and `"reward_terms": null`.

For the override path: manually write a `reward_config.json` somewhere (copy from a recent training run, or use `lsy_drone_racing/control/rl_song/config.py`'s `RewardConfig` defaults written via `python -c "import json; from dataclasses import asdict; from lsy_drone_racing.control.rl_song.config import RewardConfig; print(json.dumps(asdict(RewardConfig()), indent=2))" > /tmp/rcfg.json`):

```powershell
pixi run -e rl-train python -m lsy_drone_racing.control.rl_song.eval_sim --config level3.toml --checkpoint lsy_drone_racing/control/rl_song/checkpoints/level3_v33b_warmstart_from_v32a_seed0_300M/ --control_mode attitude --n_runs 1 --dump_trace renders/plan-B5-smoke-cfg/trace --reward_cfg /tmp/rcfg.json
```

Expected: no warning; rows have populated `"reward_total"` and `"reward_terms"` with non-NaN floats; `"r_prog"`, `"r_gate_bonus"`, etc. all present.

**Step 6: Commit**

```powershell
git add lsy_drone_racing/control/rl_song/eval_sim.py
git commit -m "feat(eval_sim): per-step reward terms with reward_config back-compat"
```

---

### Task B6: End-to-end eval-side smoke

**Files:** none (validation only)

**Step 1: Train a tiny run that has `reward_config.json` so we exercise the happy path end-to-end**

```powershell
pixi run -e rl-train python -m lsy_drone_racing.control.rl_song.train --run-name plan-B6-smoke --total-steps 50000 --init-from lsy_drone_racing/control/rl_song/checkpoints/level3_v33b_warmstart_from_v32a_seed0_300M
```

(50k steps is enough to produce a checkpoint; we just need the file.)

**Step 2: Run eval with full trace**

```powershell
pixi run -e rl-train python -m lsy_drone_racing.control.rl_song.eval_sim --config level3.toml --checkpoint lsy_drone_racing/control/rl_song/checkpoints/plan-B6-smoke/ --control_mode attitude --n_runs 4 --dump_trace renders/plan-B6-smoke/trace --record renders/plan-B6-smoke/eval.mp4
```

Expected:
- `renders/plan-B6-smoke/trace/run_meta.json` has `reward_cfg_path` populated.
- 4 episode JSONLs, each starting with `_header: true`.
- Per-row schema matches the design doc: `step`, `t`, `pos`, `vel`, `quat`, `ang_vel`, `action_policy_mean` (7-vec), `action_applied` (4-vec), `target_gate`, `gates_pos_true`, `gates_pos`, `gates_visited`, `reward_total`, `reward_terms` (10 keys).
- Video renders (the mp4 path still works, no regression).

**Step 3: Commit (just the smoke artifact — optional, can skip)**

No commit needed; this is validation only. Clean up: `Remove-Item -Recurse renders/plan-B6-smoke` and `Remove-Item -Recurse lsy_drone_racing/control/rl_song/checkpoints/plan-B6-smoke` (we don't need the tiny checkpoint).

---

## Phase C — Analyzer (~3-4 hours)

All tasks create / modify `scripts/analyze_eval_traces.py`. Single self-contained script — no new package.

### Task C1: CLI + trace loader scaffold

**Files:**
- Create: `scripts/analyze_eval_traces.py`

**Step 1: Scaffold**

```python
"""Offline analyzer for eval-trace dumps produced by ``eval_sim --dump-trace``.

Reads ``trace/episode_NNN.jsonl`` files and writes
``analysis/episode_NNN.summary.json`` plus ``analysis/run_summary.json``.

Usage
-----
    pixi run -e rl-train python scripts/analyze_eval_traces.py <trace_dir>

``<trace_dir>`` is the directory containing ``run_meta.json`` and the
per-episode JSONL files. Output is written to ``<trace_dir>/../analysis/``.

See ``docs/plans/2026-05-18-eval-trace-tooling-design.md`` for the
schema and event taxonomy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
import numpy as np

# Detector thresholds (module-level so they're easy to tune).
HOVER_WINDOW_STEPS: int = 20         # 0.4 s at 50 Hz
HOVER_XY_BBOX_M: float = 0.15
NEAR_MISS_DIST_M: float = 0.20
WOBBLE_ANG_VEL_RAD_S: float = 6.0
WOBBLE_MIN_DURATION_STEPS: int = 10  # 0.2 s
TAKEOFF_Z_M: float = 0.10
FLOOR_Z_M: float = 0.05
COLLISION_RECENT_WINDOW: int = 5     # frames pre-terminal


@dataclass(frozen=True)
class Episode:
    header: dict[str, Any]
    rows: list[dict[str, Any]]
    freq: float
```

**Step 2: Add the loader**

```python
def load_episode(jsonl_path: Path) -> Episode:
    """Read a per-episode JSONL into header + row list. Header is line 0."""
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty trace file: {jsonl_path}")
    header = json.loads(lines[0])
    if not header.get("_header", False):
        raise ValueError(f"Missing header row in {jsonl_path}")
    if header.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported schema_version={header.get('schema_version')} in {jsonl_path}"
        )
    rows = [json.loads(line) for line in lines[1:]]
    return Episode(header=header, rows=rows, freq=float(header["freq"]))


def load_run_meta(trace_dir: Path) -> dict[str, Any]:
    return json.loads((trace_dir / "run_meta.json").read_text(encoding="utf-8"))
```

**Step 3: Add the main entrypoint**

```python
def analyze(trace_dir: str) -> None:
    """Analyze a trace directory and emit summary JSONs."""
    trace = Path(trace_dir).resolve()
    if not trace.is_dir():
        raise FileNotFoundError(f"Not a directory: {trace}")
    analysis = trace.parent / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    run_meta = load_run_meta(trace)
    episode_paths = sorted(trace.glob("episode_*.jsonl"))
    episodes: list[tuple[int, Episode]] = []
    for path in episode_paths:
        idx = int(path.stem.removeprefix("episode_"))
        episodes.append((idx, load_episode(path)))

    print(f"Loaded {len(episodes)} episodes from {trace}")
    # Per-episode + rollup writers added in subsequent tasks.


if __name__ == "__main__":
    fire.Fire(analyze)
```

**Step 4: Smoke validation**

```powershell
pixi run -e rl-train python scripts/analyze_eval_traces.py renders/plan-B6-smoke/trace
```

Expected: prints "Loaded 4 episodes from ..."; no error.

**Step 5: Commit**

```powershell
git add scripts/analyze_eval_traces.py
git commit -m "feat(analyze): scaffold trace loader and CLI"
```

---

### Task C2: Outcome detection (gates_passed, finished, terminal_cause)

**Files:** Modify `scripts/analyze_eval_traces.py`.

**Step 1: Add the outcome function**

```python
def detect_outcome(ep: Episode) -> dict[str, Any]:
    """Compute ``outcome`` block: gates_passed, finished, terminal_cause."""
    n_gates = ep.header["n_gates"]
    rows = ep.rows
    last = rows[-1]
    prev_targets = [r["target_gate"] for r in rows]
    obs_targets = prev_targets[1:] + [last["target_gate"]]  # alignment doesn't matter here

    # Finished iff target_gate ever became -1.
    finished = any(t == -1 for t in prev_targets)
    if finished:
        # gates_passed = N when finished.
        gates_passed = n_gates
    else:
        # gates_passed = max target_gate ever reached.
        gates_passed = max(t for t in prev_targets if t >= 0) if prev_targets else 0

    if finished:
        terminal_cause = "finished"
    elif last["truncated"]:
        terminal_cause = "truncated"
    else:
        # Collision — object resolution comes in C6; placeholder for now.
        terminal_cause = "collision:unknown"

    return {
        "gates_passed": int(gates_passed),
        "finished": bool(finished),
        "ep_len_steps": len(rows),
        "flight_time_s": float(rows[-1]["t"]),
        "terminal_cause": terminal_cause,
    }
```

**Step 2: Smoke validation**

Temporarily wire it into `analyze()` (one episode) and print:

```python
    for idx, ep in episodes:
        print(idx, detect_outcome(ep))
```

Expected: gates_passed in [0, n_gates], flight_time matches the value `_log_episode_stats` printed during eval. Remove the print after verifying.

**Step 3: Commit**

```powershell
git add scripts/analyze_eval_traces.py
git commit -m "feat(analyze): outcome detection (gates/finished/cause)"
```

---

### Task C3: Takeoff + gate-pass events

**Files:** Modify `scripts/analyze_eval_traces.py`.

**Step 1: Helpers**

```python
def _quat_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Pure-numpy xyzw quat → 3x3 rotation matrix. Mirrors obs._quat_to_matrix."""
    x, y, z, w = quat_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)],
    ])


def _rows_pos(rows: list[dict]) -> np.ndarray:
    return np.array([r["pos"] for r in rows], dtype=np.float64)


def _rows_vel(rows: list[dict]) -> np.ndarray:
    return np.array([r["vel"] for r in rows], dtype=np.float64)
```

**Step 2: Detectors**

```python
def detect_takeoff(ep: Episode) -> dict[str, Any] | None:
    """First frame where pos.z exceeds TAKEOFF_Z_M."""
    pos = _rows_pos(ep.rows)
    above = np.where(pos[:, 2] > TAKEOFF_Z_M)[0]
    if above.size == 0:
        return None
    i = int(above[0])
    return {
        "type": "takeoff",
        "t": float(ep.rows[i]["t"]),
        "vz_at_liftoff": float(ep.rows[i]["vel"][2]),
    }


def detect_gate_passes(ep: Episode) -> list[dict[str, Any]]:
    """One ``gate_pass`` event per target_gate advance."""
    events = []
    for i in range(1, len(ep.rows)):
        prev_tg = ep.rows[i - 1]["target_gate"]
        curr_tg = ep.rows[i]["target_gate"]
        if prev_tg < 0 or curr_tg == prev_tg:
            continue
        # Either advanced to next gate or transitioned to -1 (finished).
        passed_idx = prev_tg  # the gate we just went through
        row = ep.rows[i]
        gate_pos = np.asarray(row["gates_pos_true"][passed_idx])
        gate_quat = np.asarray(row["gates_quat_true"][passed_idx])
        rot_gw = _quat_to_rotmat(gate_quat)
        # Local frame: x = forward through gate, y/z = aperture.
        rel = np.asarray(row["pos"]) - gate_pos
        local = rot_gw.T @ rel
        offset = float(np.linalg.norm(local[1:]))  # (y, z) magnitude
        vel = np.asarray(row["vel"])
        speed = float(np.linalg.norm(vel))
        forward = rot_gw[:, 0]  # gate forward axis (world frame)
        cos_angle = float(np.clip(vel @ forward / max(speed, 1e-9), -1.0, 1.0))
        events.append({
            "type": "gate_pass",
            "t": float(row["t"]),
            "gate": int(passed_idx),
            "speed": speed,
            "in_plane_offset_m": offset,
            "angle_off_normal_rad": float(np.arccos(abs(cos_angle))),
        })
    return events
```

**Step 3: Smoke validation**

Wire into `analyze()`:

```python
    for idx, ep in episodes:
        takeoff = detect_takeoff(ep)
        passes = detect_gate_passes(ep)
        print(idx, "takeoff:", takeoff, "passes:", len(passes))
```

Expected: each non-trivial episode has a takeoff with t < 0.5s and 0-4 gate-pass events. For a finishing episode, len(passes) == n_gates.

**Step 4: Commit**

```powershell
git add scripts/analyze_eval_traces.py
git commit -m "feat(analyze): takeoff and gate-pass event detection"
```

---

### Task C4: Hover detector (xy-bbox sliding window)

**Files:** Modify `scripts/analyze_eval_traces.py`.

**Step 1: Detector**

```python
def detect_hovers(ep: Episode) -> list[dict[str, Any]]:
    """Detect xy-stationary windows. Coalesced into one event per run."""
    pos = _rows_pos(ep.rows)
    n = len(pos)
    if n < HOVER_WINDOW_STEPS:
        return []
    is_hover = np.zeros(n, dtype=bool)
    for i in range(HOVER_WINDOW_STEPS - 1, n):
        window = pos[i - HOVER_WINDOW_STEPS + 1 : i + 1, :2]
        extent = window.max(axis=0) - window.min(axis=0)
        if extent.max() < HOVER_XY_BBOX_M:
            is_hover[i - HOVER_WINDOW_STEPS + 1 : i + 1] = True

    events = []
    in_run = False
    start = 0
    for i in range(n):
        if is_hover[i] and not in_run:
            in_run = True
            start = i
        elif not is_hover[i] and in_run:
            in_run = False
            events.append(_hover_event(ep, start, i - 1))
    if in_run:
        events.append(_hover_event(ep, start, n - 1))
    return events


def _hover_event(ep: Episode, i_start: int, i_end: int) -> dict[str, Any]:
    rows = ep.rows[i_start : i_end + 1]
    xy = np.array([r["pos"][:2] for r in rows])
    mean_pos = np.array([r["pos"] for r in rows]).mean(axis=0)
    # Nearest gate by centroid at the midpoint frame.
    mid = rows[len(rows) // 2]
    gates = np.asarray(mid["gates_pos_true"])
    distances = np.linalg.norm(gates - np.asarray(mid["pos"]), axis=-1)
    near_gate = int(distances.argmin())
    return {
        "type": "hover",
        "t_start": float(rows[0]["t"]),
        "t_end": float(rows[-1]["t"]),
        "duration_s": float(rows[-1]["t"] - rows[0]["t"]),
        "xy_bbox_extent_m": float((xy.max(axis=0) - xy.min(axis=0)).max()),
        "mean_pos": mean_pos.tolist(),
        "near_gate": near_gate,
    }
```

**Step 2: Smoke validation**

```python
    for idx, ep in episodes:
        hovers = detect_hovers(ep)
        print(idx, "hovers:", len(hovers), [h["duration_s"] for h in hovers])
```

Expected: 0-3 hover events per episode; the hover-after-gate-1 pattern from the level-2 deterministic-crash memory should show up as 1 hover event of ~0.5-1s duration if that checkpoint exhibits it.

**Step 3: Commit**

```powershell
git add scripts/analyze_eval_traces.py
git commit -m "feat(analyze): hover detection via xy-bbox sliding window"
```

---

### Task C5: Near-miss detector

**Files:** Modify `scripts/analyze_eval_traces.py`.

**Step 1: Port the gate-frame edge distance from `reward.py`**

```python
# Same gate-frame corner layout as obs._GATE_CORNERS_LOCAL / reward._GATE_FRAME_CORNERS_LOCAL.
# Source: lsy_drone_racing/control/rl_song/{obs,reward}.py.
_GATE_HALF_Y = 0.225  # GATE_HALF_SIZE_M[0]
_GATE_HALF_Z = 0.225  # GATE_HALF_SIZE_M[1]
# Confirm against obs.GATE_HALF_SIZE_M at implementation time and update if different.
_GATE_FRAME_CORNERS_LOCAL = np.array([
    [0.0, +_GATE_HALF_Y, +_GATE_HALF_Z],
    [0.0, +_GATE_HALF_Y, -_GATE_HALF_Z],
    [0.0, -_GATE_HALF_Y, +_GATE_HALF_Z],
    [0.0, -_GATE_HALF_Y, -_GATE_HALF_Z],
])
_GATE_FRAME_EDGES = [(0, 1), (2, 3), (0, 2), (1, 3)]


def _gate_frame_edge_dist(pos: np.ndarray, gate_pos: np.ndarray, gate_quat: np.ndarray) -> float:
    """Min distance from pos to any of the 4 gate-frame edges."""
    rot = _quat_to_rotmat(gate_quat)
    corners = (rot @ _GATE_FRAME_CORNERS_LOCAL.T).T + gate_pos  # (4, 3)
    best = np.inf
    for a, b in _GATE_FRAME_EDGES:
        ab = corners[b] - corners[a]
        ap = pos - corners[a]
        ab_sq = ab @ ab
        t = float(np.clip((ap @ ab) / max(ab_sq, 1e-12), 0.0, 1.0))
        closest = corners[a] + t * ab
        d = float(np.linalg.norm(pos - closest))
        if d < best:
            best = d
    return best
```

**Important:** at implementation time, double-check `GATE_HALF_Y` / `GATE_HALF_Z` against `obs.GATE_HALF_SIZE_M` — they may not both be 0.225.

**Step 2: Detector**

```python
def detect_near_misses(ep: Episode) -> list[dict[str, Any]]:
    """Closest gate-frame approach drops below threshold without target_gate advancing."""
    events = []
    n = len(ep.rows)
    n_gates = ep.header["n_gates"]
    # For each gate, find the row at which approach was minimum and whether it advanced.
    # Walk forward; once target_gate advances past gate g, stop tracking g.
    seen_pass = set()
    for i in range(n):
        row = ep.rows[i]
        tg = row["target_gate"]
        if tg < 0:
            break
        if tg in seen_pass:
            continue
        gate_pos = np.asarray(row["gates_pos_true"][tg])
        gate_quat = np.asarray(row["gates_quat_true"][tg])
        d = _gate_frame_edge_dist(np.asarray(row["pos"]), gate_pos, gate_quat)
        if d < NEAR_MISS_DIST_M:
            # Check whether the gate is eventually passed.
            advanced = any(r["target_gate"] != tg for r in ep.rows[i + 1 :])
            if not advanced:
                events.append({
                    "type": "near_miss",
                    "t": float(row["t"]),
                    "gate": int(tg),
                    "closest_frame_dist_m": d,
                    "passed": False,
                })
                seen_pass.add(tg)
                break  # one near-miss per episode is enough; the rest of the trace is post-failure
    return events
```

**Step 3: Smoke validation**

```python
    for idx, ep in episodes:
        misses = detect_near_misses(ep)
        print(idx, "near_misses:", misses)
```

Expected: 0-1 per non-finishing episode; sometimes none even on collision (because the drone never got close to the gate frame).

**Step 4: Commit**

```powershell
git add scripts/analyze_eval_traces.py
git commit -m "feat(analyze): near-miss detection via gate-frame edge distance"
```

---

### Task C6: Collision detector + object resolution (capsule line segment)

**Files:** Modify `scripts/analyze_eval_traces.py`.

**Step 1: Helpers**

```python
def _point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    ap = p - a
    ab_sq = ab @ ab
    t = float(np.clip((ap @ ab) / max(ab_sq, 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _resolve_collision_object(
    pos: np.ndarray,
    gates_pos: np.ndarray,
    obstacles_top: np.ndarray,
) -> tuple[str, float]:
    """Return (object_label, distance). Obstacles are vertical capsules from
    (x, y, z_top) to (x, y, 0); gates use centroid; floor is z < FLOOR_Z_M.
    """
    candidates: list[tuple[float, str]] = []
    for i, g in enumerate(gates_pos):
        candidates.append((float(np.linalg.norm(pos - g)), f"gate:{i}"))
    for i, top in enumerate(obstacles_top):
        a = np.array([top[0], top[1], 0.0])
        candidates.append((_point_to_segment_dist(pos, top, a), f"obstacle:{i}"))
    if pos[2] < FLOOR_Z_M:
        candidates.append((float(pos[2]), "floor"))
    distance, label = min(candidates, key=lambda x: x[0])
    return label, distance
```

**Step 2: Detector**

```python
def detect_collision(ep: Episode, outcome: dict[str, Any]) -> dict[str, Any] | None:
    """Build a collision event when outcome.finished is False and not truncated."""
    if outcome["finished"] or outcome["terminal_cause"] == "truncated":
        return None
    rows = ep.rows
    if len(rows) < 2:
        return None
    # Use pos[T-1] (last pre-terminal frame; design §"Collision detector").
    i = len(rows) - 2
    row = rows[i]
    pos = np.asarray(row["pos"])
    gates_pos = np.asarray(row["gates_pos_true"])
    obstacles_top = np.asarray(row["obstacles_pos_true"])
    label, distance = _resolve_collision_object(pos, gates_pos, obstacles_top)

    # 5-frame robustness window: min approach distance to inferred object.
    start = max(0, i - COLLISION_RECENT_WINDOW + 1)
    distances = []
    for j in range(start, i + 1):
        rj = rows[j]
        if label.startswith("gate:"):
            idx = int(label.split(":")[1])
            d = float(np.linalg.norm(np.asarray(rj["pos"]) - np.asarray(rj["gates_pos_true"][idx])))
        elif label.startswith("obstacle:"):
            idx = int(label.split(":")[1])
            top = np.asarray(rj["obstacles_pos_true"][idx])
            a = np.array([top[0], top[1], 0.0])
            d = _point_to_segment_dist(np.asarray(rj["pos"]), top, a)
        else:  # floor
            d = float(rj["pos"][2])
        distances.append(d)
    min_d = float(min(distances))

    return {
        "type": "collision",
        "t": float(row["t"]),
        "object": label,
        "approach_speed_50hz": float(np.linalg.norm(np.asarray(row["vel"]))),
        "last_pos_50hz_pre_terminal": row["pos"],
        "min_approach_dist_5frame_m": min_d,
    }
```

**Step 3: Update `detect_outcome` to use the resolved label**

In `detect_outcome`, replace the placeholder `"collision:unknown"` with the object label by computing collision here too — or call `detect_collision` once and pass the label back. Cleaner: have `analyze()` call `detect_collision` first, then patch `outcome["terminal_cause"]` from the event's `object` field.

```python
        outcome = detect_outcome(ep)
        collision_evt = detect_collision(ep, outcome)
        if collision_evt is not None:
            outcome["terminal_cause"] = f"collision:{collision_evt['object']}"
```

**Step 4: Smoke validation**

```python
        print(idx, outcome["terminal_cause"], collision_evt)
```

Expected: collisions resolve to `collision:obstacle:N`, `collision:gate:N`, or `collision:floor`. No `collision:unknown` unless the trace is genuinely degenerate.

**Step 5: Commit**

```powershell
git add scripts/analyze_eval_traces.py
git commit -m "feat(analyze): collision detection with capsule-segment object resolution"
```

---

### Task C7: Wobble detector

**Files:** Modify `scripts/analyze_eval_traces.py`.

**Step 1: Detector**

```python
def detect_wobbles(ep: Episode) -> list[dict[str, Any]]:
    """|ang_vel| > threshold sustained for >= WOBBLE_MIN_DURATION_STEPS."""
    n = len(ep.rows)
    mag = np.array([np.linalg.norm(r["ang_vel"]) for r in ep.rows])
    high = mag > WOBBLE_ANG_VEL_RAD_S

    events = []
    in_run = False
    start = 0
    for i in range(n):
        if high[i] and not in_run:
            in_run = True
            start = i
        elif not high[i] and in_run:
            in_run = False
            if i - start >= WOBBLE_MIN_DURATION_STEPS:
                events.append(_wobble_event(ep, mag, start, i - 1))
    if in_run and n - start >= WOBBLE_MIN_DURATION_STEPS:
        events.append(_wobble_event(ep, mag, start, n - 1))
    return events


def _wobble_event(ep: Episode, mag: np.ndarray, i_start: int, i_end: int) -> dict[str, Any]:
    return {
        "type": "wobble",
        "t_start": float(ep.rows[i_start]["t"]),
        "t_end": float(ep.rows[i_end]["t"]),
        "duration_s": float(ep.rows[i_end]["t"] - ep.rows[i_start]["t"]),
        "max_ang_vel_rad_s": float(mag[i_start : i_end + 1].max()),
    }
```

**Step 2: Smoke + commit**

```powershell
git add scripts/analyze_eval_traces.py
git commit -m "feat(analyze): wobble detection on sustained ang_vel"
```

---

### Task C8: Reward integration

**Files:** Modify `scripts/analyze_eval_traces.py`.

**Step 1: Add the integrator**

```python
def integrate_reward(ep: Episode) -> dict[str, Any] | None:
    """Sum reward terms across the episode. Returns None if any row has null terms."""
    rows = ep.rows
    if any(r["reward_terms"] is None for r in rows):
        return None
    keys = list(rows[0]["reward_terms"].keys())
    sums = {k: 0.0 for k in keys}
    for r in rows:
        for k in keys:
            sums[k] += float(r["reward_terms"][k])
    total = sum(r["reward_total"] for r in rows)
    positives = {k: v for k, v in sums.items() if v > 0}
    negatives = {k: v for k, v in sums.items() if v < 0}
    return {
        "total": float(total),
        "by_term": sums,
        "dominant_positive": max(positives, key=positives.get) if positives else None,
        "dominant_negative": min(negatives, key=negatives.get) if negatives else None,
    }
```

**Step 2: Smoke + commit**

```powershell
git add scripts/analyze_eval_traces.py
git commit -m "feat(analyze): per-episode reward integration"
```

---

### Task C9: Per-episode summary writer + headline

**Files:** Modify `scripts/analyze_eval_traces.py`.

**Step 1: Headline builder**

```python
_ANOMALY_PRIORITY = ["collision", "near_miss", "hover", "wobble"]


def build_headline(outcome: dict, events: list[dict], n_gates: int) -> str:
    by_type: dict[str, list[dict]] = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)
    primary = None
    for kind in _ANOMALY_PRIORITY:
        if kind in by_type:
            primary = by_type[kind][0]
            break

    parts = [f"Passed {outcome['gates_passed']}/{n_gates} gates"]
    if primary is not None:
        if primary["type"] == "hover":
            parts.append(f"hovered {primary['duration_s']:.2f}s near gate {primary['near_gate']}")
        elif primary["type"] == "near_miss":
            parts.append(f"near-miss gate {primary['gate']} at {primary['closest_frame_dist_m']:.2f}m")
        elif primary["type"] == "collision":
            parts.append(
                f"crashed {primary['object']} at t≈{primary['t']:.2f}s "
                f"@ ≈{primary['approach_speed_50hz']:.1f} m/s"
            )
        elif primary["type"] == "wobble":
            parts.append(f"wobbled {primary['duration_s']:.2f}s "
                         f"(peak {primary['max_ang_vel_rad_s']:.1f} rad/s)")
    if outcome["finished"]:
        parts.append(f"finished in {outcome['flight_time_s']:.2f}s")
    return "; ".join(parts)
```

**Step 2: Per-episode summarizer**

```python
def summarize_episode(ep: Episode) -> dict[str, Any]:
    outcome = detect_outcome(ep)
    events: list[dict] = []
    if (e := detect_takeoff(ep)) is not None:
        events.append(e)
    events.extend(detect_gate_passes(ep))
    events.extend(detect_hovers(ep))
    events.extend(detect_near_misses(ep))
    if (e := detect_collision(ep, outcome)) is not None:
        events.append(e)
        outcome["terminal_cause"] = f"collision:{e['object']}"
    events.extend(detect_wobbles(ep))
    events.sort(key=lambda x: x.get("t", x.get("t_start", 0.0)))

    pos = _rows_pos(ep.rows)
    vel = _rows_vel(ep.rows)
    speeds = np.linalg.norm(vel, axis=-1)
    ang_vel_mags = np.array([np.linalg.norm(r["ang_vel"]) for r in ep.rows])

    return {
        "outcome": outcome,
        "spawn": {"pos": ep.header["spawn_pos"], "quat": ep.header["spawn_quat"]},
        "events": events,
        "kinematics_metrics": {
            "max_speed": float(speeds.max()),
            "mean_speed": float(speeds.mean()),
            "max_ang_vel": float(ang_vel_mags.max()),
            "path_length_m": float(np.linalg.norm(np.diff(pos, axis=0), axis=-1).sum()),
        },
        "reward_integrated": integrate_reward(ep),
        "headline": build_headline(outcome, events, ep.header["n_gates"]),
    }
```

**Step 3: Wire into `analyze()` to write the JSON**

```python
    summaries = {}
    for idx, ep in episodes:
        s = summarize_episode(ep)
        s["episode"] = idx
        (analysis / f"episode_{idx:03d}.summary.json").write_text(
            json.dumps(s, indent=2)
        )
        summaries[idx] = s
    print(f"Wrote {len(summaries)} episode summaries")
```

**Step 4: Smoke validation**

```powershell
pixi run -e rl-train python scripts/analyze_eval_traces.py renders/plan-B6-smoke/trace
```

Expected: `renders/plan-B6-smoke/analysis/episode_NNN.summary.json` files exist; cat one and confirm:
- `outcome.terminal_cause` is `finished` / `truncated` / `collision:obstacle:N` / `collision:gate:N` / `collision:floor`.
- `events` array sorted by time, with a `takeoff` first.
- `headline` is a single sensible line.

**Step 5: Commit**

```powershell
git add scripts/analyze_eval_traces.py
git commit -m "feat(analyze): per-episode summary writer with headline"
```

---

### Task C10: Run rollup (`run_summary.json`)

**Files:** Modify `scripts/analyze_eval_traces.py`.

**Step 1: Add aggregation helpers**

```python
SPAWN_GRID_X = [(-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5)]
SPAWN_GRID_Y = [(-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5)]


def _spawn_bucket(spawn: dict) -> str:
    x, y = spawn["pos"][0], spawn["pos"][1]
    bx = next((f"x∈[{lo},{hi}]" for lo, hi in SPAWN_GRID_X if lo <= x < hi), "x∈out")
    by = next((f"y∈[{lo},{hi}]" for lo, hi in SPAWN_GRID_Y if lo <= y < hi), "y∈out")
    return f"{bx}, {by}"


def _hist(values: list, keys: list | None = None) -> dict:
    out: dict = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def build_run_summary(
    run_meta: dict, summaries: dict[int, dict], n_gates: int
) -> dict[str, Any]:
    eps = list(summaries.values())
    if not eps:
        return {"checkpoint": run_meta.get("checkpoint"), "n_episodes": 0}

    gates_passed = [e["outcome"]["gates_passed"] for e in eps]
    ep_lens = [e["outcome"]["ep_len_steps"] for e in eps]
    times = [e["outcome"]["flight_time_s"] for e in eps]
    max_speeds = [e["kinematics_metrics"]["max_speed"] for e in eps]

    aggregate = {
        "finish_rate": float(np.mean([e["outcome"]["finished"] for e in eps])),
        "gates_passed": {
            "mean": float(np.mean(gates_passed)),
            "max": int(max(gates_passed)),
            "histogram": [int((np.array(gates_passed) == g).sum()) for g in range(n_gates + 1)],
        },
        "ep_len_steps": {"mean": float(np.mean(ep_lens)), "median": float(np.median(ep_lens)),
                         "min": int(min(ep_lens)), "max": int(max(ep_lens))},
        "flight_time_s": {"mean": float(np.mean(times)), "median": float(np.median(times))},
        "max_speed": {"mean": float(np.mean(max_speeds)), "max": float(max(max_speeds))},
    }

    terminal_causes = _hist([e["outcome"]["terminal_cause"] for e in eps])
    anomalies = _hist([
        kind
        for e in eps
        for kind in {ev["type"] for ev in e["events"] if ev["type"] in {"hover", "near_miss", "wobble", "collision"}}
    ])

    # Reward roll-up: only when every episode has integrated reward.
    reward_per_episode = None
    if all(e["reward_integrated"] is not None for e in eps):
        keys = list(eps[0]["reward_integrated"]["by_term"].keys())
        sums = {k: float(np.mean([e["reward_integrated"]["by_term"][k] for e in eps]))
                for k in keys}
        negatives = {k: v for k, v in sums.items() if v < 0}
        reward_per_episode = {
            **sums,
            "dominant_negative_modal": (min(negatives, key=negatives.get) if negatives else None),
        }

    bucket_map: dict[str, list[dict]] = {}
    for e in eps:
        bucket_map.setdefault(_spawn_bucket(e["spawn"]), []).append(e)
    spawn_buckets = [
        {
            "bucket": k,
            "n": len(v),
            "finish_rate": float(np.mean([x["outcome"]["finished"] for x in v])),
            "mean_gates": float(np.mean([x["outcome"]["gates_passed"] for x in v])),
        }
        for k, v in sorted(bucket_map.items())
    ]

    episodes_block = [
        {"i": int(e["episode"]), "gates": e["outcome"]["gates_passed"],
         "finished": e["outcome"]["finished"], "headline": e["headline"]}
        for e in eps
    ]

    return {
        "checkpoint": run_meta.get("checkpoint"),
        "config": run_meta.get("config"),
        "control_mode": run_meta.get("control_mode"),
        "n_episodes": len(eps),
        "git_sha": run_meta.get("git_sha"),
        "aggregate": aggregate,
        "terminal_cause_histogram": terminal_causes,
        "anomaly_histogram": anomalies,
        "reward_per_episode": reward_per_episode,
        "spawn_buckets": spawn_buckets,
        "episodes": episodes_block,
        "investigator_notes": [],  # filled in C11
    }
```

**Step 2: Wire into `analyze()`**

```python
    n_gates = episodes[0][1].header["n_gates"]
    run_summary = build_run_summary(run_meta, summaries, n_gates)
    (analysis / "run_summary.json").write_text(json.dumps(run_summary, indent=2))
    print(f"Wrote run_summary.json")
```

**Step 3: Smoke validation**

```powershell
pixi run -e rl-train python scripts/analyze_eval_traces.py renders/plan-B6-smoke/trace
Get-Content renders/plan-B6-smoke/analysis/run_summary.json
```

Expected: structure matches the design doc. `finish_rate` between 0 and 1, `terminal_cause_histogram` keys all start with `collision:`, `finished`, or `truncated`. `episodes` block has one entry per episode with a headline.

**Step 4: Commit**

```powershell
git add scripts/analyze_eval_traces.py
git commit -m "feat(analyze): cross-episode run_summary aggregation"
```

---

### Task C11: Investigator notes

**Files:** Modify `scripts/analyze_eval_traces.py`.

**Step 1: Rule-based notes**

```python
def build_investigator_notes(summary: dict) -> list[str]:
    notes: list[str] = []
    n = summary["n_episodes"]
    if n == 0:
        return notes

    # Terminal causes > 30% of episodes.
    for cause, count in summary["terminal_cause_histogram"].items():
        if cause == "finished" or count / n <= 0.30:
            continue
        notes.append(f"{count}/{n} episodes ended in {cause} — dominant failure mode.")

    # Anomalies present in > 50% of episodes.
    for kind, count in summary["anomaly_histogram"].items():
        if count / n <= 0.50:
            continue
        notes.append(f"{kind} events present in {count}/{n} episodes.")

    # Spawn buckets > 2x off the mean.
    mean_fr = summary["aggregate"]["finish_rate"]
    for b in summary["spawn_buckets"]:
        if b["n"] < 2 or mean_fr == 0:
            continue
        ratio = b["finish_rate"] / mean_fr
        if ratio > 2.0 or ratio < 0.5:
            notes.append(
                f"Spawn bucket {b['bucket']} has finish_rate {b['finish_rate']:.2f} vs "
                f"run mean {mean_fr:.2f} — strong spawn dependence."
            )

    return notes
```

**Step 2: Wire in**

In `build_run_summary`, replace the `investigator_notes: []` placeholder with a call to `build_investigator_notes(...)` after the summary dict is otherwise complete.

**Step 3: Smoke + commit**

```powershell
git add scripts/analyze_eval_traces.py
git commit -m "feat(analyze): rule-based investigator notes"
```

---

## Phase D — End-to-end validation

### Task D1: Full smoke run on v33b checkpoint

**Files:** none (validation only).

**Step 1: Generate a `reward_config.json` for v33b**

```powershell
pixi run -e rl-train python -c "import json; from dataclasses import asdict; from lsy_drone_racing.control.rl_song.config import RewardConfig; open('lsy_drone_racing/control/rl_song/checkpoints/level3_v33b_warmstart_from_v32a_seed0_300M/reward_config.json', 'w').write(json.dumps(asdict(RewardConfig()), indent=2))"
```

This uses `RewardConfig`'s current defaults — **good enough for the smoke run, but not necessarily v33b's actual training config.** If v33b's exact reward weights matter, manually reconstruct from the wandb run config under `weilun-ang-technical-university-munich/lsy-drone-racing-rl-song`.

**Step 2: Full eval + dump**

```powershell
pixi run -e rl-train python -m lsy_drone_racing.control.rl_song.eval_sim --config level3.toml --checkpoint lsy_drone_racing/control/rl_song/checkpoints/level3_v33b_warmstart_from_v32a_seed0_300M/ --control_mode attitude --n_runs 8 --dump_trace renders/plan-D-v33b/trace --record renders/plan-D-v33b/eval.mp4
```

**Step 3: Analyze**

```powershell
pixi run -e rl-train python scripts/analyze_eval_traces.py renders/plan-D-v33b/trace
```

**Step 4: Cross-check the rendered video against the run summary**

Open `renders/plan-D-v33b/eval.mp4`. For each episode, read the corresponding `episode_NNN.summary.json` headline. The headline should match what's visible in the video: gate passes match, "hovered before gate N" matches a visible park-then-shoot, "crashed obstacle 1" matches the visible terminal frame.

If any headline-vs-video disagreement is severe, that's a tooling bug (mis-tuned detector threshold, schema mismatch, etc.) — file an issue note in the design doc's "known limitations" section and consider a follow-up task.

**Step 5: Commit (no code, optional artifact)**

Don't commit the renders dir. The tooling is done; the smoke artifacts are throwaway.

---

## Done

The tooling is now ready for overnight autoresearch:

1. Train a new checkpoint → `reward_config.json` is auto-written.
2. Eval with `--dump-trace path/to/trace` → per-step JSONL plus run meta.
3. Analyze with `scripts/analyze_eval_traces.py path/to/trace` → per-episode summary + `run_summary.json`.
4. Read `run_summary.json` (me or a Haiku-class subagent) to drive the next training iteration.

Next likely follow-ups (not in this plan):
- Wire eval+analyze into the existing overnight orchestration so each checkpoint produces summaries automatically.
- A `compare_runs.py` that diffs two `run_summary.json` files for regression detection.
- Detector threshold tuning once we see how the headlines line up against rendered videos in practice.
