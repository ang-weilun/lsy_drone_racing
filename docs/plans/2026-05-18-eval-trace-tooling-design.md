# Eval Trace Tooling Design

**Goal:** Make sim-eval episodes machine-analyzable so that overnight
autoresearch can read what happened in an evaluation run without a human
watching the rendered video. Prior reviews of rendered episodes have
repeatedly disagreed with the actual behaviour (e.g. "v32a first finish"
was read off `ep_len` alone); this tooling closes that gap.

**Architecture:** Two pieces, no script fork.

1. `eval_sim.py` gains an opt-in `--dump-trace <dir>` flag. Default
   behaviour is unchanged (no extra I/O). When set, eval writes raw
   per-step records to that directory; **no analysis is done inline**.
2. `scripts/analyze_eval_traces.py` (new). Reads a trace directory and
   produces per-episode summary JSON plus a run rollup. Re-runnable on
   old traces, so heuristic improvements re-grade prior runs without
   re-executing eval.

**Tech stack:** stdlib `json` for serialization (JSONL for timeseries,
JSON for summaries), `numpy` for kinematics derivations, reuse of
existing JAX helpers (`_quat_to_matrix`, `_gate_frame_edge_dist_sq`,
`step_reward`) from `lsy_drone_racing/control/rl_song/{obs,reward}.py`.

**Out of scope:**
- No pytest. Per CLAUDE.md the RL track validates via training/sim
  metrics; eval tooling will be validated by running it against an
  existing checkpoint and inspecting outputs.
- No LLM in the analyzer. All "narrative" output is deterministic
  template strings.
- No real-flight (`eval.py`) integration. Sim only.
- **No 500 Hz sub-step logging.** Even at sim rate the first-contact
  pose is not recoverable — contact detection runs once per env step,
  after all 10 sub-steps (`race_core.py:523-526`). 500 Hz buys
  trajectory resolution we don't need and doesn't buy impact accuracy.
  We accept the 50 Hz limit and name the affected field accordingly.

**Codex-reviewed.** This doc incorporates the findings from a codex
static review (thread 019e3bb9). Inline `(codex #N)` tags mark places
that exist specifically because of a codex finding.

---

## File layout

Per eval run (e.g. `renders/2026-05-19-v33b-level3/`):

```
trace/
  run_meta.json             # checkpoint, config, seed, n_runs, git SHA,
                            # reward_cfg snapshot, schema version
  episode_000.jsonl
  episode_001.jsonl
  ...
analysis/                   # produced by analyze_eval_traces.py
  episode_000.summary.json
  episode_001.summary.json
  ...
  run_summary.json
```

Co-locating `trace/` and `analysis/` next to the existing `.mp4` keeps
"what was rendered" and "what got dumped" together.

---

## Raw trace schema (`trace/episode_NNN.jsonl`)

One JSON object per env step (50 Hz, ~250–500 records per episode).

**Line 0 — header (`{"_header": true, ...}`):**
- `n_gates`, `n_obstacles`, `freq`, `config`, `control_mode`,
  `checkpoint`, `spawn_pos`, `spawn_quat`, `schema_version`,
  `reward_cfg_path` (resolved at eval start).
- Putting metadata in line 0 keeps every line a valid JSON object;
  `pd.read_json(..., lines=True)` reads it with no special-casing.

**Lines 1..N — per-step records.** The post-step state at simulator
time `t = (i+1) / freq` (codex #9 — the existing `curr_time` in
`_run_episode` is set *before* `env.step`, so it's one step early):

```jsonc
{
  "step": 42, "t": 0.86,

  "pos": [x, y, z], "vel": [vx, vy, vz],
  "quat": [qx, qy, qz, qw], "ang_vel": [wx, wy, wz],

  // Policy mean (7-D) is logged via a controller-side hook (see below).
  // The "applied" action is the 4-D post-projection action the env saw.
  "action_policy_mean": [...],        // shape (7,), set by the controller hook
  "action_applied":     [a0, a1, a2, a3],

  "target_gate": 1,                   // -1 once finished

  // True world state from env.unwrapped.data (un-masked).
  // Already has the leading n_envs axis on the JAX side; we squeeze
  // it out before serializing.
  "gates_pos_true":     [[x,y,z], ...],
  "gates_quat_true":    [[qx,qy,qz,qw], ...],
  "obstacles_pos_true": [[x,y,z], ...],

  // Policy view: the obs dict's gate/obstacle fields. On level 3
  // these are masked (nominal until observed by sensor range), so
  // they may differ from the "_true" copies above. Same names the
  // policy sees — no renaming.
  "gates_pos":     [[x,y,z], ...],
  "gates_quat":    [[qx,qy,qz,qw], ...],
  "obstacles_pos": [[x,y,z], ...],

  // Historical "has been within sensor range at some point" flags
  // (codex #4 — these are NOT current visibility, document semantics).
  "gates_visited":     [false, true, false, false],
  "obstacles_visited": [false, false, true, false],

  "reward_total": 0.123,
  "reward_terms": {
    "r_prog": 0.04, "r_guid": -0.01, "r_obs": -0.002,
    "r_gate_bonus": 0.0, "r_exit_vel": 0.0, "r_gate_frame": -0.001,
    "r_omega": -0.0008, "r_vel": 0.0, "r_time": -0.001, "r_terminal": 0.0
  },

  "terminated": false, "truncated": false
}
```

**Deliberately omitted:** drone-state mirrors in `obs["pos"]` /
`vel` / `quat` / `ang_vel` (redundant with top-level), full observation
vector (redundant), controller internal state (not needed).

**Size estimate:** ~700–900 bytes/row uncompressed; 400 rows ≈ 300 KB;
32 episodes ≈ ~10 MB per eval run. No gzip until it becomes a problem.

---

## Controller hook for `action_policy_mean`

The 7-D policy mean is computed inside
`_deterministic_env_action()` and projected to 4-D before
`controller.compute_control()` returns (codex #8); the raw mean is
currently discarded.

**Mechanism (concrete).** Change
`_deterministic_env_action()`'s return type from `env_action` to
`(env_action, raw_action)`. The only caller is
`controller.compute_control()`; it stashes
`self._last_policy_mean = raw_action` right before returning
`env_action`. `eval_sim.py` then reads it back via
`getattr(controller, "_last_policy_mean", None)` after each
`compute_control` call. If absent (older controller, real-flight
controller), the trace field is `null` and downstream tooling
tolerates that.

This is the only change to controller code; it's additive, doesn't
break deployment because the new return tuple is destructured at the
single existing call site.

---

## Persisting `reward_cfg` next to the checkpoint

The design originally assumed the trained-with `RewardConfig` was
written alongside the Orbax checkpoint. **It is not** —
`_save_checkpoint()` writes only params / optimizer state / normalizer
/ stage / RNG / step counters (codex #5; `train.py:907-922`).

**Fix (decided: option a).** Write `reward_config.json` **once at
training start**, into the run's checkpoint root directory — *not*
from inside `_save_checkpoint()`. The reward config doesn't mutate
during training, so writing it once at the same level as the Orbax
checkpoint dirs (e.g. `run_dir/reward_config.json`) is sufficient and
avoids changing the `_save_checkpoint()` signature or threading
`train_cfg` through to it. The write happens in `train.py`'s training
entrypoint right after the run directory is created, alongside the
existing run-config logging. One `json.dump(asdict(train_cfg.reward),
...)` call.

**Resume / warm-start guard (codex).** When `train()` reuses an
existing run directory (warm-start from a prior run), do **not**
overwrite an existing `reward_config.json`. Three cases:
1. File absent → write the current `train_cfg.reward`.
2. File present, matches `asdict(train_cfg.reward)` → no-op.
3. File present, differs → raise loudly. The two configs disagree;
   silently overwriting would mislabel every checkpoint produced by
   the prior run. The operator decides whether to delete the old
   file (intentional reward change) or fix the current
   `train_cfg.reward` (drift).

Do **not** infer a historical config from current code for an old
run directory that pre-dates this scheme. If the file is missing,
it stays missing for that run; eval handles it via the back-compat
null path below.

The eval loader resolves `reward_config.json` relative to the
checkpoint path (`<checkpoint>/../reward_config.json` for a
single-step dir, or `<checkpoint>/reward_config.json` for a run-dir
path). The resolved path is recorded in the header.

**Back-compat for older checkpoints (v25 … v33b).** No file present
**and** `--reward-cfg` not set → eval still produces traces, but
**every reward field is explicit null, end-to-end** (codex review):
- Each trace row: `"reward_total": null, "reward_terms": null`.
  Eval does **not** fall back to the base env reward, which is sparse
  and not Song's replacement reward (`race_core.py:700`).
- Per-episode summary: `"reward_integrated": null` (whole block
  omitted or set to null, not a partial dict).
- Run summary: `"reward_per_episode": null` likewise. The
  `investigator_notes` rule that references `dominant_negative_modal`
  is suppressed in this case.
- Header: `"reward_cfg_path": null`.

CLI override `--reward-cfg path/to/reward_config.json` accepts the
**same JSON format** as the file written at training start (an
`asdict(RewardConfig)` dump). Not a TOML — `RewardConfig` is a Python
dataclass under `TrainConfig.reward` (`config.py:153, 915`), not a
race-config TOML schema. Re-attaching a config for older checkpoints
means producing a JSON of the dataclass.

---

## Reward-term computation at eval time

Mirror the non-vec call site in `env_wrapper.py:177`.

Inside `_run_episode`, around `obs, reward, terminated, truncated, info
= env.step(action)`:

1. **Stash `prev_obs`** before stepping. The reward function needs both
   endpoints of the transition.
2. **Derive auxiliary flags** from `target_gate` transitions (codex #1
   — `eval_sim.py` does not `terminated |= finished` like the wrapper
   does; the env may return `target_gate == -1` without `terminated`,
   and the next step is a disabled/warp step):
   - `gate_just_passed = (prev_tg >= 0) and (obs_tg != prev_tg)`
   - `finished = (obs_tg == -1) and (prev_tg != -1)`
   - **Break the loop when `finished`**, mirroring the wrapper, so the
     trace never includes a post-finish warp step.
   - **Consistency with `curr_time`.** The existing `_run_episode` sets
     `curr_time = i / freq` *before* `env.step`, then returns it after
     break (codex). Move the assignment to *after* `env.step` and use
     `(i + 1) / freq`, so the returned/logged flight time, the
     existing `_log_episode_stats` call, and the trace row's `t` all
     refer to the same post-step instant. Otherwise the trace and the
     `ep_times` array disagree by one step.
3. **Read true gate/obstacle poses from `env.unwrapped.data`** —
   `gates_pos`, `gates_quat`, `obstacles_pos`. These already carry the
   `n_envs` axis (codex #2 — do *not* re-add it).
4. **Squeezed obs fields get `[None]` added back.** The env's
   `DroneRaceEnv.step` squeezes the leading axis out of `obs` and
   `info` (`drone_race.py:99-102`); we add it back only for these
   fields, not for `env.unwrapped.data.*`.
5. Call `step_reward(...)`, then `np.asarray(v).squeeze(0)` on outputs.
   JIT overhead is trivial at 50 Hz / `n_envs=1`; no jit-wrapper cache.

**What the reward fn actually uses (codex #3 / #7).**
- `r_prog`, `r_guid` use `true_gates_pos` / `true_gates_quat` when
  passed.
- `r_obs`, `r_gate_frame` use the **masked** `env_obs["obstacles_pos"]`
  / `env_obs["gates_pos"]` / `env_obs["gates_quat"]` — they grade the
  actor against what the actor sees.
- `true_obstacles_pos` is accepted but **ignored**. Don't bother
  passing it.

**Which `reward_cfg`?** Loaded from `reward_config.json` written next
to the checkpoint (see "Persisting `reward_cfg`" above), unless
overridden by `--reward-cfg`.

**Failure mode to flag.** If `reward_config.json` is missing *and*
`--reward-cfg` is not set, eval still produces traces but
`reward_terms` is `null` per row, and the header records
`reward_cfg_path: null`. The analyzer then skips the
`reward_integrated` section. **No silent zero-fill** (codex would not
forgive that).

---

## Analyzer: per-episode summary (`analysis/episode_NNN.summary.json`)

```jsonc
{
  "episode": 3,
  "outcome": {
    "gates_passed": 3, "finished": false,
    "ep_len_steps": 312, "flight_time_s": 6.24,
    "terminal_cause": "collision:obstacle:1"
  },
  "spawn": {"pos": [...], "quat": [...]},

  "events": [
    {"type": "takeoff",   "t": 0.18, "vz_at_liftoff": 0.62},
    {"type": "gate_pass", "t": 1.42, "gate": 0,
        "speed": 2.1, "in_plane_offset_m": 0.08,
        "angle_off_normal_rad": 0.12},
    {"type": "hover",     "t_start": 2.40, "t_end": 3.10,
        "duration_s": 0.70, "xy_bbox_extent_m": 0.11,
        "mean_pos": [...], "near_gate": 1},
    {"type": "near_miss", "t": 4.10, "gate": 2,
        "closest_frame_dist_m": 0.06, "passed": false},
    {"type": "collision", "t": 4.50, "object": "obstacle:1",
        "approach_speed_50hz": 1.8,
        "last_pos_50hz_pre_terminal": [...],   // codex #6 — see notes
        "min_approach_dist_5frame_m": 0.04},
    {"type": "wobble",    "t_start": 3.6, "t_end": 4.1,
        "duration_s": 0.5, "max_ang_vel_rad_s": 9.2}
  ],

  "kinematics_metrics": {
    "max_speed": 3.2, "mean_speed": 1.8,
    "max_ang_vel": 9.2, "path_length_m": 11.4
  },

  "reward_integrated": {
    "total": 4.21,
    "by_term": { "r_prog": 2.8, "r_gate_bonus": 1.8, "r_terminal": -1.0,
                 "r_obs": -0.3, "r_gate_frame": -0.6, "r_guid": -0.4,
                 "r_omega": -0.2, "r_exit_vel": 0.1, "r_vel": 0.0,
                 "r_time": -0.6 },
    "dominant_positive": "r_prog",
    "dominant_negative": "r_terminal"
  },

  "headline": "Passed 3/4 gates; hovered 0.70s before gate 1; crashed obstacle 1 at t≈4.50s @ ≈1.8 m/s"
}
```

### Detectors

Each detector exposes its thresholds as module-level constants for easy
tuning.

- **Takeoff:** first frame where `pos.z > 0.10 m`.

- **Hover (positional bbox, not speed).** Over a sliding window of
  `W = 20` steps (= 0.4 s at 50 Hz), the xy bounding-box extent
  `max(pos.xy) - min(pos.xy)` stays `< 0.15 m` on both axes.
  Contiguous hover windows are coalesced. Rationale: a slow gate
  approach has low `|vel|` but is not "parked"; bbox-on-position is
  the faithful definition of "the drone stopped going somewhere".

- **Gate-pass attributes.** Project the drone position at the pass
  frame into the gate's local frame using `_quat_to_matrix` (in
  `obs.py`). Aperture offset = `(y, z)` magnitude; angle-off-normal =
  angle between velocity at the pass frame and the gate forward axis.
  Use the **true** gate pose (`gates_pos_true` / `gates_quat_true`)
  for these computations, not the masked policy-view pose.

- **Near-miss.** Closest distance to gate-frame edges (reuse
  `_gate_frame_edge_dist_sq` from `reward.py`) drops below `0.20 m`
  *without* `target_gate` advancing. Computed against **true** gate
  poses.

- **Collision (50 Hz bounded).** The sim warps the drone to a reset
  point on crash, *and* contact detection runs only at the env-step
  boundary after 10 sim sub-steps (codex #6). The terminal-frame
  `pos` is the warp location; the frame before is up to 20 ms before
  actual contact. We accept this and name fields honestly:
  - `last_pos_50hz_pre_terminal = pos[T-1]`.
  - `approach_speed_50hz = |vel[T-1]|`.
  - **Object resolution** uses `argmin` distance from `pos[T-1]` to:
    - **Gates:** distance to gate centroid (point), using
      `gates_pos_true`.
    - **Obstacles:** distance to a vertical line segment from
      `(x, y, z_top)` to `(x, y, 0)` where `(x, y, z_top)` is
      `obstacles_pos_true[i]` — because that field is the **top
      marker of a vertical capsule**, not the centroid (codex #7).
      Using point-distance to a capsule top would misclassify
      lower-portion impacts as "floor".
    - **Floor:** `z < 0.05 m` at `T-1`.
  - **Robustness window.** Scan the last 5 pre-terminal frames; report
    the **minimum** approach-distance to the inferred object across
    that window as `min_approach_dist_5frame_m`. Catches glance-
    then-warp cases where the closest approach was a frame or two
    before the loop exit.

- **Wobble.** `|ang_vel| > 6 rad/s` sustained `≥ 0.2 s`.

- **Reward integration.** Straightforward sums per term. The reward
  function zeros `r_prog / r_obs / r_gate_frame / r_guid` on the
  crashed terminal step (`reward.py:455-458`); the analyzer respects
  this without special-casing — terminal-step contributions reduce
  to `r_terminal + r_time + r_omega`.

### Finish detection in the analyzer (codex #1)

The analyzer does not trust `terminated` alone for "finished" — it
detects finish by `target_gate` transitioning to `-1`, regardless of
the `terminated` flag. `eval_sim.py` is also patched to break the
loop on finish so trace files don't include a post-finish warp step.

### Headline

One line, mechanically generated. No LLM. Template:
`"Passed {N}/{G} gates; {dominant_anomaly}; {terminal_cause}"`.
`dominant_anomaly` picked from `events` by fixed priority:
`collision > near_miss > hover > wobble > none`.

The headline uses `≈` symbols around the time and approach-speed of
collision events to remind the reader that those are 50 Hz-bounded.

---

## Cross-episode rollup (`analysis/run_summary.json`)

The artifact read first by me and by Haiku-class subagents. One file
answering: *did the checkpoint improve, where does it fail, what
should I look at first?*

```jsonc
{
  "checkpoint": ".../step_000300000000",
  "config": "level3.toml",
  "control_mode": "attitude",
  "n_episodes": 32,
  "git_sha": "33db5ee",

  "aggregate": {
    "finish_rate": 0.22,
    "gates_passed":  {"mean": 1.59, "max": 4, "histogram": [3,5,10,7,7]},
    "ep_len_steps":  {"mean": 248, "median": 240, "min": 60, "max": 400},
    "flight_time_s": {"mean": 4.96, "median": 4.80},
    "max_speed":     {"mean": 2.7, "max": 3.4}
  },

  "terminal_cause_histogram": {
    "finished": 7, "collision:obstacle:1": 11,
    "collision:gate:2": 8, "collision:floor": 2, "truncated": 4
  },

  "anomaly_histogram": {
    "hover": 19, "near_miss": 6, "wobble": 12, "collision": 21
  },

  "reward_per_episode": {
    "r_prog": 2.7, "r_gate_bonus": 1.8, "r_terminal": -0.4,
    "r_obs": -0.3,
    "dominant_negative_modal": "r_terminal"
  },

  "spawn_buckets": [
    {"bucket": "x∈[0.5,1.5], y∈[-1,0]", "n": 9,
     "finish_rate": 0.55, "mean_gates": 2.8},
    {"bucket": "x∈[0.5,1.5], y∈[0,1]",  "n": 8,
     "finish_rate": 0.00, "mean_gates": 0.6}
  ],

  "episodes": [
    {"i": 0, "gates": 4, "finished": true,
     "headline": "Clean 4/4, 4.21s, peak 3.1 m/s"},
    {"i": 1, "gates": 1, "finished": false,
     "headline": "Passed 1/4; hovered 0.8s before gate 1; crashed obstacle 1 at t≈2.4s @ ≈1.6 m/s"}
  ],

  "investigator_notes": [
    "11/32 episodes ended at obstacle 1 — dominant failure mode.",
    "Spawn bucket y∈[0,1] has finish_rate 0.00 vs 0.55 in y∈[-1,0] — strong spawn dependence.",
    "Hover events present in 19/32 episodes — policy still parks before committing."
  ]
}
```

**`spawn_buckets` is the lucky-zone diagnostic.** Fixed 8-cell grid on
`(x, y)` to start; the course is bounded. Upgrade to KMeans only if
the 8-cell grid is too coarse.

**`investigator_notes` is the only opinionated output.** 2-3 lines,
deterministic rules:
- any terminal cause > 30% of episodes,
- any spawn bucket > 2× off the mean,
- any anomaly present in > 50% of episodes.

No LLM. If the rules age, edit them.

---

## Consumer model

- **Me (main thread):** read `run_summary.json` for the overview;
  drill into `episode_NNN.summary.json` when something is off; query
  `episode_NNN.jsonl` programmatically (pandas) only when the summary
  is not enough.
- **Haiku subagent (overnight autoresearch):** pointed at
  `run_summary.json` plus the per-episode summaries in one call.
  ~tens of KB total per eval run — comfortably within budget.

---

## Schema versioning

`run_meta.json` carries a `schema_version` field. Bump on any
breaking change to the trace or summary schemas. The analyzer refuses
to run on a trace whose version it does not know — failing loud is
preferable to silently misinterpreting fields.
