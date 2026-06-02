# L2 cold-start screen: ω channel + 512 saturation diagnostic — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two env-var-gated toggles (an angular-velocity obs channel; actor/critic width) so we can cold-start three matched cells on L2 — `ref`(256/52), `omegaA`(256/55), `capB`(512/52) — and seed-match-evaluate whether ω helps and whether 512 saturates on the healthy stack.

**Architecture:** Both toggles are module constants read from env vars **at import** (before tyro parses CLI args), matching the existing `ABLATE_MODE` pattern. `ACTOR_OBS_ANG_VEL_DIM` (in `rl_song/config.py`) feeds `ACTOR_OBS_DIM`, which every consumer reads symbolically, so the obs change propagates to the obs space, slicing, and normalizer automatically. The obs has exactly two encoders to keep in lockstep — the canonical JAX `rl_song/obs.py` and the numpy deploy mirror `rl_sbx/deploy_numpy/obs.py` (`rollout.py`/`env_gym.py` only *call* the canonical one). Width is `HIDDEN_SIZE` in `rl_sbx/policy.py`, picked up by `NET_ARCH`; `train.py` passes no `net_arch`, so the env var fully controls it.

**Tech Stack:** JAX/Flax + SBX PPO (`rl_sbx`), shared JAX obs/reward/config (`rl_song`), numpy deploy mirror, tyro CLI, pixi env `rl-train`, vast.ai + tmux launch.

**Testing note (repo rule):** Per project CLAUDE.md we do **not** write a pytest suite or gate on `pytest`. Verification here is: `ruff`, `python -m py_compile`, dim/`NET_ARCH` assertions via `python -c`, and — for the one piece of equivalence logic that matters — a **checkpoint-free JAX-vs-numpy encoder parity check**. The real validation is the training runs + seed-matched eval (Task 5).

---

### Task 1: ω toggle in config + canonical JAX obs encoder

**Files:**
- Modify: `lsy_drone_racing/control/rl_song/config.py` (imports; obs-dim block lines 14-67)
- Modify: `lsy_drone_racing/control/rl_song/obs.py` (import line 35; drone block lines 231-236; docstring)

- [ ] **Step 1: Add `os` import to config.py**

In `lsy_drone_racing/control/rl_song/config.py`, change the import block (currently lines 14-16):

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
```

- [ ] **Step 2: Add the ω-dim constant and fold it into `ACTOR_OBS_DIM`**

In `config.py`, replace the block at lines 59-67 (from `ACTOR_OBS_PREV_ACTION_DIM` through the `assert`) with:

```python
ACTOR_OBS_PREV_ACTION_DIM: int = 0
# Body-frame angular-velocity channel, toggled on for the L2 ω screen via the
# RL_OBS_ANG_VEL env var. Read at import — before the tyro CLI parses args —
# matching the controller_ablate ABLATE_MODE pattern. Default off preserves the
# 52-d reference obs. See docs/superpowers/specs/
# 2026-06-02-l3-obs-completion-capacity-base-design.md.
ACTOR_OBS_ANG_VEL_DIM: int = 3 if os.environ.get("RL_OBS_ANG_VEL", "0") == "1" else 0
# Per-slot obstacle: 2 body-frame xy + 1 body-frame velocity projected onto
# unit-to-obstacle + N_OBSTACLES one-hot identity + 1 visited flag.
_PER_OBSTACLE_SLOT_DIM: int = 2 + 1 + N_OBSTACLES + 1
ACTOR_OBS_OBSTACLE_DIM: int = N_NEAREST_OBSTACLES * _PER_OBSTACLE_SLOT_DIM
ACTOR_OBS_DIM: int = (
    ACTOR_OBS_DRONE_DIM
    + ACTOR_OBS_ANG_VEL_DIM
    + ACTOR_OBS_GATE_DIM
    + ACTOR_OBS_PREV_ACTION_DIM
    + ACTOR_OBS_OBSTACLE_DIM
)
assert ACTOR_OBS_DIM == 52 + ACTOR_OBS_ANG_VEL_DIM, "Actor obs layout drift"
```

(The existing `_PER_OBSTACLE_SLOT_DIM` / `ACTOR_OBS_OBSTACLE_DIM` lines are unchanged; they are repeated here only because the `ACTOR_OBS_ANG_VEL_DIM` line is inserted just above them.)

- [ ] **Step 3: Import the constant in obs.py**

In `lsy_drone_racing/control/rl_song/obs.py`, change line 35:

```python
from lsy_drone_racing.control.rl_song.config import (
    ACTOR_OBS_ANG_VEL_DIM,
    ACTOR_OBS_DIM,
    N_NEAREST_OBSTACLES,
    N_OBSTACLES,
)
```

- [ ] **Step 4: Conditionally append ω to the drone channel**

In `obs.py` `build_actor_obs`, replace the drone-channel block (lines 231-236):

```python
    # Drone channel.
    rot_wb = _quat_to_matrix(quat)
    rot_9d = rot_wb.reshape(9)
    rot_bw = rot_wb.T
    vel_body = rot_bw @ vel
    drone_parts = [rot_9d, vel_body]
    if ACTOR_OBS_ANG_VEL_DIM:
        # Body-frame body rates, appended raw (already body frame, unlike vel).
        # The flag is a static module constant, so this branch is resolved
        # before jit tracing — the graph shape stays static per run.
        drone_parts.append(env_obs["ang_vel"])
    drone_chan = jnp.concatenate(drone_parts)
```

Then update the module docstring line 11 to read:
`* drone (12, or 15 with RL_OBS_ANG_VEL=1): 9D rotation matrix, body-frame linear velocity (3), and optionally body-frame angular velocity (3).`

- [ ] **Step 5: Verify the dim toggles, format, and compile**

Run:
```bash
cd /home/exedev/lsy_drone_racing
RL_OBS_ANG_VEL=0 pixi run -e rl-train python -c "from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_DIM; print('off', ACTOR_OBS_DIM)"
RL_OBS_ANG_VEL=1 pixi run -e rl-train python -c "from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_DIM; print('on', ACTOR_OBS_DIM)"
ruff format lsy_drone_racing/control/rl_song/config.py lsy_drone_racing/control/rl_song/obs.py
ruff check lsy_drone_racing/control/rl_song/config.py lsy_drone_racing/control/rl_song/obs.py
python -m py_compile lsy_drone_racing/control/rl_song/config.py lsy_drone_racing/control/rl_song/obs.py
```
Expected: `off 52`, `on 55`, ruff clean, compile silent.

- [ ] **Step 6: Commit**

```bash
git add lsy_drone_racing/control/rl_song/config.py lsy_drone_racing/control/rl_song/obs.py
git commit -m "rl_song: optional body-frame angular-velocity obs channel (RL_OBS_ANG_VEL)"
```

---

### Task 2: ω mirror in the numpy deploy encoder + encoder parity check

**Files:**
- Modify: `lsy_drone_racing/control/rl_sbx/deploy_numpy/obs.py` (import line 16; drone block lines 95-99)
- Create: `scripts/check_obs_encoder_parity.py`

- [ ] **Step 1: Import the constant in the numpy mirror**

In `lsy_drone_racing/control/rl_sbx/deploy_numpy/obs.py`, change line 16:

```python
from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_ANG_VEL_DIM, ACTOR_OBS_DIM
```

- [ ] **Step 2: Conditionally append ω in the numpy drone channel**

In `deploy_numpy/obs.py` `build_actor_obs`, replace lines 95-99:

```python
    rot_wb = Rotation.from_quat(quat).as_matrix().astype(np.float32)
    rot_9d = rot_wb.reshape(9)
    rot_bw = rot_wb.T
    vel_body = rot_bw @ vel
    drone_parts = [rot_9d, vel_body]
    if ACTOR_OBS_ANG_VEL_DIM:
        drone_parts.append(np.asarray(env_obs["ang_vel"], dtype=np.float32))
    drone_chan = np.concatenate(drone_parts)
```

- [ ] **Step 3: Create the checkpoint-free encoder parity check**

Create `scripts/check_obs_encoder_parity.py`:

```python
"""Checkpoint-free parity check between the JAX and numpy actor-obs encoders.

Compares ``rl_song.obs.build_actor_obs`` (JAX, training) against
``rl_sbx.deploy_numpy.obs.build_actor_obs`` (numpy, deploy) on a fixed fake
observation with matched identity normalizers (mean 0, var 1). Run at both
``RL_OBS_ANG_VEL`` settings to confirm the angular-velocity toggle stays in
lockstep across the two encoders, and that the output is finite and within the
training-time clip range.

Usage::

    RL_OBS_ANG_VEL=0 pixi run -e rl-train python scripts/check_obs_encoder_parity.py
    RL_OBS_ANG_VEL=1 pixi run -e rl-train python scripts/check_obs_encoder_parity.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.rl_sbx.deploy_numpy import obs as np_obs
from lsy_drone_racing.control.rl_sbx.deploy_numpy.normalizer import (
    NORM_VAR_EPS,
    NormalizerState,
)
from lsy_drone_racing.control.rl_song import obs as jax_obs
from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_DIM

_N_GATES: int = 4
_N_OBSTACLES: int = int(np_obs.N_OBSTACLES)
_RNG_SEED: int = 20260602
_TOL: float = 1e-5


def _fake_env_obs() -> dict[str, np.ndarray]:
    """Build a deterministic unbatched env observation with project shapes."""
    rng = np.random.default_rng(_RNG_SEED)
    quat = Rotation.from_euler("xyz", rng.normal(0.0, 0.2, size=3)).as_quat()
    gates_quat = Rotation.from_euler(
        "xyz", rng.normal(0.0, 0.2, size=(_N_GATES, 3))
    ).as_quat()
    return {
        "pos": rng.normal([0.0, 0.0, 0.7], [0.5, 0.5, 0.1]).astype(np.float32),
        "quat": quat.astype(np.float32),
        "vel": rng.normal(0.0, 0.4, size=3).astype(np.float32),
        "ang_vel": rng.normal(0.0, 0.2, size=3).astype(np.float32),
        "target_gate": np.asarray(1, dtype=np.int32),
        "gates_pos": rng.normal(
            np.linspace([0.6, -0.6, 0.8], [3.2, 0.6, 1.0], _N_GATES), 0.08
        ).astype(np.float32),
        "gates_quat": gates_quat.astype(np.float32),
        "obstacles_pos": rng.normal(
            np.linspace([0.5, 0.7, 1.0], [2.8, -0.7, 1.0], _N_OBSTACLES), 0.1
        ).astype(np.float32),
        "obstacles_visited": rng.choice([False, True], size=_N_OBSTACLES).astype(bool),
    }


def main() -> None:
    """Encode the fake obs with both encoders and assert they agree."""
    env_obs = _fake_env_obs()
    prev_action = np.zeros(4, dtype=np.float32)

    jax_norm = jax_obs.init_normalizer(ACTOR_OBS_DIM)
    np_norm = NormalizerState(
        mean=np.zeros(ACTOR_OBS_DIM, dtype=np.float32),
        var=np.ones(ACTOR_OBS_DIM, dtype=np.float32),
        count=np.asarray(NORM_VAR_EPS, dtype=np.float32),
    )

    jax_out = np.asarray(
        jax_obs.build_actor_obs(env_obs, prev_action, jax_norm), dtype=np.float32
    )
    np_out = np_obs.build_actor_obs(env_obs, prev_action, np_norm)

    if jax_out.shape != (ACTOR_OBS_DIM,):
        raise ValueError(f"jax obs shape {jax_out.shape} != ({ACTOR_OBS_DIM},)")
    if not (np.all(np.isfinite(jax_out)) and np.all(np.isfinite(np_out))):
        raise ValueError("non-finite obs")
    if np.max(np.abs(jax_out)) > jax_obs.NORM_CLIP + 1e-4:
        raise ValueError("obs exceeds NORM_CLIP")
    diff = float(np.max(np.abs(jax_out - np_out)))
    np.testing.assert_allclose(jax_out, np_out, atol=_TOL, rtol=_TOL)
    print(f"ACTOR_OBS_DIM={ACTOR_OBS_DIM} max_abs_diff={diff:.3e} OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run parity at both toggle settings, format, compile**

```bash
cd /home/exedev/lsy_drone_racing
RL_OBS_ANG_VEL=0 pixi run -e rl-train python scripts/check_obs_encoder_parity.py
RL_OBS_ANG_VEL=1 pixi run -e rl-train python scripts/check_obs_encoder_parity.py
ruff format lsy_drone_racing/control/rl_sbx/deploy_numpy/obs.py scripts/check_obs_encoder_parity.py
ruff check lsy_drone_racing/control/rl_sbx/deploy_numpy/obs.py scripts/check_obs_encoder_parity.py
python -m py_compile lsy_drone_racing/control/rl_sbx/deploy_numpy/obs.py scripts/check_obs_encoder_parity.py
```
Expected: first prints `ACTOR_OBS_DIM=52 max_abs_diff=...e-... OK`, second `ACTOR_OBS_DIM=55 ... OK` (diff ≤ 1e-5), ruff clean.

- [ ] **Step 5: Commit**

```bash
git add lsy_drone_racing/control/rl_sbx/deploy_numpy/obs.py scripts/check_obs_encoder_parity.py
git commit -m "deploy_numpy: mirror angular-velocity obs channel + encoder parity check"
```

---

### Task 3: width toggle in the SBX policy

**Files:**
- Modify: `lsy_drone_racing/control/rl_sbx/policy.py` (imports lines 37-39; HIDDEN_SIZE block lines 60-71)

- [ ] **Step 1: Add `os` import**

In `lsy_drone_racing/control/rl_sbx/policy.py`, change the top imports (lines 37-39):

```python
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
```

- [ ] **Step 2: Make `HIDDEN_SIZE` env-var driven and correct the stale comment**

In `policy.py`, replace the comment + constant block at lines 60-71 with:

```python
# Hidden-layer widths for actor and critic MLPs, toggled via RL_HIDDEN_SIZE
# (default 256). Read at import — before the tyro CLI parses args.
# History: v131 ran 512 and v132 reverted it after observing a saturating
# deterministic mean |tau|/alpha_max (0.47 vs rl_song's 0.08). That verdict is
# now treated as inconclusive: it was confounded by an inert obstacle barrier
# (fixed 2026-05-31), an unclipped value function (clipped-VF — the likely real
# fix — postdates v131), the split-head experiment, and an era where no policy
# finished at all. The L2 screen re-tests 512 on the healthy stack; see
# docs/superpowers/specs/2026-06-02-l3-obs-completion-capacity-base-design.md.
HIDDEN_SIZE: int = int(os.environ.get("RL_HIDDEN_SIZE", "256"))
N_HIDDEN_LAYERS: int = 2
NET_ARCH: tuple[int, ...] = (HIDDEN_SIZE,) * N_HIDDEN_LAYERS
```

- [ ] **Step 3: Verify width toggles, format, compile**

```bash
cd /home/exedev/lsy_drone_racing
pixi run -e rl-train python -c "from lsy_drone_racing.control.rl_sbx.policy import NET_ARCH; print('default', NET_ARCH)"
RL_HIDDEN_SIZE=512 pixi run -e rl-train python -c "from lsy_drone_racing.control.rl_sbx.policy import NET_ARCH; print('wide', NET_ARCH)"
ruff format lsy_drone_racing/control/rl_sbx/policy.py
ruff check lsy_drone_racing/control/rl_sbx/policy.py
python -m py_compile lsy_drone_racing/control/rl_sbx/policy.py
```
Expected: `default (256, 256)`, `wide (512, 512)`, ruff clean.

- [ ] **Step 4: Commit**

```bash
git add lsy_drone_racing/control/rl_sbx/policy.py
git commit -m "rl_sbx: env-var actor/critic width toggle (RL_HIDDEN_SIZE); de-confound v132 note"
```

---

### Task 4: L2-screen launcher with a parity pre-flight

**Files:**
- Create: `scripts/box_launch_l2_screen.sh`

- [ ] **Step 1: Create the launcher**

Create `scripts/box_launch_l2_screen.sh` (model it on `scripts/box_launch_speed.sh`, but cold and on the single-stage L2 curriculum):

```bash
#!/usr/bin/env bash
# L2 cold-start screen launcher (runs ON the vast box).
# Cold-trains one cell of the omega / 512 screen on the single-stage L2
# curriculum. No --init-from (cold); --curriculum=default is the single L2
# stage (rl_song.config.default_curriculum / stage1_level2_phase12). The reward
# recipe is held at the SOTA spdobs03 flags so the only varying factor is the
# cell's toggle (RL_OBS_ANG_VEL / RL_HIDDEN_SIZE).
#
# Usage (on box):
#   bash box_launch_l2_screen.sh <run_name> <ang_vel 0|1> <hidden_size> [total_steps]
# Cells:
#   ref:    bash box_launch_l2_screen.sh l2scr_ref    0 256
#   omegaA: bash box_launch_l2_screen.sh l2scr_omega  1 256
#   capB:   bash box_launch_l2_screen.sh l2scr_cap512 0 512
set -euo pipefail

RUN_NAME="${1:?run_name}"
ANG_VEL="${2:?ang_vel 0|1}"
HIDDEN="${3:?hidden_size}"
TOTAL="${4:-300000000}"

REPO=/root/lsy_drone_racing
LOG="$REPO/training_logs/${RUN_NAME}.log"

export PATH="$HOME/.pixi/bin:$PATH"
export SCIPY_ARRAY_API=1
mkdir -p "$REPO/training_logs"
cd "$REPO"

# Pre-flight: the JAX/numpy encoders must agree at this cell's obs toggle before
# we burn compute. set -e aborts the launch if parity fails.
RL_OBS_ANG_VEL=$ANG_VEL pixi run -e rl-train python scripts/check_obs_encoder_parity.py

tmux kill-session -t "$RUN_NAME" 2>/dev/null || true
tmux new-session -d -s "$RUN_NAME" "
  export PATH=\"\$HOME/.pixi/bin:\$PATH\"; export SCIPY_ARRAY_API=1;
  export RL_OBS_ANG_VEL=$ANG_VEL; export RL_HIDDEN_SIZE=$HIDDEN;
  cd $REPO;
  pixi run -e rl-train python -m lsy_drone_racing.control.rl_sbx.train \
    --run-name=$RUN_NAME \
    --curriculum=default \
    --alpha-max-rad=1.4 \
    --time-penalty=0.40 \
    --omega-coef=0.005 \
    --progress-coef=15 \
    --use-obstacle-barrier --obstacle-weight=0.3 \
    --use-gate-frame-barrier --gate-frame-weight=0.5 \
    --total-timesteps=$TOTAL \
    2>&1 | tee $LOG
"
echo "launched $RUN_NAME (ang_vel=$ANG_VEL hidden=$HIDDEN total=$TOTAL) in tmux; log=$LOG"
```

- [ ] **Step 2: Lint the script and dry-check the train flags exist**

```bash
cd /home/exedev/lsy_drone_racing
bash -n scripts/box_launch_l2_screen.sh
pixi run -e rl-train python -m lsy_drone_racing.control.rl_sbx.train --help 2>&1 | rg -i 'curriculum|alpha-max|time-penalty|omega-coef|progress-coef|obstacle-barrier|gate-frame|total-timesteps|run-name' | head
```
Expected: `bash -n` silent; every flag used in the launcher appears in `--help`. (If a flag name differs, fix the launcher to match `--help` before proceeding — do not invent flags.)

- [ ] **Step 3: Commit**

```bash
git add scripts/box_launch_l2_screen.sh
git commit -m "scripts: L2 cold-start screen launcher with encoder-parity pre-flight"
```

---

### Task 5: Run the three cells and seed-matched-evaluate on L2

**Files:** none (uses the launcher + `scripts/eval_l3_seed_matched.py`). This is a run-and-decide task; no code commit.

- [ ] **Step 1: Launch the three cells (on the vast box, in parallel)**

```bash
bash scripts/box_launch_l2_screen.sh l2scr_ref    0 256
bash scripts/box_launch_l2_screen.sh l2scr_omega  1 256
bash scripts/box_launch_l2_screen.sh l2scr_cap512 0 512
```
Each runs cold on the single L2 stage. Watch the `capB` log for `|tau|/alpha_max` (the saturation diagnostic) — it should sit near the rl_song ~0.08, not climb toward ~0.47.

- [ ] **Step 2: Seed-matched L2 eval of each checkpoint (set the matching env var!)**

The eval runs `controller_numpy.py`, so the obs/width toggles MUST match how the cell was trained, or the loaded normalizer/weights mismatch the rebuilt obs/net:

```bash
cd /home/exedev/lsy_drone_racing
# ref (52-d, 256): no toggles
pixi run -e rl-train python scripts/eval_l3_seed_matched.py \
  --checkpoint <l2scr_ref dir> --config level2.toml \
  --controller rl_sbx/controller_numpy.py --control-mode attitude \
  --n-runs 100 --base-seed 0 --out l2scr_ref.json
# omegaA (55-d, 256): RL_OBS_ANG_VEL=1
RL_OBS_ANG_VEL=1 pixi run -e rl-train python scripts/eval_l3_seed_matched.py \
  --checkpoint <l2scr_omega dir> --config level2.toml \
  --controller rl_sbx/controller_numpy.py --control-mode attitude \
  --n-runs 100 --base-seed 0 --out l2scr_omega.json
# capB (52-d, 512): RL_HIDDEN_SIZE=512
RL_HIDDEN_SIZE=512 pixi run -e rl-train python scripts/eval_l3_seed_matched.py \
  --checkpoint <l2scr_cap512 dir> --config level2.toml \
  --controller rl_sbx/controller_numpy.py --control-mode attitude \
  --n-runs 100 --base-seed 0 --out l2scr_cap512.json
```
All cells share `--base-seed 0 --n-runs 100`, so results are seed-matched.

- [ ] **Step 3: Compare and decide**

Report per-cell success rate + lap time + union-of-seeds SR (the union is computed across the three JSONs' per-seed `finished` flags). Decision rule (spec §2, §8):
- **ω:** `omegaA` beating `ref` beyond seed-matched noise → ω helps; promote to L3 (spec §5). Net-neutral/negative → drop ω.
- **512:** read as a *diagnostic*, not a benefit test — clean training (`|tau|/alpha_max` not saturating, flies L2 comparably to `ref`) clears 512 for a real L3 capacity test; saturation/regression on the healthy stack is a genuine negative.

- [ ] **Step 4: Back up the eval-selected checkpoints**

Per the `gdrive-backup-best-ckpt-only` rule, push only the single eval-selected `step_*` dir per kept cell to gdrive (not whole run dirs).

---

## Self-review

**Spec coverage:** §3.1 ω channel → Tasks 1+2; §3.2 width → Task 3; §3.3 held-constant (curriculum=default cold, fixed reward recipe) → Task 4 launcher; §4 eval + pre-flight parity → Task 2 (parity) + Task 4 (pre-flight wiring) + Task 5 (seed-matched L2 eval, `|tau|/alpha_max`); §6 checklist: two encoders in lockstep (Tasks 1,2), encoder parity (Task 2), ω body-frame raw (Task 1 Step 4), no hard-coded 52/104 (`ACTOR_OBS_DIM`-driven, unchanged consumers), normalizer re-init (automatic, fresh cold run), deploy coupling (Task 5 Step 2 env-var note). All covered.

**Placeholder scan:** the only `<...>` are concrete run-dir paths in Task 5, unknowable until the runs exist — left as explicit placeholders by necessity, not vagueness. No TBD/TODO.

**Type/name consistency:** `ACTOR_OBS_ANG_VEL_DIM`, `RL_OBS_ANG_VEL`, `RL_HIDDEN_SIZE`, `HIDDEN_SIZE`, `NET_ARCH`, `build_actor_obs`, `NormalizerState`, `init_normalizer`, `NORM_VAR_EPS`, `NORM_CLIP` used consistently across tasks and match the source files read.
