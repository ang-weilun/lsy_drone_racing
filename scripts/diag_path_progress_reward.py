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
    """Convert a yaw angle (radians) to a unit quaternion [x, y, z, w]."""
    return Rotation.from_euler("z", yaw).as_quat().tolist()


def obs(
    pos: list[float], target: int, gates_pos: list[list[float]], gates_quat: list[list[float]]
) -> dict:
    """Build a minimal batched observation dict (batch size 1) for step_reward."""
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


def r_prog(
    prev_pos: list[float],
    cur_pos: list[float],
    target: int,
    gates_pos: list[list[float]],
    gates_quat: list[list[float]],
    cfg: RewardConfig,
) -> float:
    """Return the r_prog component of step_reward for a single transition."""
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


def integrated_rprog(
    positions: list[list[float]],
    target: int,
    gates_pos: list[list[float]],
    gates_quat: list[list[float]],
    cfg: RewardConfig,
) -> float:
    """Sum r_prog over consecutive positions along a scripted trajectory."""
    return sum(
        r_prog(positions[i], positions[i + 1], target, gates_pos, gates_quat, cfg)
        for i in range(len(positions) - 1)
    )


def r_prog_passstep(
    prev_pos: list[float],
    cur_pos: list[float],
    prev_target: int,
    cur_target: int,
    gates_pos: list[list[float]],
    gates_quat: list[list[float]],
    cfg: RewardConfig,
) -> float:
    """Return r_prog on the real gate-pass handoff step (gate_just_passed=True)."""
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
    """Run all scripted-trajectory checks and exit 0 on full pass, 1 on any failure."""
    base = RewardConfig()
    fix = RewardConfig(use_path_progress=True, path_exit_offset_m=0.4, path_entry_offset_m=0.4)
    ok = True
    print(f"{'scenario':<34}{'baseline':>10}{'guiding':>10}{'verdict':>9}")

    def row(
        name: str,
        prev: list[float],
        cur: list[float],
        tgt: int,
        gp: list[list[float]],
        gq: list[list[float]],
        want: str,
    ) -> None:
        nonlocal ok
        b = r_prog(prev, cur, tgt, gp, gq, base)
        g = r_prog(prev, cur, tgt, gp, gq, fix)
        passed = {"neg": g < -TOL, "pos": g > TOL, "zero": abs(g) <= TOL}[want]
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
