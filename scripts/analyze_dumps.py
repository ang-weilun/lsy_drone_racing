"""Classify and inspect failure modes from sweep_dump.py output.

Usage:
    pixi run python scripts/analyze_dumps.py --dump_dir /tmp/sfc_sweep_v3
    pixi run python scripts/analyze_dumps.py --dump_dir /tmp/sfc_sweep_v3 --inspect_seed 1
"""

from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
from scipy.spatial.transform import Rotation as R


def _load(path: Path) -> dict:
    return dict(np.load(path, allow_pickle=True))


def _gate_bars_xy(gate_pos: np.ndarray, gate_quat: np.ndarray, bar_dist: float = 0.28):
    """Return the two side-bar xy positions for a gate (matching planner geometry)."""
    gate_pos = np.asarray(gate_pos, dtype=np.float64)
    gate_quat = np.asarray(gate_quat, dtype=np.float64)
    rot = R.from_quat(gate_quat)
    right = rot.apply([0, 1, 0])
    return (gate_pos[:2] - right[:2] * bar_dist, gate_pos[:2] + right[:2] * bar_dist)


def summarize(dump_dir: str = "/tmp/sfc_sweep_v3") -> None:
    """Print a summary of all dumps in dump_dir."""
    p = Path(dump_dir)
    files = sorted(p.glob("run_*.npz"))
    print(f"# {len(files)} dumps in {dump_dir}\n")
    print(f"{'seed':>5} {'gates':>5} {'reason':>12} {'t':>5} "
          f"{'tg@last':>8} {'pos_at_term':>22} {'dist_to_g2_bar':>15}")
    rows = []
    for f in files:
        d = _load(f)
        n = int(d["meta_n_ticks"])
        if n == 0:
            continue
        seed = int(f.stem.split("_")[1])
        reason = str(d["meta_terminated_reason"])
        gates_passed_at_end = int(d["target_gate_idx"][-1])
        t_end = float(d["t"][-1])
        pos_term = d["pos"][-1]

        # Per-tick min distance from drone xy to any gate's side bars
        gates_pos = d["replan_gates_pos"][-1]
        gates_quat = d["replan_gates_quat"][-1]
        bars_xy = []
        for gp, gq in zip(gates_pos, gates_quat):
            b1, b2 = _gate_bars_xy(gp, gq)
            bars_xy.extend([b1, b2])
        bars_xy = np.asarray(bars_xy)

        # Track which gate's bars are closest at termination
        if gates_passed_at_end > 0 and gates_passed_at_end <= len(gates_pos):
            prev_gate = gates_passed_at_end - 1
            b1, b2 = _gate_bars_xy(gates_pos[prev_gate], gates_quat[prev_gate])
            d1 = np.linalg.norm(pos_term[:2] - b1)
            d2 = np.linalg.norm(pos_term[:2] - b2)
            dist_to_prev_gate_bar = min(d1, d2)
        else:
            dist_to_prev_gate_bar = float("nan")

        print(
            f"{seed:>5} {gates_passed_at_end:>5} {reason:>12} {t_end:>5.2f} "
            f"{int(d['meta_final_target_gate']):>8} "
            f"({pos_term[0]:+.2f},{pos_term[1]:+.2f},{pos_term[2]:+.2f})  "
            f"{dist_to_prev_gate_bar:>14.3f}"
        )
        rows.append(
            {
                "seed": seed,
                "gates": gates_passed_at_end,
                "reason": reason,
                "dist_to_prev_gate_bar": dist_to_prev_gate_bar,
            }
        )

    # Aggregate: failure modes
    n = len(rows)
    finishes = sum(1 for r in rows if r["reason"] == "finished")
    g3_coll = sum(1 for r in rows if r["gates"] == 3 and r["reason"] == "collision")
    g3_close_to_bar = sum(
        1
        for r in rows
        if r["gates"] == 3 and r["reason"] == "collision" and r["dist_to_prev_gate_bar"] < 0.30
    )
    print(
        f"\n# {n} runs: {finishes} finished, {g3_coll} gate-3 collisions "
        f"({g3_close_to_bar} terminate <0.30m from a gate-2 side bar)"
    )


def inspect(dump_dir: str = "/tmp/sfc_sweep_v3", seed: int = 1) -> None:
    """Inspect a single run in detail, focused on the gate-3 frame-clip mode."""
    f = Path(dump_dir) / f"run_{seed:04d}.npz"
    d = _load(f)
    print(f"# inspecting {f}")
    print(f"reason       = {d['meta_terminated_reason']}")
    print(f"final_target = {int(d['meta_final_target_gate'])}")
    n = int(d["meta_n_ticks"])
    print(f"n_ticks      = {n}")
    print(f"n_replans    = {int(d['meta_n_replans'])}")
    pos_term = d["pos"][-1]
    print(f"pos_at_term  = ({pos_term[0]:+.3f}, {pos_term[1]:+.3f}, {pos_term[2]:+.3f})")

    gates_pos = d["replan_gates_pos"][-1]
    gates_quat = d["replan_gates_quat"][-1]

    # Find the moment target_gate transitions to last (gate count - 1)
    tg = d["target_gate_idx"]
    transitions = [i for i in range(1, n) if tg[i] != tg[i - 1]]
    for ti in transitions:
        p = d["pos"][ti]
        print(
            f"  target_gate {tg[ti - 1]}->{tg[ti]} at tick {int(d['tick'][ti])} "
            f"t={d['t'][ti]:.2f} pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})"
        )

    # Final replan = the planner's path to the last gate. Show its anchors + spline samples
    last_replan = -1
    print(f"\n# Final replan (index {len(d['replan_anchors_pos']) + last_replan})")
    print(f"  target_gate_idx = {int(d['replan_target_gate_idx'][last_replan])}")
    anchors = d["replan_anchors_pos"][last_replan]
    is_gate = d["replan_anchors_is_gate"][last_replan]
    for i, (p, g) in enumerate(zip(anchors, is_gate)):
        flag = "GATE" if g else "    "
        print(f"  a{i:>2} {flag}: ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")

    # Show min distance from final-replan spline to each gate's side bars
    from scipy.interpolate import BSpline

    ctrl = d["replan_control_points"][last_replan]
    knots = d["replan_knots"][last_replan]
    sp = BSpline(knots, ctrl, 3)
    samples = sp(np.linspace(0, 1, 200))
    print("\n# Spline-sample min distance (xy) to each gate's side-bar centers:")
    for gi, (gp, gq) in enumerate(zip(gates_pos, gates_quat)):
        b1, b2 = _gate_bars_xy(gp, gq)
        for label, b in [("R", b1), ("L", b2)]:
            dists = np.linalg.norm(samples[:, :2] - b, axis=1)
            argmin = int(np.argmin(dists))
            s = samples[argmin]
            print(
                f"  gate{gi} bar-{label} at ({b[0]:+.3f},{b[1]:+.3f}): "
                f"min={dists.min():.3f}m at sample[{argmin}]=({s[0]:+.3f},{s[1]:+.3f},{s[2]:+.3f})"
            )

    # Also: along the actual drone trajectory in the last replan
    last_replan_tick = int(d["replan_tick"][last_replan])
    actual = d["pos"][d["tick"] >= last_replan_tick]
    if len(actual) > 0:
        print("\n# Actual-path min distance (xy) to each gate's side-bar centers:")
        for gi, (gp, gq) in enumerate(zip(gates_pos, gates_quat)):
            b1, b2 = _gate_bars_xy(gp, gq)
            for label, b in [("R", b1), ("L", b2)]:
                dists = np.linalg.norm(actual[:, :2] - b, axis=1)
                argmin = int(np.argmin(dists))
                s = actual[argmin]
                print(
                    f"  gate{gi} bar-{label}: min={dists.min():.3f}m "
                    f"at pos=({s[0]:+.3f},{s[1]:+.3f},{s[2]:+.3f})"
                )


if __name__ == "__main__":
    fire.Fire({"summarize": summarize, "inspect": inspect})
