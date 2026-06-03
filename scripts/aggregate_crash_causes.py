"""Aggregate crash causes from an eval-trace dump, split by gate target-ness.

Consumes the ``trace/episode_NNN.jsonl`` dumps that ``analyze_eval_traces.py``
reads (produced by ``eval_sim --dump_trace``) and answers what the run-summary
does not break out: of the episodes that ended in a collision, how many hit a
**gate frame** vs an **obstacle** vs the **floor**, and -- for gate hits -- was
it the *current target* gate or a *non-target* gate (a gate the policy is not
aiming at, which the actor obs never encodes).

The collision object label reuses :func:`analyze_eval_traces.detect_collision`
(nearest gate-centroid / obstacle-segment / floor at the last pre-terminal
frame). Because the centroid heuristic over-attributes to gates (the drone is
always near a gate centroid when passing through), every gate label is then
**validated** against the minimum distance to the gate's aperture-edge lines
over the last few frames (:func:`analyze_eval_traces._gate_frame_edge_dist`):
a genuine frame strike puts the drone within ~0.15 m of an edge, whereas a
clean centre pass sits ~0.20 m off every edge. Gate crashes are bucketed by
that distance so the centroid-mislabel rate is visible, and the target /
non-target split is reported over the *confirmed-strike* subset as well as raw.

Usage::

    pixi run -e rl-train python scripts/aggregate_crash_causes.py <trace_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_eval_traces import (  # noqa: E402
    _gate_frame_edge_dist,
    detect_collision,
    detect_outcome,
    load_episode,
    load_run_meta,
)

# Min distance (m) from drone centre to an aperture-edge line below which we
# treat a gate-labelled crash as a genuine frame strike. A clean centre pass is
# ~0.20 m off every edge; the physical frame band spans 0.20-0.36 m off-centre,
# so a strike (accounting for the ~0.05 m drone radius) lands the centre within
# ~0.15 m of the nearest edge line. 0.30 m is the borderline ceiling above which
# the gate label is almost certainly a centroid mis-attribution.
_STRIKE_DIST_M: float = 0.15
_BORDERLINE_DIST_M: float = 0.30
_FRAME_WINDOW: int = 5  # frames pre-terminal to take the min frame-edge dist over


def _min_frame_edge_dist(ep: Any, gate_idx: int) -> float:
    """Min drone-to-aperture-edge distance over the last ``_FRAME_WINDOW`` frames."""
    rows = ep.rows
    i = len(rows) - 2
    start = max(0, i - _FRAME_WINDOW + 1)
    best = float("inf")
    for j in range(start, i + 1):
        rj = rows[j]
        d = _gate_frame_edge_dist(
            np.asarray(rj["pos"]),
            np.asarray(rj["gates_pos_true"][gate_idx]),
            np.asarray(rj["gates_quat_true"][gate_idx]),
        )
        best = min(best, d)
    return best


def aggregate(trace_dir: str) -> None:
    """Tally collision causes over every episode JSONL in ``trace_dir``."""
    tdir = Path(trace_dir)
    meta = load_run_meta(tdir)
    ep_paths = sorted(tdir.glob("episode_*.jsonl"))
    if not ep_paths:
        raise FileNotFoundError(f"no episode_*.jsonl under {tdir}")

    n_total = len(ep_paths)
    n_finished = 0
    n_truncated = 0
    causes: dict[str, int] = {
        "floor": 0,
        "obstacle": 0,
        "gate_target": 0,
        "gate_nontarget": 0,
        "unknown": 0,
    }
    nontarget_detail: dict[str, int] = {"just_passed": 0, "downstream": 0, "behind": 0}
    # Gate crashes bucketed by validated frame-edge proximity.
    gate_strike: dict[str, int] = {"strike": 0, "borderline": 0, "mislabel": 0}
    # Confirmed strikes (edge dist < _STRIKE_DIST_M) split by target-ness.
    strike_split: dict[str, int] = {"target": 0, "nontarget": 0}
    strike_nontarget_detail: dict[str, int] = {"just_passed": 0, "downstream": 0, "behind": 0}
    gate_edge_dists: list[float] = []
    unknown_detail: list[dict[str, Any]] = []
    per_gate_hits: dict[int, int] = {}

    for ep_path in ep_paths:
        ep = load_episode(ep_path)
        outcome = detect_outcome(ep)
        if outcome["finished"]:
            n_finished += 1
            continue
        if outcome["terminal_cause"] == "truncated":
            n_truncated += 1
            continue

        coll = detect_collision(ep, outcome)
        if coll is None:
            causes["unknown"] += 1
            unknown_detail.append(
                {
                    "terminal_cause": outcome["terminal_cause"],
                    "ep_len_steps": outcome.get("ep_len_steps"),
                }
            )
            continue

        label = coll["object"]
        if label == "floor":
            causes["floor"] += 1
        elif label.startswith("obstacle:"):
            causes["obstacle"] += 1
        elif label.startswith("gate:"):
            gate_idx = int(label.split(":")[1])
            per_gate_hits[gate_idx] = per_gate_hits.get(gate_idx, 0) + 1
            target = int(ep.rows[len(ep.rows) - 2]["target_gate"])
            is_target = gate_idx == target
            if is_target:
                causes["gate_target"] += 1
            else:
                causes["gate_nontarget"] += 1

            if gate_idx == target - 1:
                bucket = "just_passed"
            elif target >= 0 and gate_idx > target:
                bucket = "downstream"
            else:
                bucket = "behind"
            if not is_target:
                nontarget_detail[bucket] += 1

            edge_d = _min_frame_edge_dist(ep, gate_idx)
            gate_edge_dists.append(round(edge_d, 3))
            if edge_d < _STRIKE_DIST_M:
                gate_strike["strike"] += 1
                strike_split["target" if is_target else "nontarget"] += 1
                if not is_target:
                    strike_nontarget_detail[bucket] += 1
            elif edge_d < _BORDERLINE_DIST_M:
                gate_strike["borderline"] += 1
            else:
                gate_strike["mislabel"] += 1
        else:
            causes["unknown"] += 1

    n_crash = n_total - n_finished - n_truncated
    gate_total = causes["gate_target"] + causes["gate_nontarget"]

    def _frac(x: int) -> float:
        return round(x / n_crash, 3) if n_crash else 0.0

    report: dict[str, Any] = {
        "checkpoint": meta.get("checkpoint"),
        "config": meta.get("config"),
        "n_episodes": n_total,
        "n_finished": n_finished,
        "n_truncated": n_truncated,
        "n_crash": n_crash,
        "success_rate": round(n_finished / n_total, 3),
        "crash_causes_raw": causes,
        "crash_cause_frac_of_crashes": {k: _frac(v) for k, v in causes.items()},
        "gate_frame_share_raw": _frac(gate_total),
        "obstacle_share": _frac(causes["obstacle"]),
        "nontarget_gate_detail_raw": nontarget_detail,
        "_validation": {
            "gate_label_buckets": gate_strike,
            "confirmed_strike_split": strike_split,
            "confirmed_strike_nontarget_detail": strike_nontarget_detail,
            "confirmed_gate_frame_share_of_crashes": _frac(gate_strike["strike"]),
            "gate_edge_dist_m_sorted": sorted(gate_edge_dists),
        },
        "per_gate_hit_histogram": dict(sorted(per_gate_hits.items())),
        "unknown_detail": unknown_detail,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    fire.Fire(aggregate)
