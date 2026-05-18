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
    """Read the per-run metadata JSON."""
    return json.loads((trace_dir / "run_meta.json").read_text(encoding="utf-8"))


def detect_outcome(ep: Episode) -> dict[str, Any]:
    """Compute outcome block: gates_passed, finished, terminal_cause.

    Parameters
    ----------
    ep : Episode
        Loaded episode with header and per-step rows.

    Returns
    -------
    dict
        Keys: ``gates_passed`` (int), ``finished`` (bool),
        ``ep_len_steps`` (int), ``flight_time_s`` (float),
        ``terminal_cause`` (str).

    Notes
    -----
    Finish detection uses ``target_gate`` transitioning to ``-1`` rather
    than ``terminated`` -- eval_sim breaks on finish before the env's
    sparse-reward path sets ``terminated``. ``terminal_cause`` resolves
    to ``"finished"``, ``"truncated"``, or a ``"collision:unknown"``
    placeholder; C6 patches the collision label in.
    """
    n_gates = ep.header["n_gates"]
    rows = ep.rows
    last = rows[-1]
    target_gates = [r["target_gate"] for r in rows]

    finished = any(t == -1 for t in target_gates)
    if finished:
        gates_passed = n_gates
    else:
        gates_passed = max(t for t in target_gates if t >= 0) if target_gates else 0

    if finished:
        terminal_cause = "finished"
    elif last["truncated"]:
        terminal_cause = "truncated"
    else:
        terminal_cause = "collision:unknown"  # C6 patches the label

    return {
        "gates_passed": int(gates_passed),
        "finished": bool(finished),
        "ep_len_steps": len(rows),
        "flight_time_s": float(rows[-1]["t"]),
        "terminal_cause": terminal_cause,
    }


def _quat_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    """xyzw quaternion -> 3x3 rotation matrix. Mirrors obs._quat_to_matrix."""
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
    """Stack per-step pos into (N, 3) array."""
    return np.array([r["pos"] for r in rows], dtype=np.float64)


def _rows_vel(rows: list[dict]) -> np.ndarray:
    """Stack per-step vel into (N, 3) array."""
    return np.array([r["vel"] for r in rows], dtype=np.float64)


def detect_takeoff(ep: Episode) -> dict[str, Any] | None:
    """First frame where pos.z exceeds TAKEOFF_Z_M.

    Parameters
    ----------
    ep : Episode
        Loaded episode with header and per-step rows.

    Returns
    -------
    dict or None
        Event with ``type="takeoff"``, ``t`` (seconds), and
        ``vz_at_liftoff`` (m/s). ``None`` if the drone never left the
        ground.
    """
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
    """One ``gate_pass`` event per ``target_gate`` advance.

    Parameters
    ----------
    ep : Episode
        Loaded episode with header and per-step rows.

    Returns
    -------
    list of dict
        Each event has ``type="gate_pass"``, ``t`` (seconds), ``gate``
        (int index of the gate just passed), ``speed`` (m/s),
        ``in_plane_offset_m`` (aperture-plane miss distance), and
        ``angle_off_normal_rad`` (velocity angle off gate normal, in
        ``[0, pi/2]`` since front/back passes are collapsed). Empty if
        no gates were passed.

    Notes
    -----
    Reads the ground-truth gate poses from ``gates_pos_true`` /
    ``gates_quat_true`` rather than the masked policy-view fields.
    Gate-local x-axis is forward through the aperture; (y, z) span the
    aperture plane.
    """
    events: list[dict[str, Any]] = []
    for i in range(1, len(ep.rows)):
        prev_tg = ep.rows[i - 1]["target_gate"]
        curr_tg = ep.rows[i]["target_gate"]
        if prev_tg < 0 or curr_tg == prev_tg:
            continue
        passed_idx = prev_tg
        row = ep.rows[i]
        gate_pos = np.asarray(row["gates_pos_true"][passed_idx])
        gate_quat = np.asarray(row["gates_quat_true"][passed_idx])
        rot_gw = _quat_to_rotmat(gate_quat)
        # Project drone position into gate-local frame: x = forward
        # through gate, y/z = aperture coords.
        rel = np.asarray(row["pos"]) - gate_pos
        local = rot_gw.T @ rel
        offset = float(np.linalg.norm(local[1:]))  # (y, z) magnitude
        vel = np.asarray(row["vel"])
        speed = float(np.linalg.norm(vel))
        forward = rot_gw[:, 0]
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


def analyze(trace_dir: str) -> None:
    """Analyze a trace directory and emit summary JSONs.

    Per-episode + rollup writers are added in subsequent tasks.
    """
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

    _ = run_meta  # consumed in C10 when the run rollup is wired
    print(f"Loaded {len(episodes)} episodes from {trace}")


if __name__ == "__main__":
    fire.Fire(analyze)
