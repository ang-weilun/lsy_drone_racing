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
HOVER_WINDOW_STEPS: int = 20  # 0.4 s at 50 Hz
HOVER_XY_BBOX_M: float = 0.15
NEAR_MISS_DIST_M: float = 0.20
WOBBLE_ANG_VEL_RAD_S: float = 6.0
WOBBLE_MIN_DURATION_STEPS: int = 10  # 0.2 s
TAKEOFF_Z_M: float = 0.10
FLOOR_Z_M: float = 0.05
COLLISION_RECENT_WINDOW: int = 5  # frames pre-terminal


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
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )


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
        events.append(
            {
                "type": "gate_pass",
                "t": float(row["t"]),
                "gate": int(passed_idx),
                "speed": speed,
                "in_plane_offset_m": offset,
                "angle_off_normal_rad": float(np.arccos(abs(cos_angle))),
            }
        )
    return events


def detect_hovers(ep: Episode) -> list[dict[str, Any]]:
    """Detect xy-stationary windows.

    A frame is part of a hover when, over a sliding window of
    ``HOVER_WINDOW_STEPS`` frames ending at that frame, the bounding-box
    extent on both ``x`` and ``y`` axes is below ``HOVER_XY_BBOX_M``.
    Adjacent hover frames are coalesced into one event.

    Parameters
    ----------
    ep : Episode
        Loaded episode with header and per-step rows.

    Returns
    -------
    list of dict
        Each event has type ``"hover"`` plus ``t_start``, ``t_end``,
        ``duration_s``, ``xy_bbox_extent_m``, ``mean_pos``, ``near_gate``.
    """
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

    events: list[dict[str, Any]] = []
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
    """Build a hover event from a (start, end) frame range."""
    rows = ep.rows[i_start : i_end + 1]
    xy = np.array([r["pos"][:2] for r in rows])
    mean_pos = np.array([r["pos"] for r in rows]).mean(axis=0)
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


# Gate aperture half-extents (m) — must match obs.GATE_HALF_SIZE_M.
# Source: lsy_drone_racing/control/rl_song/obs.py:46.
_GATE_HALF_Y: float = 0.20
_GATE_HALF_Z: float = 0.20

# Gate aperture corners in gate-local coords (x=0, +/- half_y, +/- half_z).
_GATE_FRAME_CORNERS_LOCAL = np.array(
    [
        [0.0, +_GATE_HALF_Y, +_GATE_HALF_Z],
        [0.0, +_GATE_HALF_Y, -_GATE_HALF_Z],
        [0.0, -_GATE_HALF_Y, +_GATE_HALF_Z],
        [0.0, -_GATE_HALF_Y, -_GATE_HALF_Z],
    ]
)
# Edge endpoint pairs: right-vertical, left-vertical, top-horiz, bottom-horiz.
_GATE_FRAME_EDGES: list[tuple[int, int]] = [(0, 1), (2, 3), (0, 2), (1, 3)]


def _gate_frame_edge_dist(pos: np.ndarray, gate_pos: np.ndarray, gate_quat: np.ndarray) -> float:
    """Minimum distance from ``pos`` to any of the four gate-frame edges.

    Mirrors ``reward._gate_frame_edge_dist_sq`` (sqrt'd for human readability).
    """
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


def detect_near_misses(ep: Episode) -> list[dict[str, Any]]:
    """Close-approach to a gate frame that did NOT result in a pass.

    Walks the episode while the same target_gate persists. If the
    drone's minimum distance to that gate's frame edges drops below
    ``NEAR_MISS_DIST_M`` and that gate is never subsequently passed,
    record one ``near_miss`` event. Stops after the first miss (rest
    of the trace is typically post-failure).
    """
    events: list[dict[str, Any]] = []
    seen_pass: set[int] = set()
    n = len(ep.rows)
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
            advanced = any(r["target_gate"] != tg for r in ep.rows[i + 1 :])
            if not advanced:
                events.append(
                    {
                        "type": "near_miss",
                        "t": float(row["t"]),
                        "gate": int(tg),
                        "closest_frame_dist_m": d,
                        "passed": False,
                    }
                )
                seen_pass.add(tg)
                break
    return events


def _point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point ``p`` to the line segment ``ab`` (3D)."""
    ab = b - a
    ap = p - a
    ab_sq = ab @ ab
    t = float(np.clip((ap @ ab) / max(ab_sq, 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _resolve_collision_object(
    pos: np.ndarray, gates_pos: np.ndarray, obstacles_top: np.ndarray
) -> tuple[str, float]:
    """Return ``(object_label, distance)`` for the nearest collision candidate.

    Parameters
    ----------
    pos : ndarray, shape (3,)
        Drone position to test.
    gates_pos : ndarray, shape (n_gates, 3)
        Ground-truth gate centroids.
    obstacles_top : ndarray, shape (n_obstacles, 3)
        Ground-truth obstacle top markers. The obstacle is modelled as
        a vertical capsule from ``(x, y, z_top)`` down to ``(x, y, 0)``;
        ``obstacles_pos`` is the top marker, not the centroid (codex
        review), so a point-to-point distance would over-estimate
        clearance for tall obstacles.

    Returns
    -------
    tuple of (str, float)
        Label of the form ``"gate:<i>"``, ``"obstacle:<i>"``, or
        ``"floor"``, and the distance to that object. The floor plane
        is added as a candidate only when ``pos[2] < FLOOR_Z_M``.
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


def detect_collision(ep: Episode, outcome: dict[str, Any]) -> dict[str, Any] | None:
    """Detect a collision event from the pre-terminal frame.

    Parameters
    ----------
    ep : Episode
        Loaded episode with header and per-step rows.
    outcome : dict
        Result of :func:`detect_outcome`. Collision is only emitted when
        the episode did not finish and was not truncated.

    Returns
    -------
    dict or None
        Event with ``type="collision"``, ``t`` (seconds), ``object``
        label (``gate:<i>`` / ``obstacle:<i>`` / ``floor``),
        ``approach_speed_50hz`` (m/s), ``last_pos_50hz_pre_terminal``
        (raw JSON list of length 3), and ``min_approach_dist_5frame_m``
        (the minimum approach distance to the inferred object across
        the last ``COLLISION_RECENT_WINDOW`` pre-terminal frames).
        ``None`` when the outcome is ``finished`` or ``truncated``, or
        the episode has fewer than two rows.

    Notes
    -----
    Uses ``pos[T-1]`` (the last pre-terminal frame) because the sim
    warps the drone before producing the terminal observation, so
    ``pos[T]`` is the reset pose rather than the impact pose (see
    ``reward.py:443-446``). At 50 Hz this is up to 20 ms before the
    actual contact; the 5-frame robustness window catches
    glance-then-warp cases where the closest approach happened a few
    steps before the terminal frame.
    """
    if outcome["finished"] or outcome["terminal_cause"] == "truncated":
        return None
    rows = ep.rows
    if len(rows) < 2:
        return None
    i = len(rows) - 2  # last pre-terminal frame
    row = rows[i]
    pos = np.asarray(row["pos"])
    gates_pos = np.asarray(row["gates_pos_true"])
    obstacles_top = np.asarray(row["obstacles_pos_true"])
    label, _ = _resolve_collision_object(pos, gates_pos, obstacles_top)

    # 5-frame robustness: minimum approach distance to the inferred object.
    start = max(0, i - COLLISION_RECENT_WINDOW + 1)
    distances: list[float] = []
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


def detect_wobbles(ep: Episode) -> list[dict[str, Any]]:
    """Detect runs of sustained high angular velocity.

    A frame is "high" when the body-rate magnitude exceeds
    ``WOBBLE_ANG_VEL_RAD_S``. Contiguous high frames are coalesced into a
    single event when the run length is at least
    ``WOBBLE_MIN_DURATION_STEPS`` frames.

    Parameters
    ----------
    ep : Episode
        Loaded episode with header and per-step rows.

    Returns
    -------
    list of dict
        Each event has ``type="wobble"`` plus ``t_start``, ``t_end``,
        ``duration_s``, and ``max_ang_vel_rad_s`` (peak magnitude over
        the run). Empty when no run is long enough.
    """
    n = len(ep.rows)
    mag = np.array([np.linalg.norm(r["ang_vel"]) for r in ep.rows])
    high = mag > WOBBLE_ANG_VEL_RAD_S

    events: list[dict[str, Any]] = []
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
    """Build a wobble event from a (start, end) frame range."""
    return {
        "type": "wobble",
        "t_start": float(ep.rows[i_start]["t"]),
        "t_end": float(ep.rows[i_end]["t"]),
        "duration_s": float(ep.rows[i_end]["t"] - ep.rows[i_start]["t"]),
        "max_ang_vel_rad_s": float(mag[i_start : i_end + 1].max()),
    }


def integrate_reward(ep: Episode) -> dict[str, Any] | None:
    """Sum reward terms across the episode.

    Parameters
    ----------
    ep : Episode
        Loaded episode with header and per-step rows.

    Returns
    -------
    dict or None
        ``None`` if any row has ``reward_terms == None`` (back-compat
        path when no ``reward_config.json`` was available at eval time).
        Otherwise: ``total`` (float), ``by_term`` (dict[str, float] of
        per-component sums), ``dominant_positive`` (term key with
        largest positive sum, or ``None`` if no positives),
        ``dominant_negative`` (term key with most negative sum, or
        ``None`` if no negatives).
    """
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
        "dominant_positive": (
            max(positives, key=lambda k: positives[k]) if positives else None
        ),
        "dominant_negative": (
            min(negatives, key=lambda k: negatives[k]) if negatives else None
        ),
    }


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
