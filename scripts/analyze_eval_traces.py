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
