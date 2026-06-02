"""Summarise an rl_sbx training log: steady instantaneous fps + prof buckets.

SB3 ``time/fps`` is cumulative, so instantaneous fps is the consecutive diff of
``total_timesteps`` over ``time_elapsed``. The first two dumps are dropped (JIT
compile inflation). Usage: ``python scripts/parse_prof_fps.py <log>``.
"""

from __future__ import annotations

import math
import re
import sys

import numpy as np

# SB3's --no-wandb table shows leaf labels, not the full "time/..." path.
_BUCKETS = ("prof_scan_s", "prof_host_s", "prof_update_plus_log_s")


def _column(log: str, key: str) -> list[float]:
    return [float(m.group(1)) for m in re.finditer(rf"{re.escape(key)}\s*\|\s*([0-9.eE+-]+)", log)]


def summarize(log: str) -> dict[str, float]:
    """Return median steady fps and median prof-bucket seconds (NaN if too short)."""
    steps = _column(log, "total_timesteps")
    elapsed = _column(log, "time_elapsed")
    n = min(len(steps), len(elapsed))
    out: dict[str, float] = {"n_dumps": float(n)}

    if n < 4:
        out["median_fps"] = math.nan
    else:
        steps_a, elapsed_a = np.array(steps[2:n]), np.array(elapsed[2:n])
        inst = np.diff(steps_a) / np.maximum(np.diff(elapsed_a), 1e-9)
        out["median_fps"] = float(np.median(inst))

    for key in _BUCKETS:
        vals = _column(log, key)
        out[key] = float(np.median(vals[2:])) if len(vals) > 2 else math.nan
    return out


def main(path: str) -> None:
    """Print the summary for a log file path."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        out = summarize(fh.read())
    print(f"steady instantaneous fps  median={out['median_fps']:,.0f}")
    for key in _BUCKETS:
        print(f"{key:24s} median={out[key]:.4f}s")


if __name__ == "__main__":
    main(sys.argv[1])
