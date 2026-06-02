"""Unit tests for the throughput-log parser (scripts/parse_prof_fps.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_SCRIPT = Path(__file__).parents[2] / "scripts" / "parse_prof_fps.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("parse_prof_fps", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dump(total: int, elapsed: int, scan: float, host: float, upd: float) -> str:
    return (
        "| time/              |          |\n"
        f"|    total_timesteps | {total} |\n"
        f"|    time_elapsed    | {elapsed} |\n"
        f"|    prof_scan_s     | {scan} |\n"
        f"|    prof_host_s     | {host} |\n"
        f"|    prof_update_plus_log_s | {upd} |\n"
    )


def _log() -> str:
    # 5 dumps; constant 4194304 steps / 40 s after the 2 compile dumps.
    rows = [
        _dump(4194304, 60, 9.0, 3.0, 4.0),
        _dump(8388608, 110, 6.0, 1.5, 2.5),
        _dump(12582912, 150, 5.0, 1.0, 2.0),
        _dump(16777216, 190, 5.0, 1.0, 2.0),
        _dump(20971520, 230, 5.0, 1.0, 2.0),
    ]
    return "".join(rows)


def test_summarize_drops_compile_dumps_and_medians():
    mod = _load()
    out = mod.summarize(_log())
    assert out["median_fps"] == pytest.approx(4194304 / 40, rel=1e-9)
    assert out["prof_scan_s"] == pytest.approx(5.0)
    assert out["prof_host_s"] == pytest.approx(1.0)
    assert out["prof_update_plus_log_s"] == pytest.approx(2.0)


def test_summarize_too_few_dumps_returns_nan():
    mod = _load()
    import math

    out = mod.summarize(_dump(1, 1, 1.0, 1.0, 1.0))
    assert math.isnan(out["median_fps"])
