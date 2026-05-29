"""Source-read constants for numpy deploy without importing JAX modules."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

# Modules that hold the obs-side constants. Search order matters only when a
# name is (accidentally) defined in more than one file; the first hit wins.
# ``config.py`` was added after ``N_OBSTACLES`` migrated out of ``obs.py`` —
# config.py is now the canonical home for track-geometry shape constants
# (``N_OBSTACLES``, ``N_NEAREST_OBSTACLES``) while ``obs.py`` retains the
# encoding-only constants (``N_FUTURE_GATES``, ``GATE_HALF_SIZE_M``, etc.).
_RL_SONG_DIR: Path = Path(__file__).resolve().parents[2] / "rl_song"
_SEARCH_PATHS: tuple[Path, ...] = (_RL_SONG_DIR / "obs.py", _RL_SONG_DIR / "config.py")


def read_rl_song_obs_constant(name: str) -> Any:
    """Read a module-level constant from the rl_song stack without importing JAX.

    Parameters
    ----------
    name : str
        Constant identifier (e.g. ``"N_OBSTACLES"``).

    Returns:
    -------
    Any
        The literal value parsed from the source file. Only constants
        defined as a top-level ``name: type = literal`` annotated
        assignment are reachable; module-level computed expressions are
        not supported.

    Raises:
    ------
    ValueError
        If ``name`` is not found as a top-level annotated assignment in
        any of the searched files.
    """
    for path in _SEARCH_PATHS:
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in module.body:
            if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
                continue
            if node.target.id == name:
                return ast.literal_eval(node.value)
    searched = ", ".join(str(p) for p in _SEARCH_PATHS)
    raise ValueError(f"Could not read {name} from any of: {searched}")
