"""Numpy observation-normalizer utilities for SBX deploy."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from lsy_drone_racing.control.rl_sbx.deploy_numpy.constants import (
    read_rl_song_obs_constant,
)

# Per-feature normalized observation clip range.
NORM_CLIP: float = float(read_rl_song_obs_constant("NORM_CLIP"))

# Variance epsilon used by the training-time Welford normalizer.
NORM_VAR_EPS: float = float(read_rl_song_obs_constant("NORM_VAR_EPS"))


class NormalizerState(NamedTuple):
    """Frozen Welford running-statistics state.

    Attributes
    ----------
    mean : ndarray, shape (ACTOR_OBS_DIM,)
        Running mean for each actor observation feature.
    var : ndarray, shape (ACTOR_OBS_DIM,)
        Running variance for each actor observation feature.
    count : ndarray, shape ()
        Running sample count. Kept for checkpoint parity; not used at deploy.
    """

    mean: npt.NDArray[np.floating]
    var: npt.NDArray[np.floating]
    count: npt.NDArray[np.floating]


def from_jax_state(state: object) -> NormalizerState:
    """Convert a checkpoint normalizer object to numpy arrays.

    Parameters
    ----------
    state : object
        Object with ``mean``, ``var``, and ``count`` attributes.

    Returns:
    -------
    NormalizerState
        Numpy-backed normalizer state.
    """
    return NormalizerState(
        mean=np.asarray(getattr(state, "mean"), dtype=np.float32),
        var=np.asarray(getattr(state, "var"), dtype=np.float32),
        count=np.asarray(getattr(state, "count"), dtype=np.float32),
    )


def apply_normalizer(
    state: NormalizerState, x: npt.NDArray[np.floating]
) -> npt.NDArray[np.float32]:
    """Normalize and clip an actor observation.

    Parameters
    ----------
    state : NormalizerState
        Frozen running statistics.
    x : ndarray, shape (ACTOR_OBS_DIM,)
        Raw actor observation.

    Returns:
    -------
    normalized : ndarray, shape (ACTOR_OBS_DIM,)
        Normalized observation clipped to the training-time range.
    """
    std = np.sqrt(state.var + NORM_VAR_EPS)
    normalized = np.clip((x - state.mean) / std, -NORM_CLIP, NORM_CLIP)
    return normalized.astype(np.float32, copy=False)
