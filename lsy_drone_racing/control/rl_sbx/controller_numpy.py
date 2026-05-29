"""Numpy-only deployment controller for the SBX-trained policy."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from drone_models.core import load_params

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.control.rl_sbx import checkpoint as ckpt
from lsy_drone_racing.control.rl_sbx.deploy_numpy import normalizer as norm_np
from lsy_drone_racing.control.rl_sbx.deploy_numpy import obs as obs_np
from lsy_drone_racing.control.rl_sbx.deploy_numpy import policy as policy_np
from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_DIM, ENV_ACTION_DIM

# Per-rotor drone-model thrust limits are multiplied to total thrust in newtons.
TOTAL_THRUST_MULTIPLIER: float = 4.0

# Repo root for resolving relative checkpoint paths.
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

# Step-directory prefix emitted by the SBX checkpoint writer.
_CHECKPOINT_PREFIX: str = "step_"


class RLSBXNumpyController(Controller):
    """Deterministic numpy deploy controller for the SBX-trained actor.

    Parameters
    ----------
    obs : dict[str, ndarray]
        Initial unbatched racing-env observation.
    info : dict
        Initial env info dictionary. Not used by this controller.
    config : dict
        Race configuration. Requires ``controller.checkpoint`` and the usual
        ``sim.physics`` / ``sim.drone_model`` entries.
    """

    def __init__(
        self, obs: dict[str, npt.NDArray[np.floating]], info: dict, config: dict
    ) -> None:
        """Load checkpoint arrays and precompute deploy constants."""
        super().__init__(obs, info, config)

        checkpoint_path = _resolve_checkpoint_path(
            _config_value(config, "controller", "checkpoint")
        )
        loaded = ckpt.load_actor_only(checkpoint_path)
        self.actor_params = _tree_to_numpy(loaded["actor_params"])
        self.actor_normalizer = norm_np.from_jax_state(loaded["actor_normalizer"])
        self.alpha_max_rad = float(loaded["tangent_alpha_max_rad"])

        physics = _config_value(config, "sim", "physics")
        drone_model = _config_value(config, "sim", "drone_model")
        drone_params = load_params(physics, drone_model)
        self.thrust_min = float(drone_params["thrust_min"] * TOTAL_THRUST_MULTIPLIER)
        self.thrust_max = float(drone_params["thrust_max"] * TOTAL_THRUST_MULTIPLIER)

        self.gate_corners_local = obs_np.gate_corners_local()
        self.prev_action_env_4vec = np.zeros((ENV_ACTION_DIM,), dtype=np.float32)

    def compute_control(
        self, obs: dict[str, npt.NDArray[np.floating]], info: dict | None = None
    ) -> npt.NDArray[np.float32]:
        """Run the actor and return a 4-d env action.

        Parameters
        ----------
        obs : dict[str, ndarray]
            Current unbatched racing-env observation.
        info : dict, optional
            Additional environment info. Not used.

        Returns:
        -------
        env_action : ndarray, shape (4,)
            Env attitude command ``[roll, pitch, yaw, thrust]``.
        """
        del info
        actor_obs = obs_np.build_actor_obs(
            obs,
            self.prev_action_env_4vec,
            self.actor_normalizer,
            gate_corners_local=self.gate_corners_local,
        )
        flat_obs = np.concatenate(
            [actor_obs, np.zeros((ACTOR_OBS_DIM,), dtype=actor_obs.dtype)], axis=-1
        )
        raw_action = policy_np.actor_mean(self.actor_params, flat_obs)
        env_action = policy_np.raw_to_env_action(
            raw_action,
            obs["quat"],
            self.thrust_min,
            self.thrust_max,
            alpha_max=self.alpha_max_rad,
        )
        self.prev_action_env_4vec = env_action
        return env_action

    def step_callback(
        self,
        action: npt.NDArray[np.floating],
        obs: dict[str, npt.NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Return ``False`` so the environment controls episode termination."""
        del action, obs, reward, terminated, truncated, info
        return False


def _tree_to_numpy(tree: Any) -> Any:
    """Recursively convert a nested checkpoint PyTree to numpy arrays."""
    if isinstance(tree, Mapping):
        return {key: _tree_to_numpy(value) for key, value in tree.items()}
    return np.asarray(tree)


def _resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    """Resolve a checkpoint path to a concrete ``step_*`` directory.

    Parameters
    ----------
    checkpoint : str or Path
        Single ``step_*`` directory or a run directory containing step dirs.

    Returns:
    -------
    path : Path
        Absolute concrete checkpoint step directory.
    """
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if path.is_dir() and path.name.startswith(_CHECKPOINT_PREFIX):
        return path
    latest = _latest_step_dir(path)
    if latest is None:
        raise FileNotFoundError(
            f"No {_CHECKPOINT_PREFIX}* subdirectories under: {path}"
        )
    return latest


def _latest_step_dir(run_dir: Path) -> Path | None:
    """Return the highest-numbered ``step_*`` subdirectory under ``run_dir``."""
    if not run_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.iterdir():
        if not path.is_dir() or not path.name.startswith(_CHECKPOINT_PREFIX):
            continue
        step_str = path.name.removeprefix(_CHECKPOINT_PREFIX)
        if step_str.isdecimal():
            candidates.append((int(step_str), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _config_value(config: Any, *keys: str) -> Any:
    """Read a nested config value from a mapping or config object."""
    value: Any = config
    for key in keys:
        if isinstance(value, Mapping):
            if key not in value:
                raise KeyError(f"Missing config key: {'.'.join(keys)}")
            value = value[key]
        else:
            if not hasattr(value, key):
                raise KeyError(f"Missing config key: {'.'.join(keys)}")
            value = getattr(value, key)
    return value
