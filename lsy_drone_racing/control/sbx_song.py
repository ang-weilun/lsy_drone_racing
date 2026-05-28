"""Deployment shim for the SBX-trained Song-style policy.

Loads actor params + actor normalizer + `tangent_alpha_max_rad` only; the
critic obs is unavailable on hardware (no privileged poses) so the loader is
structurally fenced off from any `critic.*` artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from drone_models.core import load_params

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.control.rl_sbx import checkpoint as ckpt
from lsy_drone_racing.control.rl_sbx.policy import LOG_STD_INIT, NET_ARCH, Actor
from lsy_drone_racing.control.rl_song import obs as obs_encoding
from lsy_drone_racing.control.rl_song.config import ENV_ACTION_DIM, RAW_ACTION_DIM
from lsy_drone_racing.control.rl_song.policy import raw_to_env_action

if TYPE_CHECKING:
    import numpy.typing as npt
    from jax import Array

# drone_models' thrust_min/max are per-rotor; the env command is total thrust.
TOTAL_THRUST_MULTIPLIER: float = 4.0

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT: str = "checkpoints/L2_finetune"
_MISSING: Any = object()


class RLSBXController(Controller):
    """Deterministic deploy controller for the SBX-trained actor."""

    def __init__(self, obs: dict[str, npt.NDArray[np.floating]], info: dict, config: dict) -> None:
        super().__init__(obs, info, config)

        checkpoint_path = _resolve_checkpoint_path(
            _config_value(config, "controller", "checkpoint", default=DEFAULT_CHECKPOINT)
        )
        loaded = ckpt.load_actor_only(checkpoint_path)
        self.actor_params: dict[str, Any] = loaded["actor_params"]
        self.actor_normalizer = loaded["actor_normalizer"]
        self.alpha_max_rad = float(loaded["tangent_alpha_max_rad"])

        physics = _config_value(config, "sim", "physics")
        drone_model = _config_value(config, "sim", "drone_model")
        drone_params = load_params(physics, drone_model)
        self.thrust_min = float(drone_params["thrust_min"] * TOTAL_THRUST_MULTIPLIER)
        self.thrust_max = float(drone_params["thrust_max"] * TOTAL_THRUST_MULTIPLIER)

        self.prev_action_env_4vec: Array = jnp.zeros((ENV_ACTION_DIM,), dtype=jnp.float32)

        self._actor = Actor(action_dim=RAW_ACTION_DIM, net_arch=NET_ARCH, log_std_init=LOG_STD_INIT)

        # JIT the full obs-encode -> actor -> projection chain. Without it, every
        # call retraces from Python (~45 ms on a dev GPU, over the 20 ms budget
        # at 50 Hz). Static args bake thrust bounds and alpha_max as constants.
        actor_apply = self._actor.apply

        def _forward(
            actor_params: Any,
            env_obs: dict[str, Array],
            prev_action: Array,
            normalizer: Any,
            thrust_min: float,
            thrust_max: float,
            alpha_max: float,
        ) -> Array:
            actor_obs = obs_encoding.build_actor_obs(env_obs, prev_action, normalizer)
            flat_obs = jnp.concatenate([actor_obs, jnp.zeros_like(actor_obs)], axis=-1)
            mu = actor_apply(actor_params, flat_obs[None, :])
            return raw_to_env_action(
                mu[0], env_obs["quat"], thrust_min, thrust_max, alpha_max=alpha_max
            )

        self._forward = jax.jit(
            _forward, static_argnames=("thrust_min", "thrust_max", "alpha_max")
        )

    def compute_control(
        self, obs: dict[str, npt.NDArray[np.floating]], info: dict | None = None
    ) -> npt.NDArray[np.floating]:
        """Run the actor on the masked obs and return a 4-d env action `[roll, pitch, yaw, thrust]`."""
        del info
        jax_obs = {key: jnp.asarray(value) for key, value in obs.items()}
        env_action = self._forward(
            self.actor_params,
            jax_obs,
            self.prev_action_env_4vec,
            self.actor_normalizer,
            self.thrust_min,
            self.thrust_max,
            self.alpha_max_rad,
        )
        self.prev_action_env_4vec = env_action
        return np.asarray(env_action, dtype=np.float32)

    def step_callback(
        self,
        action: npt.NDArray[np.floating],
        obs: dict[str, npt.NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        # Base default returns True (despite its comment), which breaks sim.py's
        # rollout loop after one step. The RL policy has no stopping criterion.
        del action, obs, reward, terminated, truncated, info
        return False


def _resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    """Resolve a checkpoint directory (relative paths join against the repo root)."""
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    return path


def _config_value(config: Any, *keys: str, default: Any = _MISSING) -> Any:
    """Read a nested config value from either a mapping or a ConfigDict."""
    value: Any = config
    for key in keys:
        if isinstance(value, dict):
            if key not in value:
                if default is _MISSING:
                    raise KeyError(f"Missing config key: {'.'.join(keys)}")
                return default
            value = value[key]
        else:
            if not hasattr(value, key):
                if default is _MISSING:
                    raise KeyError(f"Missing config key: {'.'.join(keys)}")
                return default
            value = getattr(value, key)
    return value
