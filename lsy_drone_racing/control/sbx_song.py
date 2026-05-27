"""Deployment shim for the SBX-trained Song-style policy.

Loads ONLY the actor parameters + actor normalizer + ``tangent_alpha_max_rad``
from a checkpoint. The :class:`Critic` class is intentionally NOT imported
here, and the loader rejects attempts to read critic artifacts — this is
risk-4 mitigation from ``docs/specs/2026-05-24-sbx-migration-design.md``
(deploy expects unavailable critic obs).

At deploy time the wrapper's flat-concat obs layout collapses to just the
actor half: the controller builds a ``(2 * ACTOR_OBS_DIM,)`` array whose
first half is the masked actor obs (via :func:`build_actor_obs` with the
loaded ``actor_normalizer``) and whose second half is zeros. The actor
module slices ``[..., :ACTOR_OBS_DIM]`` internally, so the second half is
never read.

The controller is single-drone (matches the eval pipeline and the real
deploy path).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from drone_models.core import load_params

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.control.rl_sbx import checkpoint as ckpt
from lsy_drone_racing.control.rl_sbx.policy import LOG_STD_INIT, NET_ARCH, Actor
from lsy_drone_racing.control.rl_song import obs as obs_encoding
from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_DIM, ENV_ACTION_DIM, RAW_ACTION_DIM
from lsy_drone_racing.control.rl_song.policy import raw_to_env_action

if TYPE_CHECKING:
    import numpy.typing as npt
    from jax import Array

# Mirrors ``rl_song.controller.TOTAL_THRUST_MULTIPLIER`` — the per-rotor
# ``thrust_min`` / ``thrust_max`` from ``drone_models`` is the single-rotor
# limit; the env command is total thrust across all 4 rotors.
TOTAL_THRUST_MULTIPLIER: float = 4.0

# Repo root; used to resolve ``controller.checkpoint`` when it is given as
# a path relative to the repo root rather than the eval cwd.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Hardcoded default. Lets ``config/levelN.toml`` carry only ``controller.file``
# (competition rule: only ``controller.file`` and ``env.control_mode`` may
# diff). Override by adding ``controller.checkpoint`` to the config when
# pointing at a different run.
DEFAULT_CHECKPOINT: str = "checkpoints/sbx_redesign_warm50_200M/step_000201326592"

# Step-directory prefix written by :func:`rl_sbx.checkpoint.save_step`.
_CHECKPOINT_PREFIX: str = "step_"

# Sentinel for :func:`_config_value` so callers can pass ``default=None``
# distinctly from "no default supplied, raise".
_MISSING: Any = object()


class RLSBXController(Controller):
    """Deterministic deploy-time controller for the SBX-trained actor.

    Parameters
    ----------
    obs : dict[str, ndarray]
        Initial unbatched racing-env observation (passed by the eval/deploy
        harness).
    info : dict
        Initial env info dictionary. Not used by this controller.
    config : dict
        Race configuration. ``controller.checkpoint`` is optional — when
        absent, falls back to :data:`DEFAULT_CHECKPOINT`. ``sim.physics`` /
        ``sim.drone_model`` are required.
    """

    def __init__(self, obs: dict[str, npt.NDArray[np.floating]], info: dict, config: dict) -> None:
        """Load the actor-only checkpoint and prepare the deploy actor module.

        See the class docstring for parameter semantics.
        """
        super().__init__(obs, info, config)

        checkpoint_path = _resolve_checkpoint_path(
            _config_value(config, "controller", "checkpoint", default=DEFAULT_CHECKPOINT)
        )
        loaded = ckpt.load_actor_only(checkpoint_path)
        # ``actor_state.params`` from training is the full ``{"params": ...}``
        # tree returned by ``Actor.init``; the loader hands back that tree
        # unmodified via ``msgpack_restore``. Pass it straight to ``apply``.
        self.actor_params: dict[str, Any] = loaded["actor_params"]
        self.actor_normalizer = loaded["actor_normalizer"]
        self.alpha_max_rad = float(loaded["tangent_alpha_max_rad"])

        physics = _config_value(config, "sim", "physics")
        drone_model = _config_value(config, "sim", "drone_model")
        drone_params = load_params(physics, drone_model)
        self.thrust_min = float(drone_params["thrust_min"] * TOTAL_THRUST_MULTIPLIER)
        self.thrust_max = float(drone_params["thrust_max"] * TOTAL_THRUST_MULTIPLIER)

        # Previous env action; zeros at episode start. Mirrors
        # ``rl_song.controller`` so the first-step obs layout matches what
        # the actor saw in rollout.
        self.prev_action_env_4vec: Array = jnp.zeros((ENV_ACTION_DIM,), dtype=jnp.float32)

        # Constructor args MUST match the kwargs the training-side builder
        # passed to ``Actor`` (action_dim, net_arch, log_std_init,
        # activation_fn, ortho_init). ``activation_fn`` defaults to
        # ``nn.tanh`` and ``ortho_init`` defaults to ``False`` on the Actor
        # dataclass — both match the SBX PPOPolicy defaults used at
        # training time, so we omit them and let the dataclass defaults
        # carry. ``log_std_init`` is irrelevant at deploy (no sampling)
        # but is part of the param tree shape, so pass the trained value
        # to keep the module signature identical.
        self._actor = Actor(action_dim=RAW_ACTION_DIM, net_arch=NET_ARCH, log_std_init=LOG_STD_INIT)

    def compute_control(
        self, obs: dict[str, npt.NDArray[np.floating]], info: dict | None = None
    ) -> npt.NDArray[np.floating]:
        """Run the actor on the masked obs and return a 4-d env action.

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

        Notes:
        -----
        The flat-concat layout is collapsed to ``(2 * ACTOR_OBS_DIM,)``
        with the critic half zeroed; the actor slices the first half
        internally so the second half is never read. The Gaussian mean
        (``dist.mean()``) is used directly — deterministic deploy, no
        sampling.
        """
        del info
        jax_obs = {key: jnp.asarray(value) for key, value in obs.items()}
        actor_obs = obs_encoding.build_actor_obs(
            jax_obs, self.prev_action_env_4vec, self.actor_normalizer
        )
        # Pad the critic half with zeros — actor slices ``[..., :ACTOR_OBS_DIM]``
        # and never touches the rest. Float32 to match the trained dtype.
        flat_obs = jnp.concatenate(
            [actor_obs, jnp.zeros((ACTOR_OBS_DIM,), dtype=actor_obs.dtype)], axis=-1
        )
        dist = self._actor.apply(self.actor_params, flat_obs[None, :])
        raw_action = dist.mean()[0]
        env_action = raw_to_env_action(
            raw_action,
            jax_obs["quat"],
            self.thrust_min,
            self.thrust_max,
            alpha_max=self.alpha_max_rad,
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
        """Return ``False`` so the env (not the controller) decides termination.

        The base :class:`Controller` default returns ``True`` (despite its
        comment), which would break ``sim.py``'s rollout loop after one
        step. The RL policy has no internal stopping criterion — it runs
        until the env terminates or truncates.
        """
        del action, obs, reward, terminated, truncated, info
        return False


def _resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    """Resolve a checkpoint path to a concrete ``step_*`` directory.

    Parameters
    ----------
    checkpoint : str or Path
        Either a single ``step_*`` directory or a run-level directory
        containing one or more ``step_*`` subdirectories. Relative paths
        are resolved against the repo root.

    Returns:
    -------
    Path
        Absolute path to a concrete ``step_*`` directory.

    Raises:
    ------
    FileNotFoundError
        If ``checkpoint`` does not exist or no ``step_*`` subdir is found
        under a run-level directory.
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
        raise FileNotFoundError(f"No {_CHECKPOINT_PREFIX}* subdirectories under: {path}")
    return latest


def _latest_step_dir(run_dir: Path) -> Path | None:
    """Return the highest-numbered ``step_*`` subdir under ``run_dir``."""
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


def _config_value(config: Any, *keys: str, default: Any = _MISSING) -> Any:
    """Read a nested config value from either a mapping or a ConfigDict.

    Parameters
    ----------
    config : dict or ml_collections.ConfigDict
        Root configuration object.
    *keys : str
        Dotted-path key sequence to traverse.
    default : Any, optional
        Value to return when any intermediate key is missing. If omitted,
        a missing key raises :class:`KeyError`.

    Returns:
    -------
    Any
        The value at ``config[keys[0]][keys[1]]...`` or ``default``.

    Raises:
    ------
    KeyError
        If any intermediate key is missing and no ``default`` was given.
    """
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
