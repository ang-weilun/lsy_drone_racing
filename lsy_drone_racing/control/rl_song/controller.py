"""Deployment shim for the Song-2023 RL policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import orbax.checkpoint as ocp
from drone_models.core import load_params
from jax import Array

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.control.rl_song import obs as obs_encoding
from lsy_drone_racing.control.rl_song.config import ENV_ACTION_DIM, TANGENT_ALPHA_MAX_RAD
from lsy_drone_racing.control.rl_song.obs import NormalizerState
from lsy_drone_racing.control.rl_song.policy import (
    Actor,
    deterministic_raw_action,
    raw_to_env_action,
    sample_and_log_prob,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
CHECKPOINT_PREFIX: str = "step_"
TOTAL_THRUST_MULTIPLIER: float = 4.0


class RLSongController(Controller):
    """Deterministic RL Song controller for simulated deployment."""

    def __init__(self, obs: dict[str, npt.NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller from an Orbax checkpoint.

        Parameters
        ----------
        obs : dict[str, ndarray]
            Initial unbatched racing-env observation.
        info : dict
            Initial env info dictionary. Not used by this controller.
        config : dict
            Race configuration. Requires ``controller.checkpoint`` and the
            usual ``sim.physics`` / ``sim.drone_model`` entries.
        """
        super().__init__(obs, info, config)
        checkpoint_path = _resolve_checkpoint_path(
            _config_value(config, "controller", "checkpoint")
        )
        checkpoint = _restore_checkpoint(checkpoint_path)
        self.actor_params = checkpoint["actor_params"]
        self.normalizer = _normalizer_from_checkpoint(checkpoint["normalizer"])
        self.alpha_max_rad = _load_alpha_max(checkpoint_path)

        physics = _config_value(config, "sim", "physics")
        drone_model = _config_value(config, "sim", "drone_model")
        drone_params = load_params(physics, drone_model)
        self.thrust_min = float(drone_params["thrust_min"] * TOTAL_THRUST_MULTIPLIER)
        self.thrust_max = float(drone_params["thrust_max"] * TOTAL_THRUST_MULTIPLIER)
        self.prev_action_env_4vec = jnp.zeros((ENV_ACTION_DIM,), dtype=jnp.float32)
        self._deterministic_inference = jax.jit(_deterministic_env_action, static_argnums=(5,))
        # Diagnostic: read RLSONG_STOCHASTIC_EVAL env var to switch to
        # stochastic-sample inference. The trained policy has substantial
        # post-tanh exploration variance on log_std_tangent (sigma ~0.5 on
        # 2/3 axes for the v38c checkpoint); deterministic ``project(mu)``
        # lives in a different region of env-action space than the
        # ``E[project(mu + sigma·eps)]`` distribution training optimised.
        # Setting this var enables the comparison eval per the 2026-05-20
        # train/eval-gap codex diagnosis (see handoff doc).
        import os

        self._stochastic_eval = os.environ.get("RLSONG_STOCHASTIC_EVAL", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self._eval_rng_key = jax.random.PRNGKey(int(os.environ.get("RLSONG_EVAL_SEED", "0")))
        self._stochastic_inference = jax.jit(_stochastic_env_action, static_argnums=(6,))

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

        The base :class:`Controller`'s default returns ``True`` (despite its
        comment), which would break ``sim.py``'s rollout loop after one step.
        The RL policy has no internal stopping criterion — it runs until the
        env terminates or truncates.
        """
        _ = action, obs, reward, terminated, truncated, info
        return False

    def compute_control(
        self, obs: dict[str, npt.NDArray[np.floating]], info: dict | None = None
    ) -> npt.NDArray[np.floating]:
        """Compute the next attitude command.

        Parameters
        ----------
        obs : dict[str, ndarray]
            Current unbatched racing-env observation.
        info : dict, optional
            Additional environment info. Not used.

        Returns
        -------
        action : ndarray, shape (4,)
            Env attitude command ``[roll, pitch, yaw, thrust]``.

        Notes
        -----
        Side effect: sets ``self._last_policy_mean`` to the 7-D
        pre-projection policy mean each call. The attribute is not part
        of the :class:`Controller` base contract.
        """
        _ = info
        jax_obs = _to_jax_obs(obs)
        actor_obs = obs_encoding.build_actor_obs(
            jax_obs, self.prev_action_env_4vec, self.normalizer
        )
        if self._stochastic_eval:
            self._eval_rng_key, sample_key = jax.random.split(self._eval_rng_key)
            env_action, raw_action = self._stochastic_inference(
                self.actor_params,
                actor_obs,
                jax_obs["quat"],
                self.thrust_min,
                self.thrust_max,
                sample_key,
                self.alpha_max_rad,
            )
        else:
            env_action, raw_action = self._deterministic_inference(
                self.actor_params,
                actor_obs,
                jax_obs["quat"],
                self.thrust_min,
                self.thrust_max,
                self.alpha_max_rad,
            )
        self.prev_action_env_4vec = env_action
        self._last_policy_mean = np.asarray(raw_action, dtype=np.float32)
        return np.asarray(env_action, dtype=np.float32)


def _deterministic_env_action(
    actor_params: dict[str, Any],
    actor_obs: Array,
    quat_xyzw: Array,
    thrust_min: float,
    thrust_max: float,
    alpha_max_rad: float,
) -> tuple[Array, Array]:
    """Run deterministic actor inference and raw-to-env projection.

    Parameters
    ----------
    actor_params : dict
        Actor parameters loaded from the Orbax checkpoint.
    actor_obs : Array, shape (ACTOR_OBS_DIM,)
        Normalized actor observation.
    quat_xyzw : Array, shape (4,)
        Current drone orientation as an xyzw quaternion. Read from the
        same env step the actor sees, so the composed
        ``R_target = R_current @ ΔR`` tracks the realized attitude.
    thrust_min, thrust_max : float
        Total-thrust bounds in newtons.
    alpha_max_rad : float
        Per-step rotation budget the policy was trained under. Must
        match the training-time ``α_max`` or the projection rescales
        the tangent vector against a different cap than the policy
        expects.

    Returns
    -------
    env_action : Array, shape (4,)
        Projected attitude command ``[roll, pitch, yaw, thrust]``.
    raw_action : Array, shape (RAW_ACTION_DIM,)
        Policy mean before projection. Surfaced so the controller can
        stash it on ``self._last_policy_mean`` for trace logging.
    """
    raw_action = deterministic_raw_action(actor_params, actor_obs)
    env_action = raw_to_env_action(
        raw_action, quat_xyzw, thrust_min, thrust_max, alpha_max=alpha_max_rad
    )
    return env_action, raw_action


def _stochastic_env_action(
    actor_params: dict[str, Any],
    actor_obs: Array,
    quat_xyzw: Array,
    thrust_min: float,
    thrust_max: float,
    key: Array,
    alpha_max_rad: float,
) -> tuple[Array, Array]:
    """Sample one raw action and project it (diagnostic for train/eval gap).

    Mirrors :func:`_deterministic_env_action` but draws ``raw_action`` from
    the trained Gaussian instead of taking its mean. Used only when
    ``RLSONG_STOCHASTIC_EVAL=1`` is set on the controller process — see the
    constructor docstring for the rationale (matching training-time
    projection distribution to test the deterministic projection-gap
    hypothesis).
    """
    raw_action, _ = sample_and_log_prob(actor_params, actor_obs, key)
    env_action = raw_to_env_action(
        raw_action, quat_xyzw, thrust_min, thrust_max, alpha_max=alpha_max_rad
    )
    return env_action, raw_action


def _load_alpha_max(checkpoint_path: Path) -> float:
    """Read ``tangent_alpha_max_rad`` from ``policy_config.json`` if present.

    The file lives at the *run* directory (the parent of any ``step_*``
    checkpoint) alongside ``reward_config.json``. Missing file → fall
    back to the module-level default so pre-v38 checkpoints continue to
    deploy without manual config edits.
    """
    import json

    run_dir = (
        checkpoint_path.parent
        if checkpoint_path.name.startswith(CHECKPOINT_PREFIX)
        else checkpoint_path
    )
    policy_config = run_dir / "policy_config.json"
    if not policy_config.is_file():
        return float(TANGENT_ALPHA_MAX_RAD)
    payload = json.loads(policy_config.read_text(encoding="utf-8"))
    return float(payload["tangent_alpha_max_rad"])


def _to_jax_obs(obs: dict[str, npt.NDArray[np.floating]]) -> dict[str, Array]:
    """Convert an unbatched env observation to JAX arrays."""
    return {key: jnp.asarray(value) for key, value in obs.items()}


def _config_value(config: Any, *keys: str) -> Any:
    """Read a nested config value from either a mapping or ConfigDict."""
    value = config
    for key in keys:
        if isinstance(value, dict):
            if key not in value:
                raise ValueError(f"Missing config key: {'.'.join(keys)}")
            value = value[key]
        else:
            if not hasattr(value, key):
                raise ValueError(f"Missing config key: {'.'.join(keys)}")
            value = getattr(value, key)
    return value


def _resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    """Resolve either a checkpoint directory or a run directory."""
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if path.is_dir() and path.name.startswith(CHECKPOINT_PREFIX):
        return path
    latest = _latest_checkpoint_path(path)
    if latest is None:
        raise ValueError(f"No Orbax checkpoint directories found under: {path}")
    return latest


def _latest_checkpoint_path(run_dir: Path) -> Path | None:
    """Return the newest ``step_*`` checkpoint directory under ``run_dir``."""
    if not run_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.iterdir():
        if not path.is_dir() or not path.name.startswith(CHECKPOINT_PREFIX):
            continue
        step_str = path.name.removeprefix(CHECKPOINT_PREFIX)
        if step_str.isdecimal():
            candidates.append((int(step_str), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _restore_checkpoint(path: Path) -> dict[str, Any]:
    """Restore an Orbax checkpoint directory.

    Notes
    -----
    Orbax >=0.7 refuses to deserialize ``jax.Array`` leaves without an
    explicit sharding spec, and the checkpoint was saved on a single CUDA
    device, so the implicit topology no longer matches at restore time
    (e.g. when loading on CPU for sim). We read the saved item metadata
    and request every leaf as a plain ``np.ndarray`` via
    ``ArrayRestoreArgs``; the policy code converts to ``jax.Array`` as
    needed via ``jnp.asarray``.
    """
    checkpointer = ocp.PyTreeCheckpointer()
    item_metadata = checkpointer.metadata(path).item_metadata
    restore_args = jax.tree_util.tree_map(
        lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), item_metadata
    )
    return checkpointer.restore(path, args=ocp.args.PyTreeRestore(restore_args=restore_args))


def _normalizer_from_checkpoint(data: dict[str, Array]) -> NormalizerState:
    """Restore a frozen observation normalizer from checkpoint data."""
    return NormalizerState(mean=data["mean"], var=data["var"], count=data["count"])
