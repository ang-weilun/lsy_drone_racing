"""SBX checkpoint format. Deploy loads actor params + normalizer + policy config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from flax import serialization

from lsy_drone_racing.control.rl_song.obs import NormalizerState

_STEP_DIR_TEMPLATE: str = "step_{global_step:012d}"
_ACTOR_PARAMS_FILE: str = "actor.params.msgpack"
_CRITIC_PARAMS_FILE: str = "critic.params.msgpack"
_ACTOR_NORMALIZER_FILE: str = "actor_normalizer.json"
_CRITIC_NORMALIZER_FILE: str = "critic_normalizer.json"
_POLICY_CONFIG_FILE: str = "policy_config.json"


def save_step(
    run_dir: Path,
    global_step: int,
    actor_params: Any,
    critic_params: Any,
    actor_normalizer: NormalizerState,
    critic_normalizer: NormalizerState,
    tangent_alpha_max_rad: float,
) -> Path:
    """Write a full step checkpoint and return the step directory."""
    step_dir = Path(run_dir) / _STEP_DIR_TEMPLATE.format(global_step=global_step)
    step_dir.mkdir(parents=True, exist_ok=True)

    (step_dir / _ACTOR_PARAMS_FILE).write_bytes(serialization.to_bytes(actor_params))
    (step_dir / _CRITIC_PARAMS_FILE).write_bytes(serialization.to_bytes(critic_params))

    _save_normalizer(step_dir / _ACTOR_NORMALIZER_FILE, actor_normalizer)
    _save_normalizer(step_dir / _CRITIC_NORMALIZER_FILE, critic_normalizer)

    policy_config: dict[str, Any] = {"tangent_alpha_max_rad": float(tangent_alpha_max_rad)}
    (step_dir / _POLICY_CONFIG_FILE).write_text(
        json.dumps(policy_config, indent=2), encoding="utf-8"
    )

    return step_dir


def load_actor_only(step_dir: Path) -> dict[str, Any]:
    """Load actor params, actor normalizer, and policy config. Deploy cannot see the critic."""
    step_dir = Path(step_dir)
    if not step_dir.is_dir():
        raise FileNotFoundError(f"Step directory does not exist: {step_dir}")

    actor_params_path = step_dir / _ACTOR_PARAMS_FILE
    actor_normalizer_path = step_dir / _ACTOR_NORMALIZER_FILE
    policy_config_path = step_dir / _POLICY_CONFIG_FILE
    for required in (actor_params_path, actor_normalizer_path, policy_config_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required deploy file missing: {required}")

    actor_params = serialization.msgpack_restore(actor_params_path.read_bytes())
    actor_normalizer = _load_normalizer(actor_normalizer_path)
    policy_config = json.loads(policy_config_path.read_text(encoding="utf-8"))

    return {
        "actor_params": actor_params,
        "actor_normalizer": actor_normalizer,
        "tangent_alpha_max_rad": float(policy_config["tangent_alpha_max_rad"]),
    }


def load_all(step_dir: Path, actor_template: Any, critic_template: Any) -> dict[str, Any]:
    """Load actor + critic params + both normalizers for training resume."""
    step_dir = Path(step_dir)
    if not step_dir.is_dir():
        raise FileNotFoundError(f"Step directory does not exist: {step_dir}")

    paths = {
        "actor_params": step_dir / _ACTOR_PARAMS_FILE,
        "critic_params": step_dir / _CRITIC_PARAMS_FILE,
        "actor_normalizer": step_dir / _ACTOR_NORMALIZER_FILE,
        "critic_normalizer": step_dir / _CRITIC_NORMALIZER_FILE,
        "policy_config": step_dir / _POLICY_CONFIG_FILE,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing ({name}): {path}")

    actor_params = serialization.from_bytes(actor_template, paths["actor_params"].read_bytes())
    critic_params = serialization.from_bytes(critic_template, paths["critic_params"].read_bytes())
    actor_normalizer = _load_normalizer(paths["actor_normalizer"])
    critic_normalizer = _load_normalizer(paths["critic_normalizer"])
    policy_config = json.loads(paths["policy_config"].read_text(encoding="utf-8"))

    return {
        "actor_params": actor_params,
        "critic_params": critic_params,
        "actor_normalizer": actor_normalizer,
        "critic_normalizer": critic_normalizer,
        "tangent_alpha_max_rad": float(policy_config["tangent_alpha_max_rad"]),
    }


def _save_normalizer(path: Path, state: NormalizerState) -> None:
    payload = {
        "mean": jnp.asarray(state.mean).tolist(),
        "var": jnp.asarray(state.var).tolist(),
        "count": float(jnp.asarray(state.count)),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _load_normalizer(path: Path) -> NormalizerState:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return NormalizerState(
        mean=jnp.asarray(payload["mean"], dtype=jnp.float32),
        var=jnp.asarray(payload["var"], dtype=jnp.float32),
        count=jnp.asarray(payload["count"], dtype=jnp.float32),
    )
