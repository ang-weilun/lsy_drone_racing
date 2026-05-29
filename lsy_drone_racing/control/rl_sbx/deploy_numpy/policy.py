"""Numpy actor forward pass and action projection for SBX deploy."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.rl_song.config import (
    ACTOR_OBS_DIM,
    ENV_ACTION_DIM,
    RAW_ACTION_DIM,
)


def _read_net_arch() -> tuple[int, ...]:
    """Read ``NET_ARCH`` from ``rl_sbx.policy`` without importing that module."""
    policy_path = Path(__file__).resolve().parents[1] / "policy.py"
    module = ast.parse(policy_path.read_text(encoding="utf-8"))
    constants: dict[str, Any] = {}
    for node in module.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        name = node.target.id
        constants[name] = _eval_constant_expr(node.value, constants)
        if name == "NET_ARCH":
            return tuple(int(value) for value in constants[name])
    raise ValueError(f"Could not read NET_ARCH from {policy_path}")


def _eval_constant_expr(node: ast.AST, constants: Mapping[str, Any]) -> Any:
    """Evaluate simple constant expressions used by ``rl_sbx.policy``."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_eval_constant_expr(item, constants) for item in node.elts)
    if isinstance(node, ast.Name):
        return constants[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = _eval_constant_expr(node.left, constants)
        right = _eval_constant_expr(node.right, constants)
        return left * right
    return ast.literal_eval(node)


# Raw action layout: one thrust scalar followed by a 3-d tangent vector.
_THRUST_RAW_DIM: int = 1
_TANGENT_RAW_DIM: int = 3

# Hidden-layer widths read from the SBX policy source without importing JAX/Flax.
NET_ARCH: tuple[int, ...] = _read_net_arch()

# Minimum tangent norm in radians for stable norm-preserving squash.
_TANGENT_NORM_EPS_RAD: float = 1e-8

# Negative slope of the hidden leaky-ReLU activation. Matches the default of
# ``flax.linen.leaky_relu`` and the rl_sbx Actor's ``activation_fn``.
_LEAKY_RELU_NEGATIVE_SLOPE: float = 0.01


def actor_mean(
    params: Mapping[str, Any], flat_obs: npt.NDArray[np.floating]
) -> npt.NDArray[np.float32]:
    """Run the deterministic SBX actor mean in numpy.

    Parameters
    ----------
    params : Mapping[str, Any]
        Numpy-converted Flax actor parameter tree.
    flat_obs : ndarray, shape (2 * ACTOR_OBS_DIM,)
        Flat-concat observation. Only the first actor half is consumed.

    Returns:
    -------
    raw_action : ndarray, shape (RAW_ACTION_DIM,)
        Tanh-bounded actor mean in raw action space.

    Notes:
    -----
    Mirrors the redesigned single-coupled-head SBX actor: leaky-ReLU hidden
    layers and a single tanh-bounded ``Dense(RAW_ACTION_DIM)`` head. The
    earlier v131/v132 split thrust/tangent heads were reverted to this
    layout on 2026-05-27; see the rl_sbx ``Actor`` module's ``__call__``.
    """
    actor_params = _params_subtree(params)
    x = np.asarray(flat_obs, dtype=np.float32)[..., :ACTOR_OBS_DIM]
    for layer_idx, _n_units in enumerate(NET_ARCH):
        dense = _dense_params(actor_params, layer_idx)
        x = _leaky_relu(x @ dense["kernel"] + dense["bias"])

    head_dense = _dense_params(actor_params, len(NET_ARCH))
    raw_action = np.tanh(x @ head_dense["kernel"] + head_dense["bias"])
    _validate_last_dim(raw_action, RAW_ACTION_DIM, "raw_action")
    return np.asarray(raw_action, dtype=np.float32)


def _leaky_relu(x: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """Leaky-ReLU with the flax default negative slope."""
    return np.where(x >= 0.0, x, _LEAKY_RELU_NEGATIVE_SLOPE * x).astype(
        np.float32, copy=False
    )


def raw_to_env_action(
    raw_action: npt.NDArray[np.floating],
    quat_xyzw: npt.NDArray[np.floating],
    thrust_min: float,
    thrust_max: float,
    alpha_max: float,
) -> npt.NDArray[np.float32]:
    """Project a raw policy action to an env attitude command.

    Parameters
    ----------
    raw_action : ndarray, shape (RAW_ACTION_DIM,)
        Raw policy action ``[T_raw, tau_x, tau_y, tau_z]``.
    quat_xyzw : ndarray, shape (4,)
        Current body orientation in xyzw quaternion convention.
    thrust_min : float
        Minimum total thrust in newtons.
    thrust_max : float
        Maximum total thrust in newtons.
    alpha_max : float
        Per-step tangent-vector norm budget in radians.

    Returns:
    -------
    env_action : ndarray, shape (ENV_ACTION_DIM,)
        Env command ``[roll, pitch, yaw, thrust]``.
    """
    raw = np.asarray(raw_action, dtype=np.float32)
    quat = np.asarray(quat_xyzw, dtype=np.float32)
    _validate_last_dim(raw, RAW_ACTION_DIM, "raw_action")
    _validate_last_dim(quat, 4, "quat_xyzw")

    thrust_raw = raw[..., :_THRUST_RAW_DIM]
    tangent_raw = raw[..., _THRUST_RAW_DIM:]

    thrust_range = thrust_max - thrust_min
    thrust = thrust_min + thrust_range * 0.5 * (np.tanh(thrust_raw) + 1.0)

    tau_scaled = scale_tangent(tangent_raw, alpha_max)
    delta_rotation = Rotation.from_rotvec(tau_scaled).as_matrix()
    rotation_current = Rotation.from_quat(quat).as_matrix()
    rotation_target = rotation_current @ delta_rotation
    euler_xyz = Rotation.from_matrix(rotation_target).as_euler("xyz")

    env_action = np.concatenate([euler_xyz, thrust], axis=-1)
    _validate_last_dim(env_action, ENV_ACTION_DIM, "env_action")
    return env_action.astype(np.float32, copy=False)


def scale_tangent(
    tangent_raw: npt.NDArray[np.floating], alpha_max: float
) -> npt.NDArray[np.float32]:
    """Squash ``tangent_raw`` so its norm is bounded by ``alpha_max``.

    Parameters
    ----------
    tangent_raw : ndarray, shape (..., 3)
        Unbounded tangent vector from the actor.
    alpha_max : float
        Maximum scaled tangent-vector norm in radians.

    Returns:
    -------
    tau_scaled : ndarray, shape (..., 3)
        Tangent vector with preserved direction and bounded norm.
    """
    tangent = np.asarray(tangent_raw, dtype=np.float32)
    _validate_last_dim(tangent, _TANGENT_RAW_DIM, "tangent_raw")
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    safe_norm = np.maximum(norm, _TANGENT_NORM_EPS_RAD)
    scale = np.tanh(norm) * alpha_max / safe_norm
    return np.asarray(tangent * scale, dtype=np.float32)


def _params_subtree(params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the inner Flax ``params`` subtree when present."""
    subtree = params.get("params", params)
    if not isinstance(subtree, Mapping):
        raise TypeError("actor params must be a mapping or contain a 'params' mapping")
    return subtree


def _dense_params(
    params: Mapping[str, Any], layer_idx: int
) -> Mapping[str, npt.NDArray[np.floating]]:
    """Return a named dense layer's kernel and bias."""
    key = f"Dense_{layer_idx}"
    if key not in params:
        raise KeyError(f"Missing actor parameter layer: {key}")
    layer = params[key]
    if not isinstance(layer, Mapping):
        raise TypeError(f"{key} params must be a mapping")
    for leaf in ("kernel", "bias"):
        if leaf not in layer:
            raise KeyError(f"Missing actor parameter: {key}/{leaf}")
    return layer


def _validate_last_dim(
    array: npt.NDArray[np.floating], expected_dim: int, name: str
) -> None:
    """Raise if an array's trailing dimension violates a static contract."""
    if array.shape[-1] != expected_dim:
        raise ValueError(
            f"{name} trailing dimension must be {expected_dim}; got {array.shape[-1]}"
        )
