"""Flax actor/critic networks and raw-action projection utilities.

Raw action is `[T_raw, tau]` with `tau` a local-tangent vector (Schuck et al.
2025). The projection (alpha_max scaling, exp, compose with current attitude,
Euler) is deterministic and kept outside the log-probability path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array
from jax.scipy.spatial.transform import Rotation

from lsy_drone_racing.control.rl_song.config import (
    ACTOR_OBS_DIM,
    ENV_ACTION_DIM,
    RAW_ACTION_DIM,
    TANGENT_ALPHA_MAX_RAD,
    PPOConfig,
)

HIDDEN_SIZE: int = 256
N_HIDDEN_LAYERS: int = 2
THRUST_RAW_DIM: int = 1
TANGENT_RAW_DIM: int = 3
DEFAULT_INIT_LOG_STD: float = PPOConfig().init_log_std
LOG_STD_MIN: float = -2.5
TANGENT_NORM_EPS: float = 1e-8
LOG_TWO_PI: float = 1.8378770664093453
LOG_TWO_PI_E: float = 2.8378770664093453


class Actor(nn.Module):
    """Outputs `(mu_raw, log_std_raw)` for the 4-d action distribution."""

    init_log_std: float = DEFAULT_INIT_LOG_STD

    @nn.compact
    def __call__(self, obs: Array) -> tuple[Array, Array]:
        """Return `(mu_raw, log_std_raw)`."""
        x = obs
        for _ in range(N_HIDDEN_LAYERS):
            x = nn.Dense(HIDDEN_SIZE, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(x)
            x = nn.leaky_relu(x)

        mu_raw = nn.tanh(nn.Dense(RAW_ACTION_DIM, kernel_init=nn.initializers.orthogonal(0.01))(x))

        log_std_raw = self.param(
            "log_std", nn.initializers.constant(self.init_log_std), (RAW_ACTION_DIM,)
        )
        log_std_raw = jnp.maximum(log_std_raw, LOG_STD_MIN)
        return mu_raw, jnp.broadcast_to(log_std_raw, mu_raw.shape)


class Critic(nn.Module):
    """Scalar value estimate."""

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        """Return the scalar value estimate."""
        x = obs
        for _ in range(N_HIDDEN_LAYERS):
            x = nn.Dense(HIDDEN_SIZE, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(x)
            x = nn.leaky_relu(x)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return jnp.squeeze(value, axis=-1)


def sample_and_log_prob(actor_params: dict, obs: Array, key: jax.Array) -> tuple[Array, Array]:
    """Sample a raw action and return its raw-space log probability."""
    mu_raw, log_std_raw = Actor().apply({"params": actor_params}, obs)
    raw_action = mu_raw + jnp.exp(log_std_raw) * jax.random.normal(key, shape=mu_raw.shape)
    _validate_last_dim(raw_action, RAW_ACTION_DIM, "raw_action")
    log_prob = _normal_log_prob(mu_raw, log_std_raw, raw_action)
    return raw_action, log_prob


def log_prob_of(actor_params: dict, obs: Array, raw_action: Array) -> tuple[Array, Array]:
    """Log probability and entropy of a previously sampled raw action."""
    _validate_last_dim(raw_action, RAW_ACTION_DIM, "raw_action")
    mu_raw, log_std_raw = Actor().apply({"params": actor_params}, obs)
    log_prob = _normal_log_prob(mu_raw, log_std_raw, raw_action)
    entropy = jnp.sum(0.5 * LOG_TWO_PI_E + log_std_raw, axis=-1)
    return log_prob, entropy


def deterministic_raw_action(actor_params: dict, obs: Array) -> Array:
    """Return `mu_raw` for deployment / deterministic evaluation."""
    mu_raw, _ = Actor().apply({"params": actor_params}, obs)
    _validate_last_dim(mu_raw, RAW_ACTION_DIM, "mu_raw")
    return mu_raw


def raw_to_physical_action(
    raw_action: Array,
    thrust_min: float,
    thrust_max: float,
    alpha_max: float = TANGENT_ALPHA_MAX_RAD,
) -> Array:
    """Raw `[T_raw, tau]` -> physical `[tau_scaled_rad, thrust_N]` for smoothness penalties."""
    _validate_last_dim(raw_action, RAW_ACTION_DIM, "raw_action")
    thrust_raw = raw_action[..., :THRUST_RAW_DIM]
    tangent_raw = raw_action[..., THRUST_RAW_DIM:]
    thrust_range = thrust_max - thrust_min
    thrust = thrust_min + thrust_range * 0.5 * (jnp.tanh(thrust_raw) + 1.0)
    tau_scaled = scale_tangent(tangent_raw, alpha_max)
    physical_action = jnp.concatenate([tau_scaled, thrust], axis=-1)
    _validate_last_dim(physical_action, RAW_ACTION_DIM, "physical_action")
    return physical_action


def raw_to_env_action(
    raw_action: Array,
    quat_xyzw: Array,
    thrust_min: float,
    thrust_max: float,
    alpha_max: float = TANGENT_ALPHA_MAX_RAD,
) -> Array:
    """Raw 4-vec + current quaternion -> env `[roll, pitch, yaw, thrust]`."""
    _validate_last_dim(raw_action, RAW_ACTION_DIM, "raw_action")
    _validate_last_dim(quat_xyzw, 4, "quat_xyzw")

    thrust_raw = raw_action[..., :THRUST_RAW_DIM]
    tangent_raw = raw_action[..., THRUST_RAW_DIM:]

    thrust_range = thrust_max - thrust_min
    thrust = thrust_min + thrust_range * 0.5 * (jnp.tanh(thrust_raw) + 1.0)

    tau_scaled = scale_tangent(tangent_raw, alpha_max)
    delta_rotation = Rotation.from_rotvec(tau_scaled).as_matrix()
    rotation_current = Rotation.from_quat(quat_xyzw).as_matrix()
    rotation_target = jnp.einsum("...ij,...jk->...ik", rotation_current, delta_rotation)
    euler_xyz = Rotation.from_matrix(rotation_target).as_euler("xyz")

    env_action = jnp.concatenate([euler_xyz, thrust], axis=-1)
    _validate_last_dim(env_action, ENV_ACTION_DIM, "env_action")
    return env_action


def scale_tangent(tangent_raw: Array, alpha_max: float) -> Array:
    """Squash `tangent_raw` so `||tau_scaled|| <= alpha_max`, direction preserved."""
    norm = jnp.linalg.norm(tangent_raw, axis=-1, keepdims=True)
    safe_norm = jnp.maximum(norm, TANGENT_NORM_EPS)
    scale = jnp.tanh(norm) * alpha_max / safe_norm
    return tangent_raw * scale


def _normal_log_prob(mu: Array, log_std: Array, action: Array) -> Array:
    variance_scaled = jnp.square((action - mu) / jnp.exp(log_std))
    per_dim_log_prob = -0.5 * (variance_scaled + 2.0 * log_std + LOG_TWO_PI)
    return jnp.sum(per_dim_log_prob, axis=-1)


def _validate_last_dim(array: Array, expected_dim: int, name: str) -> None:
    if array.shape[-1] != expected_dim:
        raise ValueError(f"{name} trailing dimension must be {expected_dim}; got {array.shape[-1]}")


_validate_last_dim(jnp.zeros((ACTOR_OBS_DIM,), dtype=jnp.float32), ACTOR_OBS_DIM, "obs")
