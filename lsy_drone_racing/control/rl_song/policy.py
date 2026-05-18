"""Flax actor/critic networks and raw-action projection utilities.

PPO samples in the raw 7-dimensional action space
``[T_raw, a1_x, a1_y, a1_z, a2_x, a2_y, a2_z]``. The downstream projection to
the racing environment's ``[roll, pitch, yaw, thrust]`` command is deliberately
kept out of the log-probability path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from lsy_drone_racing.control.rl_song.config import (
    ACTOR_OBS_DIM,
    ENV_ACTION_DIM,
    RAW_ACTION_DIM,
    PPOConfig,
)

HIDDEN_SIZE: int = 256
N_HIDDEN_LAYERS: int = 2
THRUST_RAW_DIM: int = 1
ROTATION_RAW_DIM: int = 6
ROTATION_VECTOR_DIM: int = 3
DEFAULT_INIT_LOG_STD: float = PPOConfig().init_log_std
ROTATION_NORM_EPS: float = 1e-8
GIMBAL_LOCK_EPS: float = 1e-6
LOG_TWO_PI: float = 1.8378770664093453
LOG_TWO_PI_E: float = 2.8378770664093453
ROT6D_IDENTITY_BIAS: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


class Actor(nn.Module):
    """Outputs ``(mu_raw, log_std_raw)`` for the 7-d action distribution."""

    init_log_std: float = DEFAULT_INIT_LOG_STD

    @nn.compact
    def __call__(self, obs: Array) -> tuple[Array, Array]:
        """Run the actor network.

        Parameters
        ----------
        obs : Array, shape (..., ACTOR_OBS_DIM)
            Normalized actor observation.

        Returns
        -------
        mu_raw : Array, shape (..., RAW_ACTION_DIM)
            Mean of the Gaussian over raw actions.
        log_std_raw : Array, shape (..., RAW_ACTION_DIM)
            Broadcast state-independent log standard deviation.
        """
        x = obs
        for _ in range(N_HIDDEN_LAYERS):
            x = nn.Dense(HIDDEN_SIZE, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(x)
            x = nn.tanh(x)

        thrust_mean = nn.Dense(THRUST_RAW_DIM, kernel_init=nn.initializers.orthogonal(0.01))(x)
        rotation_mean = nn.Dense(
            ROTATION_RAW_DIM,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.constant(jnp.asarray(ROT6D_IDENTITY_BIAS)),
        )(x)
        mu_raw = jnp.concatenate([thrust_mean, rotation_mean], axis=-1)

        thrust_log_std = self.param(
            "log_std_thrust", nn.initializers.constant(self.init_log_std), (THRUST_RAW_DIM,)
        )
        rotation_log_std = self.param(
            "log_std_rotation", nn.initializers.constant(self.init_log_std), (ROTATION_RAW_DIM,)
        )
        log_std_raw = jnp.concatenate([thrust_log_std, rotation_log_std], axis=-1)
        return mu_raw, jnp.broadcast_to(log_std_raw, mu_raw.shape)


class Critic(nn.Module):
    """Outputs a scalar value estimate from the critic observation."""

    @nn.compact
    def __call__(self, obs: Array) -> Array:
        """Run the critic network.

        Parameters
        ----------
        obs : Array, shape (..., ACTOR_OBS_DIM)
            Normalized critic observation.

        Returns
        -------
        value : Array, shape (...)
            Scalar value estimate for each leading sample.
        """
        x = obs
        for _ in range(N_HIDDEN_LAYERS):
            x = nn.Dense(HIDDEN_SIZE, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(x)
            x = nn.tanh(x)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
        return jnp.squeeze(value, axis=-1)


def sample_and_log_prob(actor_params: dict, obs: Array, key: jax.Array) -> tuple[Array, Array]:
    """Sample a raw action and return its raw-space log probability.

    Parameters
    ----------
    actor_params : dict
        Parameters for :class:`Actor`.
    obs : Array, shape (..., ACTOR_OBS_DIM)
        Normalized actor observation.
    key : jax.Array
        PRNG key for Gaussian sampling.

    Returns
    -------
    raw_action : Array, shape (..., RAW_ACTION_DIM)
        Sampled raw action. This is the tensor PPO stores in the rollout
        buffer.
    log_prob : Array, shape (...)
        ``Normal(mu_raw, sigma_raw).log_prob(raw_action).sum(-1)``.
    """
    mu_raw, log_std_raw = Actor().apply({"params": actor_params}, obs)
    raw_action = mu_raw + jnp.exp(log_std_raw) * jax.random.normal(key, shape=mu_raw.shape)
    _validate_last_dim(raw_action, RAW_ACTION_DIM, "raw_action")
    log_prob = _normal_log_prob(mu_raw, log_std_raw, raw_action)
    return raw_action, log_prob


def log_prob_of(actor_params: dict, obs: Array, raw_action: Array) -> tuple[Array, Array]:
    """Return log probability and entropy for a provided raw action.

    Parameters
    ----------
    actor_params : dict
        Parameters for :class:`Actor`.
    obs : Array, shape (..., ACTOR_OBS_DIM)
        Normalized actor observation.
    raw_action : Array, shape (..., RAW_ACTION_DIM)
        Action sampled from the raw 7-d Gaussian during rollout.

    Returns
    -------
    log_prob : Array, shape (...)
        Raw-space log probability.
    entropy : Array, shape (...)
        Entropy of the raw-space Gaussian.
    """
    _validate_last_dim(raw_action, RAW_ACTION_DIM, "raw_action")
    mu_raw, log_std_raw = Actor().apply({"params": actor_params}, obs)
    log_prob = _normal_log_prob(mu_raw, log_std_raw, raw_action)
    entropy = jnp.sum(0.5 * LOG_TWO_PI_E + log_std_raw, axis=-1)
    return log_prob, entropy


def deterministic_raw_action(actor_params: dict, obs: Array) -> Array:
    """Return ``mu_raw`` for deployment and deterministic evaluation.

    Parameters
    ----------
    actor_params : dict
        Parameters for :class:`Actor`.
    obs : Array, shape (..., ACTOR_OBS_DIM)
        Normalized actor observation.

    Returns
    -------
    raw_action : Array, shape (..., RAW_ACTION_DIM)
        Deterministic raw action equal to the Gaussian mean.
    """
    mu_raw, _ = Actor().apply({"params": actor_params}, obs)
    _validate_last_dim(mu_raw, RAW_ACTION_DIM, "mu_raw")
    return mu_raw


def raw_to_env_action(raw_action: Array, thrust_min: float, thrust_max: float) -> Array:
    """Project a raw 7-vector to the env's attitude command.

    Parameters
    ----------
    raw_action : Array, shape (..., RAW_ACTION_DIM)
        Raw policy action ``[T_raw, a1, a2]``.
    thrust_min : float
        Minimum total thrust in newtons.
    thrust_max : float
        Maximum total thrust in newtons.

    Returns
    -------
    env_action : Array, shape (..., ENV_ACTION_DIM)
        Environment command ``[roll, pitch, yaw, thrust]``.

    Notes
    -----
    The Gram-Schmidt and Euler conversion are deterministic transforms outside
    PPO's log-probability computation.
    """
    _validate_last_dim(raw_action, RAW_ACTION_DIM, "raw_action")
    thrust_raw = raw_action[..., :THRUST_RAW_DIM]
    rotation_raw = raw_action[..., THRUST_RAW_DIM:]
    a1 = rotation_raw[..., :ROTATION_VECTOR_DIM]
    a2 = rotation_raw[..., ROTATION_VECTOR_DIM:]

    thrust_range = thrust_max - thrust_min
    thrust = thrust_min + thrust_range * 0.5 * (jnp.tanh(thrust_raw) + 1.0)

    r1 = a1 / (jnp.linalg.norm(a1, axis=-1, keepdims=True) + ROTATION_NORM_EPS)
    r2_unscaled = a2 - jnp.sum(r1 * a2, axis=-1, keepdims=True) * r1
    r2 = r2_unscaled / (jnp.linalg.norm(r2_unscaled, axis=-1, keepdims=True) + ROTATION_NORM_EPS)
    r3 = jnp.cross(r1, r2, axis=-1)
    rotation_matrix = jnp.stack([r1, r2, r3], axis=-1)
    euler_xyz = _matrix_to_euler_xyz(rotation_matrix)
    env_action = jnp.concatenate([euler_xyz, thrust], axis=-1)
    _validate_last_dim(env_action, ENV_ACTION_DIM, "env_action")
    return env_action


def _normal_log_prob(mu: Array, log_std: Array, action: Array) -> Array:
    """Return summed diagonal-Gaussian log probability."""
    variance_scaled = jnp.square((action - mu) / jnp.exp(log_std))
    per_dim_log_prob = -0.5 * (variance_scaled + 2.0 * log_std + LOG_TWO_PI)
    return jnp.sum(per_dim_log_prob, axis=-1)


def _matrix_to_euler_xyz(rotation_matrix: Array) -> Array:
    """Convert a 3x3 rotation matrix to extrinsic xyz Euler angles in pure JAX.

    Parameters
    ----------
    rotation_matrix : Array, shape (..., 3, 3)
        Orthonormal matrix produced by Gram-Schmidt of the 6D rotation head.

    Returns
    -------
    Array, shape (..., 3)
        Extrinsic xyz Euler angles ``[roll, pitch, yaw]``. Matches the
        convention of
        ``scipy.spatial.transform.Rotation.from_matrix(R).as_euler('xyz')``
        and therefore the env's
        ``Rotation.from_euler('xyz', rpy).as_quat()`` round-trip.

    Notes
    -----
    Hand-rolled rather than calling
    ``jax.scipy.spatial.transform.Rotation.from_matrix`` then ``.as_euler``
    because the scipy wrapper goes through ``np.vectorize`` and adds ~70 s of
    overhead per training run in the PPO rollout's hot path. Per CLAUDE.md
    "Reimplement only when the library is incompatible with our constraints
    (real-time budget)" — cProfile data justifies the exemption.

    The gimbal-lock branch follows scipy's convention: when
    ``|cos(pitch)| < GIMBAL_LOCK_EPS`` we set ``roll = 0`` and absorb the
    residual rotation into ``yaw``.
    """
    sin_beta = -rotation_matrix[..., 2, 0]
    cos_beta = jnp.sqrt(
        jnp.square(rotation_matrix[..., 0, 0]) + jnp.square(rotation_matrix[..., 1, 0])
    )
    beta = jnp.arctan2(sin_beta, cos_beta)
    alpha = jnp.arctan2(rotation_matrix[..., 2, 1], rotation_matrix[..., 2, 2])
    gamma = jnp.arctan2(rotation_matrix[..., 1, 0], rotation_matrix[..., 0, 0])
    gimbal_lock = cos_beta < GIMBAL_LOCK_EPS
    alpha = jnp.where(gimbal_lock, jnp.zeros_like(alpha), alpha)
    gamma_locked = jnp.arctan2(-rotation_matrix[..., 0, 1], rotation_matrix[..., 1, 1])
    gamma = jnp.where(gimbal_lock, gamma_locked, gamma)
    return jnp.stack([alpha, beta, gamma], axis=-1)


def _validate_last_dim(array: Array, expected_dim: int, name: str) -> None:
    """Raise if an array's trailing dimension violates a static contract."""
    if array.shape[-1] != expected_dim:
        raise ValueError(f"{name} trailing dimension must be {expected_dim}; got {array.shape[-1]}")


_validate_last_dim(jnp.zeros((ACTOR_OBS_DIM,), dtype=jnp.float32), ACTOR_OBS_DIM, "obs")
