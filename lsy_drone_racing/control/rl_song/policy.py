"""Flax actor/critic networks and raw-action projection utilities.

PPO samples in the raw 4-dimensional action space
``[T_raw, tau_x, tau_y, tau_z]`` where ``tau`` is a local-tangent vector
``ˢτ ∈ ℝ³`` (axis-angle increment per Schuck et al. 2025). The downstream
projection — α_max scaling, exp to ``ΔR``, composition
``R_target = R_current @ ΔR``, and conversion to ``[roll, pitch, yaw,
thrust]`` — is deterministic and deliberately kept out of the
log-probability path. ``R_current`` is read from the env's
unnormalized drone quaternion at the time the action is applied. All
SO(3) primitives go through :class:`jax.scipy.spatial.transform.Rotation`
per CLAUDE.md "default to the ecosystem".

References:
----------
Schuck, J. et al. (2025). A Primer on SO(3) Action Representations
    in Deep RL. arXiv:2510.11103.
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
# Hard floor on the learned log-std parameter. v43 (Codex review): -2.0 ->
# -2.5. σ_min = exp(-2.5) ≈ 0.082; with α_max = 0.16, the floored tangent
# action has ‖τ_scaled‖ ≈ tanh(σ_min·√3)·α_max ≈ 0.023 rad/step (≈1.3°/step
# ≈ 1.1 rad/s) of stochastic exploration. The previous -2.0 floor (~2 rad/s
# residual noise) injected too much attitude dithering after the policy
# committed to a gate-passing line — useful as a cold-start anti-collapse
# guarantee, but precision-flight-hostile late in training. The v43 ent_coef
# anneal (0.02 -> 0.005) and the new KL early-stop already manage early-
# stage exploration, so the std floor's role narrows to "prevent total
# determinism for late-stage gradient flow" — a lower magnitude suffices.
LOG_STD_MIN: float = -2.5
TANGENT_NORM_EPS: float = 1e-8
LOG_TWO_PI: float = 1.8378770664093453
LOG_TWO_PI_E: float = 2.8378770664093453


class Actor(nn.Module):
    """Outputs ``(mu_raw, log_std_raw)`` for the 4-d action distribution.

    The tangent head's output bias is left at zero so the initial Gaussian
    is centered on ``ΔR = I`` (identity rotation, "command the current
    attitude"). Zhou-2019's 6D head needed a non-zero identity bias
    ``[1, 0, 0, 0, 1, 0]`` to land on identity after Gram-Schmidt; the
    tangent head has ``τ = 0 ↔ I`` natively.
    """

    init_log_std: float = DEFAULT_INIT_LOG_STD

    @nn.compact
    def __call__(self, obs: Array) -> tuple[Array, Array]:
        """Run the actor network.

        Parameters
        ----------
        obs : Array, shape (..., ACTOR_OBS_DIM)
            Normalized actor observation.

        Returns:
        -------
        mu_raw : Array, shape (..., RAW_ACTION_DIM)
            Mean of the Gaussian over raw actions ``[T_raw, tau]``.
        log_std_raw : Array, shape (..., RAW_ACTION_DIM)
            Broadcast state-independent log standard deviation.
        """
        x = obs
        for _ in range(N_HIDDEN_LAYERS):
            x = nn.Dense(HIDDEN_SIZE, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(x)
            x = nn.tanh(x)

        thrust_mean = nn.Dense(THRUST_RAW_DIM, kernel_init=nn.initializers.orthogonal(0.01))(x)
        tangent_mean = nn.Dense(TANGENT_RAW_DIM, kernel_init=nn.initializers.orthogonal(0.01))(x)
        mu_raw = jnp.concatenate([thrust_mean, tangent_mean], axis=-1)

        thrust_log_std = self.param(
            "log_std_thrust", nn.initializers.constant(self.init_log_std), (THRUST_RAW_DIM,)
        )
        tangent_log_std = self.param(
            "log_std_tangent", nn.initializers.constant(self.init_log_std), (TANGENT_RAW_DIM,)
        )
        log_std_raw = jnp.concatenate([thrust_log_std, tangent_log_std], axis=-1)
        log_std_raw = jnp.maximum(log_std_raw, LOG_STD_MIN)
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

        Returns:
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

    Returns:
    -------
    raw_action : Array, shape (..., RAW_ACTION_DIM)
        Sampled raw action ``[T_raw, tau]``. This is the tensor PPO stores
        in the rollout buffer.
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
        Action sampled from the raw 4-d Gaussian during rollout.

    Returns:
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

    Returns:
    -------
    raw_action : Array, shape (..., RAW_ACTION_DIM)
        Deterministic raw action equal to the Gaussian mean.
    """
    mu_raw, _ = Actor().apply({"params": actor_params}, obs)
    _validate_last_dim(mu_raw, RAW_ACTION_DIM, "mu_raw")
    return mu_raw


def raw_to_env_action(
    raw_action: Array,
    quat_xyzw: Array,
    thrust_min: float,
    thrust_max: float,
    alpha_max: float = TANGENT_ALPHA_MAX_RAD,
) -> Array:
    """Project a raw 4-vector + current quaternion to the env's attitude command.

    Parameters
    ----------
    raw_action : Array, shape (..., RAW_ACTION_DIM)
        Raw policy action ``[T_raw, tau_x, tau_y, tau_z]``.
    quat_xyzw : Array, shape (..., 4)
        Current drone body orientation as an xyzw quaternion. Read from
        the env state at the same step the action is applied, so the
        composed target ``R_t · ΔR`` tracks the realized attitude.
    thrust_min : float
        Minimum total thrust in newtons.
    thrust_max : float
        Maximum total thrust in newtons.
    alpha_max : float, optional
        Per-step rotation budget (rad) on ``‖τ_scaled‖``. Defaults to
        :data:`TANGENT_ALPHA_MAX_RAD`.

    Returns:
    -------
    env_action : Array, shape (..., ENV_ACTION_DIM)
        Environment command ``[roll, pitch, yaw, thrust]``.

    Notes:
    -----
    All SO(3) primitives — exp from the tangent vector, composition with
    the current orientation, and conversion to extrinsic xyz Euler — go
    through :class:`jax.scipy.spatial.transform.Rotation`. The α_max
    scaling and thrust squash are pure JAX. Everything here is
    deterministic and outside PPO's log-probability computation.
    """
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
    """Squash ``tangent_raw`` so ``‖τ_scaled‖ ≤ alpha_max``.

    Parameters
    ----------
    tangent_raw : Array, shape (..., 3)
        Unbounded network output for the tangent vector.
    alpha_max : float
        Per-step rotation budget in radians.

    Returns:
    -------
    Array, shape (..., 3)
        Tangent vector with norm bounded by ``alpha_max`` and direction
        preserved. Saturates smoothly via ``tanh(‖τ_raw‖)``; per Schuck
        2025 Hypothesis 5 this removes the wrap-around degeneracy where
        ``‖τ‖`` larger than ``α_max`` maps to the same rotation as
        ``α_max · τ̂``.
    """
    norm = jnp.linalg.norm(tangent_raw, axis=-1, keepdims=True)
    safe_norm = jnp.maximum(norm, TANGENT_NORM_EPS)
    scale = jnp.tanh(norm) * alpha_max / safe_norm
    return tangent_raw * scale


def _normal_log_prob(mu: Array, log_std: Array, action: Array) -> Array:
    """Return summed diagonal-Gaussian log probability."""
    variance_scaled = jnp.square((action - mu) / jnp.exp(log_std))
    per_dim_log_prob = -0.5 * (variance_scaled + 2.0 * log_std + LOG_TWO_PI)
    return jnp.sum(per_dim_log_prob, axis=-1)


def _validate_last_dim(array: Array, expected_dim: int, name: str) -> None:
    """Raise if an array's trailing dimension violates a static contract."""
    if array.shape[-1] != expected_dim:
        raise ValueError(f"{name} trailing dimension must be {expected_dim}; got {array.shape[-1]}")


_validate_last_dim(jnp.zeros((ACTOR_OBS_DIM,), dtype=jnp.float32), ACTOR_OBS_DIM, "obs")
