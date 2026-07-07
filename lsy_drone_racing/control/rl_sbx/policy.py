"""Asymmetric actor/critic for SBX PPO over a flat-concat observation.

The flat-concat layout `[actor_half | critic_half]` (shape `2 * ACTOR_OBS_DIM`)
is forced by SBX's single-tensor rollout buffer; each module slices its own
half. Inference-only branch: Actor returns `mu` directly (no tfp wrap), training
glue intentionally absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import flax.linen as nn
import jax.numpy as jnp

from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_DIM

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

HIDDEN_SIZE: int = 256
N_HIDDEN_LAYERS: int = 2
NET_ARCH: tuple[int, ...] = (HIDDEN_SIZE,) * N_HIDDEN_LAYERS
LOG_STD_INIT: float = -0.5


class Actor(nn.Module):
    """Flax actor; slices the actor half of the flat-concat obs."""

    action_dim: int
    net_arch: Sequence[int]
    log_std_init: float = LOG_STD_INIT
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    ortho_init: bool = False
    features_extractor: type[nn.Module] | None = None
    features_dim: int = 0

    def get_std(self) -> jnp.ndarray:
        """Placeholder std for gSDE-aware code paths; gSDE unused in this project."""
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Return the deterministic action mean `mu`."""
        x = obs[..., :ACTOR_OBS_DIM]
        hidden_kernel_init: Any
        if self.ortho_init:
            hidden_kernel_init = nn.initializers.orthogonal(jnp.sqrt(2.0))
        else:
            hidden_kernel_init = nn.initializers.lecun_normal()
        for n_units in self.net_arch:
            x = nn.Dense(n_units, kernel_init=hidden_kernel_init)(x)
            x = self.activation_fn(x)

        if self.ortho_init:
            head_kernel_init = nn.initializers.orthogonal(scale=0.01)
        else:
            head_kernel_init = nn.initializers.lecun_normal()
        mu = nn.tanh(
            nn.Dense(
                self.action_dim, kernel_init=head_kernel_init, bias_init=nn.initializers.zeros
            )(x)
        )

        # log_std declared so checkpoints from the training-time Actor (with a
        # state-independent Gaussian std) load cleanly; unused on the deploy path.
        _ = self.param("log_std", nn.initializers.constant(self.log_std_init), (self.action_dim,))
        return mu


class Critic(nn.Module):
    """Flax critic; slices the critic half of the flat-concat obs. Not used at deploy."""

    net_arch: Sequence[int]
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    features_extractor: type[nn.Module] | None = None
    features_dim: int = 0

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Return the scalar value estimate."""
        x = obs[..., ACTOR_OBS_DIM:]
        for n_units in self.net_arch:
            x = nn.Dense(n_units, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(x)
            x = self.activation_fn(x)
        return nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
