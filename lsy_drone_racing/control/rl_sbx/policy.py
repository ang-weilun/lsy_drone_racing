"""Asymmetric actor/critic policy for SBX PPO over a flat-concat observation.

SBX's stock PPO training loop (``sbx/ppo/ppo.py``) is rigidly single-tensor: it
passes one ``observations`` array to both ``actor_state.apply_fn`` and
``vf_state.apply_fn``. Dict observations are explicitly unsupported — see
``sbx/ppo/ppo.py:297`` where the rollout buffer is read as ``.observations.numpy()``.

To run an asymmetric actor/critic under that constraint the training-time env
emits a single flat-concat array of shape ``(2 * ACTOR_OBS_DIM,)``:

* ``obs[..., :ACTOR_OBS_DIM]`` is the masked actor obs (actor normalizer)
* ``obs[..., ACTOR_OBS_DIM:]`` is the privileged critic obs (critic normalizer)

The :class:`Actor` and :class:`Critic` flax modules each slice their own half
as the first operation in ``__call__``. SBX's loss code calls
``dist.log_prob(actions)`` / ``dist.entropy()`` (``sbx/ppo/ppo.py:225-235``)
unmodified — our Actor returns a ``tfd.MultivariateNormalDiag``, satisfying
both contracts.

Layer widths and activation match
:mod:`lsy_drone_racing.control.rl_song.policy` (two 256-unit ``tanh`` hidden
layers) so a hand-crafted warm-start of v83-line weights into this policy
remains theoretically possible. The head, however, is a single
``Dense(RAW_ACTION_DIM)`` with one state-independent ``log_std`` vector — i.e.
the SBX convention rather than the rl_song split-head/floored-log-std
convention. Warm-start would therefore need a separate one-time conversion
step; this file does not implement that.

Deploy-only subset
------------------
This is the inference-only branch — the training-side glue
(``AsymmetricActorCriticPolicy(PPOPolicy)``, ``optax``-built ``TrainState``s,
SBX's ``PPOPolicy.build`` machinery) is intentionally absent so the deploy
env does not need ``sbx``, ``optax``, or ``flax.training`` installed. Only
:class:`Actor` (loaded by :func:`rl_sbx.checkpoint.load_actor_only` and
called by :class:`lsy_drone_racing.control.sbx_song.RLSBXController`) and
:class:`Critic` (kept for documentation symmetry; not loaded at deploy)
remain.

References:
----------
Schuck, J. et al. (2025). A Primer on SO(3) Action Representations in Deep
    RL. arXiv:2510.11103.
Stable Baselines Jax (SBX), https://github.com/araffin/sbx, ``sbx/ppo``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import flax.linen as nn
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_DIM

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

tfd = tfp.distributions

# Hidden-layer widths for actor and critic MLPs.
# v132 (2026-05-27): reverted 512 -> 256 to match rl_song.policy.HIDDEN_SIZE.
# Diagnosis from the rl_song-20M-vs-v131-300M trace diff: v131's deterministic
# mean |tau|/alpha_max = 0.47 (saturating) vs rl_song's 0.08 (committed)
# despite same env, obs encoder, and physics. Codex review localized the
# pathology to the larger 512-wide trunk + single coupled 4D head; the
# additional capacity drives PPO's actor mean against the tanh boundary
# under SBX's update geometry. Pairs with the split thrust/tangent head
# below.
HIDDEN_SIZE: int = 256
N_HIDDEN_LAYERS: int = 2
NET_ARCH: tuple[int, ...] = (HIDDEN_SIZE,) * N_HIDDEN_LAYERS

# Initial log standard deviation for the raw 4-vec Gaussian. Matches
# ``PPOConfig.init_log_std`` (``-0.5``) from the rl_song line so cold-train
# exploration noise on the SBX-trained policy starts at the same magnitude
# the v33-onward runs successfully bootstrapped from.
LOG_STD_INIT: float = -0.5

# Floor on ``log_std`` per ``rl_song.policy.LOG_STD_MIN``. Prevents the
# learnable log-std parameter from collapsing the exploration noise to zero
# in late training (sigma ≈ 0.082 at this floor), which would freeze the
# KL signal and stop the policy from refining.
LOG_STD_MIN: float = -2.5


class Actor(nn.Module):
    """Flax actor that slices the actor half of the flat-concat obs.

    Field signature matches :class:`sbx.ppo.policies.Actor` so that
    :meth:`PPOPolicy.build` can construct us via the same keyword arguments
    it constructs the stock SBX actor with. Discrete-action machinery is
    omitted because the env's action space is :class:`gym.spaces.Box`.

    Parameters
    ----------
    action_dim : int
        Raw 4-vec action dimension ``[T_raw, tau_x, tau_y, tau_z]``.
    net_arch : Sequence[int]
        Hidden-layer widths for the MLP trunk.
    log_std_init : float, optional
        Initial value of the state-independent log standard deviation
        parameter.
    activation_fn : Callable, optional
        Activation between MLP layers; defaults to ``nn.tanh`` to match
        SBX's PPOPolicy default.
    ortho_init : bool, optional
        If True, the final action layer uses orthogonal init with
        scale ``0.01`` (matches rl_song's small-init action head).
    features_extractor, features_dim : optional
        Accepted for API parity with SBX's actor; unused (we have no image
        observations, and the flat-concat obs is already flat).
    """

    action_dim: int
    net_arch: Sequence[int]
    log_std_init: float = LOG_STD_INIT
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    ortho_init: bool = False
    features_extractor: type[nn.Module] | None = None
    features_dim: int = 0

    def get_std(self) -> jnp.ndarray:
        """Return a placeholder std for gSDE-aware code paths.

        SBX's PPOPolicy.build calls ``self.actor.reset_noise = self.reset_noise``
        and may interrogate ``get_std`` even when ``use_sde=False``. We never
        use gSDE in this project, but the attribute must exist.
        """
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> tfd.Distribution:
        """Run the actor.

        Parameters
        ----------
        obs : ndarray, shape (..., 2 * ACTOR_OBS_DIM)
            Flat-concat observation. Only the first ``ACTOR_OBS_DIM``
            features (the masked, actor-normalized half) are consumed.

        Returns:
        -------
        tfd.MultivariateNormalDiag
            Diagonal-Gaussian action distribution with mean ``mu`` and
            ``scale_diag = exp(log_std)``.
        """
        x = obs[..., :ACTOR_OBS_DIM]
        # 2026-05-25: when ``ortho_init=True``, match rl_song's hidden-layer
        # orthogonal(sqrt(2)) init alongside the output head's orthogonal(0.01).
        # rl_song.policy.Actor uses orthogonal init for ALL Dense layers; just
        # fixing the output head while leaving hidden layers at Flax default
        # (lecun_normal) is the partial-parity case that left v113f's
        # gradient flow distinct from v77's.
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

        log_std_raw = self.param(
            "log_std", nn.initializers.constant(self.log_std_init), (self.action_dim,)
        )
        # Floor ``log_std`` at ``LOG_STD_MIN = -2.5`` (sigma ≈ 0.082) per
        # ``rl_song.policy.LOG_STD_MIN`` — prevents PPO from collapsing the
        # exploration noise to zero in late training.
        log_std = jnp.maximum(log_std_raw, LOG_STD_MIN)
        log_std = jnp.broadcast_to(log_std, mu.shape)
        return tfd.MultivariateNormalDiag(loc=mu, scale_diag=jnp.exp(log_std))


class Critic(nn.Module):
    """Flax critic that slices the critic half of the flat-concat obs.

    Field signature matches :class:`sbx.ppo.policies.Critic` so SBX's
    :meth:`PPOPolicy.build` can construct us with its standard kwargs.

    Kept in the deploy bundle for documentation symmetry with the asymmetric
    training-time structure; this class is not instantiated by the deploy
    controller (see :func:`rl_sbx.checkpoint.load_actor_only`).

    Parameters
    ----------
    net_arch : Sequence[int]
        Hidden-layer widths for the MLP trunk.
    activation_fn : Callable, optional
        Activation between MLP layers; defaults to ``nn.tanh``.
    features_extractor, features_dim : optional
        Accepted for API parity; unused (flat-concat obs is already flat).
    """

    net_arch: Sequence[int]
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    features_extractor: type[nn.Module] | None = None
    features_dim: int = 0

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Run the critic.

        Parameters
        ----------
        obs : ndarray, shape (..., 2 * ACTOR_OBS_DIM)
            Flat-concat observation. Only features
            ``[ACTOR_OBS_DIM:]`` (the privileged, critic-normalized half)
            are consumed.

        Returns:
        -------
        ndarray, shape (..., 1)
            Scalar value estimate. The trailing singleton dim matches the
            stock SBX :class:`sbx.ppo.policies.Critic`; SBX's training loop
            calls ``.flatten()`` on the result.
        """
        # v127 (2026-05-26): match rl_song.policy.Critic's orthogonal init.
        # The default ``nn.Dense`` uses Flax ``lecun_normal``; on a 512→1
        # output that gives ``V(s_0) ≈ 0`` at init, so advantages ≈ raw
        # returns for the first ~100M steps until the critic catches up.
        # Biases early PPO updates toward whatever pays immediate reward
        # (hover, low body rate) — exactly the v113h-v125 failure attractor.
        # rl_song uses ``orthogonal(sqrt(2))`` for hidden + ``orthogonal(1.0)``
        # for the value head (`rl_song/policy.py:142-144`).
        x = obs[..., ACTOR_OBS_DIM:]
        for n_units in self.net_arch:
            x = nn.Dense(n_units, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(x)
            x = self.activation_fn(x)
        return nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(x)
