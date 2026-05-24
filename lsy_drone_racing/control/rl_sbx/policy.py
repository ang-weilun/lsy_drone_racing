"""Asymmetric actor/critic policy for SBX PPO over a flat-concat observation.

SBX's stock PPO training loop (``sbx/ppo/ppo.py``) is rigidly single-tensor: it
passes one ``observations`` array to both ``actor_state.apply_fn`` and
``vf_state.apply_fn``. Dict observations are explicitly unsupported — see
``sbx/ppo/ppo.py:297`` where the rollout buffer is read as ``.observations.numpy()``.

To run an asymmetric actor/critic under that constraint we make the
:class:`lsy_drone_racing.control.rl_sbx.env_gym.RLSBXVecEnv` emit a single
flat-concat array of shape ``(2 * ACTOR_OBS_DIM,)``:

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

References:
----------
Schuck, J. et al. (2025). A Primer on SO(3) Action Representations in Deep
    RL. arXiv:2510.11103.
Stable Baselines Jax (SBX), https://github.com/araffin/sbx, ``sbx/ppo``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
from flax.training.train_state import TrainState
from sbx.ppo.policies import PPOPolicy

from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_DIM

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from stable_baselines3.common.type_aliases import Schedule

tfd = tfp.distributions

# Hidden-layer widths for actor and critic MLPs. Match
# ``lsy_drone_racing.control.rl_song.policy.HIDDEN_SIZE`` / ``N_HIDDEN_LAYERS``
# so the layer geometry is identical to the v83-line policy and a manual
# warm-start (Dense kernels copied one-for-one) is possible in principle.
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

# Total flat-concat obs dimension; the env wrapper packs
# ``[actor (ACTOR_OBS_DIM) | critic (ACTOR_OBS_DIM)]``.
FLAT_CONCAT_OBS_DIM: int = 2 * ACTOR_OBS_DIM


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
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
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
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = self.activation_fn(x)

        # ``tanh`` on the mean per Song 2023 §Network and ``rl_song.policy.Actor``.
        # Bounds the policy mean to (-1, 1) so PPO's gradient w.r.t. ``mu``
        # stays informative — without tanh the v43 saturation diagnostics
        # showed ``raw_norm_mean=1.87`` and 49% sample saturation, with the
        # mean sitting outside the downstream clip boundary. The Gaussian
        # *sample* still goes through ℝ⁴ (so log-prob is unchanged); the
        # downstream ``raw_to_env_action`` squashes the sample into the env
        # action range.
        if self.ortho_init:
            mu_pre = nn.Dense(
                self.action_dim,
                kernel_init=nn.initializers.orthogonal(scale=0.01),
                bias_init=nn.initializers.zeros,
            )(x)
        else:
            mu_pre = nn.Dense(self.action_dim)(x)
        mu = nn.tanh(mu_pre)

        log_std_raw = self.param(
            "log_std", nn.initializers.constant(self.log_std_init), (self.action_dim,)
        )
        # Floor ``log_std`` at ``LOG_STD_MIN = -2.5`` (sigma ≈ 0.082) per
        # ``rl_song.policy.LOG_STD_MIN`` — prevents PPO from collapsing the
        # exploration noise to zero in late training, which would freeze the
        # KL signal and stop the policy from refining.
        log_std = jnp.maximum(log_std_raw, LOG_STD_MIN)
        log_std = jnp.broadcast_to(log_std, mu.shape)
        return tfd.MultivariateNormalDiag(loc=mu, scale_diag=jnp.exp(log_std))


class Critic(nn.Module):
    """Flax critic that slices the critic half of the flat-concat obs.

    Field signature matches :class:`sbx.ppo.policies.Critic` so SBX's
    :meth:`PPOPolicy.build` can construct us with its standard kwargs.

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
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.tanh
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
        x = obs[..., ACTOR_OBS_DIM:]
        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            x = self.activation_fn(x)
        return nn.Dense(1)(x)


class AsymmetricActorCriticPolicy(PPOPolicy):
    """PPOPolicy subclass routing flat-concat obs to slice-aware actor/critic.

    The flat-concat layout (env_gym module docstring) means SBX's PPO
    training loop is unmodified — it passes the single ``observations``
    tensor to both actor and critic, and each module's ``__call__`` slices
    its own half. As a result, the bulk of :meth:`PPOPolicy.build`,
    :meth:`PPOPolicy._predict`, :meth:`PPOPolicy._predict_all`, and the
    optimizer / ``TrainState`` plumbing are inherited unchanged.

    Parameters
    ----------
    observation_space : gym.spaces.Space
        Must be a flat ``Box`` of shape ``(2 * ACTOR_OBS_DIM,)``.
    action_space : gym.spaces.Space
        Raw 4-vec action ``Box``.
    lr_schedule : Schedule
        SB3 learning-rate schedule callable.
    **kwargs : Any
        Forwarded to :class:`sbx.ppo.policies.PPOPolicy`. Any of these
        kwargs that this class needs to fix (``actor_class``,
        ``critic_class``, ``net_arch``) are overridden with our values to
        keep call sites simple — the parent's behaviour for everything else
        (optimizer, ortho_init, log_std_init, activation_fn) is preserved.

    Raises:
    ------
    ValueError
        If ``observation_space`` is not a ``Box`` of shape
        ``(2 * ACTOR_OBS_DIM,)``.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        **kwargs: Any,
    ):
        """Construct the policy. See class docstring for parameter details."""
        if not isinstance(observation_space, gym.spaces.Box):
            raise ValueError(
                "AsymmetricActorCriticPolicy requires a Box observation space; "
                f"got {type(observation_space).__name__}."
            )
        if observation_space.shape != (FLAT_CONCAT_OBS_DIM,):
            raise ValueError(
                "AsymmetricActorCriticPolicy expects observation_space.shape == "
                f"({FLAT_CONCAT_OBS_DIM},) (= 2 * ACTOR_OBS_DIM); got "
                f"{observation_space.shape}."
            )

        # Fix the actor/critic classes and architecture. Caller-supplied
        # values for these would silently get ignored otherwise, which would
        # be surprising — raise instead.
        for forbidden in ("actor_class", "critic_class", "net_arch"):
            if forbidden in kwargs:
                raise ValueError(
                    f"AsymmetricActorCriticPolicy does not accept '{forbidden}'; "
                    "the slice-aware Actor/Critic and rl_song-matched NET_ARCH are "
                    "intrinsic to this class."
                )

        kwargs.setdefault("log_std_init", LOG_STD_INIT)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=list(NET_ARCH),
            actor_class=Actor,
            critic_class=Critic,
            **kwargs,
        )

    def build(self, key: jax.Array, lr_schedule: Schedule, max_grad_norm: float) -> jax.Array:
        """Construct ``self.actor`` / ``self.vf`` and their ``TrainState``s.

        Mirrors :meth:`sbx.ppo.policies.PPOPolicy.build` but constructs the
        Actor without SBX's image-features-extractor kwargs (our Actor and
        Critic accept ``features_extractor=None`` and ignore it, but SBX's
        stock build expands ``self.features_extractor_kwargs`` which is
        empty for a non-image policy — keeping the implementation explicit
        avoids relying on that emptiness as a load-bearing invariant).

        Parameters
        ----------
        key : jax.Array
            PRNG key; split into ``(actor_key, vf_key, kept_key)``.
        lr_schedule : Schedule
            Learning-rate schedule sampled once at ``step=1`` for
            :func:`optax.inject_hyperparams`; SBX's training loop later
            mutates ``opt_state[1].hyperparams["learning_rate"]`` per
            iteration to anneal.
        max_grad_norm : float
            Global-norm clip applied before the Adam update.

        Returns:
        -------
        jax.Array
            A fresh PRNG key the caller can continue splitting from.
        """
        key, actor_key, vf_key = jax.random.split(key, 3)
        # SBX uses ``self.key`` for gSDE exploration noise; preserve that
        # contract even though we don't use gSDE.
        key, self.key = jax.random.split(key, 2)
        self.reset_noise()

        dummy_obs = jnp.zeros((1, FLAT_CONCAT_OBS_DIM), dtype=jnp.float32)
        action_dim = int(np.prod(self.action_space.shape))

        self.actor = self.actor_class(
            action_dim=action_dim,
            net_arch=self.net_arch_pi,
            log_std_init=self.log_std_init,
            activation_fn=self.activation_fn,
            ortho_init=self.ortho_init,
        )
        # Stub used by SBX's gSDE path; harmless when ``use_sde=False``.
        self.actor.reset_noise = self.reset_noise

        # ``inject_hyperparams`` lets SBX rewrite the LR in-place per iter.
        # Adam ``eps=1e-5`` matches PPOPolicy.__init__ default.
        optimizer = optax.inject_hyperparams(self.optimizer_class)(
            learning_rate=lr_schedule(1), **self.optimizer_kwargs
        )

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, dummy_obs),
            tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optimizer),
        )

        self.vf = self.critic_class(net_arch=self.net_arch_vf, activation_fn=self.activation_fn)
        self.vf_state = TrainState.create(
            apply_fn=self.vf.apply,
            params=self.vf.init(vf_key, dummy_obs),
            tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optimizer),
        )

        # JIT the apply functions — matches SBX's PPOPolicy.build.
        self.actor.apply = jax.jit(self.actor.apply)  # type: ignore[method-assign]
        self.vf.apply = jax.jit(self.vf.apply)  # type: ignore[method-assign]

        return key
