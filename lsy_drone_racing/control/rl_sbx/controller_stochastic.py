"""Stochastic-deploy variant of :class:`RLSBXController` for the gap diagnosis.

The default deploy controller takes the Gaussian mean of the actor distribution.
v128's eval showed 0/10 deterministic finishes despite 7.65% training-time
stochastic finishes, suggesting the mean policy lives in a different (non-
finishing) mode than the action distribution.

This controller swaps ``dist.mean()`` for ``dist.sample(seed=...)`` so we can
test whether a one-shot stochastic deploy reproduces training-time behavior. Not
for real-drone deploy — diagnostic only.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from lsy_drone_racing.control.rl_sbx.controller import RLSBXController
from lsy_drone_racing.control.rl_song import obs as obs_encoding
from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_DIM
from lsy_drone_racing.control.rl_song.policy import raw_to_env_action

if TYPE_CHECKING:
    import numpy.typing as npt


_SEED_MOD: int = 2**31 - 1


class RLSBXStochasticController(RLSBXController):
    """Drop-in for :class:`RLSBXController` that samples instead of taking the mean."""

    def __init__(
        self,
        obs: dict[str, npt.NDArray[np.floating]],
        info: dict,
        config: dict,
    ) -> None:
        """Construct the parent controller and seed the sampling RNG from time.

        ``sim.py`` reinstantiates the controller per episode, so a fixed seed
        would replay the same sample sequence every episode. Seeding from the
        wall clock gives independent samples across episodes within one
        multi-run invocation.
        """
        super().__init__(obs, info, config)
        seed = int(time.monotonic_ns()) % _SEED_MOD
        self._sample_rng: jax.Array = jax.random.PRNGKey(seed)
        print(f"[STOCHASTIC] sampling actor with seed={seed}")

    def compute_control(
        self,
        obs: dict[str, npt.NDArray[np.floating]],
        info: dict | None = None,
    ) -> npt.NDArray[np.floating]:
        """Run the actor with action sampling and return a 4-d env action."""
        del info
        jax_obs = {key: jnp.asarray(value) for key, value in obs.items()}
        actor_obs = obs_encoding.build_actor_obs(
            jax_obs, self.prev_action_env_4vec, self.actor_normalizer
        )
        flat_obs = jnp.concatenate(
            [actor_obs, jnp.zeros((ACTOR_OBS_DIM,), dtype=actor_obs.dtype)], axis=-1
        )
        dist = self._actor.apply(self.actor_params, flat_obs[None, :])
        self._sample_rng, sub = jax.random.split(self._sample_rng)
        raw_action = dist.sample(seed=sub)[0]
        env_action = raw_to_env_action(
            raw_action,
            jax_obs["quat"],
            self.thrust_min,
            self.thrust_max,
            alpha_max=self.alpha_max_rad,
        )
        self.prev_action_env_4vec = env_action
        return np.asarray(env_action, dtype=np.float32)
