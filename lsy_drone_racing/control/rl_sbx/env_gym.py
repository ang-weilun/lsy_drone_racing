"""SB3 ``VecEnv`` adapter for the JAX crazyflow race env.

Bridges :class:`lsy_drone_racing.envs.drone_race.VecDroneRaceEnv` to the
Stable-Baselines3 / SBX vectorized-env protocol so the SBX PPO training loop
in :mod:`sbx` can drive the same JAX simulation the Song-2023 prototype used.

Three design choices distinguish this wrapper from
:class:`lsy_drone_racing.control.rl_song.env_wrapper.RLSongVecEnv`.

1. **Flat-concat observation space.** A single ``Box(2*ACTOR_OBS_DIM,)`` whose
   first half is the masked actor obs and second half is the privileged critic
   obs. The Task 4 ``Actor`` / ``Critic`` flax modules slice their respective
   half. ``sbx.PPO`` has no dict-obs support
   (``sbx/ppo/ppo.py:297`` calls ``rollout_data.observations.numpy()``
   unconditionally), so a single tensor is the only transport that works with
   the stock training loop. See the 2026-05-24 addendum in
   ``docs/specs/2026-05-24-sbx-migration-design.md``.
2. **Masked-geometry reward.** ``step_reward`` is called without ``true_*``
   kwargs, so the reward gradients only see what the actor sees. This is
   risk-3 mitigation from the migration design: the v85 line of attack
   removes the masked-vs-true reward leak that hid the gate-3 graze.
3. **Two independent normalizers.** ``self.actor_normalizer`` is fed the
   masked obs (what the actor will see at deploy); ``self.critic_normalizer``
   is fed the privileged obs (what only the critic ever sees). Both are
   updated externally by a SB3 callback after each rollout — this wrapper
   only exposes setters and the current state.

The wrapper depends on the JAX env's private ``_reset(data, seed=None,
mask=mask)`` to autoreset only the worlds that finished on the last step.
The public ``.reset()`` resets every world, which would discard half the
rollout buffer in an off-policy or asynchronous setting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax import Array
from stable_baselines3.common.vec_env import VecEnv

from lsy_drone_racing.control.rl_song import obs as obs_encoding
from lsy_drone_racing.control.rl_song.config import (
    ACTOR_OBS_DIM,
    ENV_ACTION_DIM,
    RAW_ACTION_DIM,
    TANGENT_ALPHA_MAX_RAD,
    RewardConfig,
)
from lsy_drone_racing.control.rl_song.policy import raw_to_env_action
from lsy_drone_racing.control.rl_song.reward import step_reward
from lsy_drone_racing.envs.race_core import obs as race_core_obs

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

# Bounds for the SB3 Box observation space. The normalizer clips every feature
# into ``±NORM_CLIP``, so anything beyond this range is a normalizer bug.
OBS_LOW: float = -obs_encoding.NORM_CLIP
OBS_HIGH: float = +obs_encoding.NORM_CLIP


class RLSBXVecEnv(VecEnv):
    """SB3 ``VecEnv`` wrapping a JAX :class:`VecDroneRaceEnv`.

    Parameters
    ----------
    jax_env : VecDroneRaceEnv
        Already-constructed vectorized JAX env. The wrapper does not create or
        configure the env — that's the caller's job (see Task 7 / the training
        CLI). The wrapper does call the env's ``.step``, ``._reset``, and
        reads ``.data``.
    reward_cfg : RewardConfig
        Reward shaping configuration consumed by
        :func:`lsy_drone_racing.control.rl_song.reward.step_reward`.
    alpha_max : float
        Tangent-space scaling for the raw-to-env action projection (Schuck
        et al. 2025).
    thrust_min, thrust_max : float
        Total-thrust bounds in newtons. Used by the raw-to-env projection to
        rescale the policy's collective-thrust output.
    n_envs : int
        Vectorization width. Must equal ``jax_env``'s vec width.
    seed : int
        Initial JAX env seed.

    Notes:
    -----
    SB3's modern (2.x) VecEnv contract is enforced:

    * ``reset(self)`` takes no args and returns just the obs.
    * ``step_wait()`` returns ``(obs, reward, done, infos)`` — a single ``done``
      flag combining terminated and truncated, with ``TimeLimit.truncated``
      stored on the per-env info dict so PPO can do correct value
      bootstrapping on timeouts.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        jax_env: Any,
        reward_cfg: RewardConfig,
        alpha_max: float = TANGENT_ALPHA_MAX_RAD,
        *,
        thrust_min: float,
        thrust_max: float,
        n_envs: int,
        seed: int,
    ):
        """Construct the wrapper. See the class docstring for parameter details."""
        observation_space = spaces.Box(
            low=OBS_LOW, high=OBS_HIGH, shape=(2 * ACTOR_OBS_DIM,), dtype=np.float32
        )
        action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(RAW_ACTION_DIM,), dtype=np.float32
        )
        super().__init__(
            num_envs=n_envs, observation_space=observation_space, action_space=action_space
        )

        self.jax_env = jax_env
        self.reward_cfg = reward_cfg
        self.alpha_max = float(alpha_max)
        self.thrust_min = float(thrust_min)
        self.thrust_max = float(thrust_max)
        self.seed_value = int(seed)

        self.actor_normalizer = obs_encoding.init_normalizer(ACTOR_OBS_DIM)
        self.critic_normalizer = obs_encoding.init_normalizer(ACTOR_OBS_DIM)

        self._prev_action: Array = jnp.zeros((n_envs, ENV_ACTION_DIM), dtype=jnp.float32)
        self._prev_env_obs: dict[str, Array] | None = None
        self._pending_actions: np.ndarray | None = None

    # ------------------------------------------------------------------
    # SB3 VecEnv interface
    # ------------------------------------------------------------------
    def reset(self) -> VecEnvObs:
        """Reset every world and return the flat-concat observation.

        Returns:
        -------
        observations : np.ndarray, shape (n_envs, 2*ACTOR_OBS_DIM)
            First half is masked actor obs, second half is privileged critic
            obs. See module docstring for the layout rationale.
        """
        env_obs, _info = self.jax_env.reset(seed=self.seed_value)
        env_obs = _to_jax_obs(env_obs)
        self._prev_env_obs = env_obs
        self._prev_action = jnp.zeros((self.num_envs, ENV_ACTION_DIM), dtype=jnp.float32)
        return self._build_obs(env_obs)

    def step_async(self, actions: np.ndarray) -> None:
        """Stash actions for the next :meth:`step_wait`."""
        self._pending_actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self) -> VecEnvStepReturn:
        """Apply the stashed actions and return ``(obs, reward, done, infos)``.

        Returns:
        -------
        obs : np.ndarray, shape (n_envs, 2*ACTOR_OBS_DIM)
            Flat-concat obs, see :meth:`reset` for layout.
        reward : np.ndarray, shape (n_envs,)
        done : np.ndarray, shape (n_envs,)
            ``terminated | truncated``. SB3 PPO uses the per-env
            ``TimeLimit.truncated`` flag on the ``infos`` list to decide
            whether to bootstrap value on timeout.
        infos : list[dict]
        """
        if self._pending_actions is None:
            raise RuntimeError("step_wait called before step_async.")
        if self._prev_env_obs is None:
            raise RuntimeError("step_wait called before reset.")

        raw_action = jnp.asarray(self._pending_actions, dtype=jnp.float32)
        self._pending_actions = None
        prev_env_obs = self._prev_env_obs

        env_action = raw_to_env_action(
            raw_action,
            jnp.asarray(prev_env_obs["quat"]),
            self.thrust_min,
            self.thrust_max,
            alpha_max=self.alpha_max,
        )

        env_obs, _env_reward, terminated, truncated, _env_info = self.jax_env.step(env_action)
        env_obs = _to_jax_obs(env_obs)

        terminated = jnp.asarray(terminated, dtype=bool)
        truncated = jnp.asarray(truncated, dtype=bool)
        current_target = env_obs["target_gate"]
        prev_target = prev_env_obs["target_gate"]
        finished = current_target < 0
        terminated = terminated | finished
        gate_just_passed = ((current_target > prev_target) & (prev_target >= 0)) | (
            finished & (prev_target >= 0)
        )

        # Masked-geometry reward: no ``true_*`` kwargs. The reward sees only
        # what the actor sees, so gradients can't exploit information the
        # policy is denied at deploy. The critic still consumes privileged
        # values via the ``critic`` channel of the dict obs.
        reward, _components = step_reward(
            env_obs,
            prev_env_obs,
            terminated,
            truncated,
            finished,
            gate_just_passed,
            self.reward_cfg,
        )

        done = terminated | truncated
        done_np = np.asarray(done)
        truncated_np = np.asarray(truncated)
        terminated_np = np.asarray(terminated)

        # Autoreset only worlds that finished. The high-level ``.reset()``
        # resets every world, which would discard fresh in-flight transitions.
        # ``race_core_obs(data)`` is shape ``(n_envs, n_drones=1, ...)`` — the
        # high-level ``VecDroneRaceEnv.step/reset`` squeezes the drone axis
        # before returning. We replicate that squeeze here so the obs dict is
        # consistent across the pre-step and post-autoreset paths.
        if bool(done_np.any()):
            self.jax_env.data, _ = self.jax_env._reset(self.jax_env.data, seed=None, mask=done)
            raw_env_obs = race_core_obs(self.jax_env.data)
            env_obs = {key: jnp.asarray(value[:, 0]) for key, value in raw_env_obs.items()}

        reset_prev_action = jnp.zeros_like(env_action)
        self._prev_action = jnp.where(done[:, None], reset_prev_action, env_action)
        self._prev_env_obs = env_obs

        obs_array = self._build_obs(env_obs)
        reward_np = np.asarray(reward, dtype=np.float32)
        infos = _build_infos(self.num_envs, terminated_np, truncated_np, obs_array)
        return obs_array, reward_np, done_np, infos

    def close(self) -> None:
        """Close the underlying JAX env."""
        if self.jax_env is not None:
            self.jax_env.close()

    def seed(self, seed: int | None = None) -> list[int | None]:
        """Set the wrapper seed.

        The JAX env is reseeded on the next :meth:`reset`. SB3's
        :class:`VecNormalize` and friends call this once at construction.
        """
        if seed is not None:
            self.seed_value = int(seed)
        return [self.seed_value] * self.num_envs

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        """Return ``getattr(self, attr_name)`` replicated per requested env.

        The wrapper is a single Python object backing all ``num_envs``
        simulated envs, so the attribute is the same across indices.
        """
        n = _resolve_indices_count(indices, self.num_envs)
        return [getattr(self, attr_name)] * n

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        """Set ``attr_name`` on the wrapper.

        Per-env setting is not supported because the wrapper holds a single
        JAX env shared across all vectorized worlds.
        """
        if indices is not None:
            raise NotImplementedError(
                "RLSBXVecEnv.set_attr does not support per-env indices: all worlds "
                "share a single backing JAX env."
            )
        setattr(self, attr_name, value)

    def env_method(
        self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any
    ) -> list[Any]:
        """Call ``method_name`` on the wrapper and replicate the result.

        Mirrors :meth:`get_attr`'s single-backing-env semantics.
        """
        n = _resolve_indices_count(indices, self.num_envs)
        result = getattr(self, method_name)(*method_args, **method_kwargs)
        return [result] * n

    def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
        """Return ``False`` for every env — this wrapper is the only layer."""
        n = _resolve_indices_count(indices, self.num_envs)
        del wrapper_class
        return [False] * n

    # ------------------------------------------------------------------
    # Normalizer accessors (consumed by the Task 6 callback)
    # ------------------------------------------------------------------
    def set_actor_normalizer(self, normalizer: obs_encoding.NormalizerState) -> None:
        """Replace the running actor-obs normalizer."""
        self.actor_normalizer = normalizer

    def set_critic_normalizer(self, normalizer: obs_encoding.NormalizerState) -> None:
        """Replace the running critic-obs normalizer."""
        self.critic_normalizer = normalizer

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _build_obs(self, env_obs: dict[str, Array]) -> np.ndarray:
        """Encode the flat-concat ``[actor | critic]`` obs of shape ``(n_envs, 2D)``.

        The actor half is built from masked geometry via
        :func:`obs_encoding.vmap_build_actor_obs`; the critic half is built
        from privileged geometry via :func:`obs_encoding.vmap_build_critic_obs`
        with ``true_*`` kwargs read straight from the JAX env's ``data``. Each
        half is normalized by its own ``NormalizerState`` before
        concatenation.
        """
        actor = obs_encoding.vmap_build_actor_obs(env_obs, self._prev_action, self.actor_normalizer)
        critic = obs_encoding.vmap_build_critic_obs(
            env_obs,
            self._prev_action,
            self.critic_normalizer,
            true_gates_pos=jnp.asarray(self.jax_env.data.gates_pos),
            true_gates_quat=jnp.asarray(self.jax_env.data.gates_quat),
            true_obstacles_pos=jnp.asarray(self.jax_env.data.obstacles_pos),
        )
        return np.concatenate(
            [np.asarray(actor, dtype=np.float32), np.asarray(critic, dtype=np.float32)], axis=-1
        )


def _to_jax_obs(env_obs: dict[str, Any]) -> dict[str, Array]:
    """Convert an env observation dict to JAX arrays."""
    return {key: jnp.asarray(value) for key, value in env_obs.items()}


def _resolve_indices_count(indices: Any, num_envs: int) -> int:
    """Return the number of envs targeted by SB3's ``indices`` argument."""
    if indices is None:
        return num_envs
    if isinstance(indices, int):
        return 1
    return len(list(indices))


def _build_infos(
    num_envs: int, terminated: np.ndarray, truncated: np.ndarray, obs: np.ndarray
) -> list[dict[str, Any]]:
    """Build the per-env info list with SB3 timeout-bootstrap fields.

    Parameters
    ----------
    num_envs : int
    terminated, truncated : np.ndarray, shape (n_envs,)
    obs : np.ndarray, shape (n_envs, 2*ACTOR_OBS_DIM)
        Already-built post-step (post-autoreset) flat-concat obs. Stored under
        ``terminal_observation`` for envs whose ``done`` flag is set, so
        SB3's PPO can bootstrap correctly on timeouts. Note this is the
        observation of the freshly-reset world, not of the terminal state
        before reset — matching the SB3 ``DummyVecEnv`` convention where
        ``terminal_observation`` is the *reset* obs that VecEnv consumers
        already have in hand.
    """
    infos: list[dict[str, Any]] = [{} for _ in range(num_envs)]
    done_mask = terminated | truncated
    for env_idx in range(num_envs):
        if not done_mask[env_idx]:
            continue
        infos[env_idx]["TimeLimit.truncated"] = bool(truncated[env_idx] and not terminated[env_idx])
        infos[env_idx]["terminal_observation"] = obs[env_idx]
    return infos
