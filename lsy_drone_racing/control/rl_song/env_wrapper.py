"""Vectorized racing-env adapter for the Song-2023 PPO pipeline."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from drone_models.core import load_params
from jax import Array
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.rl_song import obs as obs_encoding
from lsy_drone_racing.control.rl_song.config import (
    ACTOR_OBS_DIM,
    ENV_ACTION_DIM,
    RAW_ACTION_DIM,
    CurriculumConfig,
    CurriculumStage,
    RewardConfig,
    TrainConfig,
    default_curriculum,
)
from lsy_drone_racing.control.rl_song.policy import raw_to_env_action
from lsy_drone_racing.control.rl_song.reward import step_reward
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.envs.race_core import _reset_env_data
from lsy_drone_racing.envs.race_core import obs as race_core_obs
from lsy_drone_racing.utils import load_config

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
LEVEL_CONFIG_NAME: str = "level{level}.toml"
N_DRONES: int = 1
TOTAL_THRUST_MULTIPLIER: float = 4.0
TRACK_RANDOMIZATION_KEYS: frozenset[str] = frozenset(
    {"gate_pos", "gate_rpy", "obstacle_pos"}
)


class RLSongVecEnv:
    """Thin vector-env wrapper exposing PPO-ready tensors.

    Parameters
    ----------
    train_cfg : TrainConfig, optional
        Training configuration bundle.
    n_envs : int, optional
        Number of vectorized environments. Defaults to ``train_cfg.ppo.n_envs``.
    stage_idx : int, optional
        Zero-indexed curriculum stage. Defaults to
        ``train_cfg.initial_stage_index``.
    seed : int, optional
        Random seed for the env and reset perturbations.
    device : {"cpu", "gpu"}, optional
        JAX device string passed to ``VecDroneRaceEnv``.

    Notes
    -----
    The wrapped env still autoresets internally. This wrapper observes the
    env's reset mask and applies curriculum reset perturbations to newly reset
    worlds before building policy observations.
    """

    def __init__(
        self,
        train_cfg: TrainConfig | None = None,
        *,
        n_envs: int | None = None,
        stage_idx: int | None = None,
        seed: int | None = None,
        device: str = "gpu",
    ):
        self.train_cfg = TrainConfig() if train_cfg is None else train_cfg
        self.curriculum = self.train_cfg.curriculum
        self.reward_cfg = self.train_cfg.reward
        self.n_envs = self.train_cfg.ppo.n_envs if n_envs is None else n_envs
        self.seed = self.train_cfg.seed if seed is None else seed
        self.device = device
        self.max_episode_steps = self.train_cfg.max_episode_steps
        self.rng_key = jax.random.PRNGKey(self.seed)
        self.normalizer = obs_encoding.init_normalizer(ACTOR_OBS_DIM)
        self.prev_action_env_4vec = jnp.zeros(
            (self.n_envs, ENV_ACTION_DIM), dtype=jnp.float32
        )
        self.env: VecDroneRaceEnv | None = None
        self.stage_idx = -1
        self.stage = self._stage_from_index(
            self.train_cfg.initial_stage_index if stage_idx is None else stage_idx
        )

        initial_cfg = self._load_stage_config(self.stage)
        drone_params = load_params(initial_cfg.sim.physics, initial_cfg.sim.drone_model)
        self._thrust_bounds = (
            float(drone_params["thrust_min"] * TOTAL_THRUST_MULTIPLIER),
            float(drone_params["thrust_max"] * TOTAL_THRUST_MULTIPLIER),
        )
        self.current_env_obs: dict[str, Array] | None = None
        self.prev_env_obs: dict[str, Array] | None = None
        self.set_stage(self.curriculum.stages.index(self.stage))

    def reset(self, seed: int | None = None) -> tuple[dict[str, Array], dict[str, Any]]:
        """Reset every vectorized world and return actor/critic observations.

        Parameters
        ----------
        seed : int, optional
            Env seed. When provided, the wrapper's JAX reset-perturbation key is
            also reset to this seed.

        Returns
        -------
        observations : dict[str, Array]
            ``actor_obs`` and ``critic_obs`` arrays, each shaped
            ``(n_envs, ACTOR_OBS_DIM)``.
        info : dict[str, Any]
            Info dictionary returned by the underlying env.
        """
        if self.env is None:
            raise RuntimeError("RLSongVecEnv.reset called before env construction.")
        if seed is not None:
            self.rng_key = jax.random.PRNGKey(seed)

        env_obs, info = self.env.reset(seed=seed)
        self.current_env_obs = _to_jax_obs(env_obs)
        reset_mask = jnp.ones((self.n_envs,), dtype=bool)
        self._apply_reset_perturbation(reset_mask)
        self.prev_action_env_4vec = jnp.zeros(
            (self.n_envs, ENV_ACTION_DIM), dtype=jnp.float32
        )
        self.current_env_obs = self._read_env_obs()
        self.prev_env_obs = self.current_env_obs
        return self.build_observations(), info

    def step(
        self, raw_action: Array
    ) -> tuple[dict[str, Array], Array, Array, Array, dict[str, Any]]:
        """Step the env with raw policy actions.

        Parameters
        ----------
        raw_action : Array, shape (n_envs, RAW_ACTION_DIM)
            Raw 7-vector sampled by the policy. The log probability must have
            already been computed before this call.

        Returns
        -------
        observations : dict[str, Array]
            Next ``actor_obs`` and ``critic_obs`` tensors.
        reward : Array, shape (n_envs,)
            Song reward replacing the env's sparse reward.
        terminated : Array, shape (n_envs,)
            Termination flags.
        truncated : Array, shape (n_envs,)
            Timeout flags.
        info : dict[str, Any]
            Reward components and rollout metrics.
        """
        if self.env is None or self.current_env_obs is None:
            raise RuntimeError("RLSongVecEnv.step called before reset.")
        _validate_action_shape(raw_action, self.n_envs, RAW_ACTION_DIM, "raw_action")

        prev_env_obs = self.current_env_obs
        thrust_min, thrust_max = self.get_thrust_bounds()
        env_action = raw_to_env_action(jnp.asarray(raw_action), thrust_min, thrust_max)
        _validate_action_shape(env_action, self.n_envs, ENV_ACTION_DIM, "env_action")

        env_obs, _, terminated, truncated, env_info = self.env.step(env_action)
        self.current_env_obs = _to_jax_obs(env_obs)

        terminated = jnp.asarray(terminated, dtype=bool)
        truncated = jnp.asarray(truncated, dtype=bool)
        current_target = self.current_env_obs["target_gate"]
        prev_target = prev_env_obs["target_gate"]
        finished = current_target < 0
        terminated = terminated | finished
        gate_just_passed = ((current_target > prev_target) & (prev_target >= 0)) | (
            finished & (prev_target >= 0)
        )

        reward, components = step_reward(
            self.current_env_obs,
            prev_env_obs,
            terminated,
            truncated,
            finished,
            gate_just_passed,
            self.reward_cfg,
            true_gates_pos=self.true_gates_pos(),
            true_obstacles_pos=self.true_obstacles_pos(),
        )

        done = terminated | truncated
        if bool(np.asarray(jnp.any(done))):
            self._reset_done_worlds(done)
            self.current_env_obs = self._read_env_obs()

        reset_prev_action = jnp.zeros_like(env_action)
        self.prev_action_env_4vec = jnp.where(
            done[:, None], reset_prev_action, env_action
        )
        self.prev_env_obs = self.current_env_obs
        n_gates = prev_env_obs["gates_pos"].shape[1]
        target_gate_progress = jnp.where(finished, n_gates, current_target)

        info = dict(env_info)
        info.update(
            {
                "reward_components": components,
                "target_gate": current_target,
                "target_gate_progress": target_gate_progress,
                "finished": finished,
                "gate_just_passed": gate_just_passed,
                "crash": terminated & ~finished,
                "env_action": env_action,
            }
        )
        return self.build_observations(), reward, terminated, truncated, info

    def build_observations(self) -> dict[str, Array]:
        """Build normalized actor and critic observations for current env state.

        Returns
        -------
        observations : dict[str, Array]
            ``actor_obs`` and ``critic_obs`` arrays with shape
            ``(n_envs, ACTOR_OBS_DIM)``.
        """
        if self.current_env_obs is None:
            raise RuntimeError("No current env observation is available.")
        actor_obs = obs_encoding.vmap_build_actor_obs(
            self.current_env_obs, self.prev_action_env_4vec, self.normalizer
        )
        critic_obs = self._build_critic_obs(self.current_env_obs)
        return {"actor_obs": actor_obs, "critic_obs": critic_obs}

    def update_normalizer_from_batch(self, normalized_actor_obs: Array) -> None:
        """Update the running observation normalizer from a rollout batch.

        Parameters
        ----------
        normalized_actor_obs : Array, shape (n_samples, ACTOR_OBS_DIM)
            Actor observations built with the previous normalizer state.

        Notes
        -----
        ``obs.py`` currently exposes only the normalized builder. The wrapper
        inverts the affine normalization before the Welford update; clipped
        features remain clipped and are therefore conservative outliers.
        """
        batch = jnp.asarray(normalized_actor_obs)
        _validate_batch_shape(batch, ACTOR_OBS_DIM, "normalized_actor_obs")
        std = jnp.sqrt(self.normalizer.var + obs_encoding.NORM_VAR_EPS)
        raw_batch = batch * std + self.normalizer.mean
        self.normalizer = obs_encoding.update_normalizer(self.normalizer, raw_batch)

    def set_normalizer(self, normalizer: obs_encoding.NormalizerState) -> None:
        """Replace the wrapper normalizer with a frozen/restored state."""
        self.normalizer = normalizer

    def set_stage(self, stage_idx: int) -> None:
        """Switch curriculum stage and reinstantiate the vectorized env.

        Parameters
        ----------
        stage_idx : int
            Zero-indexed curriculum stage.
        """
        self.stage = self._stage_from_index(stage_idx)
        self.stage_idx = stage_idx
        if self.env is not None:
            self.env.close()
        self.env = self._make_env(self.stage)
        self.prev_action_env_4vec = jnp.zeros(
            (self.n_envs, ENV_ACTION_DIM), dtype=jnp.float32
        )
        self.reset(seed=self.seed + stage_idx)

    def get_thrust_bounds(self) -> tuple[float, float]:
        """Return cached total-thrust bounds ``(min, max)`` in newtons."""
        return self._thrust_bounds

    def true_gates_pos(self) -> Array:
        """Return unmasked true gate positions, shape ``(n_envs, n_gates, 3)``."""
        if self.env is None:
            raise RuntimeError("Env is not constructed.")
        return jnp.asarray(self.env.data.gates_pos)

    def true_obstacles_pos(self) -> Array:
        """Return true obstacle positions, shape ``(n_envs, n_obstacles, 3)``."""
        if self.env is None:
            raise RuntimeError("Env is not constructed.")
        return jnp.asarray(self.env.data.obstacles_pos)

    def render(self) -> None:
        """Render the underlying env through Crazyflow."""
        if self.env is None:
            raise RuntimeError("Env is not constructed.")
        self.env.render()

    def close(self) -> None:
        """Close the underlying env."""
        if self.env is not None:
            self.env.close()

    def _make_env(self, stage: CurriculumStage) -> VecDroneRaceEnv:
        """Create a ``VecDroneRaceEnv`` for one curriculum stage."""
        race_cfg = self._load_stage_config(stage)
        randomizations = self._stage_randomizations(race_cfg, stage)
        disturbances = self._stage_disturbances(stage)
        return VecDroneRaceEnv(
            num_envs=self.n_envs,
            freq=race_cfg.env.freq,
            sim_config=race_cfg.sim,
            track=race_cfg.env.track,
            sensor_range=race_cfg.env.sensor_range,
            control_mode="attitude",
            disturbances=disturbances,
            randomizations=randomizations,
            seed=self.seed,
            max_episode_steps=self.max_episode_steps,
            device=self.device,
        )

    def _load_stage_config(self, stage: CurriculumStage) -> ConfigDict:
        """Load and patch the TOML config for a stage without touching disk."""
        config_path = REPO_ROOT / "config" / LEVEL_CONFIG_NAME.format(level=stage.level)
        race_cfg = copy.deepcopy(load_config(config_path))
        race_cfg.env.control_mode = "attitude"
        return race_cfg

    def _stage_randomizations(
        self, race_cfg: ConfigDict, stage: CurriculumStage
    ) -> ConfigDict | None:
        """Return env randomizations that belong to the course track, not DR."""
        specs = getattr(race_cfg.env, "randomizations", ConfigDict())
        selected = ConfigDict()
        if stage.level == 3:
            for key in TRACK_RANDOMIZATION_KEYS:
                if key in specs:
                    selected[key] = specs[key]
        # TODO(stage4): translate DRSchedule mass, inertia, thrust-scale, motor
        # delay, drag, sensing-noise, latency, and wind channels into Crazyflow
        # reset/step hooks when curriculum stage 4 is enabled.
        return selected if selected else None

    def _stage_disturbances(self, stage: CurriculumStage) -> ConfigDict | None:
        """Return disturbances for a curriculum stage."""
        _ = stage
        # TODO(stage4): inject DRSchedule action/dynamics disturbances here.
        return None

    def _build_critic_obs(self, env_obs: dict[str, Array]) -> Array:
        """Build batched critic obs while preserving the stage-3 seam."""
        if hasattr(obs_encoding, "vmap_build_critic_obs"):
            return obs_encoding.vmap_build_critic_obs(
                env_obs, self.prev_action_env_4vec, self.normalizer
            )
        # TODO(stage3): once obs.py exposes privileged critic features, remove
        # this compatibility path and call its vectorized critic encoder.
        return jax.vmap(
            obs_encoding.build_critic_obs,
            in_axes=({key: 0 for key in env_obs}, 0, None),
        )(env_obs, self.prev_action_env_4vec, self.normalizer)

    def _apply_reset_perturbation(self, mask: Array) -> None:
        """Apply curriculum reset perturbations to selected env worlds."""
        if self.env is None:
            raise RuntimeError("Env is not constructed.")
        mask = jnp.asarray(mask, dtype=bool)
        if not bool(np.asarray(jnp.any(mask))):
            return

        self.rng_key, pos_key, vel_key, yaw_key = jax.random.split(self.rng_key, 4)
        states = self.env.data.sim_data.states
        pos_delta = jax.random.uniform(
            pos_key,
            shape=states.pos.shape,
            minval=-self.stage.reset_pos_perturb_m,
            maxval=self.stage.reset_pos_perturb_m,
        )
        vel = jax.random.uniform(
            vel_key,
            shape=states.vel.shape,
            minval=-self.stage.reset_vel_perturb_mps,
            maxval=self.stage.reset_vel_perturb_mps,
        )
        yaw_delta = jax.random.uniform(
            yaw_key,
            shape=states.quat.shape[:-1],
            minval=-self.stage.reset_yaw_perturb_rad,
            maxval=self.stage.reset_yaw_perturb_rad,
        )

        mask_broadcast = mask[:, None, None]
        pos = jnp.clip(
            states.pos + pos_delta,
            self.env.data.pos_limit_low,
            self.env.data.pos_limit_high,
        )
        quat = _apply_yaw_delta(states.quat, yaw_delta, mask_broadcast)
        states = states.replace(
            pos=jnp.where(mask_broadcast, pos, states.pos),
            vel=jnp.where(mask_broadcast, vel, states.vel),
            quat=quat,
        )
        sim_data = self.env.data.sim_data.replace(states=states)
        self.env.data = self.env.data.replace(sim_data=sim_data)
        self.env.data = _reset_env_data(self.env.data, mask)

    def _reset_done_worlds(self, mask: Array) -> None:
        """Reset completed worlds after terminal rewards have been computed."""
        if self.env is None:
            raise RuntimeError("Env is not constructed.")
        self.env.data, _ = self.env._reset(self.env.data, seed=None, mask=mask)
        self._apply_reset_perturbation(mask)

    def _read_env_obs(self) -> dict[str, Array]:
        """Read squeezed env observations directly from ``RaceCoreEnv.data``."""
        if self.env is None:
            raise RuntimeError("Env is not constructed.")
        env_obs = race_core_obs(self.env.data)
        return {key: jnp.asarray(value[:, 0]) for key, value in env_obs.items()}

    def _stage_from_index(self, stage_idx: int) -> CurriculumStage:
        """Validate and return a curriculum stage."""
        if stage_idx < 0 or stage_idx >= len(self.curriculum.stages):
            raise ValueError(
                f"stage_idx must be in [0, {len(self.curriculum.stages) - 1}], "
                f"got {stage_idx}"
            )
        return self.curriculum.stages[stage_idx]


def make_env(
    train_cfg: TrainConfig | None = None,
    *,
    n_envs: int | None = None,
    stage_idx: int | None = None,
    seed: int | None = None,
    device: str = "gpu",
) -> RLSongVecEnv:
    """Construct an :class:`RLSongVecEnv`.

    Parameters
    ----------
    train_cfg : TrainConfig, optional
        Training config. A default config is used when omitted.
    n_envs, stage_idx, seed, device
        Forwarded to :class:`RLSongVecEnv`.

    Returns
    -------
    RLSongVecEnv
        Wrapped vectorized racing environment.
    """
    return RLSongVecEnv(
        train_cfg,
        n_envs=n_envs,
        stage_idx=stage_idx,
        seed=seed,
        device=device,
    )


def _to_jax_obs(env_obs: dict[str, Any]) -> dict[str, Array]:
    """Convert a squeezed env observation dict to JAX arrays."""
    return {key: jnp.asarray(value) for key, value in env_obs.items()}


def _apply_yaw_delta(quat: Array, yaw_delta: Array, mask: Array) -> Array:
    """Apply a yaw delta with SciPy's rotation implementation."""
    flat_quat = np.asarray(quat).reshape(-1, quat.shape[-1])
    flat_yaw = np.asarray(yaw_delta).reshape(-1)
    rpy = Rotation.from_quat(flat_quat).as_euler("xyz")
    rpy[:, 2] = rpy[:, 2] + flat_yaw
    perturbed = Rotation.from_euler("xyz", rpy).as_quat().reshape(quat.shape)
    return jnp.where(mask, jnp.asarray(perturbed, dtype=quat.dtype), quat)


def _validate_action_shape(
    action: Array, n_envs: int, action_dim: int, name: str
) -> None:
    """Validate a batched action shape."""
    if action.shape != (n_envs, action_dim):
        raise ValueError(
            f"{name} must have shape {(n_envs, action_dim)}; got {action.shape}"
        )


def _validate_batch_shape(batch: Array, obs_dim: int, name: str) -> None:
    """Validate a two-dimensional observation batch."""
    if batch.ndim != 2 or batch.shape[-1] != obs_dim:
        raise ValueError(
            f"{name} must have shape (n_samples, {obs_dim}); got {batch.shape}"
        )


def _default_curriculum() -> CurriculumConfig:
    """Return the default curriculum for external callers."""
    return default_curriculum()
