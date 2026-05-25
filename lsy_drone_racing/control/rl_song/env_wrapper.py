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
from jax.scipy.spatial.transform import Rotation as JaxRotation
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.rl_song import obs as obs_encoding
from lsy_drone_racing.control.rl_song.config import (
    ACTOR_OBS_DIM,
    ENV_ACTION_DIM,
    RAW_ACTION_DIM,
    CurriculumConfig,
    CurriculumStage,
    TrainConfig,
    default_curriculum,
)
from lsy_drone_racing.control.rl_song.policy import raw_to_env_action
from lsy_drone_racing.control.rl_song.reward import step_reward
from lsy_drone_racing.control.rl_song.rollout import _refresh_aux_fields_after_respawn
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.envs.race_core import _reset_env_data
from lsy_drone_racing.envs.race_core import obs as race_core_obs
from lsy_drone_racing.utils import load_config

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
LEVEL_CONFIG_NAME: str = "level{level}.toml"
N_DRONES: int = 1
TOTAL_THRUST_MULTIPLIER: float = 4.0


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
        self.prev_action_env_4vec = jnp.zeros((self.n_envs, ENV_ACTION_DIM), dtype=jnp.float32)
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
        self.prev_action_env_4vec = jnp.zeros((self.n_envs, ENV_ACTION_DIM), dtype=jnp.float32)
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
            Raw 4-vector ``[T_raw, tau_x, tau_y, tau_z]`` sampled by the
            policy. The log probability must have already been computed
            before this call.

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
        env_action = raw_to_env_action(
            jnp.asarray(raw_action), jnp.asarray(prev_env_obs["quat"]), thrust_min, thrust_max
        )
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
            true_gates_quat=self.true_gates_quat(),
            true_obstacles_pos=self.true_obstacles_pos(),
        )

        done = terminated | truncated
        if bool(np.asarray(jnp.any(done))):
            self._reset_done_worlds(done)
            self.current_env_obs = self._read_env_obs()

        reset_prev_action = jnp.zeros_like(env_action)
        self.prev_action_env_4vec = jnp.where(done[:, None], reset_prev_action, env_action)
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
        self.prev_action_env_4vec = jnp.zeros((self.n_envs, ENV_ACTION_DIM), dtype=jnp.float32)
        self.reset(seed=self.seed + stage_idx)

    def get_thrust_bounds(self) -> tuple[float, float]:
        """Return cached total-thrust bounds ``(min, max)`` in newtons."""
        return self._thrust_bounds

    def true_gates_pos(self) -> Array:
        """Return unmasked true gate positions, shape ``(n_envs, n_gates, 3)``."""
        if self.env is None:
            raise RuntimeError("Env is not constructed.")
        return jnp.asarray(self.env.data.gates_pos)

    def true_gates_quat(self) -> Array:
        """Return unmasked true gate orientations, shape ``(n_envs, n_gates, 4)``."""
        if self.env is None:
            raise RuntimeError("Env is not constructed.")
        return jnp.asarray(self.env.data.gates_quat)

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
        env = VecDroneRaceEnv(
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
        # v69: optional per-env sensor_range domain randomization.
        # Deploy sensor_range=0.7 is in-distribution by intent of the [min,max]
        # range chosen at CLI; sampled once per stage construction and held
        # constant across the stage's training (re-sampled on stage switch
        # via :meth:`set_stage`). Broadcast shape (n_envs, 1, 1) matches the
        # ``(n_envs, n_drones, n_gates)`` shape that ``data.sensor_range``
        # is compared against in ``race_core._reset_env_data`` and
        # ``_update_visited_objects``.
        if stage.sensor_range_random_max > stage.sensor_range_random_min:
            rng = np.random.default_rng(self.seed + self.stage_idx + 1)
            per_env_sr = rng.uniform(
                low=float(stage.sensor_range_random_min),
                high=float(stage.sensor_range_random_max),
                size=(self.n_envs, 1, 1),
            ).astype(np.float32)
            env.data = env.data.replace(sensor_range=jnp.asarray(per_env_sr))
        return env

    def _load_stage_config(self, stage: CurriculumStage) -> ConfigDict:
        """Load and patch the TOML config for a stage without touching disk."""
        config_path = REPO_ROOT / "config" / LEVEL_CONFIG_NAME.format(level=stage.level)
        race_cfg = copy.deepcopy(load_config(config_path))
        race_cfg.env.control_mode = "attitude"
        return race_cfg

    def _stage_randomizations(
        self, race_cfg: ConfigDict, stage: CurriculumStage
    ) -> ConfigDict | None:
        """Return per-stage track-side randomizations (gate_pos, gate_rpy, obstacle_pos).

        Levels 2 / 3 use the toml's ``gate_pos`` / ``gate_rpy`` / ``obstacle_pos``
        bounds scaled by ``stage.gate_rand_scale``. The framework's
        randomize fns (post-upstream PR #91) populate ``nominal_*`` from
        the placed layout and apply the wobble on top of ``gates_pos`` /
        ``gates_quat`` / ``obstacles_pos`` so the pre-visit obs reports
        the placed layout and the post-visit obs reports placement + wobble.

        Drone-pose randomizations are excluded because
        ``_apply_reset_perturbation`` applies stage-specific drone perturbations
        itself. ``gate_rpy`` matches the eval-time distribution applied by
        ``scripts/sim.py`` on level 3.
        """
        if stage.level not in (2, 3):
            return None
        scale = float(stage.gate_rand_scale)
        if scale <= 0.0:
            return None
        cfg = ConfigDict()
        for key in ("gate_pos", "gate_rpy", "obstacle_pos"):
            if key not in race_cfg.env.randomizations:
                continue
            src = race_cfg.env.randomizations[key]
            entry = ConfigDict()
            entry.fn = src.fn
            kwargs = ConfigDict()
            kwargs.minval = [float(v) * scale for v in src.kwargs.minval]
            kwargs.maxval = [float(v) * scale for v in src.kwargs.maxval]
            entry.kwargs = kwargs
            cfg[key] = entry
        return cfg if len(cfg) else None

    def _stage_disturbances(self, stage: CurriculumStage) -> ConfigDict | None:
        """Return disturbances for a curriculum stage."""
        _ = stage
        # TODO(stage4): inject DRSchedule action/dynamics disturbances here.
        return None

    def _build_critic_obs(self, env_obs: dict[str, Array]) -> Array:
        """Build batched critic obs with privileged true gate/obstacle poses."""
        return obs_encoding.vmap_build_critic_obs(
            env_obs,
            self.prev_action_env_4vec,
            self.normalizer,
            true_gates_pos=self.true_gates_pos(),
            true_gates_quat=self.true_gates_quat(),
            true_obstacles_pos=self.true_obstacles_pos(),
        )

    def _apply_reset_perturbation(self, mask: Array) -> None:
        """Apply curriculum reset perturbations to selected env worlds."""
        if self.env is None:
            raise RuntimeError("Env is not constructed.")
        mask = jnp.asarray(mask, dtype=bool)
        if not bool(np.asarray(jnp.any(mask))):
            return

        self.rng_key, pos_key, vel_key, yaw_key = jax.random.split(self.rng_key, 4)
        states = self.env.data.sim_data.states
        # v41: dropped ``start_pos`` snapshot — seg-init no longer needs the
        # spawn position as a segment-0 anchor. Entry geometry is computed
        # from the next gate's pose only.
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
            states.pos + pos_delta, self.env.data.pos_limit_low, self.env.data.pos_limit_high
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
        self._apply_segment_init(mask)

    def _apply_segment_init(self, mask: Array) -> None:
        """Re-spawn a Bernoulli-selected subset of envs at the target gate's entry waypoint.

        Implements Phase 1 of Song et al. 2023 §III-B: with probability
        ``stage.segment_init_prob``, override the just-reset drone state so
        the drone arrives at the entry side of a uniformly-random target
        gate with velocity in that gate's traversal direction, and
        ``target_gate`` advanced to match.

        v41: see ``rollout._apply_segment_init`` docstring for the diagnosis
        that motivated dropping v29's midpoint+straight-line geometry.

        Parameters
        ----------
        mask : Array, shape (n_envs,)
            Boolean mask of envs that just reset (and are eligible for
            segment-init). Envs outside this mask are untouched.
        """
        if self.env is None:
            raise RuntimeError("Env is not constructed.")
        if self.stage.segment_init_prob <= 0.0:
            return
        mask = jnp.asarray(mask, dtype=bool)
        if not bool(np.asarray(jnp.any(mask))):
            return

        self.rng_key, bern_key, seg_key, jit_key = jax.random.split(self.rng_key, 4)
        data = self.env.data
        states = data.sim_data.states
        # Pre-wobble layout. ``nominal_gates_pos`` is set by the framework's
        # ``build_full_track_randomization_fn`` to the just-placed layout
        # (upstream PR #91) and is not mutated by the per-axis gate_pos
        # randomization that follows, so segment anchors use the placed
        # geometry rather than the post-wobble physics.
        nominal_gates_pos = data.nominal_gates_pos
        nominal_gates_quat = data.nominal_gates_quat
        n_gates = nominal_gates_pos.shape[1]

        do_seg = (
            jax.random.bernoulli(bern_key, p=self.stage.segment_init_prob, shape=(self.n_envs,))
            & mask
        )
        if not bool(np.asarray(jnp.any(do_seg))):
            return

        segment_idx = jax.random.randint(seg_key, shape=(self.n_envs,), minval=0, maxval=n_gates)
        env_arange = jnp.arange(self.n_envs)
        next_gate = nominal_gates_pos[env_arange, segment_idx]
        next_gate_quat = nominal_gates_quat[env_arange, segment_idx]

        # v41: spawn at the target gate's entry waypoint on its -x_local
        # side with velocity in the gate's +x_local direction. Replaces
        # v29's midpoint(prev_gate, next_gate) + unit(next-prev) geometry
        # — which was ~90 deg off the U-turn gate's traversal axis and
        # left Phase 2 buffer slot 2 essentially empty across the v38
        # series. Mirrors ``rollout._apply_segment_init`` exactly so the
        # eager and JIT branches share the same state distribution. See
        # the full diagnosis in that function's inline comment.
        gate_xaxis_world = JaxRotation.from_quat(next_gate_quat).apply(
            jnp.array([1.0, 0.0, 0.0])
        )  # (n_envs, 3); gate's traversal axis in world
        entry_offset_m = self.reward_cfg.lookahead_entry_offset_m
        entry_waypoint = next_gate - entry_offset_m * gate_xaxis_world

        jitter = jax.random.uniform(
            jit_key,
            shape=(self.n_envs, 3),
            minval=-self.stage.segment_init_perturb_m,
            maxval=self.stage.segment_init_perturb_m,
        )
        new_pos = jnp.clip(entry_waypoint + jitter, data.pos_limit_low, data.pos_limit_high)

        seg_vel = self.stage.segment_init_vel_mps * gate_xaxis_world  # (n_envs, 3)

        mask_b3 = do_seg[:, None, None]
        new_pos_b = new_pos[:, None, :]
        new_vel_b = seg_vel[:, None, :]
        # 2026-05-25 seg-init audit: identity quat caused nose/velocity
        # mismatch (drone body +x pointed world +x while velocity pointed
        # along gate's +x_local), and ang_vel was inherited from the
        # previous terminated episode (typically a tumble from crash).
        # Mirror the fix in rollout._apply_segment_init exactly so the
        # eager and JIT-scan branches share the same state distribution.
        yaw = jnp.arctan2(gate_xaxis_world[..., 1], gate_xaxis_world[..., 0])
        half_yaw = yaw * 0.5
        seg_quat = jnp.stack(
            [jnp.zeros_like(yaw), jnp.zeros_like(yaw), jnp.sin(half_yaw), jnp.cos(half_yaw)],
            axis=-1,
        )
        seg_quat_b = seg_quat[:, None, :]
        new_ang_vel_b = jnp.zeros_like(states.ang_vel)
        new_states = states.replace(
            pos=jnp.where(mask_b3, new_pos_b, states.pos),
            vel=jnp.where(mask_b3, new_vel_b, states.vel),
            quat=jnp.where(mask_b3, seg_quat_b, states.quat),
            ang_vel=jnp.where(mask_b3, new_ang_vel_b, states.ang_vel),
        )

        new_target = jnp.where(
            do_seg[:, None], segment_idx[:, None].astype(data.target_gate.dtype), data.target_gate
        )

        sim_data = data.sim_data.replace(states=new_states)
        self.env.data = data.replace(sim_data=sim_data, target_gate=new_target)
        # Refresh ``last_drone_pos`` / ``takeoff_pos`` / ``gates_visited`` /
        # ``obstacles_visited`` so they're consistent with the just-spawned
        # state. Mirrors the in-scan call from ``rollout._apply_segment_init``.
        self.env.data = _refresh_aux_fields_after_respawn(
            self.env.data, do_seg, new_pos, segment_idx
        )

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
                f"stage_idx must be in [0, {len(self.curriculum.stages) - 1}], got {stage_idx}"
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
    return RLSongVecEnv(train_cfg, n_envs=n_envs, stage_idx=stage_idx, seed=seed, device=device)


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


def _validate_action_shape(action: Array, n_envs: int, action_dim: int, name: str) -> None:
    """Validate a batched action shape."""
    if action.shape != (n_envs, action_dim):
        raise ValueError(f"{name} must have shape {(n_envs, action_dim)}; got {action.shape}")


def _validate_batch_shape(batch: Array, obs_dim: int, name: str) -> None:
    """Validate a two-dimensional observation batch."""
    if batch.ndim != 2 or batch.shape[-1] != obs_dim:
        raise ValueError(f"{name} must have shape (n_samples, {obs_dim}); got {batch.shape}")


def _default_curriculum() -> CurriculumConfig:
    """Return the default curriculum for external callers."""
    return default_curriculum()
