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
TRACK_RANDOMIZATION_KEYS: frozenset[str] = frozenset({"gate_pos", "gate_rpy", "obstacle_pos"})
# Per-axis half-widths of the level-3 gate / obstacle position randomization,
# in meters. Mirrored from ``config/level3.toml`` so the wrapper can apply the
# perturbation itself (and update ``nominal_gates_pos`` to match the placed
# layout) without modifying upstream framework code. The framework's
# ``build_full_track_randomization_fn`` still runs (it is triggered by
# ``track.randomize=true``) and sets ``gates_pos`` to a fresh per-episode
# layout; the wrapper then snaps the nominals onto that layout and adds the
# ±max wobble below — keeping nominal ≠ actual, like level 2.
LEVEL3_GATE_POS_PERTURB_MAX: tuple[float, float, float] = (0.15, 0.15, 0.10)
LEVEL3_OBSTACLE_POS_PERTURB_MAX: tuple[float, float, float] = (0.15, 0.15, 0.05)


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
        # Wrapper-side per-env "placed" buffers (Layer-1 layout, pre-wobble).
        # Filled by ``_apply_track_perturbation`` at every reset on level-3
        # stages. Stay ``None`` for stages where there is no track-side
        # randomization (the framework's ``nominal_gates_pos`` is then
        # already correct).
        self.placed_gates_pos: Array | None = None
        self.placed_gates_quat: Array | None = None
        self.placed_obstacles_pos: Array | None = None
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
        # Snapshot the Layer-1 placement BEFORE our wobble is applied. This
        # ensures ``placed_*`` are always valid arrays even on non-level-3
        # stages (where ``_apply_track_perturbation`` is a no-op); the
        # scanned rollout path always reads these, regardless of stage.
        self.placed_gates_pos = jnp.asarray(self.env.data.gates_pos)
        self.placed_gates_quat = jnp.asarray(self.env.data.gates_quat)
        self.placed_obstacles_pos = jnp.asarray(self.env.data.obstacles_pos)
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
        actor_env_obs = self._patch_env_obs_with_placed(self.current_env_obs)
        actor_obs = obs_encoding.vmap_build_actor_obs(
            actor_env_obs, self.prev_action_env_4vec, self.normalizer
        )
        critic_obs = self._build_critic_obs(self.current_env_obs)
        return {"actor_obs": actor_obs, "critic_obs": critic_obs}

    def _patch_env_obs_with_placed(self, env_obs: dict[str, Array]) -> dict[str, Array]:
        """Replace toml-nominal pose entries with per-env Layer-1 placement.

        Mirrors ``rollout._patch_env_obs_with_placed`` for the eager path.
        See that helper for the motivation; this version short-circuits when
        the stage applies no wrapper-side wobble (level 1) and the framework's
        nominal fields are already informative.
        """
        if self.stage.level not in (2, 3):
            return env_obs
        if (
            self.placed_gates_pos is None
            or self.placed_gates_quat is None
            or self.placed_obstacles_pos is None
        ):
            return env_obs
        patched = dict(env_obs)
        gates_visited = env_obs["gates_visited"].astype(jnp.bool_)[..., None]
        patched["gates_pos"] = jnp.where(gates_visited, env_obs["gates_pos"], self.placed_gates_pos)
        patched["gates_quat"] = jnp.where(
            gates_visited, env_obs["gates_quat"], self.placed_gates_quat
        )
        obstacles_visited = env_obs["obstacles_visited"].astype(jnp.bool_)[..., None]
        patched["obstacles_pos"] = jnp.where(
            obstacles_visited, env_obs["obstacles_pos"], self.placed_obstacles_pos
        )
        return patched

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
        """Return env randomizations that belong to the course track, not DR.

        We deliberately return ``None`` even for level 3. The framework's
        ``build_full_track_randomization_fn`` (triggered by
        ``track.randomize=true`` in ``level3.toml``) still runs — it
        regenerates the per-episode gate / obstacle XY layout. But we skip the
        ``gate_pos`` / ``gate_rpy`` / ``obstacle_pos`` perturbation steps that
        the framework would otherwise apply, because those mutate
        ``gates_pos`` without touching ``nominal_gates_pos``, leaving the
        controller's pre-visit observation stuck at the toml's ``(0, 0, z)``
        placeholder. Instead, ``_apply_reset_perturbation`` (eager) and
        ``rollout._apply_track_perturbation`` (scanned) apply the same
        per-axis perturbation *after* snapping nominal to the just-placed
        layout, so the pre-visit observation is the placement and the
        post-visit observation is placement+wobble — the level-2 convention.
        """
        _ = (race_cfg, stage)  # framework randomizations are intentionally not used.
        # TODO(stage4): translate DRSchedule mass, inertia, thrust-scale, motor
        # delay, drag, sensing-noise, latency, and wind channels into Crazyflow
        # reset/step hooks when curriculum stage 4 is enabled.
        return None

    @staticmethod
    def track_perturbation_bounds(
        stage: CurriculumStage,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return per-axis gate-pos and obstacle-pos perturbation half-widths.

        Returns ``((0, 0, 0), (0, 0, 0))`` on level 1. Otherwise returns the
        level-2/3 toml bounds (identical: ``[0.15, 0.15, 0.10]`` for gates,
        ``[0.15, 0.15, 0.05]`` for obstacles) scaled by ``stage.gate_rand_scale``.
        """
        if stage.level not in (2, 3):
            zero: tuple[float, float, float] = (0.0, 0.0, 0.0)
            return zero, zero
        scale = float(stage.gate_rand_scale)
        gx, gy, gz = LEVEL3_GATE_POS_PERTURB_MAX
        ox, oy, oz = LEVEL3_OBSTACLE_POS_PERTURB_MAX
        return ((gx * scale, gy * scale, gz * scale), (ox * scale, oy * scale, oz * scale))

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
        # Snapshot the toml start position before the small jitter on top is
        # applied. ``_apply_segment_init`` uses this as the segment-0 anchor
        # (the prev_anchor for the start → gate-0 segment).
        start_pos = states.pos
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
        self._apply_track_perturbation(mask)
        self.env.data = _reset_env_data(self.env.data, mask)
        self._apply_segment_init(mask, start_pos)

    def _apply_segment_init(self, mask: Array, start_pos: Array) -> None:
        """Re-spawn a Bernoulli-selected subset of envs at random segment centers.

        Implements Phase 1 of Song et al. 2023 §III-B: with probability
        ``stage.segment_init_prob``, override the just-reset drone state so
        the drone hovers at the midpoint of a uniformly-random path segment
        with ``target_gate`` advanced to match. The Phase-2 successful-state
        buffer is not implemented yet — the segment midpoint plus jitter is
        the only initial-state distribution.

        Parameters
        ----------
        mask : Array, shape (n_envs,)
            Boolean mask of envs that just reset (and are eligible for
            segment-init). Envs outside this mask are untouched.
        start_pos : Array, shape (n_envs, n_drones, 3)
            Per-env start position snapshotted *before* any reset
            perturbation. Used as the segment-0 anchor (prev gate for the
            "start → gate 0" segment).
        """
        if self.env is None:
            raise RuntimeError("Env is not constructed.")
        if self.stage.segment_init_prob <= 0.0:
            return
        if self.placed_gates_pos is None:
            return
        mask = jnp.asarray(mask, dtype=bool)
        if not bool(np.asarray(jnp.any(mask))):
            return

        self.rng_key, bern_key, seg_key, jit_key = jax.random.split(self.rng_key, 4)
        data = self.env.data
        states = data.sim_data.states
        n_gates = data.gates_pos.shape[1]

        do_seg = (
            jax.random.bernoulli(bern_key, p=self.stage.segment_init_prob, shape=(self.n_envs,))
            & mask
        )
        if not bool(np.asarray(jnp.any(do_seg))):
            return

        segment_idx = jax.random.randint(seg_key, shape=(self.n_envs,), minval=0, maxval=n_gates)
        env_arange = jnp.arange(self.n_envs)
        prev_idx = jnp.clip(segment_idx - 1, 0, n_gates - 1)
        prev_gate = self.placed_gates_pos[env_arange, prev_idx]
        prev_anchor = jnp.where((segment_idx == 0)[:, None], start_pos[:, 0, :], prev_gate)
        next_gate = self.placed_gates_pos[env_arange, segment_idx]
        midpoint = 0.5 * (prev_anchor + next_gate)

        jitter = jax.random.uniform(
            jit_key,
            shape=(self.n_envs, 3),
            minval=-self.stage.segment_init_perturb_m,
            maxval=self.stage.segment_init_perturb_m,
        )
        new_pos = jnp.clip(midpoint + jitter, data.pos_limit_low, data.pos_limit_high)

        # v29: velocity-aware seg-init. Matches ``rollout._apply_segment_init``;
        # see ``CurriculumStage.segment_init_vel_mps`` for motivation.
        direction = next_gate - prev_anchor
        direction_norm = jnp.linalg.norm(direction, axis=-1, keepdims=True)
        unit_direction = direction / jnp.maximum(direction_norm, 1e-6)
        seg_vel = self.stage.segment_init_vel_mps * unit_direction  # (n_envs, 3)

        mask_b3 = do_seg[:, None, None]
        new_pos_b = new_pos[:, None, :]
        new_vel_b = seg_vel[:, None, :]
        identity_quat = jnp.zeros_like(states.quat).at[..., 3].set(1.0)
        new_states = states.replace(
            pos=jnp.where(mask_b3, new_pos_b, states.pos),
            vel=jnp.where(mask_b3, new_vel_b, states.vel),
            quat=jnp.where(mask_b3, identity_quat, states.quat),
        )

        new_target = jnp.where(
            do_seg[:, None], segment_idx[:, None].astype(data.target_gate.dtype), data.target_gate
        )

        sim_data = data.sim_data.replace(states=new_states)
        self.env.data = data.replace(sim_data=sim_data, target_gate=new_target)

    def _apply_track_perturbation(self, mask: Array) -> None:
        """Snap per-env placed buffer to current layout, then add ±max wobble.

        Runs only on level-3 stages. Mirrors the JAX-traceable update in
        ``rollout._apply_reset_perturbation`` used inside the scanned rollout
        path; this eager version is invoked once per :meth:`reset` at the
        start of training (and on stage promotion). Updates two pieces of
        state for envs selected by ``mask``:

        * ``self.placed_gates_pos / quat`` and ``self.placed_obstacles_pos``:
          snapshot of the just-placed layout *before* wobble is added. These
          replace the framework's ``nominal_*`` for the actor observation
          (the framework keeps the toml's ``(0, 0, z)`` placeholder there).
        * ``env_data.gates_pos`` and ``env_data.obstacles_pos`` += wobble.

        Wobble half-widths come from :meth:`track_perturbation_bounds`,
        derived from ``level3.toml`` and scaled by ``stage.gate_rand_scale``.
        """
        if self.env is None:
            raise RuntimeError("Env is not constructed.")
        if self.stage.level not in (2, 3):
            return
        mask = jnp.asarray(mask, dtype=bool)
        if not bool(np.asarray(jnp.any(mask))):
            return
        gate_pos_max, obstacle_pos_max = self.track_perturbation_bounds(self.stage)
        gate_pos_max_arr = jnp.asarray(gate_pos_max, dtype=jnp.float32)
        obstacle_pos_max_arr = jnp.asarray(obstacle_pos_max, dtype=jnp.float32)

        self.rng_key, gate_key, obs_key = jax.random.split(self.rng_key, 3)
        data = self.env.data
        gate_delta = jax.random.uniform(
            gate_key, shape=data.gates_pos.shape, minval=-gate_pos_max_arr, maxval=gate_pos_max_arr
        )
        obs_delta = jax.random.uniform(
            obs_key,
            shape=data.obstacles_pos.shape,
            minval=-obstacle_pos_max_arr,
            maxval=obstacle_pos_max_arr,
        )
        mask_b = mask[:, None, None]
        # Snapshot the placed (pre-wobble) layout into the wrapper-side buffer.
        if self.placed_gates_pos is None:
            self.placed_gates_pos = data.gates_pos
            self.placed_gates_quat = data.gates_quat
            self.placed_obstacles_pos = data.obstacles_pos
        else:
            self.placed_gates_pos = jnp.where(mask_b, data.gates_pos, self.placed_gates_pos)
            self.placed_gates_quat = jnp.where(mask_b, data.gates_quat, self.placed_gates_quat)
            self.placed_obstacles_pos = jnp.where(
                mask_b, data.obstacles_pos, self.placed_obstacles_pos
            )
        # Apply Layer-2 wobble to gates_pos and obstacles_pos.
        new_gates_pos = jnp.where(mask_b, data.gates_pos + gate_delta, data.gates_pos)
        new_obstacles_pos = jnp.where(mask_b, data.obstacles_pos + obs_delta, data.obstacles_pos)
        self.env.data = data.replace(gates_pos=new_gates_pos, obstacles_pos=new_obstacles_pos)

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
