"""Song-2023 dense reward for the RL drone-racing controller.

The function in this module replaces the sparse reward emitted by
``VecDroneRaceEnv.step``. It consumes the post-step observation, the previous
observation, and terminal flags after ``VecDroneRaceEnv`` has squeezed the
single-drone axis, so every array has a leading ``n_envs`` dimension.

Notes
-----
``env_obs["gates_pos"]`` is intentionally insufficient for the progress term:
the racing env reports nominal gate poses until a gate is revealed by the
sensor-range mask. The wrapper must therefore pass unmasked true gate positions
through ``true_gates_pos``. If this argument is omitted, the function falls back
to ``env_obs["gates_pos"]`` for callers that intentionally use masked poses.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from lsy_drone_racing.control.rl_song.config import RewardConfig


def step_reward(
    env_obs: dict[str, Array],
    prev_env_obs: dict[str, Array],
    terminated: Array,
    truncated: Array,
    finished: Array,
    gate_just_passed: Array,
    reward_cfg: RewardConfig,
    *,
    true_gates_pos: Array | None = None,
    true_obstacles_pos: Array | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Compute one vectorized Song-style reward step.

    Parameters
    ----------
    env_obs : dict[str, Array]
        Current observation after ``VecDroneRaceEnv.step``. Values have a
        leading ``n_envs`` axis; ``pos`` is shape ``(n_envs, 3)``.
    prev_env_obs : dict[str, Array]
        Previous-step observation with the same structure as ``env_obs``.
    terminated : Array, shape (n_envs,)
        Environment termination flags after the step.
    truncated : Array, shape (n_envs,)
        Timeout flags after the step. Timeouts receive no terminal bonus or
        crash penalty.
    finished : Array, shape (n_envs,)
        Boolean mask indicating that the race finished on this transition.
    gate_just_passed : Array, shape (n_envs,)
        Boolean mask indicating that ``target_gate`` advanced on this
        transition.
    reward_cfg : RewardConfig
        Reward weights. In particular ``omega_coef`` is the 50 Hz coefficient.
    true_gates_pos : Array, shape (n_envs, n_gates, 3), optional
        Unmasked true gate positions. The wrapper should pass this from
        ``env.data.gates_pos``.
    true_obstacles_pos : Array, shape (n_envs, n_obstacles, 3), optional
        Unmasked obstacle positions. If omitted, ``env_obs["obstacles_pos"]``
        is used.

    Returns
    -------
    reward : Array, shape (n_envs,)
        Total replacement reward.
    components : dict[str, Array]
        Per-component rewards with keys ``r_prog``, ``r_omega``, ``r_obs``,
        ``r_gate_bonus``, and ``r_terminal``. Every value has shape
        ``(n_envs,)``.
    """
    _ = truncated
    gates_pos = env_obs["gates_pos"] if true_gates_pos is None else true_gates_pos
    obstacles_pos = env_obs["obstacles_pos"] if true_obstacles_pos is None else true_obstacles_pos

    prev_target = prev_env_obs["target_gate"]
    current_target = env_obs["target_gate"]
    target_idx = jnp.where(prev_target >= 0, prev_target, jnp.maximum(current_target, 0))
    env_idx = jnp.arange(gates_pos.shape[0])
    gate_pos = gates_pos[env_idx, target_idx]

    prev_pos = prev_env_obs["pos"]
    pos = env_obs["pos"]
    r_prog = jnp.linalg.norm(gate_pos - prev_pos, axis=-1)
    r_prog = reward_cfg.progress_coef * (r_prog - jnp.linalg.norm(gate_pos - pos, axis=-1))

    r_omega = -reward_cfg.omega_coef * jnp.linalg.norm(env_obs["ang_vel"], ord=1, axis=-1)

    obstacle_delta = pos[:, None, :] - obstacles_pos
    obstacle_dist_sq = jnp.sum(jnp.square(obstacle_delta), axis=-1)
    obstacle_active = 1.0 - env_obs["obstacles_visited"].astype(jnp.float32)
    obstacle_barrier = jnp.exp(-obstacle_dist_sq / jnp.square(reward_cfg.obstacle_sigma))
    r_obs = -reward_cfg.obstacle_weight * jnp.sum(obstacle_barrier * obstacle_active, axis=-1)

    # Per-gate jackpot scaling (v7). At a crossing, ``target_idx`` is still the
    # pre-step target index because of the ``prev_target >= 0`` branch above,
    # which is the index of the gate just passed. Scaling by ``(target_idx +
    # 1)`` makes gate 1 worth 1x, gate 2 worth 2x, ..., gate 4 worth 4x of the
    # base ``gate_pass_bonus``. The intent is to pull the policy through the
    # harder late-gate transitions where v5/v6 plateaued.
    gate_bonus_weight = jnp.asarray(reward_cfg.gate_pass_bonus, dtype=pos.dtype)
    gate_bonus_scale = jnp.where(
        jnp.asarray(reward_cfg.scale_gate_bonus_by_index, dtype=bool),
        target_idx.astype(pos.dtype) + 1.0,
        jnp.ones_like(r_prog),
    )
    gate_bonus_enabled = jnp.asarray(reward_cfg.use_gate_pass_bonus, dtype=bool)
    r_gate_bonus = jnp.where(
        gate_bonus_enabled & gate_just_passed,
        gate_bonus_weight * gate_bonus_scale,
        jnp.zeros_like(r_prog),
    )

    r_finish = jnp.where(finished, reward_cfg.finish_bonus, jnp.zeros_like(r_prog))
    r_crash = jnp.where(terminated & ~finished, -reward_cfg.crash_penalty, jnp.zeros_like(r_prog))
    r_terminal = r_finish + r_crash

    # v8: per-step time penalty to break the hover Q=0 attractor. Without it,
    # a random-init policy on a randomized track collects 0 progress reward
    # on average and times out at the episode cap, leaving "do nothing" as
    # the best policy under the entropy bonus. See ``RewardConfig.time_penalty``.
    r_time = jnp.full_like(r_prog, -reward_cfg.time_penalty)

    components = {
        "r_prog": r_prog,
        "r_omega": r_omega,
        "r_obs": r_obs,
        "r_gate_bonus": r_gate_bonus,
        "r_terminal": r_terminal,
        "r_time": r_time,
    }
    reward = r_prog + r_omega + r_obs + r_gate_bonus + r_terminal + r_time
    return reward, components
