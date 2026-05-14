"""Train the Song-2023 JAX PPO drone-racing controller.

This module follows CleanRL's single-file PPO style, adapted to the racing
controller's raw 7-dimensional action distribution and asymmetric actor/critic
observation interface.

References
----------
Huang, S. et al. CleanRL ``ppo_continuous_action_jax.py``.
Song, Y. et al. (2023). Reaching the limit in autonomous racing.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro
import wandb
from flax.training.train_state import TrainState
from jax import Array

from lsy_drone_racing.control.rl_song.config import (
    ACTOR_OBS_DIM,
    RAW_ACTION_DIM,
    PPOConfig,
    TrainConfig,
)
from lsy_drone_racing.control.rl_song.env_wrapper import RLSongVecEnv
from lsy_drone_racing.control.rl_song.obs import NormalizerState
from lsy_drone_racing.control.rl_song.policy import (
    Actor,
    Critic,
    log_prob_of,
)
from lsy_drone_racing.control.rl_song.rollout import (
    RolloutMetricSums,
    RolloutScanOutputs,
    RolloutStaticConfig,
    scan_rollout,
)

CHECKPOINT_DIR: Path = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_PREFIX: str = "step_"
CHECKPOINT_DIGITS: int = 12
WANDB_MODE_ONLINE: str = "online"
SECONDS_PER_LOG_RATE: float = 1.0
ADVANTAGE_EPS: float = 1e-8
VALUE_LOSS_SCALE: float = 0.5


@dataclass(frozen=True)
class CLIArgs:
    """Command-line overrides for PPO training."""

    stage: int = 1
    seed: int = 0
    total_timesteps: int | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    run_name: str | None = None
    no_wandb: bool = False


class RolloutBatch(NamedTuple):
    """Flattened PPO rollout batch.

    Fields
    ------
    actor_obs : Array, shape (batch_size, ACTOR_OBS_DIM)
    critic_obs : Array, shape (batch_size, ACTOR_OBS_DIM)
    raw_actions : Array, shape (batch_size, RAW_ACTION_DIM)
    logprobs : Array, shape (batch_size,)
    advantages : Array, shape (batch_size,)
    returns : Array, shape (batch_size,)
    values : Array, shape (batch_size,)
    """

    actor_obs: Array
    critic_obs: Array
    raw_actions: Array
    logprobs: Array
    advantages: Array
    returns: Array
    values: Array


class TrainStateBundle(NamedTuple):
    """Mutable training state grouped for checkpointing."""

    actor_state: TrainState
    critic_state: TrainState
    rng_key: Array
    global_step: int
    iteration: int


def main() -> None:
    """Parse CLI arguments and run PPO training."""
    args = tyro.cli(CLIArgs)
    train(args)


def train(args: CLIArgs) -> None:
    """Train the RL Song controller.

    Parameters
    ----------
    args : CLIArgs
        CLI overrides. ``stage`` is one-indexed and maps to the zero-indexed
        curriculum stage stored in :class:`TrainConfig`.
    """
    if args.stage < 1:
        raise ValueError(f"stage must be one-indexed and positive; got {args.stage}")

    train_cfg = _build_train_config(args)
    ppo_cfg = train_cfg.ppo
    _validate_ppo_config(ppo_cfg)
    run_name = _resolve_run_name(args, train_cfg)
    run_dir = CHECKPOINT_DIR / run_name

    rng_key = jax.random.PRNGKey(train_cfg.seed)
    actor_state, critic_state = _init_train_states(ppo_cfg, rng_key)
    restored = _restore_if_available(run_dir)
    start_stage_idx = train_cfg.initial_stage_index
    start_iteration = 1
    global_step = 0
    normalizer = None
    env_rng_key = None
    env_sim_rng_key = None

    if restored is not None:
        actor_state = actor_state.replace(
            step=restored["actor_step"],
            params=restored["actor_params"],
            opt_state=restored["actor_opt_state"],
        )
        critic_state = critic_state.replace(
            step=restored["critic_step"],
            params=restored["critic_params"],
            opt_state=restored["critic_opt_state"],
        )
        rng_key = restored["rng_key"]
        global_step = int(np.asarray(restored["global_step"]))
        start_iteration = int(np.asarray(restored["iteration"])) + 1
        start_stage_idx = int(np.asarray(restored["stage_idx"]))
        normalizer = _normalizer_from_checkpoint(restored["normalizer"])
        env_rng_key = restored["env_rng_key"]
        env_sim_rng_key = restored["env_sim_rng_key"]

    env = RLSongVecEnv(
        train_cfg,
        stage_idx=start_stage_idx,
        seed=train_cfg.seed,
        device="gpu",
    )

    wandb_run = _init_wandb(args, train_cfg, run_name, restored is not None)
    if restored is None:
        next_obs, _ = env.reset(seed=train_cfg.seed + start_stage_idx)
    else:
        if normalizer is not None:
            env.set_normalizer(normalizer)
        if env_rng_key is not None:
            env.rng_key = env_rng_key
        if env_sim_rng_key is not None:
            _set_env_sim_rng_key(env, env_sim_rng_key)
        next_obs = env.build_observations()
    episode_returns = jnp.zeros((ppo_cfg.n_envs,), dtype=jnp.float32)
    episode_lengths = jnp.zeros((ppo_cfg.n_envs,), dtype=jnp.float32)
    next_done = jnp.zeros((ppo_cfg.n_envs,), dtype=jnp.float32)
    target_gate_history: deque[float] = deque(
        maxlen=train_cfg.curriculum.promotion_window_rollouts
    )
    crash_rate_history: deque[float] = deque(
        maxlen=train_cfg.curriculum.promotion_window_rollouts
    )
    start_time = time.time()
    next_checkpoint_step = _next_checkpoint_step(
        global_step, train_cfg.checkpoint_every_steps
    )

    for iteration in range(start_iteration, ppo_cfg.n_iterations + 1):
        rollout = _collect_rollout(
            env,
            ppo_cfg,
            actor_state,
            critic_state,
            rng_key,
            next_done,
            episode_returns,
            episode_lengths,
        )
        rng_key = rollout["rng_key"]
        next_obs = rollout["next_obs"]
        next_done = rollout["next_done"]
        episode_returns = rollout["episode_returns"]
        episode_lengths = rollout["episode_lengths"]
        global_step += ppo_cfg.batch_size

        next_value = _critic_value(
            critic_state.params, next_obs["critic_obs"]
        ).reshape(-1)
        advantages, returns = _compute_gae(
            rollout["rewards"],
            rollout["dones"],
            rollout["values"],
            next_done,
            next_value,
            ppo_cfg.gamma,
            ppo_cfg.gae_lambda,
        )
        batch = _flatten_rollout(rollout, advantages, returns)
        current_ent_coef = _current_ent_coef(ppo_cfg, iteration)
        actor_state, critic_state, train_metrics = _update_policy(
            actor_state,
            critic_state,
            batch,
            ppo_cfg,
            np.random.default_rng(train_cfg.seed + iteration),
            current_ent_coef,
        )
        train_metrics["ent_coef"] = current_ent_coef

        env.update_normalizer_from_batch(
            rollout["actor_obs"].reshape((-1, ACTOR_OBS_DIM))
        )
        next_obs = env.build_observations()

        rollout_metrics = rollout["metrics"]
        target_gate_history.append(rollout_metrics["target_gate_mean"])
        crash_rate_history.append(rollout_metrics["crash_rate"])
        _maybe_promote_curriculum(
            env,
            train_cfg,
            iteration,
            target_gate_history,
            crash_rate_history,
        )

        if env.stage_idx != start_stage_idx:
            start_stage_idx = env.stage_idx
            next_obs = env.build_observations()
            next_done = jnp.zeros((ppo_cfg.n_envs,), dtype=jnp.float32)
            episode_returns = jnp.zeros((ppo_cfg.n_envs,), dtype=jnp.float32)
            episode_lengths = jnp.zeros((ppo_cfg.n_envs,), dtype=jnp.float32)

        if wandb_run is not None:
            _log_iteration(
                wandb_run,
                global_step,
                iteration,
                ppo_cfg,
                actor_state,
                train_metrics,
                rollout_metrics,
                env.stage_idx,
                start_time,
            )

        if global_step >= next_checkpoint_step:
            _save_checkpoint(
                run_dir,
                TrainStateBundle(
                    actor_state,
                    critic_state,
                    rng_key,
                    global_step,
                    iteration,
                ),
                env,
                ppo_cfg.total_timesteps,
            )
            next_checkpoint_step += train_cfg.checkpoint_every_steps

        elapsed = time.time() - start_time
        steps_per_second = int(global_step / max(elapsed, SECONDS_PER_LOG_RATE))
        print(
            f"iteration={iteration}/{ppo_cfg.n_iterations} "
            f"global_step={global_step} stage={env.stage_idx + 1} "
            f"sps={steps_per_second}"
        )

    _save_checkpoint(
        run_dir,
        TrainStateBundle(
            actor_state,
            critic_state,
            rng_key,
            global_step,
            ppo_cfg.n_iterations,
        ),
        env,
        ppo_cfg.total_timesteps,
    )
    env.close()
    if wandb_run is not None:
        wandb_run.finish()


def _build_train_config(args: CLIArgs) -> TrainConfig:
    """Create the immutable training configuration from CLI overrides."""
    cfg = TrainConfig()
    ppo_cfg = cfg.ppo
    if args.total_timesteps is not None:
        ppo_cfg = replace(ppo_cfg, total_timesteps=args.total_timesteps)
    return replace(
        cfg,
        ppo=ppo_cfg,
        seed=args.seed,
        initial_stage_index=args.stage - 1,
        wandb_project=args.wandb_project or cfg.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
    )


def _validate_ppo_config(ppo_cfg: PPOConfig) -> None:
    """Validate PPO dimensions before allocating rollout buffers."""
    if ppo_cfg.batch_size % ppo_cfg.n_minibatches != 0:
        raise ValueError(
            "PPO batch size must be divisible by n_minibatches; got "
            f"{ppo_cfg.batch_size=} and {ppo_cfg.n_minibatches=}"
        )
    expected_minibatch = ppo_cfg.batch_size // ppo_cfg.n_minibatches
    if ppo_cfg.minibatch_size != expected_minibatch:
        raise ValueError(
            f"ppo.minibatch_size must be {expected_minibatch}; got "
            f"{ppo_cfg.minibatch_size}"
        )
    if ppo_cfg.n_iterations < 1:
        raise ValueError(
            "total_timesteps must be at least one PPO batch; got "
            f"{ppo_cfg.total_timesteps=} and {ppo_cfg.batch_size=}"
        )


def _resolve_run_name(args: CLIArgs, train_cfg: TrainConfig) -> str:
    """Resolve the checkpoint and wandb run name."""
    if args.run_name is not None:
        return args.run_name
    if train_cfg.run_name is not None:
        return train_cfg.run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"stage{train_cfg.initial_stage_index + 1}_seed{train_cfg.seed}_{timestamp}"


def _init_train_states(
    ppo_cfg: PPOConfig, rng_key: Array
) -> tuple[TrainState, TrainState]:
    """Initialize separate actor and critic train states."""
    actor_key, critic_key = jax.random.split(rng_key)
    dummy_obs = jnp.zeros((1, ACTOR_OBS_DIM), dtype=jnp.float32)
    actor = Actor(init_log_std=ppo_cfg.init_log_std)
    critic = Critic()
    actor_params = actor.init(actor_key, dummy_obs)["params"]
    critic_params = critic.init(critic_key, dummy_obs)["params"]

    schedule_steps = (
        ppo_cfg.n_iterations * ppo_cfg.update_epochs * ppo_cfg.n_minibatches
    )
    lr_schedule = optax.linear_schedule(
        init_value=ppo_cfg.learning_rate,
        end_value=0.0,
        transition_steps=max(schedule_steps, 1),
    )
    actor_tx = optax.chain(
        optax.clip_by_global_norm(ppo_cfg.max_grad_norm),
        optax.adam(lr_schedule),
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(ppo_cfg.max_grad_norm),
        optax.adam(lr_schedule),
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=actor_tx,
    )
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        tx=critic_tx,
    )
    return actor_state, critic_state


def _collect_rollout(
    env: RLSongVecEnv,
    ppo_cfg: PPOConfig,
    actor_state: TrainState,
    critic_state: TrainState,
    rng_key: Array,
    next_done: Array,
    episode_returns: Array,
    episode_lengths: Array,
) -> dict[str, Any]:
    """Collect one PPO rollout with the JAX-scanned race-core path."""
    if env.env is None:
        raise RuntimeError("Cannot collect a rollout before env construction.")

    static_cfg = _rollout_static_config(env, ppo_cfg)
    scan_result = scan_rollout(
        env.env.data,
        actor_state.params,
        critic_state.params,
        env.normalizer,
        env.prev_action_env_4vec,
        rng_key,
        env.rng_key,
        next_done,
        episode_returns,
        episode_lengths,
        env.env._step,
        env.env._reset,
        static_cfg,
    )

    env.env.data = scan_result.env_data
    env.prev_action_env_4vec = scan_result.prev_action_env_4vec
    env.rng_key = scan_result.reset_rng_key
    env.current_env_obs = scan_result.next_env_obs
    env.prev_env_obs = scan_result.next_env_obs
    metrics = _rollout_metrics(scan_result.outputs, scan_result.metrics)

    return {
        "actor_obs": scan_result.outputs.actor_obs,
        "critic_obs": scan_result.outputs.critic_obs,
        "raw_actions": scan_result.outputs.raw_actions,
        "logprobs": scan_result.outputs.logprobs,
        "rewards": scan_result.outputs.rewards,
        "dones": scan_result.outputs.dones,
        "values": scan_result.outputs.values,
        "next_obs": scan_result.next_obs,
        "next_done": scan_result.next_done,
        "episode_returns": scan_result.episode_returns,
        "episode_lengths": scan_result.episode_lengths,
        "rng_key": scan_result.rng_key,
        "metrics": metrics,
    }


def _rollout_static_config(
    env: RLSongVecEnv,
    ppo_cfg: PPOConfig,
) -> RolloutStaticConfig:
    """Build the static config for the compiled rollout path."""
    thrust_min, thrust_max = env.get_thrust_bounds()
    return RolloutStaticConfig(
        n_steps=ppo_cfg.n_steps,
        n_envs=ppo_cfg.n_envs,
        thrust_min=thrust_min,
        thrust_max=thrust_max,
        max_episode_steps=env.max_episode_steps,
        reward_cfg=env.reward_cfg,
        reset_pos_perturb_m=env.stage.reset_pos_perturb_m,
        reset_vel_perturb_mps=env.stage.reset_vel_perturb_mps,
        reset_yaw_perturb_rad=env.stage.reset_yaw_perturb_rad,
    )


def _rollout_metrics(
    outputs: RolloutScanOutputs,
    metric_sums: RolloutMetricSums,
) -> dict[str, float]:
    """Convert scanned rollout tensors into Python logging metrics."""
    completed_count = metric_sums.completed_count
    completed_denominator = jnp.maximum(completed_count, 1.0)
    metrics = {
        "ep_ret": float(
            np.asarray(metric_sums.completed_return_sum / completed_denominator)
        ),
        "ep_len": float(
            np.asarray(metric_sums.completed_length_sum / completed_denominator)
        ),
        "episodes": float(np.asarray(completed_count)),
        "target_gate_mean": float(np.asarray(jnp.mean(outputs.target_gate_progress))),
        "crash_rate": float(np.asarray(jnp.mean(outputs.crash.astype(jnp.float32)))),
        "finish_rate": float(
            np.asarray(jnp.mean(outputs.finished.astype(jnp.float32)))
        ),
    }
    for key, value_component in outputs.reward_components.items():
        metrics[key] = float(np.asarray(jnp.mean(value_component)))
    return metrics


@jax.jit
def _critic_value(critic_params: dict[str, Any], critic_obs: Array) -> Array:
    """Return critic value estimates."""
    return Critic().apply({"params": critic_params}, critic_obs)


@jax.jit
def _compute_gae(
    rewards: Array,
    dones: Array,
    values: Array,
    next_done: Array,
    next_value: Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[Array, Array]:
    """Compute generalized advantage estimates.

    Parameters
    ----------
    rewards : Array, shape (n_steps, n_envs)
    dones : Array, shape (n_steps, n_envs)
    values : Array, shape (n_steps, n_envs)
    next_done : Array, shape (n_envs,)
    next_value : Array, shape (n_envs,)
    gamma, gae_lambda : float
        PPO discount and GAE lambda.

    Returns
    -------
    advantages, returns : tuple[Array, Array]
        Arrays shaped ``(n_steps, n_envs)``.
    """

    def scan_step(carry: tuple[Array, Array, Array], transition: tuple[Array, ...]):
        last_gae, next_values, next_nonterminal = carry
        reward, done, value = transition
        nonterminal = 1.0 - done
        delta = reward + gamma * next_values * next_nonterminal - value
        advantage = delta + gamma * gae_lambda * next_nonterminal * last_gae
        return (advantage, value, nonterminal), advantage

    initial = (jnp.zeros_like(next_value), next_value, 1.0 - next_done)
    _, advantages_rev = jax.lax.scan(
        scan_step,
        initial,
        (rewards[::-1], dones[::-1], values[::-1]),
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


def _flatten_rollout(
    rollout: dict[str, Any], advantages: Array, returns: Array
) -> RolloutBatch:
    """Flatten rollout tensors into PPO batch tensors."""
    return RolloutBatch(
        actor_obs=rollout["actor_obs"].reshape((-1, ACTOR_OBS_DIM)),
        critic_obs=rollout["critic_obs"].reshape((-1, ACTOR_OBS_DIM)),
        raw_actions=rollout["raw_actions"].reshape((-1, RAW_ACTION_DIM)),
        logprobs=rollout["logprobs"].reshape(-1),
        advantages=advantages.reshape(-1),
        returns=returns.reshape(-1),
        values=rollout["values"].reshape(-1),
    )


def _current_ent_coef(ppo_cfg: PPOConfig, iteration: int) -> float:
    """Linearly interpolate the entropy coefficient across training.

    Parameters
    ----------
    ppo_cfg : PPOConfig
        Training hyperparameters with ``ent_coef`` initial and
        ``ent_coef_final`` end values.
    iteration : int
        One-indexed PPO iteration number.

    Returns
    -------
    float
        Entropy coefficient for this iteration.
    """
    if ppo_cfg.n_iterations <= 1:
        return ppo_cfg.ent_coef_final
    progress = (iteration - 1) / (ppo_cfg.n_iterations - 1)
    progress = max(0.0, min(1.0, progress))
    return ppo_cfg.ent_coef + (ppo_cfg.ent_coef_final - ppo_cfg.ent_coef) * progress


def _update_policy(
    actor_state: TrainState,
    critic_state: TrainState,
    batch: RolloutBatch,
    ppo_cfg: PPOConfig,
    rng: np.random.Generator,
    ent_coef: float,
) -> tuple[TrainState, TrainState, dict[str, float]]:
    """Run PPO epochs over one flattened rollout batch."""
    batch_size = batch.actor_obs.shape[0]
    batch_indices = np.arange(batch_size)
    metrics_accum: dict[str, float] = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "old_approx_kl": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "updates": 0.0,
    }
    ent_coef_jax = jnp.asarray(ent_coef, dtype=jnp.float32)

    for _ in range(ppo_cfg.update_epochs):
        rng.shuffle(batch_indices)
        for start in range(0, batch_size, ppo_cfg.minibatch_size):
            end = start + ppo_cfg.minibatch_size
            minibatch_indices = jnp.asarray(batch_indices[start:end])
            minibatch = jax.tree_util.tree_map(
                lambda value: value[minibatch_indices], batch
            )
            actor_state, critic_state, metrics = _train_minibatch(
                actor_state, critic_state, minibatch, ent_coef_jax, ppo_cfg
            )
            for key in metrics_accum:
                if key != "updates":
                    metrics_accum[key] += float(np.asarray(metrics[key]))
            metrics_accum["updates"] += 1.0

    update_count = max(metrics_accum["updates"], 1.0)
    return actor_state, critic_state, {
        key: value / update_count
        for key, value in metrics_accum.items()
        if key != "updates"
    }


@partial(jax.jit, static_argnums=4)
def _train_minibatch(
    actor_state: TrainState,
    critic_state: TrainState,
    batch: RolloutBatch,
    ent_coef: Array,
    ppo_cfg: PPOConfig,
) -> tuple[TrainState, TrainState, dict[str, Array]]:
    """Apply one PPO minibatch update to the separate actor and critic.

    ``ent_coef`` is a runtime scalar (annealed across training) so the JIT
    cache is preserved across iterations.
    """

    def actor_loss_fn(params: dict[str, Any]) -> tuple[Array, dict[str, Array]]:
        new_logprob, entropy = log_prob_of(
            params, batch.actor_obs, batch.raw_actions
        )
        logratio = new_logprob - batch.logprobs
        ratio = jnp.exp(logratio)
        advantages = (batch.advantages - jnp.mean(batch.advantages)) / (
            jnp.std(batch.advantages) + ADVANTAGE_EPS
        )
        pg_loss_unclipped = -advantages * ratio
        pg_loss_clipped = -advantages * jnp.clip(
            ratio, 1.0 - ppo_cfg.clip_coef, 1.0 + ppo_cfg.clip_coef
        )
        policy_loss = jnp.mean(jnp.maximum(pg_loss_unclipped, pg_loss_clipped))
        entropy_loss = jnp.mean(entropy)
        old_approx_kl = jnp.mean(-logratio)
        approx_kl = jnp.mean((ratio - 1.0) - logratio)
        clip_fraction = jnp.mean(
            (jnp.abs(ratio - 1.0) > ppo_cfg.clip_coef).astype(jnp.float32)
        )
        loss = policy_loss - ent_coef * entropy_loss
        return loss, {
            "policy_loss": policy_loss,
            "entropy": entropy_loss,
            "old_approx_kl": old_approx_kl,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    def critic_loss_fn(params: dict[str, Any]) -> tuple[Array, Array]:
        new_values = Critic().apply({"params": params}, batch.critic_obs).reshape(-1)
        value_loss_unclipped = jnp.square(new_values - batch.returns)
        value_clipped = batch.values + jnp.clip(
            new_values - batch.values,
            -ppo_cfg.clip_coef,
            ppo_cfg.clip_coef,
        )
        value_loss_clipped = jnp.square(value_clipped - batch.returns)
        value_loss = VALUE_LOSS_SCALE * jnp.mean(
            jnp.maximum(value_loss_unclipped, value_loss_clipped)
        )
        return ppo_cfg.vf_coef * value_loss, value_loss

    (actor_loss, actor_metrics), actor_grads = jax.value_and_grad(
        actor_loss_fn, has_aux=True
    )(actor_state.params)
    (critic_loss, value_loss), critic_grads = jax.value_and_grad(
        critic_loss_fn, has_aux=True
    )(critic_state.params)
    actor_state = actor_state.apply_gradients(grads=actor_grads)
    critic_state = critic_state.apply_gradients(grads=critic_grads)
    metrics = {
        "policy_loss": actor_metrics["policy_loss"],
        "value_loss": value_loss,
        "entropy": actor_metrics["entropy"],
        "old_approx_kl": actor_metrics["old_approx_kl"],
        "approx_kl": actor_metrics["approx_kl"],
        "clip_fraction": actor_metrics["clip_fraction"],
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
    }
    return actor_state, critic_state, metrics


def _maybe_promote_curriculum(
    env: RLSongVecEnv,
    train_cfg: TrainConfig,
    iteration: int,
    target_gate_history: deque[float],
    crash_rate_history: deque[float],
) -> None:
    """Promote the environment to the next curriculum stage when criteria pass."""
    curriculum = train_cfg.curriculum
    if iteration % curriculum.promotion_check_iterations != 0:
        return
    if env.stage_idx >= len(curriculum.stages) - 1:
        return
    if len(target_gate_history) < curriculum.promotion_window_rollouts:
        return

    stage = curriculum.stages[env.stage_idx]
    target_gate_mean = float(np.mean(target_gate_history))
    crash_rate = float(np.mean(crash_rate_history))
    if (
        target_gate_mean >= stage.promote_target_gate_mean
        and crash_rate <= stage.promote_crash_rate_max
    ):
        env.set_stage(env.stage_idx + 1)
        target_gate_history.clear()
        crash_rate_history.clear()


def _init_wandb(
    args: CLIArgs,
    train_cfg: TrainConfig,
    run_name: str,
    resume: bool,
) -> Any | None:
    """Initialize wandb unless disabled by CLI."""
    if args.no_wandb:
        return None
    return wandb.init(
        project=train_cfg.wandb_project,
        entity=train_cfg.wandb_entity,
        name=run_name,
        id=run_name,
        resume="allow" if resume else None,
        mode=WANDB_MODE_ONLINE,
        config=_config_to_log_dict(train_cfg),
    )


def _config_to_log_dict(train_cfg: TrainConfig) -> dict[str, Any]:
    """Convert nested dataclasses to a wandb-friendly mapping."""
    return asdict(train_cfg)


def _log_iteration(
    wandb_run: Any,
    global_step: int,
    iteration: int,
    ppo_cfg: PPOConfig,
    actor_state: TrainState,
    train_metrics: dict[str, float],
    rollout_metrics: dict[str, float],
    stage_idx: int,
    start_time: float,
) -> None:
    """Log PPO and rollout metrics to wandb."""
    schedule_steps = (
        ppo_cfg.n_iterations * ppo_cfg.update_epochs * ppo_cfg.n_minibatches
    )
    lr_schedule = optax.linear_schedule(
        init_value=ppo_cfg.learning_rate,
        end_value=0.0,
        transition_steps=max(schedule_steps, 1),
    )
    elapsed = time.time() - start_time
    log_data = {
        "charts/iteration": iteration,
        "charts/learning_rate": float(np.asarray(lr_schedule(actor_state.step))),
        "charts/SPS": int(global_step / max(elapsed, SECONDS_PER_LOG_RATE)),
        "curriculum/stage": stage_idx + 1,
        "losses/policy_loss": train_metrics["policy_loss"],
        "losses/value_loss": train_metrics["value_loss"],
        "losses/entropy": train_metrics["entropy"],
        "losses/old_approx_kl": train_metrics["old_approx_kl"],
        "losses/approx_kl": train_metrics["approx_kl"],
        "losses/clip_fraction": train_metrics["clip_fraction"],
        "charts/ent_coef": train_metrics.get("ent_coef", ppo_cfg.ent_coef),
        "rollout/ep_ret": rollout_metrics["ep_ret"],
        "rollout/ep_len": rollout_metrics["ep_len"],
        "rollout/episodes": rollout_metrics["episodes"],
        "rollout/target_gate": rollout_metrics["target_gate_mean"],
        "rollout/crash_rate": rollout_metrics["crash_rate"],
        "rollout/finish_rate": rollout_metrics["finish_rate"],
    }
    for key, value in rollout_metrics.items():
        if key.startswith("r_"):
            log_data[f"rollout/{key}"] = value
    wandb_run.log(log_data, step=global_step)


def _checkpoint_path(run_dir: Path, global_step: int) -> Path:
    """Return the Orbax checkpoint directory for a training step."""
    return run_dir / f"{CHECKPOINT_PREFIX}{global_step:0{CHECKPOINT_DIGITS}d}"


def _latest_checkpoint_path(run_dir: Path) -> Path | None:
    """Return the newest checkpoint directory in a run directory."""
    if not run_dir.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.iterdir():
        if not path.is_dir() or not path.name.startswith(CHECKPOINT_PREFIX):
            continue
        step_str = path.name.removeprefix(CHECKPOINT_PREFIX)
        if step_str.isdecimal():
            candidates.append((int(step_str), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _restore_if_available(run_dir: Path) -> dict[str, Any] | None:
    """Restore the latest Orbax checkpoint if one exists."""
    checkpoint_path = _latest_checkpoint_path(run_dir)
    if checkpoint_path is None:
        return None
    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(checkpoint_path)


def _save_checkpoint(
    run_dir: Path,
    train_state: TrainStateBundle,
    env: RLSongVecEnv,
    total_timesteps: int,
) -> None:
    """Save a complete PPO training checkpoint with Orbax."""
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "actor_params": train_state.actor_state.params,
        "critic_params": train_state.critic_state.params,
        "actor_opt_state": train_state.actor_state.opt_state,
        "critic_opt_state": train_state.critic_state.opt_state,
        "actor_step": train_state.actor_state.step,
        "critic_step": train_state.critic_state.step,
        "normalizer": _normalizer_to_checkpoint(env.normalizer),
        "stage_idx": jnp.asarray(env.stage_idx, dtype=jnp.int32),
        "rng_key": train_state.rng_key,
        "env_rng_key": env.rng_key,
        "env_sim_rng_key": _env_sim_rng_key(env),
        "global_step": jnp.asarray(train_state.global_step, dtype=jnp.int32),
        "iteration": jnp.asarray(train_state.iteration, dtype=jnp.int32),
        "total_timesteps": jnp.asarray(total_timesteps, dtype=jnp.int32),
    }
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(
        _checkpoint_path(run_dir, train_state.global_step),
        checkpoint,
        force=True,
    )


def _normalizer_to_checkpoint(normalizer: NormalizerState) -> dict[str, Array]:
    """Serialize a normalizer NamedTuple as a plain mapping."""
    return {
        "mean": normalizer.mean,
        "var": normalizer.var,
        "count": normalizer.count,
    }


def _normalizer_from_checkpoint(data: dict[str, Array]) -> NormalizerState:
    """Restore a normalizer from a checkpoint mapping."""
    return NormalizerState(
        mean=data["mean"],
        var=data["var"],
        count=data["count"],
    )


def _env_sim_rng_key(env: RLSongVecEnv) -> Array:
    """Return the underlying Crazyflow simulation RNG key."""
    if env.env is None:
        raise RuntimeError("Cannot checkpoint an unconstructed environment.")
    return env.env.data.sim_data.core.rng_key


def _set_env_sim_rng_key(env: RLSongVecEnv, rng_key: Array) -> None:
    """Restore the underlying Crazyflow simulation RNG key."""
    if env.env is None:
        raise RuntimeError("Cannot restore RNG into an unconstructed environment.")
    sim_data = env.env.data.sim_data
    core = sim_data.core.replace(rng_key=rng_key)
    env.env.data = env.env.data.replace(sim_data=sim_data.replace(core=core))


def _next_checkpoint_step(global_step: int, checkpoint_every_steps: int) -> int:
    """Return the next checkpoint boundary after ``global_step``."""
    if checkpoint_every_steps <= 0:
        raise ValueError("checkpoint_every_steps must be positive.")
    return (
        global_step // checkpoint_every_steps + 1
    ) * checkpoint_every_steps


if __name__ == "__main__":
    main()
