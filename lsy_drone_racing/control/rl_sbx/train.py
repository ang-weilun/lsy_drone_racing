"""SBX PPO training entry for the rl_sbx stack.

Usage:
-----
``pixi run -e rl-train python -m lsy_drone_racing.control.rl_sbx.train
--run-name v110_sbx_baseline_cold_300M --total-timesteps 300000000``.

Construction order:
------------------
1. Build a :class:`RLSongVecEnv` wrapper using the project's existing toml /
   curriculum machinery — this gives us the inner JAX env AND a working
   :meth:`RLSongVecEnv._apply_reset_perturbation` hook for L2 seg-init.
2. Construct :class:`RLSBXVecEnv` around ``wrapper.env`` with
   ``reset_done_hook=wrapper._apply_reset_perturbation``. This restores the
   seg-init invariant that the flat-concat vec-env autoreset path would
   otherwise have lost (cf. :class:`RLSBXVecEnv` ``reset_done_hook`` doc).
3. Build :class:`sbx.PPO` with our :class:`AsymmetricActorCriticPolicy` and
   the :class:`NormalizerUpdateCallback`.
4. Save the actor + critic + both normalizers to a step-keyed dir under
   ``checkpoint_root`` via :func:`save_step`.

Defaults match the v77 cold-train recipe — see
``docs/specs/2026-05-24-sbx-migration-design.md``.

References:
----------
Stable Baselines Jax (SBX), https://github.com/araffin/sbx, ``sbx/ppo``.
Song, Y. et al. (2023). Reaching the limit in autonomous racing.
*Science Robotics* 8, eadg1462.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import fire
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from lsy_drone_racing.control.rl_sbx.callbacks import (
    EntropyAnnealCallback,
    NormalizerUpdateCallback,
    PeriodicCheckpointCallback,
)
from lsy_drone_racing.control.rl_sbx.checkpoint import save_step
from lsy_drone_racing.control.rl_sbx.env_gym import RLSBXVecEnv
from lsy_drone_racing.control.rl_sbx.jit_scan_ppo import JitScanPPO
from lsy_drone_racing.control.rl_sbx.policy import AsymmetricActorCriticPolicy
from lsy_drone_racing.control.rl_song.config import (
    TANGENT_ALPHA_MAX_RAD,
    RewardConfig,
    TrainConfig,
    _full_curriculum,
)
from lsy_drone_racing.control.rl_song.env_wrapper import RLSongVecEnv

# Curriculum selector for the ``--curriculum`` CLI flag. ``"default"`` keeps
# TrainConfig's baked-in ``default_curriculum`` (single-stage L2 + seg-init +
# phase-2). ``"full"`` swaps in :func:`_full_curriculum`, the v9/v10 seven-
# stage curriculum that exposes L3-relevant stages (stage3a/b/c, stage4_dr)
# via ``--stage-idx``.
_CURRICULUM_FACTORIES = {"default": None, "full": _full_curriculum}


class WandbScalarCallback(BaseCallback):
    """Forward SB3 logger scalars to wandb at each rollout end.

    ``WandbCallback`` from ``wandb.integration.sb3`` only forwards system
    + (PyTorch) gradient metrics. SBX's training loop accumulates
    rollout/train scalars into ``self.model.logger.name_to_value`` and
    flushes them via ``logger.dump(step=...)``. We snapshot that dict on
    each ``_on_rollout_end`` (fires after the SBX collector and right
    before the train step) and push it straight to ``wandb.log``.

    Notes:
    -----
    Replaces the ``sync_tensorboard=True`` + ``tensorboard_log=...`` path
    in the JIT-scan stack — the tensorboard event-file watcher adds disk
    I/O proportional to the metric count per iteration and the
    file-system polling cost is non-trivial at 250k+ env-steps/s.
    Direct ``wandb.log`` calls bypass that entirely.
    """

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Snapshot the SB3 logger's scalar dict and forward to wandb."""
        # ``name_to_value`` carries the keys SB3 / SBX have written since
        # the last ``dump`` (rollout/* from the previous iteration, train/*
        # from the current iteration if any). It's an ``OrderedDict``;
        # ``dict(...)`` snapshots it so wandb sees a consistent view.
        scalars = dict(self.model.logger.name_to_value)
        if not scalars:
            return
        wandb.log(scalars, step=int(self.model.num_timesteps))


# Share the rl_song wandb project so rl_sbx runs land alongside the v33-v100
# line for direct comparison plots. ``TrainConfig.wandb_project`` is the
# canonical source — pulled into a module constant here for clarity.
WANDB_PROJECT: str = TrainConfig().wandb_project

# v77 cold-train recipe defaults; see design doc §M1.
DEFAULT_TOTAL_TIMESTEPS: int = 155_000_000
DEFAULT_ALPHA_MAX_RAD: float = TANGENT_ALPHA_MAX_RAD  # 0.16 rad, Schuck 2025 best
DEFAULT_ENT_COEF: float = 0.005  # cold-train start; constant for milestone 1
DEFAULT_LEARNING_RATE: float = 3e-4

# PPO rollout shape: with n_envs=16384 from PPOConfig the buffer holds
# n_envs * n_steps = 16384 * 256 = 4_194_304 transitions per iteration.
# minibatch_size is batch_size and divides the buffer into n_minibatches.
DEFAULT_N_STEPS: int = 256
DEFAULT_N_EPOCHS: int = 3
DEFAULT_BATCH_SIZE: int = 16384


def train(
    run_name: str,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    alpha_max_rad: float = DEFAULT_ALPHA_MAX_RAD,
    ent_coef: float = DEFAULT_ENT_COEF,
    ent_coef_final: float | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    gamma: float | None = None,
    segment_init_prob: float | None = None,
    segment_init_vel_mps: float | None = None,
    phase2_prob: float | None = None,
    phase2_warmup_steps: int | None = None,
    progress_coef: float = 15.0,
    time_penalty: float = 0.10,
    guide_coef: float = 0.5,
    gate_pass_bonus: float = 10.0,
    gate_frame_weight: float = 0.0,
    obstacle_weight: float = 0.0,
    use_obstacle_barrier: bool = False,
    use_velocity_progress: bool = False,
    lookahead_coef: float = 0.0,
    lookahead_mask_through: int = 0,
    lookahead_near_plane_m: float = 0.5,
    wrong_side_coef: float = 0.0,
    wrong_side_target_min: int = 1,
    dipole_coef: float = 0.0,
    dipole_sigma: float = 0.5,
    use_path_progress: bool = False,
    path_exit_offset_m: float = 0.4,
    path_entry_offset_m: float = 0.4,
    path_progress_ks: float = 0.0,
    zero_progress_on_pass: bool = False,
    use_gate_frame_barrier: bool = False,
    omega_coef: float = 0.01,
    r_smooth_coef: float = 0.0,
    ortho_init: bool = True,
    log_std_init: float = -0.5,
    n_envs: int | None = None,
    n_steps: int = DEFAULT_N_STEPS,
    n_epochs: int = DEFAULT_N_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 0,
    checkpoint_root: str = "lsy_drone_racing/control/rl_sbx/checkpoints",
    save_freq_steps: int = 20_000_000,
    init_from: str | None = None,
    init_actor_only: bool = False,
    curriculum: str = "default",
    stage_idx: int = 0,
    wandb_project: str = WANDB_PROJECT,
    wandb_entity: str | None = None,
    no_wandb: bool = False,
    profile_throughput: bool = False,
    diag_every_n_rollouts: int = 1,
) -> None:
    """Run SBX PPO cold-train against the milestone-1 L2 seg-init curriculum.

    Parameters
    ----------
    run_name : str
        Sub-directory name under ``checkpoint_root`` for this run's
        step-keyed checkpoints.
    total_timesteps : int, optional
        Total environment steps to train for. Defaults to the milestone-1
        budget (155M).
    alpha_max_rad : float, optional
        Per-step rotation budget on ``‖τ_scaled‖`` (rad) — controls the
        raw-to-env action projection cone half-angle. Defaults to the
        rl_song module-level constant.
    ent_coef : float, optional
        Constant PPO entropy bonus for the cold-train recipe.
    learning_rate : float, optional
        Adam learning rate; passed straight to ``sbx.PPO``.
    r_smooth_coef : float, optional
        Coefficient for the action-smoothness reward term.
    n_envs : int, optional
        Vectorization width. Defaults to ``TrainConfig.ppo.n_envs``.
    n_steps : int, optional
        PPO rollout length per env per iteration.
    n_epochs : int, optional
        PPO update epochs per rollout.
    batch_size : int, optional
        Minibatch size for PPO updates.
    seed : int, optional
        JAX env + PPO seed.
    checkpoint_root : str, optional
        Run directories are created at ``<checkpoint_root>/<run_name>``.
    wandb_project : str, optional
        Wandb project name. Defaults to the same project the rl_song line
        publishes to so SBX runs land alongside v33-v100 for comparison.
    wandb_entity : str, optional
        Wandb entity (team / user). ``None`` uses the local wandb default.
    no_wandb : bool, optional
        Skip wandb init entirely (stdout-only diagnostics). Default
        ``False`` — milestone-1 needs the wandb plots for the comparison
        write-up.
    profile_throughput : bool, optional
        Log per-rollout ``time/prof_scan_s`` / ``prof_host_s`` /
        ``prof_update_plus_log_s`` (adds one ``block_until_ready`` sync).
        Profiling only; default ``False``.
    diag_every_n_rollouts : int, optional
        Emit the heavy Phase-2 buffer diagnostics every Nth rollout (default
        ``1`` = every rollout). Higher values cut the per-rollout host
        sync-storm on seg-init/phase-2 runs.

    Notes:
    -----
    The construction order (RLSongVecEnv → RLSBXVecEnv with seg-init hook
    → SBX PPO) is load-bearing — see the module docstring.
    """
    train_cfg = TrainConfig()
    if curriculum not in _CURRICULUM_FACTORIES:
        raise ValueError(
            f"Unknown curriculum {curriculum!r}; expected one of {sorted(_CURRICULUM_FACTORIES)}."
        )
    factory = _CURRICULUM_FACTORIES[curriculum]
    if factory is not None:
        train_cfg = replace(train_cfg, curriculum=factory())
    if gamma is not None:
        train_cfg = replace(train_cfg, ppo=replace(train_cfg.ppo, gamma=float(gamma)))
    n_stages = len(train_cfg.curriculum.stages)
    if not 0 <= stage_idx < n_stages:
        raise ValueError(
            f"stage_idx={stage_idx} out of range for curriculum "
            f"{curriculum!r} (n_stages={n_stages})."
        )
    effective_n_envs = train_cfg.ppo.n_envs if n_envs is None else int(n_envs)
    # 2026-05-25: v112 (`reward_cfg = RewardConfig()`) cold-trained without
    # seg-init and converged to the "barely-not-hover thrust, drone falls
    # slowly to floor" local optimum (0/100 across L0/L1/L2/L3). Discounted-
    # return analysis (gamma=0.998, max_episode=500) showed the v112 reward
    # had only a +0.6 margin between hover-then-truncate (-15.81) and
    # crash@10 (-15.20) — break-even is at time_penalty ≈ 0.0506 and the
    # default 0.05 sat 0.6 % under that. PPO had no usable gradient between
    # "stay alive doing nothing" and "try to move and crash". See
    # docs/specs/2026-05-25-reward-and-seginit-fix-design.md for the
    # archetype-by-archetype scoring table and Codex's bug audit.
    #
    # The v112 comment "three-term Song reward" was also wrong — the
    # default RewardConfig has 7+ active terms (r_prog, r_omega,
    # r_terminal, r_guid, r_gate_bonus scaled by index, r_time). This
    # construction makes the experiment label honest.
    # Defaults are the v113 recipe (time_penalty=0.10, progress_coef=15);
    # CLI flags allow overriding any of the four reward levers for variant
    # sweeps without editing this file.
    reward_cfg = RewardConfig(
        time_penalty=time_penalty,
        progress_coef=progress_coef,
        guide_coef=guide_coef,
        gate_pass_bonus=gate_pass_bonus,
        gate_frame_weight=gate_frame_weight,
        obstacle_weight=obstacle_weight,
        use_obstacle_barrier=use_obstacle_barrier,
        use_velocity_progress=use_velocity_progress,
        lookahead_coef=lookahead_coef,
        lookahead_mask_through=lookahead_mask_through,
        lookahead_near_plane_m=lookahead_near_plane_m,
        wrong_side_coef=wrong_side_coef,
        wrong_side_target_min=wrong_side_target_min,
        dipole_coef=dipole_coef,
        dipole_sigma=dipole_sigma,
        use_path_progress=use_path_progress,
        path_exit_offset_m=path_exit_offset_m,
        path_entry_offset_m=path_entry_offset_m,
        path_progress_ks=path_progress_ks,
        zero_progress_on_pass=zero_progress_on_pass,
        use_gate_frame_barrier=use_gate_frame_barrier,
        omega_coef=omega_coef,
        r_smooth_coef=r_smooth_coef,
    )

    # The wrapper's __init__ instantiates the inner JAX env via set_stage and
    # runs its own reset(seed=seed+stage_idx) — no second reset call needed.
    wrapper = RLSongVecEnv(
        train_cfg, n_envs=effective_n_envs, stage_idx=stage_idx, seed=seed, device="gpu"
    )

    # Inherit thrust bounds from the wrapper, which already loaded them via
    # ``load_params(level_toml.sim.physics, level_toml.sim.drone_model)``
    # ("first_principles" / "cf21B_500" for the level2 toml) and scaled by
    # ``TOTAL_THRUST_MULTIPLIER``. Avoids hardcoding a physics/drone-model
    # pair here that could drift out of sync with the actual training stage.
    thrust_min, thrust_max = wrapper.get_thrust_bounds()

    # Seg-init / perturbation knobs from the active curriculum stage.
    # ``JitScanPPO.collect_rollouts`` forwards these into the compiled
    # ``scan_rollout`` so the Phase-1 mid-track re-spawn and the drone-
    # state perturbation fire inside the JAX scan. The ``reset_done_hook``
    # below is the legacy step_wait path; JIT-scan bypasses ``step_wait``
    # entirely so the in-scan path is the one that matters here.
    stage = wrapper.stage
    seg_init_kwargs: dict[str, float | int] = {
        "reset_pos_perturb_m": float(stage.reset_pos_perturb_m),
        "reset_vel_perturb_mps": float(stage.reset_vel_perturb_mps),
        "reset_yaw_perturb_rad": float(stage.reset_yaw_perturb_rad),
        "segment_init_prob": float(stage.segment_init_prob),
        "segment_init_perturb_m": float(stage.segment_init_perturb_m),
        "segment_init_vel_mps": float(stage.segment_init_vel_mps),
        "phase2_prob": float(stage.phase2_prob),
        "phase2_warmup_steps": int(stage.phase2_warmup_steps),
        "phase2_capacity_per_gate": int(stage.phase2_capacity_per_gate),
    }
    if segment_init_prob is not None:
        seg_init_kwargs["segment_init_prob"] = float(segment_init_prob)
    if segment_init_vel_mps is not None:
        seg_init_kwargs["segment_init_vel_mps"] = float(segment_init_vel_mps)
    if phase2_prob is not None:
        seg_init_kwargs["phase2_prob"] = float(phase2_prob)
    if phase2_warmup_steps is not None:
        seg_init_kwargs["phase2_warmup_steps"] = int(phase2_warmup_steps)

    wandb_run = None
    if not no_wandb:
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            id=run_name,
            resume="allow",
            # SB3 scalar metrics reach wandb via ``WandbScalarCallback``
            # below — it forwards ``model.logger.name_to_value`` straight
            # to ``wandb.log`` at each ``_on_rollout_end``. The tensorboard
            # sync path is gone (event-file watcher I/O isn't free at
            # 250k+ env-steps/s, and our JIT-scan collector skips the
            # per-step callback dispatch that SBX's WandbCallback hooks
            # into for gradient metrics).
            config={
                "stack": "rl_sbx",
                "total_timesteps": total_timesteps,
                "alpha_max_rad": alpha_max_rad,
                "ent_coef": ent_coef,
                "learning_rate": learning_rate,
                "n_envs": n_envs,
                "n_steps": n_steps,
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "seed": seed,
                "stage_name": stage.name,
                **seg_init_kwargs,
            },
        )

    env = RLSBXVecEnv(
        jax_env=wrapper.env,
        reward_cfg=reward_cfg,
        alpha_max=alpha_max_rad,
        thrust_min=thrust_min,
        thrust_max=thrust_max,
        n_envs=effective_n_envs,
        seed=seed,
        reset_done_hook=wrapper._apply_reset_perturbation,
        seg_init_kwargs=seg_init_kwargs,
        phase2_capacity_per_gate=stage.phase2_capacity_per_gate,
    )

    # SBX 0.26 PPO kwargs: confirmed via inspect.signature on remote. ``device``
    # left at "auto" (JAX picks the default backend, which is the GPU the env
    # is already on).
    #
    # ``target_kl=None`` is load-bearing. SBX's ``KLAdaptiveLR`` fires
    # ``adaptive_lr.update(kl)`` PER MINIBATCH (sbx/ppo/ppo.py:332) with a
    # multiplicative factor of 1.5 per call. With our buffer geometry
    # (n_envs=16384 × n_steps=256 = 4_194_304 samples, batch_size=16384,
    # n_epochs=3) that's 256 minibatches × 3 epochs = 768 updates per
    # iteration; even a single iteration of mostly-low-KL minibatches drives
    # the adaptive LR up to ``max_learning_rate=1e-2`` (the v110 run sat at
    # LR=0.01 throughout). A 0.01 LR on a 256x256 MLP melts the policy —
    # the v110 actor's mu saturated against the ``tanh`` head and never
    # recovered. ``rl_song.PPOConfig.target_kl=0.02`` was an EARLY-STOPPING
    # threshold in the hand-rolled PPO loop, not an LR adaptation knob;
    # SBX's mechanism with the same number is the opposite semantic. Disable
    # the adaptive LR; PPO's ``clip_range`` already controls per-update
    # policy change.
    # 2026-05-25: ortho_init=True by default to match rl_song.policy.Actor's
    # output head init (orthogonal scale 0.01) and hidden layers (orthogonal
    # scale sqrt(2)). The Flax default (lecun_normal) produces output kernels
    # ~6x larger, so the initial Gaussian policy has confidently random mu
    # values that PPO struggles to recover from -- evidence: v112/v113/
    # v113b/v113d/v113e all 0/10 across L0/L1/L2 with ortho_init=False.
    # Settable to False to preserve the original v112 architecture for
    # reproducibility comparisons.
    policy_kwargs = {"ortho_init": bool(ortho_init), "log_std_init": float(log_std_init)}
    model = JitScanPPO(
        policy=AsymmetricActorCriticPolicy,
        env=env,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=train_cfg.ppo.gamma,
        gae_lambda=train_cfg.ppo.gae_lambda,
        clip_range=train_cfg.ppo.clip_coef,
        # Pessimistic value-loss clip range. Same scalar as the policy ratio
        # clip per rl_song convention (rl_song/train.py:925, 945 both use
        # ``ppo_cfg.clip_coef``). Threaded into JitScanPPO's overridden
        # ``train`` -> ``_one_update_clipped_vf`` (see jit_scan_ppo.py).
        # Stock SBX accepts this kwarg but never used it; we wire it through.
        clip_range_vf=train_cfg.ppo.clip_coef,
        ent_coef=ent_coef,
        vf_coef=train_cfg.ppo.vf_coef,
        max_grad_norm=train_cfg.ppo.max_grad_norm,
        target_kl=None,
        seed=seed,
        verbose=1,
        # No ``tensorboard_log``: scalars go via ``WandbScalarCallback``.
    )
    # Profiling-only: per-rollout scan/host/update timing into the SB3 logger.
    model.profile_throughput = profile_throughput
    model.diag_every_n_rollouts = diag_every_n_rollouts

    # 2026-05-25: optional warm-start from an existing checkpoint. The
    # checkpoint format mirrors save_step (5 files: actor.params.msgpack,
    # critic.params.msgpack, {actor,critic}_normalizer.json,
    # policy_config.json). Loads actor + critic params into the freshly
    # constructed model.policy TrainStates, and the two normalizers into
    # the env wrapper. Optimizer state stays freshly-initialized -- SBX
    # has no opt-state checkpoint to restore.
    if init_from is not None:
        from lsy_drone_racing.control.rl_sbx.checkpoint import load_all

        init_path = Path(init_from)
        # If init_from is a run dir, pick the highest step.
        if not (init_path / "actor.params.msgpack").exists():
            step_dirs = sorted(init_path.glob("step_*"))
            if not step_dirs:
                raise FileNotFoundError(f"No step_* dirs under {init_path}")
            init_path = step_dirs[-1]
        print(f"warm-start from {init_path}", flush=True)
        actor_template = model.policy.actor_state.params
        critic_template = model.policy.vf_state.params
        loaded = load_all(init_path, actor_template, critic_template)
        # Replace the TrainState params (preserves optimizer state with the
        # new shapes intact). Apply_fn / opt_state remain from model setup.
        model.policy.actor_state = model.policy.actor_state.replace(params=loaded["actor_params"])
        env.set_actor_normalizer(loaded["actor_normalizer"])
        # The critic normalizer is observation-distribution state (Welford stats
        # over the critic obs; see checkpoint.py), NOT reward geometry, so always
        # load it — a cold normalizer would feed the fresh critic unnormalized
        # inputs (Codex review #2).
        env.set_critic_normalizer(loaded["critic_normalizer"])
        if init_actor_only:
            # Reward geometry changed (guiding-path progress): only the critic
            # PARAMS encode value estimates for the old reward, so leave them
            # freshly initialized and let the critic relearn. The actor and the
            # critic-obs normalizer (loaded above) are kept. See
            # docs/superpowers/reviews/2026-05-29-guiding-path-plan-codex-review.md.
            print("actor-only warm-start: critic params reset, normalizers kept", flush=True)
        else:
            model.policy.vf_state = model.policy.vf_state.replace(params=loaded["critic_params"])

    run_dir = Path(checkpoint_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    callbacks: list = [
        NormalizerUpdateCallback(),
        PeriodicCheckpointCallback(
            run_dir=run_dir, alpha_max_rad=alpha_max_rad, save_freq_steps=save_freq_steps, verbose=1
        ),
    ]
    if ent_coef_final is not None:
        # v77 cold-train recipe annealed ent 0.005 -> 0.001 over training.
        # SBX's PPO takes ent_coef as a float at construction and never
        # schedules it; the callback mutates self.model.ent_coef at each
        # _on_rollout_end so the next iteration's update closes over the
        # new value.
        callbacks.append(
            EntropyAnnealCallback(
                ent_coef_start=ent_coef,
                ent_coef_final=ent_coef_final,
                total_timesteps=total_timesteps,
                verbose=0,
            )
        )
    if wandb_run is not None:
        callbacks.append(WandbScalarCallback())
    model.learn(total_timesteps=total_timesteps, callback=callbacks, log_interval=1)

    step_dir = save_step(
        run_dir=run_dir,
        global_step=int(model.num_timesteps),
        actor_params=model.policy.actor_state.params,
        critic_params=model.policy.vf_state.params,
        actor_normalizer=env.actor_normalizer,
        critic_normalizer=env.critic_normalizer,
        tangent_alpha_max_rad=alpha_max_rad,
    )
    print(f"Saved final checkpoint to {step_dir}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    fire.Fire(train)
