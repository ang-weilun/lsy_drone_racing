"""Static configuration for the Song-2023 RL prototype.

Type-only module: dataclasses holding PPO hyperparameters, the manual curriculum
schedule, the domain-randomization schedule, and the reward weights. No logic
beyond the curriculum-stage factory.

References
----------
Song, Y. et al. (2023). Reaching the limit in autonomous racing.
    *Science Robotics* 8, eadg1462.
See ``docs/plans/2026-05-13-rl-song-prototype-design.md`` §8–§10.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Default policy is sampled in raw 7-vec space: 1 thrust scalar + 6 rotation
# scalars (two 3-vectors that Gram-Schmidt projects onto SO(3)).
RAW_ACTION_DIM: int = 7
# Env-side action interface is 4-vec [roll, pitch, yaw, thrust].
ENV_ACTION_DIM: int = 4

# Actor obs decomposition (cf. design doc §6). Total 59 floats.
ACTOR_OBS_DRONE_DIM: int = 13  # 6D rot + body-vel + body-omega + z
ACTOR_OBS_GATE_DIM: int = 24  # 2 gates * 4 corners * 3 coords
ACTOR_OBS_VISITED_DIM: int = 2  # visited flags for the 2 future gates
ACTOR_OBS_PREV_ACTION_DIM: int = ENV_ACTION_DIM
ACTOR_OBS_OBSTACLE_DIM: int = 16  # 4 obstacles * (3 body-frame xyz + 1 visited)
ACTOR_OBS_DIM: int = (
    ACTOR_OBS_DRONE_DIM
    + ACTOR_OBS_GATE_DIM
    + ACTOR_OBS_VISITED_DIM
    + ACTOR_OBS_PREV_ACTION_DIM
    + ACTOR_OBS_OBSTACLE_DIM
)
assert ACTOR_OBS_DIM == 59, "Actor obs layout drifted from design doc §6"


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters.

    Defaults follow Song 2023 with corrections for the 50 Hz control rate (see
    design doc §8). The ``gamma=0.98`` value gives a ~1 s effective horizon at
    50 Hz, matching Song's ``gamma=0.99`` at 100 Hz.
    """

    n_envs: int = 4096
    n_steps: int = 50  # 1 s rollout at 50 Hz
    n_minibatches: int = 25  # batch_size / minibatch_size = 204800 / 8192
    minibatch_size: int = 8192
    update_epochs: int = 5
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    # Non-zero entropy bonus keeps the policy exploring. The v1 run with
    # ent_coef=0 collapsed to a stable hover (final entropy ~ -6.5); v2 with
    # ent_coef=0.005 reached "approach gate 1 but don't cross" with final
    # entropy ~ -1.7. v3 doubles it to keep exploration alive long enough to
    # discover that crossing gate 1 leads to gate 2 (which is also reachable).
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    total_timesteps: int = 100_000_000
    # Initial log-std for the raw 7-vec Gaussian; sigma ~= 0.6.
    init_log_std: float = -0.5

    @property
    def batch_size(self) -> int:
        """Total transitions per PPO update."""
        return self.n_envs * self.n_steps

    @property
    def n_iterations(self) -> int:
        """Number of PPO updates over the full training budget."""
        return self.total_timesteps // self.batch_size


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the reward function.

    The formulation is Song 2023's progress reward plus an obstacle soft
    barrier and an optional gate-pass bonus. See design doc §7.

    Notes
    -----
    ``omega_coef = 0.02`` is the 50 Hz analogue of Song's ``0.01`` at 100 Hz:
    body-rate penalty is per-step, so the per-second budget is preserved by
    doubling the coefficient when halving the step rate.
    """

    # Multiplier on the Song progress term ||g - p_{k-1}|| - ||g - p_k||. v2
    # ran at 5.0 and produced a policy that parked next to gate 1 without
    # crossing (~0.2m off the opening center, ep_ret +7.5). v3 doubles to 10.0
    # to accelerate the approach phase and the post-crossing dash to gate 2.
    progress_coef: float = 10.0
    omega_coef: float = 0.02
    # Crash penalty was 10.0; reduced because the original ratio of -10 crash
    # to ~+0.003 per-step progress collapsed the policy to a safe hover.
    crash_penalty: float = 5.0
    finish_bonus: float = 10.0
    # Obstacle soft barrier: -w_obs * sum_i exp(-||p - p_obstacle_i||^2 / sigma^2)
    obstacle_weight: float = 0.5
    obstacle_sigma: float = 0.2  # m
    # Big jackpot on gate crossing. v2 ran at 1.0 and the policy discovered
    # that crossing gate 1 forfeited ~+10 of accumulated parking reward (and
    # immediately switched the target to a far-away gate 2), so it preferred
    # to camp. v3 makes the jackpot worth ~3x the parking value to flip the
    # incentive.
    gate_pass_bonus: float = 20.0
    use_gate_pass_bonus: bool = True


@dataclass(frozen=True)
class DRSchedule:
    """Domain-randomization schedule (active at curriculum stage 4).

    Per-channel sampling ranges. See design doc §10 phase 1. ``per_episode``
    channels are sampled at reset and held fixed for the episode; ``per_step``
    channels are resampled every control step.
    """

    # Per-episode physical-parameter randomization.
    mass_rel_range: float = 0.10
    inertia_rel_range: float = 0.10
    thrust_scale_rel_range: float = 0.15
    motor_tau_range_s: tuple[float, float] = (0.015, 0.030)
    drag_rel_range: float = 0.20
    # Per-step sensing noise (Gaussian std).
    pos_noise_std_m: float = 0.01
    vel_noise_std_mps: float = 0.05
    ang_vel_noise_std_radps: float = 0.02
    # Per-episode latency (held fixed) and per-step Ornstein-Uhlenbeck wind.
    latency_range_s: tuple[float, float] = (0.0, 0.020)
    wind_force_max_n: float = 0.1
    wind_tau_s: float = 1.0


@dataclass(frozen=True)
class CurriculumStage:
    """One curriculum stage: which level, what DR, what reset perturbation.

    Promotion is checked every ``promotion_check_iterations`` PPO updates and
    requires both ``promote_target_gate_mean`` and ``promote_crash_rate_max``
    (the latter is +inf when unused).
    """

    name: str
    level: int
    use_domain_randomization: bool
    reset_pos_perturb_m: float
    reset_vel_perturb_mps: float
    reset_yaw_perturb_rad: float
    promote_target_gate_mean: float
    promote_crash_rate_max: float = float("inf")


@dataclass(frozen=True)
class CurriculumConfig:
    """Ordered curriculum stages. Stage index advances on promotion."""

    stages: tuple[CurriculumStage, ...]
    promotion_check_iterations: int = 100
    promotion_window_rollouts: int = 50


def default_curriculum() -> CurriculumConfig:
    """Return the four-stage manual curriculum from design doc §9.

    Returns
    -------
    CurriculumConfig
        Stages 1 (deterministic level-1), 2 (deterministic level-1 with reset
        perturbation), 3 (level-3 no DR), 4 (level-3 with full DR).
    """
    return CurriculumConfig(
        stages=(
            CurriculumStage(
                name="stage1_det_level1",
                level=1,
                use_domain_randomization=False,
                reset_pos_perturb_m=0.0,
                reset_vel_perturb_mps=0.0,
                reset_yaw_perturb_rad=0.0,
                promote_target_gate_mean=3.0,
            ),
            CurriculumStage(
                name="stage2_perturbed_level1",
                level=1,
                use_domain_randomization=False,
                reset_pos_perturb_m=0.2,
                reset_vel_perturb_mps=0.5,
                reset_yaw_perturb_rad=0.7853981633974483,  # pi/4
                promote_target_gate_mean=3.5,
            ),
            CurriculumStage(
                name="stage3_level3_no_dr",
                level=3,
                use_domain_randomization=False,
                reset_pos_perturb_m=0.2,
                reset_vel_perturb_mps=0.5,
                reset_yaw_perturb_rad=0.7853981633974483,
                promote_target_gate_mean=3.0,
                promote_crash_rate_max=0.3,
            ),
            CurriculumStage(
                name="stage4_level3_dr",
                level=3,
                use_domain_randomization=True,
                reset_pos_perturb_m=0.2,
                reset_vel_perturb_mps=0.5,
                reset_yaw_perturb_rad=0.7853981633974483,
                promote_target_gate_mean=float("inf"),  # terminal stage
            ),
        )
    )


@dataclass(frozen=True)
class TrainConfig:
    """Top-level training config bundle."""

    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = field(default_factory=default_curriculum)
    dr: DRSchedule = field(default_factory=DRSchedule)
    # Training loop / IO.
    seed: int = 0
    initial_stage_index: int = 0  # 0-indexed; CLI ``--stage 1`` maps to 0
    max_episode_steps: int = 500  # 10 s at 50 Hz
    checkpoint_every_steps: int = 5_000_000
    eval_video_every_steps: int = 5_000_000
    wandb_project: str = "lsy-drone-racing-rl-song"
    wandb_entity: str | None = None
    run_name: str | None = None  # defaults to <stage>_<seed>_<timestamp> in train.py
