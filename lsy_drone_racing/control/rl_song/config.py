"""Static configuration for the Song-2023 RL prototype."""

from __future__ import annotations

from dataclasses import dataclass, field

RAW_ACTION_DIM: int = 4
ENV_ACTION_DIM: int = 4
TANGENT_ALPHA_MAX_RAD: float = 0.16

N_OBSTACLES: int = 4
N_NEAREST_OBSTACLES: int = 2

ACTOR_OBS_DRONE_DIM: int = 12
ACTOR_OBS_GATE_DIM: int = 24
ACTOR_OBS_PREV_ACTION_DIM: int = 0
_PER_OBSTACLE_SLOT_DIM: int = 2 + 1 + N_OBSTACLES + 1
ACTOR_OBS_OBSTACLE_DIM: int = N_NEAREST_OBSTACLES * _PER_OBSTACLE_SLOT_DIM
ACTOR_OBS_DIM: int = (
    ACTOR_OBS_DRONE_DIM + ACTOR_OBS_GATE_DIM + ACTOR_OBS_PREV_ACTION_DIM + ACTOR_OBS_OBSTACLE_DIM
)
assert ACTOR_OBS_DIM == 52, "Actor obs layout drift"


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters."""

    n_envs: int = 16384
    n_steps: int = 250
    n_minibatches: int = 500
    minibatch_size: int = 8192
    update_epochs: int = 3
    gamma: float = 0.998
    gae_lambda: float = 0.97
    clip_coef: float = 0.2
    ent_coef: float = 0.02
    ent_coef_final: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    target_kl: float = 0.02
    total_timesteps: int = 500_000_000
    init_log_std: float = -0.5

    @property
    def batch_size(self) -> int:
        return self.n_envs * self.n_steps

    @property
    def n_iterations(self) -> int:
        return self.total_timesteps // self.batch_size


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights."""

    progress_coef: float = 10.0
    use_velocity_progress: bool = False
    lookahead_coef: float = 0.0
    lookahead_entry_offset_m: float = 0.5
    lookahead_mask_through: int = 0
    lookahead_near_plane_m: float = 0.5
    wrong_side_coef: float = 0.0
    wrong_side_target_min: int = 1
    dipole_coef: float = 0.0
    dipole_sigma: float = 0.5
    omega_coef: float = 0.02
    r_smooth_coef: float = 0.0
    crash_penalty: float = 15.0
    finish_bonus: float = 100.0
    use_obstacle_barrier: bool = False
    obstacle_weight: float = 0.0
    obstacle_sigma: float = 0.5
    use_gate_frame_barrier: bool = False
    gate_frame_weight: float = 0.0
    gate_frame_sigma: float = 0.08
    gate_pass_bonus: float = 10.0
    use_gate_pass_bonus: bool = False
    scale_gate_bonus_by_index: bool = True
    use_exit_vel_bonus: bool = False
    exit_vel_coef: float = 10.0
    exit_vel_clip_mps: float = 5.0
    use_caution: bool = False
    caution_coef: float = 0.05
    caution_peak_m: float = 1.0
    caution_kernel_m: float = 0.8
    caution_visible_threshold_m: float = 0.005
    caution_visible_factor: float = 0.0
    time_penalty: float = 0.0
    use_vel_shaping: bool = False
    vel_lat_coef: float = -0.02
    vel_back_coef: float = -0.05
    use_guide: bool = False
    guide_coef: float = 0.5
    guide_k0: float = 3.0
    guide_k1: float = 1.0
    guide_k2: float = 0.3
    use_guide_delta_phi: bool = False
    guide_kx: float = 0.5


@dataclass(frozen=True)
class DRSchedule:
    """Domain-randomization schedule (per-channel sampling ranges)."""

    mass_rel_range: float = 0.10
    inertia_rel_range: float = 0.10
    thrust_scale_rel_range: float = 0.15
    motor_tau_range_s: tuple[float, float] = (0.015, 0.030)
    drag_rel_range: float = 0.20
    pos_noise_std_m: float = 0.01
    vel_noise_std_mps: float = 0.05
    ang_vel_noise_std_radps: float = 0.02
    latency_range_s: tuple[float, float] = (0.0, 0.020)
    wind_force_max_n: float = 0.1
    wind_tau_s: float = 1.0


@dataclass(frozen=True)
class CurriculumStage:
    """One curriculum stage."""

    name: str
    level: int
    use_domain_randomization: bool
    reset_pos_perturb_m: float
    reset_vel_perturb_mps: float
    reset_yaw_perturb_rad: float
    promote_target_gate_mean: float
    promote_crash_rate_max: float = float("inf")
    gate_rand_scale: float = 1.0
    segment_init_prob: float = 0.0
    segment_init_perturb_m: float = 0.10
    segment_init_vel_mps: float = 0.0
    phase2_prob: float = 0.0
    phase2_capacity_per_gate: int = 4096
    phase2_warmup_steps: int = 0
    sensor_range_random_min: float = 0.0
    sensor_range_random_max: float = 0.0


@dataclass(frozen=True)
class CurriculumConfig:
    """Ordered curriculum stages."""

    stages: tuple[CurriculumStage, ...]
    promotion_check_iterations: int = 100
    promotion_window_rollouts: int = 50


def default_curriculum() -> CurriculumConfig:
    """Active curriculum: single level-2 stage with Phase-1+2 scaffolding."""
    return CurriculumConfig(
        stages=(
            CurriculumStage(
                name="stage1_level2_phase12",
                level=2,
                use_domain_randomization=False,
                reset_pos_perturb_m=0.0,
                reset_vel_perturb_mps=0.0,
                reset_yaw_perturb_rad=0.0,
                gate_rand_scale=1.00,
                segment_init_prob=0.40,
                segment_init_vel_mps=2.5,
                phase2_prob=0.40,
                phase2_capacity_per_gate=4096,
                phase2_warmup_steps=20_000_000,
                promote_target_gate_mean=float("inf"),
                promote_crash_rate_max=float("inf"),
            ),
        )
    )


def _default_curriculum_v42_history() -> CurriculumConfig:  # pragma: no cover
    """Two-stage L1→L2 curriculum, preserved for reinstatement."""
    pi_over_4 = 0.7853981633974483
    return CurriculumConfig(
        stages=(
            CurriculumStage(
                name="stage1_det_level1",
                level=1,
                use_domain_randomization=False,
                reset_pos_perturb_m=0.10,
                reset_vel_perturb_mps=0.0,
                reset_yaw_perturb_rad=0.3,
                gate_rand_scale=1.00,
                segment_init_prob=0.10,
                segment_init_vel_mps=2.5,
                phase2_prob=0.10,
                phase2_capacity_per_gate=4096,
                phase2_warmup_steps=20_000_000,
                promote_target_gate_mean=3.0,
                promote_crash_rate_max=0.3,
            ),
            CurriculumStage(
                name="level2_cold_v38_delta_tangent",
                level=2,
                use_domain_randomization=False,
                reset_pos_perturb_m=0.2,
                reset_vel_perturb_mps=0.0,
                reset_yaw_perturb_rad=pi_over_4,
                gate_rand_scale=1.00,
                segment_init_prob=0.30,
                segment_init_vel_mps=2.5,
                phase2_prob=0.30,
                phase2_capacity_per_gate=4096,
                phase2_warmup_steps=50_000_000,
                promote_target_gate_mean=float("inf"),
                promote_crash_rate_max=0.3,
            ),
        )
    )


def _full_curriculum() -> CurriculumConfig:
    """Legacy seven-stage curriculum, preserved for reinstatement."""
    pi_over_4 = 0.7853981633974483
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
                reset_yaw_perturb_rad=pi_over_4,
                promote_target_gate_mean=3.5,
            ),
            CurriculumStage(
                name="stage3a_level3_rand0.2",
                level=3,
                use_domain_randomization=False,
                reset_pos_perturb_m=0.2,
                reset_vel_perturb_mps=0.0,
                reset_yaw_perturb_rad=pi_over_4,
                gate_rand_scale=0.20,
                segment_init_prob=0.5,
                promote_target_gate_mean=1.8,
                promote_crash_rate_max=0.3,
            ),
            CurriculumStage(
                name="stage3b_level3_rand0.5",
                level=3,
                use_domain_randomization=False,
                reset_pos_perturb_m=0.2,
                reset_vel_perturb_mps=0.0,
                reset_yaw_perturb_rad=pi_over_4,
                gate_rand_scale=0.50,
                segment_init_prob=0.5,
                promote_target_gate_mean=1.8,
                promote_crash_rate_max=0.3,
            ),
            CurriculumStage(
                name="stage3c_level3_rand1.0",
                level=3,
                use_domain_randomization=False,
                reset_pos_perturb_m=0.2,
                reset_vel_perturb_mps=0.0,
                reset_yaw_perturb_rad=pi_over_4,
                gate_rand_scale=1.00,
                segment_init_prob=0.5,
                promote_target_gate_mean=float("inf"),
                promote_crash_rate_max=0.3,
            ),
            CurriculumStage(
                name="stage4_level3_dr",
                level=3,
                use_domain_randomization=True,
                reset_pos_perturb_m=0.2,
                reset_vel_perturb_mps=0.0,
                reset_yaw_perturb_rad=pi_over_4,
                gate_rand_scale=1.00,
                promote_target_gate_mean=float("inf"),
            ),
            CurriculumStage(
                name="level2_seginit",
                level=2,
                use_domain_randomization=False,
                reset_pos_perturb_m=0.2,
                reset_vel_perturb_mps=0.0,
                reset_yaw_perturb_rad=pi_over_4,
                gate_rand_scale=1.00,
                segment_init_prob=0.5,
                promote_target_gate_mean=float("inf"),
                promote_crash_rate_max=0.3,
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
    seed: int = 0
    initial_stage_index: int = 0
    max_episode_steps: int = 500
    tangent_alpha_max_rad: float = TANGENT_ALPHA_MAX_RAD
    checkpoint_every_steps: int = 5_000_000
    eval_video_every_steps: int = 5_000_000
    wandb_project: str = "lsy-drone-racing-rl-song"
    wandb_entity: str | None = None
    run_name: str | None = None
