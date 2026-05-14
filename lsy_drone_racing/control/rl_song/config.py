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
    # v9: rollout length 50 → 100 (1 s → 2 s at 50 Hz). With γ=0.997 the
    # effective discount horizon is ~6.9 s; a 1 s rollout forced GAE to lean
    # heavily on the critic bootstrap at the rollout boundary, which both
    # external reviewers flagged as a bias source ("γ horizon is now ~7 s
    # but PPO rollout truncates at 1 s, so GAE is over-relying on critic
    # estimation"). Doubling rollout length lets GAE compute advantages
    # from more on-policy reward and less bootstrapped value, especially
    # important now that the load-bearing reward (finish_bonus=100) only
    # arrives at the end of multi-second episodes.
    n_steps: int = 100  # 2 s rollout at 50 Hz
    n_minibatches: int = 50  # batch_size / minibatch_size = 409600 / 8192
    minibatch_size: int = 8192
    update_epochs: int = 5
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    # Initial entropy bonus. The v3 1e8-step run with constant ent_coef=0.01
    # plateaued at target_gate=1.73 because entropy kept climbing (final
    # +16.3): the bonus rewarded action-spread faster than the policy could
    # refine, so it never committed past gates 0-1. v4 introduced a linear
    # anneal from ent_coef -> ent_coef_final across training so early
    # iterations explore (discover gates 0-1) and late iterations commit
    # (refine gates 1-2-3-4). v5 (2e8 with floor 0.001) reached finish_rate
    # ~0.5% but stayed at entropy +15.9, suggesting the floor was still too
    # high. v6 drops the floor to zero so the entropy bonus fully vanishes by
    # end of training and the policy can commit deterministically.
    # v8: halve initial entropy bonus (0.01 → 0.005). Stage 3 from scratch
    # with gate randomization fell into a hover attractor at ent_coef=0.01;
    # at entropy ~25-40 the bonus (0.005-0.01 × 25-40 ≈ 0.13-0.40 per step)
    # was competitive with progress reward, so the policy never committed
    # to forward motion. Halving keeps exploration active early but lets the
    # progress signal dominate once gates are crossed.
    ent_coef: float = 0.005
    ent_coef_final: float = 0.0
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
    # v9: increased finish_bonus from 10 to 100 in tandem with shrinking the
    # per-gate jackpot below. The reward economics from v8 paid +60 for
    # reach-gate-2-then-crash vs +10 for finish, so crashing was rational.
    # Putting the load-bearing reward on race completion makes finishing
    # dominant by an order of magnitude under any realistic episode horizon.
    finish_bonus: float = 100.0
    # Obstacle soft barrier: -w_obs * sum_i exp(-||p - p_obstacle_i||^2 / sigma^2)
    obstacle_weight: float = 0.5
    obstacle_sigma: float = 0.2  # m
    # v9: shrink gate jackpot from 20 to 2. The v7/v8 jackpot of 20×(idx+1)
    # paid 20/40/60/80 per gate, dominating r_prog (~+0.17/step ≈ +8.5 per
    # 50-step rollout) by 2-10×. Song 2023 and Kaufmann 2023 use dense
    # progress as the primary signal with only a small per-gate event marker;
    # external review of the v8 results flagged the 20-80 jackpot as the
    # proximate cause of the "rush through gate 2 then crash" local optimum.
    gate_pass_bonus: float = 2.0
    use_gate_pass_bonus: bool = True
    # v9: disable per-gate scaling. Uniform 2/2/2/2 instead of 2/4/6/8 removes
    # the incentive to rush past earlier gates to bank the larger later-gate
    # jackpot. The dense progress reward already pulls the policy through
    # later gates without needing an escalating discrete payoff.
    scale_gate_bonus_by_index: bool = False
    # v8: per-step time penalty. With randomized gates the random-init policy
    # has zero progress in expectation, while a stationary "hover" policy
    # collects zero shaping reward and just times out — making hover the Q≈0
    # attractor that drives the action distribution back to uniform under the
    # entropy bonus. A small constant subtraction makes hover-timeout cost
    # 0.05 × 500 = -25, so any episode that reaches even one gate (+20 jackpot)
    # strictly dominates. Philosophy-aligned with Song 2023's "minimize lap
    # time" objective without changing the reward terms they use.
    time_penalty: float = 0.05


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
    # v8: scales the gate_pos, gate_rpy, and obstacle_pos randomization ranges
    # loaded from ``config/levelN.toml`` (and selected via
    # ``TRACK_RANDOMIZATION_KEYS`` in the env wrapper). A value of 1.0 uses the
    # full level-3 randomization budget (±0.15 m on gate_pos and obstacle_pos).
    # Smaller values produce an easier near-fixed-track regime so the policy
    # can first learn approach-to-nominal before adapting to noise; values
    # below 1.0 are intended for the ``stage3a/b/c`` warm-up sub-stages. Has
    # no effect on stages with ``level != 3`` (level 1 has no gate/obstacle
    # randomization to scale).
    gate_rand_scale: float = 1.0
    # v9 (Song 2023 §III-B Phase 1): probability that an env is re-spawned
    # at the midpoint of a random path segment (hovering, vel=0, identity
    # attitude, target_gate=k) instead of the toml start position. Covers
    # the full state space immediately so the policy is exposed to gate-3
    # and gate-4 observations from step 0, fixing the "policy never trains
    # on later-gate states because it crashed earlier" pathology. Set to
    # 0.5 on level-3 stages; harmless 0.0 default elsewhere.
    segment_init_prob: float = 0.0
    # Half-width of uniform position jitter applied to the segment midpoint
    # so the policy sees a distribution of states around each segment center
    # rather than a single point.
    segment_init_perturb_m: float = 0.10


@dataclass(frozen=True)
class CurriculumConfig:
    """Ordered curriculum stages. Stage index advances on promotion."""

    stages: tuple[CurriculumStage, ...]
    promotion_check_iterations: int = 100
    promotion_window_rollouts: int = 50


def default_curriculum() -> CurriculumConfig:
    """Return the curriculum.

    Layout
    ------
    Stages 1-2 are legacy fixed-track stages, kept for reproducibility but
    no longer used as the canonical entry point. The level-3 (track
    randomization) work is split across three v8 sub-stages with progressively
    larger gate/obstacle randomization scale, replacing the single
    ``stage3_level3_no_dr`` that route-memorizing warm-starts kept failing on:

    * ``stage3a`` (idx 2, ``gate_rand_scale=0.20``): nearly-fixed track
      (≈±0.03 m on gate_pos), where a randomly-initialized policy can
      reliably stumble onto gate 1 within feasible compute and the
      gate-pass reward starts driving credit assignment.
    * ``stage3b`` (idx 3, ``gate_rand_scale=0.50``): half-randomization,
      teaches generalization away from the nominal position before the
      full level-3 budget.
    * ``stage3c`` (idx 4, ``gate_rand_scale=1.00``): full level-3
      randomization (±0.15 m). This is the actual deployment-matching
      stage; everything before is just curriculum.

    The terminal ``stage4_level3_dr`` (idx 5, full DR) is unchanged.

    Returns
    -------
    CurriculumConfig
        Six-stage curriculum (legacy 1-2, new 3a/b/c, terminal 4).
    """
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
                # Lowered from 2.5 → 1.8 after the v8 layer-2-fix from-scratch
                # run plateaued at target_gate ~0.95 / max_gate ~1.45 (reaches
                # gate 2 on ~45% of episodes) without promoting. 2.5 was the
                # original 3-gates-passed target; with the policy already
                # consistently clearing gate 1 and reaching gate 2, promoting
                # to the harder stage 3b (scale 0.50) should force more
                # cautious flying and break the fast-but-crashy local optimum.
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
                # Stage 3 deployment target — do not auto-promote into stage 4
                # (DR) during this experiment.
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
