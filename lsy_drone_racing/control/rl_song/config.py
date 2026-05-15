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

    Defaults follow Song 2023 with corrections for the 50 Hz control rate
    (see design doc §8). ``gamma=0.997`` gives a ~6.9 s effective horizon at
    50 Hz so that the load-bearing terminal reward (``finish_bonus=100``,
    paid at the end of multi-second episodes) actually back-propagates
    through the trajectory; ``gamma=0.98`` (Song's per-step rate match)
    underweights the finish signal severely and was a v9 regression that
    silently snuck into the v10 ablations via the committed default.
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
    gamma: float = 0.997
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
    # v12 bumped 10 -> 20 to compensate for the removed time_penalty; v13A
    # (convention A + prog=20) and v12 (sign-flip + prog=20) both regressed
    # to ~11% finish / ~0.85 max_gate vs v11's 21% / 1.16. Per-episode r_prog
    # at prog=20 (~+17) exceeded the finish signal (+11), inverting the
    # reward economics so the policy preferred to harvest oscillatory r_prog
    # near gate 0 rather than commit to a pass. Reverted to 10.0; next
    # ablation (v14) is progress clipping or progress-once accounting, not
    # another coefficient bump.
    # v15: down to 1.0 to match Song 2023 Sci. Robotics §V exactly. Their
    # progress term has unit coefficient; integrated over a ~10 m level-0
    # track that lands at ~+10, comparable to a +10 finish bonus. Our v9-v14
    # progress_coef of 10 with finish_bonus 100 was internally consistent at
    # the same ratio, but the 10x absolute magnitude pushed PPO advantage
    # variance into ranges where the clip range becomes non-binding —
    # rescaling back to Song's absolute scale keeps the optimizer in the
    # regime the paper validated.
    progress_coef: float = 1.0
    # v15: down to 0.01 to match Song 2023's exact body-rate coefficient.
    # The 0.02 value here was justified earlier as the 50 Hz analogue of
    # Song's 100 Hz 0.01, but Song 2023 quotes b = 0.01 without specifying
    # control frequency and the 100 Hz figure was a misreading. Reverting
    # to the verbatim paper value.
    omega_coef: float = 0.01
    # v15: 5.0 -> 10.0 to match Song 2023 r_crash = -10.0. The earlier
    # reduction to 5.0 was motivated by "policy collapsed to safe hover under
    # -10 crash vs ~+0.003 per-step progress"; that ratio assumed
    # progress_coef = 1 with old scale, but our v9-v14 had progress_coef = 10
    # which made the relative crash penalty 5x smaller than Song's intent.
    # With v15's progress_coef back to 1.0, restoring crash to -10 gives
    # the same balance Song used.
    crash_penalty: float = 10.0
    # v9: increased finish_bonus from 10 to 100 in tandem with shrinking the
    # per-gate jackpot below. The reward economics from v8 paid +60 for
    # reach-gate-2-then-crash vs +10 for finish, so crashing was rational.
    # Putting the load-bearing reward on race completion makes finishing
    # dominant by an order of magnitude under any realistic episode horizon.
    # v15: 100 -> 10 to match Song 2023 r_finish = +10. The v9 motivation
    # (finish must dominate the gate-jackpot) is moot now that
    # gate_pass_bonus = 0. Song's r_finish ≈ integrated r_prog over a
    # successful lap, which is a deliberate design — the policy should
    # treat each per-step progress contribution as carrying equal weight
    # to the finish event.
    finish_bonus: float = 10.0
    # Obstacle soft barrier: -w_obs * sum_i exp(-||p - p_obstacle_i||^2 / sigma^2)
    obstacle_weight: float = 0.5
    obstacle_sigma: float = 0.2  # m
    # v9: shrink gate jackpot from 20 to 2. The v7/v8 jackpot of 20×(idx+1)
    # paid 20/40/60/80 per gate, dominating r_prog (~+0.17/step ≈ +8.5 per
    # 50-step rollout) by 2-10×. Song 2023 and Kaufmann 2023 use dense
    # progress as the primary signal with only a small per-gate event marker;
    # external review of the v8 results flagged the 20-80 jackpot as the
    # proximate cause of the "rush through gate 2 then crash" local optimum.
    # v11: disable. Neither Song 2023 nor Liu use a per-gate event bonus.
    # The dense progress reward + finish_bonus under high gamma should
    # cover gate transitions without an explicit discrete payoff.
    gate_pass_bonus: float = 0.0
    use_gate_pass_bonus: bool = False
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
    # v11: disable. Neither Song 2023 nor Liu use a per-step time penalty.
    # The v8 motivation (escape hover Q=0 attractor on randomized stages)
    # is now subsumed by gamma=0.997 + seg-init + Liu guidance, all of
    # which give the hover policy a strictly negative Q. Time penalty also
    # had a known downside: it priced "crash trying" cheaper than "hover
    # safely" (-5 vs -25), inflating crash rate on early-stage runs.
    # v14: re-enable at 0.02 (40% of v8's 0.05). Sim eval of v11 on level 0
    # showed 0/100 finish — the drone slips above gate 1 and parks beyond
    # the r_guid window (|x_gate| > guide_k0 = 1.5 m) where every per-step
    # reward term is identically zero. The v11 reasoning that "hover Q is
    # subsumed by foregone discounted finish_bonus" assumed r_guid covers
    # the whole flight space; it doesn't (finite support, see guide_k0).
    # 0.02 × 1500-step truncation = -30, which strictly dominates the
    # crash_penalty of -5, restoring the property that any committed
    # attempt beats indefinite hovering.
    # v15: back to 0.0. Sim eval of v14 on level 0 showed the per-step time
    # penalty made the policy retreat *further* from the gate (escape the
    # r_guid field faster) rather than commit. With r_guid also disabled
    # in v15 there is no negative-shaping zone to escape, so the time
    # penalty's role disappears. Song 2023 has no per-step time penalty.
    time_penalty: float = 0.0
    # v10: forward-flight bias in body frame (Liu eq. 8). Off by default.
    # Liu motivation is sensor-cone alignment under a 90 deg FPV depth camera
    # (the drone must point its FOV where it is going to perceive obstacles).
    # We have state-based obs, so this term solves a problem we do not have;
    # the code path is retained for ablation.
    use_vel_shaping: bool = False
    vel_lat_coef: float = -0.02
    vel_back_coef: float = -0.05
    # v10: asymmetric gate guidance field in target-gate local frame (Liu
    # eq. 6-7). Front-side shaping attracts the policy to the aperture
    # centerline, while back-side shaping penalizes off-axis wrong-side
    # approaches that symmetric r_prog cannot distinguish.
    # v15: disabled. Kaufmann 2023 Nature ("Swift") has no guidance reward
    # term — their working recipe is r_prog + r_perception + r_command -
    # r_crash. Song 2023 Sci. Robotics also drops it (their reward is just
    # gate progress, body-rate penalty, sparse crash and finish). Song 2021
    # IROS introduced the safety reward as an *optional* component "designed
    # to reduce the risk of crashing in training settings that feature large
    # track changes" and explicitly does not need it for basic racing. Liu
    # 2024 extends it for rectangular gates and adds the wrong-side penalty.
    # Our v10-v14 application has not been load-bearing for the level-0
    # cold-start task — see the 2026-05-15 v15 handoff. Reverting to the
    # Song/Kaufmann minimum.
    use_guide: bool = False
    # v13B: bumped 0.15 -> 2.0 in tandem with the switch to Δ-potential
    # shaping (see ``use_guide_delta_phi`` below). Under ΔΦ the integrated
    # r_guid over a perfectly centered pass is approximately guide_coef,
    # so 2.0 gives ~10% of per-gate r_prog (≈20 at progress_coef=20) as an
    # aperture-alignment bonus. The legacy static-field branch
    # (use_guide_delta_phi=False) is no longer well-tuned at this scale —
    # at 2.0 the per-step penalty would dominate r_prog and freeze the
    # policy.
    # v14: reverted to 0.15 in tandem with use_guide_delta_phi=False (see
    # the ``time_penalty`` block above for the level-0 failure diagnostic
    # that motivates the v14 revert + retune).
    guide_coef: float = 0.15
    # v14: widened 1.5 -> 3.0. The level-0 spawn at world (-1.5, 0.75, 0.01)
    # is ~2.1 m from gate 1 in gate-frame x, so at k0=1.5 the policy gets
    # zero r_guid signal until it has already walked itself most of the
    # way to gate 1. Widening to 3.0 puts the spawn inside the window with
    # ``guide_window**2`` ≈ 0.09 — small but non-zero gradient from step 1.
    # Combined with time_penalty=0.02 this should remove the neutral-zone
    # parking attractor that v11 found.
    guide_k0: float = 3.0
    guide_k1: float = 1.0
    guide_k2: float = 0.3
    # v13B: Δ-potential gate guidance. When True, r_guid is computed as
    # guide_coef · (Φ_t − Φ_{t-1}) with
    # Φ = aperture_score(y,z) · sigmoid(-x / guide_kx). The potential is
    # monotonic front-to-back along the gate normal, so the integrated
    # reward over a perfectly centered pass is approximately guide_coef
    # (Φ goes ~0 → ~1). Hovering produces zero r_guid, removing the
    # hover-on-approach attractor that v12's positive static field
    # created. Both endpoints use the pre-step target gate frame, so the
    # gate-transition step pays positive ΔΦ without a mask.
    # v14: disabled. v13B converged to 0% finish because pure ΔΦ has no
    # anti-loiter mechanism. Reverting to the static field with
    # time_penalty + widened guide_k0 as a less pure but functional
    # alternative.
    use_guide_delta_phi: bool = False
    guide_kx: float = 0.5


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
    """Return the active curriculum.

    v11: stripped to a single stage — ``level2_seginit`` — while we ablate
    the v10 reward (Liu guidance, no time penalty, no gate jackpot, gamma
    0.997) directly on a fixed nominal level-2 layout with ±0.15 m wobble
    and Song §III-B Phase 1 seg-init. Promotion is disabled
    (``promote_target_gate_mean=inf``) so this is effectively a no-op
    curriculum. The legacy stage1/2/3a/b/c/4 progression is preserved in
    :func:`_full_curriculum` and can be reinstated by swapping the body
    of this function back.
    """
    pi_over_4 = 0.7853981633974483
    return CurriculumConfig(
        stages=(
            CurriculumStage(
                # Misnomer kept for run-name compatibility with v11-v15
                # wandb runs. v16a actually disables seg-init entirely
                # (segment_init_prob=0.0) — every episode starts at the
                # toml drone-start position, on the ground, with vel=0.
                # The v15 sim eval showed the same hover-above-gate-1
                # failure mode v11/v14 had, with the additional regression
                # of negative ep_ret. Diagnosis is that the 50% seg-init
                # mid-track-flying episodes were teaching the policy a
                # mid-track skill that never connects back to "thread
                # gate 0 from cold start". Switching to pure cold-start
                # forces the policy to actually solve the takeoff +
                # gate-0 subtask that v7a solved in stage 1.
                name="level2_seginit",
                level=2,
                use_domain_randomization=False,
                reset_pos_perturb_m=0.2,
                reset_vel_perturb_mps=0.0,
                reset_yaw_perturb_rad=pi_over_4,
                gate_rand_scale=1.00,
                segment_init_prob=0.0,
                promote_target_gate_mean=float("inf"),
                promote_crash_rate_max=0.3,
            ),
        )
    )


def _full_curriculum() -> CurriculumConfig:
    """Legacy seven-stage curriculum (v9/v10) preserved for reinstatement.

    Layout
    ------
    Stages 1-2 are fixed-track level-1 stages. Stage3a/b/c sub-stages of
    level-3 with progressively larger gate/obstacle randomization scale.
    The terminal ``stage4_level3_dr`` adds full DR. ``level2_seginit``
    is the v11 single-stage experiment, kept here so the indices match
    what was used in committed runs.
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
    # Training loop / IO.
    seed: int = 0
    initial_stage_index: int = 0  # 0-indexed; CLI ``--stage 1`` maps to 0
    max_episode_steps: int = 500  # 10 s at 50 Hz
    checkpoint_every_steps: int = 5_000_000
    eval_video_every_steps: int = 5_000_000
    wandb_project: str = "lsy-drone-racing-rl-song"
    wandb_entity: str | None = None
    run_name: str | None = None  # defaults to <stage>_<seed>_<timestamp> in train.py
