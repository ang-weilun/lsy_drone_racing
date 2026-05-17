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
    # v22 (level-3 ablation): 0.0 -> 0.001. v21cold on level 3 (300M)
    # plateaued at ``max_gate ≈ 0.8`` from step 200M onward while
    # ``finish_rate`` kept climbing — divergence diagnostic for a
    # narrow "lucky-zone" policy: the policy refined a single approach
    # that wins when gate 1 lands in a small region of the safety-
    # limits sample space but cannot generalize across the full
    # randomized layout. The level-3 task fundamentally requires
    # reading the gate-position obs and routing accordingly; a fully-
    # committed policy (entropy → 0) cannot keep searching for that.
    # A small non-zero floor preserves exploration noise across the
    # whole run so the policy keeps probing alternative trajectories.
    # v5 (an old experiment) tried 0.001 with the v3 entropy schedule
    # and stayed at entropy +15.9 — but that was paired with
    # ent_coef=0.01 (initial), so the floor's static contribution was
    # 10× higher. At our ent_coef=0.005 initial the 0.001 floor is
    # 20% of the starting bonus instead of v5's 10%, but the new role
    # is to prevent late-stage over-commitment rather than to drive
    # early exploration.
    # v25: 0.001 -> 0.005 (constant ent_coef throughout, matches the
    # start value). v24 (warm-start + seg-init) collapsed to entropy
    # = -7.2 at 300M with floor=0.001 and start=0.001 (constant 0.001
    # schedule), confirming 0.001 is too low to prevent over-commitment
    # on the randomized layout. Bumping the floor 5× to match the
    # start value gives a flat schedule at the same magnitude v21
    # successfully trained under for the first ~100M steps; v21
    # plateaued its entropy near -2 with this magnitude before the
    # schedule annealed it down further. Keeping ent_coef constant
    # at 0.005 throughout v25 keeps that level of exploration alive
    # the whole way, blocking the lucky-zone collapse path.
    ent_coef_final: float = 0.005
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
    # track that lands at ~+10, comparable to a +10 finish bonus.
    # v17: back to 10.0. Wandb history of v16a showed early exploration
    # (max_gate ~ 0.003, crash_rate ~ 0.5%) collapse to "do nothing" by
    # step 18M when at this scale (progress_coef=1, crash_penalty=10,
    # finish_bonus=10). The crash signal (-10/episode) dominated the
    # progress signal too strongly for PPO to find the rare +10 finishes.
    # Restoring v9-v14's progress_coef = 10 gives integrated r_prog ≈ +100
    # over a successful trajectory — matched by finish_bonus = 100 below
    # and 20x the v17 crash_penalty (5), restoring the positive/negative
    # balance v11 trained under.
    progress_coef: float = 10.0
    # v20: switch r_prog from the legacy distance-delta formulation
    # (||g - p_prev|| - ||g - p||) to the Song-2023 / Kaufmann-2023
    # velocity-projection variant: project the world-frame displacement
    # onto the gate's forward-normal axis (x_local of the target gate)
    # and scale by progress_coef. The reward direction is then fixed in
    # gate frame, so lateral drift away from the traversal axis costs
    # progress while equivalent-magnitude motion along the axis pays.
    # The legacy term oscillated ~+0.034/step on v19's stationary policy
    # (~+17 over 500 steps) because rotation in place still reduces
    # ||g - p|| momentarily; the velocity-projection variant integrates
    # to zero under such rotation and is harder to "harvest" without
    # actually flying through the gate.
    # v21: back to False. The video of v20 showed the policy tilting
    # toward the gate horizontally while still on the ground, sliding,
    # and tipping over without ever lifting off. Velocity-projection
    # r_prog projects onto the gate's horizontal forward-normal axis,
    # so pure vertical motion (the load-bearing takeoff subtask) pays
    # zero r_prog gain — while horizontal tilt earns ~1.33/step. PPO
    # took the bigger per-step gradient and learned tilt-and-slide
    # instead of thrust-up-then-approach. The legacy distance-delta
    # formulation credits any motion that closes ||g - p||, including
    # vertical when the gate is above the spawn (gate 1 is at z=0.7,
    # spawn at z=0.01), which is the geometry v7a learned cold-start
    # takeoff under.
    use_velocity_progress: bool = False
    # v15: down to 0.01 to match Song 2023's exact body-rate coefficient.
    # The 0.02 value here was justified earlier as the 50 Hz analogue of
    # Song's 100 Hz 0.01, but Song 2023 quotes b = 0.01 without specifying
    # control frequency and the 100 Hz figure was a misreading. Reverting
    # to the verbatim paper value.
    omega_coef: float = 0.01
    # v15: 5.0 -> 10.0 to match Song 2023 r_crash = -10.0.
    # v17: back to 5.0. The v15 raise to 10.0 (combined with progress_coef
    # = 1) made the per-episode crash penalty dominate the per-episode
    # integrated r_prog by 5–10x, pushing PPO toward the "do nothing"
    # local optimum. v11/v14 used 5.0 with progress_coef = 10, which is
    # the balance that produced 21% biased finish in v11. Reverting.
    # v19: 5.0 -> 15.0. Pairs with the v19 reintroduction of
    # time_penalty=0.05 (see block below). With time_penalty=0.05 the
    # 500-step do-nothing baseline is r_prog - time = +18 - 25 = -7
    # per episode, which makes any crash that pays less than -7 prefer-
    # able to surviving — i.e. the policy can escape do-nothing by
    # suicide instead of by exploring. Raising crash_penalty to 15
    # makes any crash strictly worse than do-nothing (-15 < -7) so the
    # only direction of escape from the attractor is forward through a
    # gate (+20 jackpot dominates by 30+).
    crash_penalty: float = 15.0
    # v9: increased finish_bonus from 10 to 100 in tandem with shrinking the
    # per-gate jackpot below. The reward economics from v8 paid +60 for
    # reach-gate-2-then-crash vs +10 for finish, so crashing was rational.
    # Putting the load-bearing reward on race completion makes finishing
    # dominant by an order of magnitude under any realistic episode horizon.
    # v15: 100 -> 10 to match Song 2023 r_finish = +10.
    # v17: back to 100. At progress_coef = 10 (restored) and ~10 m track,
    # integrated r_prog over a finish is ~100. Matching finish_bonus = 100
    # keeps the discrete reward at the gate-pass / finish event on the
    # same order as the dense shaping, which is the v11/v14 scale that
    # produced learnable signal. With finish_bonus = 10 + progress_coef
    # = 1 (v15 scale) the value function targets were too small for PPO
    # to make meaningful updates (value_loss collapsed to 0.012).
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
    # v11: disable. Neither Song 2023 nor Liu use a per-gate event bonus.
    # The dense progress reward + finish_bonus under high gamma should
    # cover gate transitions without an explicit discrete payoff.
    # v18: re-enable at 20 with per-gate scaling (1x, 2x, 3x, 4x → 20,
    # 40, 60, 80). This is v7a's exact positive-event recipe. v15-v17 all
    # collapsed under PPO because cold-start exploration from spawn never
    # finds the +100 finish event, so crash signal dominates the gradient
    # and the policy converges to do-nothing. A per-gate event gives the
    # policy a positive gradient as it learns gate-1, gate-2, ... rather
    # than requiring it to learn the entire track at once. v7a (with this
    # exact recipe + no r_guid + no seg-init) solved level-1 to 100%
    # finish from cold start in 100M steps.
    gate_pass_bonus: float = 40.0
    use_gate_pass_bonus: bool = True
    # v9: disable per-gate scaling. Uniform 2/2/2/2 instead of 2/4/6/8 removes
    # the incentive to rush past earlier gates to bank the larger later-gate
    # jackpot. The dense progress reward already pulls the policy through
    # later gates without needing an escalating discrete payoff.
    # v18: re-enable. The v9 motivation assumed a working policy that
    # already passes early gates; v15-v17 collapsed because PPO couldn't
    # find ANY gate-passes from cold start under finish-only + small
    # negative signal. With the jackpot 20/40/60/80, the gradient toward
    # "pass gate 1" is +20 — an order of magnitude bigger than the per-
    # crash penalty -5, so the policy can learn gate-1 traversal first and
    # extend to later gates as it gets better. v7a (this exact recipe)
    # solved level 1 to 100% finish in 100M steps; v8's "rush past 1 to
    # bank gate-2's bigger jackpot" failure mode only matters if the
    # policy can already pass gates 1 and 2, which is exactly the regime
    # we are not yet in.
    # v20: double the per-gate jackpot base 20 -> 40 so the scaled
    # bonuses become 40/80/120/160. v18/v19 showed gate_pass_bonus=20
    # scaled is irrelevant when the discrete event is never triggered
    # from cold start; doubling it is cheap once gate 1 is found and
    # gives a louder gradient at the moment of first discovery.
    scale_gate_bonus_by_index: bool = True
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
    # v19: 0.0 -> 0.05. v18 with per-gate jackpot still collapsed to
    # do-nothing (finish_rate=0, max_gate=0, r_gate_bonus=0 over 842
    # episodes at 100M) because the cold-start ep_ret was +14.13 — a
    # net-positive attractor with no escape pressure. r_prog contributed
    # ~+0.036/step ≈ +18 per 500-step episode, so any time_penalty must
    # exceed 0.036/step to make do-nothing net negative. 0.05/step
    # crosses that threshold cleanly: per-episode budget is -25, which
    # combined with +18 r_prog gives a -7 do-nothing baseline. Paired
    # with crash_penalty=15 (above) and gate_pass_bonus=20 scaled, the
    # full incentive table is: crash any time <= -15; do nothing -7;
    # pass gate 1 at step 100 +33; finish +293. v14's regression was
    # caused by time_penalty interacting with the static r_guid field
    # (policy escaped the r_guid window faster); with use_guide=False
    # in v17/v18/v19 there is no field to escape, so this failure mode
    # does not apply.
    time_penalty: float = 0.05
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
    # v20: re-enable. v15-v19 all collapsed under cold-start with r_guid
    # off. Diagnosis: r_prog and gate_pass_bonus are both zero in
    # expectation under a random Gaussian policy (motion- and event-
    # based signals are zero-mean under random actions), while r_guid
    # is a *position-based* signal that depends on where the drone
    # drifts to, not how it moves. The random-walk distribution does
    # cover non-trivial neighborhoods of the gate axis, so r_guid is
    # the only term with a non-zero expectation under random rollout
    # — i.e. the only term that supplies a discoverable gradient out
    # of the cold-start basin. Re-enabling with the static Liu eq. 6-7
    # field (use_guide_delta_phi=False) and a bumped coefficient (see
    # ``guide_coef`` below).
    use_guide: bool = True
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
    # v20: 0.15 -> 0.5. With r_guid as the load-bearing cold-start signal
    # (see ``use_guide`` above), v11's coef=0.15 produced an attractor
    # weak enough that the policy could happily ignore it (v11 finish
    # was driven by seg-init scaffolding, not by r_guid magnitude).
    # Bumping ~3.3x scales the front-side aperture-alignment bonus up
    # without going as far as v13B's 2.0 (which was paired with the
    # different ΔΦ shaping that lacks the v11/v14 back-side loiter
    # penalty). The back-side penalty also scales, so any policy that
    # tries to harvest progress by orbiting behind the gate pays more.
    # v23: 0.5 -> 1.5. v22's ent_coef_final floor result ruled out
    # over-commitment as the level-3 plateau cause (max_gate=0.80 with
    # ent_floor=0.001 was identical to v21cold's max_gate=0.80 with no
    # floor). Going by per-step magnitude at the level-3 worst-case
    # spawn-to-gate distance (~2.24 m), v22's r_guid was -0.05/step on-
    # axis at ground height — ~5% of r_prog at speed, effectively
    # invisible. Bumping 3x lifts that to -0.16/step (~15% of r_prog),
    # comparable to where it sat on level 2 at typical approach
    # distances. The exit-side loiter penalty also scales 3x, which is
    # the v11-era pathology risk to watch in this run; we'll see it in
    # ep_len if the policy starts fleeing the gate plane laterally on
    # exit.
    # v24: 1.5 -> 0.5 (revert). v23 collapsed: entropy held +2.5 the whole
    # run, ep_len 31, r_prog flipped negative, max_gate stalled at 0.27.
    # The 3x loiter penalty made any off-axis exploration so costly that
    # PPO converged on flailing instead of committing to a trajectory.
    # v24 strategy shifts from "boost r_guid to crack cold-start" to
    # "warm-start from level-2 v21 + seg-init on level 3", which expects
    # the v21 reward landscape (guide_coef=0.5).
    guide_coef: float = 0.5
    # v14: widened 1.5 -> 3.0. The level-0 spawn at world (-1.5, 0.75, 0.01)
    # is ~2.1 m from gate 1 in gate-frame x, so at k0=1.5 the policy gets
    # zero r_guid signal until it has already walked itself most of the
    # way to gate 1. Widening to 3.0 puts the spawn inside the window with
    # ``guide_window**2`` ≈ 0.09 — small but non-zero gradient from step 1.
    # Combined with time_penalty=0.02 this should remove the neutral-zone
    # parking attractor that v11 found.
    # v23: widened 3.0 -> 5.0. Level-3 with full track-regen places gate 0
    # anywhere in a 4x2 m grid (config/level3.toml + envs/randomize.py
    # build_random_track_fn, border_margin=0.5). Worst-case drone-to-
    # gate distance in gate-frame x is ~2.24 m — already inside the
    # k0=3.0 window, but only with guide_window**2 ≈ 0.06, so the
    # gradient is technically present but weak even on-axis. Widening
    # to 5.0 lifts guide_window**2 to ~0.32 at the same point, ~5x more
    # gradient through the spawn region, without changing the field at
    # the gate plane itself.
    # v24: 5.0 -> 3.0 (revert in tandem with the guide_coef revert; see
    # the v24 note above).
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

    v25: single level-3 stage with the full level-3 distribution
    (``gate_rand_scale=1.0``) and **no segment-init**. Designed to be
    paired with ``--init-from`` the v21 level-2 300M checkpoint
    (``level2_seginit_seed0_v21_300M``).

    Why this combination
    --------------------
    v24 added warm-start + seg-init (``segment_init_prob=0.5``) on top of
    the same level-3 layout. The seg-init scaffolding lets half of just-
    reset envs spawn near a downstream gate. Together with the v24
    ``ent_coef_final=0.001`` floor and a constant ent_coef schedule
    (``--ent-coef-start 0.001``), the policy collapsed to entropy = -7.2
    and ep_len = 64 by 300M: it locked onto a single approach that wins
    on those favourable seg-init starts and crashes on every true ground
    spawn. Eval via :mod:`lsy_drone_racing.control.rl_song.eval_sim`
    (with the ``TruePoseObsWrapper`` patch) on level 3 with the v24
    checkpoint produced 0/8 finishes and 0/8 gates passed. The training
    ``finish_rate = 0.0002/step`` (~1.3% per episode) was 100% seg-init
    bias. See :doc:`/memory/project-v24-warmstart-seginit-failed`
    (memory) for the patched-render evidence.

    v25 removes seg-init so the policy cannot lucky-zone-collapse onto a
    narrow start distribution. Combined with the
    :class:`PPOConfig.ent_coef_final` bump 0.001 → 0.005 (see the v25
    note in :class:`PPOConfig`), the schedule now keeps a constant
    moderate entropy bonus end-to-end, which v21 successfully trained
    under for its first ~100M steps. Note progressive randomisation via
    ``gate_rand_scale < 1.0`` is *not* a useful scaffold here: level-3's
    ``track.randomize = true`` regenerates the full XY layout per
    episode regardless, so ``gate_rand_scale`` only controls additional
    wobble *on top of* an already-fully-randomised layout.

    With seg-init = 0 every completed episode is a true ground-spawn
    finish, so the existing ``rollout/finish_rate`` metric is unbiased
    (no need to plumb ``finish_rate_true_start`` through the rollout
    scan for this run).

    History
    -------
    * v24's ``level3_warmstart_seginit`` (segment_init_prob=0.5) is
      preserved in git history (commit ``d4c4fbd``).
    * v22's ``level3_rand0`` (gate_rand_scale=0.0, no seg-init) is at
      commit ``63e0f0c``.
    * The v11–v21 ``level2_seginit`` single-stage curriculum is at
      commit ``9ad7fa0``.
    * The legacy stage1/2/3a/b/c/4 progression remains in
      :func:`_full_curriculum`.
    """
    pi_over_4 = 0.7853981633974483
    return CurriculumConfig(
        stages=(
            CurriculumStage(
                name="level3_warmstart",
                level=3,
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
