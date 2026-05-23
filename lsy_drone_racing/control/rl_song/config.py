"""Static configuration for the Song-2023 RL prototype.

Type-only module: dataclasses holding PPO hyperparameters, the manual curriculum
schedule, the domain-randomization schedule, and the reward weights. No logic
beyond the curriculum-stage factory.

References:
----------
Song, Y. et al. (2023). Reaching the limit in autonomous racing.
    *Science Robotics* 8, eadg1462.
See ``docs/plans/2026-05-13-rl-song-prototype-design.md`` §8–§10.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# v38: policy is sampled in raw 4-vec space: 1 thrust scalar + 3 axis-angle
# scalars (local-tangent ˢτ on SO(3)). The 3-vec is exp-mapped (Rodrigues) to
# a delta rotation ΔR and composed with the current body orientation as
# R_target = R_current @ ΔR before the env-boundary Euler conversion. See
# Schuck et al. 2025, "A Primer on SO(3) Action Representations in Deep RL"
# (arXiv:2510.11103) Tab 2 — ˢτ is the headline-best representation across
# PPO/SAC/TD3 dense and sparse on the pure-rotation benchmark. Replaces the
# v33-v37 6D Zhou-2019 head (two 3-vecs + Gram-Schmidt → global R).
RAW_ACTION_DIM: int = 4
# Env-side action interface is 4-vec [roll, pitch, yaw, thrust].
ENV_ACTION_DIM: int = 4
# v38: per-step rotation budget for the policy's local tangent vector
# (rad). Bounds ‖τ_scaled‖ ≤ ALPHA_MAX so the network output stays inside
# the region where Exp is well-behaved (Schuck 2025 Hypothesis 1) and the
# policy does not need to learn that ‖τ‖>α_max wraps around to the same
# rotation (Hypothesis 5). 0.16 rad ≈ 9.2°/step at 50 Hz ≈ 8 rad/s, which
# matches the Schuck 2025 α_max sweep best (their {0.04, 0.08, 0.16, 0.32}
# headline) and sits inside the firmware's realistic body-rate envelope
# (~10–15 rad/s) for cf21B_500. Pair with init_log_std=-1.5 so the σ·√3
# raw-norm at init stays in the linear regime of the norm-tanh squash.
TANGENT_ALPHA_MAX_RAD: float = 0.16

# Actor obs decomposition (cf. design doc §6). Total 65 floats.
ACTOR_OBS_DRONE_DIM: int = 13  # 6D rot + body-vel + body-omega + z
ACTOR_OBS_GATE_DIM: int = 24  # 2 gates * 4 corners * 3 coords
ACTOR_OBS_VISITED_DIM: int = 2  # visited flags for the 2 future gates
# v45: explicit one-hot encoding of the current target_gate index. v44 video
# evidence showed the policy hovering near gate-1's exit / gate-2's entry
# from a true-ground-start eval (target_gate=0) — a learned attractor at a
# geometric position the training distribution heavily reinforced as
# "high value" via Phase-1 segment_idx=2 hovering. The cyclic-shift trick
# (gates[0] in obs = current target) was the only "which gate is target"
# signal; the explicit one-hot disambiguates the OOD eval state from
# geometrically-similar Phase-1/2 training states. ``target_gate = -1``
# (race finished) maps to all-zeros — the actor output on that step is
# discarded by the rollout buffer.
ACTOR_OBS_TARGET_GATE_DIM: int = 4  # one-hot over n_gates (track has 4 gates)
ACTOR_OBS_PREV_ACTION_DIM: int = ENV_ACTION_DIM
ACTOR_OBS_OBSTACLE_DIM: int = 16  # 4 obstacles * (3 body-frame xyz + 1 visited)
# v35: pre-computed obstacle-danger scalars to short-circuit a cross-channel
# interaction the policy was failing to learn from the raw obstacle channel.
# Layout: [min_clearance_xy_m, closing_speed_to_nearest_obs_mps]. See
# ``obs.build_actor_obs`` for the construction.
ACTOR_OBS_PROXIMITY_DIM: int = 2
ACTOR_OBS_DIM: int = (
    ACTOR_OBS_DRONE_DIM
    + ACTOR_OBS_GATE_DIM
    + ACTOR_OBS_VISITED_DIM
    + ACTOR_OBS_TARGET_GATE_DIM
    + ACTOR_OBS_PREV_ACTION_DIM
    + ACTOR_OBS_OBSTACLE_DIM
    + ACTOR_OBS_PROXIMITY_DIM
)
assert ACTOR_OBS_DIM == 65, "Actor obs layout drifted from design doc §6"


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

    # v40: 4096 -> 16384 (4x) to use the RTX PRO 6000 WS (Blackwell) headroom
    # the v39 run exposed: 73/97 GB VRAM and 12% GPU utilization at n_envs=4096.
    # The 5090 (32 GB) saturated VRAM at 4096 envs; the PRO 6000's 97 GB leaves
    # ~3-4x of unused capacity. Rollout buffer scales linearly with n_envs;
    # batch_size goes 409,600 -> 1,638,400 but minibatch_size stays at 8,192
    # so each PPO update still sees a 8,192-sample minibatch — total gradient
    # steps unchanged (122 iter * 5 epochs * 200 mb = 122,000, same as
    # 488 * 5 * 50 at n_envs=4096). Wall-clock for a 200M-step run should
    # drop near-linearly with envs while the GPU is under-saturated.
    # Side note: the 4096-clone init-symmetry issue called out in v38l
    # (all envs spawning identically at the toml drone state) gets worse
    # in absolute terms at 16384, so ``reset_pos_perturb_m=0.10`` / yaw
    # 0.3 rad noise from v38i is now load-bearing — bump if exploration
    # diversity wandb metrics regress.
    n_envs: int = 16384
    # v9: rollout length 50 → 100 (1 s → 2 s at 50 Hz). With γ=0.997 the
    # effective discount horizon is ~6.9 s; a 1 s rollout forced GAE to lean
    # heavily on the critic bootstrap at the rollout boundary, which both
    # external reviewers flagged as a bias source ("γ horizon is now ~7 s
    # but PPO rollout truncates at 1 s, so GAE is over-relying on critic
    # estimation"). Doubling rollout length lets GAE compute advantages
    # from more on-policy reward and less bootstrapped value, especially
    # important now that the load-bearing reward (finish_bonus=100) only
    # arrives at the end of multi-second episodes.
    # v43 (Codex review): 100 -> 250. Episode is 500 steps (10 s at 50 Hz)
    # and a typical successful lap is 200-300 steps. With n_steps=100, GAE
    # within one rollout never sees both takeoff and finish in the same
    # on-policy trajectory; the finish_bonus (+10) propagates only through
    # the value-bootstrap at the rollout boundary, which is the exact
    # transition where the cold-start critic has historically gone
    # pessimistic and suppressed gate-1 attempts (v38-v42 plateau). 250
    # gives GAE direct line-of-sight from takeoff to ~5 s of trajectory.
    n_steps: int = 250  # 5 s rollout at 50 Hz
    # v40: 50 -> 200 to keep ``minibatch_size`` constant at 8,192 after
    # bumping ``n_envs`` 4096 -> 16384. With the larger rollout buffer
    # (batch_size = 16384 * 100 = 1,638,400), the previous 50 minibatches
    # per epoch would have implied an 8x larger minibatch_size — a real
    # change in PPO optimization dynamics. Keeping minibatch_size fixed
    # and scaling n_minibatches 4x preserves the per-update gradient
    # noise level, so v40 vs v39 is a clean A/B on the reward landscape
    # (exit-waypoint r_prog) without entangled optimizer-hyperparameter
    # changes. Total gradient steps over a 200M-step run remain
    # 122 iter * 5 epochs * 200 mb = 122,000, identical to v39's
    # 488 * 5 * 50 = 122,000.
    # v43: n_steps 100 -> 250 gives batch_size = 16384 * 250 = 4,096,000;
    # keep minibatch_size=8192 so n_minibatches = 500. PPOConfig validator
    # asserts batch_size = n_minibatches * minibatch_size.
    n_minibatches: int = 500  # batch_size / minibatch_size = 4096000 / 8192
    minibatch_size: int = 8192
    # v43 (Codex review): 5 -> 3. 5 epochs over a 4M-sample on-policy
    # batch was overfitting the current distribution and accelerating
    # collapse into the gate-0 attractor — exactly the v38-v42 failure
    # mode. Fewer epochs slows policy commitment per iteration; KL
    # early-stop below provides the within-epoch safety net.
    update_epochs: int = 3
    # v43 (Codex review): 0.997 -> 0.998. With the longer n_steps=250
    # rollouts and finish_bonus arriving only at episode end (up to 10 s),
    # a slightly heavier discount keeps the finish bonus and post-U-turn
    # consequences more relevant in V(s). Effective horizon shifts from
    # ~6.9 s -> ~10.4 s at 50 Hz — roughly matched to the full episode.
    gamma: float = 0.998
    # v43 (Codex review): 0.95 -> 0.97. Pairs with n_steps=250 to reduce
    # bias in long-horizon credit assignment; we accept more variance to
    # propagate finish/crash signals further back without the critic
    # bootstrap dominating.
    gae_lambda: float = 0.97
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
    # v39: 0.005 → 0.01 for the 4D delta-tangent regime. The 7D-vs-4D
    # entropy budget differs by ~43% at the same per-dim log_std
    # (H₇ = 7·1.419 + 7·log_std vs H₄ = 4·1.419 + 4·log_std), so the v25-
    # era ent_coef=0.005 on the 7D head delivered ≈0.032 nats/step of
    # bonus pressure; the same 0.005 on 4D delivers ≈0.018. Bumping to
    # 0.01 restores ≈0.037 of pressure on the 4D head — matched to the
    # v25 7D regime that successfully shipped first-gate-passing. Paired
    # with LOG_STD_MIN=-2.0 in policy.py: the floor guarantees σ ≥ 0.135
    # regardless, this bump controls how fast σ shrinks toward it.
    # v43 (Codex review): 0.01 flat -> 0.02 -> 0.005 anneal. Codex flagged
    # the magnitude (0.01) as acceptable but the *shape* (flat) as wrong:
    # cold-start needs more exploration pressure, late training needs the
    # policy to commit to tight gate traversal. 0.02 doubles the early
    # bonus (~0.073/step at init vs Song-verbatim r_prog ~0.1/step at
    # 5 m/s, so entropy is now load-bearing at init), then anneals via
    # ent_coef_final to 0.005 — small enough for precision flight but
    # nonzero to prevent late-stage lucky-zone collapse.
    ent_coef: float = 0.02
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
    # v29: 0.005 -> 0.001 (revert to v22-era floor). v28 with the
    # constant 0.005 schedule produced a runaway entropy (+13.34 final),
    # which dissolved commitment pressure on the new r_exit_vel reward
    # — the term stayed at -0.0044 throughout training instead of
    # being driven toward zero/positive. v29 reintroduces seg-init
    # with velocity (so lucky-zone collapse via "spawned hovering"
    # is structurally blocked) and pairs it with the lower entropy
    # floor so the policy actually commits to using the seg-init
    # exposure and the exit_vel signal. The finish_rate_true_start
    # metric (added in the same commit as velocity seg-init) is the
    # unbiased indicator that exposes a lucky-zone collapse if it
    # still happens.
    # v38i: 0.001 -> 0.005 (flat schedule, no anneal). v38f-v38h all
    # plateaued at max_gate=1.000 exactly — across 4096 envs * ~100
    # rollout steps * 700 iterations of plateau = ~280M step-env
    # opportunities, no env ever passed gate 1. The 6D Zhou-2019 head
    # was tuned with init_log_std=-0.5 + this anneal schedule and that
    # combination produced enough trajectory diversity to discover gate
    # passes through v32a/v33b. The 4D delta-tangent head has a tighter
    # effective action-noise distribution (raw [-1,1] tanh-bounded then
    # scaled by alpha_max=0.16 rad), so the same anneal collapses
    # exploration before any env reaches the post-gate-1 region.
    # v39: 0.005 → 0.01 (keep flat, match ent_coef). The v38i flat-0.005
    # was still observing collapse to deterministic policy — the per-step
    # entropy bonus on the 4D head at 0.005 (≈0.018 nats/step) is too
    # small relative to the policy-gradient signal. 0.01 restores the
    # effective pressure to ~the v25 7D regime; see ent_coef rationale.
    # v43 (Codex review): 0.01 flat -> 0.005 (anneal endpoint, paired with
    # ent_coef=0.02 above). Linear anneal across total_timesteps so the
    # policy commits late while preserving enough exploration noise to
    # avoid the late-stage lucky-zone collapse documented in v22-v25.
    ent_coef_final: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    # v43 (Codex review): KL early-stop target for the SB3-style per-epoch
    # check in ``train._train_iteration``. Set to 0.0 to disable. With the
    # standard 1.5x multiplier the actual stop threshold is 0.03 mean
    # approx_kl over a full epoch's minibatches. Directly attacks the
    # v38-v42 failure mode where PPO "confidently optimized away from
    # gate-1 attempts" — the KL stop aborts the update before a single
    # iteration's policy step locks in a no-attempt distribution.
    target_kl: float = 0.02
    # v33: bumped default 100M -> 500M to match the v32a-era launch
    # convention (every recent run has overridden this on the CLI). The
    # entropy anneal endpoint and the LR cosine endpoint both read
    # ``total_timesteps``, so a stale default silently mis-anneals if a
    # launch forgets the override.
    total_timesteps: int = 500_000_000
    # Initial log-std for the raw 4-vec Gaussian (T_raw, τ_x, τ_y, τ_z).
    # σ = exp(-0.5) ≈ 0.61. With α_max=0.16 the initial typical per-step
    # rotation is tanh(σ·√3)·α_max ≈ tanh(1.05)·0.16 ≈ 0.125 rad/step
    # (~7°/step at 50 Hz ≈ 6 rad/s body-rate) — aggressive but inside the
    # cf21B_500 firmware's tracking envelope. Prior runs with a lower
    # init_log_std collapsed to a deterministic policy; keeping σ high at
    # init gives PPO headroom before LOG_STD_MIN in policy.py clamps
    # further reduction.
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

    Notes:
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
    # v36: 10.0 -> 1.0. Strip back to Song 2023's literal gate-progress
    # coefficient. v9-v35 used 10× scaling (so r_prog could be matched
    # against scaled crash_penalty / finish_bonus / gate_pass_bonus
    # event signals); v36 also strips those events back to Song's ±10
    # values, so the 10× scaling no longer serves a purpose. Per-step
    # r_prog at 5 m/s is now ~0.1, vs finish_bonus +10 and crash -10 —
    # the same proportions Song trained under.
    # v38: 1.0 -> 10.0. v36's stripped reward cold-train collapsed at
    # 500M (0/32 gates, see ``default_curriculum`` v37b history note).
    # Diagnosis: under v36 magnitudes, the per-step r_prog at 5 m/s
    # (~0.1) is dwarfed by the variance from random Gaussian exploration
    # noise, and ``r_terminal=-10`` arrives infrequently and discounted —
    # there is no per-step f(p_t) gradient large enough to bootstrap
    # gate-finding from random init. Restoring the v9-v33 10× scaling
    # (and the matching event-reward magnitudes below) gives PPO a
    # 1+ reward / step signal at racing speed, which is the v7a-style
    # cold-start recipe that demonstrably broke level 1 to 100% finish
    # in 100M steps. The new delta-tangent action head cannot warm-start
    # from any v33-v37 checkpoint (head shape changed) so the cold-start
    # gradient must be load-bearing again.
    # v43: 10.0 -> 1.0. Song-verbatim revert. The 10× scale-up was
    # internal balancing against jackpot + guide + obstacle add-ons, all
    # of which v43 strips. With those gone, Song's literal value restores
    # the documented event:r_prog ratio (finish +10 vs integrated r_prog
    # ≈ +6.6 over a clean lap on the 6.61 m level-1 path).
    # v47: 1.0 -> 10.0. Bare Song magnitudes left a finished lap paying
    # only -6 net at v46's anti-hover-augmented reward (r_prog +6.6 +
    # finish +10 - time -10 - guide -10 - omega -2). Crash dominated,
    # producing a "commit-and-crash" attractor instead of a finish-
    # incentivised one. Restoring v38's 10× scaling: integrated r_prog
    # ≈ +66 over a clean lap puts finish at +100 net (with gate_pass
    # bonus +100 below) vs crash -15, restoring the v42 positive
    # gradient that produced 100% L0 gate-0 traversal.
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
    # v38k: pre-gate-0 entry-position shaping. v38f-v38i all plateaued at
    # ``max_gate=1.000`` exactly across 280M+ step-env opportunities.
    # Geometry analysis (2026-05-22 codex consult): under the post-
    # 2025-10-20 ``Update gate orientation and pass check`` commit
    # (``23415dc``), gates are crossed along their local +x axis; gate
    # 0's yaw=-0.78 points local +x in world (+0.71, -0.70), gate 1's
    # yaw=2.35 points local +x in world (-0.70, +0.71). The track has
    # a U-turn between gate 0 and gate 1: the drone must overshoot
    # gate 0 in +x, decelerate, then re-cross gate 1's plane in the
    # (-x, +y) direction (last_x_local_g1 < 0 -> > 0). v38f-v38j's
    # natural +x post-gate-0 coast lands the drone on x_local_g1 < 0
    # immediately and OOBs at x=2.5 still on x_local_g1 < 0 — gate 1
    # is geometrically unreachable from that trajectory shape.
    # v38j's mistake: rewarded distance-closing to gate 1's *center*,
    # which is naturally closed as the drone coasts +x past gate 0.
    # Look-ahead paid +~0.3/step for the *wrong* behavior.
    # v38k: reward distance-closing to a virtual entry waypoint at
    # ``gate1_pos - lookahead_entry_offset_m * gate1_local_x_in_world``,
    # i.e., 0.5 m back from gate 1 on its -x_local side. With gate 1
    # yaw=2.35 this puts the waypoint at world ~(1.40, 0.39). Approach
    # from spawn (-1.5, 0.75) toward this waypoint biases the policy
    # to enter gate 0 with a +x velocity vector that *naturally feeds
    # into the U-turn approach geometry* — gate 0 exit aligned toward
    # (+1.40, +0.39) instead of straight +x lands the drone with
    # momentum that the base r_prog (distance-closing to gate 1
    # center) can then complete the U-turn from.
    # Masked to ``target_idx == 0`` only (post-gate-0, base r_prog
    # handles gate 1). 0.0 disables; preserves v38i. Coefficient
    # reduced 0.3 -> 0.2 vs v38j to keep per-step magnitude
    # commensurate with ``obstacle_weight=0.6 * sigma=0.5`` barrier
    # the policy was tuned around. Codex pre-launch critique
    # 2026-05-22 flagged that >0.2 risks the entry-side bias
    # dominating the obstacle gradient on the gate-0 approach.
    # v42: 0.2 -> 0.0 (disable). v41b rendered videos surfaced a
    # gate-frame-clip failure mode: with r_prog (toward gate 0 center)
    # and r_lookahead (toward gate 1 entry waypoint at (1.40, 0.40, 1.2))
    # both as distance-delta terms, their gradients dominate at different
    # moments along the gate-0 approach. Far from gate 0 r_prog dominates
    # 5:1 and the drone tracks gate 0. AT the gate-0 plane, r_prog's
    # gradient flattens (distance-to-gate-0 at local minimum) and r_lookahead
    # remains gradient-active — pulling +y (toward gate 1's entry at
    # y=0.40 vs gate 0's center at y=0.25). The policy banks left right
    # as it crosses gate 0's plane and clips the gate's left post.
    # Disabling collapses the trade-off: gate-0 reliability should jump
    # toward 100% on level 1, but the policy loses the v38k mechanism
    # for setting up the U-turn approach to gate 1. v41b's other v41
    # changes (Phase 2 replay at p=0.10, seg-init at gate's +x_local
    # entry waypoint) still train the post-gate-1 value function and
    # provide on-rails gate-1-approach experience, so the post-gate-0
    # learning signal is not lost — just the at-the-plane bias is.
    lookahead_coef: float = 0.0
    # v38k: distance behind ``target_idx + 1`` along its -x_local axis
    # for the entry waypoint. With gate 1 nominal x=1.05 and randomization
    # +/-0.15 m, the offset places the waypoint at world x in
    # [1.25, 1.55] — well inside the OOB envelope (pos_limit_high[0]=2.5).
    # Range 0.3-0.8 m: smaller is closer to gate-center shaping
    # (degenerates back to v38j), larger pulls the drone further past
    # gate 0 before any reward gradient flips sign.
    lookahead_entry_offset_m: float = 0.5
    # v15: down to 0.01 to match Song 2023's exact body-rate coefficient.
    # The 0.02 value here was justified earlier as the 50 Hz analogue of
    # Song's 100 Hz 0.01, but Song 2023 quotes b = 0.01 without specifying
    # control frequency and the 100 Hz figure was a misreading. Reverting
    # to the verbatim paper value.
    # v33: 0.01 -> 0.003. v32a evals showed the drone flying slow, and
    # the per-step magnitude of r_omega = 0.01 * |ang_vel|_L1 at racing
    # body rates (~10 rad/s per axis -> L1 ≈ 30) is -0.3/step, which
    # over a 500-step episode is -150 — larger than ``finish_bonus``
    # and comparable to the entire gate jackpot total. The policy was
    # structurally biased toward gentle slow turns. 0.003 keeps the
    # penalty against thrashing/oscillation (still -0.09/step at 30
    # rad/s, integrates to -45 over 500 steps, ~half the finish bonus)
    # but no longer dominates the speed economics. Song's b=0.01 was
    # justified for the lighter quad in his paper; our cf21B at this
    # control rate hits higher peak rates routinely.
    # v36: 0.003 -> 0.01. Codex pre-launch review flagged 0.02 (literal
    # Song-at-50Hz) as too punitive at the v36 reward scale: with the
    # L1 norm in reward.py at L1(ω) ≈ 5 (modest cornering) the per-step
    # omega penalty is -0.1, exactly cancelling r_prog at 5 m/s; at
    # L1(ω) ≈ 10 (aggressive racing) the penalty dwarfs the +10 finish.
    # 0.01 keeps the term active as a smoothness prior (integrated
    # -3 to -5 / episode at racing rates, comparable to +6 r_prog over
    # a clean lap) without cancelling forward motion. v33's 0.003 was
    # effectively zero (-1 / episode in eval), too weak to discourage
    # thrashing.
    # v38: keep at 0.01. With the v38 progress_coef=10 the per-step
    # r_prog at 5 m/s is ~1.0 and the per-step r_omega at L1(ω)=10 is
    # ~0.1 — a 10× ratio that preserves a meaningful smoothness prior
    # without inverting the forward-motion incentive.
    # v43: 0.01 -> 0.02. Song-verbatim at our 50 Hz step rate. Song's
    # literal ``b = 0.01`` is at 100 Hz; the per-second body-rate budget
    # is preserved by doubling the coefficient when halving the step
    # rate. v36's codex-flagged "too punitive at L1(ω)=5" concern was
    # against the L1 norm + v36 reward scale; with v43 reverting to L2
    # norm (see reward.py) and Song's progress_coef=1, the per-step
    # r_omega at ‖ω‖₂ ≈ 3 rad/s is ~-0.06 vs r_prog ~0.1 — within Song's
    # own balance.
    omega_coef: float = 0.02
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
    # v33: 15 -> 100. v32a banks gate jackpots (40 / 80 / 120 / 160) and
    # the finish bonus (100) on the way to a finish, and v33 adds another
    # +50 per gate from the bumped ``exit_vel_coef``. A clean crash after
    # gates 1+2 with v33's exit-velocity term pays 40 + 50 + 80 + 50 = 220
    # before crash; with the old crash_penalty=15 the crash netted +205
    # (strictly more than just standing still), so risky terminal lines
    # were the dominant policy. crash_penalty=100 brings the same crash
    # to net +120 — still positive, but a clean continuation banking the
    # next gate (+120 + ~50 + further events) dominates. crash_penalty=50
    # (an earlier v33 draft) still netted +170 after two gates and was
    # caught as too weak in the codex pre-launch review.
    # v36: 100.0 -> 10.0. Strip back to Song 2023's literal crash penalty.
    # v33 had bumped to 100 because the v33 gate_pass + exit_vel events
    # totaled ~+200 over a successful trajectory and needed a matching-
    # magnitude crash penalty to make bank-and-crash unprofitable. With
    # those event bonuses removed in v36, +10 finish vs -10 crash matches
    # Song's proportions and Song's value-function dynamics.
    # v38: 10.0 -> 100.0. Restored alongside progress_coef=10 and the
    # event bonuses below. Same magnitude as v33-v35; deters bank-and-
    # crash now that a successful trajectory banks ~+200 in event reward.
    # v38c: 100.0 -> 15.0. v38a/v38b iter-250 traces showed a "hit gate 1,
    # drift" attractor: under crash_penalty=100, the expected return on
    # a gate-2 attempt (positive r_prog + jackpot only on success vs
    # -100 + partial r_prog on a typical mid-track crash) was net-
    # negative compared with the +15 reward of touching gate 1 and
    # timing out. Policy correctly learned to STOP TRYING after gate 1.
    # Dropping to 15 (the v19 cold-start-breaking value, used before
    # v33's exit-velocity bonus banked an extra +50/gate) restores
    # positive expected return on gate-attempt under modest success
    # probability — see v18-v19 history above. We do NOT have v33's
    # +50/gate exit-vel signal active (use_exit_vel_bonus=False in v38),
    # so v33's reason for crash_penalty=100 (matching the larger banked
    # event reward) doesn't apply at v38 magnitudes.
    # v43: 15.0 -> 10.0. Song-verbatim. v38c's 15 was a v19 hold-over
    # paired with time_penalty=0.05 and gate_pass_bonus=20-scaled. v43
    # drops all three add-ons; Song's literal ±10 is the matched pair
    # that holds the finish:crash ratio at 1:1.
    # v47: 10.0 -> 15.0. v38c value, paired with v47 progress_coef=10 and
    # finish_bonus=100 below. With v42's full reward stack, integrated
    # event reward over a clean lap is ~+200 (gate jackpot scaled
    # 10+20+30+40=100 + finish 100), so a 15-point crash penalty is
    # deterrent without making crash-trying strictly worse than do-
    # nothing (which time_penalty=0.05 already handles at -25/episode).
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
    # v36: 100.0 -> 10.0. Strip back to Song 2023's literal finish bonus.
    # Matched 1:1 with crash_penalty above.
    # v38: 10.0 -> 100.0. Mirrors crash_penalty restoration.
    # v43: 100.0 -> 10.0. Song-verbatim, matched 1:1 with crash_penalty.
    # v38's 100 was a 10× scale-up to match the v18 jackpot magnitude
    # (+200 integrated event reward); with jackpot stripped in v43, +10
    # is the only consistent value.
    # v47: 10.0 -> 100.0. With progress_coef=10 and integrated r_prog ≈
    # +66 over a clean lap, finish_bonus needs to be on the same order
    # for the value function to weight the lap-finish event meaningfully.
    # 100 was v38/v42's value and pairs with the jackpot scaling below
    # (1+2+3+4 × 10 = +100 event reward for completing the four gates,
    # plus +100 finish = +200 event total, dominating the dense -10
    # time + -10 guide cost).
    finish_bonus: float = 100.0
    # Obstacle soft barrier: -w_obs * sum_i exp(-||p - p_obstacle_i||^2 / sigma^2)
    # v32: bump sigma from 0.2 → 0.3 m so the penalty has a meaningful
    # avoidance gradient at safe-but-close distances (old 0.2 m gave
    # ~exp(-6.25) = 0.002 at 0.5 m, no learning signal until contact).
    # weight stays at 0.2: per-step worst case ~4 obstacles × 1.0 × 0.2
    # = -0.8/step, manageable cumulative over a 100-step episode.
    # v33: 0.2 -> 0.8. v32a evals showed the drone still grazes obstacle
    # capsules. With weight=0.2, the per-step penalty for being right
    # next to a single obstacle was -0.2 — small enough that PPO would
    # absorb it for any nominal progress gain. At 0.8 the same single-
    # obstacle close-approach is -0.8/step, comparable to a step's
    # ``r_prog`` at racing speed (so PPO actually has to route around).
    # Codex's pre-launch review suggested bumping above the 0.6 draft
    # because the realistic case is "one obstacle close", not "all 4
    # obstacles touching", so the per-step magnitude needs to scale to
    # the realistic worst case, not the cumulative bound.
    # v38f: 0.8 -> 0.0. The v38a-v38e cold-train series (level 1 + level 2,
    # five runs) all failed deterministic eval at 0/20 finishes; the
    # 2026-05-20 level-2 handoff diagnosis isolates the secondary cause as
    # this barrier + the gate-frame barrier biasing the cold-start policy
    # toward conservative trajectories that never solve takeoff -> gate 1.
    # v7a (the only recipe that cleared level 1 to 100 % deterministic
    # finish in 100M steps) had no obstacle barrier; the term was a v33
    # addition for level-3 obstacle avoidance, which is not what stage 1
    # exercises. Disabled for v38f cold-train; restore for the level-2
    # warm-start stage if obstacle proximity becomes a failure mode.
    # v38g: 0.0 -> 0.2. v38f trace analysis on deterministic level 0
    # showed all 20 episodes pass gate 0 cleanly (a v38a-v38e first), but
    # the trained policy exits gate 0 at vel ~(5.7, -0.75, 2.1) i.e.
    # mostly +x rather than the gate-normal direction (~-45 deg, gate 0
    # yaw=-0.78 rad). 80 ms past gate 0 the drone is at XY (0.99, 0.19),
    # 6 cm from obstacle 1's vertical pole at (1.0, 0.25). Reward at the
    # terminal step is exactly the crash_penalty (-15) - confirmed
    # obstacle hit, not OOB / timeout. v38g restores the v32-era mild
    # obstacle barrier (0.2, half of v33's 0.8) so PPO gets a per-step
    # gradient about obstacle proximity without re-introducing v38e's
    # 'touch gate 1, die' barrier-too-strong trap. obstacle_sigma stays
    # at 0.5 (v34 widening, ~1.0 m felt zone). See
    # ``project_v38f_obstacle1_hit`` memory for the full trace evidence.
    # v38h: 0.2 -> 0.6. v38g eval: same level0 20/20 gate-0 passes as
    # v38f but level1 5->15, level2 2->7 (3x more robust under
    # randomization). However, deterministic trajectory at w=0.2 still
    # hits obstacle 1 — diagnostic computed r_obs profile along the
    # v38g trace and found w=0.2 produces -0.19/step at peak obstacle-1
    # proximity (d=0.10 m), well below r_prog=+0.42/step. The smallest
    # weight whose per-step danger-zone penalty meaningfully exceeds
    # r_prog is w=0.6 (-0.58/step, ~1.4x r_prog), which forces PPO to
    # find an evasive trajectory rather than treating obstacle 1 as
    # acceptable cost-of-doing-business for the +20 gate-0 jackpot.
    # w=0.8 risks discouraging gate-0 attempts (episode total -7.8 vs
    # gate jackpot +20). w=0.4 is at parity with r_prog (-0.39 vs
    # +0.42), marginal. Per Codex review 2026-05-22, gate_frame_weight
    # is the wrong knob for the "drone passes gate sideways"
    # pathology (point-distance barrier, not perpendicular-pass term);
    # see ``feedback_gate_frame_is_point_barrier`` memory.
    # v43: 0.6 -> 0.0. Song-verbatim has no obstacle term. Note: L2 has
    # vertical-pole obstacles that Song's tracks did not, so the policy
    # learns avoidance only through the -10 crash penalty without this
    # term — this is exactly the test of whether Song's minimum recipe
    # generalizes to a track with obstacles.
    obstacle_weight: float = 0.0
    # v34: 0.3 -> 0.5. v33b eval traces showed the policy was effectively
    # ignoring the obstacle channel: in 3/8 eval episodes the drone flew a
    # near-straight path from spawn to gate 0 and hit obstacle 0 within
    # 0.84 s of takeoff, even though obstacle 0 sat <0.11 m off the straight
    # line in placed-pose. With sigma=0.3 the Gaussian barrier is below
    # -0.05/step at d>=0.5 m, giving only ~0.4 s of felt gradient at 1.7 m/s
    # cruise — too short to learn a detour against the continuous r_prog
    # pull. sigma=0.5 keeps the same peak penalty (-0.8 at d=0) but extends
    # the felt zone to ~1.0 m, giving ~0.6 s of advance signal so the
    # policy has enough lead time to curve around. Side effect: 24% of
    # gate-obstacle XY pairs are within 0.7 m, so passing through a
    # near-obstacle gate now incurs ~-8 obstacle penalty over ~10 steps —
    # still well under the +50 gate bonus, but actively biases the policy
    # toward gate-passing lines that maximize lateral obstacle clearance.
    obstacle_sigma: float = 0.5  # m
    # v32: Gate-frame soft barrier. Penalize proximity to gate frame
    # edges (4 line segments per gate, connecting the opening corners
    # from ``obs.GATE_HALF_SIZE_M``). The barrier extends ~sigma into
    # the passage; with passage half-width 0.20 m and sigma=0.08 m,
    # passage center gives exp(-(0.20/0.08)^2) ≈ 0.0019 (negligible),
    # while approaching the rim at d=0.10 m gives exp(-(0.10/0.08)^2)
    # ≈ 0.21 (strong gradient). Applied to all gates equally — the
    # drone wants to fly through the passage center, not graze any
    # frame.
    # v33: 0.2 -> 1.2 (same rationale as obstacle_weight bump). v32a's
    # gate-frame hits show the 0.2 weight was insufficient — at d=0.10 m
    # from the nearest edge the per-step sum across the 4 edges of one
    # gate is ≈ 0.21 × 0.2 ≈ 0.042 (codex's careful re-estimate; my
    # earlier 0.67 number assumed all 4 edges sit at 10 cm
    # simultaneously, which is geometry-impossible for the square
    # opening). 1.2× lifts the same close-graze to -0.25/step, and a
    # full second of grazing (~50 steps) costs -12.6 — outweighs the
    # marginal r_prog gain from a tighter line.
    # v36: 1.2 -> 0.0. Strip the gate-frame Gaussian barrier entirely. Song
    # 2023 has no equivalent term and explicitly relies on the value
    # function learning that grazing-near-frame states have low value
    # because they lead to crashes. Keeping the barrier was over-pricing
    # tight gate-pass lines and pushing the policy onto wide approaches
    # that brought it closer to obstacles next to the gates. Removed.
    # The constant ``gate_frame_sigma`` is left at 0.08 in case a future
    # version re-enables the term, but with weight=0 the barrier
    # contribution is zero regardless of sigma.
    # v38: 0.0 -> 1.2. Re-enabled at the v33 magnitude. With the v38
    # restoration of the cold-start jackpot + guide scaffold, the
    # gate-frame barrier prevents the "graze the rim then bank the
    # gate-pass jackpot" attractor; v33b/v34 evals showed the policy
    # still grazes frames under barrier=0.8, and v36's barrier=0 left
    # gate-edge collisions un-priced. 1.2 keeps the per-step penalty at
    # ~-0.25/step near a 10-cm edge graze (~50 steps of grazing costs
    # -12, well under the gate-bonus magnitude so the gradient is
    # informative, not prohibitive).
    # v38f: 1.2 -> 0.0. Same diagnosis as ``obstacle_weight`` above —
    # the gate-frame Gaussian barrier was a v32/v33 level-3 collision-
    # avoidance addition that biases the cold-start policy toward wide
    # approaches and prevents the v7a-style "fly through the aperture"
    # commitment behaviour. v7a's recipe had no gate-frame barrier and
    # solved level 1 to 100 % deterministic finish in 100M steps.
    # Disabled for v38f cold-train.
    gate_frame_weight: float = 0.0
    gate_frame_sigma: float = 0.08  # m
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
    # v38d: 40.0 -> 20.0. v38c stage-2 iter-500 traces showed a plateau
    # at ~15% finish on level 2 with bimodal episode lengths (24% finish
    # at ~3 s + 76% crash at ~0.8 s). Diagnosis: at gate_pass_bonus=40
    # scaled, total event reward over a successful lap is +500 (gates
    # 40+80+120+160 + finish 100) vs ~+170 integrated r_prog — event
    # reward dominates 3:1 and incentivises "rush past gate, sacrifice
    # precision, gamble on the jackpot". Halving to 20 (= v18/v7a's
    # exact value) makes event reward +200 + +100 = +300 vs +170 prog —
    # closer to balanced. Per-step r_prog gradient becomes load-bearing
    # so the policy is pulled toward smooth flight, not gate-banking.
    # v39: 20.0 -> 10.0. v38d's reference "+170 integrated r_prog" was
    # measured on a longer track. Re-integrating against the level-1
    # layout (path spawn->g0->g1->g2->g3 = 6.61 m, progress_coef=10)
    # gives integrated r_prog ≈ +66. With gate_pass_bonus=20 scaled
    # 1x..4x = +200 plus finish_bonus +100, event:r_prog = +300/+66 =
    # 4.5x — well above v38d's own 3x "rush-and-crash" threshold. The
    # actual v38f-v38l plateau (max_gate=1.000 exactly across ~280M
    # step-env opportunities, no env ever passing gate 1) is the failure
    # mode v38d was trying to prevent: bank gate-0 jackpot, attempt
    # gate-1 high-variance, crash beats commit. Cutting base to 10
    # (gates pay 10/20/30/40 = +100 scaled) brings event:r_prog to
    # +200/+66 = 3.0x while preserving v38's "jackpot scaffold for the
    # 4D head's cold-start regime" rationale (cold-train needs sparse
    # discrete signal that v36's no-jackpot run lacked, just not at
    # 2x dominance over the dense gradient that has to learn the
    # post-gate-0 U-turn geometry). Keep ``scale_gate_bonus_by_index``
    # on; codex review 2026-05-22 recommended halving the base over
    # disabling the scaling, since the discovery gradient at gate 1
    # (the actual plateau) needs to remain stronger than at gate 0.
    gate_pass_bonus: float = 10.0
    # v36: True -> False. Strip the per-gate jackpot. Song 2023 has no
    # gate-pass event bonus and explicitly relies on the gate-progress
    # term alone to drive gate-by-gate behaviour (once a gate is passed,
    # the target switches, so the next step's progress is suddenly large
    # — that's the implicit "bonus"). The v18 jackpot was a v15-v17
    # cold-start scaffold; we don't need it once the value function has
    # learned the gate sequence.
    # v38: False -> True. The v36 cold-train collapse (0/32 gates @ 500M)
    # is exactly the failure mode the jackpot scaffold was designed to
    # prevent. With the new delta-tangent action head we have no working
    # policy to warm-start from, so the cold-start gradient must come
    # back via the jackpot.
    # v43: True -> False. Song-verbatim has no jackpot. The v18 cold-
    # start scaffold rationale (random policy never finds finish_bonus)
    # is real but unverified under the current 4D delta-tangent action
    # head + bare L2 curriculum; v43 is the clean test of whether the
    # gate-progress gradient alone bootstraps under those conditions.
    # v47: False -> True. Per-gate event reward (scaled 10/20/30/40 with
    # scale_gate_bonus_by_index below) — the v18 cold-start scaffold
    # that v36's no-jackpot run lacked, the v37b/v42 history block
    # identified as load-bearing. With v46 anti-hover pressure now in
    # place, the jackpot provides the *positive* gradient toward
    # gate-passing that bare Song's distance-delta r_prog (which
    # plateaus at the gate plane) doesn't.
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
    # v28: exit-velocity bonus at gate-pass. v25/v26 level-3 evals (8-episode
    # patched renders) showed a hover-then-shoot trajectory pattern: the
    # policy brakes to ~zero velocity at every gate, then accelerates from
    # rest to the next gate. Greedy ``r_prog`` rewards distance reduction to
    # the *current* target only, so there is no signal that "arrive at gate
    # N with velocity aimed at gate N+1" is preferred to "arrive at zero
    # velocity". This term supplies that signal sparsely: at the moment a
    # gate is passed, reward ``exit_vel_coef * (v · unit(next_gate_pos -
    # pos))``. Signed, so it also punishes crossing a gate while moving
    # backward (toward gate N-1). Disabled on the finish step (target_gate
    # = -1, no "next" gate). Same triggering cadence as ``r_gate_bonus`` to
    # avoid interfering with the dense r_prog gradient. The term is
    # subordinate by design: at 3 m/s aligned velocity per pass, contributes
    # +6 vs gate_pass_bonus's +40-160. Does *not* penalize any maneuver —
    # extreme attitudes (e.g. v7's reverse-into-gate-4 trick) that preserve
    # exit-velocity-toward-next-gate get a larger bonus, not a smaller one.
    # v33: 2.0 -> 10.0 with ``v_to_next`` clamped to ``±exit_vel_clip``
    # in ``reward.step_reward``. v32a's value paid 4-15% of the per-gate
    # jackpot at a 3 m/s aligned exit; PPO routinely ignored it. 10×
    # makes a clean 5 m/s aligned exit pay ``10 * 5 = +50`` per gate,
    # comparable to the gate-1 jackpot itself. The clip prevents
    # arbitrary outliers (e.g. brief 10+ m/s body-frame artifacts after
    # a hard pitch) from minting one-shot rewards that destabilize the
    # value function.
    # v36: True -> False. Strip the exit-velocity bonus. Song 2023 does
    # not use one; v33's bump to coef=10 + clip 5 m/s never produced a
    # net-positive integrated value in eval (averaged -3 / episode in
    # v33b, -2.3 / episode in v34, -8.6 / episode in v35) — the policy
    # was passing gates at velocities that didn't reliably point at the
    # next gate, so the term was a small *negative* on average. Removed.
    # v38: keep at False. v33-v35 evidence is that this term is a small
    # *negative* on average — the policy converges faster without it,
    # and we want fast iteration on the new action head.
    use_exit_vel_bonus: bool = False
    exit_vel_coef: float = 10.0
    # Hard cap on the velocity scalar fed into the exit-velocity bonus.
    # See ``reward.step_reward``: ``r_exit_vel = exit_vel_coef *
    # clip(v · dir_to_next, -exit_vel_clip, exit_vel_clip)``.
    exit_vel_clip_mps: float = 5.0
    # v64: Caution shaping for randomized-track regime (level 3 or any
    # stage with gate_rand_scale > 0). Penalizes high speed near the
    # current target gate when the gate is NOT YET sensor-visible. The
    # racing env masks ``env_obs["gates_pos"]`` to the nominal/placed
    # location until the gate enters ``sensor_range`` (= 0.7 m on level
    # 3 toml); after that the masked obs matches the true wobbled gate.
    # Without this shaping, the policy on a randomized track flies at
    # ~5 m/s toward the nominal location, gets only ~0.14 s (7 steps at
    # 50 Hz) of reaction time after sensor reveal, and crashes into the
    # actually-positioned gate frame. r_caution = -caution_coef *
    # ||vel|| * exp(-((dist - peak_m)/kernel_m)^2) * visibility_factor,
    # where visibility_factor is 1.0 when ||masked - true|| > threshold
    # (gate not yet revealed) and ``caution_visible_factor`` otherwise.
    # On deterministic tracks ||masked - true|| = 0 always, so with the
    # default ``caution_visible_factor = 0`` the term is dormant on L0/
    # L1/L2. Off by default (opt-in via --use-caution).
    use_caution: bool = False
    caution_coef: float = 0.05
    # Peak penalty distance (m) — the kernel exp(-((dist-peak)/width)^2)
    # is maximal at dist = peak_m. Defaults to 1.0 m which is just outside
    # the level-3 sensor_range so the penalty incentivises braking before
    # the reveal point.
    caution_peak_m: float = 1.0
    # Gaussian width (m) of the caution kernel. Tapers to negligible
    # beyond ~2.5 × kernel_m from the peak.
    caution_kernel_m: float = 0.8
    # Threshold (m) on ||masked_target_pos - true_target_pos||; below
    # this the gate is treated as sensor-visible. Default 0.005 m
    # (5 mm) is conservatively below any meaningful wobble.
    caution_visible_threshold_m: float = 0.005
    # Penalty multiplier when the gate is sensor-visible. Default 0.0
    # means caution turns off completely once the sensor reveals the
    # true position, so the policy can commit aggressively in that
    # final window. Raise to 0.2-0.5 if late-window aggression is too
    # much.
    caution_visible_factor: float = 0.0
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
    # v33: 0.05 -> 0.10. v32a achieves first-lap finishes but at slow
    # lap times. With time_penalty=0.05, the gap between a 200-step
    # fast lap and a 500-step slow lap is only 0.05 × 300 = 15 reward,
    # which is rounding error next to the ~600 of event-based reward.
    # Doubling to 0.10 widens the fast-vs-slow gap to 30 — meaningful
    # against the +50 v33 exit-velocity bonus per gate. The do-nothing
    # baseline tightens too (was -25 per 500-step timeout, now -50),
    # but the v33 crash_penalty=50 still makes any crash strictly
    # worse than do-nothing, so policy escape from a hover attractor
    # remains via forward motion, not suicide.
    # v36: 0.15 -> 0.0. Strip the explicit time penalty. Song 2023 does
    # not use one and relies on the finish_bonus (+10) plus the discount
    # factor γ=0.997 to bias the policy toward faster laps (earlier
    # finish = less discounted bonus). At γ=0.997 a 200-step lap pays
    # 10·γ^200 ≈ 5.49 while a 500-step lap pays 10·γ^500 ≈ 2.23 — a
    # 3.26 reward differential, much larger than the v34 time_penalty's
    # 0.15·300 = 45 *only if the policy actually finishes*. Without a
    # time penalty, the do-nothing baseline is 0 instead of -75; combined
    # with crash_penalty=-10 the policy still has a clear bias toward
    # forward motion (any positive r_prog) over standing still.
    # v38: 0.0 -> 0.05. With the v38 progress_coef=10 the per-step r_prog
    # at 5 m/s is ~1.0 and the do-nothing baseline is ~0 → no per-step
    # cost to hovering, which is exactly the v15-v19 failure mode. 0.05
    # was the v19 cold-start-breaking value and still leaves a 200-step
    # fast lap at +10 net (vs +400 r_prog total) — small relative to
    # the event reward but enough to make hover-timeout strictly worse
    # than do-nothing.
    # v43: 0.05 -> 0.0. Song-verbatim has no per-step time penalty.
    # Song relies on γ-discounted finish_bonus to drive faster laps:
    # at γ=0.997, a 200-step lap pays 10·γ^200 ≈ 5.5 vs a 500-step
    # lap's 10·γ^500 ≈ 2.2 — a 3.3 differential that biases toward
    # speed without an explicit per-step cost.
    # v46: 0.0 -> 0.05. Re-introduce v8/v19 anti-hover pressure. Bare-Song
    # reward at v43-v45 produced a stable "hover above gate 0" attractor
    # — render evidence (v45 L0 renders) showed the policy taking off,
    # climbing past gate 0's altitude, drifting laterally past gate 0
    # without crossing the aperture, and hovering there. r_prog plateaus
    # at the gate plane (distance(g_K, pos) ~ 0 once close), so without
    # time_penalty hovering near gate K pays ~0 per step (better than
    # crashing through). Per-step 0.05 makes a 500-step timeout cost -25;
    # hovering's per-episode return drops from ~-5 to ~-30, strictly
    # worse than even a crash (-10). Policy is now incentivised to either
    # finish (+16.6 net at progress_coef=1.0) or attempt-and-crash (-10).
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
    # v36: True -> False. The Liu loiter penalty was a v15-v17 cold-start
    # scaffold that supplied a position-dependent gradient when r_prog
    # alone could not escape the do-nothing basin. v36 strips back to
    # Song 2023's pure gate-progress reward, which does not have a
    # spatial guidance term. Cold-start risk is mitigated instead by
    # segment_init (mid-track spawn) and the Phase 2 buffer (past gate-
    # pass replay). The codex pre-launch review caught that this default
    # was still True after v33-v35; under the v36 stripped reward, an
    # active r_guid would dominate the per-step magnitude and totally
    # undermine the "strip to Song" hypothesis.
    # v38: False -> True. The seg-init + Phase 2 buffer mitigations
    # explicitly failed for v36 cold-train (0/32 gates @ 500M), confirming
    # the v20 diagnosis that r_guid is the only term with a non-zero
    # expectation under a random Gaussian policy. The position-dependent
    # gradient is load-bearing for the random-walk distribution to
    # discover the gate-axis direction. Coefficient kept at v21/v24's
    # guide_coef=0.5 (the validated working value).
    # v43: True -> False. Song-verbatim has no spatial guidance term.
    # The v38 re-enable rationale ("r_guid is the only term with a non-
    # zero expectation under a random Gaussian policy") was tied to the
    # 7-vec action head; the current 4D delta-tangent head's exploration
    # distribution is different (per the v42 wandb traces) and the
    # guide-field gradient may instead be the source of the gate-1
    # U-turn pathology by biasing entry geometry. v43 tests Song's
    # claim that the value function alone discovers safe-state regions.
    # v46: False -> True. Re-introduce the v10/v11/v13A static Liu loiter
    # field (use_guide_delta_phi=False below, so the original Liu eq. 6-7
    # formulation, not the Δ-potential variant). Combined with the v46
    # time_penalty re-add above, this attacks the v45 "fly over gate 0
    # and hover" failure mode at two layers: time_penalty makes ANY
    # hovering strictly negative-value (global anti-hover); r_guid adds
    # a position-dependent loiter penalty SPECIFICALLY in the gate-plane
    # vicinity that's larger when off-axis from the traversal line. The
    # combination biases the policy toward axial approach + commit-through
    # rather than over-fly-and-hover. guide_coef=0.5 is the v21-validated
    # working magnitude; field shape uses default guide_k0=3.0, k1=1.0,
    # k2=0.3.
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
    # loaded from ``config/levelN.toml`` and forwarded to the framework via
    # :meth:`RLSongVecEnv._stage_randomizations`. A value of 1.0 uses the
    # full level-3 randomization budget (±0.15 m on gate_pos / obstacle_pos
    # and ±0.05 / ±0.1 / ±0.2 rad on gate_rpy). Smaller values produce an
    # easier near-fixed-track regime so the policy can first learn
    # approach-to-nominal before adapting to noise; values below 1.0 are
    # intended for the ``stage3a/b/c`` warm-up sub-stages. Has no effect on
    # stages with ``level != 3`` (level 1 has no gate/obstacle randomization
    # to scale).
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
    # v29: velocity-aware seg-init speed. When >0, a seg-init re-spawn gives
    # the drone velocity ``segment_init_vel_mps`` * unit(next_gate - prev_anchor)
    # instead of the zero-velocity hover the original Song §III-B recipe used.
    # Motivation: v25-v28 level-3 eval renders showed the policy hover-and-
    # survives when it reaches target_gate >= 2 because that obs region was
    # OOD during training (no seg-init exposure to mid-track states). Re-
    # enabling seg-init at p=0.5 with zero velocity caused v24's lucky-zone
    # collapse — the policy over-fit to "spawned hovering at a convenient
    # midpoint then accelerate". Giving the seg-init spawn a non-zero
    # velocity in the direction of the next gate removes the trivially-
    # exploitable "spawned hovering" state distribution while still putting
    # the policy at later-gate approach poses for training.
    segment_init_vel_mps: float = 0.0
    # Song 2023 §III-B Phase 2 successful-state buffer. With probability
    # ``phase2_prob`` an env that just reset is re-spawned from a state
    # sampled out of a per-gate stratified buffer of past successful
    # gate-pass events. Distinct from Phase 1 seg-init: the buffer
    # captures real trajectory states the policy has reached, instead of
    # synthetic segment midpoints.
    #
    # Gated by a warm-up: until ``global_step >= phase2_warmup_steps`` the
    # effective probability is 0.0 (the buffer needs time to populate, and
    # an empty replay is wasted compute). Cross-warmup retraces the JIT
    # cache once, then runs at the configured ``phase2_prob`` for the rest
    # of the stage.
    phase2_prob: float = 0.0
    phase2_capacity_per_gate: int = 4096
    phase2_warmup_steps: int = 0


@dataclass(frozen=True)
class CurriculumConfig:
    """Ordered curriculum stages. Stage index advances on promotion."""

    stages: tuple[CurriculumStage, ...]
    promotion_check_iterations: int = 100
    promotion_window_rollouts: int = 50


def default_curriculum() -> CurriculumConfig:
    """Return the active curriculum.

    v44 (2026-05-22): single level-2 stage with Song 2023's Phase-1 +
    Phase-2 distribution scaffolding reinstated, reward and PPO knobs
    otherwise unchanged from v43.

    The v43 bare-L2 + Song-verbatim run collapsed to a clean hover
    attractor — 8,299 episodes, 498/500 steps each, 0/4096 gate passes,
    0 crashes, ep_ret −3.3 dominated by integrated r_omega. The diagnosis
    matches v36 / v37b history verbatim: under bare ``r_prog + r_omega +
    r_terminal``, random-Gaussian exploration produces ~zero expected
    r_prog while hover pays only r_omega (~−5/episode), and any committed
    motion that crashes pays −10. Hover is strictly best until the value
    function sees a positive return — but with no curriculum scaffolding,
    the policy collapses to hover before random exploration ever reaches
    a gate at true-start frequency. The wandb buffer fill diagnostic
    (``phase2_buffer_fill_g1 = 609`` accidental passes recorded but
    *discarded* because phase2_prob was 0) is the direct evidence that
    Phase 2 replay would have been load-bearing.

    v44 fix: reinstate the v41 Lever-B / Phase-2 settings at p=0.10
    each. Phase 1 seg-init (``segment_init_prob = 0.10``,
    ``segment_init_vel_mps = 2.5``) puts ~10% of envs at mid-track
    segment centers with velocity along the next gate's traversal axis,
    so r_prog is immediately positive from those spawns — anchoring V to
    positive returns and breaking the hover attractor on the true-start
    90%. Phase 2 replay (``phase2_prob = 0.10``,
    ``phase2_warmup_steps = 20_000_000``) re-spawns from real past
    gate-pass states once the buffer has filled, exposing the critic to
    the entire reachable forward-progress distribution.

    Reward stays Song-verbatim (3 terms: ``r_prog + r_omega + r_terminal``,
    no jackpot, no guide, no time penalty, no obstacle barrier, no
    lookahead) per the v43 design. PPO hyperparameters stay at the
    Codex-reviewed values (``n_steps=250``, ``ent_coef 0.02 → 0.005``
    anneal, ``LOG_STD_MIN = −2.5``, ``gamma=0.998``, ``gae_lambda=0.97``,
    ``update_epochs=3``, ``target_kl=0.02``). ``reset_pos_perturb_m = 0``
    is preserved because the level-2 toml randomization (drone_pos ±0.10 m
    / drone_rpy ±0.10 rad) handles the symmetry-break job that motivated
    v38i's 0.10 m perturb on deterministic L1.

    Predecessor handoffs and the prose v43 hover-collapse diagnosis live
    in ``docs/handoffs/`` (untracked); :func:`_default_curriculum_v42_history`
    preserves the v42 two-stage L1→L2 layout, :func:`_full_curriculum`
    preserves the legacy seven-stage progression.
    """
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
    """Return the v42 multi-stage curriculum, preserved for reinstatement.

    v38 (2026-05-20): single level-2 cold-train stage under the new
    Schuck-2025 delta-tangent action head. Goal — ship a controller
    that clears ≥50% finish on a 20-run ``config/level2.toml`` eval
    with average lap time near 3 s. The new action head (raw 4-vec
    instead of 7-vec) makes every v33-v37 checkpoint structurally
    incompatible (``Dense_2`` shape, ``log_std_rotation`` → ``log_std_tangent``),
    so warm-start is unavailable and the cold-start gradient must do
    the work. The active reward defaults restore the v33b jackpot +
    Liu r_guid scaffold that historically broke cold-start (v7a-style)
    — see the ``RewardConfig`` v38 notes for the term-by-term rationale.
    Phase-1 seg-init + Phase-2 buffer remain from v36/v37b for mid-track
    state coverage. Per-step rotation budget ``α_max`` is overridable
    from the train CLI via ``--alpha-max-rad``; default 0.04 rad,
    recommended sweep ``{0.04, 0.08, 0.16}`` per Schuck 2025 + the
    2026-05-20 SO(3) handoff §7.4.

    v36 history note: cold-train under a Song-2023-stripped reward +
    the proximity-obs additions from v35. v33b/v34/v35 each demonstrated
    that the accumulated reward complexity (gate_pass jackpot, exit-vel
    bonus, gate-frame Gaussian barrier, time penalty, 10× crash/finish
    events) had us in a local optimum that further tuning could not
    escape — v35 broke the obstacle:0 dominance via obs engineering
    but introduced new mid-track failures, and tied v33b on finish_rate
    (6/32 vs 7/32). The hypothesis for v36: our reward additions are
    net-negative and the cleanest path forward is to revert to Song's
    minimal gate-progress reward and trust the value function to do the
    safe-state work.

    Per Song 2023 §"Gate Progress" (Scaramuzza et al., Science Robotics):
    ``r(k) = ||g_k - p_{k-1}|| - ||g_k - p_k|| - b·||ω_k||`` with
    ``b=0.01`` at 100 Hz (doubled to ``0.02`` at our 50 Hz), plus +10 on
    finish / -10 on collision. **Nothing else.** The paper explicitly
    argues that the value function alone learns to assign low values to
    risky states (near gate borders), which removes the need for explicit
    barrier penalties.

    Our level-3 tracks have obstacles that Song's did not, so we keep the
    Gaussian obstacle barrier from v33-v35 (``obstacle_weight=0.8``,
    ``obstacle_sigma=0.5``) as our minimal addition. We also keep the v35
    proximity obs features (dim 61) because they are observation
    engineering, not reward — and v35 demonstrably broke the obstacle:0
    blindness via those features.

    Reward terms removed for v36:

    * ``gate_pass_bonus`` (was +40 to +160 per gate; v18 cold-start
      scaffold no longer needed).
    * ``exit_vel_bonus`` (was up to +50 per gate; integrated *negative*
      in every v33b / v34 / v35 eval, was paying for nothing).
    * ``gate_frame_weight`` (was 1.2; Song uses value-function bonuses
      not per-step barriers; this was over-pricing tight gate lines and
      pushing the policy onto wide approaches that brought it closer to
      obstacles next to the gates).
    * ``time_penalty`` (was 0.15; Song relies on γ-discounted
      finish_bonus to drive speed; explicit time penalty made the
      do-nothing baseline negative which created suicide attractor under
      a cold-train).

    Magnitudes rescaled 10× down:

    * ``progress_coef`` 10.0 -> 1.0 (Song's literal).
    * ``crash_penalty`` 100 -> 10 (Song's literal).
    * ``finish_bonus`` 100 -> 10 (Song's literal).
    * ``omega_coef`` 0.003 -> 0.01. Song's literal at 50 Hz would be
      0.02 (b=0.01 at 100 Hz, doubled for the half step rate), but
      codex's pre-launch review flagged 0.02 as too punitive at the
      stripped-reward scale (per-step omega penalty cancels r_prog at
      L1(ω)=5). 0.01 keeps the term active as a smoothness prior
      without overpowering forward motion.

    Loiter penalty (Liu r_guid) explicitly disabled (``use_guide=False``).
    Song 2023 has no spatial guidance term; cold-start risk is mitigated
    via ``segment_init_prob=0.30`` (mid-track spawn) and
    ``phase2_prob=0.30`` (past-gate-pass replay buffer) instead.

    Cold-train (no ``--init-from``) for 500M timesteps. The reward
    landscape is too different from v33-v35 to warm-start usefully — the
    value function trained on +160 gate-jackpots will mispredict the
    return distribution under +10 finish-only events. Cold-train risk
    mitigations: keep ``segment_init_prob=0.30`` with ``segment_init_vel_mps
    =2.5`` (v29-v33 default — phase-1 seg-init that exposed the policy
    to mid-track approach states), keep ``phase2_prob=0.30`` with 50 M
    warmup (phase-2 buffer of past gate-pass states). These are training
    curriculum, not reward, and are preserved from v33-v35.

    v37b update (2026-05-19): the v36 cold-train collapsed (eval@500M:
    0/32 gates, 29/32 floor crashes, median ep_len 0.24 s). With
    ``use_guide``, ``time_penalty``, ``use_gate_pass_bonus``,
    ``use_exit_vel_bonus``, and ``gate_frame_weight`` all stripped, there
    is no ``f(p_t)`` position-based reward left to break the cold-start
    exploration trap — exactly the v15-v19 failure mode our prior
    iteration documented. The above "warm-start unhelpful" rationale
    applies only to value miscalibration; it does not address the
    cold-start gradient problem, which dominated.

    v37b inverts the warm-start decision: same reward and curriculum
    as v36, but ``--init-from`` the v35 checkpoint
    (``level3_v35_proximity_obs_warmstart_from_v34_seed0_300M``) which
    is already a competent gate-passing policy at the matching 61-dim
    obs. Critic miscalibration risk is accepted; PPO value clipping
    plus ~50M steps of re-alignment is expected to handle it. CLI:
    ``--init-from`` parent run dir, ``--ent_coef_start 0.001``
    (matches ``ent_coef_final=0.001`` → flat schedule, preserves v35's
    low-entropy commitment), ``--total_timesteps 300_000_000``,
    ``--seed 0``, ``--stage 1``. Tests whether the stripped Gate-Progress
    reward preserves a competent policy (orthogonal question to whether
    it can bootstrap one).

    v35: proximity-obs warm-start on top of v34 (which tied v33b within
    noise — 6/32 finishes vs 7/32, same obstacle:0 dominant failure mode).
    The v34 eval traces showed the policy is generating near-zero roll
    commands as it approaches obstacles (e.g. episode 6: drone distance to
    obstacle dropped 0.39 m → 0.16 m while roll stayed in [-0.05, +0.16]).
    The raw obstacle channel contains the information in principle, but
    PPO did not learn the cross-channel multiplicative interaction
    (self_velocity · obstacle_direction → roll command) from sparse
    reward across 300M v33b + 300M v34 steps.

    v35 pre-computes that interaction as two scalar obs features —
    ``min_clearance_xy_m`` and ``closing_speed_to_nearest_obs_mps`` —
    appended to the actor / critic obs (dim 59 → 61). Rewards inherit
    v34 unchanged (``obstacle_sigma=0.5``, ``time_penalty=0.15``,
    ``crash_penalty=100``). Warm-start from v34 with zero-padded first-
    layer kernels so the existing 59 features keep their learned weights
    and the two new features start with no influence; PPO retrains the
    new input weights from zero over the 300M-step run.

    v34: obstacle-pricing fix on top of v33b (current SOTA, 7/32 finishes).
    Eval traces of v33b on level 3 (``renders/plan-D1-smoke/``) showed
    3/8 episodes crashing into obstacle 0 within 0.84 s of takeoff, on
    near-straight spawn-to-gate-0 trajectories with the obstacle <0.11 m
    off the line. Diagnosis: v33's r_obs Gaussian barrier (sigma=0.3) is
    below -0.05/step at d>=0.5 m, so the policy gets <0.4 s of felt
    gradient at cruise speed — not enough lead time to learn a detour
    against the continuous r_prog pull. Two parameter changes warm-started
    from v33b, everything else unchanged:

    * ``obstacle_sigma`` **0.3 -> 0.5**. Same peak penalty at d=0, but the
      felt zone extends to ~1.0 m. 24% of gate-obstacle XY pairs are
      within 0.7 m on level 3, so passing a near-obstacle gate now pays
      ~-8 obstacle penalty — still well under the +50 gate bonus, but
      actively biases the policy toward gate-passing lines that maximize
      lateral clearance.
    * ``time_penalty`` **0.10 -> 0.15**. v33b's clean laps run ~6 s; the
      0.10 budget widened the fast/slow gap to 30 reward, 0.15 widens it
      further to 45 reward (~1 gate-bonus differential).

    All other v33 parameters retained: ``crash_penalty=100``,
    ``obstacle_weight=0.8``, ``gate_frame_weight=1.2``,
    ``omega_coef=0.003``, ``exit_vel_coef=10.0``, Phase 2 mix.

    v33: obs/reward-economics overhaul on top of v32a (first lap-finishing
    controller). Same Phase 2 mix and same cold-train discipline as v32a,
    but with eight changes prompted by the post-v32a codex+Claude review:

    Observation/geometry consistency
        * **Obstacle obs at drone altitude** (``obs.py``): encode the
          body-frame vector to ``[obs_x, obs_y, drone_z]`` instead of the
          top marker. The reward already does XY-only distance to a
          vertical pole; the obs now exposes the same geometry instead of
          a "ball above-and-to-the-side" feature.
        * **Safety reward against actor-visible poses** (``reward.py``):
          ``r_obs`` and ``r_gate_frame`` use post-wobble truth for
          visited objects, pre-wobble placement for unvisited objects.
          v32a graded the actor on a ±0.15 m XY perturbation it could
          not see until sensor range, which broke the per-step
          avoidance gradient on the (large majority of) unvisited
          frames.
        * **r_gate_frame masked to target ± 1**: drops the per-step
          avoidance gradient from gates outside the actor's
          ``N_FUTURE_GATES=2`` observation window.

    Speed economics
        * ``omega_coef`` **0.01 -> 0.003**. At 10 rad/s axis rates the
          old penalty integrated to ~-150 over a 500-step episode,
          larger than ``finish_bonus`` and structurally biasing the
          policy toward slow gentle turns.
        * ``exit_vel_coef`` **2.0 -> 10.0** with ``v_to_next`` clipped to
          ±5 m/s. v32a's term paid only ~5-15% of the per-gate jackpot,
          easy to ignore; the bump makes a clean 5 m/s aligned exit
          worth +50, comparable to the gate-1 jackpot.
        * ``time_penalty`` **0.05 -> 0.10**. v32a's gap between a
          200-step fast lap and a 500-step slow lap was a rounding
          error next to event-based reward; the bump widens it to a
          gate-bonus-worth differential.

    Safety pricing
        * ``crash_penalty`` **15 -> 50**. v32a's 15 was lower than the
          banked early-gate jackpots (40 + 80 = 120), making risky
          terminal-gate lines net-positive. 50 still leaves a marginal
          additional gate worth attempting but stops "bank-and-crash"
          from being a profitable strategy.
        * ``obstacle_weight`` **0.2 -> 0.6**, ``gate_frame_weight``
          **0.2 -> 0.8**. v32a's barriers maxed at ~-0.8/step (all 4
          obstacles touching) and ~-0.04/step (gate-frame graze), too
          weak relative to ~+10/step ``r_prog`` at full speed. The
          bumps carve actual no-go zones.

    Segment-init obstacle-visibility (``rollout.py``)
        * ``_refresh_aux_fields_after_respawn`` now recomputes
          ``obstacles_visited`` from XY sensor range instead of setting
          a blanket-True. The blanket-True made mid-track respawns
          claim sensor discovery of obstacles outside range, which
          contaminates the actor obs feature distribution.

    All else (Phase 2 mix, seg-init mix, layout-restoring buffer, cold
    train at 500M, anneal entropy 0.005 -> 0.001) carries over from v32a
    unchanged.

    v32a (history note): reward fixes for the obstacle / gate-frame collision
    problem v30 / v31 evals exposed, with the v32 bugs codex caught corrected:
    r_obs uses XY-only distance (obstacles are vertical capsules from
    z≈1.55 down to floor — full 3D distance was dominated by the
    vertical offset, keeping r_obs near zero even right next to the
    capsule), the in-scan reward call now passes ``true_gates_quat``
    (was silently using nominal toml quats for unvisited randomized
    gates → r_gate_frame and r_guid were oriented against the wrong
    gates during training), and dense position-dependent terms
    (r_prog / r_obs / r_gate_frame / r_guid) are zeroed on crash
    steps to avoid the warp-location spurious gradient. Same Phase 2
    mix as v31
    (``segment_init_prob=0.30`` velocity-aware 2.5 m/s, ``phase2_prob=0.30``,
    warmup 50M, 40 / 30 / 30 % per-reset partition) and the same
    layout-restoring buffer, but with two reward changes:

    * ``r_obs``: drop the ``obstacle_active = 1 - obstacles_visited``
      mask (was zeroing the penalty exactly when the small-sigma
      barrier became non-negligible) and bump ``obstacle_sigma`` from
      0.2 → 0.3 m so the avoidance gradient extends to safe-but-close
      distances.
    * ``r_gate_frame`` (new): Gaussian barrier on point-to-segment
      distance to each gate's 4 frame edges. Replaces the missing
      collision signal — v30 / v31 drones crashed into gate bezels
      because there was no per-step penalty for it.

    **Cold-train 500M** rather than warm-starting from v26: v26's
    policy ignores both new reward terms (since they were near-zero
    during its training), and the un-learning cost outweighs the
    transferable gate-passage knowledge. v7-style fresh init with
    annealing entropy.

    v29 (history note): single level-3 stage with full distribution
    (``gate_rand_scale=1.0``), Song §III-B seg-init at
    ``segment_init_prob=0.5``, and **velocity-aware seg-init**
    (``segment_init_vel_mps=2.5``). Designed to cold-train from scratch
    with an annealing entropy schedule
    (``PPOConfig.ent_coef_final=0.001``) so the policy commits to using
    the seg-init exposure.

    Why this combination
    --------------------
    User-observed eval problem from v25/v26/v28 patched renders: the
    drone reliably navigates gates 0 and 1 from a ground spawn but then
    **hovers indefinitely at gate 2** without crossing. Diagnosis: with
    seg-init disabled, ~86% of training samples have ``target_gate=0``
    and the policy never trained on the "approach gate 2 from past
    gate 1" obs distribution. When eval succeeds at the first two
    gates, the post-gate-1 state is OOD and the policy defaults to a
    hover-survive attractor (no crash penalty, just ``r_time`` ticks).

    The fix needs *training exposure* to mid-track states, since reward
    shaping only works for states the policy actually visits. Seg-init
    is the existing mechanism for that. But v24 enabled it at p=0.5
    with zero-velocity spawns and collapsed onto a lucky-zone strategy:
    eval was 0/8 finishes, 0/8 gates from a true ground spawn despite
    training metrics looking strong. The lucky-zone vulnerability was
    "spawned hovering at convenient midpoint → accelerate" — a state
    distribution the policy could over-fit because real flight never
    produces hovering at those midpoints.

    Velocity-aware seg-init removes that vulnerability: the re-spawn
    velocity is ``segment_init_vel_mps * unit(next_gate - prev_anchor)``,
    so the seg-init state distribution becomes "drone in transit
    between gates" rather than "drone hovering at a lucky pose".
    Paired with the lower entropy floor and the now-tracked
    ``finish_rate_true_start`` metric (which exposes lucky-zone
    collapse if it happens despite the velocity), v29 directly attacks
    the third-gate hover problem.

    History
    -------
    * v25's ``level3_warmstart`` (seg-init disabled) is at commit
      ``9514e7e``.
    * v24's ``level3_warmstart_seginit`` (segment_init_prob=0.5, zero
      velocity) at commit ``d4c4fbd``.
    * v22's ``level3_rand0`` at commit ``63e0f0c``.
    * The v11–v21 ``level2_seginit`` curriculum at commit ``9ad7fa0``.
    * The legacy stage1/2/3a/b/c/4 progression remains in
      :func:`_full_curriculum`.
    """
    pi_over_4 = 0.7853981633974483
    return CurriculumConfig(
        stages=(
            # v38b: stage 1 = deterministic level-1 cold-train. This is
            # the v7a recipe that solved level 1 to 100% finish in 100M
            # steps under the v18-era reward scaffold. The single-stage
            # v38a attempt against level 2 plateaued at max_gate ≈ 1.1
            # under both α_max=0.08 and α_max=0.16, suggesting a "hit
            # gate 1, drift" local optimum: the high gate-pose
            # randomization of level 2 + crash_penalty=100 made the
            # expected return on a gate-2 attempt net-negative for the
            # cold-start policy, so it stopped trying. v38b breaks that
            # by mastering the deterministic-gate task first, then
            # transferring to level 2 with a competent gate-passing
            # policy and only the gate-pose-adaptation problem left.
            CurriculumStage(
                # v38e: seg-init + phase-2 DISABLED for cold-start. The
                # v38c run achieved 24% overall finish_rate but
                # finish_rate_true_start = 0 — the policy only learned
                # gate-passing from mid-track spawns and never solved
                # the takeoff→gate-1 subtask. Deterministic deployment
                # always starts from true ground spawn so eval crashed
                # 0/20. v7a's recipe also had seg-init=0 for stage 1 →
                # 100% true-start finish on level 1.
                # v38i: bumped reset_pos_perturb_m from 0.0 -> 0.10 and
                # reset_yaw_perturb_rad from 0.0 -> 0.3. Under
                # deterministic level 1 with 0 perturbation, all 4096
                # vmapped envs init at exactly the same toml drone
                # state. Combined with low policy entropy after early
                # training, the 4096 trajectories become near-identical
                # — we have 4096 rollouts but ~1 effective sample for
                # exploration. v38f-v38h max_gate=1.000 exactly forever
                # (across ~280M step-env opportunities, no env ever
                # passed gate 1) is direct evidence. Small init noise
                # gives the 4096 envs different trajectories, breaking
                # the 4096-clone symmetry without changing the eval
                # distribution (level1.toml has zero pose randomization;
                # deployment is still the toml start state).
                # v38l (2026-05-22): re-enable Phase-1 seg-init at low
                # probability (``segment_init_prob=0.10``) with velocity-
                # aware spawn (``segment_init_vel_mps=2.5``). Wandb
                # diagnostic on v38i/v38j surfaced a climb-then-fall in
                # ``rollout/target_gate`` (peak 0.23-0.26 at step ~14M,
                # collapse to 0.084 by step ~25M) that the end-of-training
                # summary completely hid. Critical: during the collapse
                # ``approx_kl`` stayed active (0.005-0.006), entropy
                # stayed high (3.79 → 3.64), and ``value_loss`` dropped
                # 5.5 → 0.16 — i.e. PPO confidently optimized AWAY from
                # gate-1 attempts toward a lower-variance gate-0-only
                # attractor (NOT exploration starvation). Mechanism: the
                # policy's only post-gate-1 rollouts are crashes (no
                # gate 1→2 navigation learned yet), so the advantage of
                # attempting gate 1 has high variance vs the stable
                # gate-0 trajectory. Lever B (Codex ranking, B > A > C):
                # seg-init at 10% with velocity gives the critic a real
                # distribution of post-gate-1 trajectories that aren't
                # all crashes. ``finish_rate_true_start`` remains the
                # unbiased true-spawn audit (the v38e revert rationale
                # holds at higher p ≥ 0.30; v24 at p=0.5 zero-vel
                # collapsed onto a lucky-zone strategy that produced
                # 0/8 finishes from true ground spawn). At p=0.10 the
                # 90% true-spawn distribution still drives the
                # takeoff→gate-1 subtask while the 10% mid-track
                # exposure breaks the variance asymmetry. Keep
                # v38k's entry-waypoint look-ahead intact — it gave
                # a small discovery bump (peak target_gate 0.26 vs
                # 0.23) and recovered the v38j level-2 regression.
                name="stage1_det_level1",
                level=1,
                use_domain_randomization=False,
                reset_pos_perturb_m=0.10,
                reset_vel_perturb_mps=0.0,
                reset_yaw_perturb_rad=0.3,
                gate_rand_scale=1.00,  # ignored on level=1 (no toml gate noise)
                segment_init_prob=0.10,  # v38l: Lever B
                segment_init_vel_mps=2.5,  # v38l: Lever B (velocity-aware)
                # v41: enable Phase 2 replay at stage 1. The buffer is being
                # populated (slot 3 saturates at 4096 within ~40M steps from
                # segment-2 seg-init spawns; with the v41 seg-init fix slot 2
                # should also fill) but the policy was never replayed from
                # those successful post-gate-K states — every v38-v40 run
                # had ``phase2_prob=0.0``. Wandb diagnosis 2026-05-22:
                # ``rollout/target_gate`` climbs to a peak ~20-25M then
                # *drops* and stagnates while ``max_gate`` plateaus at 1.1,
                # because the value function learns V(post-gate-0) ≈ "crash"
                # from the policy's own death trajectories and actively
                # pushes the policy away from gate-1 attempts. Phase 2
                # replay breaks the self-reinforcing dynamic by feeding the
                # critic real post-gate-K trajectories that don't crash
                # (the buffer only stores ``~done_bool`` events).
                # p=0.10 matches segment_init_prob — kept low so
                # ``finish_rate_true_start`` stays the unbiased true-start
                # audit metric.
                # warmup=20M lets the buffer accumulate non-trivial fill
                # (slot 3 saturates at ~40M historically; slot 2 will be
                # the new variable with v41 seg-init) before replay starts
                # consuming entries. Smaller than stage 2's 50M because
                # this is a cold start with no warm-start checkpoint —
                # we want the policy to escape the post-gate-0 trap as
                # early as possible once buffer is non-empty.
                phase2_prob=0.10,
                phase2_capacity_per_gate=4096,
                phase2_warmup_steps=20_000_000,
                promote_target_gate_mean=3.0,
                promote_crash_rate_max=0.3,
            ),
            CurriculumStage(
                # v38b: stage 2 = level 2 deployment distribution.
                # Reached via auto-promotion from stage 1 (when
                # target_gate_mean >= 3.0) or via --init-from a stage-1
                # checkpoint on a second launch. Level 2 toml gives
                # gate_pos ±0.15 m / gate_rpy yaw ±0.20 rad /
                # obstacle_pos ±0.15 m / drone_pos ±0.10 m / drone_rpy
                # ±0.10 rad / mass/inertia DR + action/dynamics noise.
                # gate_rand_scale=1.0 means the level-2 toml ranges are
                # used verbatim; reset_pos_perturb on top gives the
                # policy ~2× the deployment perturbation margin.
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
    # Per-step rotation budget on ``‖τ_scaled‖`` (rad). Single source of truth
    # for the active α_max — overridable from the train CLI via
    # ``--alpha-max-rad`` to sweep {0.04, 0.08, 0.16} without code edits.
    # The module-level :data:`TANGENT_ALPHA_MAX_RAD` is the default fallback;
    # downstream (rollout, controller, train logging) reads this attribute so
    # the value flowing through env action projection and saturation
    # diagnostics is always consistent with the active run config.
    tangent_alpha_max_rad: float = TANGENT_ALPHA_MAX_RAD
    checkpoint_every_steps: int = 5_000_000
    eval_video_every_steps: int = 5_000_000
    wandb_project: str = "lsy-drone-racing-rl-song"
    wandb_entity: str | None = None
    run_name: str | None = None  # defaults to <stage>_<seed>_<timestamp> in train.py
