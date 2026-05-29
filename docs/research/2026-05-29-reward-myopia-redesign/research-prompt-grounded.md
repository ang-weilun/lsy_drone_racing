# Grounded research prompt (method note)

The first deep-research pass came back with recommendations that were already
implemented in our code (multi-gate body-frame observation, plane-crossing gate
switching, curriculum) because it had no visibility into our implementation. We
then authored the grounded prompt below — which explicitly states what is already
built and points at the genuinely open questions — and re-ran the research. The
second and third passes (reports 2 and 3) were materially better as a result.

This file is kept as a record of the method: *ground the research prompt in the
actual implementation so the review does not re-recommend solved problems.*

---

````markdown
# Research request: fixing myopic point-to-point flight in an RL drone-racing progress reward (the "next gate is behind/beside" turn problem)

## What I'm building
An autonomous drone-racing controller for a **Crazyflie** flying a **fully randomized** track of gates and obstacles. State (drone + gate + obstacle poses) comes from a **Vicon mocap** system; a PC runs the policy and streams commands; the Crazyflie's onboard low-level controller closes the inner loop. We command **collective thrust + body-rate-style attitude (CTBR-like, the Song/Kaufmann interface)**, clipped at a max angular rate α_max (currently 0.32 rad/step on the hard level), at **50 Hz**. No onboard camera — full state is available from mocap, so any "perception" term would be for *anticipation*, not state estimation.

We use **PPO** (massively parallel, ~16k envs, asymmetric actor-critic with a privileged critic), following the **UZH RPG line** (Song 2021/2023 "Reaching the limit", Kaufmann 2023 "Swift"). We deliberately formulate the objective as **gate-progress** rather than time-optimal trajectory tracking, per Song 2023's argument.

## What is ALREADY implemented — please do NOT re-recommend these
- **Multi-gate observation in the drone body frame.** 52-d obs: drone = 9-d rotation matrix + 3-d body-frame velocity; gates = current target gate's 4 corners in body frame + (next gate corners − target gate corners) delta; obstacles = 2 nearest. Current + next gate (N_FUTURE_GATES=2); gate orientation implicit in corners.
- **Directional plane-crossing gate-switching within the aperture** (not centre-proximity).
- **Asymmetric actor-critic** (critic gets true poses; actor gets sensor-masked poses).
- **Curriculum + domain randomization**: mid-track segment-init with gate-aligned initial velocity; success-state replay buffer; randomized gate/obstacle poses; sensor-range randomization.
- Action-smoothness and body-rate penalties are wired in.

## The reward we actually use (this is where the problem lives)
- **Progress (the suspect): `r_prog = k · (‖g_center − p_{t−1}‖ − ‖g_center − p_t‖)`** — change in Euclidean distance to the current target gate's centre. Direction-blind.
- Body-rate penalty; per-step time penalty (a speed lever); terminal crash (terminates); terminal finish. γ=0.998.
- Available-but-disabled shaping levers: entry-waypoint lookahead, wrong-side/overshoot penalty, signed "dipole" potential, exit-velocity bonus, gate-frame Gaussian barrier, obstacle barrier.

## The problem (a local optimum, in the current best policy)
Slow, point-to-point flight. When the successor gate is **behind** (≈180°) the drone flies through the current gate and **reverses out**; when the successor is **to the side**, it straight-lines to the next centre and **clips the frame of the gate it just passed**.

**Mechanism we believe:** after crossing gate N's plane the target advances to N+1 and the direction-blind centre-distance `r_prog` rewards shrinking the straight-line distance to N+1's centre — which points backward/sideways through N's frame. The plane-crossing switch prevents re-counting but not the backward pull.

**Subtlety to engage directly:** naive path-projection (progress along the segment between gate *centres*, Song 2021) does NOT fix the 180° case — when N+1 is directly behind, that segment points backward too. We need formulations that produce through-and-turn behaviour even when the successor is behind/lateral.

## What I want from this research
Literature-grounded, cited (favour the RPG line; verify the actual reward equations), ranked recommendations with exact formulations, addressing:
1. Progress-reward geometry that does NOT reward reversing when the next gate is behind/beside (smoothed guiding path vs entry-waypoint vs corridor vs topological path); for each, whether/why it avoids the backward-reward pathology on a 180°.
2. Exit-momentum and heading/anticipation terms — published role and magnitude vs progress.
3. Reward vs exploration/curriculum: is this a reward-shaping or curriculum problem in the literature?
4. Anti-myopia / horizon: discount, n-step returns, value horizon, how far ahead the next gate(s) should enter the reward.
5. Observation refinement (only if evidence supports it given we already have current+next gate in body frame): next gate relative to current gate's pose / explicit gate normal.

## Constraints / preferences
- Crazyflie scale (modest thrust-to-weight, CTBR/attitude, 50 Hz) — tune magnitudes accordingly.
- Fully randomized layouts — the fix must generalise, not memorise.
- Prefer methods that compose with warm-start fine-tuning and single-lever experiments; open to replacing the core progress term if the evidence says symptom-layering won't fix the root.
- For each recommendation: exact formula, source + equation reference, expected effect on the 180°/lateral cases, main failure/tuning risk. Flag where the literature disagrees or a claim is unverified.
````
