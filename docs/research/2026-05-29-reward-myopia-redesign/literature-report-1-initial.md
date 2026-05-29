> **Archival note.** First deep-research pass, reproduced verbatim. This pass had
> no visibility into our implementation and therefore recommended several things
> we had already built (multi-gate body-frame observation, plane-crossing gate
> switching, curriculum). Kept for the record; superseded by reports 2 and 3,
> which were run against the grounded prompt in `research-prompt-grounded.md`.

---

# Fixing "Point-to-Point" Myopia in RL Drone Racing: Observation, Reward, Gate-Switching, and Curriculum Solutions

## TL;DR

- Your myopic, slow, reverse-out-and-collide behavior is driven by two coupled root causes: (1) an observation horizon that only meaningfully exposes the current gate center, and (2) a progress reward measured to the gate *center* combined with *center-based instant* target switching — fix the observation (multi-gate, body-frame, with gate orientation) and the switching logic (directional plane crossing + a target point placed past the gate plane) first, then add a look-ahead/heading term.
- The literature is consistent: Song et al. (IROS 2021) showed that adding *future gates* to the observation improves success rate and lap time, and that a *path-projected* progress reward (not Euclidean distance-to-center) plus a *safety reward* keyed to the gate avoids the degenerate point-to-point line; Swift (Nature 2023) keeps a center-progress reward but adds a perception/heading term and treats a gate as passed on actual traversal, not center proximity.
- Highest-impact, lowest-risk fixes for your crazyflow/lsy_drone_racing PPO setup, in order: (1) represent current + 2 future gates in the drone body frame including each gate's normal/orientation; (2) place the progress target a short distance *past* the gate plane along the gate normal and switch on *directional plane crossing*; (3) switch progress to a path-projection formulation; (4) add a gate-heading/perception-alignment term; (5) wrap it all in a speed/track-difficulty curriculum.

## Key Findings

**1. The standard progress reward has two distinct lineages, and the difference matters for your bug.**

- **Song et al. (IROS 2021)** — "Autonomous Drone Racing with Deep Reinforcement Learning" — uses a **path-projection progress reward**. The track is represented as line segments connecting gate centers; the reward each step is the change in the drone's *projected coordinate along that path*, i.e. r_p(t) = s(p_t) − s(p_{t−1}), where s(·) projects the drone position onto the segment connecting the previous gate to the next gate. This rewards motion *along the racing line*, not motion that merely reduces straight-line distance to the center. They add a **safety reward** that encourages passage near the middle of the four gate corners (implicitly respecting gate orientation), plus a command penalty and a terminal crash penalty.
- **Swift (Kaufmann et al., Nature 2023)** uses a simpler **center-distance progress reward**. Verbatim, the total reward is r_t = r_t^prog + r_t^perc + r_t^cmd − r_t^crash, with:
  - r_t^prog = λ1 (d_{t−1}^Gate − d_t^Gate), where d_t^Gate is the distance from the vehicle center of mass to the *center of the next gate*;
  - r_t^perc = λ2·exp(λ3·δ_cam⁴), where δ_cam is the angle between the camera optical axis and the next gate center;
  - r_t^cmd = λ4‖a_t^ω‖ + λ5‖a_t − a_{t−1}‖² (body-rate and action-smoothness penalties);
  - r_t^crash = 5.0 when the platform goes below ground or collides with a gate (which also terminates the episode).
- **Critical interpretation:** Your reward is the Swift-style center-distance form, but you are *missing the perception/heading term* and you have *center-based instant switching*. A pure center-distance progress reward with instant center switching is exactly the configuration that produces point-to-point myopia: once the gate switches at the center, the drone is rewarded for collapsing the straight-line distance to the next center, which can point backward (180° gates → reverse out) or sideways (off-to-the-side gates → straight-line into the frame just crossed).

**2. Future-gate observations are the single most-cited fix for anticipation/racing lines.**

- Song et al. (2021)'s headline contribution is "relative gate observations," and they report a **gate-observation ablation** showing that including information about *future gates* (beyond the immediate next gate) improves success rate and lap times. Follow-up work (Liu, "Learning Generalizable Policy for Obstacle-Aware Autonomous Drone Racing," 2024) adopts this directly: "We include information about two future waypoints based on the result of the gate observation experiment in [Song 2021]: including information about two future gates can improve success rate and lap times." Each waypoint is encoded in the **drone body frame** as the relative gate position plus the unit vectors and lengths from the drone body-frame origin to the four corners (a 17-dim vector per waypoint there).
- **Swift, by contrast, encodes only the single next gate** as the relative position of the four gate corners with respect to the vehicle (R¹²), inside a R³¹ observation (R¹⁵ state with rotation matrix + R¹² gate corners + R⁴ previous action). Swift compensates for the lack of an explicit future-gate horizon with a strong value function over a long horizon.
- **Why body frame for all gates matters:** Expressing every gate (current + N future) in the drone body frame makes the policy's input/output mapping translation- and rotation-equivariant. Encoding a future gate relative to the *current target gate's* pose additionally tells the policy the *exit direction* it should carry through the current gate.

**3. Center-based instant switching is the direct cause of "reverse-out" and "frame collision."**

- When the active gate switches the instant the drone reaches the gate *center*, the progress target jumps to the next gate center while the drone is still *inside* the current gate plane. The robust alternatives: (a) define gate passing as a **directional plane crossing**; (b) place the **progress target point a short distance past the gate plane along the gate normal**; (c) **switch with look-ahead**, exposing the next gate in the observation well before the current one is cleared.

**4. Progress-reward-only RL reliably converges to slow myopic local optima.**

- Penicka, Song, Kaufmann & Scaramuzza (RA-L 2022) state that "naive optimization of the reward formulation results in suboptimal performance due to local minima," and solve it with a **two-stage curriculum** plus progress along the **topological guiding path**.

**5. RL progress reward is the discrete-time analog of CPC's complementary progress constraints (Foehn & Scaramuzza 2020).**

## Recommendations (in order)

1. Observation fix: current + 2 future gates, body frame, with gate normal; asymmetric actor-critic.
2. Switching + target-point fix: directional plane crossing; target ≈0.2–0.5 m past the gate plane; wrong-side penalty.
3. Reward-form fix: path-projection progress to the past-plane target; strong terminal crash; directional gate-pass bonus.
4. Heading/anticipation term: r_perc = λ2·exp(λ3·δ⁴) on the angle to the next gate; action-difference smoothness penalty.
5. Curriculum + randomization: cap per-step progress and anneal; randomize gate poses; reset at random gates.

## Caveats

Exact Swift λ weights and the verbatim Song 2021 reward equations were not verified from the primary PDFs in this pass. Crazyflie-scale magnitudes differ greatly from the racing-quad sources and must be re-tuned.
