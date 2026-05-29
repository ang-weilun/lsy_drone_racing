# Liu 2024 — gate-frame guidance reward (source extract)

Verbatim reward logic extracted from Liu 2024's open-source implementation
(`github.com/ErcBunny/IsaacGymEnvs`, default branch `main`), file
`isaacgymenvs/tasks/drone_racing/mdp/reward.py`, function `_compute_script`
(`@torch.jit.script`). Deployed weights from `isaacgymenvs/cfg/task/DRBase.yaml`.

This is the reference for the "augment" route (Route B). The key correction it
resolves: **Liu does not use path-projection progress.** Progress stays
centre-distance (Swift form); the geometric fix is a separate gate-frame
potential.

## Total reward

```
reward = r_progress + r_perception + r_cmd + r_collision + r_guidance + r_waypoint + r_timeout
```

## Progress (centre-distance, zeroed on the passing step)

```python
dist_to_wp = ‖wp_pos[next_wp_id] − drone_pos‖
r_progress = k_progress * (last_dist_to_wp − dist_to_wp) * (~wp_passing)
#   * (~wp_passing): progress is set to 0 on the step a waypoint is passed,
#   "to avoid undesired negative progress" at the target hand-off.
#   (Same defect class as our v85 "r_prog leak" fix.)
```

## Guidance (the gate-frame funnel + wrong-side rejection)

```python
# drone position expressed in the TARGET-GATE frame
x, y, z = quat_rotate_inverse(wp_q, drone_pos − wp_pos)
#   x = signed distance along the gate normal: + = behind/exit side, − = approach side
#   y, z = in-plane offset from the aperture centre
w, h = wp_width[next_wp_id], wp_height[next_wp_id]

# window along the gate-normal axis (triangular, peaks at the gate plane)
layer_x   = clamp(1 − |x| / guidance_x_thresh, min=0)     # guidance_x_thresh = 3.0 m
guidance_x = −(layer_x ** 2)                               # the −f²(x) term, ≤ 0

# aperture-aware in-plane Gaussian width
tol      = 0.5 if x > 0 else guidance_tol                  # guidance_tol = 0.2
yz_scale = (1 − guidance_x) * tol * sqrt((y²+z²) / ((y/w)² + (z/h)²))
#   the sqrt(...) makes the Gaussian width scale with the gate's half-extent in
#   the radial direction → automatically adapts to gate size.

guidance_yz = ( k_rejection * exp(−0.5 (y²+z²) / yz_scale)   if x > 0      # behind gate → reject
              else 1 − exp(−0.5 (y²+z²) / yz_scale) )                      # approaching → funnel to axis

r_guidance = k_guidance * guidance_x * guidance_yz
```

Behaviour:
- **Approach side (x ≤ 0):** `−f²·(1 − exp)` — zero on the gate axis, increasingly
  negative off-axis, strongest at the plane. A smooth funnel pulling the drone
  onto the gate axis *before* it reaches the gate → suppresses lateral frame-clip.
- **Behind / wrong side (x > 0):** `−f²·k_rejection·exp` — maximum penalty right
  behind the gate near its axis (≈ −k_guidance·k_rejection), decaying off-axis.
  A "do not be behind the gate" repulsion → discourages wrong-side approach and
  reversing back through the plane. Recomputed against the *current* target, so
  after passing N it acts on N+1.

## Other terms (for completeness)

```python
# perception — CAMERA term; not applicable to our mocap (no onboard camera)
r_perception = k_perception * exp(k_cam_dev * cam_dev**4)   # cam_dev = angle(camera_x_axis, dir_to_gate)
# command
r_cmd = k_cmd_ang_vel * ‖action[roll,pitch,yaw]‖ + k_cmd_diff * ‖action − last_action‖
# events
r_collision = k_collision * drone_collision        # terminal
r_waypoint  = k_waypoint  * wp_passing             # sparse gate-pass bonus
r_timeout   = k_timeout   * timeout
# extra (DRBase.yaml): velocity shaping
r_extra = k_vel_lateral * |v_lateral| + k_vel_backward * |v_backward|   # Liu Eq. 8 style
```

## Deployed weights (`DRBase.yaml` reward block)

| key | value | note |
|---|---|---|
| `k_progress` | 1.0 | centre-distance |
| `k_guidance` | 1.0 | **co-equal with progress** |
| `k_rejection` | 2.0 | wrong-side multiplier |
| `k_waypoint` | 5.0 | gate-pass bonus |
| `k_collision` | −10.0 | terminal |
| `k_timeout` | −10.0 | |
| `k_perception` / `k_cam_dev` | 0.02 / −10.0 | camera only — drop for mocap |
| `k_cmd_ang_vel` / `k_cmd_diff` | −4e-4 / −2e-4 | body-rate + action-smoothness |
| `guidance_x_thresh` | 3.0 | normal-axis window (**metres**) |
| `guidance_tol` | 0.2 | |
| `k_vel_lateral` / `k_vel_backward` | −1e-3 / −5e-3 | velocity shaping |

## Gate switching (`waypoint_tracker.py`)

Liu's `WaypointTracker._compute_script` is the same scheme as our `gate_passed`:
intersect the drone's path segment with the gate plane, require the parametric
`t ∈ [0,1)`, the correct crossing direction (`drone_pos_diff · x_axis > 0`),
the intersection inside the aperture (`|proj_y| < w/2`, `|proj_z| < h/2`), and the
previous waypoint already passed. `next_wp_id` = first not-yet-passed waypoint.

## Scale caveat

Liu's platform: 0.76 kg quad, gates 1.2–3.0 m, 25 Hz, with an onboard camera.
Ours: ~30 g Crazyflie, ~0.45 m gates, 50 Hz, mocap (no camera). The in-plane
geometry auto-scales through the aperture normalisation, but `guidance_x_thresh`
(3 m) and the co-equal `k_guidance` (= `k_progress`) must be re-tuned downward at
Crazyflie scale, and the camera perception term is dropped.

## Mapping to our code

Liu's guidance ≈ our existing **disabled** levers, of which Liu is an
aperture-aware refinement:
- guidance funnel + rejection ≈ our `dipole` (signed front/behind potential) +
  `wrong_side` penalty;
- velocity shaping ≈ our `vel_shaping` (`vel_lat_coef`, `vel_back_coef`);
- centre-progress + zero-on-pass ≈ our `r_prog` + the v85 leak fix.
