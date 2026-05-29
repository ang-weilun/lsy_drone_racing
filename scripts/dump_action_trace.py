"""Dump one deterministic L0 episode's action trace and compute utilization ratios.

Loads an SBX checkpoint, runs one episode with deterministic mean actions,
captures per-step ``tau_scaled`` (body-frame per-step rotation command) and
``env_action`` ``[roll, pitch, yaw, thrust]``, then prints summary statistics
of how much of the drone's physical envelope the policy is using.

Usage:

    pixi run -e rl-train python scripts/dump_action_trace.py \\
        --checkpoint <path> --out /tmp/trace.npz [--config level0.toml]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fire
import gymnasium
import jax.numpy as jnp
import numpy as np
from drone_models.core import load_params
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.control.rl_sbx.controller import RLSBXController
from lsy_drone_racing.control.rl_song import obs as obs_encoding
from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_DIM
from lsy_drone_racing.control.rl_song.policy import (
    THRUST_RAW_DIM,
    raw_to_env_action,
    scale_tangent,
)
from lsy_drone_racing.utils import load_config

# 50 Hz env step.
ENV_FREQ_HZ: float = 50.0


def _run_one_episode(
    checkpoint: str,
    config: str,
) -> dict[str, np.ndarray]:
    """Run one deterministic episode and return per-step arrays."""
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / config)
    cfg.controller.checkpoint = checkpoint
    cfg.env.control_mode = "attitude"
    cfg.sim.render = False

    env = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    )
    env = JaxToNumpy(env)
    obs, info = env.reset()
    controller = RLSBXController(obs, info, dict(cfg))

    tau_scaled_list: list[np.ndarray] = []
    env_action_list: list[np.ndarray] = []
    target_gate_list: list[int] = []
    pos_list: list[np.ndarray] = []
    quat_list: list[np.ndarray] = []
    vel_list: list[np.ndarray] = []
    ang_vel_list: list[np.ndarray] = []
    terminated = False
    truncated = False

    while not (terminated or truncated):
        jax_obs = {k: jnp.asarray(v) for k, v in obs.items()}
        actor_obs = obs_encoding.build_actor_obs(
            jax_obs, controller.prev_action_env_4vec, controller.actor_normalizer
        )
        flat_obs = jnp.concatenate(
            [actor_obs, jnp.zeros((ACTOR_OBS_DIM,), dtype=actor_obs.dtype)], axis=-1
        )
        dist = controller._actor.apply(controller.actor_params, flat_obs[None, :])
        raw_action = dist.mean()[0]
        tau_raw = raw_action[THRUST_RAW_DIM:]
        tau_scaled = scale_tangent(tau_raw, controller.alpha_max_rad)
        env_action = raw_to_env_action(
            raw_action,
            jax_obs["quat"],
            controller.thrust_min,
            controller.thrust_max,
            alpha_max=controller.alpha_max_rad,
        )
        controller.prev_action_env_4vec = env_action

        tau_scaled_list.append(np.asarray(tau_scaled, dtype=np.float64))
        env_action_list.append(np.asarray(env_action, dtype=np.float64))
        target_gate_list.append(int(np.asarray(obs["target_gate"]).item()))
        pos_list.append(np.asarray(obs["pos"], dtype=np.float64))
        quat_list.append(np.asarray(obs["quat"], dtype=np.float64))
        vel_list.append(np.asarray(obs["vel"], dtype=np.float64))
        ang_vel_list.append(np.asarray(obs["ang_vel"], dtype=np.float64))

        obs, _reward, terminated, truncated, _info = env.step(
            np.asarray(env_action, dtype=np.float32)
        )

    finished = (
        int(np.asarray(obs["target_gate"]).item()) == -1
    )

    return {
        "tau_scaled": np.stack(tau_scaled_list),
        "env_action": np.stack(env_action_list),
        "target_gate": np.asarray(target_gate_list, dtype=np.int32),
        "pos": np.stack(pos_list),
        "quat": np.stack(quat_list),
        "vel": np.stack(vel_list),
        "ang_vel": np.stack(ang_vel_list),
        "alpha_max_rad": np.asarray(controller.alpha_max_rad),
        "thrust_min": np.asarray(controller.thrust_min),
        "thrust_max": np.asarray(controller.thrust_max),
        "finished": np.asarray(finished),
    }


def _print_envelope_ratios(trace: dict[str, np.ndarray], drone_model: str) -> None:
    """Print utilization stats comparing commanded actions to the drone's envelope."""
    params: dict[str, Any] = load_params("first_principles", drone_model)
    mass = float(params["mass"])
    thrust_min_motor = float(params["thrust_min"])
    thrust_max_motor = float(params["thrust_max"])
    thrust_min_total = 4.0 * thrust_min_motor
    thrust_max_total = 4.0 * thrust_max_motor
    weight_n = mass * 9.81

    alpha_max = float(trace["alpha_max_rad"])
    tau = trace["tau_scaled"]  # (T, 3)
    env_action = trace["env_action"]  # (T, 4)
    ang_vel = trace["ang_vel"]  # (T, 3) actual measured body rates
    thrust = env_action[:, 3]
    target_gate = trace["target_gate"]
    n_steps = tau.shape[0]

    print()
    print("=" * 64)
    print(f"Trace length: {n_steps} steps ({n_steps / ENV_FREQ_HZ:.2f} s)")
    print(f"Episode finished: {bool(trace['finished'])}")
    print(f"Drone model: {drone_model}, mass {mass*1000:.1f} g, weight {weight_n:.3f} N")
    print(
        f"Thrust envelope: min {thrust_min_total:.3f} N, max {thrust_max_total:.3f} N, "
        f"TWR {thrust_max_total/weight_n:.2f}"
    )
    print(f"alpha_max budget: {alpha_max:.3f} rad/step = "
          f"{alpha_max * ENV_FREQ_HZ:.1f} rad/s")
    print("=" * 64)

    # 1. tau_scaled utilization (per-step rotation magnitude / alpha_max).
    tau_norm = np.linalg.norm(tau, axis=-1)
    tau_ratio = tau_norm / max(alpha_max, 1e-9)
    print()
    print("[1] tau_scaled magnitude (commanded rotation per step):")
    print(f"    mean ratio:   {tau_ratio.mean():.3f} of alpha_max")
    print(f"    p50:          {np.percentile(tau_ratio, 50):.3f}")
    print(f"    p95:          {np.percentile(tau_ratio, 95):.3f}")
    print(f"    max:          {tau_ratio.max():.3f}")
    print(f"    fraction of steps at >0.9 of alpha_max: "
          f"{(tau_ratio > 0.9).mean():.1%}")
    print(f"    fraction of steps at <0.1 of alpha_max: "
          f"{(tau_ratio < 0.1).mean():.1%}")

    # 2. Per-axis tau_scaled utilization (signed, so we can see direction bias).
    axis_names = ("roll", "pitch", "yaw")
    print()
    print("[2] tau_scaled per-axis (commanded rotation per step, signed):")
    for axis_i, name in enumerate(axis_names):
        comp = tau[:, axis_i]
        ratio = comp / alpha_max
        print(f"    {name} (tau_{name[0]}/alpha_max):  "
              f"mean {ratio.mean():+.3f}, abs-mean {np.abs(ratio).mean():.3f}, "
              f"min {ratio.min():+.3f}, max {ratio.max():+.3f}, "
              f"|·|>0.9: {(np.abs(ratio) > 0.9).mean():.1%}")

    # 3. Equivalent body rate (commanded rate if held for the full step).
    omega_cmd = tau * ENV_FREQ_HZ  # rad/s
    omega_cmd_norm = np.linalg.norm(omega_cmd, axis=-1)
    print()
    print(f"[3] Equivalent commanded body rate (tau * {ENV_FREQ_HZ:.0f} Hz, rad/s):")
    print(f"    mean ||omega_cmd||: {omega_cmd_norm.mean():.2f} rad/s")
    print(f"    p95:                {np.percentile(omega_cmd_norm, 95):.2f} rad/s")
    print(f"    max:                {omega_cmd_norm.max():.2f} rad/s")
    print(f"    policy ceiling (alpha_max * 50): "
          f"{alpha_max * ENV_FREQ_HZ:.2f} rad/s")

    # 4. Actual measured body rate from obs (for comparison).
    omega_actual_norm = np.linalg.norm(ang_vel, axis=-1)
    print()
    print("[4] Actual measured body rate (env obs ang_vel, rad/s):")
    print(f"    mean ||omega_actual||: {omega_actual_norm.mean():.2f} rad/s")
    print(f"    p95:                   {np.percentile(omega_actual_norm, 95):.2f} rad/s")
    print(f"    max:                   {omega_actual_norm.max():.2f} rad/s")

    # 5. Thrust utilization.
    hover_frac = weight_n / thrust_max_total
    thrust_ratio = (thrust - thrust_min_total) / (thrust_max_total - thrust_min_total)
    thrust_above_hover = thrust - weight_n
    print()
    print("[5] Thrust:")
    print(f"    hover thrust:    {weight_n:.3f} N ({hover_frac:.1%} of max)")
    print(f"    mean thrust:     {thrust.mean():.3f} N ({thrust.mean()/thrust_max_total:.1%} of max)")
    print(f"    p50:             {np.percentile(thrust, 50):.3f} N "
          f"({np.percentile(thrust, 50)/thrust_max_total:.1%})")
    print(f"    p95:             {np.percentile(thrust, 95):.3f} N "
          f"({np.percentile(thrust, 95)/thrust_max_total:.1%})")
    print(f"    max:             {thrust.max():.3f} N ({thrust.max()/thrust_max_total:.1%})")
    print(f"    min:             {thrust.min():.3f} N ({thrust.min()/thrust_max_total:.1%})")
    print(f"    fraction of steps at >0.95 of max: "
          f"{(thrust > 0.95 * thrust_max_total).mean():.1%}")
    print(f"    fraction of steps at <1.05 of hover: "
          f"{(thrust < 1.05 * weight_n).mean():.1%}")
    print(f"    mean above hover: {thrust_above_hover.mean():+.3f} N "
          f"(net vertical accel {thrust_above_hover.mean()/mass:+.2f} m/s²)")

    # 6. Per-gate breakdown if multiple gates touched.
    print()
    print("[6] Per-gate target window (commanded action stats while heading to each gate):")
    for g in sorted(set(target_gate.tolist())):
        if g < 0:
            continue
        mask = target_gate == g
        n = int(mask.sum())
        if n < 5:
            continue
        tau_g = tau[mask]
        thrust_g = thrust[mask]
        omega_g = np.linalg.norm(tau_g * ENV_FREQ_HZ, axis=-1)
        print(
            f"    target={g} ({n} steps): "
            f"||omega_cmd|| mean {omega_g.mean():.2f} max {omega_g.max():.2f} rad/s, "
            f"thrust mean {thrust_g.mean()/thrust_max_total:.1%} max {thrust_g.max()/thrust_max_total:.1%} of max"
        )


def main(
    checkpoint: str,
    out: str = "/tmp/trace.npz",
    config: str = "level0.toml",
    drone_model: str = "cf21B_500",
) -> None:
    """Entry point. See module docstring for usage."""
    trace = _run_one_episode(checkpoint=checkpoint, config=config)
    np.savez(out, **trace)
    print(f"\nWrote trace: {out}")
    _print_envelope_ratios(trace, drone_model=drone_model)


if __name__ == "__main__":
    fire.Fire(main)
