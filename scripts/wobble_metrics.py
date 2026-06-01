"""Quantify twitchiness/wobble from a dumped action trace (.npz).

Reads a trace produced by ``scripts/dump_action_trace.py`` and computes
scalar wobble metrics that isolate *jerky* flight from *aggressive-but-smooth*
flight. The headline metric is command jerk — the per-step change in the
``[roll, pitch, yaw, thrust]`` command — which is exactly what the ``r_smooth``
reward term penalizes. A sustained aggressive roll has high body rates but low
jerk; wobble is rapid command reversals → high jerk and frequent sign changes.

Usage
-----
    pixi run -e rl-train python scripts/wobble_metrics.py --trace /tmp/trace.npz
    # compare two traces:
    pixi run -e rl-train python scripts/wobble_metrics.py \
        --trace /tmp/before.npz --baseline /tmp/after.npz
"""

from __future__ import annotations

import numpy as np

ENV_FREQ_HZ: float = 50.0  # control rate; sets the per-step dt for rate units


def _sign_changes(signal: np.ndarray) -> int:
    """Count zero-crossings of a 1-D signal (oscillation proxy)."""
    s = np.sign(signal)
    s = s[s != 0]
    if s.size < 2:
        return 0
    return int(np.sum(np.abs(np.diff(s)) > 0))


def wobble_metrics(trace: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute scalar wobble metrics from a trace dict.

    Parameters
    ----------
    trace : dict of ndarray
        Must contain ``env_action`` (T, 4) = ``[roll, pitch, yaw, thrust]`` and
        ``ang_vel`` (T, 3) body rates. Optional ``tau_scaled`` (T, 3).

    Returns
    -------
    metrics : dict of float
        ``cmd_jerk_mean`` / ``cmd_jerk_p95`` — L2 norm of the per-step command
        delta (the r_smooth proxy), mean and 95th percentile.
        ``angvel_rms`` — RMS body-rate magnitude (aggression level; NOT wobble).
        ``angvel_jerk_mean`` — mean per-step change in body rate (rad/s per step).
        ``angvel_sign_changes_per_s`` — body-rate axis reversals per second
        (high = oscillation/wobble), averaged over roll/pitch/yaw.
    """
    env_action = np.asarray(trace["env_action"], dtype=np.float64)  # (T, 4)
    ang_vel = np.asarray(trace["ang_vel"], dtype=np.float64)  # (T, 3)
    n_steps = env_action.shape[0]
    duration_s = n_steps / ENV_FREQ_HZ

    cmd_delta = np.diff(env_action, axis=0)  # (T-1, 4)
    cmd_jerk = np.linalg.norm(cmd_delta, axis=-1)  # (T-1,)

    angvel_mag = np.linalg.norm(ang_vel, axis=-1)  # (T,)
    angvel_delta = np.linalg.norm(np.diff(ang_vel, axis=0), axis=-1)  # (T-1,)

    sign_changes = sum(_sign_changes(ang_vel[:, ax]) for ax in range(3))
    sign_changes_per_s = sign_changes / max(duration_s, 1e-9) / 3.0

    return {
        "cmd_jerk_mean": float(cmd_jerk.mean()),
        "cmd_jerk_p95": float(np.percentile(cmd_jerk, 95)),
        "angvel_rms": float(np.sqrt(np.mean(angvel_mag**2))),
        "angvel_jerk_mean": float(angvel_delta.mean()),
        "angvel_sign_changes_per_s": float(sign_changes_per_s),
        "duration_s": float(duration_s),
        "finished": bool(np.asarray(trace.get("finished", False))),
    }


def _print_metrics(label: str, metrics: dict[str, float]) -> None:
    """Print one trace's metrics in a compact block."""
    print(f"=== {label} ===")
    print(f"  duration:                  {metrics['duration_s']:.2f} s"
          f"  finished={metrics['finished']}")
    print(f"  cmd_jerk_mean (r_smooth):  {metrics['cmd_jerk_mean']:.5f}  <- headline wobble")
    print(f"  cmd_jerk_p95:              {metrics['cmd_jerk_p95']:.5f}")
    print(f"  angvel_rms (aggression):   {metrics['angvel_rms']:.4f} rad/s")
    print(f"  angvel_jerk_mean:          {metrics['angvel_jerk_mean']:.5f} rad/s/step")
    print(f"  angvel_sign_changes/s:     {metrics['angvel_sign_changes_per_s']:.2f}  <- oscillation")


def main(trace: str, baseline: str | None = None) -> None:
    """Print wobble metrics for one trace, or a before/after comparison.

    Parameters
    ----------
    trace : str
        Path to the ``.npz`` trace (the "after"/primary trace).
    baseline : str, optional
        Path to a second ``.npz`` trace (the "before"). When given, prints both
        and the percentage change of each metric (trace relative to baseline).
    """
    primary = dict(np.load(trace, allow_pickle=True))
    m_primary = wobble_metrics(primary)

    if baseline is None:
        _print_metrics(trace, m_primary)
        return

    base = dict(np.load(baseline, allow_pickle=True))
    m_base = wobble_metrics(base)
    _print_metrics(f"BEFORE  {baseline}", m_base)
    print()
    _print_metrics(f"AFTER   {trace}", m_primary)
    print()
    print("=== change (after vs before) ===")
    for key in ("cmd_jerk_mean", "cmd_jerk_p95", "angvel_rms",
                "angvel_jerk_mean", "angvel_sign_changes_per_s"):
        b, a = m_base[key], m_primary[key]
        pct = (a - b) / b * 100.0 if abs(b) > 1e-12 else float("nan")
        print(f"  {key:28s} {b:.5f} -> {a:.5f}  ({pct:+.1f}%)")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
