"""Run 20-seed deterministic eval on level0/1/2 in a single Python process.

Reusing one process means Python startup + module imports + (often) JIT cache
are amortized across the three eval matrices. ~2x faster than three back-to-back
``eval_sim`` invocations.

Emits the composite metric ``L2_lap_mean_finished_only / max(L2_finish_frac, 0.05)``
on stdout, lower-is-better. The per-level results are echoed for the
``post_hoc_select`` wrapper to grep.

Usage
-----
    pixi run -e rl-train python scripts/eval_3level.py \
        --checkpoint <run_or_view_dir> \
        --n-runs 20
"""

from __future__ import annotations

import math

import fire

from lsy_drone_racing.control.rl_song import eval_sim


def main(checkpoint: str, n_runs: int = 20) -> None:
    """Eval one checkpoint on level0/1/2 deterministically.

    Parameters
    ----------
    checkpoint : str
        Path passed to ``eval_sim.simulate(checkpoint=...)``. May be a run
        directory (uses the latest step) or a view directory holding a
        single ``step_NNN`` symlink plus the two config jsons.
    n_runs : int
        Episodes per level. Defaults to 20 to match the project's eval
        matrix convention.
    """
    finished = {}
    mean_lap = {}
    for level in (0, 1, 2):
        ep_times = eval_sim.simulate(
            config=f"level{level}.toml",
            checkpoint=checkpoint,
            control_mode="attitude",
            n_runs=n_runs,
            render=False,
        )
        ok = [t for t in ep_times if t is not None]
        finished[level] = len(ok)
        mean_lap[level] = sum(ok) / len(ok) if ok else math.nan
        lap_str = f"{mean_lap[level]:.3f}" if ok else "NA"
        print(f"L{level}: {finished[level]}/{n_runs} @ {lap_str}", flush=True)

    l2_finish_frac = finished[2] / n_runs
    if math.isnan(mean_lap[2]):
        metric = math.inf
    else:
        denom = max(l2_finish_frac, 0.05)
        metric = mean_lap[2] / denom
    metric_str = "inf" if math.isinf(metric) else f"{metric:.2f}"
    print(f"COMPOSITE_METRIC={metric_str}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
