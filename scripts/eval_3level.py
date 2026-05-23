"""Run N-seed deterministic eval over multiple level configs in one process.

Reusing one process amortizes Python startup, module imports, and the JAX
JIT cache across configs. Roughly 2x faster than back-to-back ``eval_sim``
invocations.

Two metrics emitted on stdout:

- ``COMPOSITE_METRIC`` — historical L2 metric
  ``L2_lap_mean_finished_only / max(L2_finish_frac, 0.05)``, lower-is-better.
  Only emitted when level 2 is in ``--levels``.
- ``L3_FINISHES`` — finish count on level 3, primary metric for L3 work.
  Only emitted when level 3 is in ``--levels``.

Usage
-----
    pixi run -e rl-train python scripts/eval_3level.py \\
        --checkpoint <run_or_view_dir> \\
        --n-runs 20 \\
        --levels 0,1,2,3
"""

from __future__ import annotations

import math

import fire

from lsy_drone_racing.control.rl_song import eval_sim


def main(checkpoint: str, n_runs: int = 20, levels: str = "0,1,2") -> None:
    """Eval one checkpoint on the requested levels deterministically.

    Parameters
    ----------
    checkpoint : str
        Path passed to ``eval_sim.simulate(checkpoint=...)``. May be a run
        directory (uses the latest step) or a view directory holding a
        single ``step_NNN`` symlink plus the two config jsons.
    n_runs : int
        Episodes per level. Defaults to 20 to match the project's eval
        matrix convention.
    levels : str
        Comma-separated level indices to evaluate (e.g. ``"0,1,2"`` or
        ``"0,1,2,3"`` to include the L3 OOD config).
    """
    # fire auto-parses comma-separated args to a tuple; accept either form.
    if isinstance(levels, str):
        requested_levels = tuple(int(s) for s in levels.split(","))
    else:
        requested_levels = tuple(int(s) for s in levels)
    finished: dict[int, int] = {}
    mean_lap: dict[int, float] = {}
    for level in requested_levels:
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

    if 2 in finished:
        l2_finish_frac = finished[2] / n_runs
        if math.isnan(mean_lap[2]):
            metric = math.inf
        else:
            denom = max(l2_finish_frac, 0.05)
            metric = mean_lap[2] / denom
        metric_str = "inf" if math.isinf(metric) else f"{metric:.2f}"
        print(f"COMPOSITE_METRIC={metric_str}", flush=True)

    if 3 in finished:
        print(f"L3_FINISHES={finished[3]}/{n_runs}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
