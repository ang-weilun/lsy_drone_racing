"""Per-tick trace recording for the SFC-MPCC controller.

Enabled by setting the ``MPCC_TRACE_DIR`` environment variable to an output
directory. Each episode produces one compressed ``.npz`` archive with
per-tick state/solver telemetry, per-replan plan geometry, and episode
metadata. ``scripts/trace_autopsy.py`` consumes these files.

Disabled (env var unset) the controller never constructs a recorder, so the
flight code path only pays a single ``is None`` check per tick.

Note: tracing is sim-only in practice — ``scripts/deploy.py`` never calls
``episode_callback``, so traces are not saved on hardware.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import CubicSpline

TRACE_DIR_ENV_VAR = "MPCC_TRACE_DIR"
MAX_TICKS = 1500  # DroneRacing-v0 episode cap: 30 s at 50 Hz
SPLINE_SAMPLES = 200  # dense reference-spline samples stored per replan

# Keys stored per replan event, written to the npz as ``replan<NN>_<key>``.
# ``corridor_A``, ``corridor_b``, and ``corridor_offsets`` are stored for
# offline corridor debugging and are not read by ``scripts/trace_autopsy.py``.
REPLAN_KEYS = (
    "spline",
    "capsules",
    "corridor_A",
    "corridor_b",
    "corridor_offsets",
    "gates_pos",
    "gates_quat",
    "obstacles_pos",
)


def make_recorder_if_enabled(n_horizon: int) -> TraceRecorder | None:
    """Create a :class:`TraceRecorder` if ``MPCC_TRACE_DIR`` is set, else None.

    Args:
        n_horizon: Number of MPC shooting intervals N (the recorder stores N+1 nodes).

    Returns:
        A recorder writing into the configured directory, or None when tracing is off.
    """
    trace_dir = os.environ.get(TRACE_DIR_ENV_VAR, "")
    if not trace_dir:
        return None
    return TraceRecorder(Path(trace_dir), n_horizon)


def contour_lag_errors(pos: NDArray, theta: float, spline: CubicSpline) -> tuple[float, float]:
    """Compute contour (lateral) and lag (along-track) errors at the current progress.

    Mirrors the e_c / e_l decomposition used in the acados cost (sfc_mpcc.py),
    evaluated on the arc-length reference spline.

    Args:
        pos: Drone position, shape (3,).
        theta: Current path-progress parameter (arc length, m).
        spline: Arc-length parameterized reference ``CubicSpline``.

    Returns:
        Tuple ``(e_contour, e_lag)``: contour error magnitude (m, >= 0) and
        signed lag error (m, positive = ahead of the reference point).
    """
    p_d = spline(theta)
    tangent = spline(theta, 1)
    t_norm = (
        float(np.linalg.norm(tangent)) + 1e-6
    )  # mirrors t_norm epsilon in sfc_mpcc.py acados cost
    e = np.asarray(pos, dtype=np.float64) - p_d
    e_lag = float(np.dot(tangent, e) / t_norm)
    e_contour = float(np.linalg.norm(e - e_lag * tangent / t_norm))
    return e_contour, e_lag


class TraceRecorder:
    """Accumulates one episode of MPCC telemetry and writes it as one ``.npz``."""

    def __init__(self, out_dir: Path, n_horizon: int) -> None:
        """Preallocate per-tick buffers for a full-length episode.

        Args:
            out_dir: Directory to write trace files into (created if missing).
            n_horizon: Number of MPC shooting intervals N.
        """
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._n = 0
        n_nodes = n_horizon + 1

        self.pos = np.full((MAX_TICKS, 3), np.nan, np.float32)
        self.vel = np.full((MAX_TICKS, 3), np.nan, np.float32)
        self.quat = np.full((MAX_TICKS, 4), np.nan, np.float32)
        self.ang_vel = np.full((MAX_TICKS, 3), np.nan, np.float32)
        self.action = np.full((MAX_TICKS, 4), np.nan, np.float32)
        self.target_gate = np.full(MAX_TICKS, -2, np.int16)
        self.status = np.full(MAX_TICKS, -1, np.int8)
        self.solve_time = np.full(MAX_TICKS, np.nan, np.float32)
        self.fallback = np.zeros(MAX_TICKS, bool)
        self.theta = np.full(MAX_TICKS, np.nan, np.float32)
        self.v_theta = np.full(MAX_TICKS, np.nan, np.float32)
        self.e_contour = np.full(MAX_TICKS, np.nan, np.float32)
        self.e_lag = np.full(MAX_TICKS, np.nan, np.float32)
        self.horizon_pos = np.full((MAX_TICKS, n_nodes, 3), np.nan, np.float32)
        self.horizon_theta = np.full((MAX_TICKS, n_nodes), np.nan, np.float32)

        self._replans: list[dict[str, NDArray]] = []
        self._replan_ticks: list[int] = []
        self._replan_reasons: list[str] = []
        self._meta: dict[str, float | int | str] = {}
        self._terminated = False
        self._truncated = False
        self._last_target_gate = 0
        self._overflow_warned = False

    def set_meta(self, **meta: float | int | str) -> None:
        """Store episode-level metadata (seed, freq, n_gates, ...)."""
        self._meta.update(meta)

    def record_tick(
        self,
        obs: dict[str, NDArray],
        action: NDArray,
        status: int,
        solve_time: float,
        fallback: bool,
        theta: float,
        v_theta: float,
        e_contour: float,
        e_lag: float,
        horizon_pos: NDArray,
        horizon_theta: NDArray,
    ) -> None:
        """Record one control tick.

        Ticks beyond ``MAX_TICKS`` are silently dropped after a one-time warning;
        the valid prefix already captured is preserved and saved normally.

        Args:
            obs: Environment observation dict for this tick.
            action: Commanded ``[roll, pitch, yaw, thrust]``, shape (4,).
            status: acados solver return status.
            solve_time: Wall-clock seconds spent in the solve loop.
            fallback: True when the PD hover fallback produced the action.
            theta: Path progress at solve time (m).
            v_theta: Path progress rate at solve time (m/s).
            e_contour: Contour error magnitude (m).
            e_lag: Signed lag error (m).
            horizon_pos: Predicted positions over the horizon, shape (N+1, 3).
            horizon_theta: Predicted progress over the horizon, shape (N+1,).
        """
        i = self._n
        if i >= MAX_TICKS:
            # A full diagnostic buffer must never abort flight: keep the valid
            # prefix and drop further ticks (deploy episodes have no tick cap).
            if not self._overflow_warned:
                logger.warning(f"trace buffer full after {MAX_TICKS} ticks; dropping further ticks")
                self._overflow_warned = True
            return
        self.pos[i] = obs["pos"]
        self.vel[i] = obs["vel"]
        self.quat[i] = obs["quat"]
        self.ang_vel[i] = obs["ang_vel"]
        self.action[i] = action
        self.target_gate[i] = int(obs["target_gate"])
        self.status[i] = status
        self.solve_time[i] = solve_time
        self.fallback[i] = fallback
        self.theta[i] = theta
        self.v_theta[i] = v_theta
        self.e_contour[i] = e_contour
        self.e_lag[i] = e_lag
        self.horizon_pos[i] = horizon_pos
        self.horizon_theta[i] = horizon_theta
        self._n = i + 1

    def record_replan(self, tick: int, reason: str, **geometry: NDArray) -> None:
        """Record a (re)plan event with its full geometry.

        Args:
            tick: Controller tick at which the plan was built (0 = initial plan).
            reason: Planner-reported trigger (init / gate_passed / gate_jitter / ...).
            **geometry: Arrays for every key in :data:`REPLAN_KEYS`.
        """
        missing = set(REPLAN_KEYS) - set(geometry)
        if missing:
            raise ValueError(f"replan geometry missing keys: {sorted(missing)}")
        self._replan_ticks.append(tick)
        self._replan_reasons.append(reason)
        self._replans.append({k: np.asarray(geometry[k]) for k in REPLAN_KEYS})

    def record_step_result(self, target_gate: int, terminated: bool, truncated: bool) -> None:
        """Record post-step termination flags (called from ``step_callback``)."""
        self._last_target_gate = target_gate
        self._terminated = terminated
        self._truncated = truncated

    def save(self) -> Path:
        """Write the episode trace as one compressed ``.npz`` and return its path."""
        n = self._n
        finished = self._last_target_gate == -1
        n_gates = int(self._meta.get("n_gates", 0))
        gates_passed = n_gates if finished else max(0, self._last_target_gate)
        if finished:
            outcome = "finished"
        elif self._terminated:
            outcome = "crashed"
        else:
            outcome = "timeout"

        data: dict[str, NDArray] = {
            "pos": self.pos[:n],
            "vel": self.vel[:n],
            "quat": self.quat[:n],
            "ang_vel": self.ang_vel[:n],
            "action": self.action[:n],
            "target_gate": self.target_gate[:n],
            "status": self.status[:n],
            "solve_time": self.solve_time[:n],
            "fallback": self.fallback[:n],
            "theta": self.theta[:n],
            "v_theta": self.v_theta[:n],
            "e_contour": self.e_contour[:n],
            "e_lag": self.e_lag[:n],
            "horizon_pos": self.horizon_pos[:n],
            "horizon_theta": self.horizon_theta[:n],
            "replan_ticks": np.asarray(self._replan_ticks, np.int32),
            "replan_reasons": np.asarray(self._replan_reasons),
            "outcome": np.asarray(outcome),
            "gates_passed": np.asarray(gates_passed, np.int32),
            "final_tick": np.asarray(n, np.int32),
        }
        for i, replan in enumerate(self._replans):
            for key, value in replan.items():
                data[f"replan{i:02d}_{key}"] = value
        for key, value in self._meta.items():
            data[f"meta_{key}"] = np.asarray(value)

        seed = self._meta.get("seed", "x")
        path = self._out_dir / f"trace_seed{seed}_{time.time_ns()}.npz"
        np.savez_compressed(path, **data)
        return path
