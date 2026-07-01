"""Point-mass-model (PMM) time-optimal reference primitives.

Builds a per-axis point-mass time-optimal reference path through drone-racing
gates: 1D bounded-acceleration double-integrator solves, composed across axes
for 3D synchronization, velocity sampling, and graph-based path selection.

References:
----------
.. [1] A. Romero, S. Sun, P. Foehn, and D. Scaramuzza,
       "Model Predictive Contouring Control for Time-Optimal Quadrotor
       Flight," IEEE Transactions on Robotics, 2022. arXiv:2108.13205.
.. [2] A. Romero, R. Penicka, and D. Scaramuzza,
       "Time-Optimal Online Replanning for Agile Quadrotor Flight,"
       IEEE Robotics and Automation Letters, 2022. arXiv:2203.09839.
"""

from __future__ import annotations

import heapq
import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Numerical tolerance for feasibility checks (t1, t2 in [0, T], |a| <= a_max) and
# for detecting the constant-velocity degenerate branch (|a| ~ 0).
_FEAS_TOL = 1e-9
_A_DEGENERATE_TOL = 1e-9

# Below this displacement two waypoints are treated as coincident, so the
# direction toward the next gate is undefined and falls back to the gate normal.
_DIR_DEGENERATE_TOL = 1e-9

# Slack on the pole-capsule clearance test ``dist >= radius - tol``. Tiny: the
# capsule radius already carries its safety margin upstream, so this only
# absorbs floating-point round-off at the boundary.
_CLEARANCE_TOL = 1e-6

# Interval-intersection synchronization (:func:`_sync_time`) parameters: bracket
# + bisect each infeasible axis's upper feasibility boundary ``T_high_i``.
_SYNC_BRACKET_GROWTH = 1.5  # geometric factor to grow the feasible upper bracket
_SYNC_BRACKET_MAX_FACTOR = 64.0  # cap the bracket at T_lo * this factor
_SYNC_BRACKET_ABS_CAP = 30.0  # absolute ceiling on the bracket time, in seconds
_SYNC_BISECT_TOL = 1e-3  # bisection tolerance on the feasibility boundary, in seconds
_SYNC_MAX_RAISES = 4  # max times T_star is raised to clear a later axis's gap


def min_time_1d(x0: float, v0: float, x1: float, v1: float, a_max: float) -> float | None:
    """Minimum time for a 1D double integrator |a|<=a_max to go (x0,v0)->(x1,v1).

    Bang-bang with one switch; tries both control orderings and returns the
    smaller feasible total time.

    Parameters
    ----------
    x0, v0 : float
        Initial position and velocity.
    x1, v1 : float
        Target position and velocity.
    a_max : float
        Acceleration bound (the control magnitude is at most ``a_max``).

    Returns:
    -------
    float or None
        The minimum maneuver time, or ``None`` if no feasible bang-bang
        profile reaches the target.
    """
    dx = x1 - x0
    best: float | None = None
    for a1 in (a_max, -a_max):
        disc = 0.5 * (v0 * v0 + v1 * v1) + a1 * dx
        if disc < 0.0:
            continue
        root = math.sqrt(disc)
        for t2 in ((-v1 + root) / a1, (-v1 - root) / a1):
            t1 = t2 + (v1 - v0) / a1
            if t1 >= -1e-9 and t2 >= -1e-9:
                total = max(t1, 0.0) + max(t2, 0.0)
                if best is None or total < best:
                    best = total
    return best


def fixed_time_profile_1d(
    x0: float, v0: float, x1: float, v1: float, a_max: float, T: float
) -> Callable[[npt.ArrayLike], npt.NDArray[np.float64]]:
    """Single-switch bang-bang profile reaching ``(x1, v1)`` in exactly time ``T``.

    Signed control magnitude ``a`` applied ``+a`` then ``-a`` (a negative ``a``
    flips the ordering); the root whose switch times lie in ``[0, T]`` and whose
    ``|a| <= a_max`` is selected.

    Parameters
    ----------
    x0, v0 : float
        Initial position and velocity.
    x1, v1 : float
        Target position and velocity at time ``T``.
    a_max : float
        Acceleration bound; the control magnitude is at most ``a_max``.
    T : float
        Total maneuver time. Must satisfy ``T >= min_time_1d(...)``; otherwise no
        bang-bang profile of magnitude ``<= a_max`` can reach the target.

    Returns:
    -------
    callable
        ``pos(t)`` mapping a scalar or array of times ``t in [0, T]`` to position,
        vectorized over a NumPy array. Phase 1 (``0 <= t <= t1``) uses ``+a``;
        phase 2 (``t1 < t <= T``) uses ``-a`` from the phase-1 endpoint.

    Raises:
    ------
    ValueError
        If ``T <= 0`` or neither root yields a feasible profile (``T`` is below
        the 1D minimum time for the given ``a_max``).
    """
    if T <= 0.0:
        raise ValueError(f"fixed-time profile needs T > 0, got T={T}")

    dv = v1 - v0
    pos_const = (x1 - x0) - 0.5 * (v0 + v1) * T

    # Constant-velocity degenerate branch: a == 0 solves the constraints exactly
    # only when there is no velocity change and the displacement matches v0*T.
    if abs(dv) < _A_DEGENERATE_TOL and abs(pos_const) < _A_DEGENERATE_TOL:
        return lambda t: x0 + v0 * np.asarray(t, dtype=np.float64)

    disc = 4.0 * pos_const * pos_const + T * T * dv * dv
    if disc < 0.0:  # guarded; disc is a sum of squares so this is unreachable
        raise ValueError("fixed-time profile discriminant negative (no real root)")
    root = math.sqrt(disc)

    for accel in ((2.0 * pos_const + root) / (T * T), (2.0 * pos_const - root) / (T * T)):
        if abs(accel) < _A_DEGENERATE_TOL:
            # |a| ~ 0 reduces to constant velocity; only valid if it closes the gap.
            if abs(dv) < _A_DEGENERATE_TOL and abs(pos_const) < _A_DEGENERATE_TOL:
                return lambda t: x0 + v0 * np.asarray(t, dtype=np.float64)
            continue
        t1 = 0.5 * (T + dv / accel)
        t2 = T - t1
        if -_FEAS_TOL <= t1 <= T + _FEAS_TOL and -_FEAS_TOL <= t2 <= T + _FEAS_TOL:
            if abs(accel) <= a_max + _FEAS_TOL:
                return _make_two_phase_pos(x0, v0, accel, t1, T)

    raise ValueError(
        f"no feasible fixed-time profile for T={T} (likely T < min_time_1d; "
        f"need a larger T or smaller |a|)"
    )


def _make_two_phase_pos(
    x0: float, v0: float, accel: float, t1: float, T: float
) -> Callable[[npt.ArrayLike], npt.NDArray[np.float64]]:
    """Build the vectorized position evaluator for a two-phase bang-bang profile.

    Parameters
    ----------
    x0, v0 : float
        Initial position and velocity.
    accel : float
        Signed control magnitude (``+accel`` in phase 1, ``-accel`` in phase 2).
    t1 : float
        Switch time (phase-1 duration).
    T : float
        Total maneuver time.

    Returns:
    -------
    callable
        ``pos(t)`` mapping scalar/array times in ``[0, T]`` to position.
    """
    xs = x0 + v0 * t1 + 0.5 * accel * t1 * t1  # phase-1 endpoint position
    vs = v0 + accel * t1  # phase-1 endpoint velocity

    def pos(t: npt.ArrayLike) -> npt.NDArray[np.float64]:
        ts = np.asarray(t, dtype=np.float64)
        phase1 = x0 + v0 * ts + 0.5 * accel * ts * ts
        tau = ts - t1
        phase2 = xs + vs * tau - 0.5 * accel * tau * tau
        return np.where(ts <= t1, phase1, phase2)

    return pos


def pmm_segment(
    p0: npt.ArrayLike,
    v0: npt.ArrayLike,
    p1: npt.ArrayLike,
    v1: npt.ArrayLike,
    a_max: float,
    n_samples: int,
) -> tuple[npt.NDArray[np.float64], float]:
    """3D point-mass time-optimal segment with axes synchronized by intersection.

    Each axis runs a decoupled fixed-time bang-bang profile of duration
    ``T_star``, the smallest time feasible for every axis (interval
    intersection, see :func:`_sync_time`; Romero et al., RA-L 2022,
    arXiv:2203.09839) rather than ``max_i(min_time_i)``, which can land in a
    fast axis's feasibility gap.

    Parameters
    ----------
    p0 : array_like, shape (3,)
        Initial position.
    v0 : array_like, shape (3,)
        Initial velocity.
    p1 : array_like, shape (3,)
        Target position.
    v1 : array_like, shape (3,)
        Target velocity.
    a_max : float
        Per-axis acceleration bound.
    n_samples : int
        Number of time samples (inclusive of both endpoints).

    Returns:
    -------
    pts : ndarray, shape (n_samples, 3)
        Sampled positions along the synchronized segment.
    T_star : float
        Segment duration (smallest time feasible for every axis).

    Raises:
    ------
    ValueError
        If no time is feasible for all three axes (an axis has no minimum time, or
        no common feasible time exists within the synchronization search cap).
    """
    p0 = np.asarray(p0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)

    t_star = _sync_time(p0, v0, p1, v1, a_max)
    if t_star is None:
        raise ValueError("no common feasible synchronization time for all axes")

    ts = np.linspace(0.0, t_star, n_samples)
    pts: npt.NDArray[np.float64] = np.empty((n_samples, 3), dtype=np.float64)
    if t_star <= 0.0:  # no motion on any axis; all samples sit at the start
        pts[:] = p0
        return pts, t_star

    for i in range(3):
        prof = fixed_time_profile_1d(
            float(p0[i]), float(v0[i]), float(p1[i]), float(v1[i]), a_max, t_star
        )
        pts[:, i] = prof(ts)
    return pts, t_star


def _fixed_time_feasible_1d(
    x0: float, v0: float, x1: float, v1: float, a_max: float, T: float
) -> bool:
    """Whether a single-switch fixed-time bang-bang profile reaches the target.

    Mirrors the root-feasibility test of :func:`fixed_time_profile_1d` without
    allocating the position closure, so the planner can score an edge's
    reachability cheaply.

    Parameters
    ----------
    x0, v0 : float
        Initial position and velocity.
    x1, v1 : float
        Target position and velocity.
    a_max : float
        Acceleration bound.
    T : float
        Maneuver time to test.

    Returns:
    -------
    bool
        ``True`` if a feasible single-switch profile of magnitude ``<= a_max``
        reaches ``(x1, v1)`` in exactly ``T``.
    """
    if T <= 0.0:
        return False
    dv = v1 - v0
    pos_const = (x1 - x0) - 0.5 * (v0 + v1) * T
    if abs(dv) < _A_DEGENERATE_TOL and abs(pos_const) < _A_DEGENERATE_TOL:
        return True
    disc = 4.0 * pos_const * pos_const + T * T * dv * dv
    root = math.sqrt(disc)
    for accel in ((2.0 * pos_const + root) / (T * T), (2.0 * pos_const - root) / (T * T)):
        if abs(accel) < _A_DEGENERATE_TOL:
            continue
        t1 = 0.5 * (T + dv / accel)
        t2 = T - t1
        if -_FEAS_TOL <= t1 <= T + _FEAS_TOL and -_FEAS_TOL <= t2 <= T + _FEAS_TOL:
            if abs(accel) <= a_max + _FEAS_TOL:
                return True
    return False


def _axis_upper_feasible_time(
    x0: float, v0: float, x1: float, v1: float, a_max: float, t_lo: float
) -> float | None:
    """Smallest feasible fixed-time ``>= t_lo`` for one axis, by bracket + bisection.

    Grows a feasible upper bracket geometrically, then bisects the feasibility
    boundary, returning the feasible side.

    Parameters
    ----------
    x0, v0 : float
        Initial position and velocity.
    x1, v1 : float
        Target position and velocity.
    a_max : float
        Acceleration bound.
    t_lo : float
        Lower bound (infeasible for this axis); the search starts here.

    Returns:
    -------
    float or None
        The start ``T_high`` of the axis's terminal feasible interval (feasible
        side of the boundary), or ``None`` if no feasible time is found below the
        bracket cap.
    """
    cap = min(t_lo * _SYNC_BRACKET_MAX_FACTOR, _SYNC_BRACKET_ABS_CAP)
    t_hi = t_lo * _SYNC_BRACKET_GROWTH
    while t_hi <= cap and not _fixed_time_feasible_1d(x0, v0, x1, v1, a_max, t_hi):
        t_hi *= _SYNC_BRACKET_GROWTH
    if not _fixed_time_feasible_1d(x0, v0, x1, v1, a_max, t_hi):
        return None  # no feasible time within the cap -> edge is genuinely infeasible

    lo, hi = t_lo, t_hi  # invariant: lo infeasible, hi feasible
    while hi - lo > _SYNC_BISECT_TOL:
        mid = 0.5 * (lo + hi)
        if _fixed_time_feasible_1d(x0, v0, x1, v1, a_max, mid):
            hi = mid
        else:
            lo = mid
    return hi  # feasible side of the boundary


def _sync_time(
    p0: npt.NDArray[np.float64],
    v0: npt.NDArray[np.float64],
    p1: npt.NDArray[np.float64],
    v1: npt.NDArray[np.float64],
    a_max: float,
) -> float | None:
    """Smallest fixed-time feasible for ALL three axes (interval intersection).

    A fast, short-displacement axis can have a gap before its terminal
    ``[T_high, inf)`` feasible interval, so the naive ``max_i(min_time_1d_i)``
    can drop an otherwise reachable edge. Synchronizes instead to the start of
    the intersection of the per-axis terminal feasible intervals (Romero et
    al., RA-L 2022, arXiv:2203.09839), raising ``T_star`` and re-checking
    (bounded by :data:`_SYNC_MAX_RAISES`) since a higher ``T_star`` can land in
    a different axis's gap.

    Parameters
    ----------
    p0, v0 : ndarray, shape (3,)
        Initial position and velocity.
    p1, v1 : ndarray, shape (3,)
        Target position and velocity.
    a_max : float
        Per-axis acceleration bound.

    Returns:
    -------
    float or None
        The synchronized segment time (feasible for all axes), ``0.0`` for the
        no-motion case, or ``None`` if any axis has no minimum time or no common
        feasible time exists within the search cap.
    """
    t_lo = 0.0
    for i in range(3):
        t_i = min_time_1d(float(p0[i]), float(v0[i]), float(p1[i]), float(v1[i]), a_max)
        if t_i is None:
            return None
        t_lo = max(t_lo, t_i)
    if t_lo <= 0.0:
        return 0.0

    t_star = t_lo
    for _ in range(_SYNC_MAX_RAISES):
        raised = t_star
        all_feasible = True
        for i in range(3):
            if _fixed_time_feasible_1d(
                float(p0[i]), float(v0[i]), float(p1[i]), float(v1[i]), a_max, t_star
            ):
                continue
            all_feasible = False
            t_high = _axis_upper_feasible_time(
                float(p0[i]), float(v0[i]), float(p1[i]), float(v1[i]), a_max, t_star
            )
            if t_high is None:
                return None  # this axis has no feasible time within the cap
            raised = max(raised, t_high)
        if all_feasible:
            return t_star
        t_star = raised
    # A final check after the last raise: every axis must be feasible at t_star.
    for i in range(3):
        if not _fixed_time_feasible_1d(
            float(p0[i]), float(v0[i]), float(p1[i]), float(v1[i]), a_max, t_star
        ):
            return None
    return t_star


def pmm_segment_time(
    p0: npt.ArrayLike, v0: npt.ArrayLike, p1: npt.ArrayLike, v1: npt.ArrayLike, a_max: float
) -> float | None:
    """Synchronized point-mass segment time, or ``None`` if unreachable.

    Cheap companion to :func:`pmm_segment` for graph search: returns just
    ``t_star`` without sampling positions. See :func:`_sync_time`.

    Parameters
    ----------
    p0, v0 : array_like, shape (3,)
        Initial position and velocity.
    p1, v1 : array_like, shape (3,)
        Target position and velocity.
    a_max : float
        Per-axis acceleration bound.

    Returns:
    -------
    float or None
        ``t_star`` (smallest time feasible for every axis), or ``None`` if no
        common feasible time exists.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.float64)
    return _sync_time(p0, v0, p1, v1, a_max)


def _unit(vec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | None:
    """Return ``vec`` normalized to unit length, or ``None`` if it is ~zero.

    Parameters
    ----------
    vec : ndarray, shape (3,)
        Vector to normalize.

    Returns:
    -------
    ndarray of shape (3,) or None
        The unit vector, or ``None`` when ``||vec||`` is below the degeneracy
        tolerance (so a direction cannot be defined).
    """
    norm = float(np.linalg.norm(vec))
    if norm < _DIR_DEGENERATE_TOL:
        return None
    return vec / norm


def sample_gate_velocities(
    gate_pos: npt.ArrayLike,
    next_pos: npt.ArrayLike | None,
    gate_normal: npt.ArrayLike | None,
    n_dir: int,
    n_mag: int,
    v_lo: float,
    v_hi: float,
    half_angle: float,
) -> npt.NDArray[np.float64]:
    """Cone-sample candidate velocity vectors at a gate.

    The cone axis points toward the *next* waypoint (the direction that carries
    momentum through the gate), falling back to ``gate_normal`` when
    ``next_pos`` is absent or coincident. Single-pass cone sampling; the
    iterative cone-refocus of Romero et al. (RA-L 2022, arXiv:2203.09839) is
    not implemented.

    Parameters
    ----------
    gate_pos : array_like, shape (3,)
        Gate center position (cone apex).
    next_pos : array_like, shape (3,) or None
        Next waypoint position. The cone axis points from ``gate_pos`` to it.
        If ``None`` (or coincident with ``gate_pos``), ``gate_normal`` is used.
    gate_normal : array_like, shape (3,) or None
        Fallback cone axis (gate facing direction) used when ``next_pos`` does
        not yield a valid direction.
    n_dir : int
        Number of directions sampled on the cone (>= 1). Direction 0 is the
        axis itself; the rest are spread on the cone surface.
    n_mag : int
        Number of speed magnitudes (>= 1) sampled in ``[v_lo, v_hi]``.
    v_lo, v_hi : float
        Inclusive speed range for the sampled magnitudes.
    half_angle : float
        Cone half-angle in radians.

    Returns:
    -------
    ndarray, shape (n_dir * n_mag, 3)
        Candidate velocity vectors, ordered direction-major then magnitude.

    Raises:
    ------
    ValueError
        If ``n_dir < 1`` or ``n_mag < 1``, or no valid cone axis can be formed
        (both ``next_pos`` and ``gate_normal`` degenerate).
    """
    if n_dir < 1 or n_mag < 1:
        raise ValueError(f"n_dir and n_mag must be >= 1, got n_dir={n_dir}, n_mag={n_mag}")

    gate_pos = np.asarray(gate_pos, dtype=np.float64)

    axis: npt.NDArray[np.float64] | None = None
    if next_pos is not None:
        axis = _unit(np.asarray(next_pos, dtype=np.float64) - gate_pos)
    if axis is None and gate_normal is not None:
        axis = _unit(np.asarray(gate_normal, dtype=np.float64))
    if axis is None:
        raise ValueError("no valid cone axis: next_pos and gate_normal are both degenerate")

    directions = _cone_directions(axis, n_dir, half_angle)
    magnitudes = np.linspace(v_lo, v_hi, n_mag)
    # Outer product over (direction, magnitude); flatten to (n_dir * n_mag, 3).
    return (directions[:, None, :] * magnitudes[None, :, None]).reshape(-1, 3)


def _cone_directions(
    axis: npt.NDArray[np.float64], n_dir: int, half_angle: float
) -> npt.NDArray[np.float64]:
    """Unit directions on a cone of given half-angle around ``axis``.

    Direction 0 is ``axis`` itself; the remaining ``n_dir - 1`` directions lie
    on the cone surface, azimuthally equispaced about ``axis``.

    Parameters
    ----------
    axis : ndarray, shape (3,)
        Unit cone axis.
    n_dir : int
        Number of directions (>= 1).
    half_angle : float
        Cone half-angle in radians.

    Returns:
    -------
    ndarray, shape (n_dir, 3)
        Unit direction vectors.
    """
    dirs: npt.NDArray[np.float64] = np.empty((n_dir, 3), dtype=np.float64)
    dirs[0] = axis
    if n_dir == 1:
        return dirs

    # Build an orthonormal basis (u, w) spanning the plane perpendicular to axis.
    # Pick the reference axis least aligned with `axis` for numerical stability.
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    w = np.cross(axis, u)

    azimuths = np.linspace(0.0, 2.0 * math.pi, n_dir - 1, endpoint=False)
    cos_h = math.cos(half_angle)
    sin_h = math.sin(half_angle)
    # cone(phi) = cos(h) * axis + sin(h) * (cos(phi) u + sin(phi) w).
    dirs[1:] = cos_h * axis[None, :] + sin_h * (
        np.cos(azimuths)[:, None] * u[None, :] + np.sin(azimuths)[:, None] * w[None, :]
    )
    return dirs


def _segment_clears_capsules(
    p0: npt.NDArray[np.float64],
    v0: npt.NDArray[np.float64],
    p1: npt.NDArray[np.float64],
    v1: npt.NDArray[np.float64],
    a_max: float,
    t_star: float,
    keepout_capsules: Sequence[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]],
    n_check: int,
) -> bool:
    """Whether a coarse sampling of a point-mass segment clears all keep-out capsules.

    A point clears a capsule iff its distance to the capsule axis segment is at
    least the capsule radius (margin already included, slackened by
    :data:`_CLEARANCE_TOL`). Reuses the caller's ``t_star`` to avoid re-solving
    the three 1D minimum-time problems; ``n_check`` is kept small, distinct from
    the dense ``n_per_seg`` used for the final path, to stay real-time.

    Parameters
    ----------
    p0, v0 : ndarray, shape (3,)
        Segment start position and velocity.
    p1, v1 : ndarray, shape (3,)
        Segment end position and velocity.
    a_max : float
        Per-axis acceleration bound.
    t_star : float
        Synchronized segment duration (``pmm_segment_time`` result for this edge).
        Must be ``> 0``; the caller already excludes the zero-motion case.
    keepout_capsules : sequence of (ndarray, ndarray, float)
        Capsules ``(c1, c2, radius)`` with endpoints ``c1, c2`` of shape ``(3,)``
        (the capsule axis segment, e.g. a vertical pole or a gate frame bar) and
        ``radius`` already including the safety margin. A sample point within
        ``radius`` of any capsule axis fails the clearance test.
    n_check : int
        Number of coarse samples along the segment (inclusive of endpoints).

    Returns:
    -------
    bool
        ``True`` if every sampled point clears every capsule by its radius.

    Notes:
    -----
    Coarse sampling can miss a brief between-samples dip; the capsule radius's
    upstream margin absorbs small overshoots.
    """
    ts = np.linspace(0.0, t_star, n_check)
    seg_pts: npt.NDArray[np.float64] = np.empty((n_check, 3), dtype=np.float64)
    for i in range(3):
        prof = fixed_time_profile_1d(
            float(p0[i]), float(v0[i]), float(p1[i]), float(v1[i]), a_max, t_star
        )
        seg_pts[:, i] = prof(ts)

    for c1, c2, radius in keepout_capsules:
        axis = c2 - c1
        axis_sq = float(np.dot(axis, axis))
        rel = seg_pts - c1  # (n_check, 3) offset from the capsule's first endpoint
        if axis_sq > _A_DEGENERATE_TOL:
            # Project each sample onto the capsule axis, clamped to the segment.
            proj = np.clip((rel @ axis) / axis_sq, 0.0, 1.0)
            closest = c1 + proj[:, None] * axis
        else:
            closest = np.broadcast_to(c1, seg_pts.shape)
        dist = np.linalg.norm(seg_pts - closest, axis=1)
        if bool(np.any(dist < radius - _CLEARANCE_TOL)):
            return False
    return True


def _gate_crossing_velocities(
    gate_pos: npt.NDArray[np.float64],
    next_pos: npt.NDArray[np.float64] | None,
    gate_normal: npt.NDArray[np.float64],
    v_n_min: float,
    n_dir: int,
    n_mag: int,
    v_lo: float,
    v_hi: float,
    half_angle: float,
    normal_weight: float = 0.5,
) -> npt.NDArray[np.float64]:
    """Candidate gate velocities constrained to cross the plane along +normal.

    Like :func:`sample_gate_velocities`, but the cone axis is re-centered on a
    ``normal_weight``-blend of the next-waypoint direction and the gate normal
    (``0.5`` = equal blend; ``1.0`` = pure normal), and samples below
    ``v_n_min`` normal component are dropped, falling back to the most
    normal-aligned samples if that empties the layer.

    Parameters
    ----------
    gate_pos : ndarray, shape (3,)
        Gate center (cone apex).
    next_pos : ndarray, shape (3,) or None
        Next waypoint; the cone axis blends ``next_pos - gate_pos`` with the
        normal. ``None`` (last gate) uses the normal alone.
    gate_normal : ndarray, shape (3,)
        Gate facing direction; crossing must have a positive component along it.
    v_n_min : float
        Minimum required ``dot(v, normalize(gate_normal))``.
    n_dir, n_mag, v_lo, v_hi, half_angle
        Cone-sampling parameters (see :func:`sample_gate_velocities`).

    Returns:
    -------
    ndarray, shape (<= n_dir * n_mag, 3)
        Candidate crossing velocities (at least one row).
    """
    n_hat = _unit(gate_normal)
    if n_hat is None:
        raise ValueError("gate_normal is degenerate")
    nd = _unit(np.asarray(next_pos, dtype=np.float64) - gate_pos) if next_pos is not None else None
    axis = _unit((1.0 - normal_weight) * nd + normal_weight * n_hat) if nd is not None else n_hat
    if axis is None:  # next direction antiparallel to the normal -> cross along normal
        axis = n_hat
    dirs = _cone_directions(axis, n_dir, half_angle)
    mags = np.linspace(v_lo, v_hi, n_mag)
    vels = (dirs[:, None, :] * mags[None, :, None]).reshape(-1, 3)
    v_n = vels @ n_hat
    keep = vels[v_n >= v_n_min]
    if keep.shape[0] == 0:
        keep = vels[np.argsort(v_n)[-n_mag:]]  # most +normal-aligned samples
    return keep


def plan_pmm_path(
    waypoints: Sequence[npt.ArrayLike] | npt.NDArray[np.float64],
    start_vel: npt.ArrayLike,
    a_max: float,
    n_dir: int,
    n_mag: int,
    v_lo: float,
    v_hi: float,
    half_angle: float,
    n_per_seg: int,
    keepout_capsules: Sequence[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]]
    | None = None,
    corridor_check_samples: int = 10,
    gate_normals: Sequence[npt.ArrayLike | None] | None = None,
    cross_v_n_min: float = 0.0,
    cross_normal_weight: float = 0.5,
) -> tuple[npt.NDArray[np.float64], float] | tuple[None, str]:
    """Min-time point-mass path through gate waypoints via layered Dijkstra.

    Each gate is a layer of candidate ``(position, velocity)`` nodes (velocities
    from :func:`sample_gate_velocities`); layer 0 is the start node. Edge cost
    is the point-mass segment time; Dijkstra finds the min-total-time chain,
    whose segments are re-sampled and concatenated into one path. Velocity-
    search method of Romero et al. (RA-L 2022, arXiv:2203.09839).

    Parameters
    ----------
    waypoints : sequence of array_like, shape (3,)
        Ordered positions ``[p_start, gate_1, ..., gate_K]`` (``p_start`` is the
        current drone position; the rest are gate centers).
    start_vel : array_like, shape (3,)
        Velocity at the start node.
    a_max : float
        Per-axis acceleration bound passed to :func:`pmm_segment`.
    n_dir, n_mag : int
        Cone direction / magnitude sample counts per gate.
    v_lo, v_hi : float
        Speed range for the gate velocity samples.
    half_angle : float
        Cone half-angle in radians for the gate velocity samples.
    n_per_seg : int
        Number of samples per segment (inclusive of both endpoints).
    keepout_capsules : sequence of (ndarray, ndarray, float) or None, optional
        Keep-out capsules ``(c1, c2, radius)`` with axis endpoints ``c1, c2`` of
        shape ``(3,)`` (a pole's vertical segment, or a gate frame bar) and
        ``radius`` already including the safety margin. An edge is REJECTED if
        any coarse sample on its (curved) point-mass segment comes within
        ``radius`` of any capsule axis -- lets the line bulge off the straight
        waypoint chord while still clearing obstacles, unlike a straight-line
        polytope. Gate frame bar radii leave the aperture open (so a gate's own
        frame doesn't block its waypoint) while still routing around bystander
        gate frames. ``None`` (default) disables the obstacle check.
    corridor_check_samples : int, optional
        Number of COARSE samples per segment used for the capsule clearance test
        (default 10). Deliberately much smaller than ``n_per_seg`` to keep the
        per-edge cost real-time; the capsule radii carry their own margin
        upstream. Unused when ``keepout_capsules`` is ``None``.

    Returns:
    -------
    tuple of (ndarray of shape (M, 3), float)
        The concatenated path and its total time, on success.
    tuple of (None, str)
        ``(None, reason)`` if no feasible path exists (too-short waypoint list,
        no reachable velocity assignment, or no assignment clearing every
        keep-out capsule). Never silently falls back to a gate-pinned line; the
        caller must act conservatively (hold, slow, or hover).

    Raises:
    ------
    ValueError
        If ``n_per_seg < 2`` (a segment needs both endpoints).
    """
    if n_per_seg < 2:
        raise ValueError(f"n_per_seg must be >= 2, got {n_per_seg}")

    pts = [np.asarray(wp, dtype=np.float64) for wp in waypoints]
    n_wp = len(pts)
    if n_wp < 2:
        return None, f"need at least 2 waypoints (start + 1 gate), got {n_wp}"

    n_gates = n_wp - 1  # layers 1..n_gates

    # Velocity samples per gate layer. The last gate has no "next" waypoint, so
    # its cone axis falls back to the previous-segment direction.
    layer_vels: list[npt.NDArray[np.float64]] = []
    for i in range(1, n_wp):
        if i < n_wp - 1:
            next_pos: npt.NDArray[np.float64] | None = pts[i + 1]
            normal: npt.NDArray[np.float64] | None = None
        else:
            next_pos = None
            normal = pts[i] - pts[i - 1]  # carry the incoming heading through
        gnorm: npt.NDArray[np.float64] | None = None
        if gate_normals is not None and gate_normals[i] is not None:
            gnorm = np.asarray(gate_normals[i], dtype=np.float64)
        if gnorm is not None and cross_v_n_min > 0.0:
            # Structural gate-crossing constraint (replaces blind anchors): force a
            # +normal velocity component so the min-time path crosses the plane.
            layer_vels.append(
                _gate_crossing_velocities(
                    pts[i],
                    next_pos,
                    gnorm,
                    cross_v_n_min,
                    n_dir,
                    n_mag,
                    v_lo,
                    v_hi,
                    half_angle,
                    normal_weight=cross_normal_weight,
                )
            )
        else:
            layer_vels.append(
                sample_gate_velocities(
                    pts[i], next_pos, normal, n_dir, n_mag, v_lo, v_hi, half_angle
                )
            )

    # Node ids: 0 = start; gate i (1..n_gates) nodes are offset blocks.
    layer_offset = [0]  # layer_offset[i] = first node id of layer i
    layer_offset.append(1)  # layer 1 starts at id 1
    for i in range(1, n_gates):
        layer_offset.append(layer_offset[-1] + len(layer_vels[i - 1]))
    n_nodes = layer_offset[-1] + len(layer_vels[-1])

    dist: npt.NDArray[np.float64] = np.full(n_nodes, np.inf)
    prev: npt.NDArray[np.int64] = np.full(n_nodes, -1, dtype=np.int64)
    dist[0] = 0.0

    start_vel = np.asarray(start_vel, dtype=np.float64)

    def node_state(
        node: int, layer: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Position and velocity of a node (layer 0 is the start node)."""
        if layer == 0:
            return pts[0], start_vel
        return pts[layer], layer_vels[layer - 1][node - layer_offset[layer]]

    # Layered DAG: edges only go from layer L to layer L+1. A forward sweep with a
    # heap is a clean Dijkstra; the layering guarantees no negative or back edges.
    heap: list[tuple[float, int, int]] = [(0.0, 0, 0)]  # (dist, node, layer)
    while heap:
        d, node, layer = heapq.heappop(heap)
        if d > dist[node]:
            continue
        if layer == n_gates:
            continue  # last layer has no outgoing edges
        p_a, v_a = node_state(node, layer)
        next_layer = layer + 1
        base = layer_offset[next_layer]
        for j, v_b in enumerate(layer_vels[next_layer - 1]):
            # Cheap time-only edge cost during search; positions are sampled only
            # for the chosen chain below. None = unreachable pairing -> skip it.
            seg_t = pmm_segment_time(p_a, v_a, pts[next_layer], v_b, a_max)
            if seg_t is None:  # this velocity pairing is unreachable; route around it
                continue
            # Only sample positions for the capsule clearance check once the edge
            # passed the cheap reachability test -- doomed edges never pay the
            # sampling cost. Reuse the known seg_t so the helper need not re-solve
            # min-time. A zero-duration edge is a single point at p_a; the
            # endpoints checked at adjacent edges already cover it, so it is
            # trivially clear.
            if (
                keepout_capsules is not None
                and seg_t > 0.0
                and not _segment_clears_capsules(
                    p_a,
                    v_a,
                    pts[next_layer],
                    v_b,
                    a_max,
                    seg_t,
                    keepout_capsules,
                    corridor_check_samples,
                )
            ):
                continue  # bulges into a keep-out capsule -> reject this velocity pairing
            nb = base + j
            nd = d + seg_t
            if nd < dist[nb]:
                dist[nb] = nd
                prev[nb] = node
                heapq.heappush(heap, (nd, nb, next_layer))

    # Cheapest node in the last layer.
    last_base = layer_offset[n_gates]
    last_slice = dist[last_base : last_base + len(layer_vels[-1])]
    best_local = int(np.argmin(last_slice))
    total_time = float(last_slice[best_local])
    if not math.isfinite(total_time):
        if keepout_capsules is not None:
            return None, "pmm_infeasible: no keepout-clear velocity assignment"
        return None, "no feasible point-mass path through the gate velocity samples"
    best_node = last_base + best_local

    # Reconstruct the node chain start -> ... -> best_node.
    chain: list[int] = []
    cur = best_node
    while cur != -1:
        chain.append(cur)
        cur = int(prev[cur])
    chain.reverse()

    # Re-evaluate the chosen segments and concatenate, dropping the shared gate
    # point between consecutive segments to keep the path continuous and not
    # double-counted.
    segments: list[npt.NDArray[np.float64]] = []
    for layer in range(1, n_gates + 1):
        p_a, v_a = node_state(chain[layer - 1], layer - 1)
        p_b, v_b = node_state(chain[layer], layer)
        seg_pts = pmm_segment(p_a, v_a, p_b, v_b, a_max, n_per_seg)[0]
        segments.append(seg_pts if layer == 1 else seg_pts[1:])
    path = np.concatenate(segments, axis=0)
    return path, total_time
