"""Geometry utilities for computing drone path corridors and obstacle avoidance."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.planner.util_types import Capsule, FlightCorridor, SkeletonPoint

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.sfc_planner_mpc_config import PlannerConfig


def closest_points_segments(
    p1: NDArray, q1: NDArray, p2: NDArray, q2: NDArray
) -> tuple[NDArray, NDArray]:
    """Find the closest points between two line segments.

    Args:
        p1: Start point of the first segment.
        q1: End point of the first segment.
        p2: Start point of the second segment.
        q2: End point of the second segment.

    Returns:
        A tuple containing the closest points on both segments.
    """
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)

    if a <= 1e-6 and e <= 1e-6:
        return p1, p2
    if a <= 1e-6:
        s = np.clip(f / e, 0.0, 1.0)
        return p1, p2 + s * d2

    c = np.dot(d1, r)
    if e <= 1e-6:
        t = np.clip(-c / a, 0.0, 1.0)
        return p1 + t * d1, p2

    b = np.dot(d1, d2)
    denom = a * e - b * b

    if denom != 0.0:
        t = np.clip((b * f - c * e) / denom, 0.0, 1.0)
    else:
        t = 0.0

    s = (b * t + f) / e
    if s < 0.0:
        s = 0.0
        t = np.clip(-c / a, 0.0, 1.0)
    elif s > 1.0:
        s = 1.0
        t = np.clip((b - c) / a, 0.0, 1.0)

    return p1 + t * d1, p2 + s * d2


def get_all_obstacle_capsules(
    obstacles_pos: NDArray, gates_pos: NDArray, gates_quat: NDArray, config: PlannerConfig
) -> list[Capsule]:
    """Create capsule representations for all obstacles and gates.

    Args:
        obstacles_pos: Array of positions for cylindrical obstacles.
        gates_pos: Array of gate positions.
        gates_quat: Array of gate orientations (quaternions).
        config: Planner configuration containing dimensions and margins.

    Returns:
        A list of Capsule objects representing the obstacles and gate structures.
    """
    capsules = []
    margin = config.safety_margin

    for p in obstacles_pos:
        capsules.append(
            Capsule(
                np.array([p[0], p[1], 0.0]),
                np.array([p[0], p[1], config.pole_height]),
                config.pole_radius + margin,
                False,
            )
        )

    for gate_i, (pos, quat) in enumerate(zip(gates_pos, gates_quat)):
        rot = R.from_quat(quat)
        up = rot.apply([0, 0, 1])
        right = rot.apply([0, 1, 0])

        stand_h = pos[2] - config.gate_outer / 2.0
        if stand_h > 0:
            capsules.append(
                Capsule(
                    pos - up * (config.gate_outer / 2.0),
                    pos - up * (config.gate_outer / 2.0 + stand_h),
                    config.gate_stand_radius + margin,
                    True,
                    gate_i,
                )
            )

        bar_dist = config.gate_bar_dist
        bar_radius = config.gate_bar_radius + margin
        half_outer = config.gate_outer / 2.0

        capsules.append(
            Capsule(
                pos + up * bar_dist - right * half_outer,
                pos + up * bar_dist + right * half_outer,
                bar_radius,
                True,
                gate_i,
            )
        )
        capsules.append(
            Capsule(
                pos - up * bar_dist - right * half_outer,
                pos - up * bar_dist + right * half_outer,
                bar_radius,
                True,
                gate_i,
            )
        )
        capsules.append(
            Capsule(
                pos - right * bar_dist + up * half_outer,
                pos - right * bar_dist - up * half_outer,
                bar_radius,
                True,
                gate_i,
            )
        )
        capsules.append(
            Capsule(
                pos + right * bar_dist + up * half_outer,
                pos + right * bar_dist - up * half_outer,
                bar_radius,
                True,
                gate_i,
            )
        )

    return capsules


def generate_flight_corridors(
    skeleton_path: list[SkeletonPoint],
    capsules: list[Capsule],
    gates_pos: NDArray,
    gates_quat: NDArray,
    config: PlannerConfig,
) -> list[FlightCorridor]:
    """Generate convex flight corridors around a path skeleton.

    Args:
        skeleton_path: The sequence of skeleton points representing the reference path.
        capsules: The obstacle capsules to avoid.
        gates_pos: Array of gate positions.
        gates_quat: Array of gate orientations (quaternions).
        config: Planner configuration containing corridor sizing limits.

    Returns:
        A list of FlightCorridor objects, each containing the separating hyperplanes
        (A, b) defining the safe region for the corresponding path segment.
    """
    gate_normals = R.from_quat(gates_quat).apply([1.0, 0.0, 0.0]) if len(gates_quat) > 0 else []
    corridors = []

    for i in range(len(skeleton_path) - 1):
        pt1 = skeleton_path[i]
        pt2 = skeleton_path[i + 1]
        corr = FlightCorridor(
            pt1.pos,
            pt2.pos,
            limit_low=np.array(config.ROOM_LIMIT_LOW),
            limit_high=np.array(config.ROOM_LIMIT_HIGH),
        )

        for cap in capsules:
            if cap.is_gate and cap.gate_idx is not None:
                if (
                    getattr(pt1, "gate_idx", None) == cap.gate_idx
                    or getattr(pt2, "gate_idx", None) == cap.gate_idx
                ):
                    g_pos = gates_pos[cap.gate_idx]
                    g_normal = gate_normals[cap.gate_idx]
                    d1 = float(np.dot(pt1.pos - g_pos, g_normal))
                    d2 = float(np.dot(pt2.pos - g_pos, g_normal))
                    near_radius = config.gate_outer / 2.0 + 0.10
                    if d1 * d2 < 0.0:
                        t = -d1 / (d2 - d1)
                        crossing = pt1.pos + t * (pt2.pos - pt1.pos)
                        if np.linalg.norm(crossing - g_pos) < near_radius:
                            continue
                    elif d1 == 0.0 and np.linalg.norm(pt1.pos - g_pos) < near_radius:
                        continue
                    elif d2 == 0.0 and np.linalg.norm(pt2.pos - g_pos) < near_radius:
                        continue

            c1, c2 = closest_points_segments(pt1.pos, pt2.pos, cap.p1, cap.p2)
            vec = c1 - c2
            dist = np.linalg.norm(vec)

            if dist > 1e-5:
                n = vec / dist
            else:
                d1_vec = pt2.pos - pt1.pos
                perp = (
                    np.array([-d1_vec[1], d1_vec[0], 0])
                    if np.linalg.norm(d1_vec[:2]) > 1e-5
                    else np.array([1.0, 0.0, 0.0])
                )
                n = perp / np.linalg.norm(perp)

            effective_radius = min(cap.radius, dist - 0.005)
            plane_p = c2 + n * effective_radius
            corr.add_halfspace(-n, plane_p)

        corridors.append(corr)
    return corridors
