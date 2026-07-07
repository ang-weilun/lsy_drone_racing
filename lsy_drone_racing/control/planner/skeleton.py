"""Skeleton path planning for drone navigation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.planner.util_types import SkeletonPoint

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.sfc_planner_mpc_config import PlannerConfig


def build_analytical_skeleton(
    current_pos: NDArray,
    current_vel: NDArray,
    gates_pos: NDArray,
    gates_quat: NDArray,
    target_gate_idx: int,
    config: PlannerConfig,
) -> list[SkeletonPoint]:
    """Build an analytical path skeleton using cubic Hermite splines.

    Args:
        current_pos: The current position of the drone.
        current_vel: The current velocity of the drone.
        gates_pos: The positions of the gates.
        gates_quat: The orientations of the gates (quaternions).
        target_gate_idx: The index of the next gate to fly through.
        config: The planner configuration.

    Returns:
        A list of SkeletonPoint objects representing the raw reference path.
    """
    gate_normals = R.from_quat(gates_quat).apply([1.0, 0.0, 0.0]) if len(gates_quat) > 0 else []
    raw_path = [SkeletonPoint(current_pos, False, None, None, None)]

    def cubic_hermite_spline(
        p0: NDArray, m0: NDArray, p1: NDArray, m1: NDArray, t: float
    ) -> NDArray:
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    points_and_attrs = []

    points_and_attrs.append(
        {
            "pos": current_pos,
            "dir": current_vel / np.linalg.norm(current_vel)
            if np.linalg.norm(current_vel) > 1e-3
            else current_vel,
            "is_drone": True,
        }
    )

    for i in range(target_gate_idx, len(gates_pos)):
        pos = gates_pos[i].copy()
        normal = gate_normals[i].copy()
        rot = R.from_quat(gates_quat[i])
        right = rot.apply([0, 1, 0])
        up = rot.apply([0, 0, 1])

        pre_pos = pos - normal * config.pre_gate_entry_dist
        points_and_attrs.append(
            {"pos": pre_pos, "dir": normal, "gate_idx": i, "is_waypoint": True, "is_drone": False}
        )

        points_and_attrs.append(
            {
                "pos": pos,
                "dir": normal,
                "normal": normal,
                "right": right,
                "up": up,
                "gate_idx": i,
                "is_tube": True,
                "is_drone": False,
            }
        )

        post_pos = pos + normal * config.anchor_gap
        points_and_attrs.append(
            {"pos": post_pos, "dir": normal, "gate_idx": i, "is_waypoint": True, "is_drone": False}
        )

    if len(gates_pos) > 0 and target_gate_idx <= len(gates_pos):
        last_gate_idx = len(gates_pos) - 1
        last_pos = gates_pos[last_gate_idx]
        last_normal = gate_normals[last_gate_idx]
        finish_pos = last_pos + last_normal * config.FINISH_LINE_EXT_DIST
        points_and_attrs.append(
            {"pos": finish_pos, "dir": last_normal, "gate_idx": last_gate_idx, "is_drone": False}
        )

    for i in range(len(points_and_attrs) - 1):
        pt0 = points_and_attrs[i]
        pt1 = points_and_attrs[i + 1]
        dist = np.linalg.norm(pt1["pos"] - pt0["pos"])

        if pt0["is_drone"]:
            m0 = current_vel * config.HERMITE_TANGENT_SCALE_DRONE
        else:
            m0 = pt0["dir"] * dist * config.HERMITE_TANGENT_SCALE_GATE

        m1 = pt1["dir"] * dist * config.HERMITE_TANGENT_SCALE_GATE

        samples = config.HERMITE_SAMPLES_PER_SEGMENT
        is_tube = (
            pt0.get("is_tube", False)
            and pt1.get("is_tube", False)
            and pt0.get("gate_idx") == pt1.get("gate_idx")
        )
        for j in range(1, samples):
            t = j / samples
            pt = cubic_hermite_spline(pt0["pos"], m0, pt1["pos"], m1, t)
            raw_path.append(
                SkeletonPoint(
                    pt,
                    False,
                    pt1.get("normal") if is_tube else None,
                    pt1.get("right") if is_tube else None,
                    pt1.get("up") if is_tube else None,
                    gate_idx=pt1.get("gate_idx"),
                    is_in_tube=is_tube,
                )
            )

        raw_path.append(
            SkeletonPoint(
                pt1["pos"],
                "normal" in pt1,
                pt1.get("normal"),
                pt1.get("right"),
                pt1.get("up"),
                gate_idx=pt1.get("gate_idx"),
                is_in_tube=pt1.get("is_tube", False),
                is_waypoint=pt1.get("is_waypoint", False),
            )
        )

    return raw_path


def apply_3d_obstacle_repulsion(
    raw_path: list[SkeletonPoint],
    obstacles_pos: NDArray,
    gates_pos: NDArray,
    gates_quat: NDArray,
    config: PlannerConfig,
) -> list[SkeletonPoint]:
    """Apply a repulsion vector field to the path skeleton to avoid obstacles.

    Args:
        raw_path: The initial path skeleton.
        obstacles_pos: The positions of cylindrical obstacles.
        gates_pos: The positions of the gates.
        gates_quat: The orientations of the gates (quaternions).
        config: The planner configuration.

    Returns:
        A list of SkeletonPoint objects representing the repelled, safer path.
    """
    margin = config.OBSTACLE_AVOIDANCE_MARGIN
    capsules = []

    for p in obstacles_pos:
        capsules.append(
            (
                np.array([p[0], p[1], 0.0]),
                np.array([p[0], p[1], config.pole_height]),
                config.pole_radius + margin,
            )
        )

    for j, (p, q) in enumerate(zip(gates_pos, gates_quat)):
        rot = R.from_quat(q)
        right = rot.apply([0, 1, 0])
        up = rot.apply([0, 0, 1])
        bar_dist = config.gate_bar_dist
        obs_radius = config.gate_bar_radius + margin
        half_outer = config.gate_outer / 2.0
        capsules.append(
            (
                p + up * bar_dist - right * half_outer,
                p + up * bar_dist + right * half_outer,
                obs_radius,
            )
        )
        capsules.append(
            (
                p - up * bar_dist - right * half_outer,
                p - up * bar_dist + right * half_outer,
                obs_radius,
            )
        )
        capsules.append(
            (
                p - right * bar_dist + up * half_outer,
                p - right * bar_dist - up * half_outer,
                obs_radius,
            )
        )
        capsules.append(
            (
                p + right * bar_dist + up * half_outer,
                p + right * bar_dist - up * half_outer,
                obs_radius,
            )
        )

    path = raw_path
    for _ in range(3):
        new_path = []
        for i, pt in enumerate(path):
            if pt.is_gate or i == 0 or i == len(path) - 1:
                new_path.append(pt)
                continue

            curr_pos = pt.pos.copy()
            push_accum = np.zeros(3)

            for c1, c2, safe_radius in capsules:
                v = c2 - c1
                w = curr_pos - c1
                v_sq = np.dot(v, v)
                t = np.clip(np.dot(w, v) / v_sq, 0.0, 1.0) if v_sq > 1e-6 else 0.0
                closest = c1 + t * v
                diff = curr_pos - closest
                dist = np.linalg.norm(diff)

                if dist < safe_radius:
                    push_dir = diff / dist if dist > 1e-6 else np.array([1.0, 0.0, 0.0])
                    push_amount = safe_radius - dist + config.OBSTACLE_AVOIDANCE_PUSH_EXTRA
                    push_accum += push_dir * push_amount

            if np.linalg.norm(push_accum) > 0:
                new_pos = curr_pos + push_accum
                new_path.append(
                    SkeletonPoint(
                        new_pos, pt.is_gate, pt.gate_normal, pt.gate_right, pt.gate_up, pt.gate_idx
                    )
                )
            else:
                new_path.append(pt)
        path = new_path

    return path


def calculate_anchors(
    current_pos: NDArray,
    current_vel: NDArray,
    obstacles_pos: NDArray,
    gates_pos: NDArray,
    gates_quat: NDArray,
    target_gate_idx: int,
    config: PlannerConfig,
) -> list[SkeletonPoint]:
    """Calculate the final anchor points for the path skeleton.

    Args:
        current_pos: The current position of the drone.
        current_vel: The current velocity of the drone.
        obstacles_pos: The positions of cylindrical obstacles.
        gates_pos: The positions of the gates.
        gates_quat: The orientations of the gates (quaternions).
        target_gate_idx: The index of the next gate to fly through.
        config: The planner configuration.

    Returns:
        A list of SkeletonPoint objects clipped to the safety limits.
    """
    raw_path = build_analytical_skeleton(
        current_pos, current_vel, gates_pos, gates_quat, target_gate_idx, config
    )
    path = apply_3d_obstacle_repulsion(raw_path, obstacles_pos, gates_pos, gates_quat, config)

    low = np.array(config.CORRIDOR_LIMIT_LOW) + config.CORRIDOR_BUFFER
    high = np.array(config.CORRIDOR_LIMIT_HIGH) - config.CORRIDOR_BUFFER
    clipped_path = []
    for pt in path:
        clipped_pos = np.clip(pt.pos, low, high)
        clipped_path.append(
            SkeletonPoint(
                clipped_pos,
                pt.is_gate,
                pt.gate_normal,
                pt.gate_right,
                pt.gate_up,
                gate_idx=getattr(pt, "gate_idx", None),
            )
        )

    return clipped_path
