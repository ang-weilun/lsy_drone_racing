import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

class Capsule(NamedTuple):
    """Capsule representation for obstacle avoidance.
    
    A capsule is defined by a line segment between `start` and `end` and a `radius`.
    It can optionally be associated with a specific gate.
    """
    p1: NDArray
    p2: NDArray
    radius: float
    is_gate: bool
    gate_idx: int | None = None


def get_obstacle_capsules(obstacles_pos: NDArray | list[NDArray], config) -> list[Capsule]:
    """Generate capsules for cylindrical obstacles.
    
    Args:
        obstacles_pos: List or array of obstacle positions [N, 3]
        config: EnvironmentConfig containing safety margins and dimensions
    Returns:
        List of obstacle capsules.
    """
    capsules = []
    margin = config.safety_margin
    radius = config.pole_radius + margin

    for p in obstacles_pos:
        capsules.append(
            Capsule(
                np.array([p[0], p[1], 0.0]),
                np.array([p[0], p[1], config.pole_height]),
                radius,
                False,
            )
        )
    return capsules


def get_gate_capsules(gates_pos: NDArray | list[NDArray], gates_quat: NDArray | list[NDArray], config) -> list[Capsule]:
    """Generate capsules for gate frames and stands.
    
    Args:
        gates_pos: List or array of gate positions [N, 3]
        gates_quat: List or array of gate quaternions [N, 4]
        config: EnvironmentConfig containing safety margins and dimensions
    Returns:
        List of gate capsules.
    """
    capsules = []
    margin = config.safety_margin

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

        # Top horizontal bar
        capsules.append(
            Capsule(
                pos + up * bar_dist - right * half_outer,
                pos + up * bar_dist + right * half_outer,
                bar_radius,
                True,
                gate_i,
            )
        )
        # Bottom horizontal bar
        capsules.append(
            Capsule(
                pos - up * bar_dist - right * half_outer,
                pos - up * bar_dist + right * half_outer,
                bar_radius,
                True,
                gate_i,
            )
        )
        # Right vertical bar (from drone perspective)
        capsules.append(
            Capsule(
                pos - up * bar_dist + right * bar_dist,
                pos + up * bar_dist + right * bar_dist,
                bar_radius,
                True,
                gate_i,
            )
        )
        # Left vertical bar (from drone perspective)
        capsules.append(
            Capsule(
                pos - up * bar_dist - right * bar_dist,
                pos + up * bar_dist - right * bar_dist,
                bar_radius,
                True,
                gate_i,
            )
        )

    return capsules
