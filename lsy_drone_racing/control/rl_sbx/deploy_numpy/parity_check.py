"""One-shot parity check between JAX and numpy SBX deploy controllers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.rl_sbx.controller import RLSBXController
from lsy_drone_racing.control.rl_sbx.controller_numpy import RLSBXNumpyController
from lsy_drone_racing.control.rl_sbx.deploy_numpy.obs import N_OBSTACLES

# Number of gates in the current race-track observation layout.
_N_GATES: int = 4

# Fixed seed for deterministic diagnostic observations.
_RNG_SEED: int = 20260526

# Default drone model used by project level configs.
_DEFAULT_DRONE_MODEL: str = "cf21B_500"

# Default physics backend used by project level configs.
_DEFAULT_PHYSICS: str = "first_principles"

# Absolute and relative tolerance for action parity.
_ACTION_TOL: float = 1e-5


def main() -> None:
    """Run the parity check from a checkpoint path CLI argument."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path, help="SBX run directory or concrete step_* dir"
    )
    args = parser.parse_args()

    config = {
        "controller": {"checkpoint": str(args.checkpoint)},
        "sim": {"physics": _DEFAULT_PHYSICS, "drone_model": _DEFAULT_DRONE_MODEL},
    }
    env_obs = _fake_env_obs()

    jax_controller = RLSBXController(env_obs, {}, config)
    numpy_controller = RLSBXNumpyController(env_obs, {}, config)

    jax_action = np.asarray(jax_controller.compute_control(env_obs), dtype=np.float32)
    numpy_action = np.asarray(
        numpy_controller.compute_control(env_obs), dtype=np.float32
    )

    diff = np.max(np.abs(jax_action - numpy_action))
    np.testing.assert_allclose(
        jax_action, numpy_action, atol=_ACTION_TOL, rtol=_ACTION_TOL
    )
    print(f"max_abs_diff={diff:.8g}")
    print(f"jax_action={jax_action}")
    print(f"numpy_action={numpy_action}")


def _fake_env_obs() -> dict[str, npt.NDArray[np.generic]]:
    """Build a deterministic unbatched env observation with project shapes."""
    rng = np.random.default_rng(_RNG_SEED)
    quat = Rotation.from_euler("xyz", rng.normal(0.0, 0.2, size=3)).as_quat()
    gates_quat = Rotation.from_euler(
        "xyz", rng.normal(0.0, 0.2, size=(_N_GATES, 3))
    ).as_quat()
    return {
        "pos": rng.normal(
            loc=[0.0, 0.0, 0.7], scale=[0.5, 0.5, 0.1]
        ).astype(np.float32),
        "quat": quat.astype(np.float32),
        "vel": rng.normal(0.0, 0.4, size=3).astype(np.float32),
        "ang_vel": rng.normal(0.0, 0.2, size=3).astype(np.float32),
        "target_gate": np.asarray(1, dtype=np.int32),
        "gates_pos": rng.normal(
            loc=np.linspace([0.6, -0.6, 0.8], [3.2, 0.6, 1.0], _N_GATES),
            scale=0.08,
        ).astype(np.float32),
        "gates_quat": gates_quat.astype(np.float32),
        "obstacles_pos": rng.normal(
            loc=np.linspace([0.5, 0.7, 1.0], [2.8, -0.7, 1.0], N_OBSTACLES),
            scale=0.1,
        ).astype(np.float32),
        "obstacles_visited": rng.choice([False, True], size=N_OBSTACLES).astype(bool),
    }


if __name__ == "__main__":
    main()
