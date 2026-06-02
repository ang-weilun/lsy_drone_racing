"""Checkpoint-free parity check between the JAX and numpy actor-obs encoders.

Compares ``rl_song.obs.build_actor_obs`` (JAX, training) against
``rl_sbx.deploy_numpy.obs.build_actor_obs`` (numpy, deploy) on a fixed fake
observation with matched identity normalizers (mean 0, var 1). Run at both
``RL_OBS_ANG_VEL`` settings to confirm the angular-velocity toggle stays in
lockstep across the two encoders, and that the output is finite and within the
training-time clip range.

Usage
-----
::

    RL_OBS_ANG_VEL=0 pixi run -e rl-train python scripts/check_obs_encoder_parity.py
    RL_OBS_ANG_VEL=1 pixi run -e rl-train python scripts/check_obs_encoder_parity.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.rl_sbx.deploy_numpy import obs as np_obs
from lsy_drone_racing.control.rl_sbx.deploy_numpy.normalizer import NORM_VAR_EPS, NormalizerState
from lsy_drone_racing.control.rl_song import obs as jax_obs
from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_DIM

# Track geometry for the fake observation.
_N_GATES: int = 4
_N_OBSTACLES: int = int(np_obs.N_OBSTACLES)
# Fixed seed for a deterministic diagnostic observation.
_RNG_SEED: int = 20260602
# Absolute/relative tolerance for JAX-vs-numpy encoder parity.
_TOL: float = 1e-5


def _fake_env_obs() -> dict[str, np.ndarray]:
    """Build a deterministic unbatched env observation with project shapes."""
    rng = np.random.default_rng(_RNG_SEED)
    quat = Rotation.from_euler("xyz", rng.normal(0.0, 0.2, size=3)).as_quat()
    gates_quat = Rotation.from_euler("xyz", rng.normal(0.0, 0.2, size=(_N_GATES, 3))).as_quat()
    return {
        "pos": rng.normal([0.0, 0.0, 0.7], [0.5, 0.5, 0.1]).astype(np.float32),
        "quat": quat.astype(np.float32),
        "vel": rng.normal(0.0, 0.4, size=3).astype(np.float32),
        "ang_vel": rng.normal(0.0, 0.2, size=3).astype(np.float32),
        "target_gate": np.asarray(1, dtype=np.int32),
        "gates_pos": rng.normal(
            np.linspace([0.6, -0.6, 0.8], [3.2, 0.6, 1.0], _N_GATES), 0.08
        ).astype(np.float32),
        "gates_quat": gates_quat.astype(np.float32),
        "obstacles_pos": rng.normal(
            np.linspace([0.5, 0.7, 1.0], [2.8, -0.7, 1.0], _N_OBSTACLES), 0.1
        ).astype(np.float32),
        "obstacles_visited": rng.choice([False, True], size=_N_OBSTACLES).astype(bool),
    }


def main() -> None:
    """Encode the fake obs with both encoders and assert they agree."""
    env_obs = _fake_env_obs()
    prev_action = np.zeros(4, dtype=np.float32)

    jax_norm = jax_obs.init_normalizer(ACTOR_OBS_DIM)
    np_norm = NormalizerState(
        mean=np.zeros(ACTOR_OBS_DIM, dtype=np.float32),
        var=np.ones(ACTOR_OBS_DIM, dtype=np.float32),
        count=np.asarray(NORM_VAR_EPS, dtype=np.float32),
    )

    jax_out = np.asarray(jax_obs.build_actor_obs(env_obs, prev_action, jax_norm), dtype=np.float32)
    np_out = np_obs.build_actor_obs(env_obs, prev_action, np_norm)

    if jax_out.shape != (ACTOR_OBS_DIM,):
        raise ValueError(f"jax obs shape {jax_out.shape} != ({ACTOR_OBS_DIM},)")
    if not (np.all(np.isfinite(jax_out)) and np.all(np.isfinite(np_out))):
        raise ValueError("non-finite observation")
    if np.max(np.abs(jax_out)) > jax_obs.NORM_CLIP + 1e-4:
        raise ValueError("observation exceeds NORM_CLIP")
    diff = float(np.max(np.abs(jax_out - np_out)))
    np.testing.assert_allclose(jax_out, np_out, atol=_TOL, rtol=_TOL)
    print(f"ACTOR_OBS_DIM={ACTOR_OBS_DIM} max_abs_diff={diff:.3e} OK")


if __name__ == "__main__":
    main()
