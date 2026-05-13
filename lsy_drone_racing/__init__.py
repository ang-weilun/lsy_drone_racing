"""LSY drone racing package for the Autonomous Drone Racing class @ TUM."""

import lsy_drone_racing.envs  # noqa: F401, register environments with gymnasium

try:
    # crazyflow is only available in pixi envs that include the `sim` feature.
    # The `lsy_drone_racing.rl.*` modules don't need it, so let those work on
    # a bare laptop install for smoke tests.
    from crazyflow.utils import enable_cache
except ImportError:
    pass
else:
    enable_cache()  # Enable persistent caching of jax functions
