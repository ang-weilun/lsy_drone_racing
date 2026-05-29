"""Vectorized stochastic evaluation on the clean level-3 seed set."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import fire
import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.sim.sim import seed_sim
from jax import Array
from jax.scipy.spatial.transform import Rotation as JaxRotation

from lsy_drone_racing.control.rl_sbx.checkpoint import load_all
from lsy_drone_racing.control.rl_sbx.policy import (
    FLAT_CONCAT_OBS_DIM,
    LOG_STD_INIT,
    NET_ARCH,
    Actor,
    Critic,
)
from lsy_drone_racing.control.rl_song import obs as obs_encoding
from lsy_drone_racing.control.rl_song.config import (
    ACTOR_OBS_DIM,
    ENV_ACTION_DIM,
    RAW_ACTION_DIM,
    CurriculumConfig,
    CurriculumStage,
    TrainConfig,
)
from lsy_drone_racing.control.rl_song.env_wrapper import RLSongVecEnv
from lsy_drone_racing.control.rl_song.policy import raw_to_env_action
from lsy_drone_racing.envs.race_core import EnvData, _reset_env_data, rng_spec2fn
from lsy_drone_racing.envs.race_core import obs as race_core_obs
from lsy_drone_racing.envs.randomize import build_random_track_fn

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "snapshots" / "ckpt_redesign_L3_600M"
CLEAN_SEEDS_PATH = REPO_ROOT / "snapshots" / "clean_l3_seeds.json"
STEP_PREFIX = "step_"


def main(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    *,
    seeds_path: str | Path = CLEAN_SEEDS_PATH,
    n: int | None = None,
    sample_seed: int = 0,
    device: str = "gpu",
    print_timing: bool = False,
) -> dict[str, Any]:
    """Run stochastic vectorized eval and print the legacy filtered-seed format."""
    seeds = _load_clean_seeds(Path(seeds_path))
    if n is not None:
        seeds = seeds[: int(n)]
    if not seeds:
        raise ValueError("No seeds selected for evaluation.")

    ckpt_dir = _resolve_checkpoint_path(Path(checkpoint))
    train_cfg = _eval_train_config(seed=int(seeds[0]))
    env = RLSongVecEnv(
        train_cfg, n_envs=len(seeds), stage_idx=0, seed=int(seeds[0]), device=device
    )
    _disable_autoreset(env)

    actor_template, critic_template = _checkpoint_templates()
    loaded = load_all(ckpt_dir, actor_template, critic_template)
    env.set_normalizer(loaded["actor_normalizer"])

    _reset_vec_env_with_clean_seeds(env, seeds)
    env.current_env_obs = _single_drone_obs(race_core_obs(env.env.data))
    env.prev_env_obs = env.current_env_obs
    env.prev_action_env_4vec = jnp.zeros((len(seeds), ENV_ACTION_DIM), dtype=jnp.float32)
    env.prev_physical_action = jnp.zeros((len(seeds), RAW_ACTION_DIM), dtype=jnp.float32)

    thrust_min, thrust_max = env.get_thrust_bounds()
    sample_keys = jax.random.split(jax.random.PRNGKey(sample_seed), len(seeds))
    started = time.perf_counter()
    lap_times, finished, _done = _eval_scan(
        env.env.data,
        loaded["actor_params"],
        loaded["actor_normalizer"],
        sample_keys,
        env.env._step,
        len(seeds),
        train_cfg.max_episode_steps,
        int(env.env.settings.freq),
        float(thrust_min),
        float(thrust_max),
        float(loaded["tangent_alpha_max_rad"]),
    )
    jax.block_until_ready(lap_times)
    elapsed = time.perf_counter() - started
    env.close()

    lap_times_np = np.asarray(lap_times)
    finished_np = np.asarray(finished, dtype=bool)
    results = [
        float(lap_times_np[i]) if bool(finished_np[i]) else None for i in range(len(seeds))
    ]
    _print_results(seeds, results)
    if print_timing:
        steps = train_cfg.max_episode_steps * len(seeds)
        print(f"eval wall time: {elapsed:.3f}s ({steps / elapsed:.1f} env-steps/s)")
    return {
        "seeds": seeds,
        "lap_times_s": results,
        "finished": int(np.sum(finished_np)),
        "total": len(seeds),
        "elapsed_s": elapsed,
    }


def _eval_train_config(seed: int) -> TrainConfig:
    stage = CurriculumStage(
        name="eval_level3_clean",
        level=3,
        use_domain_randomization=False,
        reset_pos_perturb_m=0.0,
        reset_vel_perturb_mps=0.0,
        reset_yaw_perturb_rad=0.0,
        gate_rand_scale=1.0,
        segment_init_prob=0.0,
        promote_target_gate_mean=float("inf"),
    )
    return replace(
        TrainConfig(),
        seed=seed,
        initial_stage_index=0,
        curriculum=CurriculumConfig(stages=(stage,)),
    )


def _checkpoint_templates() -> tuple[Any, Any]:
    obs = jnp.zeros((1, FLAT_CONCAT_OBS_DIM), dtype=jnp.float32)
    actor = Actor(action_dim=RAW_ACTION_DIM, net_arch=NET_ARCH, log_std_init=LOG_STD_INIT)
    critic = Critic(net_arch=NET_ARCH)
    actor_template = actor.init(jax.random.PRNGKey(0), obs)
    critic_template = critic.init(jax.random.PRNGKey(1), obs)
    return actor_template, critic_template


def _reset_vec_env_with_clean_seeds(env: RLSongVecEnv, seeds: list[int]) -> None:
    """Reset one vector env, then assign each slot its clean L3 track seed.

    ``SimCore.n_worlds`` is a static field, so this must not stack data from
    one-world envs.  Instead, keep the 50-world ``EnvData`` treedef and replace
    only the batched track leaves with values generated from each clean seed's
    single-world reset stream.
    """
    if env.env is None:
        raise RuntimeError("RLSongVecEnv has no underlying env.")
    if len(seeds) != env.n_envs:
        raise ValueError(f"Expected {env.n_envs} seeds, got {len(seeds)}.")

    env.reset(seed=int(seeds[0]))
    data = env.env.data
    race_cfg = env._load_stage_config(env.stage)
    randomizations = env._stage_randomizations(race_cfg, env.stage)
    subkeys = _single_world_track_subkeys(data.sim_data, seeds)

    randomization_count = int(bool(race_cfg.env.track.randomize))
    if randomizations is not None:
        randomization_count += len(randomizations)
    if randomization_count == 0:
        env.env.data = _reset_env_data(data, jnp.ones((env.n_envs,), dtype=bool))
        return

    keys = jax.vmap(lambda key: jax.random.split(key, randomization_count))(subkeys)
    key_idx = 0
    gates_pos = data.gates_pos
    gates_quat = data.gates_quat
    obstacles_pos = data.obstacles_pos
    nominal_gates_pos = data.nominal_gates_pos
    nominal_gates_quat = data.nominal_gates_quat
    nominal_obstacles_pos = data.nominal_obstacles_pos

    if race_cfg.env.track.randomize:
        generate_track = build_random_track_fn(
            [gate["pos"][2] for gate in race_cfg.env.track.gates],
            [obstacle["pos"][2] for obstacle in race_cfg.env.track.obstacles],
            race_cfg.env.track.safety_limits.pos_limit_low,
            race_cfg.env.track.safety_limits.pos_limit_high,
        )
        gates_pos, gates_quat, obstacles_pos = jax.vmap(generate_track)(keys[:, key_idx])
        nominal_gates_pos = gates_pos
        nominal_gates_quat = gates_quat
        nominal_obstacles_pos = obstacles_pos
        key_idx += 1

    if randomizations is not None:
        for target, spec in sorted(randomizations.items()):
            rng = rng_spec2fn(spec)
            slot_keys = keys[:, key_idx]
            key_idx += 1
            match target:
                case "gate_pos":
                    gates_pos = gates_pos + _vmap_single_world_rng(
                        rng, slot_keys, gates_pos.shape[1:]
                    )
                case "gate_rpy":
                    delta = _vmap_single_world_rng(rng, slot_keys, gates_pos.shape[1:])
                    gate_rpy = JaxRotation.from_quat(gates_quat).as_euler("xyz")
                    gates_quat = JaxRotation.from_euler("xyz", gate_rpy + delta).as_quat()
                case "obstacle_pos":
                    obstacles_pos = obstacles_pos + _vmap_single_world_rng(
                        rng, slot_keys, obstacles_pos.shape[1:]
                    )
                case _:
                    raise ValueError(f"Unexpected eval track randomization target: {target}")

    env.env.data = data.replace(
        gates_pos=gates_pos,
        gates_quat=gates_quat,
        obstacles_pos=obstacles_pos,
        nominal_gates_pos=nominal_gates_pos,
        nominal_gates_quat=nominal_gates_quat,
        nominal_obstacles_pos=nominal_obstacles_pos,
    )
    env.env.data = _reset_env_data(env.env.data, jnp.ones((env.n_envs,), dtype=bool))


def _single_world_track_subkeys(sim_data: Any, seeds: list[int]) -> Array:
    seed_array = jnp.asarray(seeds, dtype=jnp.uint32)

    def reset_subkey(seed: Array) -> Array:
        seeded = seed_sim(sim_data, seed, sim_data.core.device)
        return jax.random.split(seeded.core.rng_key, 2)[1]

    return jax.vmap(reset_subkey)(seed_array)


def _vmap_single_world_rng(rng: Any, keys: Array, slot_shape: tuple[int, ...]) -> Array:
    return jax.vmap(lambda key: rng(key, shape=(1, *slot_shape))[0])(keys)


def _disable_autoreset(env: RLSongVecEnv) -> None:
    if env.env is None:
        raise RuntimeError("RLSongVecEnv has no underlying env.")
    env.env.settings = env.env.settings.replace(autoreset=False)
    env.env._step = env.env.build_step_fn()


@jax.jit
def _sample_actions(actor_params: dict[str, Any], flat_obs: Array, sample_keys: Array) -> Array:
    dist = Actor(action_dim=RAW_ACTION_DIM, net_arch=NET_ARCH, log_std_init=LOG_STD_INIT).apply(
        actor_params, flat_obs
    )
    mu = dist.mean()
    std = dist.stddev()
    eps = jax.vmap(lambda key: jax.random.normal(key, (RAW_ACTION_DIM,)))(sample_keys)
    return mu + std * eps


def _eval_scan(
    env_data: EnvData,
    actor_params: dict[str, Any],
    normalizer: obs_encoding.NormalizerState,
    sample_keys: Array,
    env_step_fn: Any,
    n_envs: int,
    max_steps: int,
    env_freq_hz: int,
    thrust_min: float,
    thrust_max: float,
    alpha_max_rad: float,
) -> tuple[Array, Array, Array]:
    @jax.jit
    def run(data: EnvData, keys: Array) -> tuple[Array, Array, Array]:
        init = (
            data,
            jnp.zeros((n_envs, ENV_ACTION_DIM), dtype=jnp.float32),
            keys,
            jnp.zeros((n_envs,), dtype=bool),
            jnp.zeros((n_envs,), dtype=bool),
            jnp.full((n_envs,), jnp.nan, dtype=jnp.float32),
        )

        def body(carry: tuple[EnvData, Array, Array, Array, Array, Array], step_idx: Array):
            data, prev_action, keys, done_mask, finished_mask, lap_times = carry
            env_obs = _single_drone_obs(race_core_obs(data))
            actor_obs = obs_encoding.vmap_build_actor_obs(env_obs, prev_action, normalizer)
            flat_obs = jnp.concatenate(
                [actor_obs, jnp.zeros((n_envs, ACTOR_OBS_DIM), dtype=actor_obs.dtype)], axis=-1
            )
            split = jax.vmap(jax.random.split)(keys)
            next_keys = split[:, 0]
            action_keys = split[:, 1]
            raw_action = _sample_actions(actor_params, flat_obs, action_keys)
            env_action = raw_to_env_action(
                raw_action,
                env_obs["quat"],
                thrust_min,
                thrust_max,
                alpha_max=alpha_max_rad,
            )
            env_action = jnp.where(done_mask[:, None], jnp.zeros_like(env_action), env_action)

            stepped_data, (next_obs_full, _, terminated_full, truncated_full, _) = env_step_fn(
                data, env_action
            )
            stepped_data = _freeze_done_worlds(stepped_data, data, done_mask, n_envs)
            next_env_obs = _single_drone_obs(next_obs_full)
            terminated = terminated_full[:, 0].astype(jnp.bool_)
            truncated = truncated_full[:, 0].astype(jnp.bool_)
            finished = next_env_obs["target_gate"] < 0
            terminated = terminated | finished

            active = ~done_mask
            step_done = active & (terminated | truncated)
            step_finished = step_done & finished
            next_done_mask = done_mask | step_done
            next_finished_mask = finished_mask | step_finished
            lap_s = (step_idx.astype(jnp.float32) + 1.0) / float(env_freq_hz)
            next_lap_times = jnp.where(step_finished, lap_s, lap_times)
            next_prev_action = jnp.where(
                next_done_mask[:, None], jnp.zeros_like(env_action), env_action
            )
            return (
                stepped_data,
                next_prev_action,
                next_keys,
                next_done_mask,
                next_finished_mask,
                next_lap_times,
            ), None

        final, _ = jax.lax.scan(body, init, jnp.arange(max_steps, dtype=jnp.int32))
        _, _, _, done_mask, finished_mask, lap_times = final
        return lap_times, finished_mask, done_mask

    return run(env_data, sample_keys)


def _freeze_done_worlds(
    new_data: EnvData, old_data: EnvData, done_mask: Array, n_envs: int
) -> EnvData:
    def freeze(new_leaf: Any, old_leaf: Any) -> Any:
        if hasattr(new_leaf, "shape") and new_leaf.shape and new_leaf.shape[0] == n_envs:
            mask_shape = (n_envs,) + (1,) * (new_leaf.ndim - 1)
            return jnp.where(done_mask.reshape(mask_shape), old_leaf, new_leaf)
        return new_leaf

    return jax.tree_util.tree_map(freeze, new_data, old_data)


def _single_drone_obs(obs: dict[str, Array]) -> dict[str, Array]:
    return {key: jnp.asarray(value[:, 0]) for key, value in obs.items()}


def _load_clean_seeds(path: Path) -> list[int]:
    path = _repo_path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return [int(seed) for seed in payload["clean_seeds"]]


def _resolve_checkpoint_path(path: Path) -> Path:
    path = _repo_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if path.is_dir() and (path / "actor.params.msgpack").is_file():
        return path
    candidates: list[tuple[int, Path]] = []
    for child in path.glob(f"{STEP_PREFIX}*"):
        step = child.name.removeprefix(STEP_PREFIX)
        if child.is_dir() and step.isdecimal():
            candidates.append((int(step), child))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files or {STEP_PREFIX}* dirs under: {path}")
    return max(candidates, key=lambda item: item[0])[1]


def _repo_path(path: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _print_results(seeds: list[int], lap_times: list[float | None]) -> None:
    for seed, lap in zip(seeds, lap_times, strict=True):
        if lap is None:
            print(f"seed={seed:3d}: CRASH")
        else:
            print(f"seed={seed:3d}: {lap:.3f}")

    finished_times = [lap for lap in lap_times if lap is not None]
    print(f"finished: {len(finished_times)}/{len(seeds)}")
    if finished_times:
        arr = np.asarray(finished_times, dtype=np.float32)
        print(
            "lap times (s): "
            f"min={np.min(arr):.3f}, "
            f"mean={np.mean(arr):.3f}, "
            f"median={np.median(arr):.3f}, "
            f"max={np.max(arr):.3f}"
        )
    else:
        print("lap times (s): min=nan, mean=nan, median=nan, max=nan")


if __name__ == "__main__":
    fire.Fire(main, serialize=lambda _: None)
