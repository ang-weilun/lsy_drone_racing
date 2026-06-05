import time
import numpy as np
import fire
import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from pathlib import Path

from lsy_drone_racing.utils import load_config, load_controller

def profile(config_file="level0.toml", controller_file="sfc_mpcc.py", n_runs=1):
    config = load_config(Path(__file__).parents[1] / "config" / config_file)
    config.sim.render = False
    
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_cls = load_controller(control_path / controller_file)
    
    env = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)
    
    execution_times = []
    
    for _ in range(n_runs):
        obs, info = env.reset()
        controller = controller_cls(obs, info, config)
        
        while True:
            start = time.perf_counter()
            action = controller.compute_control(obs, info)
            end = time.perf_counter()
            execution_times.append(end - start)
            
            obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = controller.step_callback(action, obs, reward, terminated, truncated, info)
            if terminated or truncated or controller_finished:
                break
        controller.episode_callback()
    env.close()
    
    times_ms = np.array(execution_times) * 1000
    if len(times_ms) == 0:
        print("No execution times recorded.")
        return
        
    print("\n" + "="*40)
    print("--- Controller Profiling Results ---")
    print("="*40)
    print(f"Total control steps : {len(times_ms)}")
    print(f"Average time        : {np.mean(times_ms):.3f} ms ({1000/np.mean(times_ms):.1f} Hz)")
    print(f"Median time         : {np.median(times_ms):.3f} ms ({1000/np.median(times_ms):.1f} Hz)")
    print(f"Min time            : {np.min(times_ms):.3f} ms")
    print(f"Max time            : {np.max(times_ms):.3f} ms")
    print(f"99th percentile     : {np.percentile(times_ms, 99):.3f} ms")
    print("----------------------------------------")
    print("Target              : 500 Hz (2.0 ms per step)")
    print("="*40)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    fire.Fire(profile)
