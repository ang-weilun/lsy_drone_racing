"""Competition evaluation script.

Note:
    Please do not alter this script or ask the course supervisors first!
"""

import logging
from pathlib import Path

import numpy as np
from sim import simulate

from lsy_drone_racing.utils import load_config

logger = logging.getLogger(__name__)


def main():
    """Run the simulation N times and save the results as 'evaluation.csv'."""
    n_runs = 50
    config_file = "level2.toml"
    config = load_config(Path(__file__).parents[1] / "config" / config_file)
    ep_times = simulate(
        config=config_file, controller=config.controller.file, n_runs=n_runs, render=False
    )

    # Log the number of failed runs if any
    if n_failed := len([x for x in ep_times if x is None]):
        logger.warning(f"{n_failed} run{'' if n_failed == 1 else 's'} failed out of {n_runs}!")
    else:
        logger.info("All runs completed successfully!")

    # Abort if more than half of the runs failed
    if (success_rate := 1 - n_failed / n_runs) < 0.5:
        logger.error("More than 50% of all runs failed! Aborting evaluation.")
        raise RuntimeError("Too many runs failed!")

    successful_times = [x for x in ep_times if x is not None]
    successful_times_avg = np.mean(successful_times)
    successful_times_std = np.std(successful_times)
    successful_times_median = np.median(successful_times)
    successful_times_min = np.min(successful_times)
    successful_times_max = np.max(successful_times)
    
    logger.info(f"Average Time (s): {successful_times_avg:.3f}")
    logger.info(f"Std Dev (s): {successful_times_std:.3f}")
    logger.info(f"Median Time (s): {successful_times_median:.3f}")
    logger.info(f"Min Time (s): {successful_times_min:.3f}")
    logger.info(f"Max Time (s): {successful_times_max:.3f}")
    logger.info(f"Success Rate: {success_rate * 100}%")
    file = Path(__file__).parents[1] / "evaluation.csv"
    with open(file, "w") as f:
        f.write(f"{successful_times_avg},{success_rate},")
    logger.info(f"Results saved in {file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
