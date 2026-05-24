"""SBX PPO training entry for the rl_sbx stack.

Usage:
-----
``pixi run -e rl-train python -m lsy_drone_racing.control.rl_sbx.train
--run-name v110_sbx_baseline_cold_300M --total-timesteps 300000000``.

Construction order:
------------------
1. Build a :class:`RLSongVecEnv` wrapper using the project's existing toml /
   curriculum machinery — this gives us the inner JAX env AND a working
   :meth:`RLSongVecEnv._apply_reset_perturbation` hook for L2 seg-init.
2. Construct :class:`RLSBXVecEnv` around ``wrapper.env`` with
   ``reset_done_hook=wrapper._apply_reset_perturbation``. This restores the
   seg-init invariant that the flat-concat vec-env autoreset path would
   otherwise have lost (cf. :class:`RLSBXVecEnv` ``reset_done_hook`` doc).
3. Build :class:`sbx.PPO` with our :class:`AsymmetricActorCriticPolicy` and
   the :class:`NormalizerUpdateCallback`.
4. Save the actor + critic + both normalizers to a step-keyed dir under
   ``checkpoint_root`` via :func:`save_step`.

Defaults match the v77 cold-train recipe — see
``docs/specs/2026-05-24-sbx-migration-design.md``.

References:
----------
Stable Baselines Jax (SBX), https://github.com/araffin/sbx, ``sbx/ppo``.
Song, Y. et al. (2023). Reaching the limit in autonomous racing.
*Science Robotics* 8, eadg1462.
"""

from __future__ import annotations

from pathlib import Path

import fire
from drone_models.core import load_params
from sbx import PPO

from lsy_drone_racing.control.rl_sbx.callbacks import NormalizerUpdateCallback
from lsy_drone_racing.control.rl_sbx.checkpoint import save_step
from lsy_drone_racing.control.rl_sbx.env_gym import RLSBXVecEnv
from lsy_drone_racing.control.rl_sbx.policy import AsymmetricActorCriticPolicy
from lsy_drone_racing.control.rl_song.config import TANGENT_ALPHA_MAX_RAD, RewardConfig, TrainConfig
from lsy_drone_racing.control.rl_song.env_wrapper import RLSongVecEnv

# Matches rl_song.controller / env_wrapper: the per-rotor sys_id thrust is
# scaled by the rotor count (4 for cf2x) to obtain the collective-thrust bound
# the attitude controller commands. Keep in sync with
# ``rl_song.env_wrapper.TOTAL_THRUST_MULTIPLIER``.
THRUST_MULTIPLIER: float = 4.0

# v77 cold-train recipe defaults; see design doc §M1.
DEFAULT_TOTAL_TIMESTEPS: int = 155_000_000
DEFAULT_ALPHA_MAX_RAD: float = TANGENT_ALPHA_MAX_RAD  # 0.16 rad, Schuck 2025 best
DEFAULT_ENT_COEF: float = 0.005  # cold-train start; constant for milestone 1
DEFAULT_LEARNING_RATE: float = 3e-4

# PPO rollout shape: with n_envs=16384 from PPOConfig the buffer holds
# n_envs * n_steps = 16384 * 256 = 4_194_304 transitions per iteration.
# minibatch_size is batch_size and divides the buffer into n_minibatches.
DEFAULT_N_STEPS: int = 256
DEFAULT_N_EPOCHS: int = 3
DEFAULT_BATCH_SIZE: int = 16384


def train(
    run_name: str,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    alpha_max_rad: float = DEFAULT_ALPHA_MAX_RAD,
    ent_coef: float = DEFAULT_ENT_COEF,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    n_envs: int | None = None,
    n_steps: int = DEFAULT_N_STEPS,
    n_epochs: int = DEFAULT_N_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 0,
    checkpoint_root: str = "lsy_drone_racing/control/rl_sbx/checkpoints",
) -> None:
    """Run SBX PPO cold-train against the milestone-1 L2 seg-init curriculum.

    Parameters
    ----------
    run_name : str
        Sub-directory name under ``checkpoint_root`` for this run's
        step-keyed checkpoints.
    total_timesteps : int, optional
        Total environment steps to train for. Defaults to the milestone-1
        budget (155M).
    alpha_max_rad : float, optional
        Per-step rotation budget on ``‖τ_scaled‖`` (rad) — controls the
        raw-to-env action projection cone half-angle. Defaults to the
        rl_song module-level constant.
    ent_coef : float, optional
        Constant PPO entropy bonus for the cold-train recipe.
    learning_rate : float, optional
        Adam learning rate; passed straight to ``sbx.PPO``.
    n_envs : int, optional
        Vectorization width. Defaults to ``TrainConfig.ppo.n_envs``.
    n_steps : int, optional
        PPO rollout length per env per iteration.
    n_epochs : int, optional
        PPO update epochs per rollout.
    batch_size : int, optional
        Minibatch size for PPO updates.
    seed : int, optional
        JAX env + PPO seed.
    checkpoint_root : str, optional
        Run directories are created at ``<checkpoint_root>/<run_name>``.

    Notes:
    -----
    The construction order (RLSongVecEnv → RLSBXVecEnv with seg-init hook
    → SBX PPO) is load-bearing — see the module docstring.
    """
    train_cfg = TrainConfig()
    effective_n_envs = train_cfg.ppo.n_envs if n_envs is None else int(n_envs)
    # v77 baseline reward: no gate_frame / obstacle weight, three-term Song
    # reward (r_prog + r_omega + r_terminal). RewardConfig defaults match.
    reward_cfg = RewardConfig()

    # The wrapper's __init__ instantiates the inner JAX env via set_stage and
    # runs its own reset(seed=seed+stage_idx) — no second reset call needed.
    wrapper = RLSongVecEnv(train_cfg, n_envs=effective_n_envs, stage_idx=0, seed=seed, device="gpu")

    # Mirror rl_song.env_wrapper.RLSongVecEnv: per-rotor thrust bounds from
    # the cf2x_L250 sys_id, scaled by the four-rotor count to give the
    # collective-thrust envelope the attitude controller commands.
    drone_params = load_params("sys_id", "cf2x_L250")
    thrust_min = float(drone_params["thrust_min"] * THRUST_MULTIPLIER)
    thrust_max = float(drone_params["thrust_max"] * THRUST_MULTIPLIER)

    env = RLSBXVecEnv(
        jax_env=wrapper.env,
        reward_cfg=reward_cfg,
        alpha_max=alpha_max_rad,
        thrust_min=thrust_min,
        thrust_max=thrust_max,
        n_envs=effective_n_envs,
        seed=seed,
        reset_done_hook=wrapper._apply_reset_perturbation,
    )

    # SBX 0.26 PPO kwargs: confirmed via inspect.signature on remote. ``device``
    # left at "auto" (JAX picks the default backend, which is the GPU the env
    # is already on).
    model = PPO(
        policy=AsymmetricActorCriticPolicy,
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=train_cfg.ppo.gamma,
        gae_lambda=train_cfg.ppo.gae_lambda,
        clip_range=train_cfg.ppo.clip_coef,
        ent_coef=ent_coef,
        vf_coef=train_cfg.ppo.vf_coef,
        max_grad_norm=train_cfg.ppo.max_grad_norm,
        target_kl=train_cfg.ppo.target_kl,
        seed=seed,
        verbose=1,
    )

    callbacks = [NormalizerUpdateCallback()]
    model.learn(total_timesteps=total_timesteps, callback=callbacks, log_interval=1)

    run_dir = Path(checkpoint_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    step_dir = save_step(
        run_dir=run_dir,
        global_step=int(model.num_timesteps),
        actor_params=model.policy.actor_state.params,
        critic_params=model.policy.vf_state.params,
        actor_normalizer=env.actor_normalizer,
        critic_normalizer=env.critic_normalizer,
        tangent_alpha_max_rad=alpha_max_rad,
    )
    print(f"Saved final checkpoint to {step_dir}")


if __name__ == "__main__":
    fire.Fire(train)
