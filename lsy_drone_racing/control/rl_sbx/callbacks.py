"""SBX callbacks for the rl_sbx stack.

``NormalizerUpdateCallback`` runs the Welford running-stat updates on the
env's two ``NormalizerState`` instances at each ``_on_rollout_end``. Updates
are independent — actor and critic normalizers are NEVER cross-pollinated.

``PeriodicCheckpointCallback`` writes a ``save_step`` checkpoint every
``save_freq_steps`` env steps so post-hoc selection works on long runs
(handoff note: "final step overtrains" on >100M cold-trains).

The rollout buffer stores normalized obs (the env wrapper normalizes before
returning). We invert the affine normalization to recover raw values, then
run ``update_normalizer`` on the raw batch. This mirrors
``RLSongVecEnv.update_normalizer_from_batch``.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from lsy_drone_racing.control.rl_sbx.checkpoint import save_step
from lsy_drone_racing.control.rl_song.config import ACTOR_OBS_DIM
from lsy_drone_racing.control.rl_song.obs import NORM_VAR_EPS, NormalizerState, update_normalizer

# Flat-concat obs layout (see ``RLSBXVecEnv._build_obs``):
#   [0 : ACTOR_OBS_DIM)            → masked actor obs
#   [ACTOR_OBS_DIM : 2*ACTOR_OBS_DIM) → privileged critic obs
ACTOR_SLICE = slice(0, ACTOR_OBS_DIM)
CRITIC_SLICE = slice(ACTOR_OBS_DIM, 2 * ACTOR_OBS_DIM)


def _find_underlying_env(env: object) -> object:
    """Drill through ``VecEnv`` wrappers to find the ``RLSBXVecEnv``.

    Parameters:
    ----------
    env : object
        The initial env (typically ``BaseCallback.training_env``), which may
        be wrapped in ``VecMonitor``, ``VecNormalize``, etc.

    Returns:
    -------
    object
        The first env in the wrapper chain exposing ``actor_normalizer``.

    Raises:
    ------
    RuntimeError
        If no env in the chain exposes ``actor_normalizer``.
    """
    while not hasattr(env, "actor_normalizer"):
        if hasattr(env, "venv"):
            env = env.venv
        elif hasattr(env, "env"):
            env = env.env
        else:
            raise RuntimeError("Could not find RLSBXVecEnv under training_env wrappers.")
    return env


class NormalizerUpdateCallback(BaseCallback):
    """Welford-update both normalizers after each rollout buffer fill.

    The rollout buffer at ``_on_rollout_end`` contains ``n_steps * n_envs``
    samples of shape ``(2*ACTOR_OBS_DIM,)`` that have already been normalized
    by the env wrapper. We split each sample into actor / critic halves,
    invert the affine ``(x - mean) / sqrt(var + eps)`` to recover raw values,
    and feed each half to its respective ``NormalizerState`` independently.

    Parameters:
    ----------
    verbose : int, optional
        SB3 verbosity, propagated to ``BaseCallback``.

    Notes:
    -----
    Inversion is exact (no information loss): the env wrapper normalizes with
    the stats valid at the start of the rollout, and those same stats live on
    the env object until this callback fires.
    """

    def __init__(self, verbose: int = 0) -> None:
        """Initialize the callback. See class docstring for parameter details."""
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Update both ``NormalizerState`` instances from the rollout buffer."""
        env = _find_underlying_env(self.training_env)
        rollout_buffer = self.model.rollout_buffer
        observations = rollout_buffer.observations
        if observations is None:
            raise RuntimeError("rollout_buffer.observations is None at _on_rollout_end.")
        # Flatten (n_steps, n_envs, obs_dim) → (n_steps * n_envs, obs_dim).
        flat_obs = np.asarray(observations).reshape(-1, observations.shape[-1])
        actor_normalized = flat_obs[:, ACTOR_SLICE]
        critic_normalized = flat_obs[:, CRITIC_SLICE]

        # v126: single-normalizer ablation. Both halves of the flat-concat obs
        # are normalized against the same Welford state (``actor_normalizer``
        # is the canonical state; ``critic_normalizer`` is an alias). Update
        # statistics from the actor batch only — this matches rl_song's
        # reference behaviour where ``rollout.scan_rollout`` runs the Welford
        # update from a single sample stream. The critic-side raw values
        # (privileged gate positions) cover a subset of the actor's masked
        # values (nominal default → revealed truth as sensor range fires), so
        # adding them as a second update would double-count the post-reveal
        # samples in the running mean/var. ``set_actor_normalizer`` rebinds
        # both attributes; the explicit ``set_critic_normalizer`` call is
        # unnecessary but kept for clarity at the assignment site.
        actor_raw = self._invert(actor_normalized, env.actor_normalizer)
        _ = critic_normalized  # privileged values intentionally not folded in
        new_norm = update_normalizer(env.actor_normalizer, jnp.asarray(actor_raw))
        env.set_actor_normalizer(new_norm)

    @staticmethod
    def _invert(normalized: np.ndarray, state: NormalizerState) -> np.ndarray:
        """Recover raw obs values from post-normalization obs.

        Parameters:
        ----------
        normalized : ndarray, shape (n_samples, obs_dim)
            Observations as stored in the rollout buffer.
        state : NormalizerState
            The normalizer state in effect during the rollout.

        Returns:
        -------
        raw : ndarray, shape (n_samples, obs_dim)
            Raw obs values, recovered via ``z * sqrt(var + eps) + mean``.

        Notes:
        -----
        The forward map is ``normalize(x) = (x - mean) / sqrt(var + eps)``;
        this is its exact inverse.
        """
        std = np.sqrt(np.asarray(state.var) + NORM_VAR_EPS)
        mean = np.asarray(state.mean)
        return normalized * std + mean


class EntropyAnnealCallback(BaseCallback):
    """Linearly anneal ``model.ent_coef`` from start to final over training.

    SBX's PPO accepts ``ent_coef`` as a float at construction and never
    schedules it. The v77 cold-train recipe used ent_coef 0.005 -> 0.001
    annealed linearly with timesteps -- a stronger commit-pressure
    schedule than v112's constant 0.005. Restore that on the SBX stack.

    Parameters:
    ----------
    ent_coef_start : float
        Initial entropy coefficient (also the model's construction-time
        value; this callback overwrites it at the first rollout end).
    ent_coef_final : float
        Final entropy coefficient at ``total_timesteps``.
    total_timesteps : int
        Schedule horizon; should match ``model.learn(total_timesteps=...)``.
    verbose : int, optional
        SB3 verbosity, propagated to ``BaseCallback``.

    Notes:
    -----
    The schedule is linear in env steps (not iterations), so the
    instantaneous value at step ``t`` is ``start + (final - start) *
    min(1, t / total)``. After ``total`` the value clamps at ``final``.
    """

    def __init__(
        self,
        ent_coef_start: float,
        ent_coef_final: float,
        total_timesteps: int,
        verbose: int = 0,
    ) -> None:
        """Initialize the callback. See class docstring for parameter details."""
        super().__init__(verbose)
        self.ent_coef_start = float(ent_coef_start)
        self.ent_coef_final = float(ent_coef_final)
        self.total_timesteps = int(total_timesteps)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Update ``model.ent_coef`` per the linear schedule."""
        t = int(self.model.num_timesteps)
        frac = min(1.0, t / max(1, self.total_timesteps))
        new_ent = self.ent_coef_start + (self.ent_coef_final - self.ent_coef_start) * frac
        # ``sbx.PPO`` reads ``self.ent_coef`` per ``_one_update`` call inside
        # the JIT, so updating the Python attribute is sufficient -- the next
        # iteration's update will close over the new value.
        self.model.ent_coef = float(new_ent)


class PeriodicCheckpointCallback(BaseCallback):
    """Write a ``save_step`` checkpoint every ``save_freq_steps`` env steps.

    Enables post-hoc checkpoint selection on cold-trains and warm-starts
    longer than ~100 M steps. The 2026-05-25 handoff documented that the
    final step of v56 / v77 over-trained; the working checkpoint was one
    earlier ``step_NNN``. The previous ``rl_sbx/train.py`` only saved the
    final step, blocking that selection — fixed by this callback.

    Parameters:
    ----------
    run_dir : Path
        Existing run directory; per-step subdirs created under it.
    alpha_max_rad : float
        Forwarded to :func:`save_step` so eval can reconstruct the policy
        with the same tangent-space rotation budget used at training.
    save_freq_steps : int
        Number of *env steps* between saves (not gradient updates). The
        callback fires on ``_on_rollout_end`` (once per PPO iteration), so
        the effective cadence is rounded up to the nearest iteration
        boundary. At ``n_envs=16384 × n_steps=256`` that's ~4.19 M per
        iteration, so a request of 10 M saves every 3rd iter.
    verbose : int, optional
        SB3 verbosity, propagated to ``BaseCallback``.

    Notes:
    -----
    The callback never deletes checkpoints — disk grows linearly with run
    length. A 155 M cold-train at ``save_freq_steps=20_000_000`` produces
    ~8 step-dirs; each is ~1 MB so this is negligible vs the wandb run.
    """

    def __init__(
        self,
        run_dir: Path,
        alpha_max_rad: float,
        save_freq_steps: int = 20_000_000,
        verbose: int = 0,
    ) -> None:
        """Initialize the callback. See class docstring for parameter details."""
        super().__init__(verbose)
        self.run_dir = run_dir
        self.alpha_max_rad = float(alpha_max_rad)
        self.save_freq_steps = int(save_freq_steps)
        self._last_save_step: int = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Save a checkpoint if the cumulative timestep crossed the cadence."""
        current = int(self.model.num_timesteps)
        if current - self._last_save_step < self.save_freq_steps:
            return
        env = _find_underlying_env(self.training_env)
        step_dir = save_step(
            run_dir=self.run_dir,
            global_step=current,
            actor_params=self.model.policy.actor_state.params,
            critic_params=self.model.policy.vf_state.params,
            actor_normalizer=env.actor_normalizer,
            critic_normalizer=env.critic_normalizer,
            tangent_alpha_max_rad=self.alpha_max_rad,
        )
        self._last_save_step = current
        if self.verbose:
            print(f"[checkpoint] step {current} -> {step_dir}", flush=True)
