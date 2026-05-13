"""Pluggable agent + env factories.

The training script never imports SB3 / sbx / crazyflow directly. It
asks this module for an agent and a vec env based on ``cfg.builder``.
Wire your real implementation in by writing two functions that match
the protocols below and registering them under a name.

Builders are picked by config:

    cfg = {"builder": "noop_smoke", ...}

The ``noop_smoke`` builder is intentionally trivial — it doesn't even
import crazyflow — so the resume cycle can be exercised on a laptop
with just torch installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np
import torch

log = logging.getLogger(__name__)


class Agent(Protocol):
    """Duck-typed protocol the training loop needs."""

    def act(self, obs: Any) -> Any: ...
    def update(self, batch: Any) -> dict[str, float]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class VecEnv(Protocol):
    """Duck-typed vec env. Need state_dict for norm running stats."""

    def reset(self) -> Any: ...
    def step(self, action: Any) -> tuple[Any, Any, Any, Any]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


# ---- registry -----------------------------------------------------------

_BUILDERS: dict[str, tuple[Callable[..., VecEnv], Callable[..., Agent]]] = {}


def register(name: str, env_fn: Callable[..., VecEnv], agent_fn: Callable[..., Agent]) -> None:
    _BUILDERS[name] = (env_fn, agent_fn)


def build_vec_env(cfg: dict[str, Any]) -> VecEnv:
    name = cfg["builder"]
    if name not in _BUILDERS:
        raise KeyError(f"Unknown builder {name!r}. Registered: {sorted(_BUILDERS)}")
    return _BUILDERS[name][0](cfg)


def build_agent(cfg: dict[str, Any], env: VecEnv) -> Agent:
    name = cfg["builder"]
    if name not in _BUILDERS:
        raise KeyError(f"Unknown builder {name!r}. Registered: {sorted(_BUILDERS)}")
    return _BUILDERS[name][1](cfg, env)


# ---- noop_smoke builder (laptop-friendly) -------------------------------
#
# Used by tests/integration/test_resume_smoke.py. The "env" is a counter
# and the "agent" is a counter — together they prove that global_step,
# RNG state, model params, replay buffer, and env norm state all survive
# a SIGTERM-and-restart cycle.


@dataclass
class _SmokeEnv:
    obs_dim: int = 4
    n_envs: int = 2
    running_mean: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    running_var: np.ndarray = field(default_factory=lambda: np.ones(4, dtype=np.float32))
    seen: int = 0

    def reset(self) -> np.ndarray:
        return np.random.randn(self.n_envs, self.obs_dim).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        obs = np.random.randn(self.n_envs, self.obs_dim).astype(np.float32)
        # Cheap running-stats update so we can verify the env state was
        # actually round-tripped.
        self.seen += self.n_envs
        delta = obs.mean(axis=0) - self.running_mean
        self.running_mean = self.running_mean + delta / max(self.seen, 1)
        self.running_var = self.running_var * 0.99 + obs.var(axis=0) * 0.01
        reward = np.zeros(self.n_envs, dtype=np.float32)
        done = np.zeros(self.n_envs, dtype=bool)
        return obs, reward, done, {}

    def state_dict(self) -> dict[str, Any]:
        return {
            "running_mean": self.running_mean.copy(),
            "running_var": self.running_var.copy(),
            "seen": self.seen,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.running_mean = np.asarray(state["running_mean"], dtype=np.float32)
        self.running_var = np.asarray(state["running_var"], dtype=np.float32)
        self.seen = int(state["seen"])


class _SmokeAgent:
    def __init__(self, obs_dim: int = 4, action_dim: int = 2) -> None:
        torch.manual_seed(0)
        self.model = torch.nn.Linear(obs_dim, action_dim)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        self.replay: list[float] = []
        self.updates = 0

    def act(self, obs: Any) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32)
            return self.model(t).numpy()

    def update(self, batch: Any) -> dict[str, float]:
        x = torch.randn(8, self.model.in_features)
        y = torch.randn(8, self.model.out_features)
        pred = self.model(x)
        loss = ((pred - y) ** 2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.replay.append(float(loss.detach()))
        self.updates += 1
        return {"loss": float(loss.detach()), "updates": self.updates}

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "replay": list(self.replay),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        self.replay = list(state["replay"])
        self.updates = int(state["updates"])


def _build_smoke_env(cfg: dict[str, Any]) -> _SmokeEnv:
    return _SmokeEnv(obs_dim=cfg.get("obs_dim", 4), n_envs=cfg.get("n_envs", 2))


def _build_smoke_agent(cfg: dict[str, Any], env: _SmokeEnv) -> _SmokeAgent:
    return _SmokeAgent(obs_dim=env.obs_dim, action_dim=cfg.get("action_dim", 2))


register("noop_smoke", _build_smoke_env, _build_smoke_agent)


# ---- placeholders for real builders -------------------------------------
#
# Replace the bodies with your SB3 / sbx wiring. The training loop never
# changes — only what you put inside these functions.


def _build_sb3_env(cfg: dict[str, Any]) -> VecEnv:
    raise NotImplementedError(
        "sb3 env builder not wired yet — wire your crazyflow vec env here. "
        "It must expose .reset/.step/.state_dict/.load_state_dict for the "
        "normalization running stats."
    )


def _build_sb3_agent(cfg: dict[str, Any], env: VecEnv) -> Agent:
    raise NotImplementedError(
        "sb3 agent builder not wired yet — wrap your SB3/sbx policy here. "
        "state_dict must include model + optimizer + replay buffer."
    )


register("sb3_ppo", _build_sb3_env, _build_sb3_agent)
register("sbx_sac", _build_sb3_env, _build_sb3_agent)
