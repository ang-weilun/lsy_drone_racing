import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BasePlanner(ABC):
    """Abstract base class for all trajectory planners in the mpcc framework."""

    def __init__(self, obs: dict[str, np.ndarray], freq: int, config: Any) -> None:
        self.config = config
        self.freq = freq
        
        self.gates_pos = obs.get("gates_pos", np.array([])).copy()
        self.gates_quat = obs.get("gates_quat", np.array([])).copy()
        self.obstacles_pos = obs.get("obstacles_pos", np.array([])).copy()
        
        self.target_gate_idx = int(obs.get("target_gate", 0))
        if self.target_gate_idx == -1:
            self.target_gate_idx = len(self.gates_pos)

        self._tick = 0
        self._last_replan_tick = -getattr(self.config, "REPLAN_DEBOUNCE_TICKS", 1000)
        
        self.replan_events: list[dict] = []
        self.last_replan_event: dict | None = None
        self._traj_history: list[np.ndarray] = []

    def episode_reset(self) -> None:
        """Reset the planner state for a new episode."""
        self.target_gate_idx = 0
        self._tick = 0
        self._last_replan_tick = -getattr(self.config, "REPLAN_DEBOUNCE_TICKS", 1000)
        self.replan_events = []
        self.last_replan_event = None
        self._traj_history = []

    def add_trajectory_point(self, pos: np.ndarray) -> None:
        """Add a position point to the flown trajectory history buffer."""
        self._traj_history.append(pos.copy())

    def get_trajectory_history(self) -> np.ndarray:
        """Retrieve the recorded flown trajectory history."""
        if len(self._traj_history) == 0:
            return np.empty((0, 3))
        return np.array(self._traj_history)

    @abstractmethod
    def update(self, obs: dict[str, np.ndarray]) -> bool:
        """Update the planner state and replan if necessary.
        
        Args:
            obs: The current observation dict.
            
        Returns:
            bool: True if a replan occurred, False otherwise.
        """
        pass

    @abstractmethod
    def des_pos_spline(self, u: np.ndarray | float) -> np.ndarray:
        """Evaluate the desired position spline.
        
        Args:
            u: Normalized arc-length parameter in [0, 1].
            
        Returns:
            The evaluated position(s).
        """
        pass
