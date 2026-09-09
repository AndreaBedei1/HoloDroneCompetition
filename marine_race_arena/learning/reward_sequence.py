"""Reliability-first PPO reward for ordered multi-gate sequences.

Privileged referee state is used only for training reward and termination.  It
is never appended to the policy observation and this object has no control API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from marine_race_arena.learning.config import ACTION_DIM
from marine_race_arena.learning.reward_v3 import (
    MultiGateRewardConfig,
    MultiGateTrainingReward,
)


@dataclass
class SequenceRewardConfig(MultiGateRewardConfig):
    beacon_progress_scale: float = 1.5
    gate_crossing_bonus: float = 30.0
    next_beacon_acquisition_bonus: float = 3.0
    next_beacon_alignment_scale: float = 2.0
    next_gate_visual_acquisition_bonus: float = 4.0
    next_gate_stable_visual_bonus: float = 2.0
    post_gate_forward_scale: float = 0.6
    post_acquisition_forward_scale: float = 0.5
    previous_gate_return_penalty: float = 12.0
    missed_gate_penalty: float = 30.0
    collision_penalty: float = 20.0
    out_of_bounds_penalty: float = 25.0
    wrong_direction_penalty: float = 25.0
    timeout_penalty: float = 40.0
    action_change_penalty: float = 0.06
    post_gate_window_steps: int = 150
    completion_bonus_per_gate: float = 8.0
    jerk_change_penalty: float = 0.025
    efficiency_unlocked: bool = False

    def set_efficiency_unlocked(self, unlocked: bool) -> None:
        self.efficiency_unlocked = bool(unlocked)
        self.set_phase("efficiency" if unlocked else "reliability")


class SequenceTrainingReward(MultiGateTrainingReward):
    """Potential progress plus explicit next-target and anti-return shaping."""

    def __init__(self, config: Optional[SequenceRewardConfig] = None) -> None:
        super().__init__(config or SequenceRewardConfig(reward_phase="reliability"))
        self._previous_action_delta = np.zeros(ACTION_DIM, dtype=np.float32)

    def reset(self, env: Any) -> None:
        super().reset(env)
        self._previous_action_delta = np.zeros(ACTION_DIM, dtype=np.float32)

    def __call__(
        self,
        env: Any,
        step: Any,
        gate_delta: int,
        action: Any = None,
    ) -> Tuple[float, Dict[str, float]]:
        previous_action = self._state.previous_action.copy()
        reward, components = super().__call__(env, step, gate_delta, action)
        action_array = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        action_delta = action_array - previous_action
        jerk = float(np.linalg.norm(action_delta - self._previous_action_delta))
        components["jerk_change_penalty"] = -float(
            self.config.jerk_change_penalty * jerk
        )
        self._previous_action_delta = action_delta

        state = env.episode.context.referee.states[env.episode.participant_id]
        status = getattr(state.status, "value", str(state.status))
        expected = max(1, len(env.episode.context.config.track.gate_sequence))
        if (step.terminated or step.truncated) and status == "FINISHED":
            components["sequence_length_completion"] = float(
                min(100.0, self.config.completion_bonus_per_gate * expected)
            )
        else:
            components["sequence_length_completion"] = 0.0

        reward = float(np.clip(sum(components.values()), -150.0, 150.0))
        return reward, components
