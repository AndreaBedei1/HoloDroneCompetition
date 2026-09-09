"""Training-only DAgger expert for multi-gate transition demonstrations.

The learned v3 transfer policy controls the approach and first crossing so the
recorded transition state matches deployment. The deterministic expert is
stepped in shadow and supplies recorded actions only after the local tracker
advances. This controller is never valid as the final runtime controller.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from marine_race_arena.controllers.official_baselines import (
    RuleGateCenterThenCommitController,
)
from marine_race_arena.learning.rl_multigate_controller import (
    RLMultigateController,
)
from marine_race_arena.participants.controller_interface import BaseController


class MultigateDAggerExpertController(BaseController):
    """Collect transition labels on states induced by the learned policy."""

    debug_only = True
    uses_ground_truth = False
    training_only = True

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._learned = RLMultigateController(model_path=model_path)
        self._expert = RuleGateCenterThenCommitController()
        self.expert_action_steps = 0
        self.learned_action_steps = 0

    def reset(self, mission_info: Mapping[str, Any]) -> None:
        self._learned.reset(mission_info)
        self._expert.reset(mission_info)
        self.expert_action_steps = 0
        self.learned_action_steps = 0

    def step(self, observation: Mapping[str, Any]) -> dict:
        learned_command = self._learned.step(observation)
        expert_command = self._expert.step(observation)
        tracker = self._learned.tracker
        if tracker is not None and tracker.local_completed >= 1:
            self.expert_action_steps += 1
            return expert_command
        self.learned_action_steps += 1
        return learned_command

    def close(self) -> None:
        self._learned.close()
        self._expert.close()
