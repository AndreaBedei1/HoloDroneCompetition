"""Persistent legal course-progress context for PPO sequence policies."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping, Optional, Sequence

from marine_race_arena.learning.config_sequence import SequenceLearningContext
from marine_race_arena.learning.config_v3 import MultiGateLearningContext
from marine_race_arena.learning.tracker_context_v3 import OnboardMultiGateContextTracker
from marine_race_arena.controllers.local_course_tracker import (
    LocalCourseTracker,
    LocalCourseTrackerConfig,
)


# The sequence policy must learn the turn toward gate k+1 before it can curl
# back toward gate k. These values still require camera + beacon + DVL evidence,
# but confirm the exit soon after the acoustic range turnaround rather than
# waiting for 1.2 m of post-gate travel. No pose, referee, or collision signal
# participates in this decision.
SEQUENCE_TRACKER_CONFIG = LocalCourseTrackerConfig(
    min_commit_displacement_m=1.00,
    close_range_required_packets=3,
    range_rise_margin_m=0.45,
    rear_exit_range_rise_margin_m=0.25,
    rear_exit_range_rise_required_packets=2,
    rear_exit_min_rear_packets=3,
    rear_bearing_min_deg=95.0,
    rear_bearing_required_packets=1,
    visual_disappear_frames=3,
    exit_clearance_s=1.0,
)


class OnboardSequenceContextTracker(OnboardMultiGateContextTracker):
    """Expose local progression state without any action-generation API."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._target_acquired = False

    def reset(self, first_observation=None) -> None:
        super().reset(first_observation)
        self._tracker = LocalCourseTracker(
            initial_beacon_id=self.initial_beacon_id,
            total_beacons=self.total_beacons,
            laps=self.laps,
            config=SEQUENCE_TRACKER_CONFIG,
        )
        self._target_acquired = False

    def context(
        self,
        observation: Mapping[str, Any],
        *,
        dt: Optional[float],
        prev_action: Optional[Sequence[float]] = None,
    ) -> SequenceLearningContext:
        base = super().context(observation, dt=dt, prev_action=prev_action)
        if base.expected_beacon_changed:
            self._target_acquired = False
        elif self._previous_beacon_id is not None and (
            base.beacon_range_delta_present or base.visual_lock
        ):
            self._target_acquired = True
        total = max(1, self.total_beacons * self.laps)
        current = min(total - 1, max(0, int(self.tracker.local_completed)))
        copied = {
            field.name: getattr(base, field.name)
            for field in fields(MultiGateLearningContext)
        }
        context = SequenceLearningContext(
            **copied,
            current_gate_index=current,
            remaining_gate_count=max(0, total - current),
            sequence_progress=current / total,
            previous_gate_cleared=self._previous_beacon_id is not None,
            new_target_acquired_since_crossing=self._target_acquired,
            time_since_confirmed_crossing_steps=(
                self._steps_since_change if self._previous_beacon_id is not None else 0
            ),
        )
        self._last_context = context
        return context
