"""Shared onboard-only context source for the learning observation encoder.

Drives a controller-side :class:`LocalCourseTracker` from the official
observation to produce a :class:`LearningContext` (expected beacon, tracker
phase, local progress, visual lock). The Gym env, the trajectory recorder and the
deployable RL controller all use this one component, so the observation encoding
is identical at training and inference time.

Only legal onboard information is used: received beacon packets, the FrontCamera
frame and DVL velocity, plus the controller's own previous action.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from marine_race_arena.controllers.local_course_tracker import LocalCourseTracker
from marine_race_arena.learning.config import ACTION_DIM, LearningContext
from marine_race_arena.learning.observation_encoder import _depth_m


class OnboardContextTracker:
    """Produce a legal :class:`LearningContext` for each official observation."""

    def __init__(self, total_beacons: int, laps: int = 1, initial_beacon_id: str = "B01") -> None:
        self.total_beacons = max(1, int(total_beacons))
        self.laps = max(1, int(laps))
        self.initial_beacon_id = initial_beacon_id
        self._tracker: Optional[LocalCourseTracker] = None
        self._depth_ref: Optional[float] = None

    def reset(self, first_observation: Optional[Mapping[str, Any]] = None) -> None:
        self._tracker = LocalCourseTracker(
            initial_beacon_id=self.initial_beacon_id,
            total_beacons=self.total_beacons,
            laps=self.laps,
        )
        sensors = (first_observation or {}).get("sensors") or {}
        self._depth_ref = _depth_m(sensors)

    @property
    def tracker(self) -> LocalCourseTracker:
        if self._tracker is None:
            raise RuntimeError("OnboardContextTracker.reset() must be called first.")
        return self._tracker

    def context(
        self,
        observation: Mapping[str, Any],
        *,
        dt: float,
        prev_action: Optional[Sequence[float]] = None,
    ) -> LearningContext:
        """Advance the tracker on this observation and return the encoding context."""
        tracker = self.tracker
        sensors = (observation or {}).get("sensors") or {}
        tracker.update(
            local_time_s=float((observation or {}).get("local_time_s", 0.0)),
            beacons=(observation or {}).get("beacons") or [],
            camera_image=sensors.get("FrontCamera"),
            dvl_velocity=sensors.get("DVLSensor"),
            dt=dt,
        )
        visual_lock = getattr(tracker, "_latest_visual_target", None) is not None
        action = list(prev_action) if prev_action is not None else [0.0] * ACTION_DIM
        return LearningContext(
            expected_beacon_id=tracker.expected_beacon_id,
            tracker_phase=tracker.phase,
            local_beacon_index=tracker.local_beacon_index,
            local_lap=tracker.local_lap,
            total_beacons=tracker.total_beacons,
            laps=tracker.laps,
            depth_reference_m=self._depth_ref,
            visual_lock=visual_lock,
            prev_action=action,
        )
