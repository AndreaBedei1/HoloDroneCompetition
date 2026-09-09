"""Minimal onboard gate-progression and temporal context for learned control.

The wrapped :class:`LocalCourseTracker` may select the expected beacon and
confirm passage, but this class has no action-generation API.  All temporal
features come from official sensor packets, local history, and the previously
applied policy action.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Mapping, Optional, Sequence

from marine_race_arena.controllers.local_course_tracker import LocalCourseTracker
from marine_race_arena.learning.config import ACTION_DIM
from marine_race_arena.learning.config_v3 import (
    RECENT_GATE_HISTORY_STEPS,
    RECENT_LOSS_WINDOW_STEPS,
    RECENT_VISION_WINDOW_STEPS,
    MultiGateLearningContext,
)
from marine_race_arena.learning.observation_encoder import (
    _depth_m,
    _dvl_body_velocity,
    _finite,
    _select_beacon_packet,
)

_CENTER_X_THRESHOLD = 0.18
_CENTER_Y_THRESHOLD = 0.20
_LARGE_AREA_THRESHOLD = 0.08
_RECEDING_DELTA_M = 0.04
_REAR_SECTOR_DEG = 100.0


def _camera_present(image: Any) -> bool:
    if image is None:
        return False
    shape = getattr(image, "shape", None)
    if shape is not None:
        return len(shape) >= 2 and int(shape[0]) > 0 and int(shape[1]) > 0
    return (
        isinstance(image, list)
        and bool(image)
        and isinstance(image[0], list)
        and bool(image[0])
    )


class OnboardMultiGateContextTracker:
    """Produce v3 contexts while retaining only legal progression authority."""

    def __init__(
        self,
        total_beacons: int,
        laps: int = 1,
        initial_beacon_id: str = "B01",
    ) -> None:
        self.total_beacons = max(1, int(total_beacons))
        self.laps = max(1, int(laps))
        self.initial_beacon_id = str(initial_beacon_id)
        self._tracker: Optional[LocalCourseTracker] = None
        self._depth_ref: Optional[float] = None
        self._step = 0
        self._last_local_time_s: Optional[float] = None
        self._last_seen_step: Optional[int] = None
        self._last_seen = None
        self._visual_history: Deque[tuple[bool, bool]] = deque(maxlen=RECENT_GATE_HISTORY_STEPS)
        self._loss_started_step: Optional[int] = None
        self._loss_displacement_m = 0.0
        self._loss_dvl_present = False
        self._previous_range_m: Optional[float] = None
        self._recent_ranges: Deque[float] = deque(maxlen=30)
        self._steps_since_change = 0
        self._previous_beacon_id: Optional[str] = None
        self._previous_gate_rear = False
        self._previous_gate_bearing_present = False
        self._last_context: Optional[MultiGateLearningContext] = None

    def reset(self, first_observation: Optional[Mapping[str, Any]] = None) -> None:
        self._tracker = LocalCourseTracker(
            initial_beacon_id=self.initial_beacon_id,
            total_beacons=self.total_beacons,
            laps=self.laps,
        )
        sensors = (first_observation or {}).get("sensors") or {}
        self._depth_ref = _depth_m(sensors)
        self._step = 0
        self._last_local_time_s = None
        self._last_seen_step = None
        self._last_seen = None
        self._visual_history.clear()
        self._loss_started_step = None
        self._loss_displacement_m = 0.0
        self._loss_dvl_present = False
        self._previous_range_m = None
        self._recent_ranges.clear()
        self._steps_since_change = 0
        self._previous_beacon_id = None
        self._previous_gate_rear = False
        self._previous_gate_bearing_present = False
        self._last_context = None

    @property
    def tracker(self) -> LocalCourseTracker:
        if self._tracker is None:
            raise RuntimeError("OnboardMultiGateContextTracker.reset() must be called first.")
        return self._tracker

    @property
    def started(self) -> bool:
        return self._tracker is not None

    @property
    def last_context(self) -> Optional[MultiGateLearningContext]:
        return self._last_context

    def context(
        self,
        observation: Mapping[str, Any],
        *,
        dt: Optional[float],
        prev_action: Optional[Sequence[float]] = None,
    ) -> MultiGateLearningContext:
        tracker = self.tracker
        obs = observation if isinstance(observation, Mapping) else {}
        sensors = obs.get("sensors") if isinstance(obs.get("sensors"), Mapping) else {}
        camera_image = sensors.get("FrontCamera")
        camera_present = _camera_present(camera_image)
        expected_before = tracker.expected_beacon_id
        local_time_s = _finite(obs.get("local_time_s"), 0.0)
        if dt is None:
            effective_dt = (
                max(0.0, local_time_s - self._last_local_time_s)
                if self._last_local_time_s is not None
                else 0.0
            )
        else:
            effective_dt = max(0.0, _finite(dt, 0.0))
        self._last_local_time_s = local_time_s

        tracker.update(
            local_time_s=local_time_s,
            beacons=obs.get("beacons") or [],
            camera_image=camera_image,
            dvl_velocity=sensors.get("DVLSensor"),
            dt=dt,
        )
        expected_after = tracker.expected_beacon_id
        changed = expected_after != expected_before
        if changed:
            self._previous_beacon_id = expected_before
            self._steps_since_change = 0
            self._previous_range_m = None
            self._recent_ranges.clear()
            self._previous_gate_rear = True
            self._previous_gate_bearing_present = True
        else:
            self._steps_since_change += 1

        target = getattr(tracker, "_latest_visual_target", None)
        visible = target is not None
        if visible:
            centered = (
                abs(_finite(target.center_x)) <= _CENTER_X_THRESHOLD
                and abs(_finite(target.center_y)) <= _CENTER_Y_THRESHOLD
            )
            large = _finite(target.area_fraction) >= _LARGE_AREA_THRESHOLD
            self._last_seen_step = self._step
            self._last_seen = target
            self._visual_history.append((centered, large))
            self._loss_started_step = None
            self._loss_displacement_m = 0.0
            self._loss_dvl_present = False
        else:
            if camera_present and self._last_seen_step is not None and self._loss_started_step is None:
                self._loss_started_step = self._step
            self._visual_history.append((False, False))
            if self._loss_started_step is not None:
                dvl = _dvl_body_velocity(sensors)
                if dvl is not None:
                    self._loss_dvl_present = True
                    self._loss_displacement_m += max(0.0, dvl[0]) * effective_dt

        packet = _select_beacon_packet(obs.get("beacons") or [], expected_after)
        range_delta_m = 0.0
        range_delta_present = False
        if packet is not None:
            current_range = max(0.0, _finite(packet.get("range_m"), 0.0))
            if self._previous_range_m is not None:
                range_delta_m = current_range - self._previous_range_m
                range_delta_present = True
            self._previous_range_m = current_range
            self._recent_ranges.append(current_range)

        if self._previous_beacon_id:
            previous_packet = _select_beacon_packet(
                obs.get("beacons") or [], self._previous_beacon_id
            )
            if previous_packet is not None:
                self._previous_gate_bearing_present = True
                self._previous_gate_rear = (
                    abs(_finite(previous_packet.get("bearing_deg"), 0.0)) >= _REAR_SECTOR_DEG
                )

        steps_since_seen = (
            self._step - self._last_seen_step if self._last_seen_step is not None else 0
        )
        recently_seen = (
            self._last_seen_step is not None
            and steps_since_seen <= RECENT_VISION_WINDOW_STEPS
        )
        recently_lost = (
            not visible
            and self._loss_started_step is not None
            and self._step - self._loss_started_step <= RECENT_LOSS_WINDOW_STEPS
        )
        last = self._last_seen
        action = list(prev_action) if prev_action is not None else [0.0] * ACTION_DIM
        recent = list(self._visual_history)
        context = MultiGateLearningContext(
            expected_beacon_id=expected_after,
            tracker_phase=tracker.phase,
            local_beacon_index=tracker.local_beacon_index,
            local_lap=tracker.local_lap,
            total_beacons=tracker.total_beacons,
            laps=tracker.laps,
            depth_reference_m=self._depth_ref,
            visual_lock=visible,
            visual_target=target,
            use_tracked_visual_target=True,
            prev_action=action,
            camera_present=camera_present,
            vision_recently_seen=recently_seen,
            vision_recently_lost=recently_lost,
            steps_since_gate_seen=steps_since_seen,
            steps_since_gate_seen_present=self._last_seen_step is not None,
            last_seen_center_x=_finite(getattr(last, "center_x", 0.0)),
            last_seen_center_y=_finite(getattr(last, "center_y", 0.0)),
            last_seen_area_fraction=_finite(getattr(last, "area_fraction", 0.0)),
            last_seen_confidence=_finite(getattr(last, "confidence", 0.0)),
            last_seen_present=last is not None,
            gate_was_recently_centered=any(item[0] for item in recent),
            gate_was_recently_large=any(item[1] for item in recent),
            forward_displacement_since_visual_loss_m=self._loss_displacement_m,
            forward_displacement_since_visual_loss_present=self._loss_dvl_present,
            beacon_range_delta_m=range_delta_m,
            beacon_range_delta_present=range_delta_present,
            beacon_range_min_recent_m=min(self._recent_ranges) if self._recent_ranges else 0.0,
            beacon_range_min_recent_present=bool(self._recent_ranges),
            beacon_now_receding=range_delta_present and range_delta_m >= _RECEDING_DELTA_M,
            expected_beacon_changed=changed,
            # Zero before the first real change so a transferred policy sees a
            # neutral transition block throughout single-gate approach.
            steps_since_beacon_change=(
                self._steps_since_change
                if self._previous_beacon_id is not None
                else 0
            ),
            previous_gate_in_rear_sector=self._previous_gate_rear,
            previous_gate_bearing_present=self._previous_gate_bearing_present,
        )
        self._last_context = context
        self._step += 1
        return context
