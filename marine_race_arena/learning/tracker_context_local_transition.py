"""Minimal legal temporal state for the local transition policy."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

from marine_race_arena.controllers.local_course_tracker import (
    LocalCourseTracker,
    LocalCourseTrackerConfig,
)
from marine_race_arena.learning.config import ACTION_DIM
from marine_race_arena.learning.config_local_transition import (
    TARGET_CHANGED_RECENT_WINDOW_S,
    LocalTransitionLearningContext,
)
from marine_race_arena.learning.observation_encoder import (
    _depth_m,
    _finite,
    _select_beacon_packet,
)


# HoloOcean's close gate image is commonly cropped into disconnected bars, so
# its detected centroid is not stable enough to be a mandatory pre-passage
# condition.  This local-transition-only configuration permits a COMMIT from
# consecutive close camera + expected-beacon evidence, then retains the full
# independent DVL/range/disappearance/rear-bearing exit confirmation.

#: Aperture and beacon placement, from the official track files: every circuit
#: uses ``track.gate_inner_size_m = [1.5, 1.5]`` and mounts the beacon at
#: ``beacon.position_offset = [0, 0, 0.35]``, i.e. 0.35 m up the gate's own
#: up-axis rather than at the centre of the aperture.
GATE_APERTURE_HALF_SIZE_M = 0.75
BEACON_UP_OFFSET_M = 0.35

#: How far from the beacon a *legitimate* pass can be.  The far corner of the
#: aperture, diagonally opposite the beacon, is
#: ``hypot(0.75, 0.75 + 0.35) = 1.33 m`` away -- so any envelope below that
#: rejects passes that really did go through the gate.
#:
#: This is the correction for a measured defect, not a loosening.  The value
#: here used to be a hand-picked 0.60 m, described as rejecting a pass outside
#: the 1.5 m aperture.  It does not: with the beacon 0.35 m high, 0.60 m admits
#: only a pass within 0.49 m laterally or 0.25 m below the centre, and rejects
#: the rest of a perfectly valid transit.  The tracker has no recovery path, so
#: one rejected passage leaves it waiting for a gate the rover has already left
#: -- permanently.  Measured on Vertical Serpent, same sensor stream, same
#: episode: at 0.60 m the tracker advanced 5 times against 16 crossings and
#: agreed with actual progress on 32.9% of steps; at the aperture-derived
#: envelope it advanced 16 times.  Every observation recorded after the stall
#: pointed backwards while the expert's action drove forwards, which is not a
#: policy the learner can fit.
PASSAGE_ENVELOPE_M = math.hypot(
    GATE_APERTURE_HALF_SIZE_M, GATE_APERTURE_HALF_SIZE_M + BEACON_UP_OFFSET_M
)

LOCAL_TRANSITION_TRACKER_CONFIG = LocalCourseTrackerConfig(
    proximity_commit_enabled=True,
    proximity_commit_range_m=1.0,
    proximity_commit_bearing_deg=25.0,
    proximity_commit_confidence_threshold=0.35,
    proximity_commit_required_frames=2,
    min_commit_displacement_m=0.10,
    min_range_for_passage_m=PASSAGE_ENVELOPE_M,
    close_range_required_packets=2,
    range_rise_margin_m=0.30,
    rear_exit_range_rise_margin_m=0.15,
    rear_exit_range_rise_required_packets=2,
    rear_exit_min_rear_packets=2,
    rear_bearing_min_deg=90.0,
    rear_bearing_required_packets=2,
    visual_disappear_frames=2,
    exit_clearance_s=0.0,
)


def _wrapped_angle_delta_deg(current: float, previous: float) -> float:
    """Shortest signed current-minus-previous angular displacement."""

    return (float(current) - float(previous) + 180.0) % 360.0 - 180.0


class OnboardLocalTransitionContextTracker:
    """Track only the expected target and valid consecutive sensor derivatives."""

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
        self._last_local_time_s: Optional[float] = None
        self._previous_beacon: Optional[tuple[str, float, float, float]] = None
        self._previous_vision: Optional[tuple[float, float]] = None
        self._time_since_change_s: Optional[float] = None
        self._last_context = LocalTransitionLearningContext()

    @property
    def tracker(self) -> LocalCourseTracker:
        if self._tracker is None:
            raise RuntimeError("local transition tracker must be reset first")
        return self._tracker

    @property
    def last_context(self) -> LocalTransitionLearningContext:
        return self._last_context

    def reset(self, first_observation: Optional[Mapping[str, Any]] = None) -> None:
        self._tracker = LocalCourseTracker(
            initial_beacon_id=self.initial_beacon_id,
            total_beacons=self.total_beacons,
            laps=self.laps,
            config=LOCAL_TRANSITION_TRACKER_CONFIG,
        )
        observation = first_observation or {}
        sensors = observation.get("sensors") if isinstance(observation, Mapping) else {}
        self._depth_ref = _depth_m(sensors if isinstance(sensors, Mapping) else {})
        self._last_local_time_s = None
        self._previous_beacon = None
        self._previous_vision = None
        self._time_since_change_s = None
        self._last_context = LocalTransitionLearningContext(
            expected_beacon_id=self.initial_beacon_id,
            depth_reference_m=self._depth_ref,
        )

    def context(
        self,
        observation: Mapping[str, Any],
        *,
        dt: Optional[float],
        prev_action: Optional[Sequence[float]] = None,
    ) -> LocalTransitionLearningContext:
        obs = observation if isinstance(observation, Mapping) else {}
        sensors = obs.get("sensors") if isinstance(obs.get("sensors"), Mapping) else {}
        local_time_s = _finite(obs.get("local_time_s"), 0.0)
        if dt is None:
            effective_dt = (
                max(0.0, local_time_s - self._last_local_time_s)
                if self._last_local_time_s is not None else 0.0
            )
        else:
            effective_dt = max(0.0, _finite(dt, 0.0))
        self._last_local_time_s = local_time_s

        tracker = self.tracker
        expected_before = tracker.expected_beacon_id
        tracker.update(
            local_time_s=local_time_s,
            beacons=obs.get("beacons") or [],
            camera_image=sensors.get("FrontCamera"),
            dvl_velocity=sensors.get("DVLSensor"),
            dt=effective_dt,
        )
        expected_after = tracker.expected_beacon_id
        changed = expected_after != expected_before
        if changed:
            self._previous_beacon = None
            self._previous_vision = None
            self._time_since_change_s = 0.0
        elif self._time_since_change_s is not None:
            self._time_since_change_s += effective_dt

        bearing_rate = elevation_rate = range_rate = 0.0
        beacon_rate_valid = False
        packet = _select_beacon_packet(obs.get("beacons") or [], expected_after)
        if packet is not None:
            current = (
                str(expected_after),
                _finite(packet.get("bearing_deg"), 0.0),
                _finite(packet.get("elevation_deg"), 0.0),
                max(0.0, _finite(packet.get("range_m"), 0.0)),
            )
            if (
                not changed
                and effective_dt > 0.0
                and self._previous_beacon is not None
                and self._previous_beacon[0] == current[0]
            ):
                bearing_rate = _wrapped_angle_delta_deg(
                    current[1], self._previous_beacon[1]
                ) / effective_dt
                elevation_rate = (current[2] - self._previous_beacon[2]) / effective_dt
                range_rate = (current[3] - self._previous_beacon[3]) / effective_dt
                beacon_rate_valid = all(math.isfinite(value) for value in (
                    bearing_rate, elevation_rate, range_rate
                ))
            self._previous_beacon = current
        else:
            # A packet after a gap is a reacquisition, not a consecutive sample.
            self._previous_beacon = None

        x_rate = y_rate = 0.0
        vision_rate_valid = False
        target = getattr(tracker, "_latest_visual_target", None)
        if target is not None:
            current_vision = (
                _finite(getattr(target, "center_x", 0.0), 0.0),
                _finite(getattr(target, "center_y", 0.0), 0.0),
            )
            if not changed and effective_dt > 0.0 and self._previous_vision is not None:
                x_rate = (current_vision[0] - self._previous_vision[0]) / effective_dt
                y_rate = (current_vision[1] - self._previous_vision[1]) / effective_dt
                vision_rate_valid = math.isfinite(x_rate) and math.isfinite(y_rate)
            self._previous_vision = current_vision
        else:
            # Loss invalidates the next derivative across reacquisition.
            self._previous_vision = None

        action = list(prev_action) if prev_action is not None else [0.0] * ACTION_DIM
        since_change = self._time_since_change_s or 0.0
        context = LocalTransitionLearningContext(
            expected_beacon_id=expected_after,
            depth_reference_m=self._depth_ref,
            visual_target=target,
            use_tracked_visual_target=True,
            prev_action=action,
            vision_center_x_rate_s=x_rate,
            vision_center_y_rate_s=y_rate,
            vision_rate_valid=vision_rate_valid,
            beacon_bearing_rate_deg_s=bearing_rate,
            beacon_elevation_rate_deg_s=elevation_rate,
            beacon_range_rate_m_s=range_rate,
            beacon_rate_valid=beacon_rate_valid,
            target_changed_recently=(
                self._time_since_change_s is not None
                and since_change <= TARGET_CHANGED_RECENT_WINDOW_S
            ),
            time_since_target_change_s=since_change,
        )
        self._last_context = context
        return context
