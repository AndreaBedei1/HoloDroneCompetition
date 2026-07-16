"""Controller-side course progression from onboard information only.

The :class:`LocalCourseTracker` is the participant's own estimate of where it
is in the mission. It consumes exactly what a controller legally observes —
the received beacon packet list, the FrontCamera image, and DVL velocity — and
maintains the expected beacon, the locally completed count, the local lap and
a progression phase. The referee never confirms anything to it; if the tracker
is wrong, the referee simply scores the failure.

State machine::

    SEARCH -> APPROACH -> VISUAL_ALIGN -> COMMIT -> VERIFY_EXIT -> ADVANCE
       ^         ^             ^            |            |           |
       +---------+-------------+---- regressions --------+           v
                                                        next beacon / FINISHED

Gate passage is advanced only on robust multi-sensor evidence, all onboard:

1. the expected gate was detected by the FrontCamera and stayed large and
   centered for ``commit_required_frames`` consecutive frames (entry into
   COMMIT), or a partially cropped gate remained consistent with a very close,
   forward expected beacon for ``close_commit_required_frames``;
2. forward displacement after COMMIT, integrated from DVL surge velocity,
   reached ``min_commit_displacement_m``;
3. the expected beacon's (jump-filtered) range reached a local minimum below
   ``min_range_for_passage_m`` and then either rose by the normal
   ``range_rise_margin_m`` or produced the smaller, persistent rear-sector
   turnaround used when the rover is physically constrained by a gate frame;
4. the previously large-and-centered gate disappeared from the camera for
   ``visual_disappear_frames`` consecutive frames with the camera running;
5. two fresh packets place the same expected beacon at least
   ``rear_bearing_min_deg`` into the rear sector.  This final onboard cue
   proves the rover continued past the beacon instead of stopping just before
   the gate after a noisy range turnaround.

Single missing camera frames, single beacon dropouts, out-of-range silence,
one-packet range discontinuities and pre-lock visual disappearance never
advance the tracker; timeouts regress the phase and keep the same expected
beacon. Thresholds are configurable through :class:`LocalCourseTrackerConfig`
and covered by unit tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from marine_race_arena.controllers.vision import (
    VisionTarget,
    select_visual_target_for_beacon,
    vision_targets_from_camera,
)

PHASE_SEARCH = "SEARCH"
PHASE_APPROACH = "APPROACH"
PHASE_VISUAL_ALIGN = "VISUAL_ALIGN"
PHASE_COMMIT = "COMMIT"
PHASE_VERIFY_EXIT = "VERIFY_EXIT"
PHASE_ADVANCE = "ADVANCE"
PHASE_FINISHED = "FINISHED"

STATUS_RUNNING = "RUNNING"
STATUS_FINISHED = "FINISHED"


@dataclass(frozen=True)
class LocalCourseTrackerConfig:
    """Tunable thresholds; values are documented onboard quantities only."""

    #: Steer on the last received expected-beacon packet for at most this long.
    beacon_hold_s: float = 1.6
    #: Reject a single received range that jumps this far from the filtered
    #: value; accept it only when the following packet confirms it.
    range_jump_reject_m: float = 3.0
    #: Filtered expected-beacon range must be at most this to allow COMMIT.
    commit_range_m: float = 3.2
    #: Expected-beacon bearing magnitude must be at most this to allow COMMIT.
    commit_bearing_deg: float = 8.0
    #: Image-center tolerances and detection quality required for a centered frame.
    commit_center_x_threshold: float = 0.13
    commit_center_y_threshold: float = 0.32
    commit_confidence_threshold: float = 0.42
    commit_area_threshold: float = 0.030
    #: Consecutive centered frames required to enter COMMIT.
    commit_required_frames: int = 4
    #: At very short range the gate often extends below the camera frame and
    #: its visual centroid becomes biased.  A separate, still conservative
    #: close-range lock prevents the controller from steering away after the
    #: aperture is already directly ahead.  It remains camera + beacon only.
    close_commit_range_m: float = 1.6
    close_commit_bearing_deg: float = 8.0
    close_commit_center_x_threshold: float = 0.30
    close_commit_center_y_threshold: float = 0.50
    close_commit_confidence_threshold: float = 0.45
    close_commit_area_threshold: float = 0.08
    #: The close-range rescue is valid only during the first approach to the
    #: expected beacon.  A gate reacquired after the rover already passed it
    #: has a much larger range rise and must not trigger a return crossing.
    close_commit_range_rise_max_m: float = 0.35
    close_commit_required_frames: int = 2
    #: DVL-integrated forward displacement after COMMIT required for passage.
    min_commit_displacement_m: float = 1.2
    #: Give up a COMMIT that produced no displacement after this long.
    commit_timeout_s: float = 14.0
    #: The filtered range minimum must be below this for passage evidence
    #: (a pass through the aperture goes right under the gate beacon; a miss
    #: beside the frame keeps a larger closest range).
    min_range_for_passage_m: float = 1.30
    #: Require several received packets inside the close-passage envelope so a
    #: single noisy low range or a collision rebound cannot authorize advance.
    close_range_required_packets: int = 4
    #: A close-range packet only counts while the expected beacon is still in
    #: the forward sector.  After a true passage it may swing behind, but a
    #: side-on closest approach must not establish passage evidence.
    close_range_max_bearing_deg: float = 50.0
    #: The filtered range must rise this far above its minimum for passage.
    #: Set several noise standard deviations above the configured beacon range
    #: noise so a rise can not be produced by measurement noise alone.
    range_rise_margin_m: float = 0.9
    #: A rover that has crossed but remains in contact with the gate frame may
    #: be unable to create the full 0.9 m rise.  Permit a smaller turnaround
    #: only after a sustained rear-sector transition, and only when the margin
    #: persists in fresh, consecutive packets that all still place the beacon
    #: behind the rover.  This is an onboard acoustic recovery, not a collision
    #: sensor or referee signal.
    rear_exit_range_rise_margin_m: float = 0.35
    rear_exit_range_rise_required_packets: int = 3
    rear_exit_min_rear_packets: int = 6
    #: Passage is confirmed only after the expected beacon has persistently
    #: moved into the rear sector.  A 100-degree threshold remains conservative
    #: under moderate heading error while rejecting a beacon still beside or
    #: ahead of the rover.
    rear_bearing_min_deg: float = 100.0
    rear_bearing_required_packets: int = 2
    #: Consecutive camera-on frames without a large centered gate for passage.
    visual_disappear_frames: int = 6
    #: A detection at least this large (area fraction) still counts as "gate visible".
    disappear_area_fraction: float = 0.018
    #: Rule-book gate aperture width (public competition constant) and camera
    #: FOV, used to estimate how large the expected gate would appear at the
    #: current beacon range. A detection much smaller than that (for example
    #: the next gate far ahead) does not block disappearance evidence.
    nominal_gate_width_m: float = 1.5
    camera_fov_deg: float = 90.0
    #: Fraction of the expected apparent area below which a detection is
    #: considered "not the expected gate" for disappearance purposes.
    disappear_expected_area_factor: float = 0.35
    #: Abort a COMMIT when the gate slides out of the image sideways: the
    #: smoothed detection offset sits beyond ``commit_abort_center_x`` for
    #: ``commit_abort_frames`` consecutive frames while the vehicle is still
    #: ``commit_abort_min_range_m`` or more away. Below that range the frame
    #: is only partially visible and the detection centroid is biased, so
    #: nothing is recorded there.
    commit_abort_center_x: float = 0.60
    commit_abort_frames: int = 4
    commit_abort_min_range_m: float = 2.0
    #: Smoothing factor for the recorded commit-phase detection offset.
    commit_center_x_ema_alpha: float = 0.35
    #: Give up VERIFY_EXIT without full evidence after this long (no advance).
    verify_timeout_s: float = 16.0
    #: Exit-clearance hold after an advancement before hunting the next beacon.
    exit_clearance_s: float = 2.5
    #: Lose the visual-align phase after this long without any detection.
    visual_align_loss_s: float = 1.2


@dataclass
class BeaconEstimate:
    """The controller's held view of its expected beacon."""

    beacon_id: str
    bearing_deg: float
    elevation_deg: float
    range_m: float
    signal_strength: float
    received_at_s: float
    age_s: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "beacon_id": self.beacon_id,
            "bearing_deg": self.bearing_deg,
            "elevation_deg": self.elevation_deg,
            "range_m": self.range_m,
            "signal_strength": self.signal_strength,
            "received_at_s": self.received_at_s,
            "age_s": self.age_s,
        }


@dataclass
class TrackerStep:
    """Everything a controller needs from one tracker update."""

    phase: str
    status: str
    expected_beacon_id: str
    local_beacon_index: int
    local_completed: int
    local_lap: int
    beacon: Optional[BeaconEstimate]
    visual_target: Optional[VisionTarget]
    visual_targets: List[VisionTarget] = field(default_factory=list)
    just_advanced: bool = False
    finished: bool = False


class LocalCourseTracker:
    """Local mission progression estimated purely from onboard sensing."""

    def __init__(
        self,
        *,
        initial_beacon_id: str,
        total_beacons: int,
        laps: int = 1,
        config: Optional[LocalCourseTrackerConfig] = None,
    ) -> None:
        if total_beacons <= 0:
            raise ValueError("total_beacons must be positive.")
        self.config = config or LocalCourseTrackerConfig()
        self.total_beacons = int(total_beacons)
        self.laps = max(1, int(laps))
        initial_index = _beacon_index_from_id(initial_beacon_id, self.total_beacons)
        self.local_beacon_index = initial_index
        self.local_lap = 1
        self.local_completed = 0
        self.status = STATUS_RUNNING
        self.phase = PHASE_SEARCH

        self._last_time_s: Optional[float] = None
        self._held: Optional[BeaconEstimate] = None
        self._pending_range: Optional[float] = None
        self._filtered_range: Optional[float] = None
        self._last_evidence_packet_received_at_s: Optional[float] = None
        self._min_range: Optional[float] = None
        # Monotonic for the entire expected-beacon attempt.  Unlike the exit
        # evidence minimum, this is deliberately not reset by a failed COMMIT:
        # it keeps the close-range rescue from firing on a later recrossing.
        self._rescue_min_range: Optional[float] = None
        self._close_range_streak = 0
        self._close_range_confirmed = False
        self._rear_bearing_streak = 0
        self._rear_bearing_confirmed = False
        self._rear_exit_range_rise_streak = 0
        self._rear_exit_range_rise_confirmed = False
        self._centered_streak = 0
        self._close_commit_streak = 0
        self._disappear_streak = 0
        self._commit_started_at: Optional[float] = None
        self._verify_started_at: Optional[float] = None
        self._commit_displacement_m = 0.0
        self._advance_started_at: Optional[float] = None
        self._last_visual_at: Optional[float] = None
        self._offcenter_streak = 0
        self._last_commit_center_x: Optional[float] = None
        self._commit_entry_reason: Optional[str] = None
        self._last_commit_entry_evidence: Optional[Dict[str, Any]] = None
        self._advancements = 0
        self._latest_visual_target: Optional[VisionTarget] = None
        self._last_advancement_evidence: Optional[Dict[str, Any]] = None
        # Bounded, controller-local evidence histories.  They contain only
        # received packet measurements and camera-derived detections, never
        # configured geometry or referee state.
        self._recent_beacon_history: List[Dict[str, Any]] = []
        self._recent_visual_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ API

    @property
    def expected_beacon_id(self) -> str:
        return _beacon_id_from_index(self.local_beacon_index)

    @property
    def finished(self) -> bool:
        return self.status == STATUS_FINISHED

    def update(
        self,
        *,
        local_time_s: float,
        beacons: Any,
        camera_image: Any = None,
        dvl_velocity: Optional[Any] = None,
        dt: Optional[float] = None,
    ) -> TrackerStep:
        """Advance the local estimate by one control step.

        ``beacons`` is the received packet list from the official observation;
        ``camera_image`` the FrontCamera frame (or None); ``dvl_velocity`` the
        DVL body-frame velocity vector (or None).
        """
        time_s = float(local_time_s)
        step_dt = self._step_dt(time_s, dt)
        self._last_time_s = time_s

        if self.finished:
            return self._result(None, [], just_advanced=False)

        self._ingest_beacons(beacons, time_s)
        held = self._current_estimate(time_s)

        camera_present = _camera_frame_present(camera_image)
        visual_targets = vision_targets_from_camera(camera_image) if camera_present else []
        visual_target = select_visual_target_for_beacon(
            visual_targets,
            held.bearing_deg if held is not None else None,
            held.range_m if held is not None else None,
        )
        self._latest_visual_target = visual_target
        if visual_target is not None:
            self._last_visual_at = time_s
        self._recent_visual_history.append(
            {
                "local_time_s": time_s,
                "camera_present": camera_present,
                "detected": visual_target is not None,
                "center_x": visual_target.center_x if visual_target is not None else None,
                "center_y": visual_target.center_y if visual_target is not None else None,
                "confidence": visual_target.confidence if visual_target is not None else None,
                "area_fraction": visual_target.area_fraction if visual_target is not None else None,
            }
        )
        del self._recent_visual_history[:-20]

        forward_velocity = _forward_velocity(dvl_velocity)
        if self.phase in (PHASE_COMMIT, PHASE_VERIFY_EXIT) and forward_velocity is not None:
            self._commit_displacement_m += forward_velocity * step_dt

        just_advanced = self._advance_phase_machine(
            time_s=time_s,
            held=held,
            camera_present=camera_present,
            visual_target=visual_target,
        )
        return self._result(held, visual_targets, visual_target=visual_target, just_advanced=just_advanced)

    def diagnostics(self) -> Dict[str, Any]:
        """Debug/log-only snapshot; never fed back into the observation."""
        range_rise_m = None
        if self._filtered_range is not None and self._min_range is not None:
            range_rise_m = max(0.0, self._filtered_range - self._min_range)
        rescue_range_rise_m = None
        if self._filtered_range is not None and self._rescue_min_range is not None:
            rescue_range_rise_m = max(
                0.0, self._filtered_range - self._rescue_min_range
            )
        visual = self._latest_visual_target
        return {
            "phase": self.phase,
            "status": self.status,
            "expected_beacon_id": self.expected_beacon_id,
            "local_beacon_index": self.local_beacon_index,
            "local_completed": self.local_completed,
            "local_lap": self.local_lap,
            "filtered_range_m": self._filtered_range,
            "min_range_m": self._min_range,
            "range_rise_m": range_rise_m,
            "rescue_min_range_m": self._rescue_min_range,
            "rescue_range_rise_m": rescue_range_rise_m,
            "close_range_streak": self._close_range_streak,
            "close_range_confirmed": self._close_range_confirmed,
            "rear_bearing_streak": self._rear_bearing_streak,
            "rear_bearing_confirmed": self._rear_bearing_confirmed,
            "rear_exit_range_rise_streak": self._rear_exit_range_rise_streak,
            "rear_exit_range_rise_confirmed": self._rear_exit_range_rise_confirmed,
            "last_evidence_packet_received_at_s": (
                self._last_evidence_packet_received_at_s
            ),
            "commit_displacement_m": round(self._commit_displacement_m, 3),
            "commit_active": self.phase in (PHASE_COMMIT, PHASE_VERIFY_EXIT),
            "centered_streak": self._centered_streak,
            "close_commit_streak": self._close_commit_streak,
            "disappear_streak": self._disappear_streak,
            "advancements": self._advancements,
            "recent_beacon_samples": len(self._recent_beacon_history),
            "recent_visual_samples": len(self._recent_visual_history),
            "visual_detected": visual is not None,
            "visual_center_x": visual.center_x if visual is not None else None,
            "visual_center_y": visual.center_y if visual is not None else None,
            "visual_confidence": visual.confidence if visual is not None else None,
            "visual_area_fraction": visual.area_fraction if visual is not None else None,
            "last_commit_center_x": self._last_commit_center_x,
            "commit_entry_reason": self._commit_entry_reason,
            "last_commit_entry_evidence": self._last_commit_entry_evidence,
            "last_advancement_evidence": self._last_advancement_evidence,
        }

    # -------------------------------------------------------------- internals

    def _step_dt(self, time_s: float, dt: Optional[float]) -> float:
        if dt is not None and dt > 0.0:
            return float(dt)
        if self._last_time_s is None:
            return 0.0
        return max(0.0, time_s - self._last_time_s)

    def _ingest_beacons(self, beacons: Any, time_s: float) -> None:
        packet = _latest_packet_for(beacons, self.expected_beacon_id)
        if packet is None:
            return
        raw_range = _safe_float(packet.get("range_m"), None)
        if raw_range is None:
            return
        received_at_s = _safe_float(packet.get("received_at_s"), time_s) or time_s
        if (
            self._last_evidence_packet_received_at_s is not None
            and received_at_s <= self._last_evidence_packet_received_at_s
        ):
            # Held or replayed packets remain stale measurements; they cannot
            # build any persistence counter or confirm a rejected range jump.
            return
        self._last_evidence_packet_received_at_s = received_at_s
        filtered, range_sample_accepted = self._filter_range(raw_range)
        self._held = BeaconEstimate(
            beacon_id=self.expected_beacon_id,
            bearing_deg=_safe_float(packet.get("bearing_deg"), 0.0) or 0.0,
            elevation_deg=_safe_float(packet.get("elevation_deg"), 0.0) or 0.0,
            range_m=filtered,
            signal_strength=_safe_float(packet.get("signal_strength"), 0.0) or 0.0,
            received_at_s=received_at_s,
            age_s=0.0,
        )
        self._recent_beacon_history.append(
            {
                "beacon_id": self.expected_beacon_id,
                "bearing_deg": self._held.bearing_deg,
                "elevation_deg": self._held.elevation_deg,
                "range_m": self._held.range_m,
                "signal_strength": self._held.signal_strength,
                "received_at_s": self._held.received_at_s,
            }
        )
        del self._recent_beacon_history[:-20]
        if not range_sample_accepted:
            # The bearing may still be useful for navigation through _held,
            # but a packet whose range was rejected cannot confirm passage.
            return
        if self._rescue_min_range is None or filtered < self._rescue_min_range:
            self._rescue_min_range = filtered
        if self.phase in (PHASE_APPROACH, PHASE_VISUAL_ALIGN, PHASE_COMMIT, PHASE_VERIFY_EXIT):
            if self._min_range is None or filtered < self._min_range:
                self._min_range = filtered
        if self.phase in (PHASE_COMMIT, PHASE_VERIFY_EXIT):
            if (
                filtered <= self.config.min_range_for_passage_m
                and abs(self._held.bearing_deg) <= self.config.close_range_max_bearing_deg
            ):
                self._close_range_streak += 1
                if self._close_range_streak >= self.config.close_range_required_packets:
                    self._close_range_confirmed = True
            elif not self._close_range_confirmed:
                self._close_range_streak = 0
            if self._close_range_confirmed:
                if abs(self._held.bearing_deg) >= self.config.rear_bearing_min_deg:
                    self._rear_bearing_streak += 1
                    if (
                        self._rear_bearing_streak
                        >= self.config.rear_bearing_required_packets
                    ):
                        self._rear_bearing_confirmed = True
                elif not self._rear_bearing_confirmed:
                    self._rear_bearing_streak = 0

            rear_exit_turnaround = (
                self._min_range is not None
                and filtered - self._min_range
                >= self.config.rear_exit_range_rise_margin_m
                and abs(self._held.bearing_deg) >= self.config.rear_bearing_min_deg
                and self._rear_bearing_streak
                >= self.config.rear_exit_min_rear_packets
            )
            if rear_exit_turnaround:
                self._rear_exit_range_rise_streak += 1
                if (
                    self._rear_exit_range_rise_streak
                    >= self.config.rear_exit_range_rise_required_packets
                ):
                    self._rear_exit_range_rise_confirmed = True
            elif not self._rear_exit_range_rise_confirmed:
                self._rear_exit_range_rise_streak = 0

    def _filter_range(self, raw_range: float) -> tuple[float, bool]:
        """Reject one-packet range discontinuities; accept confirmed jumps."""
        if self._filtered_range is None:
            self._filtered_range = raw_range
            self._pending_range = None
            return raw_range, True
        if abs(raw_range - self._filtered_range) <= self.config.range_jump_reject_m:
            self._filtered_range = raw_range
            self._pending_range = None
            return raw_range, True
        if (
            self._pending_range is not None
            and abs(raw_range - self._pending_range) <= self.config.range_jump_reject_m
        ):
            # Two consecutive packets agree on the new level: accept it.
            self._filtered_range = raw_range
            self._pending_range = None
            return raw_range, True
        self._pending_range = raw_range
        return self._filtered_range, False

    def _current_estimate(self, time_s: float) -> Optional[BeaconEstimate]:
        if self._held is None:
            return None
        age = max(0.0, time_s - self._held.received_at_s)
        if age > self.config.beacon_hold_s:
            return None
        held = BeaconEstimate(**{**self._held.__dict__, "age_s": age})
        return held

    def _advance_phase_machine(
        self,
        *,
        time_s: float,
        held: Optional[BeaconEstimate],
        camera_present: bool,
        visual_target: Optional[VisionTarget],
    ) -> bool:
        config = self.config

        if self.phase == PHASE_ADVANCE:
            if (
                self._advance_started_at is not None
                and time_s - self._advance_started_at >= config.exit_clearance_s
            ):
                self._advance_started_at = None
                self.phase = PHASE_SEARCH
            return False

        if self.phase in (PHASE_SEARCH, PHASE_APPROACH):
            if held is None:
                self.phase = PHASE_SEARCH
            else:
                self.phase = PHASE_APPROACH
                if visual_target is not None and visual_target.confidence >= 0.35:
                    self.phase = PHASE_VISUAL_ALIGN
            return False

        if self.phase == PHASE_VISUAL_ALIGN:
            if visual_target is None:
                # COMMIT locks are explicitly consecutive visual frames.  A
                # missing or rejected detection breaks both streaks even when
                # VISUAL_ALIGN itself is held briefly for reacquisition.
                self._centered_streak = 0
                self._close_commit_streak = 0
                lost_for = (
                    time_s - self._last_visual_at
                    if self._last_visual_at is not None
                    else config.visual_align_loss_s + 1.0
                )
                if lost_for > config.visual_align_loss_s:
                    self.phase = PHASE_APPROACH if held is not None else PHASE_SEARCH
                return False
            centered = (
                abs(visual_target.center_x) <= config.commit_center_x_threshold
                and abs(visual_target.center_y) <= config.commit_center_y_threshold
                and visual_target.confidence >= config.commit_confidence_threshold
                and (
                    visual_target.area_fraction >= config.commit_area_threshold
                    or visual_target.confidence >= 0.60
                )
                and held is not None
                and held.range_m <= config.commit_range_m
                and abs(held.bearing_deg) <= config.commit_bearing_deg
            )
            close_centered = (
                abs(visual_target.center_x) <= config.close_commit_center_x_threshold
                and abs(visual_target.center_y) <= config.close_commit_center_y_threshold
                and visual_target.confidence >= config.close_commit_confidence_threshold
                and visual_target.area_fraction >= config.close_commit_area_threshold
                and held is not None
                and held.range_m <= config.close_commit_range_m
                and abs(held.bearing_deg) <= config.close_commit_bearing_deg
                and self._rescue_min_range is not None
                and held.range_m - self._rescue_min_range
                <= config.close_commit_range_rise_max_m
            )
            self._centered_streak = self._centered_streak + 1 if centered else 0
            self._close_commit_streak = (
                self._close_commit_streak + 1 if close_centered else 0
            )
            normal_commit = self._centered_streak >= config.commit_required_frames
            close_commit = (
                self._close_commit_streak >= config.close_commit_required_frames
            )
            if normal_commit or close_commit:
                self._enter_commit(
                    time_s=time_s,
                    reason="standard_visual_lock" if normal_commit else "close_range_rescue",
                    held=held,
                    visual_target=visual_target,
                )
            return False

        if self.phase == PHASE_COMMIT:
            self._update_disappear_streak(camera_present, visual_target)
            if self._commit_slid_out_sideways(held, camera_present, visual_target):
                # The gate is sliding out of the image while still meters
                # away: the push is heading beside the frame, not through it.
                self._regress_after_failed_commit(held, visual_target)
                return False
            if self._commit_displacement_m >= config.min_commit_displacement_m:
                self.phase = PHASE_VERIFY_EXIT
                self._verify_started_at = time_s
                return self._try_confirm_exit(time_s)
            if (
                self._commit_started_at is not None
                and time_s - self._commit_started_at > config.commit_timeout_s
            ):
                # Commit produced no displacement: blocked or misdetected.
                self._regress_after_failed_commit(held, visual_target)
            return False

        if self.phase == PHASE_VERIFY_EXIT:
            self._update_disappear_streak(camera_present, visual_target)
            if self._commit_slid_out_sideways(held, camera_present, visual_target):
                self._regress_after_failed_commit(held, visual_target)
                return False
            confirmed = self._try_confirm_exit(time_s)
            if confirmed:
                return True
            if (
                self._verify_started_at is not None
                and time_s - self._verify_started_at > config.verify_timeout_s
            ):
                self._regress_after_failed_commit(held, visual_target)
            return False

        return False

    def _enter_commit(
        self,
        *,
        time_s: float,
        reason: str,
        held: Optional[BeaconEstimate],
        visual_target: Optional[VisionTarget],
    ) -> None:
        """Enter COMMIT and retain the strictly-onboard evidence snapshot."""
        rescue_range_rise_m = None
        if held is not None and self._rescue_min_range is not None:
            rescue_range_rise_m = max(0.0, held.range_m - self._rescue_min_range)
        self.phase = PHASE_COMMIT
        self._commit_started_at = time_s
        self._commit_displacement_m = 0.0
        self._disappear_streak = 0
        self._offcenter_streak = 0
        self._commit_entry_reason = reason
        self._last_commit_entry_evidence = {
            "local_time_s": time_s,
            "beacon_id": self.expected_beacon_id,
            "reason": reason,
            "filtered_range_m": held.range_m if held is not None else None,
            "rescue_min_range_m": self._rescue_min_range,
            "rescue_range_rise_m": rescue_range_rise_m,
            "bearing_deg": held.bearing_deg if held is not None else None,
            "visual_center_x": visual_target.center_x if visual_target is not None else None,
            "visual_center_y": visual_target.center_y if visual_target is not None else None,
            "visual_confidence": visual_target.confidence if visual_target is not None else None,
            "visual_area_fraction": (
                visual_target.area_fraction if visual_target is not None else None
            ),
            "centered_streak": self._centered_streak,
            "close_commit_streak": self._close_commit_streak,
        }
        self._close_commit_streak = 0
        self._last_commit_center_x = (
            visual_target.center_x if visual_target is not None else None
        )

    def _commit_slid_out_sideways(
        self,
        held: Optional[BeaconEstimate],
        camera_present: bool,
        visual_target: Optional[VisionTarget],
    ) -> bool:
        """Detect a push that is passing beside the gate instead of through it.

        Only meaningful while the vehicle is still ``commit_abort_min_range_m``
        or more from the beacon: in the final meter the bars legitimately
        expand past the image edges. The last roughly-ahead observation is
        also recorded here for the advancement evidence.
        """
        config = self.config
        if self._rear_bearing_confirmed:
            # Once fresh acoustic packets put the expected beacon behind the
            # rover, a later camera target cannot be the pre-passage gate
            # sliding out sideways.  It is commonly a fragment of the passed
            # frame or the next gate, so the pre-passage abort no longer
            # applies.
            return False
        beyond_breakup_zone = (
            self._filtered_range is not None
            and self._filtered_range > config.commit_abort_min_range_m
        )
        if camera_present and visual_target is not None and beyond_breakup_zone:
            alpha = config.commit_center_x_ema_alpha
            if self._last_commit_center_x is None:
                self._last_commit_center_x = visual_target.center_x
            else:
                self._last_commit_center_x = (
                    (1.0 - alpha) * self._last_commit_center_x + alpha * visual_target.center_x
                )
            if abs(self._last_commit_center_x) >= config.commit_abort_center_x:
                self._offcenter_streak += 1
            else:
                self._offcenter_streak = 0
        return self._offcenter_streak >= config.commit_abort_frames

    def _update_disappear_streak(
        self,
        camera_present: bool,
        visual_target: Optional[VisionTarget],
    ) -> None:
        if not self._close_range_confirmed:
            # Visual loss before a verified close approach is not post-passage
            # evidence and must not be banked for a later range turnaround.
            self._disappear_streak = 0
            return
        if not camera_present:
            # A missing camera frame is not evidence the gate disappeared.
            return
        gate_still_visible = (
            visual_target is not None
            and visual_target.area_fraction >= self._visible_gate_area_threshold()
        )
        self._disappear_streak = 0 if gate_still_visible else self._disappear_streak + 1

    def _visible_gate_area_threshold(self) -> float:
        """Minimum detection area that still counts as 'the expected gate'.

        Scaled by the apparent size the expected gate would have at the
        current filtered beacon range, so a smaller gate further along the
        course does not masquerade as the expected one after a passage.
        """
        config = self.config
        threshold = config.disappear_area_fraction
        if self._filtered_range is not None and self._filtered_range > 0.5:
            half_fov = math.radians(config.camera_fov_deg / 2.0)
            apparent_width = (config.nominal_gate_width_m + 0.4) / (
                2.0 * self._filtered_range * math.tan(half_fov)
            )
            expected_area = min(1.0, apparent_width) ** 2
            threshold = max(threshold, config.disappear_expected_area_factor * expected_area)
        return threshold

    def _exit_evidence_complete(self) -> bool:
        config = self.config
        if self._commit_displacement_m < config.min_commit_displacement_m:
            return False
        if self._min_range is None or self._min_range > config.min_range_for_passage_m:
            return False
        if not self._close_range_confirmed:
            return False
        if self._filtered_range is None:
            return False
        normal_range_turnaround = (
            self._filtered_range - self._min_range >= config.range_rise_margin_m
        )
        if (
            not normal_range_turnaround
            and not self._rear_exit_range_rise_confirmed
        ):
            return False
        if self._disappear_streak < config.visual_disappear_frames:
            return False
        if not self._rear_bearing_confirmed:
            return False
        return True

    def _try_confirm_exit(self, time_s: float) -> bool:
        if not self._exit_evidence_complete():
            return False
        self._advance(time_s)
        return True

    def _advance(self, time_s: float) -> None:
        range_rise_m = None
        if self._filtered_range is not None and self._min_range is not None:
            range_rise_m = max(0.0, self._filtered_range - self._min_range)
        self._last_advancement_evidence = {
            "local_time_s": time_s,
            "beacon_id": self.expected_beacon_id,
            "filtered_range_m": self._filtered_range,
            "min_range_m": self._min_range,
            "range_rise_m": range_rise_m,
            "close_range_streak": self._close_range_streak,
            "close_range_confirmed": self._close_range_confirmed,
            "rear_bearing_streak": self._rear_bearing_streak,
            "rear_bearing_confirmed": self._rear_bearing_confirmed,
            "rear_exit_range_rise_streak": self._rear_exit_range_rise_streak,
            "rear_exit_range_rise_confirmed": self._rear_exit_range_rise_confirmed,
            "last_bearing_deg": self._held.bearing_deg if self._held is not None else None,
            "commit_displacement_m": round(self._commit_displacement_m, 3),
            "disappear_streak": self._disappear_streak,
            "last_commit_center_x": self._last_commit_center_x,
        }
        self.local_completed += 1
        self._advancements += 1
        final_beacon = self.local_beacon_index == self.total_beacons - 1
        if final_beacon and self.local_lap >= self.laps:
            self.status = STATUS_FINISHED
            self.phase = PHASE_FINISHED
        else:
            if final_beacon:
                self.local_beacon_index = 0
                self.local_lap += 1
            else:
                self.local_beacon_index += 1
            self.phase = PHASE_ADVANCE
            self._advance_started_at = time_s
        # Reset per-beacon evidence for the new expected beacon.
        self._held = None
        self._pending_range = None
        self._filtered_range = None
        self._last_evidence_packet_received_at_s = None
        self._min_range = None
        self._rescue_min_range = None
        self._close_range_streak = 0
        self._close_range_confirmed = False
        self._rear_bearing_streak = 0
        self._rear_bearing_confirmed = False
        self._rear_exit_range_rise_streak = 0
        self._rear_exit_range_rise_confirmed = False
        self._centered_streak = 0
        self._close_commit_streak = 0
        self._disappear_streak = 0
        self._commit_started_at = None
        self._verify_started_at = None
        self._commit_displacement_m = 0.0
        self._last_visual_at = None
        self._offcenter_streak = 0
        self._last_commit_center_x = None
        self._commit_entry_reason = None
        self._recent_beacon_history.clear()
        self._recent_visual_history.clear()
        self._latest_visual_target = None

    def _regress_after_failed_commit(
        self,
        held: Optional[BeaconEstimate],
        visual_target: Optional[VisionTarget],
    ) -> None:
        self._centered_streak = 0
        self._close_commit_streak = 0
        self._disappear_streak = 0
        self._commit_started_at = None
        self._verify_started_at = None
        self._commit_displacement_m = 0.0
        self._offcenter_streak = 0
        self._last_commit_center_x = None
        self._commit_entry_reason = None
        # The min-range history also restarts: a failed attempt must not lend
        # its close approach to a later, unrelated range rise.
        self._min_range = None
        self._close_range_streak = 0
        self._close_range_confirmed = False
        self._rear_bearing_streak = 0
        self._rear_bearing_confirmed = False
        self._rear_exit_range_rise_streak = 0
        self._rear_exit_range_rise_confirmed = False
        if visual_target is not None:
            self.phase = PHASE_VISUAL_ALIGN
        elif held is not None:
            self.phase = PHASE_APPROACH
        else:
            self.phase = PHASE_SEARCH

    def _result(
        self,
        held: Optional[BeaconEstimate],
        visual_targets: List[VisionTarget],
        visual_target: Optional[VisionTarget] = None,
        just_advanced: bool = False,
    ) -> TrackerStep:
        return TrackerStep(
            phase=self.phase,
            status=self.status,
            expected_beacon_id=self.expected_beacon_id,
            local_beacon_index=self.local_beacon_index,
            local_completed=self.local_completed,
            local_lap=self.local_lap,
            beacon=held,
            visual_target=visual_target,
            visual_targets=visual_targets,
            just_advanced=just_advanced,
            finished=self.finished,
        )


def _beacon_id_from_index(index: int) -> str:
    return f"B{index + 1:02d}"


def _beacon_index_from_id(beacon_id: str, total_beacons: int) -> int:
    text = str(beacon_id or "").strip().upper()
    if not text.startswith("B"):
        raise ValueError(f"Invalid beacon id '{beacon_id}'.")
    try:
        index = int(text[1:]) - 1
    except ValueError as exc:
        raise ValueError(f"Invalid beacon id '{beacon_id}'.") from exc
    if not 0 <= index < total_beacons:
        raise ValueError(
            f"Beacon id '{beacon_id}' is outside the mission range B01..B{total_beacons:02d}."
        )
    return index


def _latest_packet_for(beacons: Any, beacon_id: str) -> Optional[Mapping[str, Any]]:
    if not isinstance(beacons, (list, tuple)):
        return None
    latest: Optional[Mapping[str, Any]] = None
    latest_time = float("-inf")
    for packet in beacons:
        if not isinstance(packet, Mapping):
            continue
        if str(packet.get("beacon_id")) != beacon_id:
            continue
        received = _safe_float(packet.get("received_at_s"), 0.0) or 0.0
        if received >= latest_time:
            latest = packet
            latest_time = received
    return latest


def _camera_frame_present(image: Any) -> bool:
    if image is None:
        return False
    shape = getattr(image, "shape", None)
    if shape is not None:
        return len(shape) >= 2 and int(shape[0]) > 0 and int(shape[1]) > 0
    if isinstance(image, list):
        return bool(image) and isinstance(image[0], list) and bool(image[0])
    return False


def _forward_velocity(dvl_velocity: Any) -> Optional[float]:
    if dvl_velocity is None:
        return None
    value = dvl_velocity
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        for key in ("vx", "x", "velocity_x"):
            if key in value:
                return _safe_float(value.get(key), None)
        return None
    if isinstance(value, (list, tuple)) and value:
        return _safe_float(value[0], None)
    return _safe_float(value, None)


def _safe_float(value: Any, default: Optional[float]) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    if converted != converted or converted in (float("inf"), float("-inf")):
        return default
    return converted
