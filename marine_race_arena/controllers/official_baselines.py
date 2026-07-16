"""Official onboard-only baseline controllers.

Both controllers consume exclusively the official observation contract —
``local_time_s``, onboard ``sensors`` (FrontCamera, DepthSensor, IMUSensor,
DVLSensor, optional CollisionSensor), and the received ``beacons`` packet
list — plus the static mission information given at reset. Course progression
(which beacon is expected, when a gate counts as passed, when the mission is
finished) is estimated entirely by the controller's own
:class:`~marine_race_arena.controllers.local_course_tracker.LocalCourseTracker`.
No referee feedback of any kind is read, and none exists in the observation.

The two controllers share the approach behavior and differ only in how they
traverse the final meters of a gate:

* :class:`RuleGateBaselineController` keeps visual-servoing on the gate all the
  way through the aperture (continuous servo).
* :class:`RuleGateCenterThenCommitController` freezes its heading and depth
  once the tracker enters COMMIT and pushes straight through (center-then-
  commit), ignoring image-center chatter in the last meters.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping

from marine_race_arena.controllers.local_course_tracker import (
    PHASE_ADVANCE,
    PHASE_COMMIT,
    PHASE_VERIFY_EXIT,
    LocalCourseTracker,
    TrackerStep,
)
from marine_race_arena.controllers.vision import VisionTarget
from marine_race_arena.participants.controller_interface import BaseController

LOGGER = logging.getLogger(__name__)


class RuleGateBaselineController(BaseController):
    """Official rule-based gate controller: beacon approach + visual servo."""

    debug_only = False
    uses_ground_truth = False
    align_before_surge_bearing_deg = 24.0
    turn_in_place_bearing_deg = 34.0
    visual_yaw_only_error_threshold = 0.18
    turn_latch_bearing_deg = 75.0
    alignment_brake_range_m = 3.0
    alignment_brake_surge = -0.10
    visual_required_range_m = 5.5

    def reset(self, mission_info: dict[str, Any]) -> None:
        limits = _command_limits(mission_info)
        self.max_surge = min(limits["surge"], 0.46)
        self.max_sway = min(limits["sway"], 0.30)
        self.max_heave = min(limits["heave"], 0.34)
        self.max_yaw = min(limits["yaw"], 0.16)
        self.participant_id = str(mission_info.get("participant_id", ""))
        self.tracker = LocalCourseTracker(
            initial_beacon_id=str(mission_info.get("initial_beacon_id", "B01")),
            total_beacons=int(mission_info.get("total_beacons", 1)),
            laps=int(mission_info.get("laps", 1)),
        )
        self._last_command = _zero_command()
        self._turn_sign = 0.0
        self._visual_turn_active = False
        self._nominal_depth_m: float | None = None
        self._exit_hold_depth_m: float | None = None
        self._passage_depth_m: float | None = None

    def step(self, observation: dict[str, Any]) -> dict[str, float]:
        sensors = _mapping(observation.get("sensors"))
        tracker_step = self.tracker.update(
            local_time_s=_safe_float(observation.get("local_time_s"), 0.0),
            beacons=observation.get("beacons", []),
            camera_image=sensors.get("FrontCamera"),
            dvl_velocity=sensors.get("DVLSensor"),
        )
        self._update_depth_hold(sensors)
        if tracker_step.just_advanced:
            self._on_local_advance(sensors)

        if tracker_step.finished:
            # Local mission complete: hold position permanently.
            self._last_command = _zero_command()
            return _zero_command()

        target = self._phase_target(tracker_step, sensors)
        smoothed = _smooth_command(self._last_command, target, alpha=0.45)
        command = _limit_command_delta(
            self._last_command,
            smoothed,
            limits={"surge": 0.10, "sway": 0.05, "heave": 0.05, "yaw": 0.03},
        )
        command["yaw"] = _clamp(command["yaw"], -self.max_yaw, self.max_yaw)
        self._last_command = command
        return dict(command)

    def close(self) -> None:
        pass

    # ----------------------------------------------------------- behaviors

    def _on_local_advance(self, sensors: Mapping[str, Any]) -> None:
        self._turn_sign = 0.0
        self._visual_turn_active = False
        self._exit_hold_depth_m = _depth_m_from_sensors(sensors)
        # Each gate establishes a new onboard depth reference. Keeping the
        # original spawn depth forever would fight legitimate vertical course
        # changes such as Vertical Serpent.
        if self._exit_hold_depth_m is not None:
            self._nominal_depth_m = self._exit_hold_depth_m
        self._passage_depth_m = None
        self._last_command = dict(self._last_command)
        self._last_command["surge"] = min(float(self._last_command.get("surge", 0.0)), 0.08)

    def _phase_target(self, tracker_step: TrackerStep, sensors: Mapping[str, Any]) -> dict[str, float]:
        if tracker_step.phase == PHASE_ADVANCE:
            return self._exit_gate_command(tracker_step, sensors)
        if tracker_step.phase in (PHASE_COMMIT, PHASE_VERIFY_EXIT):
            return self._through_gate_servo(tracker_step, sensors)
        self._passage_depth_m = None
        return self._navigation_command(tracker_step, sensors)

    def _through_gate_servo(
        self,
        tracker_step: TrackerStep,
        sensors: Mapping[str, Any],
    ) -> dict[str, float]:
        """Continuous-servo passage: keep chasing the image through the aperture.

        Unlike the far-field approach, the passage never brakes or backs off:
        once the local tracker has a stable centered lock, the vehicle keeps a
        forward surge floor through the plane, servos laterally on the image
        while it is visible, and holds the lock depth when the bars leave the
        field of view in the final meters.
        """
        if self._passage_depth_m is None:
            self._passage_depth_m = _depth_m_from_sensors(sensors)
        visual_target = tracker_step.visual_target
        yaw_damping = self._yaw_damping_command(sensors)
        if visual_target is not None:
            error_x = visual_target.center_x
            error_y = visual_target.center_y
            surge = 0.24 + 0.08 * _clamp(visual_target.area_fraction / 0.12, 0.0, 1.0)
            surge = max(0.20, surge * (1.0 - 0.45 * min(1.0, abs(error_x))))
            heave = self._depth_hold_heave(sensors, target_depth_m=self._passage_depth_m)
            # The camera is useful for fine vertical centering, but its blob
            # centroid becomes biased toward the last visible bar as the gate
            # fills the image.  Bound that correction so it cannot overpower
            # the real DepthSensor hold during the physical passage.
            visual_heave = _clamp(-0.30 * error_y, -0.04, 0.04)
            heave = _clamp(heave + visual_heave, -self.max_heave, self.max_heave)
            return {
                "surge": _clamp(surge, 0.20, self.max_surge),
                # COMMIT is a heading lock. Translate toward the remaining
                # image error, but do not rotate after the near-field contour
                # becomes partial; doing so made curved-gate exits turn almost
                # 90 degrees and lodge the rover in the frame.
                "sway": _clamp(-0.18 * error_x, -0.14, 0.14),
                "heave": heave,
                "yaw": yaw_damping,
            }
        return {
            "surge": 0.22,
            "sway": 0.0,
            "heave": self._depth_hold_heave(sensors, target_depth_m=self._passage_depth_m),
            "yaw": yaw_damping,
        }

    def _navigation_command(self, tracker_step: TrackerStep, sensors: Mapping[str, Any]) -> dict[str, float]:
        visual_target = tracker_step.visual_target
        target = self._beacon_fallback(tracker_step, sensors, visual_target is not None)
        if visual_target is not None and visual_target.confidence >= 0.35:
            target = self._visual_gate_command(tracker_step, sensors, visual_target, target)
        return target

    def _beacon_fallback(
        self,
        tracker_step: TrackerStep,
        sensors: Mapping[str, Any],
        has_visual_target: bool,
    ) -> dict[str, float]:
        beacon = tracker_step.beacon
        if beacon is None:
            # No packet from the expected beacon: hold depth and search slowly.
            return {"surge": 0.08, "sway": 0.0, "heave": self._depth_hold_heave(sensors), "yaw": 0.06}

        bearing_deg = _clamp(beacon.bearing_deg, -100.0, 100.0)
        elevation_deg = _clamp(beacon.elevation_deg, -40.0, 40.0)
        range_m = max(0.0, beacon.range_m)
        bearing_abs = abs(bearing_deg)
        yaw = self._yaw_command_for_bearing(
            bearing_deg,
            max_yaw=self.max_yaw,
            gain_deg=220.0,
            deadband_deg=4.0,
        )
        heave = self._mixed_heave(_centerline_elevation_deg(elevation_deg, range_m), sensors, visual_error_y=None)
        if bearing_abs >= self.align_before_surge_bearing_deg:
            return {
                "surge": self._surge_when_not_aligned(range_m),
                "sway": 0.0,
                "heave": heave,
                "yaw": yaw,
            }
        if range_m <= self.visual_required_range_m and not has_visual_target:
            return {
                "surge": self._surge_when_not_aligned(range_m),
                "sway": 0.0,
                "heave": heave,
                "yaw": yaw,
            }

        alignment = _clamp(1.0 - bearing_abs / self.align_before_surge_bearing_deg, 0.25, 1.0)
        range_speed = 0.10 + 0.24 * _clamp(range_m / 8.0, 0.0, 1.0)
        if range_m <= self.visual_required_range_m:
            range_speed = min(range_speed, 0.16)
        surge = _clamp(range_speed * alignment, 0.06, self.max_surge)
        return {"surge": surge, "sway": 0.0, "heave": heave, "yaw": yaw}

    def _visual_gate_command(
        self,
        tracker_step: TrackerStep,
        sensors: Mapping[str, Any],
        target: VisionTarget,
        fallback: Mapping[str, float],
    ) -> dict[str, float]:
        beacon = tracker_step.beacon
        bearing_deg = beacon.bearing_deg if beacon is not None else 0.0
        bearing_abs = abs(bearing_deg)
        range_m = max(0.0, beacon.range_m) if beacon is not None else 0.0
        error_x = target.center_x
        error_y = target.center_y
        centered_x = abs(error_x) <= 0.13
        centered_y = abs(error_y) <= 0.30
        # The high-surge visual branch must use the same bearing envelope as
        # LocalCourseTracker COMMIT. Otherwise the rover can cross the plane
        # while the tracker is still (correctly) refusing an oblique commit.
        roughly_aligned = bearing_abs <= self.tracker.config.commit_bearing_deg

        # Hysteresis on the turn-in-place decision: without it the command
        # chatters between the beacon-turn and visual-servo branches when the
        # bearing sits exactly on the threshold, freezing the vehicle.
        if self._visual_turn_active:
            self._visual_turn_active = bearing_abs > self.align_before_surge_bearing_deg
        else:
            self._visual_turn_active = bearing_abs >= self.turn_in_place_bearing_deg + 2.0

        if self._visual_turn_active or bearing_abs >= self.turn_in_place_bearing_deg + 2.0:
            yaw = self._yaw_command_for_bearing(
                bearing_deg,
                max_yaw=self.max_yaw,
                gain_deg=220.0,
                deadband_deg=5.0,
            )
            surge = self._surge_when_not_aligned(range_m)
            sway = 0.0
        elif abs(error_x) > 0.08:
            # Turn and translate toward the detection (negative center_x means
            # the gate sits camera-left, which is a positive-yaw turn).
            yaw = _clamp(-0.11 * error_x, -0.09, 0.09)
            sway = (
                _clamp(-0.26 * error_x, -self.max_sway, self.max_sway)
                if abs(error_x) <= 0.45
                else 0.0
            )
            if abs(error_x) > self.visual_yaw_only_error_threshold or range_m <= 4.0:
                surge = self._surge_when_not_aligned(range_m)
            else:
                # Keep a small approach creep while fine-centering at distance;
                # without it the servo can hover indefinitely just outside the
                # commit envelope with a slightly off-center image.
                surge = 0.10
        else:
            yaw = _clamp(0.45 * float(fallback.get("yaw", 0.0)), -0.04, 0.04)
            sway = _clamp(-0.20 * error_x, -self.max_sway, self.max_sway)
            surge = 0.10 if range_m > 2.5 else 0.0

        heave = self._mixed_heave(
            _centerline_elevation_deg(beacon.elevation_deg if beacon is not None else 0.0, range_m),
            sensors,
            visual_error_y=error_y,
        )
        if centered_x and centered_y and roughly_aligned:
            surge = 0.24 + 0.08 * _clamp(target.area_fraction / 0.12, 0.0, 1.0)
            if range_m <= 2.5:
                surge = min(surge, 0.22)
            elif range_m <= 5.0:
                surge = min(surge, 0.28)
        elif bearing_abs >= self.align_before_surge_bearing_deg:
            surge = self._surge_when_not_aligned(range_m)
        return {
            "surge": _clamp(surge, self.alignment_brake_surge, self.max_surge),
            "sway": sway,
            "heave": heave,
            "yaw": yaw,
        }

    def _yaw_command_for_bearing(
        self,
        bearing_deg: float,
        max_yaw: float,
        gain_deg: float,
        deadband_deg: float,
    ) -> float:
        bearing_abs = abs(bearing_deg)
        if bearing_abs < self.align_before_surge_bearing_deg:
            self._turn_sign = 0.0
            yaw_bearing = bearing_deg
        elif bearing_abs >= self.turn_latch_bearing_deg:
            if self._turn_sign == 0.0:
                self._turn_sign = 1.0 if bearing_deg >= 0.0 else -1.0
            yaw_bearing = self._turn_sign * min(bearing_abs, 100.0)
        else:
            self._turn_sign = 1.0 if bearing_deg >= 0.0 else -1.0
            yaw_bearing = bearing_deg
        return _yaw_command_from_bearing(
            yaw_bearing,
            max_yaw=max_yaw,
            gain_deg=gain_deg,
            deadband_deg=deadband_deg,
        )

    def _surge_when_not_aligned(self, range_m: float) -> float:
        if 0.0 < range_m <= self.alignment_brake_range_m:
            return self.alignment_brake_surge
        return 0.0

    def _exit_gate_command(
        self,
        tracker_step: TrackerStep,
        sensors: Mapping[str, Any],
    ) -> dict[str, float]:
        yaw = 0.0
        beacon = tracker_step.beacon
        if beacon is not None and abs(beacon.bearing_deg) <= 25.0:
            yaw = self._yaw_command_for_bearing(
                beacon.bearing_deg,
                max_yaw=0.06,
                gain_deg=260.0,
                deadband_deg=8.0,
            )
        return {
            "surge": 0.24,
            "sway": 0.0,
            "heave": self._depth_hold_heave(sensors, target_depth_m=self._exit_hold_depth_m),
            "yaw": yaw,
        }

    # --------------------------------------------------------- depth control

    def _update_depth_hold(self, sensors: Mapping[str, Any]) -> None:
        if self._nominal_depth_m is not None:
            return
        depth = _depth_m_from_sensors(sensors)
        if depth is not None and math.isfinite(depth):
            self._nominal_depth_m = depth

    def _depth_hold_heave(self, sensors: Mapping[str, Any], target_depth_m: float | None = None) -> float:
        hold_depth_m = self._nominal_depth_m if target_depth_m is None else target_depth_m
        if hold_depth_m is None:
            return 0.0
        depth = _depth_m_from_sensors(sensors)
        if depth is None:
            return 0.0
        depth_error_m = depth - hold_depth_m
        depth_rate = _vertical_velocity_from_sensors(sensors)
        correction = 0.20 * depth_error_m
        if depth_rate is not None:
            correction -= 0.08 * depth_rate
        return _clamp(correction, -0.16, 0.16)

    def _yaw_damping_command(self, sensors: Mapping[str, Any]) -> float:
        """Brake residual COMMIT rotation using only the onboard IMU gyro.

        On BlueROV2 a positive high-level yaw command produces a negative
        ``IMUSensor`` body-z angular rate (verified by the reproducible probe),
        so a damping command has the same sign as the measured rate.
        """

        yaw_rate = _yaw_rate_from_sensors(sensors)
        if yaw_rate is None:
            return 0.0
        return _clamp(0.18 * yaw_rate, -0.07, 0.07)

    def _mixed_heave(
        self,
        beacon_elevation_deg: float,
        sensors: Mapping[str, Any],
        visual_error_y: float | None,
    ) -> float:
        beacon_heave = _heave_command(beacon_elevation_deg, sensors, self.max_heave)
        depth_hold = self._depth_hold_heave(sensors)
        heave = 0.70 * beacon_heave + 0.30 * depth_hold
        if visual_error_y is not None:
            # A partial/near-field gate blob can have a strongly biased image
            # centroid.  Treat it as a bounded trim while beacon elevation and
            # the onboard depth loop remain the primary vertical evidence.
            heave += _clamp(-0.42 * visual_error_y, -0.05, 0.05)
        return _clamp(heave, -self.max_heave, self.max_heave)


class RuleGateCenterThenCommitController(RuleGateBaselineController):
    """Rule baseline with a distinct center-then-commit gate passage.

    The approach, observation contract, command caps and smoothing match
    :class:`RuleGateBaselineController`. Once its local tracker locks a
    centered gate and enters COMMIT, the controller stops chasing image-center
    changes and pushes straight through the aperture with depth hold and a
    small acoustic-heading correction, until the tracker confirms the exit.
    """

    def reset(self, mission_info: dict[str, Any]) -> None:
        super().reset(mission_info)
        self._commit_depth_m: float | None = None
        self._commit_area_fraction = 0.0
        self._commit_logged = False

    @property
    def commit_active(self) -> bool:
        return self.tracker.phase in (PHASE_COMMIT, PHASE_VERIFY_EXIT)

    def _phase_target(self, tracker_step: TrackerStep, sensors: Mapping[str, Any]) -> dict[str, float]:
        if tracker_step.phase in (PHASE_COMMIT, PHASE_VERIFY_EXIT):
            if not self._commit_logged:
                self._commit_depth_m = _depth_m_from_sensors(sensors)
                if tracker_step.visual_target is not None:
                    self._commit_area_fraction = tracker_step.visual_target.area_fraction
                LOGGER.info(
                    "Center-then-commit locked local beacon %s.",
                    tracker_step.expected_beacon_id,
                )
                self._commit_logged = True
            return self._commit_command(tracker_step, sensors)
        self._commit_logged = False
        self._commit_depth_m = None
        return super()._phase_target(tracker_step, sensors)

    def _commit_command(
        self,
        tracker_step: TrackerStep,
        sensors: Mapping[str, Any],
    ) -> dict[str, float]:
        beacon = tracker_step.beacon
        range_m = max(0.0, beacon.range_m) if beacon is not None else 0.0
        if tracker_step.visual_target is not None:
            self._commit_area_fraction = tracker_step.visual_target.area_fraction
        surge = 0.24 + 0.08 * _clamp(self._commit_area_fraction / 0.12, 0.0, 1.0)
        if 0.0 < range_m <= 2.5:
            surge = min(surge, 0.22)
        elif range_m <= 5.0:
            surge = min(surge, 0.28)

        yaw = self._yaw_damping_command(sensors)
        sway = 0.0
        if beacon is not None and abs(beacon.bearing_deg) <= 25.0:
            # Small acoustic lateral correction toward the gate axis; the
            # commit ignores image-center chatter and damps yaw entirely from
            # the onboard gyro rather than turning back toward the beacon.
            sway = _clamp(0.25 * math.sin(math.radians(beacon.bearing_deg)), -0.15, 0.15)
        return {
            "surge": _clamp(surge, 0.0, self.max_surge),
            "sway": sway,
            "heave": self._depth_hold_heave(sensors, target_depth_m=self._commit_depth_m),
            "yaw": yaw,
        }


# ---------------------------------------------------------------- helpers


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _command_limits(mission_info: Mapping[str, Any]) -> dict[str, float]:
    """Positive per-axis command magnitudes from the static mission info."""
    limits = mission_info.get("command_limits")
    result: dict[str, float] = {}
    for axis in ("surge", "sway", "heave", "yaw"):
        magnitude = 0.95
        if isinstance(limits, Mapping):
            bounds = limits.get(axis)
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                magnitude = min(abs(_safe_float(bounds[0], -0.95)), abs(_safe_float(bounds[1], 0.95)))
        result[axis] = _clamp(magnitude, 0.1, 1.0)
    return result


def _safe_float(value: Any, default: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(converted):
        return default
    return converted


def _safe_optional_float(value: Any) -> float | None:
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if hasattr(value, "tolist"):
        value = value.tolist()
        if isinstance(value, list):
            if not value:
                return None
            value = value[0]
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _depth_m_from_sensors(sensors: Mapping[str, Any]) -> float | None:
    """Depth in meters (positive down) from the onboard pressure sensor.

    The DepthSensor reports the z position (negative underwater); depth is the
    negated value, computed onboard from the sensor reading alone.
    """
    reading = _safe_optional_float(sensors.get("DepthSensor"))
    if reading is None:
        return None
    return -reading


def _vertical_velocity_from_sensors(sensors: Mapping[str, Any]) -> float | None:
    dvl = sensors.get("DVLSensor")
    if dvl is None:
        return None
    if hasattr(dvl, "tolist"):
        dvl = dvl.tolist()
    if isinstance(dvl, Mapping):
        for key in ("z", "vz", "velocity_z", "vertical_velocity"):
            if key in dvl:
                return _safe_float(dvl.get(key), 0.0)
        return None
    if isinstance(dvl, (list, tuple)) and len(dvl) >= 3:
        return _safe_float(dvl[2], 0.0)
    return None


def _yaw_rate_from_sensors(sensors: Mapping[str, Any]) -> float | None:
    """Body-z gyro rate from the approved HoloOcean IMUSensor payload."""

    imu = sensors.get("IMUSensor")
    if imu is None:
        return None
    if hasattr(imu, "tolist"):
        imu = imu.tolist()
    if isinstance(imu, Mapping):
        angular = imu.get("angular_velocity", imu.get("gyro"))
        if isinstance(angular, Mapping):
            for key in ("z", "yaw", "wz"):
                if key in angular:
                    return _safe_optional_float(angular.get(key))
        if isinstance(angular, (list, tuple)) and len(angular) >= 3:
            return _safe_optional_float(angular[2])
        return None
    # HoloOcean 2.3.0 ReturnBias=True: [acceleration, angular_velocity,
    # acceleration_bias, angular_velocity_bias], each a three-vector.
    if isinstance(imu, (list, tuple)) and len(imu) >= 2:
        angular = imu[1]
        if isinstance(angular, (list, tuple)) and len(angular) >= 3:
            return _safe_optional_float(angular[2])
    return None


def _yaw_command_from_bearing(
    bearing_deg: float,
    max_yaw: float,
    gain_deg: float,
    deadband_deg: float,
) -> float:
    if abs(bearing_deg) <= deadband_deg:
        return 0.0
    adjusted = math.copysign(abs(bearing_deg) - deadband_deg, bearing_deg)
    return _clamp(adjusted / gain_deg, -max_yaw, max_yaw)


def _centerline_elevation_deg(elevation_deg: float, range_m: float) -> float:
    beacon_height_bias_deg = math.degrees(math.atan2(0.30, max(1.0, range_m)))
    return _clamp(elevation_deg - beacon_height_bias_deg, -35.0, 35.0)


def _heave_command(elevation_deg: float, sensors: Mapping[str, Any], max_heave: float) -> float:
    elevation_rad = math.radians(elevation_deg)
    heave = _clamp(0.95 * math.sin(elevation_rad), -max_heave, max_heave)
    depth_rate = _vertical_velocity_from_sensors(sensors)
    if depth_rate is not None:
        heave = _clamp(heave - 0.12 * depth_rate, -max_heave, max_heave)
    return heave


def _smooth_command(previous: Mapping[str, float], target: Mapping[str, float], alpha: float) -> dict[str, float]:
    return {
        key: _clamp((1.0 - alpha) * float(previous.get(key, 0.0)) + alpha * float(target.get(key, 0.0)), -1.0, 1.0)
        for key in ("surge", "sway", "heave", "yaw")
    }


def _limit_command_delta(
    previous: Mapping[str, float],
    target: Mapping[str, float],
    limits: Mapping[str, float],
) -> dict[str, float]:
    return {
        key: float(previous.get(key, 0.0))
        + _clamp(
            float(target.get(key, 0.0)) - float(previous.get(key, 0.0)),
            -float(limits.get(key, 1.0)),
            float(limits.get(key, 1.0)),
        )
        for key in ("surge", "sway", "heave", "yaw")
    }


def _zero_command() -> dict[str, float]:
    return {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
