"""Reproducible official baseline controllers for benchmark evaluation."""

from __future__ import annotations

import math
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping

from marine_race_arena.participants.controller_interface import BaseController

LOGGER = logging.getLogger(__name__)

PHASE_APPROACH_GATE = "APPROACH_GATE"
PHASE_TRANSIT_GATE = "TRANSIT_GATE"
PHASE_EXIT_GATE = "EXIT_GATE"

VISION_PHASE_ACOUSTIC_APPROACH = "ACOUSTIC_APPROACH"
VISION_PHASE_TURN_TO_BEACON = "TURN_TO_BEACON"
VISION_PHASE_VISUAL_ALIGN = "VISUAL_ALIGN"
VISION_PHASE_VISUAL_TRANSIT = "VISUAL_TRANSIT"
VISION_PHASE_EXIT_GATE = "EXIT_GATE"


class RuleGateBaselineController(BaseController):
    """Simple official rule-based gate controller using beacon plus FrontCamera."""

    debug_only = False
    uses_ground_truth = False
    align_before_surge_bearing_deg = 24.0
    turn_in_place_bearing_deg = 34.0
    visual_center_x_threshold = 0.13
    visual_center_y_threshold = 0.30
    visual_yaw_only_error_threshold = 0.18
    turn_latch_bearing_deg = 75.0
    alignment_brake_range_m = 3.0
    alignment_brake_surge = -0.10
    visual_required_range_m = 5.5
    exit_steps_after_gate = 35

    def reset(self, race_info: dict[str, Any]) -> None:
        max_command = _clamp(_safe_float(race_info.get("max_command"), 0.85), 0.1, 1.0)
        self.max_surge = min(max_command, 0.46)
        self.max_sway = min(max_command, 0.30)
        self.max_heave = min(max_command, 0.34)
        self.max_yaw = min(max_command, 0.16)
        self._last_command = _zero_command()
        self._turn_sign = 0.0
        self._last_target_key = None
        self._last_completed_gates = 0
        self._exit_steps_remaining = 0
        self._nominal_depth_m = None
        self._exit_hold_depth_m = None

    def step(self, observation: dict[str, Any]) -> dict[str, float]:
        beacon = _mapping(observation.get("beacon"))
        sensors = _mapping(observation.get("sensors"))
        race = _mapping(observation.get("race"))
        self._update_depth_hold(sensors)
        completed_gates = max(0, int(_safe_float(race.get("completed_gates"), self._last_completed_gates)))
        if completed_gates > self._last_completed_gates:
            self._last_completed_gates = completed_gates
            self._exit_steps_remaining = self.exit_steps_after_gate
            self._exit_hold_depth_m = self._current_depth_m(sensors)
            self._last_command = dict(self._last_command)
            self._last_command["surge"] = min(float(self._last_command.get("surge", 0.0)), 0.08)
        target_key = self._target_key(beacon, race)
        if target_key != self._last_target_key:
            self._turn_sign = 0.0
            self._last_target_key = target_key
            self._last_command = dict(self._last_command)
            self._last_command["surge"] = min(float(self._last_command.get("surge", 0.0)), 0.08)
        visual_target = _vision_target_from_camera(sensors.get("FrontCamera"))
        if self._exit_steps_remaining > 0:
            target = self._exit_gate_command(beacon, sensors)
            self._exit_steps_remaining -= 1
        else:
            target = self._beacon_fallback(beacon, sensors, visual_target is not None)
            if visual_target is not None and visual_target.confidence >= 0.35:
                target = self._visual_gate_command(beacon, sensors, visual_target, target)
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

    def _beacon_fallback(
        self,
        beacon: Mapping[str, Any],
        sensors: Mapping[str, Any],
        has_visual_target: bool,
    ) -> dict[str, float]:
        if not bool(beacon.get("valid")):
            return {"surge": 0.0, "sway": 0.0, "heave": self._depth_hold_heave(sensors), "yaw": 0.0}

        bearing_deg = _clamp(_safe_float(beacon.get("bearing_deg"), 0.0), -100.0, 100.0)
        elevation_deg = _clamp(_safe_float(beacon.get("elevation_deg"), 0.0), -40.0, 40.0)
        range_m = max(0.0, _safe_float(beacon.get("range_m"), 0.0))
        bearing_abs = abs(bearing_deg)
        yaw = self._yaw_command_for_bearing(
            bearing_deg,
            max_yaw=self.max_yaw,
            gain_deg=220.0,
            deadband_deg=4.0,
        )
        heave = self._mixed_heave(_centerline_elevation_deg(elevation_deg, range_m), sensors, visual_error_y=None)
        if bearing_abs >= self.turn_in_place_bearing_deg:
            return {
                "surge": self._surge_when_not_aligned(range_m),
                "sway": 0.0,
                "heave": heave,
                "yaw": yaw,
            }
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
        beacon: Mapping[str, Any],
        sensors: Mapping[str, Any],
        target: "VisionTarget",
        fallback: Mapping[str, float],
    ) -> dict[str, float]:
        bearing_deg = _safe_float(beacon.get("bearing_deg"), 0.0) if bool(beacon.get("valid")) else 0.0
        bearing_abs = abs(bearing_deg)
        range_m = max(0.0, _safe_float(beacon.get("range_m"), 0.0))
        error_x = target.center_x
        error_y = target.center_y
        centered_x = abs(error_x) <= self.visual_center_x_threshold
        centered_y = abs(error_y) <= self.visual_center_y_threshold
        roughly_aligned = bearing_abs <= self.align_before_surge_bearing_deg

        if bearing_abs >= self.turn_in_place_bearing_deg:
            yaw = self._yaw_command_for_bearing(
                bearing_deg,
                max_yaw=self.max_yaw,
                gain_deg=220.0,
                deadband_deg=5.0,
            )
            surge = self._surge_when_not_aligned(range_m)
            sway = 0.0
        elif abs(error_x) > 0.08:
            yaw = _clamp(0.11 * error_x, -0.09, 0.09)
            sway = (
                _clamp(-0.26 * error_x, -self.max_sway, self.max_sway)
                if abs(error_x) <= 0.45
                else 0.0
            )
            surge = (
                self._surge_when_not_aligned(range_m)
                if abs(error_x) > self.visual_yaw_only_error_threshold or range_m <= 4.0
                else 0.0
            )
        else:
            yaw = _clamp(0.45 * float(fallback.get("yaw", 0.0)), -0.04, 0.04)
            sway = _clamp(-0.20 * error_x, -self.max_sway, self.max_sway)
            surge = 0.0

        heave = self._mixed_heave(
            _centerline_elevation_deg(_safe_float(beacon.get("elevation_deg"), 0.0), range_m),
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
        beacon: Mapping[str, Any],
        sensors: Mapping[str, Any],
    ) -> dict[str, float]:
        yaw = 0.0
        if bool(beacon.get("valid")):
            yaw = self._yaw_command_for_bearing(
                _safe_float(beacon.get("bearing_deg"), 0.0),
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

    def _update_depth_hold(self, sensors: Mapping[str, Any]) -> None:
        if self._nominal_depth_m is not None:
            return
        depth = self._current_depth_m(sensors)
        if depth is not None and math.isfinite(depth):
            self._nominal_depth_m = depth

    def _current_depth_m(self, sensors: Mapping[str, Any]) -> float | None:
        depth = _safe_optional_float(sensors.get("depth_m"))
        if depth is None:
            depth = _safe_optional_float(sensors.get("DepthSensor"))
        return depth

    def _depth_hold_heave(self, sensors: Mapping[str, Any], target_depth_m: float | None = None) -> float:
        hold_depth_m = self._nominal_depth_m if target_depth_m is None else target_depth_m
        if hold_depth_m is None:
            return 0.0
        depth = self._current_depth_m(sensors)
        if depth is None:
            return 0.0
        depth_error_m = depth - hold_depth_m
        depth_rate = _vertical_velocity_from_sensors(sensors)
        correction = 0.20 * depth_error_m
        if depth_rate is not None:
            correction -= 0.08 * depth_rate
        return _clamp(correction, -0.16, 0.16)

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
            heave -= 0.42 * visual_error_y
        depth = _safe_optional_float(sensors.get("depth_m"))
        if depth is None:
            depth = _safe_optional_float(sensors.get("DepthSensor"))
        if depth is not None and self._nominal_depth_m is not None:
            depth_error_m = depth - self._nominal_depth_m
            if depth_error_m > 0.65 and beacon_elevation_deg > -5.0:
                heave = max(heave, 0.04)
            elif depth_error_m < -0.65 and beacon_elevation_deg < 5.0:
                heave = min(heave, -0.04)
        return _clamp(heave, -self.max_heave, self.max_heave)

    def _target_key(
        self,
        beacon: Mapping[str, Any],
        race: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        target_id = (
            beacon.get("target_gate_id")
            or beacon.get("next_gate_id")
            or race.get("target_gate_id")
            or race.get("next_gate_id")
        )
        target_index = beacon.get("target_sequence_index")
        if target_index is None:
            target_index = beacon.get("next_gate_index")
        if target_index is None:
            target_index = race.get("target_sequence_index")
        if target_index is None:
            target_index = race.get("next_gate_index")
        return (target_id, target_index)


class AcousticBaselineController(BaseController):
    """Deterministic beacon-only baseline using official observation fields."""

    debug_only = False
    uses_ground_truth = False

    def reset(self, race_info: dict[str, Any]) -> None:
        max_command = _clamp(_safe_float(race_info.get("max_command"), 0.85), 0.1, 1.0)
        self.max_surge = min(max_command, 0.88)
        self.max_sway = min(max_command, 0.55)
        self.max_heave = min(max_command, 0.50)
        self.max_yaw = min(max_command, 0.46)
        self._last_command = _zero_command()
        self._last_completed_gates = 0
        self._phase = PHASE_APPROACH_GATE
        self._exit_steps_remaining = 0
        self._step_count = 0
        self._verbose = _env_flag("MARINE_RACE_ACOUSTIC_BASELINE_VERBOSE")

    @property
    def phase(self) -> str:
        return self._phase

    def step(self, observation: dict[str, Any]) -> dict[str, float]:
        beacon = _mapping(observation.get("beacon"))
        sensors = _mapping(observation.get("sensors"))
        race = _mapping(observation.get("race"))
        self._update_phase(beacon, race)
        target = self._target_command(beacon, sensors, race)
        previous = self._last_command
        self._last_command = _smooth_command(previous, target, alpha=0.55)
        self._last_command["yaw"] = _rate_limit(
            float(previous.get("yaw", 0.0)),
            float(target.get("yaw", 0.0)),
            max_delta=0.075,
        )
        self._step_count += 1
        self._log_diagnostics(beacon, target)
        return dict(self._last_command)

    def close(self) -> None:
        pass

    def _update_phase(self, beacon: Mapping[str, Any], race: Mapping[str, Any]) -> None:
        completed_gates = max(0, int(_safe_float(race.get("completed_gates"), self._last_completed_gates)))
        if completed_gates > self._last_completed_gates:
            self._phase = PHASE_EXIT_GATE
            self._exit_steps_remaining = 8
            self._last_completed_gates = completed_gates
            return

        if self._phase == PHASE_EXIT_GATE:
            self._exit_steps_remaining -= 1
            if self._exit_steps_remaining > 0:
                return
            self._phase = PHASE_APPROACH_GATE

        if not bool(beacon.get("valid")):
            self._phase = PHASE_APPROACH_GATE
            return

        range_m = max(0.0, _safe_float(beacon.get("range_m"), 0.0))
        bearing_abs = abs(_safe_float(beacon.get("bearing_deg"), 0.0))
        if range_m <= 2.6 and bearing_abs <= 65.0:
            self._phase = PHASE_TRANSIT_GATE
        elif range_m >= 3.4:
            self._phase = PHASE_APPROACH_GATE

    def _target_command(
        self,
        beacon: Mapping[str, Any],
        sensors: Mapping[str, Any],
        race: Mapping[str, Any],
    ) -> dict[str, float]:
        if not bool(beacon.get("valid")):
            return {"surge": 0.12, "sway": 0.0, "heave": 0.0, "yaw": 0.08}

        bearing_deg = _clamp(_safe_float(beacon.get("bearing_deg"), 0.0), -120.0, 120.0)
        elevation_deg = _clamp(_safe_float(beacon.get("elevation_deg"), 0.0), -45.0, 45.0)
        range_m = max(0.0, _safe_float(beacon.get("range_m"), 0.0))
        bearing_rad = math.radians(bearing_deg)
        centerline_elevation_deg = _centerline_elevation_deg(elevation_deg, range_m)
        elevation_rad = math.radians(centerline_elevation_deg)

        if self._phase == PHASE_EXIT_GATE:
            surge = 0.74
            sway = _clamp(0.10 * math.sin(bearing_rad), -0.12, 0.12)
            yaw = _yaw_command_from_bearing(bearing_deg, max_yaw=0.18, gain_deg=95.0, deadband_deg=8.0)
            heave = _heave_command(centerline_elevation_deg, sensors, self.max_heave)
            return {"surge": surge, "sway": sway, "heave": heave, "yaw": yaw}

        if self._phase == PHASE_TRANSIT_GATE:
            speed = 0.52 + 0.10 * _clamp(range_m / 2.6, 0.0, 1.0)
            sway_gain = 0.28
            yaw = _yaw_command_from_bearing(bearing_deg, max_yaw=0.24, gain_deg=90.0, deadband_deg=5.0)
        else:
            range_factor = _clamp(range_m / 10.0, 0.0, 1.0)
            alignment_factor = _clamp(math.cos(abs(bearing_rad)), 0.45, 1.0)
            speed = (0.34 + 0.54 * range_factor) * alignment_factor
            if range_m < 4.0:
                speed = min(speed, 0.64)
            sway_gain = 0.46
            yaw = _yaw_command_from_bearing(bearing_deg, max_yaw=self.max_yaw, gain_deg=80.0, deadband_deg=3.0)

        surge = _clamp(speed * math.cos(elevation_rad), -self.max_surge, self.max_surge)
        if abs(bearing_deg) > 70.0:
            surge = min(surge, 0.48)
        sway = _clamp(sway_gain * math.sin(bearing_rad), -self.max_sway, self.max_sway)
        heave = _heave_command(centerline_elevation_deg, sensors, self.max_heave)

        return {"surge": surge, "sway": sway, "heave": heave, "yaw": yaw}

    def _log_diagnostics(self, beacon: Mapping[str, Any], command: Mapping[str, float]) -> None:
        if not self._verbose or self._step_count % 10 != 0:
            return
        LOGGER.info(
            "acoustic_baseline phase=%s range=%.2f bearing=%.2f elevation=%.2f "
            "surge=%.3f yaw=%.3f heave=%.3f",
            self._phase,
            _safe_float(beacon.get("range_m"), -1.0),
            _safe_float(beacon.get("bearing_deg"), 0.0),
            _safe_float(beacon.get("elevation_deg"), 0.0),
            command.get("surge", 0.0),
            command.get("yaw", 0.0),
            command.get("heave", 0.0),
        )


class AcousticVisionBaselineController(BaseController):
    """Acoustic baseline with deterministic front-camera local alignment."""

    debug_only = False
    uses_ground_truth = False

    def reset(self, race_info: dict[str, Any]) -> None:
        self.acoustic = AcousticBaselineController()
        self.acoustic.reset(race_info)

    def step(self, observation: dict[str, Any]) -> dict[str, float]:
        command = self.acoustic.step(observation)
        sensors = _mapping(observation.get("sensors"))
        beacon = _mapping(observation.get("beacon"))
        target = _vision_target_from_camera(sensors.get("FrontCamera"))
        if target is None:
            return command

        range_m = _safe_float(beacon.get("range_m"), 20.0)
        if range_m > 14.0 and target.confidence < 0.45:
            return command

        local_weight = 0.35 if range_m > 10.0 else 0.60
        horizontal_correction = -target.center_x
        vertical_correction = -target.center_y
        command["sway"] = _clamp(
            (1.0 - local_weight) * command["sway"] + local_weight * 0.30 * horizontal_correction,
            -0.45,
            0.45,
        )
        command["yaw"] = _clamp(
            (1.0 - local_weight) * command["yaw"] + local_weight * 0.45 * horizontal_correction,
            -0.40,
            0.40,
        )
        command["heave"] = _clamp(
            command["heave"] + local_weight * 0.20 * vertical_correction,
            -0.45,
            0.45,
        )
        command["surge"] = _clamp(command["surge"] * (1.0 - 0.35 * abs(target.center_x)), 0.05, 0.72)
        return command

    def close(self) -> None:
        self.acoustic.close()


class VisionGateBaselineController(BaseController):
    """Official vision-servo baseline for HoloOcean gate traversal."""

    debug_only = False
    uses_ground_truth = False

    def reset(self, race_info: dict[str, Any]) -> None:
        max_command = _clamp(_safe_float(race_info.get("max_command"), 0.85), 0.1, 1.0)
        self.acoustic = AcousticBaselineController()
        self.acoustic.reset(race_info)
        self.max_surge = min(max_command, 0.72)
        self.max_sway = min(max_command, 0.50)
        self.max_heave = min(max_command, 0.48)
        self.max_yaw = min(max_command, 0.08)
        self._phase = VISION_PHASE_ACOUSTIC_APPROACH
        self._last_command = _zero_command()
        self._last_completed_gates = 0
        self._last_visual_target: VisionTarget | None = None
        self._visual_lock_steps = 0
        self._exit_steps_remaining = 0
        self._transit_steps_remaining = 0
        self._recovery_steps_remaining = 0
        self._recovery_mode = "none"
        self._collision_streak = 0
        self._turn_direction = 0.0
        self._turn_release_steps = 0
        self._step_count = 0
        self._verbose = _env_flag("MARINE_RACE_VISION_GATE_BASELINE_VERBOSE")

    @property
    def phase(self) -> str:
        return self._phase

    def step(self, observation: dict[str, Any]) -> dict[str, float]:
        beacon = _mapping(observation.get("beacon"))
        sensors = _mapping(observation.get("sensors"))
        race = _mapping(observation.get("race"))
        visual_target = self._locked_visual_target(
            self._visual_candidate(beacon, _vision_targets_from_camera(sensors.get("FrontCamera")))
        )
        target_sequence_index = int(_safe_float(race.get("target_sequence_index"), 0.0))
        if _collision_active_from_sensors(sensors):
            self._collision_streak += 1
        else:
            self._collision_streak = 0
        if (
            target_sequence_index >= 2
            and self._collision_streak >= 6
            and self._recovery_steps_remaining <= 0
        ):
            if self._phase == VISION_PHASE_EXIT_GATE or _safe_float(beacon.get("range_m"), 0.0) < 2.0:
                self._recovery_steps_remaining = 20
                self._recovery_mode = "scrub"
            else:
                self._recovery_steps_remaining = 18
                self._recovery_mode = "backoff"
            self._collision_streak = 0
        self._update_phase(beacon, race, visual_target)
        acoustic_command = self._acoustic_fallback_command(observation)
        if self._recovery_steps_remaining > 0:
            target_command = self._recovery_command(beacon, acoustic_command, visual_target)
            self._recovery_steps_remaining -= 1
            if self._recovery_steps_remaining <= 0:
                self._recovery_mode = "none"
        else:
            target_command = self._phase_command(acoustic_command, visual_target)
        smoothed = _smooth_command(self._last_command, target_command, alpha=0.48)
        command = _limit_command_delta(
            self._last_command,
            smoothed,
            limits={"surge": 0.08, "sway": 0.08, "heave": 0.07, "yaw": 0.025},
        )
        command["yaw"] = _clamp(command["yaw"], -self.max_yaw, self.max_yaw)
        self._last_command = command
        self._step_count += 1
        self._log_diagnostics(visual_target, command)
        return dict(command)

    def close(self) -> None:
        self.acoustic.close()

    def _acoustic_fallback_command(self, observation: dict[str, Any]) -> dict[str, float]:
        command = self.acoustic.step(observation)
        beacon = _mapping(observation.get("beacon"))
        if not bool(beacon.get("valid")):
            command["yaw"] = 0.0
            command["surge"] = _clamp(command["surge"], 0.08, self.max_surge)
            command["sway"] = _clamp(command["sway"], -self.max_sway, self.max_sway)
            command["heave"] = _clamp(command["heave"], -self.max_heave, self.max_heave)
            return command

        range_m = max(0.0, _safe_float(beacon.get("range_m"), 0.0))
        bearing_deg = _clamp(_safe_float(beacon.get("bearing_deg"), 0.0), -179.0, 179.0)
        bearing_abs = abs(bearing_deg)
        bearing_rad = math.radians(bearing_deg)
        range_factor = _clamp(range_m / 9.0, 0.0, 1.0)
        if self._phase == VISION_PHASE_TURN_TO_BEACON:
            turn_direction = self._turn_yaw_direction(bearing_deg)
            yaw_limit = 0.045 if range_m < 4.0 else min(self.max_yaw, 0.060)
            yaw_magnitude = _clamp((bearing_abs - 8.0) / 340.0, 0.018, yaw_limit)
            turn_amount = _clamp((bearing_abs - 28.0) / 70.0, 0.0, 1.0)
            turn_surge = 0.20 + 0.12 * _clamp(range_m / 7.0, 0.0, 1.0) - 0.08 * turn_amount
            command["surge"] = _clamp(
                turn_surge,
                0.12,
                0.32,
            )
            command["sway"] = _clamp(0.16 * math.sin(bearing_rad), -0.18, 0.18)
            command["yaw"] = -turn_direction * yaw_magnitude
        else:
            speed = 0.24 + 0.42 * range_factor
            turn_slowdown = _clamp(1.0 - bearing_abs / 115.0, 0.24, 1.0)
            command["surge"] = _clamp(speed * turn_slowdown, 0.06, self.max_surge)
            command["sway"] = _clamp(0.18 * math.sin(bearing_rad), -self.max_sway, self.max_sway)
            command["yaw"] = _yaw_command_from_bearing(
                -bearing_deg,
                max_yaw=0.055 if range_m < 3.2 else min(self.max_yaw, 0.12),
                gain_deg=170.0,
                deadband_deg=10.0,
            )
        command["heave"] = _clamp(command["heave"], -self.max_heave, self.max_heave)
        return command

    def _turn_yaw_direction(self, bearing_deg: float) -> float:
        if abs(bearing_deg) <= 1.0:
            return self._turn_direction or 0.0
        return math.copysign(1.0, bearing_deg)

    def _locked_visual_target(self, detected: VisionTarget | None) -> VisionTarget | None:
        if detected is not None and detected.confidence >= 0.38:
            self._last_visual_target = detected
            self._visual_lock_steps = 10
            return detected
        if self._last_visual_target is not None and self._visual_lock_steps > 0:
            self._visual_lock_steps -= 1
            return self._last_visual_target.with_confidence(self._last_visual_target.confidence * 0.75)
        self._last_visual_target = None
        return None

    def _visual_candidate(self, beacon: Mapping[str, Any], detected: list[VisionTarget]) -> VisionTarget | None:
        if not detected or self._phase == VISION_PHASE_EXIT_GATE:
            return None
        selected = _select_visual_target_for_beacon(detected, beacon)
        if selected is None:
            return None
        if self._phase in {VISION_PHASE_VISUAL_ALIGN, VISION_PHASE_VISUAL_TRANSIT}:
            if bool(beacon.get("valid")):
                range_m = max(0.0, _safe_float(beacon.get("range_m"), 999.0))
                bearing_deg = _safe_float(beacon.get("bearing_deg"), 180.0)
                if range_m > 3.2 and (abs(bearing_deg) > 55.0 or _vision_conflicts_with_beacon(selected, bearing_deg)):
                    return None
            return selected
        if not bool(beacon.get("valid")):
            return selected if selected.confidence >= 0.72 and abs(selected.center_x) <= 0.35 else None
        return selected

    def _update_phase(
        self,
        beacon: Mapping[str, Any],
        race: Mapping[str, Any],
        visual_target: VisionTarget | None,
    ) -> None:
        completed_gates = max(0, int(_safe_float(race.get("completed_gates"), self._last_completed_gates)))
        if completed_gates > self._last_completed_gates:
            self._phase = VISION_PHASE_EXIT_GATE
            self._exit_steps_remaining = 24
            self._transit_steps_remaining = 0
            self._visual_lock_steps = 0
            self._last_visual_target = None
            self._recovery_steps_remaining = 0
            self._recovery_mode = "none"
            self._collision_streak = 0
            self._turn_direction = 0.0
            self._turn_release_steps = 0
            self._last_completed_gates = completed_gates
            return

        if self._phase == VISION_PHASE_EXIT_GATE:
            self._exit_steps_remaining -= 1
            if self._exit_steps_remaining > 0:
                return
            self._phase = VISION_PHASE_ACOUSTIC_APPROACH

        bearing_deg = _safe_float(beacon.get("bearing_deg"), 0.0)
        bearing_abs = abs(bearing_deg)
        range_m = max(0.0, _safe_float(beacon.get("range_m"), 0.0))
        beacon_valid = bool(beacon.get("valid"))

        if self._phase in {VISION_PHASE_VISUAL_ALIGN, VISION_PHASE_VISUAL_TRANSIT} and _beacon_points_away(beacon):
            self._phase = VISION_PHASE_ACOUSTIC_APPROACH
            self._transit_steps_remaining = 0
            self._visual_lock_steps = 0
            self._last_visual_target = None
            self._turn_release_steps = 0
            return

        if visual_target is None:
            if self._phase == VISION_PHASE_VISUAL_TRANSIT and self._transit_steps_remaining > 0:
                self._transit_steps_remaining -= 1
                return
            if beacon_valid and (
                range_m >= 3.2
                and (bearing_abs >= 35.0 or (self._phase == VISION_PHASE_TURN_TO_BEACON and bearing_abs > 24.0))
            ):
                self._phase = VISION_PHASE_TURN_TO_BEACON
                if bearing_abs < 120.0 or self._turn_direction == 0.0:
                    self._turn_direction = math.copysign(1.0, bearing_deg or 1.0)
                self._turn_release_steps = 0
                return
            if self._phase == VISION_PHASE_TURN_TO_BEACON and (bearing_abs <= 24.0 or range_m < 3.2):
                self._turn_release_steps += 1
                if self._turn_release_steps < 4:
                    return
            self._turn_direction = 0.0
            self._turn_release_steps = 0
            self._phase = VISION_PHASE_ACOUSTIC_APPROACH
            return

        self._turn_direction = 0.0
        self._turn_release_steps = 0
        if self._phase in {VISION_PHASE_ACOUSTIC_APPROACH, VISION_PHASE_TURN_TO_BEACON}:
            self._phase = VISION_PHASE_VISUAL_ALIGN

        if self._phase == VISION_PHASE_VISUAL_ALIGN:
            horizontal_center_limit = 0.16 if range_m < 4.0 else 0.14
            vertical_center_limit = 0.40 if range_m < 4.0 else 0.30
            centered = (
                abs(visual_target.center_x) <= horizontal_center_limit
                and abs(visual_target.center_y) <= vertical_center_limit
            )
            large_enough = visual_target.area_fraction >= 0.030 or visual_target.confidence >= 0.62
            if centered and large_enough:
                self._phase = VISION_PHASE_VISUAL_TRANSIT
                self._transit_steps_remaining = 22
            return

        if self._phase == VISION_PHASE_VISUAL_TRANSIT:
            self._transit_steps_remaining = max(0, self._transit_steps_remaining - 1)
            if (
                self._transit_steps_remaining <= 0
                and visual_target.area_fraction < 0.010
                and visual_target.confidence < 0.35
            ):
                self._phase = VISION_PHASE_ACOUSTIC_APPROACH

    def _phase_command(
        self,
        acoustic_command: Mapping[str, float],
        visual_target: VisionTarget | None,
    ) -> dict[str, float]:
        if self._phase == VISION_PHASE_EXIT_GATE:
            return {"surge": 0.68, "sway": 0.0, "heave": 0.0, "yaw": 0.0}
        if visual_target is None or self._phase in {VISION_PHASE_ACOUSTIC_APPROACH, VISION_PHASE_TURN_TO_BEACON}:
            return dict(acoustic_command)

        error_x = visual_target.center_x
        error_y = visual_target.center_y
        acoustic_heave = _clamp(float(acoustic_command.get("heave", 0.0)), -self.max_heave, self.max_heave)
        if self._phase == VISION_PHASE_VISUAL_TRANSIT:
            return {
                "surge": _clamp(0.48 + 0.08 * min(1.0, visual_target.area_fraction / 0.15), 0.38, 0.58),
                "sway": _clamp(-0.38 * error_x, -self.max_sway, self.max_sway),
                "heave": _clamp(0.50 * acoustic_heave - 0.28 * error_y, -self.max_heave, self.max_heave),
                "yaw": _clamp(0.030 * error_x, -0.030, 0.030),
            }
        return {
            "surge": _clamp(0.18 + 0.16 * visual_target.confidence, 0.14, 0.38),
            "sway": _clamp(-0.55 * error_x, -self.max_sway, self.max_sway),
            "heave": _clamp(acoustic_heave - 0.38 * error_y, -self.max_heave, self.max_heave),
            "yaw": _clamp(0.09 * error_x, -0.065, 0.065),
        }

    def _recovery_command(
        self,
        beacon: Mapping[str, Any],
        acoustic_command: Mapping[str, float],
        visual_target: VisionTarget | None,
    ) -> dict[str, float]:
        bearing_deg = _safe_float(beacon.get("bearing_deg"), 0.0)
        yaw = _yaw_command_from_bearing(-bearing_deg, max_yaw=0.08, gain_deg=150.0, deadband_deg=8.0)
        sway = 0.0
        heave = _clamp(float(acoustic_command.get("heave", 0.0)), -self.max_heave, self.max_heave)
        if self._recovery_mode == "scrub":
            side_step = 1.0 if (self._step_count // 14) % 2 == 0 else -1.0
            if abs(bearing_deg) > 25.0:
                side_step = math.copysign(1.0, bearing_deg)
            if visual_target is not None and abs(visual_target.center_x) > 0.08:
                side_step = -math.copysign(1.0, visual_target.center_x)
            return {
                "surge": 0.50,
                "sway": 0.42 * side_step,
                "heave": _clamp(heave - 0.08, -self.max_heave, self.max_heave),
                "yaw": 0.0,
            }
        if visual_target is not None:
            sway = _clamp(-0.22 * visual_target.center_x, -0.22, 0.22)
            heave = _clamp(0.50 * heave - 0.20 * visual_target.center_y, -self.max_heave, self.max_heave)
            yaw = _clamp(0.08 * visual_target.center_x + 0.50 * yaw, -0.08, 0.08)
        else:
            side_step = 1.0 if (self._step_count // 18) % 2 == 0 else -1.0
            sway = 0.24 * side_step
            heave = _clamp(heave - 0.12, -self.max_heave, self.max_heave)
        return {"surge": -0.26, "sway": sway, "heave": heave, "yaw": yaw}

    def _log_diagnostics(self, visual_target: VisionTarget | None, command: Mapping[str, float]) -> None:
        if not self._verbose or self._step_count % 10 != 0:
            return
        confidence = visual_target.confidence if visual_target is not None else 0.0
        center_x = visual_target.center_x if visual_target is not None else 0.0
        center_y = visual_target.center_y if visual_target is not None else 0.0
        LOGGER.info(
            "vision_gate_baseline phase=%s visual_confidence=%.3f gate_center=(%.3f, %.3f) "
            "error_x=%.3f error_y=%.3f surge=%.3f sway=%.3f heave=%.3f yaw=%.3f",
            self._phase,
            confidence,
            center_x,
            center_y,
            center_x,
            center_y,
            command.get("surge", 0.0),
            command.get("sway", 0.0),
            command.get("heave", 0.0),
            command.get("yaw", 0.0),
        )


SMOOTH_PHASE_APPROACH = "APPROACH_GATE"
SMOOTH_PHASE_TRANSIT = "TRANSIT_GATE"
SMOOTH_PHASE_EXIT = "EXIT_GATE"


class SmoothGateBaselineController(BaseController):
    """Conservative, smoothness-oriented variation of the rule gate baseline.

    It keeps the same legal inputs (acoustic beacon plus the derived heading and
    depth sensors) and the same approach/transit/exit phase structure as the
    official rule baseline, but deliberately trades speed for smoothness: a lower
    cruise surge, an earlier and gentler brake into the gate, a wider transit
    capture window, softer yaw gains, a longer settle on exit, and tighter command
    smoothing and rate limits. The intended effect is a slower official time with
    lower-overshoot, lower-jerk motion, so the benchmark can compare two legal
    controllers that differ in timing and behaviour rather than only in code.

    Fusion also differs from the rule baseline: this controller servos on the
    beacon alone. It ignores the front camera, which makes it robust when no
    camera is available (for example under the kinematic fallback adapter) while
    remaining a legal official-observation controller. It never reads ground
    truth, referee internals, or hidden gate geometry.
    """

    debug_only = False
    uses_ground_truth = False

    # Conservative command envelope; cf. the faster acoustic beacon baseline.
    cruise_surge = 0.40
    transit_surge = 0.44
    exit_surge = 0.40
    brake_range_m = 3.0  # begin easing off surge within this range of the gate
    transit_range_m = 3.0  # commit to the transit phase inside this range
    transit_bearing_deg = 55.0
    exit_steps = 12

    def reset(self, race_info: dict[str, Any]) -> None:
        max_command = _clamp(_safe_float(race_info.get("max_command"), 0.85), 0.1, 1.0)
        self.max_surge = min(max_command, 0.42)
        self.max_sway = min(max_command, 0.26)
        self.max_heave = min(max_command, 0.34)
        self.max_yaw = min(max_command, 0.13)
        self._last_command = _zero_command()
        self._last_completed_gates = 0
        self._phase = SMOOTH_PHASE_APPROACH
        self._exit_steps_remaining = 0

    @property
    def phase(self) -> str:
        return self._phase

    def step(self, observation: dict[str, Any]) -> dict[str, float]:
        beacon = _mapping(observation.get("beacon"))
        sensors = _mapping(observation.get("sensors"))
        race = _mapping(observation.get("race"))
        self._update_phase(beacon, race)
        target = self._target_command(beacon, sensors)
        smoothed = _smooth_command(self._last_command, target, alpha=0.35)
        command = _limit_command_delta(
            self._last_command,
            smoothed,
            limits={"surge": 0.06, "sway": 0.04, "heave": 0.04, "yaw": 0.02},
        )
        command["yaw"] = _clamp(command["yaw"], -self.max_yaw, self.max_yaw)
        self._last_command = command
        return dict(command)

    def close(self) -> None:
        pass

    def _update_phase(self, beacon: Mapping[str, Any], race: Mapping[str, Any]) -> None:
        completed_gates = max(0, int(_safe_float(race.get("completed_gates"), self._last_completed_gates)))
        if completed_gates > self._last_completed_gates:
            self._phase = SMOOTH_PHASE_EXIT
            self._exit_steps_remaining = self.exit_steps
            self._last_completed_gates = completed_gates
            return

        if self._phase == SMOOTH_PHASE_EXIT:
            self._exit_steps_remaining -= 1
            if self._exit_steps_remaining > 0:
                return
            self._phase = SMOOTH_PHASE_APPROACH

        if not bool(beacon.get("valid")):
            self._phase = SMOOTH_PHASE_APPROACH
            return

        range_m = max(0.0, _safe_float(beacon.get("range_m"), 0.0))
        bearing_abs = abs(_safe_float(beacon.get("bearing_deg"), 0.0))
        if range_m <= self.transit_range_m and bearing_abs <= self.transit_bearing_deg:
            self._phase = SMOOTH_PHASE_TRANSIT
        elif range_m >= self.transit_range_m + 0.8:
            self._phase = SMOOTH_PHASE_APPROACH

    def _target_command(
        self,
        beacon: Mapping[str, Any],
        sensors: Mapping[str, Any],
    ) -> dict[str, float]:
        if not bool(beacon.get("valid")):
            return {"surge": 0.10, "sway": 0.0, "heave": 0.0, "yaw": 0.06}

        bearing_deg = _clamp(_safe_float(beacon.get("bearing_deg"), 0.0), -120.0, 120.0)
        elevation_deg = _clamp(_safe_float(beacon.get("elevation_deg"), 0.0), -45.0, 45.0)
        range_m = max(0.0, _safe_float(beacon.get("range_m"), 0.0))
        bearing_rad = math.radians(bearing_deg)
        centerline_elevation_deg = _centerline_elevation_deg(elevation_deg, range_m)
        heave = _heave_command(centerline_elevation_deg, sensors, self.max_heave)

        if self._phase == SMOOTH_PHASE_EXIT:
            yaw = _yaw_command_from_bearing(bearing_deg, max_yaw=0.10, gain_deg=140.0, deadband_deg=8.0)
            sway = _clamp(0.06 * math.sin(bearing_rad), -0.10, 0.10)
            return {"surge": self.exit_surge, "sway": sway, "heave": heave, "yaw": yaw}

        if self._phase == SMOOTH_PHASE_TRANSIT:
            yaw = _yaw_command_from_bearing(bearing_deg, max_yaw=0.10, gain_deg=110.0, deadband_deg=4.0)
            sway = _clamp(0.20 * math.sin(bearing_rad), -self.max_sway, self.max_sway)
            surge = _clamp(self.transit_surge, -self.max_surge, self.max_surge)
            return {"surge": surge, "sway": sway, "heave": heave, "yaw": yaw}

        # APPROACH: conservative speed with a strong alignment slowdown and an
        # early, gentle brake as the gate gets close.
        alignment = _clamp(math.cos(bearing_rad), 0.40, 1.0)
        range_factor = _clamp(range_m / 9.0, 0.0, 1.0)
        speed = self.cruise_surge * alignment * (0.60 + 0.40 * range_factor)
        if range_m <= self.brake_range_m:
            speed *= _clamp(range_m / self.brake_range_m, 0.50, 1.0)
        surge = _clamp(speed, 0.05, self.max_surge)
        yaw = _yaw_command_from_bearing(bearing_deg, max_yaw=self.max_yaw, gain_deg=95.0, deadband_deg=3.0)
        sway = _clamp(0.16 * math.sin(bearing_rad), -self.max_sway, self.max_sway)
        return {"surge": surge, "sway": sway, "heave": heave, "yaw": yaw}


@dataclass(frozen=True)
class VisionTarget:
    center_x: float
    center_y: float
    confidence: float
    area_fraction: float = 0.0
    width_fraction: float = 0.0
    height_fraction: float = 0.0

    def with_confidence(self, confidence: float) -> "VisionTarget":
        return VisionTarget(
            center_x=self.center_x,
            center_y=self.center_y,
            confidence=confidence,
            area_fraction=self.area_fraction,
            width_fraction=self.width_fraction,
            height_fraction=self.height_fraction,
        )


def _vision_target_from_camera(image: Any) -> VisionTarget | None:
    return _select_default_visual_target(_vision_targets_from_camera(image))


def _vision_targets_from_camera(image: Any) -> list[VisionTarget]:
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        height = int(shape[0])
        width = int(shape[1])
    elif isinstance(image, list) and image and isinstance(image[0], list):
        height = len(image)
        width = len(image[0])
    else:
        return []
    if width <= 0 or height <= 0:
        return []

    step = max(1, int(max(width, height) / 80))
    sampled = 0
    selected_cells: dict[tuple[int, int], tuple[int, int]] = {}

    for y in range(0, height, step):
        grid_y = y // step
        for x in range(0, width, step):
            pixel = _pixel_channels(image, x, y)
            if pixel is None:
                continue
            sampled += 1
            if not _looks_like_gate_bar_pixel(pixel):
                continue
            selected_cells[(x // step, grid_y)] = (x, y)

    if sampled == 0 or len(selected_cells) < max(8, int(0.004 * sampled)):
        return []

    targets: list[VisionTarget] = []
    all_points = list(selected_cells.values())
    global_target = _vision_target_from_points(all_points, sampled, width, height)
    if global_target is not None:
        targets.append(global_target)

    visited: set[tuple[int, int]] = set()
    for cell in selected_cells:
        if cell in visited:
            continue
        stack = [cell]
        visited.add(cell)
        component_points: list[tuple[int, int]] = []
        while stack:
            current = stack.pop()
            component_points.append(selected_cells[current])
            cx, cy = current
            for nx in range(cx - 1, cx + 2):
                for ny in range(cy - 1, cy + 2):
                    neighbor = (nx, ny)
                    if neighbor in visited or neighbor not in selected_cells:
                        continue
                    visited.add(neighbor)
                    stack.append(neighbor)
        if len(component_points) < max(6, int(0.0015 * sampled)):
            continue
        target = _vision_target_from_points(component_points, sampled, width, height)
        if target is not None:
            targets.append(target)

    deduped: list[VisionTarget] = []
    for target in sorted(targets, key=lambda item: item.confidence, reverse=True):
        if any(abs(target.center_x - existing.center_x) < 0.04 and abs(target.center_y - existing.center_y) < 0.04 for existing in deduped):
            continue
        deduped.append(target)
    return deduped[:8]


def _vision_target_from_points(
    points: list[tuple[int, int]],
    sampled: int,
    width: int,
    height: int,
) -> VisionTarget | None:
    if not points:
        return None
    selected = len(points)
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    box_width = max_x - min_x + 1
    box_height = max_y - min_y + 1
    if box_width < width * 0.025 or box_height < height * 0.025:
        return None

    center_x_px = sum(point[0] for point in points) / selected
    center_y_px = sum(point[1] for point in points) / selected
    center_x = _clamp((center_x_px - (width - 1) * 0.5) / max(1.0, (width - 1) * 0.5), -1.0, 1.0)
    center_y = _clamp((center_y_px - (height - 1) * 0.5) / max(1.0, (height - 1) * 0.5), -1.0, 1.0)
    coverage = selected / sampled
    box_area = (box_width * box_height) / max(1.0, width * height)
    width_fraction = box_width / max(1.0, width)
    height_fraction = box_height / max(1.0, height)
    aspect_ratio = width_fraction / max(1e-6, height_fraction)
    aspect_score = _clamp(1.0 - abs(math.log(max(1e-6, aspect_ratio))) / math.log(3.0), 0.0, 1.0)
    center_score = _clamp(1.0 - 0.45 * abs(center_x) - 0.30 * abs(center_y), 0.0, 1.0)
    confidence = _clamp(
        0.30 * min(1.0, coverage / 0.04)
        + 0.30 * min(1.0, box_area / 0.20)
        + 0.20 * aspect_score
        + 0.20 * center_score,
        0.0,
        1.0,
    )
    return VisionTarget(
        center_x=center_x,
        center_y=center_y,
        confidence=confidence,
        area_fraction=box_area,
        width_fraction=width_fraction,
        height_fraction=height_fraction,
    )


def _pixel_channels(image: Any, x: int, y: int) -> tuple[float, float, float] | None:
    try:
        pixel = image[y][x]
    except (TypeError, IndexError, KeyError):
        return None
    if hasattr(pixel, "tolist"):
        pixel = pixel.tolist()
    if isinstance(pixel, (int, float)):
        value = float(pixel)
        return (value, value, value)
    if not isinstance(pixel, (list, tuple)) or len(pixel) < 3:
        return None
    try:
        return (float(pixel[0]), float(pixel[1]), float(pixel[2]))
    except (TypeError, ValueError):
        return None


def _looks_like_gate_bar_pixel(pixel: tuple[float, float, float]) -> bool:
    high = max(pixel)
    low = min(pixel)
    mean = sum(pixel) / 3.0
    saturation = high - low
    return (high >= 115.0 and saturation >= 35.0) or mean >= 190.0


def _select_visual_target_for_beacon(
    targets: list[VisionTarget],
    beacon: Mapping[str, Any],
) -> VisionTarget | None:
    if not targets:
        return None
    if not bool(beacon.get("valid")):
        return _select_default_visual_target(targets)

    bearing_deg = _safe_float(beacon.get("bearing_deg"), 0.0)
    bearing_abs = abs(bearing_deg)
    range_m = max(0.0, _safe_float(beacon.get("range_m"), 999.0))
    close_default = _select_default_visual_target(targets)
    if range_m <= 3.2 and close_default is not None and close_default.confidence >= 0.55:
        if bearing_abs <= 45.0 or not _vision_conflicts_with_beacon(close_default, bearing_deg):
            return close_default

    scored: list[tuple[float, VisionTarget]] = []
    for target in targets:
        if target.confidence < 0.38:
            continue
        centered_score = _clamp(1.0 - 0.75 * abs(target.center_x) - 0.25 * abs(target.center_y), 0.0, 1.0)
        size_score = _clamp(target.area_fraction / 0.12, 0.0, 1.0)
        if bearing_abs <= 25.0:
            scored.append((target.confidence + 0.35 * centered_score + 0.20 * size_score, target))
            continue

        # Positive acoustic bearing means the selected gate is to camera-left.
        expected_side_amount = -math.copysign(1.0, bearing_deg) * target.center_x
        min_offset = 0.10 if bearing_abs < 45.0 else 0.18
        if expected_side_amount < min_offset:
            continue
        if bearing_abs > 70.0 and range_m > 3.0 and target.confidence < 0.86:
            continue
        side_score = _clamp(expected_side_amount / 0.55, 0.0, 1.0)
        scored.append((target.confidence + 0.65 * side_score + 0.15 * size_score, target))
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _select_default_visual_target(targets: list[VisionTarget]) -> VisionTarget | None:
    if not targets:
        return None
    return max(
        targets,
        key=lambda target: (
            target.confidence
            + 0.30 * _clamp(1.0 - abs(target.center_x), 0.0, 1.0)
            + 0.15 * _clamp(1.0 - abs(target.center_y), 0.0, 1.0)
        ),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_float(value: Any, default: float) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(converted):
        return default
    return converted


def _safe_optional_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def _vertical_velocity_from_sensors(sensors: Mapping[str, Any]) -> float | None:
    for key in ("vertical_velocity_m_s", "heave_velocity_m_s", "velocity_z_m_s"):
        if key in sensors:
            return _safe_float(sensors.get(key), 0.0)
    dvl = sensors.get("DVLSensor") or sensors.get("DVL") or sensors.get("VelocitySensor")
    if isinstance(dvl, Mapping):
        for key in ("z", "vz", "velocity_z", "vertical_velocity"):
            if key in dvl:
                return _safe_float(dvl.get(key), 0.0)
    if isinstance(dvl, (list, tuple)) and len(dvl) >= 3:
        return _safe_float(dvl[2], 0.0)
    return None


def _collision_active_from_sensors(sensors: Mapping[str, Any]) -> bool:
    for key, value in sensors.items():
        if "collision" not in str(key).lower() and "contact" not in str(key).lower():
            continue
        if hasattr(value, "any"):
            try:
                return bool(value.any())
            except Exception:
                return bool(value)
        if isinstance(value, Mapping):
            return any(bool(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(bool(item) for item in value)
        return bool(value)
    return False


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


def _beacon_points_away(beacon: Mapping[str, Any]) -> bool:
    if not bool(beacon.get("valid")):
        return False
    range_m = max(0.0, _safe_float(beacon.get("range_m"), 0.0))
    bearing_abs = abs(_safe_float(beacon.get("bearing_deg"), 0.0))
    return range_m >= 3.5 and bearing_abs >= 95.0


def _vision_conflicts_with_beacon(target: VisionTarget, bearing_deg: float) -> bool:
    if abs(bearing_deg) <= 25.0 or abs(target.center_x) <= 0.10:
        return False
    return bearing_deg * target.center_x > 0.0


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


def _rate_limit(previous: float, target: float, max_delta: float) -> float:
    return previous + _clamp(target - previous, -max_delta, max_delta)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
