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


@dataclass(frozen=True)
class VisionTarget:
    center_x: float
    center_y: float
    confidence: float


def _vision_target_from_camera(image: Any) -> VisionTarget | None:
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        height = int(shape[0])
        width = int(shape[1])
    elif isinstance(image, list) and image and isinstance(image[0], list):
        height = len(image)
        width = len(image[0])
    else:
        return None
    if width <= 0 or height <= 0:
        return None

    step = max(1, int(max(width, height) / 80))
    selected = 0
    sampled = 0
    sum_x = 0.0
    sum_y = 0.0
    min_x = width
    max_x = -1
    min_y = height
    max_y = -1

    for y in range(0, height, step):
        for x in range(0, width, step):
            pixel = _pixel_channels(image, x, y)
            if pixel is None:
                continue
            sampled += 1
            if not _looks_like_gate_bar_pixel(pixel):
                continue
            selected += 1
            sum_x += x
            sum_y += y
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    if sampled == 0 or selected < max(8, int(0.004 * sampled)):
        return None
    box_width = max_x - min_x + 1
    box_height = max_y - min_y + 1
    if box_width < width * 0.03 or box_height < height * 0.03:
        return None

    center_x_px = sum_x / selected
    center_y_px = sum_y / selected
    center_x = _clamp((center_x_px - (width - 1) * 0.5) / max(1.0, (width - 1) * 0.5), -1.0, 1.0)
    center_y = _clamp((center_y_px - (height - 1) * 0.5) / max(1.0, (height - 1) * 0.5), -1.0, 1.0)
    coverage = selected / sampled
    box_area = (box_width * box_height) / max(1.0, width * height)
    confidence = _clamp(0.45 * min(1.0, coverage / 0.04) + 0.55 * min(1.0, box_area / 0.20), 0.0, 1.0)
    return VisionTarget(center_x=center_x, center_y=center_y, confidence=confidence)


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


def _zero_command() -> dict[str, float]:
    return {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _rate_limit(previous: float, target: float, max_delta: float) -> float:
    return previous + _clamp(target - previous, -max_delta, max_delta)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
