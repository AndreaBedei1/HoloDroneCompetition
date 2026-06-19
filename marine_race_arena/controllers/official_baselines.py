"""Reproducible official baseline controllers for benchmark evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from marine_race_arena.participants.controller_interface import BaseController


class AcousticBaselineController(BaseController):
    """Deterministic beacon-only baseline using official observation fields."""

    debug_only = False
    uses_ground_truth = False

    def reset(self, race_info: dict[str, Any]) -> None:
        max_command = _clamp(_safe_float(race_info.get("max_command"), 0.85), 0.1, 1.0)
        self.max_surge = min(max_command, 0.72)
        self.max_sway = min(max_command, 0.42)
        self.max_heave = min(max_command, 0.45)
        self.max_yaw = min(max_command, 0.38)
        self._last_command = _zero_command()

    def step(self, observation: dict[str, Any]) -> dict[str, float]:
        beacon = _mapping(observation.get("beacon"))
        sensors = _mapping(observation.get("sensors"))
        race = _mapping(observation.get("race"))
        target = self._target_command(beacon, sensors, race)
        self._last_command = _smooth_command(self._last_command, target, alpha=0.45)
        return dict(self._last_command)

    def close(self) -> None:
        pass

    def _target_command(
        self,
        beacon: Mapping[str, Any],
        sensors: Mapping[str, Any],
        race: Mapping[str, Any],
    ) -> dict[str, float]:
        del race
        if not bool(beacon.get("valid")):
            return {"surge": 0.08, "sway": 0.0, "heave": 0.0, "yaw": 0.10}

        bearing_deg = _clamp(_safe_float(beacon.get("bearing_deg"), 0.0), -120.0, 120.0)
        elevation_deg = _clamp(_safe_float(beacon.get("elevation_deg"), 0.0), -45.0, 45.0)
        range_m = max(0.0, _safe_float(beacon.get("range_m"), 0.0))
        bearing_rad = math.radians(bearing_deg)
        elevation_rad = math.radians(elevation_deg)

        range_factor = _clamp(range_m / 8.0, 0.0, 1.0)
        near_gate_factor = _clamp(range_m / 2.0, 0.25, 1.0)
        alignment_factor = _clamp(math.cos(abs(bearing_rad)), 0.25, 1.0)
        speed = (0.14 + 0.58 * range_factor) * near_gate_factor * alignment_factor

        surge = _clamp(speed * math.cos(elevation_rad), -self.max_surge, self.max_surge)
        sway = _clamp(0.50 * math.sin(bearing_rad), -self.max_sway, self.max_sway)
        heave = _clamp(0.70 * math.sin(elevation_rad), -self.max_heave, self.max_heave)
        yaw = _clamp(bearing_deg / 85.0, -self.max_yaw, self.max_yaw)

        depth_rate = _vertical_velocity_from_sensors(sensors)
        if depth_rate is not None:
            heave = _clamp(heave - 0.08 * depth_rate, -self.max_heave, self.max_heave)

        return {"surge": surge, "sway": sway, "heave": heave, "yaw": yaw}


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


def _smooth_command(previous: Mapping[str, float], target: Mapping[str, float], alpha: float) -> dict[str, float]:
    return {
        key: _clamp((1.0 - alpha) * float(previous.get(key, 0.0)) + alpha * float(target.get(key, 0.0)), -1.0, 1.0)
        for key in ("surge", "sway", "heave", "yaw")
    }


def _zero_command() -> dict[str, float]:
    return {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
