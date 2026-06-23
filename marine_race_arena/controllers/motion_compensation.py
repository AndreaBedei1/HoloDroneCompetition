"""Optional motion compensation layers for high-level controller commands."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


MOTION_COMPENSATION_NONE = "none"
MOTION_COMPENSATION_DVL_PI = "dvl_pi"
MOTION_COMPENSATION_MODES = (MOTION_COMPENSATION_NONE, MOTION_COMPENSATION_DVL_PI)


@dataclass
class MotionCompensationDiagnostics:
    active: bool = False
    reason: str = "not_run"
    raw_command: dict[str, float] = field(default_factory=dict)
    compensated_command: dict[str, float] = field(default_factory=dict)
    dvl_velocity_body: tuple[float, float] | None = None
    velocity_error: tuple[float, float] | None = None
    integral_error: tuple[float, float] = (0.0, 0.0)

    def as_event_payload(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "reason": self.reason,
            "raw_command": dict(self.raw_command),
            "compensated_command": dict(self.compensated_command),
            "dvl_velocity_body": _tuple_to_list(self.dvl_velocity_body),
            "velocity_error": _tuple_to_list(self.velocity_error),
            "integral_error": list(self.integral_error),
        }


class NoMotionCompensator:
    """Pass-through compensator used for the default no-compensation mode."""

    mode = MOTION_COMPENSATION_NONE

    def __init__(self) -> None:
        self._last_diagnostics = MotionCompensationDiagnostics(reason="disabled")

    def reset(self) -> None:
        self._last_diagnostics = MotionCompensationDiagnostics(reason="disabled")

    def compensate(self, command: Mapping[str, Any], observation_or_sensors: Mapping[str, Any], dt: float) -> dict[str, Any]:
        del observation_or_sensors, dt
        copied = dict(command)
        high_level = _safe_high_level_command(command)
        self._last_diagnostics = MotionCompensationDiagnostics(
            active=False,
            reason="disabled",
            raw_command=high_level,
            compensated_command=dict(high_level),
            integral_error=(0.0, 0.0),
        )
        return copied

    def diagnostics(self) -> MotionCompensationDiagnostics:
        return self._last_diagnostics


class DVLCurrentCompensator:
    """PI compensation for horizontal body-frame velocity errors.

    The compensator only edits high-level surge and sway commands. It does not
    use pose, ground-truth velocity, yaw feedback, or gate information.
    """

    mode = MOTION_COMPENSATION_DVL_PI

    def __init__(
        self,
        *,
        command_to_velocity_m_s: float = 1.25,
        kp: float = 0.18,
        ki: float = 0.04,
        integral_limit_m_s: float = 0.45,
        max_correction: float = 0.14,
        command_limit: float = 1.0,
        max_compensated_reverse_surge: float = 0.06,
        lateral_slowdown_error_m_s: float = 0.22,
        max_surge_with_lateral_error: float = 0.14,
        yaw_fade_start: float = 0.04,
        yaw_fade_full: float = 0.12,
        min_turn_sway_scale: float = 0.45,
    ) -> None:
        self.command_to_velocity_m_s = max(0.01, float(command_to_velocity_m_s))
        self.kp = float(kp)
        self.ki = float(ki)
        self.integral_limit_m_s = abs(float(integral_limit_m_s))
        self.max_correction = abs(float(max_correction))
        self.command_limit = abs(float(command_limit))
        self.max_compensated_reverse_surge = abs(float(max_compensated_reverse_surge))
        self.lateral_slowdown_error_m_s = abs(float(lateral_slowdown_error_m_s))
        self.max_surge_with_lateral_error = abs(float(max_surge_with_lateral_error))
        self.yaw_fade_start = abs(float(yaw_fade_start))
        self.yaw_fade_full = max(self.yaw_fade_start + 0.001, abs(float(yaw_fade_full)))
        self.min_turn_sway_scale = _clamp(abs(float(min_turn_sway_scale)), 0.0, 1.0)
        self.integral_error = [0.0, 0.0]
        self._last_velocity_error: list[float | None] = [None, None]
        self._last_diagnostics = MotionCompensationDiagnostics(reason="not_run")

    def reset(self) -> None:
        self.integral_error = [0.0, 0.0]
        self._last_velocity_error = [None, None]
        self._last_diagnostics = MotionCompensationDiagnostics(reason="reset")

    def compensate(self, command: Mapping[str, Any], observation_or_sensors: Mapping[str, Any], dt: float) -> dict[str, Any]:
        if "thrusters" in command:
            copied = dict(command)
            self._last_diagnostics = MotionCompensationDiagnostics(
                active=False,
                reason="thruster_command_unsupported",
                compensated_command=_safe_high_level_command({}),
            )
            return copied

        raw = _safe_high_level_command(command)
        sensors = _sensors_mapping(observation_or_sensors)
        velocity = extract_body_velocity_xy(sensors)
        if velocity is None:
            self._last_diagnostics = MotionCompensationDiagnostics(
                active=False,
                reason="missing_dvl_velocity",
                raw_command=raw,
                compensated_command=dict(raw),
                integral_error=tuple(self.integral_error),
            )
            return dict(raw)

        dt_s = _safe_dt(dt)
        desired_velocity = (
            raw["surge"] * self.command_to_velocity_m_s,
            raw["sway"] * self.command_to_velocity_m_s,
        )
        velocity_error = (
            desired_velocity[0] - velocity[0],
            desired_velocity[1] - velocity[1],
        )
        self._decay_integral_on_error_reversal(0, velocity_error[0])
        self._decay_integral_on_error_reversal(1, velocity_error[1])
        self.integral_error[0] = _clamp(
            self.integral_error[0] + velocity_error[0] * dt_s,
            -self.integral_limit_m_s,
            self.integral_limit_m_s,
        )
        self.integral_error[1] = _clamp(
            self.integral_error[1] + velocity_error[1] * dt_s,
            -self.integral_limit_m_s,
            self.integral_limit_m_s,
        )
        correction_surge = _clamp(
            (self.kp * velocity_error[0] + self.ki * self.integral_error[0]) / self.command_to_velocity_m_s,
            -self.max_correction,
            self.max_correction,
        )
        correction_sway = _clamp(
            (self.kp * velocity_error[1] + self.ki * self.integral_error[1]) / self.command_to_velocity_m_s,
            -self.max_correction,
            self.max_correction,
        )
        turn_scale = self._turn_compensation_scale(raw["yaw"])
        correction_surge *= max(0.5, turn_scale)
        correction_sway *= max(self.min_turn_sway_scale, turn_scale)
        compensated = dict(raw)
        compensated["surge"] = _clamp(raw["surge"] + correction_surge, -self.command_limit, self.command_limit)
        compensated["sway"] = _clamp(raw["sway"] + correction_sway, -self.command_limit, self.command_limit)
        if raw["surge"] >= 0.0:
            compensated["surge"] = max(compensated["surge"], -self.max_compensated_reverse_surge)
        if abs(velocity_error[1]) >= self.lateral_slowdown_error_m_s and compensated["surge"] > 0.0:
            compensated["surge"] = min(compensated["surge"], self.max_surge_with_lateral_error)
        self._last_diagnostics = MotionCompensationDiagnostics(
            active=True,
            reason="dvl_pi",
            raw_command=raw,
            compensated_command=dict(compensated),
            dvl_velocity_body=velocity,
            velocity_error=velocity_error,
            integral_error=tuple(self.integral_error),
        )
        return compensated

    def diagnostics(self) -> MotionCompensationDiagnostics:
        return self._last_diagnostics

    def _decay_integral_on_error_reversal(self, axis: int, error: float) -> None:
        previous = self._last_velocity_error[axis]
        if previous is not None and previous * error < 0.0:
            self.integral_error[axis] *= 0.35
        self._last_velocity_error[axis] = error

    def _turn_compensation_scale(self, yaw_command: float) -> float:
        yaw_abs = abs(yaw_command)
        if yaw_abs <= self.yaw_fade_start:
            return 1.0
        if yaw_abs >= self.yaw_fade_full:
            return 0.0
        return 1.0 - (yaw_abs - self.yaw_fade_start) / (self.yaw_fade_full - self.yaw_fade_start)


def make_motion_compensator(mode: str | None) -> NoMotionCompensator | DVLCurrentCompensator:
    normalized = normalize_motion_compensation_mode(mode)
    if normalized == MOTION_COMPENSATION_DVL_PI:
        return DVLCurrentCompensator()
    return NoMotionCompensator()


def normalize_motion_compensation_mode(mode: str | None) -> str:
    if mode is None:
        return MOTION_COMPENSATION_NONE
    normalized = str(mode).strip().lower()
    if normalized not in MOTION_COMPENSATION_MODES:
        raise ValueError(
            f"motion_compensation '{mode}' is not supported. "
            f"Supported modes: {', '.join(MOTION_COMPENSATION_MODES)}."
        )
    return normalized


def extract_body_velocity_xy(sensors: Mapping[str, Any]) -> tuple[float, float] | None:
    for key in ("DVLSensor", "DVL", "dvl", "VelocitySensor", "velocity_sensor"):
        if key not in sensors:
            continue
        velocity = _extract_velocity_pair(sensors.get(key))
        if velocity is not None:
            return velocity
    for key_pair in (("surge_velocity_m_s", "sway_velocity_m_s"), ("velocity_x_m_s", "velocity_y_m_s")):
        if key_pair[0] in sensors and key_pair[1] in sensors:
            return _velocity_pair_from_values(sensors.get(key_pair[0]), sensors.get(key_pair[1]))
    return None


def _sensors_mapping(observation_or_sensors: Mapping[str, Any]) -> Mapping[str, Any]:
    sensors = observation_or_sensors.get("sensors")
    return sensors if isinstance(sensors, Mapping) else observation_or_sensors


def _extract_velocity_pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        for nested_key in (
            "velocity",
            "Velocity",
            "vel",
            "linear_velocity",
            "body_velocity",
            "velocity_body",
            "dvl_velocity_body",
        ):
            if nested_key in value:
                nested = _extract_velocity_pair(value.get(nested_key))
                if nested is not None:
                    return nested
        for key_pair in (
            ("x", "y"),
            ("vx", "vy"),
            ("surge", "sway"),
            ("u", "v"),
            ("velocity_x", "velocity_y"),
            ("velocity_x_m_s", "velocity_y_m_s"),
        ):
            if key_pair[0] in value and key_pair[1] in value:
                return _velocity_pair_from_values(value.get(key_pair[0]), value.get(key_pair[1]))
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _velocity_pair_from_values(value[0], value[1])
    if hasattr(value, "tolist"):
        try:
            return _extract_velocity_pair(value.tolist())
        except Exception:
            return None
    return None


def _velocity_pair_from_values(x_value: Any, y_value: Any) -> tuple[float, float] | None:
    x = _safe_float(x_value)
    y = _safe_float(y_value)
    if x is None or y is None:
        return None
    return (x, y)


def _safe_high_level_command(command: Mapping[str, Any]) -> dict[str, float]:
    return {
        "surge": _clamp(_safe_float(command.get("surge"), default=0.0) or 0.0, -1.0, 1.0),
        "sway": _clamp(_safe_float(command.get("sway"), default=0.0) or 0.0, -1.0, 1.0),
        "heave": _clamp(_safe_float(command.get("heave"), default=0.0) or 0.0, -1.0, 1.0),
        "yaw": _clamp(_safe_float(command.get("yaw"), default=0.0) or 0.0, -1.0, 1.0),
    }


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(converted):
        return default
    return converted


def _safe_dt(dt: float) -> float:
    converted = _safe_float(dt, default=0.0) or 0.0
    return _clamp(converted, 0.001, 1.0)


def _tuple_to_list(value: tuple[float, float] | None) -> list[float] | None:
    return None if value is None else [float(value[0]), float(value[1])]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
