"""Optional motion compensation layer for high-level controller commands.

Only the pass-through (no-compensation) layer ships. Designing a controller
that rejects current-induced drift from the legal observation is left to future
work; the selection point below is kept so such a layer can be added later
without changing the runner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping


MOTION_COMPENSATION_NONE = "none"
MOTION_COMPENSATION_MODES = (MOTION_COMPENSATION_NONE,)


@dataclass
class MotionCompensationDiagnostics:
    active: bool = False
    reason: str = "not_run"
    raw_command: dict[str, float] = field(default_factory=dict)
    compensated_command: dict[str, float] = field(default_factory=dict)

    def as_event_payload(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "reason": self.reason,
            "raw_command": dict(self.raw_command),
            "compensated_command": dict(self.compensated_command),
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
        )
        return copied

    def diagnostics(self) -> MotionCompensationDiagnostics:
        return self._last_diagnostics


def make_motion_compensator(mode: str | None) -> NoMotionCompensator:
    # Validates the mode (rejecting anything other than "none") and returns the
    # pass-through layer. A future compensator would be dispatched here.
    normalize_motion_compensation_mode(mode)
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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
