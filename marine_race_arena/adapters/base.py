"""Simulator adapter interface for marine races."""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

from marine_race_arena.arena.arena_builder import Arena
from marine_race_arena.arena.gate_factory import VisualGate
from marine_race_arena.config.schema import TrackConfig, Vector3
from marine_race_arena.participants.participant import RaceParticipant

LOGGER = logging.getLogger(__name__)


class RaceAdapterError(RuntimeError):
    """Base error for simulator adapter failures."""


class RaceAdapterUnavailable(RaceAdapterError):
    """Raised when an adapter cannot initialize in the current environment."""


@dataclass(frozen=True)
class AdapterParticipantState:
    participant_id: str
    position: Vector3
    rotation_rpy_deg: Vector3
    raw_sensors: Dict[str, Any] = field(default_factory=dict)


class BaseRaceAdapter(abc.ABC):
    """Hide simulator-specific details from the race runner."""

    name = "base"
    thruster_limit = 1.0

    def __init__(
        self,
        config: TrackConfig,
        arena: Arena,
        seed: Optional[int] = None,
        headless: bool = False,
        record: bool = False,
    ):
        self.config = config
        self.arena = arena
        self.seed = seed
        self.headless = headless
        self.record = record

    @abc.abstractmethod
    def initialize(self) -> None:
        """Initialize simulator resources that do not require participants."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the simulator state."""

    @abc.abstractmethod
    def spawn_participants(self, participants: Mapping[str, RaceParticipant]) -> None:
        """Spawn or initialize participants."""

    @abc.abstractmethod
    def spawn_visual_gates(self, visual_gates: Iterable[VisualGate]) -> None:
        """Spawn optional gate visuals or preserve debug metadata."""

    @abc.abstractmethod
    def get_participant_state(self, participant_id: str) -> AdapterParticipantState:
        """Return ground-truth participant state for the referee/debug path."""

    @abc.abstractmethod
    def get_allowed_sensor_data(self, participant_id: str, sensor_profile: Any) -> Dict[str, Any]:
        """Return controller-safe sensor data."""

    @abc.abstractmethod
    def apply_command(self, participant_id: str, command: Mapping[str, Any], control_mode: str) -> None:
        """Queue or apply a participant command."""

    @abc.abstractmethod
    def get_collision_state(self, participant_id: str) -> bool:
        """Return whether a collision occurred for this participant in the latest tick."""

    @abc.abstractmethod
    def get_current_time(self) -> float:
        """Return adapter simulation time in seconds."""

    @abc.abstractmethod
    def step(self, dt: float) -> None:
        """Advance the simulator by one race tick."""

    @abc.abstractmethod
    def close(self) -> None:
        """Close simulator resources."""

    def clamp_high_level_command(
        self,
        command: Mapping[str, Any],
        participant_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Clamp high-level body-frame commands to safe numeric ranges."""

        safe = {
            "surge": _clamp(_float(command.get("surge", 0.0)), -1.0, 1.0),
            "sway": _clamp(_float(command.get("sway", 0.0)), -1.0, 1.0),
            "heave": _clamp(_float(command.get("heave", 0.0)), -1.0, 1.0),
            "yaw": _clamp(_float(command.get("yaw", 0.0)), -1.0, 1.0),
        }
        if participant_id is not None:
            try:
                position = self.get_participant_state(participant_id).position
            except RaceAdapterError:
                position = None
            if position is not None:
                # z is positive upward. Negative heave drives deeper; positive heave drives toward surface.
                if position[2] <= self.config.world.bounds.z_min + 0.25 and safe["heave"] < 0.0:
                    safe["heave"] = 0.0
                if position[2] >= self.config.world.bounds.z_max - 0.25 and safe["heave"] > 0.0:
                    safe["heave"] = 0.0
        return safe

    def clamp_thruster_command(self, command: Mapping[str, Any]) -> list[float]:
        values = command.get("thrusters", [])
        if not isinstance(values, list):
            values = []
        clamped = [_clamp(_float(value), -self.thruster_limit, self.thruster_limit) for value in values[:8]]
        while len(clamped) < 8:
            clamped.append(0.0)
        return clamped

    def command_to_bluerov2_thrusters(
        self,
        participant_id: str,
        command: Mapping[str, Any],
        control_mode: str,
    ) -> list[float]:
        """Map race commands to BlueROV2 control-scheme-0 thruster values.

        BlueROV2 thruster control expects 8 values. This mapping is conservative
        and intended as an adapter-level baseline, not a precision controller.
        """

        if "thrusters" in command or control_mode == "thrusters":
            return self.clamp_thruster_command(command)

        safe = self.clamp_high_level_command(command, participant_id=participant_id)
        surge = safe["surge"]
        sway = safe["sway"]
        heave = safe["heave"]
        yaw = safe["yaw"]
        vertical = [
            heave,
            heave,
            heave,
            heave,
        ]
        horizontal = [
            surge + sway - 0.35 * yaw,
            surge - sway + 0.35 * yaw,
            surge + sway + 0.35 * yaw,
            surge - sway - 0.35 * yaw,
        ]
        return [
            _clamp(value * self.thruster_limit, -self.thruster_limit, self.thruster_limit)
            for value in vertical + horizontal
        ]

    def filter_sensor_data(
        self,
        raw_sensor_data: Mapping[str, Any],
        sensor_profile: Any,
        official_mode: bool,
    ) -> Dict[str, Any]:
        """Filter simulator sensor data before a participant controller sees it."""

        allowed_names = _configured_sensor_names(sensor_profile)
        filtered: Dict[str, Any] = {}
        for name, value in raw_sensor_data.items():
            if official_mode and _is_ground_truth_sensor(name):
                continue
            if allowed_names and name not in allowed_names and _canonical_sensor_name(name) not in allowed_names:
                continue
            filtered[name] = _json_safe(value)
        return filtered


def _configured_sensor_names(sensor_profile: Any) -> set[str]:
    if not isinstance(sensor_profile, Mapping):
        return set()
    values = []
    for key in ("allowed", "allowed_sensors", "sensors", "holoocean_sensors"):
        configured = sensor_profile.get(key)
        if isinstance(configured, list):
            values.extend(configured)
    names: set[str] = set()
    for value in values:
        if isinstance(value, str):
            names.add(value)
            names.add(_canonical_sensor_name(value))
        elif isinstance(value, Mapping):
            sensor_name = value.get("sensor_name") or value.get("sensor_type")
            if sensor_name:
                names.add(str(sensor_name))
                names.add(_canonical_sensor_name(str(sensor_name)))
    return names


def _is_ground_truth_sensor(sensor_name: str) -> bool:
    canonical = _canonical_sensor_name(sensor_name)
    return canonical in {
        "posesensor",
        "locationsensor",
        "rotationsensor",
        "dynamicssensor",
        "ground_truth",
        "debug_ground_truth",
        "exact_pose",
        "exact_position",
    }


def _canonical_sensor_name(sensor_name: str) -> str:
    return sensor_name.replace("_", "").replace("-", "").lower()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
