"""Dataclasses used by the marine race arena configuration system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Vector3 = Tuple[float, float, float]
Vector2 = Tuple[float, float]


@dataclass(frozen=True)
class BoundsConfig:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains(self, position: Vector3) -> bool:
        x, y, z = position
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )


@dataclass(frozen=True)
class RaceConfig:
    name: str
    format: str
    laps: int
    expected_gates_per_lap: int
    timing_mode: str
    max_duration_s: float
    official_mode: bool


@dataclass(frozen=True)
class WorldConfig:
    package: str
    map: str
    arena_origin: Vector3
    bounds: BoundsConfig
    preferred_environment: str = "OpenWater-Hovering"
    fallback_environment: str = "PierHarbor-Hovering"


@dataclass(frozen=True)
class TrackSettings:
    declared_length_m: float
    length_tolerance_m: float
    gate_inner_size_m: Vector2
    gate_bar_thickness_m: float
    gate_depth_m: float
    gate_sequence: List[str]


@dataclass(frozen=True)
class StartConfig:
    position: Vector3
    rotation_rpy_deg: Vector3


@dataclass(frozen=True)
class FinishConfig:
    gate_id: str


@dataclass(frozen=True)
class BeaconConfig:
    enabled: bool = True
    id: Optional[str] = None
    mode: str = "active_when_target"
    position_offset: Vector3 = (0.0, 0.0, 0.35)
    range_m: float = 50.0
    noise_std: float = 0.0
    dropout_probability: float = 0.0
    update_rate_hz: float = 10.0
    message: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateConfig:
    id: str
    type: str
    position: Vector3
    rotation_rpy_deg: Vector3
    inner_size_m: Vector2
    bar_thickness_m: float
    color: Any
    passage_direction: Vector3
    linked_gate: Optional[str] = None
    beacon: Optional[BeaconConfig] = None


@dataclass(frozen=True)
class CurrentConfig:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParticipantConfig:
    id: str
    vehicle: str
    controller: str
    controller_class: Optional[str]
    spawn: Dict[str, Any]
    sensors: Any
    control_mode: str
    official_sensor_profile: bool


@dataclass(frozen=True)
class RefereeConfig:
    gate_validation: Dict[str, Any] = field(default_factory=dict)
    penalties: Dict[str, Any] = field(default_factory=dict)
    scoring: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackConfig:
    race: RaceConfig
    world: WorldConfig
    track: TrackSettings
    start: StartConfig
    finish: FinishConfig
    gates: List[GateConfig]
    beacon: BeaconConfig
    currents: List[CurrentConfig]
    participants: List[ParticipantConfig]
    referee: RefereeConfig
    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def gate_by_id(self, gate_id: str) -> GateConfig:
        for gate in self.gates:
            if gate.id == gate_id:
                return gate
        raise KeyError(f"Unknown gate id: {gate_id}")

