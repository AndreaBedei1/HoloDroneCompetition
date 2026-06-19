"""JSON loader for marine race track configurations."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from marine_race_arena.config.benchmark_tasks import (
    BenchmarkTaskConfig,
    normalize_benchmark_task_mode,
)
from marine_race_arena.config.schema import (
    BeaconConfig,
    BoundsConfig,
    CurrentConfig,
    FinishConfig,
    GateConfig,
    ParticipantConfig,
    RaceConfig,
    RefereeConfig,
    StartConfig,
    TrackConfig,
    TrackSettings,
    Vector2,
    Vector3,
    WorldConfig,
)
from marine_race_arena.config.validation import TrackValidationError, validate_track_config


class TrackConfigLoadError(ValueError):
    """Raised when a track JSON file is malformed before semantic validation."""


DEFAULT_OFFICIAL_SENSOR_PROFILE: Dict[str, Any] = {
    "profile": "official_vision_acoustic",
    "allowed_sensors": [
        "DepthSensor",
        "IMUSensor",
        "DVLSensor",
        "VelocitySensor",
        "CollisionSensor",
        "FrontCamera",
        "depth_m",
        "heading_yaw_deg",
        "environment_current_m_s",
        "current_physical_coupling_active",
        "current_coupling_method",
        "control_mode",
    ],
    "holoocean_sensors": [
        {"sensor_type": "DepthSensor", "socket": "DepthSocket", "Hz": 30, "configuration": {"Sigma": 0.0}},
        {"sensor_type": "IMUSensor", "socket": "IMUSocket", "Hz": 30, "configuration": {"ReturnBias": True}},
        {
            "sensor_type": "DVLSensor",
            "socket": "DVLSocket",
            "Hz": 15,
            "configuration": {"Elevation": 22.5, "ReturnRange": True, "MaxRange": 50},
        },
        {
            "sensor_type": "RGBCamera",
            "sensor_name": "FrontCamera",
            "socket": "CameraSocket",
            "rotation": [0.0, 0.0, 0.0],
            "Hz": 30,
            "configuration": {"CaptureWidth": 640, "CaptureHeight": 480, "FovAngle": 90.0},
        },
    ],
}


def load_track_config(
    track_path: str | Path,
    debug: bool = False,
    benchmark_task: str | None = None,
) -> TrackConfig:
    """Load and validate a track JSON file.

    Args:
        track_path: Path to a JSON track file.
        debug: When true, semantic validation errors are reported as warnings by the caller
            only if parsing succeeded. Parser errors are always fatal.
        benchmark_task: Optional CLI/code override for benchmark task validation.

    Returns:
        Parsed TrackConfig.
    """

    path = Path(track_path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise TrackConfigLoadError(f"Could not parse JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise TrackConfigLoadError(f"Could not read track config {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise TrackConfigLoadError("Track JSON root must be an object.")

    config = parse_track_config(raw)
    if benchmark_task is not None:
        config = with_benchmark_task(config, benchmark_task)
    result = validate_track_config(config)
    if result.errors and not debug:
        raise TrackValidationError(result.errors, result.warnings)
    return config


def parse_track_config(raw: Mapping[str, Any]) -> TrackConfig:
    track = _parse_track(_required_mapping(raw, "track"))
    race = _parse_race(_required_mapping(raw, "race"), expected_gates_per_lap=len(track.gate_sequence))
    benchmark_task = _parse_benchmark_task(raw.get("benchmark_task"))
    world = _parse_world(_required_mapping(raw, "world"))
    start = _parse_start(_required_mapping(raw, "start"))
    finish = _parse_finish(_required_mapping(raw, "finish"))
    global_beacon = _parse_beacon(raw.get("beacon", {}), default_id=None)
    gates = [
        _parse_gate(gate, track, global_beacon)
        for gate in _required_list(raw, "gates")
    ]
    currents = [_parse_current(current) for current in raw.get("currents", [])]
    participants = [_parse_participant(participant) for participant in raw.get("participants", [])]
    if not participants:
        participants = [
            ParticipantConfig(
                id="bluerov2_01",
                vehicle="BlueROV2",
                controller="pygame",
                controller_class=None,
                spawn={"position": start.position, "rotation_rpy_deg": start.rotation_rpy_deg},
                sensors=dict(DEFAULT_OFFICIAL_SENSOR_PROFILE),
                control_mode="high_level",
                official_sensor_profile=True,
            )
        ]
    referee = _parse_referee(raw.get("referee", {}))
    obstacles = list(raw.get("obstacles", []))
    return TrackConfig(
        race=race,
        benchmark_task=benchmark_task,
        world=world,
        track=track,
        start=start,
        finish=finish,
        gates=gates,
        beacon=global_beacon,
        currents=currents,
        participants=participants,
        referee=referee,
        obstacles=obstacles,
        raw=dict(raw),
    )


def with_benchmark_task(config: TrackConfig, mode: str | None) -> TrackConfig:
    return replace(
        config,
        benchmark_task=BenchmarkTaskConfig(mode=normalize_benchmark_task_mode(mode)),
    )


def _parse_race(raw: Mapping[str, Any], expected_gates_per_lap: int) -> RaceConfig:
    return RaceConfig(
        name=str(_required(raw, "name")),
        format=str(raw.get("format", "time_trial")),
        laps=int(raw.get("laps", 1)),
        expected_gates_per_lap=int(raw.get("expected_gates_per_lap", expected_gates_per_lap)),
        timing_mode=str(raw.get("timing_mode", "first_gate_to_last_gate")),
        max_duration_s=float(raw.get("max_duration_s", 600.0)),
        official_mode=bool(raw.get("official_mode", False)),
    )


def _parse_benchmark_task(raw: Any) -> BenchmarkTaskConfig:
    if raw is None:
        return BenchmarkTaskConfig()
    if isinstance(raw, str):
        return BenchmarkTaskConfig(mode=normalize_benchmark_task_mode(raw))
    if not isinstance(raw, Mapping):
        raise TrackConfigLoadError("benchmark_task must be a string or an object with a mode field.")
    if "mode" not in raw:
        raise TrackConfigLoadError("benchmark_task object must contain a mode field.")
    return BenchmarkTaskConfig(mode=normalize_benchmark_task_mode(str(raw["mode"])))


def _parse_world(raw: Mapping[str, Any]) -> WorldConfig:
    bounds_raw = _required_mapping(raw, "bounds")
    return WorldConfig(
        package=str(raw.get("package", "Ocean")),
        map=str(raw.get("map", raw.get("preferred_environment", "OpenWater-Hovering"))),
        arena_origin=_vector3(raw.get("arena_origin", [0.0, 0.0, 0.0]), "world.arena_origin"),
        bounds=BoundsConfig(
            x_min=float(_required(bounds_raw, "x_min")),
            x_max=float(_required(bounds_raw, "x_max")),
            y_min=float(_required(bounds_raw, "y_min")),
            y_max=float(_required(bounds_raw, "y_max")),
            z_min=float(_required(bounds_raw, "z_min")),
            z_max=float(_required(bounds_raw, "z_max")),
        ),
        preferred_environment=str(raw.get("preferred_environment", "OpenWater-Hovering")),
        fallback_environment=str(raw.get("fallback_environment", "PierHarbor-Hovering")),
    )


def _parse_track(raw: Mapping[str, Any]) -> TrackSettings:
    return TrackSettings(
        declared_length_m=float(_required(raw, "declared_length_m")),
        length_tolerance_m=float(raw.get("length_tolerance_m", 2.0)),
        gate_inner_size_m=_vector2_or_scalar(
            raw.get("gate_inner_size_m", [1.5, 1.5]), "track.gate_inner_size_m"
        ),
        gate_bar_thickness_m=float(raw.get("gate_bar_thickness_m", 0.18)),
        gate_depth_m=float(raw.get("gate_depth_m", 0.20)),
        gate_sequence=[str(gate_id) for gate_id in _required_list(raw, "gate_sequence")],
    )


def _parse_start(raw: Mapping[str, Any]) -> StartConfig:
    return StartConfig(
        position=_vector3(_required(raw, "position"), "start.position"),
        rotation_rpy_deg=_vector3(raw.get("rotation_rpy_deg", [0.0, 0.0, 0.0]), "start.rotation_rpy_deg"),
    )


def _parse_finish(raw: Mapping[str, Any]) -> FinishConfig:
    return FinishConfig(gate_id=str(_required(raw, "gate_id")))


def _parse_beacon(raw: Any, default_id: Optional[str]) -> BeaconConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TrackConfigLoadError("beacon must be an object when provided.")
    return BeaconConfig(
        enabled=bool(raw.get("enabled", True)),
        id=str(raw.get("id")) if raw.get("id") is not None else default_id,
        mode=str(raw.get("mode", "active_when_target")),
        position_offset=_vector3(raw.get("position_offset", [0.0, 0.0, 0.35]), "beacon.position_offset"),
        range_m=float(raw.get("range_m", 50.0)),
        noise_std=float(raw.get("noise_std", 0.0)),
        dropout_probability=float(raw.get("dropout_probability", 0.0)),
        update_rate_hz=float(raw.get("update_rate_hz", 10.0)),
        message=dict(raw.get("message", {})),
    )


def _parse_gate(raw: Any, track: TrackSettings, global_beacon: BeaconConfig) -> GateConfig:
    if not isinstance(raw, Mapping):
        raise TrackConfigLoadError("Each gate entry must be an object.")
    gate_id = str(_required(raw, "id"))
    beacon_raw = raw.get("beacon")
    if beacon_raw is None:
        beacon = BeaconConfig(
            enabled=global_beacon.enabled,
            id=f"B_{gate_id}" if global_beacon.enabled else None,
            mode=global_beacon.mode,
            position_offset=global_beacon.position_offset,
            range_m=global_beacon.range_m,
            noise_std=global_beacon.noise_std,
            dropout_probability=global_beacon.dropout_probability,
            update_rate_hz=global_beacon.update_rate_hz,
            message=dict(global_beacon.message),
        )
    else:
        merged = dict(global_beacon.__dict__)
        merged.update(dict(beacon_raw))
        if merged.get("id") is None and merged.get("enabled", True):
            merged["id"] = f"B_{gate_id}"
        beacon = _parse_beacon(merged, default_id=f"B_{gate_id}")

    return GateConfig(
        id=gate_id,
        type=str(raw.get("type", "single")),
        position=_vector3(_required(raw, "position"), f"gates.{gate_id}.position"),
        rotation_rpy_deg=_vector3(
            _required(raw, "rotation_rpy_deg"), f"gates.{gate_id}.rotation_rpy_deg"
        ),
        inner_size_m=_vector2_or_scalar(raw.get("inner_size_m", track.gate_inner_size_m), f"{gate_id}.inner_size_m"),
        bar_thickness_m=float(raw.get("bar_thickness_m", track.gate_bar_thickness_m)),
        color=raw.get("color", "#00ff88"),
        passage_direction=_vector3(
            _required(raw, "passage_direction"), f"gates.{gate_id}.passage_direction"
        ),
        linked_gate=str(raw["linked_gate"]) if raw.get("linked_gate") is not None else None,
        beacon=beacon,
    )


def _parse_current(raw: Any) -> CurrentConfig:
    if not isinstance(raw, Mapping):
        raise TrackConfigLoadError("Each current entry must be an object.")
    current_type = str(_required(raw, "type"))
    params = dict(raw)
    params.pop("type", None)
    return CurrentConfig(type=current_type, params=params)


def _parse_participant(raw: Any) -> ParticipantConfig:
    if not isinstance(raw, Mapping):
        raise TrackConfigLoadError("Each participant entry must be an object.")
    return ParticipantConfig(
        id=str(_required(raw, "id")),
        vehicle=str(raw.get("vehicle", "BlueROV2")),
        controller=str(raw.get("controller", "pygame")),
        controller_class=str(raw["controller_class"]) if raw.get("controller_class") is not None else None,
        spawn=dict(raw.get("spawn", {})),
        sensors=raw.get("sensors", {}),
        control_mode=str(raw.get("control_mode", "high_level")),
        official_sensor_profile=bool(raw.get("official_sensor_profile", True)),
    )


def _parse_referee(raw: Any) -> RefereeConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TrackConfigLoadError("referee must be an object when provided.")
    gate_validation_defaults = {
        "vehicle_model": "center_point",
        "vehicle_clearance_margin_m": 0.0,
        "stuck_timeout_s": 45.0,
        "stuck_speed_threshold_m_s": 0.02,
        "timeout_enabled": False,
        "collision_penalty_cooldown_s": 1.0,
        "out_of_bounds_penalty_cooldown_s": 1.0,
    }
    penalty_defaults = {
        "minor_collision_s": 5.0,
        "gate_collision_s": 10.0,
        "out_of_bounds_s": 10.0,
        "stuck_s": 15.0,
        "wrong_direction_s": 0.0,
        "missed_gate_dnf": True,
        "severe_collision_dnf": False,
        "out_of_bounds_dnf": False,
        "wrong_direction_dsq": False,
    }
    scoring_defaults = {
        "rank_finished_by": "penalized_time",
        "rank_unfinished_by": "completed_gates",
    }
    gate_validation_defaults.update(dict(raw.get("gate_validation", {})))
    penalty_defaults.update(dict(raw.get("penalties", {})))
    scoring_defaults.update(dict(raw.get("scoring", {})))
    return RefereeConfig(
        gate_validation=gate_validation_defaults,
        penalties=penalty_defaults,
        scoring=scoring_defaults,
    )


def _required(raw: Mapping[str, Any], key: str) -> Any:
    if key not in raw:
        raise TrackConfigLoadError(f"Missing required field '{key}'.")
    return raw[key]


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _required(raw, key)
    if not isinstance(value, Mapping):
        raise TrackConfigLoadError(f"Field '{key}' must be an object.")
    return value


def _required_list(raw: Mapping[str, Any], key: str) -> list[Any]:
    value = _required(raw, key)
    if not isinstance(value, list):
        raise TrackConfigLoadError(f"Field '{key}' must be a list.")
    return value


def _vector3(value: Any, field_name: str) -> Vector3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise TrackConfigLoadError(f"{field_name} must contain exactly 3 numeric values.")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as exc:
        raise TrackConfigLoadError(f"{field_name} must contain numeric values.") from exc


def _vector2_or_scalar(value: Any, field_name: str) -> Vector2:
    if isinstance(value, (int, float)):
        size = float(value)
        return (size, size)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TrackConfigLoadError(f"{field_name} must be a number or 2-value list.")
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise TrackConfigLoadError(f"{field_name} must contain numeric values.") from exc
