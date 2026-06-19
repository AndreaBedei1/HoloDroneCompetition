"""Explicit validation for marine race track configurations."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from marine_race_arena.arena.obstacle import (
    Obstacle,
    ObstacleConfigError,
    resolve_active_obstacles,
    validate_active_obstacles,
)
from marine_race_arena.config.benchmark_tasks import (
    BENCHMARK_TASK_CLEAN_GATE,
    BENCHMARK_TASK_CURRENT_GATE,
    BENCHMARK_TASK_MODES,
    BENCHMARK_TASK_MULTI_ROV,
    BENCHMARK_TASK_OBSTACLE_GATE,
    STRONG_CURRENT_MIN_SPEED_M_S,
)
from marine_race_arena.config.schema import GateConfig, TrackConfig, Vector3


class TrackValidationError(ValueError):
    """Raised when a track configuration cannot be used safely."""

    def __init__(self, errors: Iterable[str], warnings: Optional[Iterable[str]] = None):
        self.errors = list(errors)
        self.warnings = list(warnings or [])
        message = "Invalid marine race track configuration:\n" + "\n".join(
            f"- {error}" for error in self.errors
        )
        if self.warnings:
            message += "\nWarnings:\n" + "\n".join(f"- {warning}" for warning in self.warnings)
        super().__init__(message)


@dataclass
class ValidationResult:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def raise_if_needed(self) -> None:
        if self.errors:
            raise TrackValidationError(self.errors, self.warnings)


ALLOWED_RACE_FORMATS = {"ai_grand_challenge", "time_trial", "multi_rover", "drag_race"}
ALLOWED_TIMING_MODES = {"green_to_finish", "first_gate_to_last_gate"}
ALLOWED_GATE_TYPES = {"single", "double", "vertical_double", "split_s_upper", "split_s_lower"}
ALLOWED_BEACON_MODES = {"always_on", "active_when_target", "sequential_channel"}
ALLOWED_CURRENT_TYPES = {"constant", "localized_jet", "sinusoidal", "vortex"}
ALLOWED_CONTROL_MODES = {"high_level", "thrusters"}
BUILT_IN_CONTROLLERS = {
    "oracle",
    "acoustic",
    "keyboard",
    "manual",
    "manual_keyboard",
    "pygame",
    "pygame_keyboard",
    "student_template",
}


def validate_track_config(config: TrackConfig, strict: bool = True) -> ValidationResult:
    """Validate a fully parsed track config.

    Args:
        config: Parsed track configuration.
        strict: When true, warnings are kept as warnings and errors remain fatal.

    Returns:
        A ValidationResult containing errors and warnings.
    """

    result = ValidationResult()
    _validate_race(config, result)
    _validate_world(config, result)
    _validate_track_and_gates(config, result)
    _validate_beacons(config, result)
    _validate_currents(config, result)
    _validate_participants(config, result)
    _validate_referee(config, result)
    active_obstacles = _validate_obstacles(config, result)
    _validate_benchmark_task(config, active_obstacles, result)
    if strict:
        return result
    return result


def compute_declared_path_length_m(config: TrackConfig, include_start: bool = True) -> float:
    """Compute the one-pass path length used for declared-length validation."""

    gate_by_id = {gate.id: gate for gate in config.gates}
    positions: List[Vector3] = []
    if include_start:
        positions.append(config.start.position)
    for gate_id in config.track.gate_sequence:
        gate = gate_by_id.get(gate_id)
        if gate is not None:
            positions.append(gate.position)
    if len(positions) < 2:
        return 0.0
    return sum(_distance(a, b) for a, b in zip(positions, positions[1:]))


def _validate_race(config: TrackConfig, result: ValidationResult) -> None:
    race = config.race
    if not race.name:
        result.error("race.name must not be empty.")
    if race.format not in ALLOWED_RACE_FORMATS:
        result.error(f"race.format '{race.format}' is not supported.")
    if race.laps <= 0:
        result.error("race.laps must be positive.")
    if race.expected_gates_per_lap <= 0:
        result.error("race.expected_gates_per_lap must be positive.")
    if race.timing_mode not in ALLOWED_TIMING_MODES:
        result.error(f"race.timing_mode '{race.timing_mode}' is not supported.")
    if race.max_duration_s <= 0:
        result.error("race.max_duration_s must be positive.")
    if race.expected_gates_per_lap != len(config.track.gate_sequence):
        result.error(
            "race.expected_gates_per_lap must match the number of ids in track.gate_sequence."
        )


def _validate_world(config: TrackConfig, result: ValidationResult) -> None:
    bounds = config.world.bounds
    if bounds.x_min >= bounds.x_max:
        result.error("world.bounds.x_min must be less than x_max.")
    if bounds.y_min >= bounds.y_max:
        result.error("world.bounds.y_min must be less than y_max.")
    if bounds.z_min >= bounds.z_max:
        result.error("world.bounds.z_min must be less than z_max.")
    if bounds.z_max > 1.0:
        result.warn("world.bounds.z_max is above the water surface for a typical underwater map.")
    if not config.world.preferred_environment:
        result.warn("world.preferred_environment is empty; OpenWater-Hovering is recommended.")
    if not bounds.contains(config.start.position):
        result.error("start.position is outside world.bounds.")
    if config.start.position[2] < bounds.z_min:
        result.error("start.position is below z_min and unsafe.")
    if config.start.position[2] < -7.0:
        result.warn("start.position is deeper than -7.0 m; verify the map floor clearance.")


def _validate_track_and_gates(config: TrackConfig, result: ValidationResult) -> None:
    if config.track.declared_length_m <= 0:
        result.error("track.declared_length_m must be positive.")
    if config.track.length_tolerance_m < 0:
        result.error("track.length_tolerance_m must be zero or positive.")
    if config.track.gate_inner_size_m[0] <= 0 or config.track.gate_inner_size_m[1] <= 0:
        result.error("track.gate_inner_size_m dimensions must be positive.")
    if config.track.gate_bar_thickness_m <= 0:
        result.error("track.gate_bar_thickness_m must be positive.")
    if config.track.gate_depth_m <= 0:
        result.error("track.gate_depth_m must be positive.")

    gate_ids: Dict[str, GateConfig] = {}
    for gate in config.gates:
        if gate.id in gate_ids:
            result.error(f"Duplicated gate id '{gate.id}'.")
        gate_ids[gate.id] = gate
        _validate_gate(config, gate, result)

    for gate_id in config.track.gate_sequence:
        if gate_id not in gate_ids:
            result.error(f"track.gate_sequence references unknown gate '{gate_id}'.")
    if config.finish.gate_id not in gate_ids:
        result.error(f"finish.gate_id references unknown gate '{config.finish.gate_id}'.")
    elif config.track.gate_sequence and config.finish.gate_id != config.track.gate_sequence[-1]:
        result.warn("finish.gate_id is not the final id in track.gate_sequence.")

    for gate in config.gates:
        if gate.linked_gate and gate.linked_gate not in gate_ids:
            result.error(f"Gate '{gate.id}' links to unknown gate '{gate.linked_gate}'.")
        if gate.type == "split_s_upper":
            _validate_split_s_pair(gate, gate_ids.get(gate.linked_gate or ""), result)
        if gate.type == "split_s_lower":
            linked = gate_ids.get(gate.linked_gate or "")
            if linked and linked.type != "split_s_upper":
                result.error(f"Split-S lower gate '{gate.id}' must link to a split_s_upper gate.")

    computed_length = compute_declared_path_length_m(config)
    if abs(computed_length - config.track.declared_length_m) > config.track.length_tolerance_m:
        result.error(
            "track.declared_length_m does not match the computed start-to-sequence path length "
            f"({computed_length:.2f} m) within tolerance {config.track.length_tolerance_m:.2f} m."
        )


def _validate_gate(config: TrackConfig, gate: GateConfig, result: ValidationResult) -> None:
    if not gate.id:
        result.error("Gate id must not be empty.")
    if gate.type not in ALLOWED_GATE_TYPES:
        result.error(f"Gate '{gate.id}' has unsupported type '{gate.type}'.")
    if not config.world.bounds.contains(gate.position):
        result.error(f"Gate '{gate.id}' is outside world.bounds.")
    if gate.position[2] < config.world.bounds.z_min:
        result.error(f"Gate '{gate.id}' is below z_min and unsafe.")
    if gate.position[2] < -7.0:
        result.warn(f"Gate '{gate.id}' is deeper than -7.0 m; verify map floor clearance.")
    if gate.inner_size_m[0] <= 0 or gate.inner_size_m[1] <= 0:
        result.error(f"Gate '{gate.id}' inner_size_m dimensions must be positive.")
    if gate.bar_thickness_m <= 0:
        result.error(f"Gate '{gate.id}' bar_thickness_m must be positive.")
    if _norm(gate.passage_direction) <= 1e-9:
        result.error(f"Gate '{gate.id}' passage_direction must be nonzero.")
    else:
        passage_yaw = _horizontal_yaw_deg(gate.passage_direction)
        if passage_yaw is not None:
            yaw_error = abs(_angle_delta_deg(gate.rotation_rpy_deg[2], passage_yaw))
            if yaw_error > 5.0:
                result.warn(
                    f"Gate '{gate.id}' rotation_rpy_deg yaw differs from passage_direction yaw "
                    f"by {yaw_error:.1f} degrees; visual gates use passage_direction as the source of truth."
                )
    if not all(math.isfinite(value) for value in gate.rotation_rpy_deg):
        result.error(f"Gate '{gate.id}' rotation_rpy_deg contains a non-finite value.")
    if any(abs(value) > 360.0 for value in gate.rotation_rpy_deg):
        result.warn(f"Gate '{gate.id}' rotation_rpy_deg uses values outside +/-360 degrees.")


def _validate_split_s_pair(
    upper_gate: GateConfig, lower_gate: Optional[GateConfig], result: ValidationResult
) -> None:
    if lower_gate is None:
        result.error(f"Split-S upper gate '{upper_gate.id}' must link to a lower gate.")
        return
    if lower_gate.type != "split_s_lower":
        result.error(f"Split-S upper gate '{upper_gate.id}' must link to a split_s_lower gate.")
    if upper_gate.position[2] <= lower_gate.position[2]:
        result.error(
            f"Split-S upper gate '{upper_gate.id}' should be above lower gate '{lower_gate.id}'."
        )
    direction_dot = _dot(_normalize(upper_gate.passage_direction), _normalize(lower_gate.passage_direction))
    if direction_dot > 0.7:
        result.warn(
            f"Split-S gates '{upper_gate.id}' and '{lower_gate.id}' have very similar directions."
        )


def _validate_beacons(config: TrackConfig, result: ValidationResult) -> None:
    if config.beacon.mode not in ALLOWED_BEACON_MODES:
        result.error(f"beacon.mode '{config.beacon.mode}' is not supported.")
    seen = set()
    for gate in config.gates:
        beacon = gate.beacon
        if beacon is None or not beacon.enabled:
            result.warn(f"Gate '{gate.id}' has no enabled acoustic beacon.")
            continue
        if beacon.mode not in ALLOWED_BEACON_MODES:
            result.error(f"Gate '{gate.id}' beacon mode '{beacon.mode}' is not supported.")
        if not beacon.id:
            result.error(f"Gate '{gate.id}' has an enabled beacon without an id.")
            continue
        if beacon.id in seen:
            result.error(f"Duplicated beacon id '{beacon.id}'.")
        seen.add(beacon.id)
        if beacon.range_m <= 0:
            result.error(f"Beacon '{beacon.id}' range_m must be positive.")
        if beacon.noise_std < 0:
            result.error(f"Beacon '{beacon.id}' noise_std must be zero or positive.")
        if not 0.0 <= beacon.dropout_probability <= 1.0:
            result.error(f"Beacon '{beacon.id}' dropout_probability must be in [0, 1].")
        if beacon.update_rate_hz <= 0:
            result.error(f"Beacon '{beacon.id}' update_rate_hz must be positive.")


def _validate_currents(config: TrackConfig, result: ValidationResult) -> None:
    for index, current in enumerate(config.currents):
        if current.type not in ALLOWED_CURRENT_TYPES:
            result.error(f"Current #{index} has unsupported type '{current.type}'.")
        if current.type == "constant":
            _require_vector(current.params, "velocity", f"Current #{index}", result)
        if current.type == "localized_jet":
            _require_vector(current.params, "center", f"Current #{index}", result)
            _require_vector(current.params, "velocity", f"Current #{index}", result)
            if float(current.params.get("radius", 0.0)) <= 0:
                result.error(f"Current #{index} localized_jet.radius must be positive.")
        if current.type == "sinusoidal":
            axis = current.params.get("axis")
            if axis not in {"x", "y", "z"}:
                result.error(f"Current #{index} sinusoidal.axis must be one of x, y, z.")
            frequency = current.params.get("frequency_hz", current.params.get("frequency", 0.0))
            if float(frequency) < 0:
                result.error(f"Current #{index} sinusoidal.frequency_hz must be zero or positive.")
        if current.type == "vortex":
            _require_vector(current.params, "center", f"Current #{index}", result)
            if float(current.params.get("radius", 0.0)) <= 0:
                result.error(f"Current #{index} vortex.radius must be positive.")
            if float(current.params.get("tangential_speed", 0.0)) < 0:
                result.error(f"Current #{index} vortex.tangential_speed must be zero or positive.")
            falloff = current.params.get("falloff", "gaussian")
            if falloff not in {"linear", "gaussian"}:
                result.error(f"Current #{index} vortex.falloff must be linear or gaussian.")


def _validate_participants(config: TrackConfig, result: ValidationResult) -> None:
    seen = set()
    for participant in config.participants:
        if not participant.id:
            result.error("Participant id must not be empty.")
        if participant.id in seen:
            result.error(f"Duplicated participant id '{participant.id}'.")
        seen.add(participant.id)
        if participant.control_mode not in ALLOWED_CONTROL_MODES:
            result.error(
                f"Participant '{participant.id}' has unsupported control_mode "
                f"'{participant.control_mode}'."
            )
        if participant.spawn:
            position = participant.spawn.get("position")
            if position is not None:
                try:
                    spawn_position = tuple(float(value) for value in position)
                except (TypeError, ValueError):
                    result.error(f"Participant '{participant.id}' spawn.position is not numeric.")
                else:
                    if len(spawn_position) != 3:
                        result.error(f"Participant '{participant.id}' spawn.position must have 3 values.")
                    elif not config.world.bounds.contains(spawn_position):  # type: ignore[arg-type]
                        result.error(f"Participant '{participant.id}' spawn.position is outside bounds.")
        if not _controller_reference_looks_valid(participant.controller):
            result.warn(
                f"Participant '{participant.id}' controller '{participant.controller}' was not found locally. "
                "This is allowed for external modules but must resolve at race time."
            )


def _validate_referee(config: TrackConfig, result: ValidationResult) -> None:
    penalties = config.referee.penalties
    for key in (
        "minor_collision_s",
        "gate_collision_s",
        "out_of_bounds_s",
        "stuck_s",
        "wrong_direction_s",
    ):
        if key in penalties and float(penalties[key]) < 0:
            result.error(f"referee.penalties.{key} must be zero or positive.")
    stuck_timeout = config.referee.gate_validation.get("stuck_timeout_s")
    if stuck_timeout is not None and float(stuck_timeout) <= 0:
        result.error("referee.gate_validation.stuck_timeout_s must be positive.")
    collision_cooldown = config.referee.gate_validation.get("collision_penalty_cooldown_s")
    if collision_cooldown is not None and float(collision_cooldown) < 0:
        result.error("referee.gate_validation.collision_penalty_cooldown_s must be zero or positive.")
    out_of_bounds_cooldown = config.referee.gate_validation.get("out_of_bounds_penalty_cooldown_s")
    if out_of_bounds_cooldown is not None and float(out_of_bounds_cooldown) < 0:
        result.error("referee.gate_validation.out_of_bounds_penalty_cooldown_s must be zero or positive.")
    clearance_margin = config.referee.gate_validation.get("vehicle_clearance_margin_m")
    if clearance_margin is not None and float(clearance_margin) < 0:
        result.error("referee.gate_validation.vehicle_clearance_margin_m must be zero or positive.")


def _validate_obstacles(config: TrackConfig, result: ValidationResult) -> List[Obstacle]:
    try:
        active_obstacles = resolve_active_obstacles(config)
    except ObstacleConfigError as exc:
        for error in exc.errors:
            result.error(error)
        active_obstacles = []
    for error in validate_active_obstacles(config, active_obstacles):
        result.error(error)
    return active_obstacles


def _validate_benchmark_task(
    config: TrackConfig,
    active_obstacles: List[Obstacle],
    result: ValidationResult,
) -> None:
    mode = config.benchmark_task.mode
    if mode is None:
        return
    if mode not in BENCHMARK_TASK_MODES:
        result.error(
            f"benchmark_task.mode '{mode}' is not supported. "
            f"Allowed modes: {', '.join(BENCHMARK_TASK_MODES)}."
        )
        return

    if mode == BENCHMARK_TASK_CLEAN_GATE:
        _require_single_rov_task(config, mode, result)
        if active_obstacles:
            result.error("benchmark_task clean_gate must not activate obstacles.")
        if config.currents:
            result.error("benchmark_task clean_gate must not configure currents.")
        return

    if mode == BENCHMARK_TASK_OBSTACLE_GATE:
        _require_single_rov_task(config, mode, result)
        if config.currents:
            result.error("benchmark_task obstacle_gate must not configure currents.")
        if not active_obstacles:
            result.error("benchmark_task obstacle_gate requires at least one active static obstacle.")
        return

    if mode == BENCHMARK_TASK_CURRENT_GATE:
        _require_single_rov_task(config, mode, result)
        if not config.currents:
            result.error("benchmark_task current_gate requires at least one marine current.")
        elif _max_configured_current_speed(config) < STRONG_CURRENT_MIN_SPEED_M_S:
            result.error(
                "benchmark_task current_gate requires at least one configured current with "
                f"speed >= {STRONG_CURRENT_MIN_SPEED_M_S:.2f} m/s."
            )
        return

    if mode == BENCHMARK_TASK_MULTI_ROV:
        if len(config.participants) < 2:
            result.error("benchmark_task multi_rov requires at least two participants.")
        result.warn(
            "benchmark_task multi_rov is parsed and validated, but execution still uses the "
            "current shared-course referee model."
        )


def _require_single_rov_task(config: TrackConfig, mode: str, result: ValidationResult) -> None:
    if len(config.participants) != 1:
        result.error(f"benchmark_task {mode} requires exactly one participant.")
    if not config.gates:
        result.error(f"benchmark_task {mode} requires at least one gate.")


def _max_configured_current_speed(config: TrackConfig) -> float:
    speeds = [_configured_current_speed(current.params, current.type) for current in config.currents]
    return max(speeds, default=0.0)


def _configured_current_speed(params: Mapping[str, Any], current_type: str) -> float:
    if current_type == "constant":
        return _vector_speed(params.get("velocity"))
    if current_type == "localized_jet":
        return _vector_speed(params.get("velocity"))
    if current_type == "sinusoidal":
        base_speed = _vector_speed(params.get("base_velocity"))
        return base_speed + abs(_safe_float(params.get("amplitude", 0.0), 0.0))
    if current_type == "vortex":
        tangential = _safe_float(params.get("tangential_speed", 0.0), 0.0)
        vertical = _safe_float(params.get("vertical_speed", 0.0), 0.0)
        return math.hypot(tangential, vertical)
    return 0.0


def _vector_speed(value: Any) -> float:
    vector = _parse_vector3(value)
    if vector is None:
        return 0.0
    return _norm(vector)


def _parse_vector3(value: Any) -> Optional[Vector3]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _controller_reference_looks_valid(reference: str) -> bool:
    if not reference:
        return False
    if reference in BUILT_IN_CONTROLLERS:
        return True
    if reference.endswith(".py"):
        return Path(reference).exists()
    module_name = reference
    if ":" in module_name:
        module_name = module_name.split(":", 1)[0]
    elif "." in module_name:
        parts = module_name.split(".")
        if len(parts) > 1 and parts[-1][:1].isupper():
            module_name = ".".join(parts[:-1])
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _require_vector(params: Dict[str, object], key: str, owner: str, result: ValidationResult) -> None:
    value = params.get(key)
    if not isinstance(value, list) or len(value) != 3:
        result.error(f"{owner} requires a 3-value '{key}' vector.")
        return
    try:
        [float(component) for component in value]
    except (TypeError, ValueError):
        result.error(f"{owner}.{key} must contain numeric values.")


def _distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vector3) -> Vector3:
    length = _norm(vector)
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _horizontal_yaw_deg(vector: Vector3) -> Optional[float]:
    horizontal_norm = math.hypot(vector[0], vector[1])
    if horizontal_norm <= 1e-9:
        return None
    return math.degrees(math.atan2(vector[1], vector[0]))


def _angle_delta_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0
