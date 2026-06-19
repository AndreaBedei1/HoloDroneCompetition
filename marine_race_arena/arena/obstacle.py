"""Static obstacle definitions, generation, validation helpers, and collisions."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from marine_race_arena.config.schema import GateConfig, TrackConfig, Vector3

OBSTACLE_MODE_NONE = "none"
OBSTACLE_MODE_FIXED = "fixed"
OBSTACLE_MODE_RANDOM = "random"
OBSTACLE_MODES = (OBSTACLE_MODE_NONE, OBSTACLE_MODE_FIXED, OBSTACLE_MODE_RANDOM)

OBSTACLE_DENSITY_LOW = "low"
OBSTACLE_DENSITY_MEDIUM = "medium"
OBSTACLE_DENSITY_HIGH = "high"
OBSTACLE_DENSITIES = (OBSTACLE_DENSITY_LOW, OBSTACLE_DENSITY_MEDIUM, OBSTACLE_DENSITY_HIGH)

DEFAULT_OBSTACLE_SIZE_M = (0.7, 0.7, 0.7)
DEFAULT_OBSTACLE_PENALTY_S = 5.0
DEFAULT_VEHICLE_COLLISION_RADIUS_M = 0.35


class ObstacleConfigError(ValueError):
    """Raised when active obstacle configuration cannot be resolved."""

    def __init__(self, errors: Iterable[str]):
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class Obstacle:
    id: str
    type: str
    position: Vector3
    size: Vector3
    rotation_rpy_deg: Vector3
    collision: bool
    penalty_s: float
    between_gates: tuple[str, str]
    source: str = "fixed"

    @property
    def bounding_radius_m(self) -> float:
        return 0.5 * math.sqrt(self.size[0] ** 2 + self.size[1] ** 2 + self.size[2] ** 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "position": list(self.position),
            "size": list(self.size),
            "rotation_rpy_deg": list(self.rotation_rpy_deg),
            "collision": self.collision,
            "penalty_s": self.penalty_s,
            "between_gates": list(self.between_gates),
            "source": self.source,
        }


def effective_obstacle_mode(config: TrackConfig) -> str:
    configured = config.obstacle_generation.mode
    if configured:
        return configured.strip().lower()
    if config.obstacles:
        return OBSTACLE_MODE_FIXED
    return OBSTACLE_MODE_NONE


def resolve_active_obstacles(config: TrackConfig) -> list[Obstacle]:
    mode = effective_obstacle_mode(config)
    if mode == OBSTACLE_MODE_NONE:
        return []
    if mode == OBSTACLE_MODE_FIXED:
        return _parse_fixed_obstacles(config.obstacles)
    if mode == OBSTACLE_MODE_RANDOM:
        return generate_random_obstacles(config)
    raise ObstacleConfigError([f"obstacle_generation.mode '{mode}' is not supported."])


def generate_random_obstacles(config: TrackConfig) -> list[Obstacle]:
    intervals = _gate_intervals(config)
    if not intervals:
        return []
    density = config.obstacle_generation.density.strip().lower()
    fraction = {
        OBSTACLE_DENSITY_LOW: 0.25,
        OBSTACLE_DENSITY_MEDIUM: 0.5,
        OBSTACLE_DENSITY_HIGH: 0.75,
    }.get(density, 0.5)
    count = max(1, min(len(intervals), int(math.ceil(len(intervals) * fraction))))
    rng = random.Random(config.obstacle_generation.seed if config.obstacle_generation.seed is not None else 0)
    interval_indices = list(range(len(intervals)))
    rng.shuffle(interval_indices)
    selected = sorted(interval_indices[:count])

    obstacles: list[Obstacle] = []
    size = _size_for_density(density)
    min_clearance = max(0.0, config.obstacle_generation.min_clearance_m)
    for obstacle_index, interval_index in enumerate(selected, start=1):
        left_gate, right_gate = intervals[interval_index]
        position, yaw_deg = _generated_position_between_gates(
            left_gate,
            right_gate,
            size=size,
            min_clearance_m=min_clearance,
            rng=rng,
        )
        obstacles.append(
            Obstacle(
                id=f"OBS_R{obstacle_index:02d}",
                type="box",
                position=position,
                size=size,
                rotation_rpy_deg=(0.0, 0.0, yaw_deg),
                collision=True,
                penalty_s=DEFAULT_OBSTACLE_PENALTY_S,
                between_gates=(left_gate.id, right_gate.id),
                source="random",
            )
        )
    return obstacles


def validate_active_obstacles(config: TrackConfig, obstacles: Sequence[Obstacle]) -> list[str]:
    errors: list[str] = []
    mode = effective_obstacle_mode(config)
    if mode not in OBSTACLE_MODES:
        errors.append(f"obstacle_generation.mode '{mode}' is not supported.")
    density = config.obstacle_generation.density.strip().lower()
    if density not in OBSTACLE_DENSITIES:
        errors.append(f"obstacle_generation.density '{config.obstacle_generation.density}' is not supported.")
    if config.obstacle_generation.min_clearance_m < 0.0:
        errors.append("obstacle_generation.min_clearance_m must be zero or positive.")

    seen_ids: set[str] = set()
    gate_sequence_index = {gate_id: index for index, gate_id in enumerate(config.track.gate_sequence)}
    gate_by_id = {gate.id: gate for gate in config.gates}
    for obstacle in obstacles:
        owner = f"Obstacle '{obstacle.id}'"
        if not obstacle.id:
            errors.append("Obstacle id must not be empty.")
        elif obstacle.id in seen_ids:
            errors.append(f"Duplicated obstacle id '{obstacle.id}'.")
        seen_ids.add(obstacle.id)

        if obstacle.type != "box":
            errors.append(f"{owner} has unsupported type '{obstacle.type}'. Only box obstacles are supported.")
        if any(component <= 0.0 for component in obstacle.size):
            errors.append(f"{owner}.size dimensions must be positive.")
        if obstacle.penalty_s < 0.0:
            errors.append(f"{owner}.penalty_s must be zero or positive.")
        if not config.world.bounds.contains(obstacle.position):
            errors.append(f"{owner}.position is outside world.bounds.")
        left_id, right_id = obstacle.between_gates
        if left_id not in gate_sequence_index or right_id not in gate_sequence_index:
            errors.append(f"{owner}.between_gates must reference ids in track.gate_sequence.")
            continue
        if abs(gate_sequence_index[left_id] - gate_sequence_index[right_id]) != 1:
            errors.append(f"{owner}.between_gates must reference adjacent gate ids.")
            continue
        left_gate = gate_by_id[left_id]
        right_gate = gate_by_id[right_id]
        _validate_obstacle_clearance(config, obstacle, left_gate, right_gate, owner, errors)
    return errors


def obstacle_collision_events(
    obstacles: Sequence[Obstacle],
    previous_position: Vector3,
    current_position: Vector3,
    vehicle_radius_m: float = DEFAULT_VEHICLE_COLLISION_RADIUS_M,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for obstacle in obstacles:
        if not obstacle.collision:
            continue
        radius = obstacle.bounding_radius_m + max(0.0, vehicle_radius_m)
        if _distance_point_to_segment(obstacle.position, previous_position, current_position) <= radius:
            events.append(
                {
                    "obstacle_id": obstacle.id,
                    "obstacle_type": obstacle.type,
                    "penalty_s": obstacle.penalty_s,
                    "position": current_position,
                }
            )
    return events


def _parse_fixed_obstacles(raw_obstacles: Sequence[Mapping[str, Any]]) -> list[Obstacle]:
    obstacles: list[Obstacle] = []
    errors: list[str] = []
    for index, raw in enumerate(raw_obstacles):
        owner = f"Obstacle #{index}"
        if not isinstance(raw, Mapping):
            errors.append(f"{owner} must be an object.")
            continue
        try:
            obstacle = Obstacle(
                id=str(_required(raw, "id", owner)),
                type=str(_required(raw, "type", owner)),
                position=_vector3(_required(raw, "position", owner), f"{owner}.position"),
                size=_vector3(_required(raw, "size", owner), f"{owner}.size"),
                rotation_rpy_deg=_vector3(
                    _required(raw, "rotation_rpy_deg", owner), f"{owner}.rotation_rpy_deg"
                ),
                collision=bool(_required(raw, "collision", owner)),
                penalty_s=float(_required(raw, "penalty_s", owner)),
                between_gates=_between_gates(_required(raw, "between_gates", owner), owner),
                source="fixed",
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        obstacles.append(obstacle)
    if errors:
        raise ObstacleConfigError(errors)
    return obstacles


def _required(raw: Mapping[str, Any], key: str, owner: str) -> Any:
    if key not in raw:
        raise ValueError(f"{owner} requires '{key}'.")
    return raw[key]


def _vector3(value: Any, field_name: str) -> Vector3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 numeric values.")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain numeric values.") from exc


def _between_gates(value: Any, owner: str) -> tuple[str, str]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{owner}.between_gates must contain two gate ids.")
    return (str(value[0]), str(value[1]))


def _gate_intervals(config: TrackConfig) -> list[tuple[GateConfig, GateConfig]]:
    gate_by_id = {gate.id: gate for gate in config.gates}
    intervals: list[tuple[GateConfig, GateConfig]] = []
    for left_id, right_id in zip(config.track.gate_sequence, config.track.gate_sequence[1:]):
        left_gate = gate_by_id.get(left_id)
        right_gate = gate_by_id.get(right_id)
        if left_gate is not None and right_gate is not None:
            intervals.append((left_gate, right_gate))
    return intervals


def _size_for_density(density: str) -> Vector3:
    if density == OBSTACLE_DENSITY_LOW:
        return (0.55, 0.55, 0.55)
    if density == OBSTACLE_DENSITY_HIGH:
        return (0.9, 0.9, 0.9)
    return (0.7, 0.7, 0.7)


def _generated_position_between_gates(
    left_gate: GateConfig,
    right_gate: GateConfig,
    size: Vector3,
    min_clearance_m: float,
    rng: random.Random,
) -> tuple[Vector3, float]:
    start = left_gate.position
    end = right_gate.position
    direction = _sub(end, start)
    horizontal_length = math.hypot(direction[0], direction[1])
    if horizontal_length <= 1e-9:
        unit_direction = (1.0, 0.0, 0.0)
        perpendicular = (0.0, 1.0, 0.0)
    else:
        unit_direction = (direction[0] / horizontal_length, direction[1] / horizontal_length, 0.0)
        perpendicular = (-unit_direction[1], unit_direction[0], 0.0)
    t = rng.uniform(0.35, 0.65)
    side = -1.0 if rng.random() < 0.5 else 1.0
    radius = 0.5 * math.sqrt(size[0] ** 2 + size[1] ** 2 + size[2] ** 2)
    offset_m = min_clearance_m + radius + rng.uniform(0.35, 0.9)
    base = _lerp(start, end, t)
    position = _add(base, _scale(perpendicular, side * offset_m))
    yaw_deg = math.degrees(math.atan2(unit_direction[1], unit_direction[0]))
    return (
        (round(position[0], 3), round(position[1], 3), round(position[2], 3)),
        round(yaw_deg, 3),
    )


def _validate_obstacle_clearance(
    config: TrackConfig,
    obstacle: Obstacle,
    left_gate: GateConfig,
    right_gate: GateConfig,
    owner: str,
    errors: list[str],
) -> None:
    min_clearance = max(0.0, config.obstacle_generation.min_clearance_m)
    radius = obstacle.bounding_radius_m
    projection = _segment_projection_t(obstacle.position, left_gate.position, right_gate.position)
    if projection < 0.0 or projection > 1.0:
        errors.append(f"{owner}.position must project between its between_gates pair.")
    for label, point in (
        ("left gate aperture", left_gate.position),
        ("right gate aperture", right_gate.position),
        ("start spawn", config.start.position),
    ):
        if _distance(obstacle.position, point) < radius + min_clearance:
            errors.append(f"{owner} is too close to {label}.")
    try:
        finish_gate = config.gate_by_id(config.finish.gate_id)
    except KeyError:
        finish_gate = None
    if finish_gate is not None and _distance(obstacle.position, finish_gate.position) < radius + min_clearance:
        errors.append(f"{owner} is too close to finish gate aperture.")
    centerline_clearance = _distance_point_to_segment(
        obstacle.position, left_gate.position, right_gate.position
    )
    if centerline_clearance <= radius + min_clearance:
        errors.append(f"{owner} is too close to the valid path centerline between gates.")


def _distance_point_to_segment(point: Vector3, start: Vector3, end: Vector3) -> float:
    t = _segment_projection_t(point, start, end)
    t = max(0.0, min(1.0, t))
    closest = _lerp(start, end, t)
    return _distance(point, closest)


def _segment_projection_t(point: Vector3, start: Vector3, end: Vector3) -> float:
    segment = _sub(end, start)
    length_sq = _dot(segment, segment)
    if length_sq <= 1e-12:
        return 0.0
    return _dot(_sub(point, start), segment) / length_sq


def _lerp(a: Vector3, b: Vector3, t: float) -> Vector3:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
