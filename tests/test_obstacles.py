from __future__ import annotations

import copy
import json
from pathlib import Path

from marine_race_arena.adapters.fallback_adapter import FallbackRaceAdapter
from marine_race_arena.adapters.holoocean_adapter import HoloOceanRaceAdapter
from marine_race_arena.adapters.visual_spawner import HoloOceanVisualSpawner
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.arena.obstacle import resolve_active_obstacles
from marine_race_arena.config.benchmark_tasks import (
    BENCHMARK_TASK_CLEAN_GATE,
    BENCHMARK_TASK_CURRENT_GATE,
    BENCHMARK_TASK_OBSTACLE_GATE,
)
from marine_race_arena.config.loader import parse_track_config, with_obstacle_options
from marine_race_arena.config.validation import validate_track_config
from marine_race_arena.participants.participant import RaceParticipant
from marine_race_arena.referee.race_state import ParticipantStatus
from marine_race_arena.referee.referee import Referee


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


class FakeSpawnPropEnv:
    def __init__(self) -> None:
        self.calls = []

    def spawn_prop(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_obstacle_config_validation_rejects_unsupported_shape() -> None:
    raw = _raw_horseshoe()
    raw["benchmark_task"] = BENCHMARK_TASK_OBSTACLE_GATE
    obstacle = _static_obstacle()
    obstacle["type"] = "cylinder"
    raw["obstacles"] = [obstacle]

    result = validate_track_config(parse_track_config(raw))

    assert any("Only box obstacles are supported" in error for error in result.errors)


def test_fixed_obstacle_resolution_is_deterministic() -> None:
    raw = _raw_horseshoe()
    raw["benchmark_task"] = BENCHMARK_TASK_OBSTACLE_GATE
    raw["obstacles"] = [_static_obstacle()]
    config = parse_track_config(raw)

    first = [obstacle.to_dict() for obstacle in resolve_active_obstacles(config)]
    second = [obstacle.to_dict() for obstacle in resolve_active_obstacles(config)]

    assert first == second
    assert validate_track_config(config).errors == []


def test_random_obstacle_generation_is_deterministic_from_seed() -> None:
    raw = _raw_horseshoe()
    raw["benchmark_task"] = BENCHMARK_TASK_OBSTACLE_GATE
    config = with_obstacle_options(
        parse_track_config(raw),
        mode="random",
        density="medium",
        seed=42,
    )

    first = [obstacle.to_dict() for obstacle in resolve_active_obstacles(config)]
    second = [obstacle.to_dict() for obstacle in resolve_active_obstacles(config)]
    different_seed = with_obstacle_options(config, seed=7)
    different = [obstacle.to_dict() for obstacle in resolve_active_obstacles(different_seed)]

    assert first == second
    assert first != different
    assert validate_track_config(config).errors == []


def test_obstacles_none_produces_no_active_obstacles() -> None:
    raw = _raw_horseshoe()
    raw["benchmark_task"] = BENCHMARK_TASK_CLEAN_GATE
    raw["obstacles"] = [_static_obstacle()]
    config = with_obstacle_options(parse_track_config(raw), mode="none")

    assert resolve_active_obstacles(config) == []
    assert validate_track_config(config).errors == []


def test_obstacle_gate_requires_active_obstacles() -> None:
    raw = _raw_horseshoe()
    raw["benchmark_task"] = BENCHMARK_TASK_OBSTACLE_GATE
    raw["obstacles"] = []
    config = with_obstacle_options(parse_track_config(raw), mode="none")

    result = validate_track_config(config)

    assert any("obstacle_gate requires at least one active static obstacle" in error for error in result.errors)


def test_current_gate_can_ignore_configured_obstacles() -> None:
    raw = _raw_horseshoe()
    raw["benchmark_task"] = BENCHMARK_TASK_CURRENT_GATE
    raw["currents"] = [{"type": "constant", "velocity": [0.8, 0.0, 0.0]}]
    raw["obstacles"] = [_static_obstacle()]
    config = with_obstacle_options(parse_track_config(raw), mode="none")

    assert resolve_active_obstacles(config) == []
    assert validate_track_config(config).errors == []


def test_fallback_obstacle_collision_adds_penalty_and_does_not_dnf() -> None:
    raw = _raw_horseshoe()
    raw["benchmark_task"] = BENCHMARK_TASK_OBSTACLE_GATE
    raw["obstacles"] = [_static_obstacle()]
    config = parse_track_config(raw)
    arena = ArenaBuilder(config).build()
    obstacle = arena.obstacles[0]
    participant_config = copy.deepcopy(config.participants[0])
    participant = RaceParticipant(
        config=participant_config,
        controller=object(),
        position=(obstacle.position[0] - 1.0, obstacle.position[1], obstacle.position[2]),
        rotation_rpy_deg=participant_config.spawn["rotation_rpy_deg"],
    )
    adapter = FallbackRaceAdapter(config, arena)
    adapter.initialize()
    adapter.spawn_participants({participant.id: participant})
    previous_position = participant.position
    current_position = (obstacle.position[0] + 1.0, obstacle.position[1], obstacle.position[2])
    obstacle_collisions = adapter.get_obstacle_collision_events(
        participant.id,
        previous_position=previous_position,
        current_position=current_position,
    )
    referee = Referee(config, arena.gate_map, arena.bounds)
    referee.register_participants([participant.id])
    referee.start_race(0.0)

    events = referee.update(
        participant.id,
        previous_position,
        current_position,
        1.0,
        obstacle_collisions=obstacle_collisions,
    )
    state = referee.states[participant.id]

    assert any(event["event"] == "obstacle_collision" for event in events)
    assert state.status == ParticipantStatus.RUNNING
    assert state.obstacle_collision_events == 1
    assert state.penalties_s == obstacle.penalty_s


def test_holoocean_obstacle_spawning_uses_physical_box_props() -> None:
    raw = _raw_horseshoe()
    raw["benchmark_task"] = BENCHMARK_TASK_OBSTACLE_GATE
    raw["obstacles"] = [_static_obstacle()]
    config = parse_track_config(raw)
    arena = ArenaBuilder(config).build()
    env = FakeSpawnPropEnv()
    adapter = HoloOceanRaceAdapter(config, arena)
    adapter.env = env
    adapter.visual_spawner = HoloOceanVisualSpawner(env)

    adapter.spawn_obstacles(arena.obstacles)

    assert len(env.calls) == 1
    args, kwargs = env.calls[0]
    assert args == ("box",)
    assert kwargs["tag"] == "OBS01"
    assert kwargs["sim_physics"] is True
    assert kwargs["scale"] == [0.7, 0.7, 0.7]


def _raw_horseshoe() -> dict:
    return json.loads((TRACK_DIR / "marine_race_horseshoe_bay.json").read_text(encoding="utf-8"))


def _static_obstacle() -> dict:
    return {
        "id": "OBS01",
        "type": "box",
        "position": [-28.2, -6.25, -4.05],
        "size": [0.7, 0.7, 0.7],
        "rotation_rpy_deg": [0.0, 0.0, 33.7],
        "collision": True,
        "penalty_s": 5.0,
        "between_gates": ["G01", "G02"],
    }
