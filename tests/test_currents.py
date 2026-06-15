from __future__ import annotations

import math
from pathlib import Path

from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.arena.currents import CurrentFieldManager
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.config.schema import CurrentConfig


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


def test_vortex_outside_radius_returns_zero() -> None:
    manager = CurrentFieldManager([_vortex(clockwise=False)])

    assert manager.get_current_at((20.0, 0.0, -4.0), 0.0) == (0.0, 0.0, 0.0)


def test_vortex_inside_radius_returns_tangential_velocity() -> None:
    manager = CurrentFieldManager([_vortex(clockwise=False, vertical_speed=0.0)])

    velocity = manager.get_current_at((5.0, 0.0, -4.0), 0.0)

    assert abs(velocity[0]) < 1e-9
    assert velocity[1] > 0.0
    assert abs(velocity[2]) < 1e-9


def test_vortex_clockwise_reverses_tangential_direction() -> None:
    counter_clockwise = CurrentFieldManager([_vortex(clockwise=False, vertical_speed=0.0)])
    clockwise = CurrentFieldManager([_vortex(clockwise=True, vertical_speed=0.0)])

    ccw_velocity = counter_clockwise.get_current_at((5.0, 0.0, -4.0), 0.0)
    cw_velocity = clockwise.get_current_at((5.0, 0.0, -4.0), 0.0)

    assert math.isclose(ccw_velocity[0], -cw_velocity[0], abs_tol=1e-9)
    assert math.isclose(ccw_velocity[1], -cw_velocity[1], abs_tol=1e-9)


def test_vortex_center_is_safe_and_keeps_vertical_component() -> None:
    manager = CurrentFieldManager([_vortex(clockwise=False, vertical_speed=0.04)])

    velocity = manager.get_current_at((0.0, 0.0, -4.0), 0.0)

    assert velocity == (0.0, 0.0, 0.04)


def test_combined_currents_are_summed() -> None:
    manager = CurrentFieldManager(
        [
            CurrentConfig(type="constant", params={"velocity": [0.1, 0.0, 0.0]}),
            _vortex(clockwise=False, tangential_speed=0.2, vertical_speed=0.0, falloff="linear"),
        ]
    )

    velocity = manager.get_current_at((5.0, 0.0, -4.0), 0.0)

    assert math.isclose(velocity[0], 0.1, abs_tol=1e-9)
    assert velocity[1] > 0.0


def test_no_current_benchmark_tracks_evaluate_zero_current() -> None:
    for track_name in ("marine_race_horseshoe_bay.json", "marine_race_vertical_serpent.json"):
        config = load_track_config(TRACK_DIR / track_name)
        arena = ArenaBuilder(config).build()

        assert config.currents == []
        assert arena.current_manager.get_current_at(config.start.position, 0.0) == (0.0, 0.0, 0.0)


def test_mixed_endurance_has_nonzero_strong_current_zones() -> None:
    config = load_track_config(TRACK_DIR / "marine_race_mixed_endurance.json")
    arena = ArenaBuilder(config).build()

    speeds = [
        _norm(arena.current_manager.get_current_at(gate.position, 0.0))
        for gate in config.gates
    ]

    assert any(speed > 0.1 for speed in speeds)
    assert any(current.type == "vortex" for current in config.currents)


def _vortex(
    clockwise: bool,
    tangential_speed: float = 0.45,
    vertical_speed: float = 0.0,
    falloff: str = "gaussian",
) -> CurrentConfig:
    return CurrentConfig(
        type="vortex",
        params={
            "center": [0.0, 0.0, -4.0],
            "radius": 10.0,
            "tangential_speed": tangential_speed,
            "vertical_speed": vertical_speed,
            "falloff": falloff,
            "clockwise": clockwise,
        },
    )


def _norm(value: tuple[float, float, float]) -> float:
    return math.sqrt(value[0] ** 2 + value[1] ** 2 + value[2] ** 2)
