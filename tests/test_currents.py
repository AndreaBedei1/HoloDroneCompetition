from __future__ import annotations

import math
from pathlib import Path

import pytest

from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.arena.currents import CurrentFieldManager
from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE, BENCHMARK_TASK_CURRENT_GATE
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.config.schema import CurrentConfig
from marine_race_arena.config.validation import validate_track_config


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"
OFFICIAL_TRACKS = (
    "marine_race_horseshoe_bay.json",
    "marine_race_vertical_serpent.json",
    "marine_race_mixed_endurance.json",
)


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


def test_all_official_tracks_validate_clean_gate_with_current_profile_none() -> None:
    for track_name in OFFICIAL_TRACKS:
        config = load_track_config(
            TRACK_DIR / track_name,
            benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
            current_profile="none",
        )

        result = validate_track_config(config)

        assert result.errors == []
        assert config.selected_current_profile == "none"
        assert config.currents == []


def test_all_official_tracks_validate_current_gate_with_medium_or_strong_currents() -> None:
    for track_name in OFFICIAL_TRACKS:
        for profile in ("medium", "strong"):
            config = load_track_config(
                TRACK_DIR / track_name,
                benchmark_task=BENCHMARK_TASK_CURRENT_GATE,
                current_profile=profile,
            )

            result = validate_track_config(config)

            assert result.errors == []
            assert config.selected_current_profile == profile
            assert len(config.currents) == 5
            assert [current.type for current in config.currents] == [
                "constant",
                "localized_jet",
                "localized_jet",
                "vortex",
                "sinusoidal",
            ]


def test_current_profile_centers_are_inside_track_bounds() -> None:
    for track_name in OFFICIAL_TRACKS:
        for profile in ("medium", "strong"):
            config = load_track_config(
                TRACK_DIR / track_name,
                benchmark_task=BENCHMARK_TASK_CURRENT_GATE,
                current_profile=profile,
            )

            for current in config.currents:
                center = current.params.get("center")
                if center is not None:
                    assert config.world.bounds.contains(tuple(center))


def test_clean_gate_rejects_active_current_profile() -> None:
    config = load_track_config(
        TRACK_DIR / "marine_race_horseshoe_bay.json",
        debug=True,
        benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
        current_profile="medium",
    )

    result = validate_track_config(config)

    assert any("clean_gate must not configure currents" in error for error in result.errors)


def test_current_gate_rejects_current_profile_none() -> None:
    config = load_track_config(
        TRACK_DIR / "marine_race_horseshoe_bay.json",
        debug=True,
        benchmark_task=BENCHMARK_TASK_CURRENT_GATE,
        current_profile="none",
    )

    result = validate_track_config(config)

    assert any("current_gate requires at least one marine current" in error for error in result.errors)


def test_mixed_endurance_current_profile_strong_matches_existing_intensities() -> None:
    default_config = load_track_config(TRACK_DIR / "marine_race_mixed_endurance.json")
    strong_config = load_track_config(
        TRACK_DIR / "marine_race_mixed_endurance.json",
        benchmark_task=BENCHMARK_TASK_CURRENT_GATE,
        current_profile="strong",
    )

    assert _current_intensity_signature(strong_config.currents) == _current_intensity_signature(
        default_config.currents
    )


def test_current_profile_medium_is_half_intensity_of_strong() -> None:
    for track_name in OFFICIAL_TRACKS:
        strong_config = load_track_config(
            TRACK_DIR / track_name,
            benchmark_task=BENCHMARK_TASK_CURRENT_GATE,
            current_profile="strong",
        )
        medium_config = load_track_config(
            TRACK_DIR / track_name,
            benchmark_task=BENCHMARK_TASK_CURRENT_GATE,
            current_profile="medium",
        )

        assert len(medium_config.currents) == len(strong_config.currents)
        for medium, strong in zip(medium_config.currents, strong_config.currents):
            assert medium.type == strong.type
            _assert_medium_params(medium.params, strong.params)


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


def _current_signature(currents: list[CurrentConfig]) -> list[tuple[str, dict]]:
    return [(current.type, current.params) for current in currents]


def _current_intensity_signature(currents: list[CurrentConfig]) -> list[tuple]:
    signature = []
    for current in currents:
        params = current.params
        if current.type in {"constant", "localized_jet"}:
            signature.append((current.type, tuple(params["velocity"])))
        elif current.type == "vortex":
            signature.append((current.type, params["tangential_speed"], params["vertical_speed"]))
        elif current.type == "sinusoidal":
            signature.append((current.type, params["amplitude"]))
        else:
            signature.append((current.type,))
    return signature


def _assert_medium_params(medium: dict, strong: dict) -> None:
    scaled_keys = {"velocity", "base_velocity"}
    scaled_scalar_keys = {"tangential_speed", "vertical_speed", "amplitude"}
    assert medium.keys() == strong.keys()
    for key, strong_value in strong.items():
        medium_value = medium[key]
        if key in scaled_keys:
            assert medium_value == pytest.approx([float(value) * 0.5 for value in strong_value])
        elif key in scaled_scalar_keys:
            assert float(medium_value) == pytest.approx(float(strong_value) * 0.5)
        else:
            assert medium_value == strong_value
