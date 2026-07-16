from __future__ import annotations

from pathlib import Path

import pytest

from marine_race_arena.arena.obstacle import resolve_active_obstacles
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.config.validation import validate_track_config


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"
OFFICIAL_TRACKS = (
    "marine_race_horseshoe_bay.json",
    "marine_race_vertical_serpent.json",
    "marine_race_mixed_endurance.json",
)


@pytest.mark.parametrize("track_name", OFFICIAL_TRACKS)
@pytest.mark.parametrize("profile", ("medium", "strong"))
def test_every_official_track_supports_named_physical_current_profiles(
    track_name: str, profile: str
) -> None:
    path = TRACK_DIR / track_name
    clean = load_track_config(
        path,
        benchmark_task="clean_gate",
        current_profile="none",
        obstacles="none",
    )
    configured = load_track_config(
        path,
        benchmark_task="current_gate",
        current_profile=profile,
        obstacles="none",
    )

    assert len(configured.currents) == 5
    assert configured.selected_current_profile == profile
    assert configured.gates == clean.gates
    assert configured.referee == clean.referee
    assert validate_track_config(configured).errors == []


@pytest.mark.parametrize("track_name", OFFICIAL_TRACKS)
def test_every_official_track_supports_seeded_static_obstacles(
    track_name: str,
) -> None:
    path = TRACK_DIR / track_name
    clean = load_track_config(
        path,
        benchmark_task="clean_gate",
        current_profile="none",
        obstacles="none",
    )
    configured = load_track_config(
        path,
        benchmark_task="obstacle_gate",
        current_profile="none",
        obstacles="random",
        obstacle_density="medium",
        obstacle_physics="static",
        seed=0,
    )
    obstacles = resolve_active_obstacles(configured)

    assert obstacles
    assert all(obstacle.type == "box" for obstacle in obstacles)
    assert configured.gates == clean.gates
    assert configured.referee == clean.referee
    assert validate_track_config(configured).errors == []
