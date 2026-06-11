from __future__ import annotations

from pathlib import Path

from marine_race_arena.adapters.fallback_adapter import FallbackRaceAdapter
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.participants.participant import RaceParticipant


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


def _adapter():
    config = load_track_config(TRACK_DIR / "abu_dhabi_marine_easy.json")
    arena = ArenaBuilder(config).build()
    participant_config = config.participants[0]
    participant = RaceParticipant(
        config=participant_config,
        controller=object(),
        position=tuple(participant_config.spawn["position"]),
        rotation_rpy_deg=tuple(participant_config.spawn["rotation_rpy_deg"]),
    )
    adapter = FallbackRaceAdapter(config, arena)
    adapter.initialize()
    adapter.spawn_participants({participant.id: participant})
    return adapter, participant


def test_high_level_command_mapping_is_clamped() -> None:
    adapter, participant = _adapter()
    thrusters = adapter.command_to_bluerov2_thrusters(
        participant.id,
        {"surge": 5.0, "sway": -5.0, "heave": 5.0, "yaw": -5.0},
        "high_level",
    )
    assert len(thrusters) == 8
    assert all(-adapter.thruster_limit <= value <= adapter.thruster_limit for value in thrusters)


def test_high_level_yaw_maps_to_counterrotating_horizontal_thrusters() -> None:
    adapter, participant = _adapter()
    thrusters = adapter.command_to_bluerov2_thrusters(
        participant.id,
        {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 1.0},
        "high_level",
    )

    assert thrusters[:4] == [0.0, 0.0, 0.0, 0.0]
    assert thrusters[4] < 0.0
    assert thrusters[5] > 0.0
    assert thrusters[6] > 0.0
    assert thrusters[7] < 0.0


def test_thruster_command_is_padded_and_clamped() -> None:
    adapter, participant = _adapter()
    thrusters = adapter.command_to_bluerov2_thrusters(
        participant.id,
        {"thrusters": [99.0, -99.0, 0.25]},
        "thrusters",
    )
    assert len(thrusters) == 8
    assert thrusters[0] == adapter.thruster_limit
    assert thrusters[1] == -adapter.thruster_limit
    assert thrusters[3:] == [0.0, 0.0, 0.0, 0.0, 0.0]
