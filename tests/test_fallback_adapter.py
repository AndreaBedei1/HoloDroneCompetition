from __future__ import annotations

from pathlib import Path

from marine_race_arena.adapters.fallback_adapter import FallbackRaceAdapter
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.participants.participant import RaceParticipant


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


def _adapter_and_participant(z_override: float | None = None):
    config = load_track_config(TRACK_DIR / "abu_dhabi_marine_easy.json")
    arena = ArenaBuilder(config).build()
    participant_config = config.participants[0]
    position = tuple(participant_config.spawn["position"])
    if z_override is not None:
        position = (position[0], position[1], z_override)
    participant = RaceParticipant(
        config=participant_config,
        controller=object(),
        position=position,
        rotation_rpy_deg=tuple(participant_config.spawn["rotation_rpy_deg"]),
    )
    adapter = FallbackRaceAdapter(config, arena)
    adapter.initialize()
    adapter.spawn_participants({participant.id: participant})
    return config, adapter, participant


def test_fallback_adapter_moves_forward() -> None:
    _, adapter, participant = _adapter_and_participant()
    before = adapter.get_participant_state(participant.id).position
    adapter.apply_command(participant.id, {"surge": 0.5}, "high_level")
    adapter.step(0.2)
    after = adapter.get_participant_state(participant.id).position
    assert after[0] > before[0]


def test_heave_is_clamped_near_z_min() -> None:
    config, adapter, participant = _adapter_and_participant(z_override=-7.9)
    assert config.world.bounds.z_min == -8.0
    command = adapter.clamp_high_level_command({"heave": -1.0}, participant_id=participant.id)
    assert command["heave"] == 0.0


def test_heave_is_clamped_near_z_max() -> None:
    config, adapter, participant = _adapter_and_participant(z_override=-1.1)
    assert config.world.bounds.z_max == -1.0
    command = adapter.clamp_high_level_command({"heave": 1.0}, participant_id=participant.id)
    assert command["heave"] == 0.0
