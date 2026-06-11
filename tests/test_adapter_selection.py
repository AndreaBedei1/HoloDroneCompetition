from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from marine_race_arena.adapters import AdapterSelectionError, FallbackRaceAdapter, RaceAdapterUnavailable, select_adapter
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.controllers.oracle_gate_follower import OracleGateFollowerController
from marine_race_arena.participants.participant import RaceParticipant
from marine_race_arena.scripts.run_marine_race import _reject_invalid_official_controllers


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


def _config_and_arena():
    config = load_track_config(TRACK_DIR / "abu_dhabi_marine_easy.json")
    return config, ArenaBuilder(config).build()


def test_adapter_fallback_selects_fallback() -> None:
    config, arena = _config_and_arena()
    adapter = select_adapter("fallback", config, arena, allow_fallback=False)
    assert isinstance(adapter, FallbackRaceAdapter)


def test_auto_without_allow_fallback_fails_when_holoocean_unavailable() -> None:
    config, arena = _config_and_arena()
    with patch(
        "marine_race_arena.adapters.HoloOceanRaceAdapter.initialize",
        side_effect=RaceAdapterUnavailable("forced unavailable"),
    ):
        try:
            select_adapter("auto", config, arena, allow_fallback=False)
        except AdapterSelectionError as exc:
            assert "fallback is not allowed" in str(exc)
        else:
            raise AssertionError("auto adapter silently fell back")


def test_auto_with_allow_fallback_uses_fallback_when_holoocean_unavailable() -> None:
    config, arena = _config_and_arena()
    with patch(
        "marine_race_arena.adapters.HoloOceanRaceAdapter.initialize",
        side_effect=RaceAdapterUnavailable("forced unavailable"),
    ):
        adapter = select_adapter("auto", config, arena, allow_fallback=True)
    assert isinstance(adapter, FallbackRaceAdapter)


def test_official_mode_blocks_oracle_controller() -> None:
    config, _ = _config_and_arena()
    config = replace(config, race=replace(config.race, official_mode=True))
    participant_config = config.participants[0]
    participant = RaceParticipant(
        config=participant_config,
        controller=OracleGateFollowerController(),
        position=tuple(participant_config.spawn["position"]),
        rotation_rpy_deg=tuple(participant_config.spawn["rotation_rpy_deg"]),
    )
    try:
        _reject_invalid_official_controllers(config, {participant.id: participant})
    except Exception as exc:
        assert "not allowed in official mode" in str(exc)
    else:
        raise AssertionError("official mode accepted oracle controller")

