from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from marine_race_arena.adapters import AdapterSelectionError, FallbackRaceAdapter, RaceAdapterUnavailable, select_adapter
from marine_race_arena.adapters.holoocean_adapter import HoloOceanRaceAdapter
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.participants.participant import RaceParticipant


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


def _config_arena_participant():
    config = load_track_config(TRACK_DIR / "marine_race_horseshoe_bay.json")
    arena = ArenaBuilder(config).build()
    participant_config = config.participants[0]
    participant = RaceParticipant(
        config=participant_config,
        controller=object(),
        position=tuple(participant_config.spawn["position"]),
        rotation_rpy_deg=tuple(participant_config.spawn["rotation_rpy_deg"]),
    )
    return config, arena, {participant.id: participant}


def test_auto_adapter_does_not_silently_fallback_without_permission() -> None:
    config, arena, _ = _config_arena_participant()
    with patch(
        "marine_race_arena.adapters.HoloOceanRaceAdapter.initialize",
        side_effect=RaceAdapterUnavailable("forced unavailable"),
    ):
        try:
            select_adapter("auto", config, arena, allow_fallback=False)
        except AdapterSelectionError as exc:
            assert "fallback is not allowed" in str(exc)
        else:
            raise AssertionError("auto adapter silently fell back without permission")


def test_official_auto_adapter_does_not_fallback_without_permission() -> None:
    config, arena, _ = _config_arena_participant()
    config = replace(config, race=replace(config.race, official_mode=True))
    with patch(
        "marine_race_arena.adapters.HoloOceanRaceAdapter.initialize",
        side_effect=RaceAdapterUnavailable("forced unavailable"),
    ):
        try:
            select_adapter("auto", config, arena, allow_fallback=False)
        except AdapterSelectionError as exc:
            assert "fallback is not allowed" in str(exc)
        else:
            raise AssertionError("official auto adapter silently fell back without permission")


def test_auto_adapter_can_fallback_when_allowed() -> None:
    config, arena, _ = _config_arena_participant()
    with patch(
        "marine_race_arena.adapters.HoloOceanRaceAdapter.initialize",
        side_effect=RaceAdapterUnavailable("forced unavailable"),
    ):
        adapter = select_adapter("auto", config, arena, allow_fallback=True)
    assert isinstance(adapter, FallbackRaceAdapter)


def test_fallback_adapter_moves_after_high_level_command() -> None:
    config, arena, participants = _config_arena_participant()
    adapter = FallbackRaceAdapter(config, arena)
    adapter.initialize()
    adapter.spawn_participants(participants)
    before = adapter.get_participant_state("bluerov2_01").position

    adapter.apply_command("bluerov2_01", {"surge": 1.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}, "high_level")
    adapter.step(0.1)
    after = adapter.get_participant_state("bluerov2_01").position

    assert after[0] > before[0]


def test_command_mapping_clamps_thrusters() -> None:
    config, arena, participants = _config_arena_participant()
    adapter = FallbackRaceAdapter(config, arena)
    adapter.initialize()
    adapter.spawn_participants(participants)

    thrusters = adapter.command_to_bluerov2_thrusters(
        "bluerov2_01",
        {"surge": 4.0, "sway": -4.0, "heave": 3.0, "yaw": 2.0},
        "high_level",
    )

    assert len(thrusters) == 8
    assert all(-1.0 <= value <= 1.0 for value in thrusters)


def test_official_filter_removes_ground_truth_pose() -> None:
    config, arena, _ = _config_arena_participant()
    adapter = FallbackRaceAdapter(config, arena)
    raw = {
        "DepthSensor": 4.0,
        "IMUSensor": [[0.0, 0.0, 0.0]],
        "PoseSensor": [[1.0, 0.0, 0.0, 2.0]],
        "LocationSensor": [1.0, 2.0, -4.0],
    }

    filtered = adapter.filter_sensor_data(raw, {"allowed_sensors": ["DepthSensor", "IMUSensor", "PoseSensor"]}, official_mode=True)

    assert "DepthSensor" in filtered
    assert "IMUSensor" in filtered
    assert "PoseSensor" not in filtered
    assert "LocationSensor" not in filtered


def test_holoocean_close_uses_context_manager_and_drops_environment_references() -> None:
    config, arena, _ = _config_arena_participant()

    class ExitOnlyEnvironment:
        def __init__(self) -> None:
            self.exit_calls: list[tuple[object, object, object]] = []

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            self.exit_calls.append((exc_type, exc, traceback))

    env = ExitOnlyEnvironment()
    adapter = HoloOceanRaceAdapter(config, arena)
    adapter.env = env
    adapter.visual_spawner = object()  # type: ignore[assignment]

    adapter.close()

    assert env.exit_calls == [(None, None, None)]
    assert adapter.env is None
    assert adapter.visual_spawner is None


def test_holoocean_close_force_reaps_a_leaked_windows_world_process() -> None:
    config, arena, _ = _config_arena_participant()

    class LeakedWorldProcess:
        pid = 43210

        def __init__(self) -> None:
            self.kill_calls = 0
            self.stopped = False

        def poll(self):
            return 0 if self.stopped else None

        def kill(self) -> None:
            self.kill_calls += 1

        def wait(self, timeout: int):
            if not self.stopped:
                raise subprocess.TimeoutExpired("Holodeck", timeout)
            return 0

    class LeakingEnvironment:
        def __init__(self) -> None:
            self._world_process = LeakedWorldProcess()
            self.exit_calls = 0

        def __exit__(self, exc_type, exc, traceback) -> None:
            self.exit_calls += 1

    env = LeakingEnvironment()
    adapter = HoloOceanRaceAdapter(config, arena)
    adapter.env = env

    def taskkill(command, **kwargs):
        assert command == ["taskkill", "/PID", "43210", "/T", "/F"]
        env._world_process.stopped = True
        return subprocess.CompletedProcess(command, 0)

    with (
        patch("marine_race_arena.adapters.holoocean_adapter.os.name", "nt"),
        patch(
            "marine_race_arena.adapters.holoocean_adapter.subprocess.run",
            side_effect=taskkill,
        ) as run,
    ):
        adapter.close()

    assert env.exit_calls == 1
    assert env._world_process.kill_calls == 1
    run.assert_called_once()
    assert adapter.env is None
