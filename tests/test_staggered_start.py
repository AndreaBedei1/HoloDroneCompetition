from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from marine_race_arena.adapters.fallback_adapter import FallbackRaceAdapter
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config, parse_track_config
from marine_race_arena.participants.participant import RaceParticipant
from marine_race_arena.referee.race_state import ParticipantStatus
from marine_race_arena.referee.referee import Referee
from marine_race_arena.scripts.run_marine_race import _run_race_loop, _with_staggered_participants


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


def test_participant_spawn_start_delay_is_parsed() -> None:
    raw = json.loads((TRACK_DIR / "marine_race_horseshoe_bay.json").read_text(encoding="utf-8"))
    raw["participants"] = copy.deepcopy(raw["participants"])
    raw["participants"][0]["spawn"]["start_delay_s"] = 12.5

    config = parse_track_config(raw)

    assert config.participants[0].start_delay_s == 12.5
    assert config.participants[0].spawn["start_delay_s"] == 12.5


def test_cli_staggered_start_generates_offsets_and_delays() -> None:
    config = load_track_config(TRACK_DIR / "marine_race_horseshoe_bay.json")
    args = argparse.Namespace(
        staggered_start=True,
        num_rovers=3,
        start_gap_s=20.0,
        staggered_lateral_offset_m=2.5,
    )

    updated = _with_staggered_participants(config, args)

    assert [participant.id for participant in updated.participants] == [
        "bluerov2_01",
        "bluerov2_02",
        "bluerov2_03",
    ]
    assert [participant.start_delay_s for participant in updated.participants] == [0.0, 20.0, 40.0]
    positions = [tuple(participant.spawn["position"]) for participant in updated.participants]
    assert len(set(positions)) == 3
    assert all(updated.world.bounds.contains(position) for position in positions)
    assert _horizontal_distance(positions[0], positions[1]) > 2.0
    assert _horizontal_distance(positions[0], positions[2]) > 2.0


def test_referee_staggered_release_timing_and_waiting_gate_timeout() -> None:
    referee = _referee(["p1", "p2"])

    referee.start_race(0.0, start_delays={"p1": 0.0, "p2": 2.0})
    referee.gate_timeout_stuck("p2", time_s=1.0, timeout_s=1.0)

    assert referee.states["p1"].status == ParticipantStatus.RUNNING
    assert referee.states["p1"].release_time_s == 0.0
    assert referee.states["p2"].status == ParticipantStatus.NOT_STARTED
    assert referee.states["p2"].stuck_events == 0

    referee.release_participant("p2", 2.0)

    state = referee.states["p2"]
    assert state.status == ParticipantStatus.RUNNING
    assert state.release_time_s == 2.0
    assert state.green_start_time == 2.0
    assert state.stuck_accumulator_s == 0.0


def test_waiting_participant_gets_zero_commands_but_no_controller_steps() -> None:
    context = _run_short_staggered_fallback()

    assert context.controllers["p_run"].step_calls > 0
    assert context.controllers["p_wait"].step_calls == 0
    wait_commands = context.adapter.commands_by_participant["p_wait"]
    assert wait_commands
    assert all(command == {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0} for command in wait_commands)


def test_waiting_participant_does_not_accumulate_stuck_or_gate_timeout() -> None:
    context = _run_short_staggered_fallback(gate_timeout_s=0.05)
    wait_state = context.referee.states["p_wait"]

    assert wait_state.status == ParticipantStatus.NOT_STARTED
    assert wait_state.stuck_events == 0
    assert wait_state.stuck_accumulator_s == 0.0


def test_released_participant_runs_and_summary_contains_all_participants() -> None:
    context = _run_short_staggered_fallback()
    run_state = context.referee.states["p_run"]

    assert run_state.status == ParticipantStatus.RUNNING
    assert run_state.release_time_s == 0.0
    assert len(context.summary["participants"]) == 2
    assert len(context.summary["ranking"]) == 2
    by_id = {participant["participant_id"]: participant for participant in context.summary["participants"]}
    assert by_id["p_run"]["start_delay_s"] == 0.0
    assert by_id["p_wait"]["start_delay_s"] == 10.0
    assert by_id["p_wait"]["status"] == "NOT_STARTED"


class _CountingController:
    def __init__(self) -> None:
        self.step_calls = 0

    def reset(self, mission_info: dict[str, Any]) -> None:
        pass

    def step(self, observation: dict[str, Any]) -> dict[str, float]:
        self.step_calls += 1
        return {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}

    def close(self) -> None:
        pass


class _RecordingFallbackAdapter(FallbackRaceAdapter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.commands_by_participant: dict[str, list[dict[str, float]]] = {}

    def apply_command(self, participant_id: str, command: Mapping[str, Any], control_mode: str) -> None:
        safe = self.clamp_high_level_command(command, participant_id=participant_id)
        self.commands_by_participant.setdefault(participant_id, []).append(safe)
        super().apply_command(participant_id, safe, control_mode)


class _RunContext:
    def __init__(
        self,
        *,
        controllers: dict[str, _CountingController],
        adapter: _RecordingFallbackAdapter,
        referee: Referee,
        summary: dict[str, Any],
    ) -> None:
        self.controllers = controllers
        self.adapter = adapter
        self.referee = referee
        self.summary = summary


def _run_short_staggered_fallback(gate_timeout_s: float | None = None) -> _RunContext:
    config = load_track_config(TRACK_DIR / "marine_race_horseshoe_bay.json")
    config = replace(config, race=replace(config.race, max_duration_s=0.2, official_mode=False))
    base = config.participants[0]
    run_spawn = dict(base.spawn)
    wait_spawn = dict(base.spawn)
    run_spawn["start_delay_s"] = 0.0
    wait_spawn["start_delay_s"] = 10.0
    run_config = replace(base, id="p_run", spawn=run_spawn, start_delay_s=0.0)
    wait_config = replace(base, id="p_wait", spawn=wait_spawn, start_delay_s=10.0)
    config = replace(config, participants=[run_config, wait_config])
    arena = ArenaBuilder(config).build()
    controllers = {"p_run": _CountingController(), "p_wait": _CountingController()}
    participants = {
        participant_config.id: RaceParticipant(
            config=participant_config,
            controller=controllers[participant_config.id],
            position=tuple(participant_config.spawn["position"]),
            rotation_rpy_deg=tuple(participant_config.spawn["rotation_rpy_deg"]),
        )
        for participant_config in config.participants
    }
    adapter = _RecordingFallbackAdapter(config, arena)
    adapter.initialize()
    adapter.spawn_participants(participants)
    adapter.reset()
    referee = Referee(config, arena.gate_map, arena.bounds)
    referee.register_participants(participants.keys())

    summary = _run_race_loop(
        config=config,
        arena=arena,
        referee=referee,
        adapter=adapter,
        participants=participants,
        dt=0.1,
        gate_timeout_s=gate_timeout_s,
        log_participant_states=False,
    )
    return _RunContext(controllers=controllers, adapter=adapter, referee=referee, summary=summary)


def _referee(participant_ids: list[str]) -> Referee:
    config = load_track_config(TRACK_DIR / "marine_race_horseshoe_bay.json")
    arena = ArenaBuilder(config).build()
    referee = Referee(config, arena.gate_map, arena.bounds)
    referee.register_participants(participant_ids)
    return referee


def _horizontal_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
