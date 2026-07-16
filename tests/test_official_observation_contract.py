"""Contract tests: the official observation and mission info carry no referee data.

These tests execute the real runner code paths (observation builder, mission
info, race loop) on the fallback adapter and assert, field by field, that
nothing referee-derived, ground-truth-derived or environment-privileged ever
reaches a controller.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from marine_race_arena.adapters.fallback_adapter import FallbackRaceAdapter
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.arena.acoustic_comms import AcousticCommsChannel, CommsConfig
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.controllers.official_baselines import (
    RuleGateBaselineController,
    RuleGateCenterThenCommitController,
)
from marine_race_arena.participants.controller_interface import BaseController
from marine_race_arena.participants.participant import RaceParticipant
from marine_race_arena.participants.sensor_profile import (
    FORBIDDEN_SENSOR_KEYS,
    OFFICIAL_SENSOR_ALLOWLIST,
    build_observation,
)
from marine_race_arena.referee.referee import Referee
from marine_race_arena.referee.race_state import ParticipantRaceState, ParticipantStatus
from marine_race_arena.scripts import run_marine_race
from marine_race_arena.scripts.run_marine_race import (
    _build_controller_observation,
    _log_controller_command,
    _log_controller_local_state,
    _log_comms_deliveries,
    _mission_info,
    _needs_local_finish_tail,
    _run_race_loop,
)

TRACK = "marine_race_arena/tracks/tests/three_gate_s_curve.json"

FORBIDDEN_OBSERVATION_KEYS = {
    "race",
    "beacon",
    "time_s",
    "participant_id",
    "debug_ground_truth",
    "target_gate_id",
    "target_sequence_index",
    "completed_gates",
    "status",
    "lap",
    "official_time_started",
}

FORBIDDEN_MISSION_KEYS = {
    "race_name",
    "format",
    "timing_mode",
    "official_mode",
    "benchmark_task",
    "obstacle_mode",
    "obstacle_density",
    "obstacle_physics",
    "current_profile",
    "active_current_count",
    "motion_compensation",
    "max_duration_s",
    "adapter",
    "bounds",
    "initial_target_gate_id",
    "gates_per_lap",
    "max_command",
    "target_gate_id",
}

FORBIDDEN_PACKET_KEYS = {
    "valid",
    "reason",
    "active_beacon_id",
    "target_gate_id",
    "sequence_index",
    "noise_level",
    "mode",
    "message",
    "channel",
    "visible_beacon_ids",
    "exact_gate_center",
    "exact_gate_normal",
    "exact_beacon_position",
}


def _setup(track=TRACK, official=True, participants=None):
    config = load_track_config(track)
    config = replace(config, race=replace(config.race, official_mode=official, max_duration_s=60.0))
    if participants is not None:
        config = replace(config, participants=participants)
    arena = ArenaBuilder(config, seed=0).build()
    return config, arena


def _make_participant(config, controller=None):
    pc = config.participants[0]
    controller = controller or RuleGateBaselineController()
    participant = RaceParticipant(
        config=pc,
        controller=controller,
        position=tuple(pc.spawn["position"]),
        rotation_rpy_deg=tuple(pc.spawn["rotation_rpy_deg"]),
    )
    return pc, participant


def _observation(config, arena, participant, adapter, comms_inbox=None):
    state = adapter.get_participant_state(participant.id)
    return _build_controller_observation(
        config=config,
        arena=arena,
        adapter=adapter,
        participant=participant,
        participant_state=state,
        release_time_s=0.0,
        comms_inbox=comms_inbox,
    )


def _adapter_for(config, arena, participant):
    adapter = FallbackRaceAdapter(config, arena)
    adapter.initialize()
    adapter.spawn_participants({participant.id: participant})
    adapter.reset()
    adapter.spawn_visual_gates(arena.visual_gates)
    return adapter


class _EventCollector:
    def __init__(self):
        self.events = []

    def log_event(self, event_type, time_s, participant_id=None, **payload):
        self.events.append(
            {
                "event": event_type,
                "time_s": time_s,
                "participant_id": participant_id,
                **payload,
            }
        )


# ------------------------------------------------------------ observation


def test_official_observation_contains_only_approved_top_level_fields():
    config, arena = _setup()
    _, participant = _make_participant(config)
    adapter = _adapter_for(config, arena, participant)
    observation = _observation(config, arena, participant, adapter)
    assert set(observation.keys()) <= {"local_time_s", "sensors", "beacons", "comms"}
    assert "comms" not in observation  # comms only present when enabled
    assert not FORBIDDEN_OBSERVATION_KEYS & set(observation.keys())


def test_official_observation_sensors_are_allowlisted_onboard_sensors():
    config, arena = _setup()
    _, participant = _make_participant(config)
    adapter = _adapter_for(config, arena, participant)
    observation = _observation(config, arena, participant, adapter)
    sensors = observation["sensors"]
    assert set(sensors.keys()) <= OFFICIAL_SENSOR_ALLOWLIST
    assert not FORBIDDEN_SENSOR_KEYS & set(sensors.keys())
    for name in ("heading_yaw_deg", "depth_m", "VelocitySensor", "PoseSensor",
                 "environment_current_m_s", "current_physical_coupling_active",
                 "current_coupling_method", "control_mode"):
        assert name not in sensors


def test_official_observation_beacons_are_pure_packets():
    config, arena = _setup()
    _, participant = _make_participant(config)
    adapter = _adapter_for(config, arena, participant)
    observation = _observation(config, arena, participant, adapter)
    assert isinstance(observation["beacons"], list)
    assert observation["beacons"], "spawn is in range of the test track beacons"
    for packet in observation["beacons"]:
        assert set(packet.keys()) == {
            "beacon_id", "bearing_deg", "elevation_deg", "range_m",
            "signal_strength", "received_at_s",
        }
        assert not FORBIDDEN_PACKET_KEYS & set(packet.keys())


def test_local_time_is_participant_local_from_release():
    config, arena = _setup()
    _, participant = _make_participant(config)
    adapter = _adapter_for(config, arena, participant)
    for _ in range(7):
        adapter.step(0.1)
    state = adapter.get_participant_state(participant.id)
    observation = _build_controller_observation(
        config=config, arena=arena, adapter=adapter, participant=participant,
        participant_state=state, release_time_s=0.5,
    )
    assert observation["local_time_s"] == pytest.approx(0.2, abs=1e-6)


def test_observation_builder_takes_no_referee():
    signature = inspect.signature(_build_controller_observation)
    assert "referee" not in signature.parameters
    source = inspect.getsource(_build_controller_observation)
    assert "expected_gate_id" not in source
    assert "race_progress" not in source


def test_build_observation_never_emits_race_or_debug_in_official_mode():
    observation = build_observation(
        local_time_s=1.0,
        sensor_data={"DepthSensor": [-3.0], "PoseSensor": [[1.0]]},
        beacon_packets=[],
        official_mode=True,
        debug_ground_truth={"own_position": (0, 0, 0)},
    )
    assert "race" not in observation
    assert "debug_ground_truth" not in observation
    assert "PoseSensor" not in observation["sensors"]


def test_comms_block_present_only_when_enabled_and_local_clocked():
    config, arena = _setup()
    _, participant = _make_participant(config)
    adapter = _adapter_for(config, arena, participant)
    inbox = [{"from": "other", "payload": {"local_beacon_index": 2}, "sent_at_s": 9.0,
              "received_at_s": 10.0}]
    observation = _observation(config, arena, participant, adapter, comms_inbox=inbox)
    assert "comms" in observation
    message = observation["comms"]["inbox"][0]
    assert set(message.keys()) == {"from", "payload", "received_at_s"}
    assert "sent_at_s" not in message
    assert message["received_at_s"] == pytest.approx(10.0)  # release at 0.0


def test_controller_command_logger_records_every_heartbeat_inside_throttle_window():
    logger = _EventCollector()
    referee = SimpleNamespace(logger=logger)
    last_log_times = {}
    for time_s, beacon_index in ((1.0, 2), (1.5, 3)):
        _log_controller_command(
            referee=referee,
            participant_id="bluerov2_01",
            time_s=time_s,
            local_time_s=time_s,
            command={
                "surge": 0.1,
                "sway": 0.0,
                "heave": 0.0,
                "yaw": 0.0,
                "message": {
                    "local_beacon_index": beacon_index,
                    "local_lap": 1,
                    "local_status": "RUNNING",
                },
            },
            last_log_times=last_log_times,
        )

    assert [event["message"]["local_beacon_index"] for event in logger.events] == [2, 3]


def test_controller_command_logger_still_throttles_commands_without_messages():
    logger = _EventCollector()
    referee = SimpleNamespace(logger=logger)
    last_log_times = {}
    for time_s in (1.0, 1.5):
        _log_controller_command(
            referee=referee,
            participant_id="bluerov2_01",
            time_s=time_s,
            local_time_s=time_s,
            command={"surge": 0.1, "sway": 0.0, "heave": 0.0, "yaw": 0.0},
            last_log_times=last_log_times,
        )

    assert len(logger.events) == 1


def test_comms_delivery_log_is_offline_timing_only_and_omits_payload():
    logger = _EventCollector()
    referee = SimpleNamespace(logger=logger)
    _log_comms_deliveries(
        referee=referee,
        receiver_id="bluerov2_02",
        inbox=[
            {
                "from": "bluerov2_01",
                "payload": {"local_beacon_index": 4},
                "sent_at_s": 10.0,
                "received_at_s": 10.075,
            }
        ],
    )

    assert logger.events == [
        {
            "event": "comms_delivery",
            "time_s": 10.075,
            "participant_id": "bluerov2_02",
            "sender_id": "bluerov2_01",
            "sent_at_s": 10.0,
            "received_at_s": 10.075,
            "latency_s": pytest.approx(0.075),
        }
    ]
    assert "payload" not in logger.events[0]


def test_local_state_logger_emits_immediately_when_hold_decision_changes():
    logger = _EventCollector()
    referee = SimpleNamespace(logger=logger)
    controller = SimpleNamespace(
        tracker=SimpleNamespace(diagnostics=lambda: {"advancements": 2}),
        coordination_diagnostics={
            "is_holding": False,
            "hold_reason": None,
            "decision_reason": "gap_satisfied",
            "hold_steps": 0,
            "advance_steps": 1,
        },
    )
    last_log_times = {}
    _log_controller_local_state(
        referee=referee,
        controller=controller,
        participant_id="bluerov2_02",
        time_s=5.0,
        last_log_times=last_log_times,
    )
    controller.coordination_diagnostics = {
        **controller.coordination_diagnostics,
        "is_holding": True,
        "hold_reason": "gap_below_minimum",
        "decision_reason": "gap_below_minimum",
        "hold_steps": 1,
    }
    _log_controller_local_state(
        referee=referee,
        controller=controller,
        participant_id="bluerov2_02",
        time_s=5.1,
        last_log_times=last_log_times,
    )

    assert len(logger.events) == 2
    assert logger.events[-1]["coordination_is_holding"] is True
    assert logger.events[-1]["coordination_hold_reason"] == "gap_below_minimum"


# ------------------------------------------------------------ mission info


def test_mission_info_contains_only_approved_fields_single_rover():
    config, _ = _setup()
    info = _mission_info(config, config.participants[0].id)
    assert set(info.keys()) == {
        "participant_id", "initial_beacon_id", "total_beacons", "laps", "command_limits",
    }
    assert info["initial_beacon_id"] == "B01"
    assert info["total_beacons"] == len(config.track.gate_sequence)
    assert not FORBIDDEN_MISSION_KEYS & set(info.keys())


def test_mission_info_fleet_block_carries_static_release_order_only():
    config, _ = _setup()
    base = config.participants[0]
    fleet = [
        replace(base, id="bluerov2_01", start_delay_s=0.0),
        replace(base, id="bluerov2_02", start_delay_s=8.0),
        replace(base, id="bluerov2_03", start_delay_s=16.0),
    ]
    config = replace(config, participants=fleet)
    info = _mission_info(config, "bluerov2_02")
    assert set(info.keys()) == {
        "participant_id", "initial_beacon_id", "total_beacons", "laps",
        "command_limits", "fleet",
    }
    assert info["fleet"] == {
        "participant_order": ["bluerov2_01", "bluerov2_02", "bluerov2_03"],
        "release_index": 1,
        "predecessor_id": "bluerov2_01",
    }
    leader = _mission_info(config, "bluerov2_01")
    assert leader["fleet"]["predecessor_id"] is None


# ------------------------------------------------------- end-to-end sweep


class _ObservationRecorder(BaseController):
    uses_ground_truth = False
    debug_only = False

    def __init__(self):
        self.observations = []
        self.mission_info = None

    def reset(self, mission_info):
        self.mission_info = mission_info

    def step(self, observation):
        self.observations.append(observation)
        return {"surge": 0.3, "sway": 0.0, "heave": 0.0, "yaw": 0.0}

    def close(self):
        pass


def test_full_race_loop_never_leaks_forbidden_fields():
    config, arena = _setup()
    config = replace(config, race=replace(config.race, max_duration_s=6.0))
    recorder = _ObservationRecorder()
    pc, participant = _make_participant(config)
    participant = RaceParticipant(
        config=pc, controller=recorder,
        position=tuple(pc.spawn["position"]),
        rotation_rpy_deg=tuple(pc.spawn["rotation_rpy_deg"]),
    )
    adapter = _adapter_for(config, arena, participant)
    referee = Referee(config, arena.gate_map, arena.bounds)
    referee.register_participants([participant.id])
    recorder.reset(_mission_info(config, participant.id))
    _run_race_loop(
        config=config, arena=arena, referee=referee, adapter=adapter,
        participants={participant.id: participant}, dt=0.1,
    )
    assert recorder.observations, "controller must have been stepped"
    for observation in recorder.observations:
        assert set(observation.keys()) <= {"local_time_s", "sensors", "beacons"}
        assert not FORBIDDEN_OBSERVATION_KEYS & set(observation.keys())
        for packet in observation["beacons"]:
            assert not FORBIDDEN_PACKET_KEYS & set(packet.keys())
    assert not FORBIDDEN_MISSION_KEYS & set(recorder.mission_info.keys())


def test_runner_source_has_no_referee_navigation_calls():
    """The controller-facing paths never consult the referee.

    The referee may keep using privileged state for its own event logging
    (``_log_participant_states``), but the observation builder and the mission
    info given to controllers must not touch it.
    """
    for function in (_build_controller_observation, _mission_info):
        assert "referee" not in inspect.signature(function).parameters
        source = inspect.getsource(function)
        assert "referee." not in source
        assert "expected_gate_id" not in source
        assert "race_progress" not in source
    assert "race_progress" not in inspect.getsource(run_marine_race)


def test_post_finish_local_confirmation_tail_is_bounded_and_finish_only():
    class Tracker:
        finished = False

    class Controller:
        tracker = Tracker()

    state = ParticipantRaceState(
        participant_id="p01",
        status=ParticipantStatus.FINISHED,
        official_finish_time=10.0,
    )

    assert _needs_local_finish_tail(state, Controller(), 17.99, grace_s=8.0)
    assert not _needs_local_finish_tail(state, Controller(), 18.01, grace_s=8.0)
    Controller.tracker.finished = True
    assert not _needs_local_finish_tail(state, Controller(), 10.1, grace_s=8.0)
    Controller.tracker.finished = False
    state.status = ParticipantStatus.DNF
    assert not _needs_local_finish_tail(state, Controller(), 10.1, grace_s=8.0)


def test_official_controllers_never_read_race_fields():
    import marine_race_arena.controllers.official_baselines as baselines
    import marine_race_arena.controllers.leader_follower as leader_follower
    import marine_race_arena.controllers.local_course_tracker as tracker
    import marine_race_arena.controllers.student_template as student

    for module in (baselines, leader_follower, tracker, student):
        source = inspect.getsource(module)
        assert 'observation.get("race")' not in source
        assert "observation[\"race\"]" not in source
        assert "completed_gates" not in source
        assert "target_sequence_index" not in source
        assert "expected_gate_id" not in source


def test_repository_wide_forbidden_field_audit():
    """No controller module may reference referee-provided fields at all."""
    controllers_dir = Path("marine_race_arena/controllers")
    forbidden_markers = (
        'observation["race"]',
        "observation.get(\"race\")",
        "observation['race']",
        "observation.get('race')",
        "race_progress",
        "expected_gate_id",
        "initial_target_gate_id",
        "target_sequence_index",
    )
    offenders = []
    for path in controllers_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in forbidden_markers:
            if marker in text:
                offenders.append(f"{path.name}: {marker}")
    assert not offenders, f"controllers reference forbidden fields: {offenders}"


def test_both_official_controllers_run_on_the_official_contract():
    for controller_type in (RuleGateBaselineController, RuleGateCenterThenCommitController):
        config, arena = _setup()
        controller = controller_type()
        pc, participant = _make_participant(config, controller)
        adapter = _adapter_for(config, arena, participant)
        controller.reset(_mission_info(config, pc.id))
        observation = _observation(config, arena, participant, adapter)
        command = controller.step(observation)
        assert set(command.keys()) >= {"surge", "sway", "heave", "yaw"}
        # Missing packets and missing camera are handled safely.
        command = controller.step({"local_time_s": 1.0, "sensors": {}, "beacons": []})
        assert all(abs(float(command[axis])) <= 1.0 for axis in ("surge", "sway", "heave", "yaw"))
