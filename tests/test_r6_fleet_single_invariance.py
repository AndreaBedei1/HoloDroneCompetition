"""R6: single-vehicle behaviour is unchanged when fleet mode is disabled.

R6 is an architectural invariant, verified here with deterministic regression
tests on the fallback (engine-free) adapter -- no HoloOcean run is launched:

* the official observation carries no ``comms`` field when communication is off;
* enabling the communication path adds only ``comms`` and leaves the other
  observation fields untouched;
* a single-vehicle run is deterministic (same inputs -> same commands, score);
* adding a second vehicle and the team-aggregation path does not change the
  first vehicle's observations or its referee score.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

from marine_race_arena.adapters.fallback_adapter import FallbackRaceAdapter
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.participants.participant import RaceParticipant
from marine_race_arena.participants.sensor_profile import build_observation
from marine_race_arena.referee.referee import Referee
from marine_race_arena.scripts.run_marine_race import _mission_info, _run_race_loop

TRACK = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks" / "marine_race_horseshoe_bay.json"

_SCORE_FIELDS = (
    "status",
    "completed_gates",
    "official_time_s",
    "green_to_finish_time_s",
    "penalties_s",
    "penalized_time_s",
    "collisions",
    "obstacle_collisions",
    "out_of_bounds_events",
    "stuck_events",
)


class _RecordingStub:
    """Deterministic onboard controller that records the observations it sees."""

    uses_ground_truth = False

    def __init__(self) -> None:
        self.observations: List[Dict[str, Any]] = []
        self.commands: List[Dict[str, float]] = []

    def reset(self, mission_info: Dict[str, Any]) -> None:
        self.observations = []
        self.commands = []

    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        # Record only the JSON-safe, comparable parts of the observation.
        self.observations.append(
            {
                "local_time_s": observation.get("local_time_s"),
                "beacons": observation.get("beacons"),
                "sensor_keys": sorted(observation.get("sensors", {}).keys()),
                "has_comms": "comms" in observation,
            }
        )
        command = {"surge": 0.3, "sway": 0.0, "heave": 0.0, "yaw": 0.0}
        self.commands.append(command)
        return command

    def close(self) -> None:
        pass


def _run(participant_specs, *, duration_s: float = 1.5, dt: float = 0.1):
    """Run a deterministic fallback race with the given participant ids/spawns."""
    config = load_track_config(TRACK)
    config = replace(config, race=replace(config.race, max_duration_s=duration_s, official_mode=True))
    base = config.participants[0]
    participant_configs = []
    for spec in participant_specs:
        spawn = dict(base.spawn)
        spawn["position"] = list(spec["position"])
        spawn["start_delay_s"] = 0.0
        participant_configs.append(
            replace(base, id=spec["id"], spawn=spawn, start_delay_s=0.0)
        )
    config = replace(config, participants=participant_configs)
    arena = ArenaBuilder(config, seed=0).build()
    controllers = {spec["id"]: _RecordingStub() for spec in participant_specs}
    participants = {
        pc.id: RaceParticipant(
            config=pc,
            controller=controllers[pc.id],
            position=tuple(pc.spawn["position"]),
            rotation_rpy_deg=tuple(pc.spawn["rotation_rpy_deg"]),
        )
        for pc in config.participants
    }
    adapter = FallbackRaceAdapter(config, arena)
    adapter.initialize()
    adapter.spawn_participants(participants)
    adapter.reset()
    referee = Referee(config, arena.gate_map, arena.bounds)
    referee.register_participants(participants.keys())
    for participant in participants.values():
        participant.controller.reset(_mission_info(config, participant.id))
    summary = _run_race_loop(
        config=config,
        arena=arena,
        referee=referee,
        adapter=adapter,
        participants=participants,
        dt=dt,
        gate_timeout_s=None,
        log_participant_states=False,
        comms_channel=None,  # communication disabled
    )
    return controllers, summary


def _row(summary, participant_id):
    for row in summary["participants"]:
        if row["participant_id"] == participant_id:
            return {k: row.get(k) for k in _SCORE_FIELDS}
    raise AssertionError(f"{participant_id} not in summary")


# --------------------------------------------------------------------------- #
# Observation builder: comms is gated, other fields are invariant.
# --------------------------------------------------------------------------- #
def test_observation_omits_comms_when_disabled():
    obs = build_observation(
        local_time_s=1.0,
        sensor_data={"DepthSensor": [3.0]},
        beacon_packets=[],
        official_mode=True,
        comms_inbox=None,
    )
    assert "comms" not in obs


def test_enabling_comms_path_only_adds_comms_field():
    common = dict(
        local_time_s=1.0,
        sensor_data={"DepthSensor": [3.0]},
        beacon_packets=[{"beacon_id": "B01", "range_m": 5.0}],
        official_mode=True,
    )
    without = build_observation(**common, comms_inbox=None)
    with_comms = build_observation(**common, comms_inbox=[])
    # The only difference is the presence of the comms inbox.
    assert "comms" not in without
    assert with_comms["comms"] == {"inbox": []}
    for key in ("local_time_s", "sensors", "beacons"):
        assert without[key] == with_comms[key]


# --------------------------------------------------------------------------- #
# Runner: no comms field ever reaches a controller when comms is disabled.
# --------------------------------------------------------------------------- #
def test_single_rover_controller_never_sees_comms():
    controllers, _ = _run([{"id": "solo", "position": (0.0, 0.0, -3.5)}])
    seen = controllers["solo"].observations
    assert seen  # the controller actually ran
    assert all(record["has_comms"] is False for record in seen)


def test_single_rover_run_is_deterministic():
    _, summary_a = _run([{"id": "solo", "position": (0.0, 0.0, -3.5)}])
    _, summary_b = _run([{"id": "solo", "position": (0.0, 0.0, -3.5)}])
    assert _row(summary_a, "solo") == _row(summary_b, "solo")
    assert "team_summary" not in summary_a  # single rover -> no team aggregation


# --------------------------------------------------------------------------- #
# Fleet aggregation is additive: it does not change the first rover.
# --------------------------------------------------------------------------- #
def test_second_rover_and_team_aggregation_do_not_change_first_rover():
    solo_controllers, solo_summary = _run([{"id": "bluerov2_01", "position": (0.0, 0.0, -3.5)}])
    fleet_controllers, fleet_summary = _run(
        [
            {"id": "bluerov2_01", "position": (0.0, 0.0, -3.5)},
            {"id": "bluerov2_02", "position": (0.0, 12.0, -3.5)},  # spaced apart
        ]
    )
    # (c, d) identical referee score for the first rover.
    assert _row(solo_summary, "bluerov2_01") == _row(fleet_summary, "bluerov2_01")
    # (a, e) identical observation stream and commands for the first rover.
    assert (
        solo_controllers["bluerov2_01"].observations
        == fleet_controllers["bluerov2_01"].observations
    )
    assert (
        solo_controllers["bluerov2_01"].commands
        == fleet_controllers["bluerov2_01"].commands
    )
    # The fleet run does compute team aggregation; the solo run does not.
    assert "team_summary" in fleet_summary
    assert "team_summary" not in solo_summary
    # And the first rover still never sees a comms field with comms disabled.
    assert all(rec["has_comms"] is False for rec in fleet_controllers["bluerov2_01"].observations)
