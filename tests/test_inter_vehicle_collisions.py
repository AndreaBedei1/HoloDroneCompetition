from __future__ import annotations

from pathlib import Path

from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.referee.race_state import ParticipantStatus
from marine_race_arena.referee.referee import Referee


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


def test_no_inter_vehicle_event_when_rovers_are_far_apart() -> None:
    referee = _fleet_referee()

    events = referee.detect_inter_vehicle_collisions(
        1.0,
        {
            "rover_a": (0.0, 0.0, -4.0),
            "rover_b": (5.0, 0.0, -4.0),
        },
    )

    assert events == []
    assert referee.inter_vehicle_collision_events == 0


def test_one_event_when_two_released_rovers_are_within_threshold() -> None:
    referee = _fleet_referee()

    events = referee.detect_inter_vehicle_collisions(
        1.0,
        {
            "rover_a": (0.0, 0.0, -4.0),
            "rover_b": (0.4, 0.0, -4.0),
        },
    )

    assert len(events) == 1
    assert events[0]["event"] == "inter_vehicle_collision"
    assert events[0]["participant_a"] == "rover_a"
    assert events[0]["participant_b"] == "rover_b"
    assert referee.inter_vehicle_collision_events == 1
    assert referee.states["rover_a"].involved_inter_vehicle_collisions == 1
    assert referee.states["rover_b"].involved_inter_vehicle_collisions == 1


def test_waiting_rover_is_ignored_by_inter_vehicle_detector() -> None:
    referee = _fleet_referee(start_delays={"rover_a": 0.0, "rover_b": 10.0})

    events = referee.detect_inter_vehicle_collisions(
        1.0,
        {
            "rover_a": (0.0, 0.0, -4.0),
            "rover_b": (0.4, 0.0, -4.0),
        },
    )

    assert referee.states["rover_b"].status == ParticipantStatus.NOT_STARTED
    assert events == []
    assert referee.inter_vehicle_collision_events == 0


def test_cooldown_and_hysteresis_prevent_repeated_inter_vehicle_events() -> None:
    referee = _fleet_referee(cooldown_s=1.0, release_threshold_m=1.0)
    close_positions = {"rover_a": (0.0, 0.0, -4.0), "rover_b": (0.4, 0.0, -4.0)}
    far_positions = {"rover_a": (0.0, 0.0, -4.0), "rover_b": (1.5, 0.0, -4.0)}

    first_events = referee.detect_inter_vehicle_collisions(1.0, close_positions)
    cooldown_events = referee.detect_inter_vehicle_collisions(1.5, close_positions)
    stuck_together_events = referee.detect_inter_vehicle_collisions(3.0, close_positions)
    separated_events = referee.detect_inter_vehicle_collisions(3.1, far_positions)
    second_events = referee.detect_inter_vehicle_collisions(4.2, close_positions)

    assert len(first_events) == 1
    assert cooldown_events == []
    assert stuck_together_events == []
    assert separated_events == []
    assert len(second_events) == 1
    assert referee.inter_vehicle_collision_events == 2


def test_inter_vehicle_collision_counts_once_at_team_level_not_per_rover() -> None:
    referee = _fleet_referee(mode="penalize")

    referee.detect_inter_vehicle_collisions(
        1.0,
        {
            "rover_a": (0.0, 0.0, -4.0),
            "rover_b": (0.4, 0.0, -4.0),
        },
    )
    summary = referee.summary()
    team_summary = summary["team_summary"]

    assert team_summary["total_inter_vehicle_collisions"] == 1
    assert team_summary["total_collisions"] == 1
    assert referee.states["rover_a"].collision_events == 0
    assert referee.states["rover_b"].collision_events == 0
    assert referee.states["rover_a"].involved_inter_vehicle_collisions == 1
    assert referee.states["rover_b"].involved_inter_vehicle_collisions == 1


def test_team_summary_aggregates_penalties_and_finish_time() -> None:
    referee = _fleet_referee(mode="penalize")
    referee.detect_inter_vehicle_collisions(
        1.0,
        {
            "rover_a": (0.0, 0.0, -4.0),
            "rover_b": (0.4, 0.0, -4.0),
        },
    )
    state_a = referee.states["rover_a"]
    state_b = referee.states["rover_b"]
    state_a.collision_events = 2
    state_a.obstacle_collision_events = 1
    state_a.penalties_s += 10.0
    state_a.status = ParticipantStatus.FINISHED
    state_b.status = ParticipantStatus.FINISHED
    state_a.release_time_s = 0.0
    state_b.release_time_s = 5.0
    state_a.official_finish_time = 100.0
    state_b.official_finish_time = 120.0

    team_summary = referee.summary()["team_summary"]

    assert team_summary["rover_count"] == 2
    assert team_summary["all_rovers_finished"] is True
    assert team_summary["team_start_time_s"] == 0.0
    assert team_summary["team_finish_time_s"] == 120.0
    assert team_summary["team_elapsed_time_s"] == 120.0
    assert team_summary["total_gate_collisions"] == 1
    assert team_summary["total_obstacle_collisions"] == 1
    assert team_summary["total_inter_vehicle_collisions"] == 1
    assert team_summary["total_collisions"] == 3
    assert team_summary["total_penalties_s"] == 15.0
    assert team_summary["team_penalized_time_s"] == 135.0


def test_per_rover_summaries_remain_available_in_fleet_mode() -> None:
    summary = _fleet_referee().summary()

    assert len(summary["participants"]) == 2
    assert summary["ranking"] == ["rover_a", "rover_b"]
    assert "team_summary" in summary
    assert all("involved_inter_vehicle_collisions" in row for row in summary["participants"])


def test_single_rover_summary_has_no_team_scoring_regression() -> None:
    config = load_track_config(TRACK_DIR / "marine_race_horseshoe_bay.json")
    arena = ArenaBuilder(config).build()
    referee = Referee(config, arena.gate_map, arena.bounds)
    referee.register_participants(["solo"])
    referee.start_race(0.0)

    summary = referee.summary()

    assert "team_summary" not in summary
    assert len(summary["participants"]) == 1
    assert "involved_inter_vehicle_collisions" not in summary["participants"][0]


def _fleet_referee(
    *,
    mode: str = "diagnostic",
    cooldown_s: float = 1.0,
    release_threshold_m: float | None = None,
    start_delays: dict[str, float] | None = None,
) -> Referee:
    config = load_track_config(TRACK_DIR / "marine_race_horseshoe_bay.json")
    arena = ArenaBuilder(config).build()
    referee = Referee(
        config,
        arena.gate_map,
        arena.bounds,
        inter_vehicle_collision_mode=mode,
        inter_vehicle_collision_xy_threshold_m=0.8,
        inter_vehicle_collision_z_threshold_m=0.75,
        inter_vehicle_collision_release_threshold_m=release_threshold_m,
        inter_vehicle_collision_cooldown_s=cooldown_s,
        team_id="test_team",
    )
    referee.register_participants(["rover_a", "rover_b"])
    referee.start_race(0.0, start_delays=start_delays or {"rover_a": 0.0, "rover_b": 0.0})
    return referee
