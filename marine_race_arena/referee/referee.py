"""Race referee for gate validation, timing, DNF rules, and summaries."""

from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable, List, Mapping, Optional

from marine_race_arena.arena.bounds import ArenaBounds
from marine_race_arena.arena.gate import Gate
from marine_race_arena.config.schema import TrackConfig, Vector3
from marine_race_arena.referee.gate_validation import validate_gate_crossing
from marine_race_arena.referee.logger import RaceLogger
from marine_race_arena.referee.race_state import ParticipantRaceState, ParticipantStatus
from marine_race_arena.referee.scoring import penalized_time_s, rank_participants

INTER_VEHICLE_COLLISION_OFF = "off"
INTER_VEHICLE_COLLISION_DIAGNOSTIC = "diagnostic"
INTER_VEHICLE_COLLISION_PENALIZE = "penalize"
INTER_VEHICLE_COLLISION_MODES = (
    INTER_VEHICLE_COLLISION_OFF,
    INTER_VEHICLE_COLLISION_DIAGNOSTIC,
    INTER_VEHICLE_COLLISION_PENALIZE,
)


class Referee:
    def __init__(
        self,
        config: TrackConfig,
        gate_map: Dict[str, Gate],
        bounds: ArenaBounds,
        logger: Optional[RaceLogger] = None,
        inter_vehicle_collision_mode: str = INTER_VEHICLE_COLLISION_OFF,
        inter_vehicle_collision_xy_threshold_m: float = 0.8,
        inter_vehicle_collision_z_threshold_m: float = 0.75,
        inter_vehicle_collision_release_threshold_m: Optional[float] = None,
        inter_vehicle_collision_cooldown_s: float = 1.0,
        team_id: str = "fleet_01",
    ):
        self.config = config
        self.gate_map = gate_map
        self.bounds = bounds
        self.logger = logger
        self.gate_sequence = list(config.track.gate_sequence)
        self.states: Dict[str, ParticipantRaceState] = {}
        self.latest_positions: Dict[str, Vector3] = {}
        self._green_time: Optional[float] = None
        self.vehicle_clearance_margin_m = max(
            0.0,
            float(config.referee.gate_validation.get("vehicle_clearance_margin_m", 0.0)),
        )
        self.team_id = team_id
        if inter_vehicle_collision_mode not in INTER_VEHICLE_COLLISION_MODES:
            raise ValueError(
                "inter_vehicle_collision_mode must be one of "
                f"{', '.join(INTER_VEHICLE_COLLISION_MODES)}."
            )
        self.inter_vehicle_collision_mode = inter_vehicle_collision_mode
        self.inter_vehicle_collision_xy_threshold_m = max(
            0.0, float(inter_vehicle_collision_xy_threshold_m)
        )
        self.inter_vehicle_collision_z_threshold_m = max(
            0.0, float(inter_vehicle_collision_z_threshold_m)
        )
        if inter_vehicle_collision_release_threshold_m is None:
            inter_vehicle_collision_release_threshold_m = max(
                self.inter_vehicle_collision_xy_threshold_m + 0.25,
                self.inter_vehicle_collision_xy_threshold_m * 1.25,
            )
        self.inter_vehicle_collision_release_threshold_m = max(
            self.inter_vehicle_collision_xy_threshold_m,
            float(inter_vehicle_collision_release_threshold_m),
        )
        self.inter_vehicle_collision_cooldown_s = max(0.0, float(inter_vehicle_collision_cooldown_s))
        self.inter_vehicle_collision_events = 0
        self.inter_vehicle_collision_penalties_s = 0.0
        self._inter_vehicle_pair_last_event_time: Dict[tuple[str, str], float] = {}
        self._inter_vehicle_pair_active: set[tuple[str, str]] = set()

    def register_participants(self, participant_ids: Iterable[str]) -> None:
        for participant_id in participant_ids:
            self.states[participant_id] = ParticipantRaceState(participant_id=participant_id)

    def start_race(self, time_s: float, start_delays: Optional[Mapping[str, float]] = None) -> None:
        self._green_time = time_s
        log_releases = start_delays is not None
        self._log("race_start", time_s, race_name=self.config.race.name)
        for state in self.states.values():
            start_delay_s = max(0.0, float((start_delays or {}).get(state.participant_id, 0.0)))
            state.start_delay_s = start_delay_s
            if start_delay_s <= 0.0:
                self._release_state(state, time_s, log_event=log_releases)
            else:
                state.status = ParticipantStatus.NOT_STARTED
                state.green_start_time = None
                state.release_time_s = None
                state.last_motion_time = None
                state.last_update_time = None

    def release_participant(self, participant_id: str, time_s: float) -> None:
        state = self.states.get(participant_id)
        if state is None or state.is_terminal or state.status == ParticipantStatus.RUNNING:
            return
        self._release_state(state, time_s, log_event=True)

    def manual_stop(self, participant_ids: Iterable[str], time_s: float) -> None:
        for participant_id in participant_ids:
            state = self.states.get(participant_id)
            if state is None or state.is_terminal:
                continue
            state.status = ParticipantStatus.MANUAL_STOP
            state.last_update_time = time_s
            self._log("manual_stop", time_s, participant_id, reason="manual_stop")

    def gate_timeout_stuck(self, participant_id: str, time_s: float, timeout_s: float) -> None:
        state = self.states.get(participant_id)
        if state is None or state.is_terminal or state.status == ParticipantStatus.NOT_STARTED:
            return
        expected_gate_id = self.expected_gate_id(participant_id)
        state.status = ParticipantStatus.STUCK
        state.stuck_events += 1
        state.last_update_time = time_s
        self._log(
            "stuck",
            time_s,
            participant_id,
            reason="gate_timeout",
            duration_s=timeout_s,
            expected_gate_id=expected_gate_id,
        )
        self._log("dnf", time_s, participant_id, reason="gate_timeout_stuck")

    def update(
        self,
        participant_id: str,
        previous_position: Vector3,
        current_position: Vector3,
        time_s: float,
        collision: bool = False,
        obstacle_collisions: Optional[Iterable[Dict[str, object]]] = None,
        controller_error: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        state = self.states[participant_id]
        self.latest_positions[participant_id] = current_position
        events: List[Dict[str, object]] = []
        if state.is_terminal:
            return events
        if state.status == ParticipantStatus.NOT_STARTED:
            return events

        if controller_error is not None:
            state.controller_error = controller_error
            state.status = ParticipantStatus.CONTROLLER_ERROR
            event = {"event": "controller_error", "message": controller_error}
            events.append(event)
            self._log("controller_error", time_s, participant_id, message=controller_error)
            return events

        bounds_reason = self.bounds.violation_reason(current_position)
        if bounds_reason is not None:
            self._handle_out_of_bounds(state, bounds_reason, current_position, time_s, events)

        if collision:
            self._handle_collision(state, current_position, time_s, events)
        for obstacle_collision in obstacle_collisions or []:
            self._handle_obstacle_collision(state, obstacle_collision, current_position, time_s, events)

        self._update_stuck_state(state, previous_position, current_position, time_s, events)
        if state.is_terminal:
            return events

        if self._duration_exceeded(state, time_s):
            state.status = ParticipantStatus.TIMEOUT
            events.append({"event": "timeout"})
            self._log("dnf", time_s, participant_id, reason="timeout")
            return events

        gate_events = self._validate_gate_progression(
            state, previous_position, current_position, time_s
        )
        events.extend(gate_events)
        return events

    def expected_gate_id(self, participant_id: str) -> str:
        state = self.states[participant_id]
        return self.gate_sequence[state.expected_gate_index]

    def race_progress(self, participant_id: str) -> Dict[str, object]:
        state = self.states[participant_id]
        return {
            "status": state.status.value,
            "lap": state.current_lap,
            "laps": self.config.race.laps,
            "completed_gates": state.valid_gate_crossings,
            "target_gate_id": self.expected_gate_id(participant_id),
            "target_sequence_index": state.expected_gate_index,
            "official_time_started": state.official_start_time is not None,
        }

    def detect_inter_vehicle_collisions(
        self,
        time_s: float,
        positions: Optional[Mapping[str, Vector3]] = None,
    ) -> List[Dict[str, object]]:
        if self.inter_vehicle_collision_mode == INTER_VEHICLE_COLLISION_OFF:
            return []
        running_ids = sorted(
            participant_id
            for participant_id, state in self.states.items()
            if state.status == ParticipantStatus.RUNNING
        )
        if len(running_ids) < 2:
            return []

        current_positions = positions if positions is not None else self.latest_positions
        events: List[Dict[str, object]] = []
        for participant_a, participant_b in itertools.combinations(running_ids, 2):
            position_a = current_positions.get(participant_a)
            position_b = current_positions.get(participant_b)
            if position_a is None or position_b is None:
                continue
            pair = _ordered_pair(participant_a, participant_b)
            horizontal_distance_m = _horizontal_distance(position_a, position_b)
            vertical_distance_m = abs(position_a[2] - position_b[2])
            distance_3d_m = _distance(position_a, position_b)
            within_threshold = (
                horizontal_distance_m <= self.inter_vehicle_collision_xy_threshold_m
                and vertical_distance_m <= self.inter_vehicle_collision_z_threshold_m
            )
            if not within_threshold:
                if (
                    pair in self._inter_vehicle_pair_active
                    and horizontal_distance_m >= self.inter_vehicle_collision_release_threshold_m
                ) or vertical_distance_m > self.inter_vehicle_collision_z_threshold_m:
                    self._inter_vehicle_pair_active.discard(pair)
                continue

            if pair in self._inter_vehicle_pair_active:
                continue
            last_event_time = self._inter_vehicle_pair_last_event_time.get(pair)
            if not _cooldown_elapsed(last_event_time, time_s, self.inter_vehicle_collision_cooldown_s):
                continue

            event = self._record_inter_vehicle_collision(
                participant_a=participant_a,
                participant_b=participant_b,
                position_a=position_a,
                position_b=position_b,
                horizontal_distance_m=horizontal_distance_m,
                vertical_distance_m=vertical_distance_m,
                distance_3d_m=distance_3d_m,
                time_s=time_s,
            )
            self._inter_vehicle_pair_active.add(pair)
            self._inter_vehicle_pair_last_event_time[pair] = time_s
            events.append(event)
        return events

    def summary(self) -> Dict[str, object]:
        ranking = rank_participants(
            self.states.values(), self.gate_sequence, self.gate_map, self.latest_positions
        )
        fleet_mode = len(self.states) > 1
        participant_summaries = []
        for rank, state in enumerate(ranking, start=1):
            row = {
                "rank": rank,
                "participant_id": state.participant_id,
                "status": state.status.value,
                "start_delay_s": state.start_delay_s,
                "release_time_s": state.release_time_s,
                "official_time_s": state.official_time_s,
                "green_to_finish_time_s": state.green_to_finish_time_s,
                "penalties_s": state.penalties_s,
                "penalized_time_s": penalized_time_s(state),
                "completed_gates": state.valid_gate_crossings,
                "lap": state.current_lap,
                "collisions": state.collision_events,
                "obstacle_collisions": state.obstacle_collision_events,
                "wrong_direction_crossings": state.wrong_direction_crossings,
                "missed_gate_attempts": state.missed_gate_attempts,
                "out_of_bounds_events": state.out_of_bounds_events,
                "stuck_events": state.stuck_events,
            }
            if fleet_mode:
                row["involved_inter_vehicle_collisions"] = state.involved_inter_vehicle_collisions
            participant_summaries.append(row)
        summary = {
            "race_name": self.config.race.name,
            "environment": self.config.world.map,
            "timing_mode": self.config.race.timing_mode,
            "participants": participant_summaries,
            "ranking": [state.participant_id for state in ranking],
        }
        if fleet_mode:
            summary["team_summary"] = self._team_summary()
        return summary

    def _release_state(
        self,
        state: ParticipantRaceState,
        time_s: float,
        log_event: bool,
    ) -> None:
        state.status = ParticipantStatus.RUNNING
        state.green_start_time = time_s
        state.release_time_s = time_s
        state.last_motion_time = time_s
        state.last_update_time = time_s
        state.stuck_accumulator_s = 0.0
        state.stuck_penalty_active = False
        if log_event:
            self._log(
                "participant_released",
                time_s,
                state.participant_id,
                start_delay_s=state.start_delay_s,
                release_time_s=time_s,
            )

    def _validate_gate_progression(
        self,
        state: ParticipantRaceState,
        previous_position: Vector3,
        current_position: Vector3,
        time_s: float,
    ) -> List[Dict[str, object]]:
        events: List[Dict[str, object]] = []
        expected_gate_id = self.gate_sequence[state.expected_gate_index]
        expected_gate = self.gate_map[expected_gate_id]
        result = validate_gate_crossing(
            expected_gate,
            previous_position,
            current_position,
            clearance_margin_m=self.vehicle_clearance_margin_m,
        )

        if result.valid:
            self._handle_valid_gate(state, expected_gate, result.intersection, time_s, events)
            return events

        if result.reason == "wrong_direction":
            state.wrong_direction_crossings += 1
            events.append({"event": "wrong_direction", "gate_id": expected_gate.id})
            self._log("wrong_direction", time_s, state.participant_id, gate_id=expected_gate.id)
            return events

        for gate_id, gate in self.gate_map.items():
            if gate_id == expected_gate_id:
                continue
            if _crossed_gate_aperture(
                gate,
                previous_position,
                current_position,
                self.vehicle_clearance_margin_m,
            ):
                state.missed_gate_attempts += 1
                events.append({"event": "missed_gate", "expected_gate_id": expected_gate_id, "crossed_gate_id": gate_id})
                self._log(
                    "penalty",
                    time_s,
                    state.participant_id,
                    reason="missed_gate",
                    expected_gate_id=expected_gate_id,
                    crossed_gate_id=gate_id,
                )
                if bool(self.config.referee.penalties.get("missed_gate_dnf", True)):
                    state.status = ParticipantStatus.DNF
                    self._log("dnf", time_s, state.participant_id, reason="missed_gate")
                break
        return events

    def _handle_valid_gate(
        self,
        state: ParticipantRaceState,
        gate: Gate,
        intersection: Optional[Vector3],
        time_s: float,
        events: List[Dict[str, object]],
    ) -> None:
        if (
            self.config.race.timing_mode == "first_gate_to_last_gate"
            and state.official_start_time is None
            and state.current_lap == 1
            and state.expected_gate_index == 0
        ):
            state.official_start_time = time_s

        state.valid_gate_crossings += 1
        event = {
            "event": "gate_passed",
            "gate_id": gate.id,
            "lap": state.current_lap,
            "sequence_index": state.expected_gate_index,
            "intersection": intersection,
        }
        events.append(event)
        self._log(
            "gate_passed",
            time_s,
            state.participant_id,
            gate_id=gate.id,
            lap=state.current_lap,
            sequence_index=state.expected_gate_index,
            intersection=intersection,
        )

        final_gate_in_lap = state.expected_gate_index == len(self.gate_sequence) - 1
        final_lap = state.current_lap == self.config.race.laps
        if final_gate_in_lap and final_lap:
            self._finish_participant(state, time_s)
            return
        if final_gate_in_lap:
            self._log("lap_completed", time_s, state.participant_id, lap=state.current_lap)
            state.current_lap += 1
            state.expected_gate_index = 0
            return
        state.expected_gate_index += 1

    def _finish_participant(self, state: ParticipantRaceState, time_s: float) -> None:
        if self.config.race.timing_mode == "green_to_finish" and state.green_start_time is not None:
            state.official_start_time = state.green_start_time
        state.official_finish_time = time_s
        if state.green_start_time is not None:
            state.green_to_finish_time_s = time_s - state.green_start_time
        state.status = ParticipantStatus.FINISHED
        self._log(
            "race_finish",
            time_s,
            state.participant_id,
            official_time_s=state.official_time_s,
            green_to_finish_time_s=state.green_to_finish_time_s,
            penalties_s=state.penalties_s,
        )

    def _handle_collision(
        self,
        state: ParticipantRaceState,
        current_position: Vector3,
        time_s: float,
        events: List[Dict[str, object]],
    ) -> None:
        cooldown_s = float(self.config.referee.gate_validation.get("collision_penalty_cooldown_s", 1.0))
        if not _cooldown_elapsed(state.last_collision_penalty_time, time_s, cooldown_s):
            return
        penalty = float(self.config.referee.penalties.get("minor_collision_s", 5.0))
        state.collision_events += 1
        state.penalties_s += penalty
        state.last_collision_penalty_time = time_s
        events.append({"event": "collision", "penalty_s": penalty, "position": current_position})
        self._log("collision", time_s, state.participant_id, penalty_s=penalty, position=current_position)

    def _handle_obstacle_collision(
        self,
        state: ParticipantRaceState,
        obstacle_collision: Dict[str, object],
        current_position: Vector3,
        time_s: float,
        events: List[Dict[str, object]],
    ) -> None:
        cooldown_s = float(self.config.referee.gate_validation.get("collision_penalty_cooldown_s", 1.0))
        if not _cooldown_elapsed(state.last_obstacle_collision_penalty_time, time_s, cooldown_s):
            return
        penalty = float(obstacle_collision.get("penalty_s", self.config.referee.penalties.get("minor_collision_s", 5.0)))
        obstacle_id = str(obstacle_collision.get("obstacle_id", "unknown_obstacle"))
        state.collision_events += 1
        state.obstacle_collision_events += 1
        state.penalties_s += penalty
        state.last_obstacle_collision_penalty_time = time_s
        event = {
            "event": "obstacle_collision",
            "obstacle_id": obstacle_id,
            "penalty_s": penalty,
            "position": obstacle_collision.get("position", current_position),
        }
        events.append(event)
        self._log(
            "obstacle_collision",
            time_s,
            state.participant_id,
            obstacle_id=obstacle_id,
            penalty_s=penalty,
            position=event["position"],
        )

    def _record_inter_vehicle_collision(
        self,
        *,
        participant_a: str,
        participant_b: str,
        position_a: Vector3,
        position_b: Vector3,
        horizontal_distance_m: float,
        vertical_distance_m: float,
        distance_3d_m: float,
        time_s: float,
    ) -> Dict[str, object]:
        state_a = self.states[participant_a]
        state_b = self.states[participant_b]
        penalty_s = 0.0
        if self.inter_vehicle_collision_mode == INTER_VEHICLE_COLLISION_PENALIZE:
            penalty_s = float(self.config.referee.penalties.get("minor_collision_s", 5.0))
            self.inter_vehicle_collision_penalties_s += penalty_s
        self.inter_vehicle_collision_events += 1
        state_a.involved_inter_vehicle_collisions += 1
        state_b.involved_inter_vehicle_collisions += 1
        event = {
            "event": "inter_vehicle_collision",
            "team_id": self.team_id,
            "participant_a": participant_a,
            "participant_b": participant_b,
            "time_s": time_s,
            "horizontal_distance_m": horizontal_distance_m,
            "vertical_distance_m": vertical_distance_m,
            "distance_3d_m": distance_3d_m,
            "position_a": position_a,
            "position_b": position_b,
            "completed_gates_a": state_a.valid_gate_crossings,
            "completed_gates_b": state_b.valid_gate_crossings,
            "target_gate_a": self._target_gate_id_or_none(state_a),
            "target_gate_b": self._target_gate_id_or_none(state_b),
            "mode": self.inter_vehicle_collision_mode,
            "penalty_s": penalty_s,
        }
        self._log(
            "inter_vehicle_collision",
            time_s,
            None,
            team_id=self.team_id,
            participant_a=participant_a,
            participant_b=participant_b,
            horizontal_distance_m=horizontal_distance_m,
            vertical_distance_m=vertical_distance_m,
            distance_3d_m=distance_3d_m,
            position_a=position_a,
            position_b=position_b,
            completed_gates_a=state_a.valid_gate_crossings,
            completed_gates_b=state_b.valid_gate_crossings,
            target_gate_a=event["target_gate_a"],
            target_gate_b=event["target_gate_b"],
            mode=self.inter_vehicle_collision_mode,
            penalty_s=penalty_s,
        )
        return event

    def _target_gate_id_or_none(self, state: ParticipantRaceState) -> Optional[str]:
        if state.is_terminal or not self.gate_sequence:
            return None
        return self.gate_sequence[state.expected_gate_index]

    def _team_summary(self) -> Dict[str, object]:
        states = list(self.states.values())
        rover_count = len(states)
        expected_gates_per_rover = int(self.config.race.laps) * len(self.gate_sequence)
        expected_total_gates = expected_gates_per_rover * rover_count
        total_completed_gates = sum(state.valid_gate_crossings for state in states)
        all_rovers_finished = all(state.status == ParticipantStatus.FINISHED for state in states)
        release_times = [state.release_time_s for state in states if state.release_time_s is not None]
        team_start_time_s = min(release_times) if release_times else self._green_time
        finish_times = [
            state.official_finish_time for state in states if state.official_finish_time is not None
        ]
        team_finish_time_s = max(finish_times) if all_rovers_finished and finish_times else None
        team_elapsed_time_s = None
        if team_start_time_s is not None and team_finish_time_s is not None:
            team_elapsed_time_s = team_finish_time_s - team_start_time_s
        total_obstacle_collisions = sum(state.obstacle_collision_events for state in states)
        total_gate_collisions = sum(
            max(0, state.collision_events - state.obstacle_collision_events)
            for state in states
        )
        total_inter_vehicle_collisions = self.inter_vehicle_collision_events
        total_collisions = (
            total_gate_collisions
            + total_obstacle_collisions
            + total_inter_vehicle_collisions
        )
        total_penalties_s = (
            sum(state.penalties_s for state in states) + self.inter_vehicle_collision_penalties_s
        )
        team_penalized_time_s = (
            team_elapsed_time_s + total_penalties_s
            if team_elapsed_time_s is not None
            else None
        )
        return {
            "team_id": self.team_id,
            "rover_count": rover_count,
            "expected_total_gates": expected_total_gates,
            "total_completed_gates": total_completed_gates,
            "all_rovers_finished": all_rovers_finished,
            "team_start_time_s": team_start_time_s,
            "team_finish_time_s": team_finish_time_s,
            "team_elapsed_time_s": team_elapsed_time_s,
            "total_gate_collisions": total_gate_collisions,
            "total_obstacle_collisions": total_obstacle_collisions,
            "total_inter_vehicle_collisions": total_inter_vehicle_collisions,
            "total_collisions": total_collisions,
            "total_penalties_s": total_penalties_s,
            "team_penalized_time_s": team_penalized_time_s,
            "inter_vehicle_collision_mode": self.inter_vehicle_collision_mode,
        }

    def _handle_out_of_bounds(
        self,
        state: ParticipantRaceState,
        reason: str,
        current_position: Vector3,
        time_s: float,
        events: List[Dict[str, object]],
    ) -> None:
        cooldown_s = float(self.config.referee.gate_validation.get("out_of_bounds_penalty_cooldown_s", 1.0))
        if not _cooldown_elapsed(state.last_out_of_bounds_penalty_time, time_s, cooldown_s):
            return
        penalty = float(self.config.referee.penalties.get("out_of_bounds_s", 10.0))
        state.out_of_bounds_events += 1
        state.penalties_s += penalty
        state.last_out_of_bounds_penalty_time = time_s
        events.append(
            {
                "event": "out_of_bounds",
                "reason": reason,
                "penalty_s": penalty,
                "position": current_position,
            }
        )
        self._log(
            "out_of_bounds",
            time_s,
            state.participant_id,
            reason=reason,
            penalty_s=penalty,
            position=current_position,
        )

    def _update_stuck_state(
        self,
        state: ParticipantRaceState,
        previous_position: Vector3,
        current_position: Vector3,
        time_s: float,
        events: List[Dict[str, object]],
    ) -> None:
        threshold = float(self.config.referee.gate_validation.get("stuck_speed_threshold_m_s", 0.02))
        timeout_s = float(self.config.referee.gate_validation.get("stuck_timeout_s", 30.0))
        previous_update_time = state.last_update_time if state.last_update_time is not None else time_s
        dt = max(0.0, time_s - previous_update_time)
        state.last_update_time = time_s
        distance = _distance(previous_position, current_position)
        speed = distance / dt if dt > 0 else 0.0
        if speed < threshold:
            state.stuck_accumulator_s += dt
        else:
            state.stuck_accumulator_s = 0.0
            state.stuck_penalty_active = False
            state.last_motion_time = time_s
        if state.stuck_accumulator_s >= timeout_s and not state.stuck_penalty_active:
            penalty = float(self.config.referee.penalties.get("stuck_s", 15.0))
            state.stuck_events += 1
            state.penalties_s += penalty
            state.stuck_penalty_active = True
            events.append(
                {
                    "event": "stuck",
                    "duration_s": state.stuck_accumulator_s,
                    "penalty_s": penalty,
                }
            )
            self._log(
                "stuck",
                time_s,
                state.participant_id,
                duration_s=state.stuck_accumulator_s,
                penalty_s=penalty,
            )

    def _duration_exceeded(self, state: ParticipantRaceState, time_s: float) -> bool:
        if not bool(self.config.referee.gate_validation.get("timeout_enabled", False)):
            return False
        if state.green_start_time is None:
            return False
        return (time_s - state.green_start_time) > self.config.race.max_duration_s

    def _log(
        self,
        event_type: str,
        time_s: float,
        participant_id: Optional[str] = None,
        **payload: object,
    ) -> None:
        if self.logger is not None:
            self.logger.log_event(event_type, time_s, participant_id, **payload)


def _distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _horizontal_distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _ordered_pair(participant_a: str, participant_b: str) -> tuple[str, str]:
    ordered = sorted((participant_a, participant_b))
    return (ordered[0], ordered[1])


def _cooldown_elapsed(previous_time_s: Optional[float], time_s: float, cooldown_s: float) -> bool:
    if previous_time_s is None:
        return True
    return (time_s - previous_time_s) >= max(0.0, cooldown_s)


def _crossed_gate_aperture(
    gate: Gate,
    previous_position: Vector3,
    current_position: Vector3,
    clearance_margin_m: float,
) -> bool:
    if not gate.crossed_between(previous_position, current_position):
        return False
    intersection = gate.intersection_point(previous_position, current_position)
    if intersection is None:
        return False
    return gate.is_point_inside_aperture(intersection, margin_m=clearance_margin_m)
