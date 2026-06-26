"""Race referee for gate validation, timing, DNF rules, and summaries."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional

from marine_race_arena.arena.bounds import ArenaBounds
from marine_race_arena.arena.gate import Gate
from marine_race_arena.config.schema import TrackConfig, Vector3
from marine_race_arena.referee.gate_validation import validate_gate_crossing
from marine_race_arena.referee.logger import RaceLogger
from marine_race_arena.referee.race_state import ParticipantRaceState, ParticipantStatus
from marine_race_arena.referee.scoring import penalized_time_s, rank_participants


class Referee:
    def __init__(
        self,
        config: TrackConfig,
        gate_map: Dict[str, Gate],
        bounds: ArenaBounds,
        logger: Optional[RaceLogger] = None,
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

    def summary(self) -> Dict[str, object]:
        ranking = rank_participants(
            self.states.values(), self.gate_sequence, self.gate_map, self.latest_positions
        )
        participant_summaries = []
        for rank, state in enumerate(ranking, start=1):
            participant_summaries.append(
                {
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
            )
        return {
            "race_name": self.config.race.name,
            "environment": self.config.world.map,
            "timing_mode": self.config.race.timing_mode,
            "participants": participant_summaries,
            "ranking": [state.participant_id for state in ranking],
        }

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
