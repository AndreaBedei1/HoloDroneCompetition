"""Race state dataclasses and status names."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from marine_race_arena.config.schema import Vector3


class ParticipantStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    DNF = "DNF"
    DSQ = "DSQ"
    TIMEOUT = "TIMEOUT"
    CONTROLLER_ERROR = "CONTROLLER_ERROR"


@dataclass
class ParticipantRaceState:
    participant_id: str
    status: ParticipantStatus = ParticipantStatus.NOT_STARTED
    current_lap: int = 1
    expected_gate_index: int = 0
    valid_gate_crossings: int = 0
    missed_gate_attempts: int = 0
    wrong_direction_crossings: int = 0
    collision_events: int = 0
    out_of_bounds_events: int = 0
    penalties_s: float = 0.0
    official_start_time: Optional[float] = None
    official_finish_time: Optional[float] = None
    green_start_time: Optional[float] = None
    green_to_finish_time_s: Optional[float] = None
    last_position: Optional[Vector3] = None
    last_motion_time: Optional[float] = None
    last_update_time: Optional[float] = None
    stuck_accumulator_s: float = 0.0
    controller_error: Optional[str] = None
    last_event: Dict[str, object] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            ParticipantStatus.FINISHED,
            ParticipantStatus.DNF,
            ParticipantStatus.DSQ,
            ParticipantStatus.TIMEOUT,
            ParticipantStatus.CONTROLLER_ERROR,
        }

    @property
    def official_time_s(self) -> Optional[float]:
        if self.official_start_time is None or self.official_finish_time is None:
            return None
        return self.official_finish_time - self.official_start_time
