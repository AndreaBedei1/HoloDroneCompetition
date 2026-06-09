"""Participant runtime state wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from marine_race_arena.config.schema import ParticipantConfig, Vector3


@dataclass
class RaceParticipant:
    config: ParticipantConfig
    controller: Any
    position: Vector3
    rotation_rpy_deg: Vector3

    @property
    def id(self) -> str:
        return self.config.id

