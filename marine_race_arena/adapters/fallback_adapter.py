"""Fallback point-vehicle adapter used when no simulator is requested.

The fallback adapter is an engine-free debug/test substrate. It synthesizes
the same onboard sensor set the HoloOcean adapter exposes (DepthSensor,
IMUSensor, DVLSensor, CollisionSensor and an optional synthetic FrontCamera)
so the onboard-only controller contract can be exercised without the engine.
It never exposes ground-truth pose, world-frame velocity, or environment
current state to controllers.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, Mapping, Optional

from marine_race_arena.adapters.base import AdapterParticipantState, BaseRaceAdapter, RaceAdapterError
from marine_race_arena.adapters.fallback_camera import SyntheticGateCamera
from marine_race_arena.arena.gate_factory import VisualGate
from marine_race_arena.arena.obstacle import Obstacle
from marine_race_arena.config.schema import Vector3
from marine_race_arena.participants.participant import RaceParticipant

LOGGER = logging.getLogger(__name__)


class FallbackRaceAdapter(BaseRaceAdapter):
    """Simple kinematic adapter that preserves the original fallback behavior."""

    name = "fallback"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._participants: Dict[str, RaceParticipant] = {}
        self._states: Dict[str, AdapterParticipantState] = {}
        self._commands: Dict[str, Mapping[str, Any]] = {}
        self._body_velocities: Dict[str, Vector3] = {}
        self._yaw_rates_rad_s: Dict[str, float] = {}
        self._time_s = 0.0
        self._camera: Optional[SyntheticGateCamera] = None

    def initialize(self) -> None:
        LOGGER.info("Using fallback point-vehicle adapter; no HoloOcean physics are active.")

    def reset(self) -> None:
        self._time_s = 0.0

    def spawn_participants(self, participants: Mapping[str, RaceParticipant]) -> None:
        self._participants = dict(participants)
        self._states = {
            participant_id: AdapterParticipantState(
                participant_id=participant_id,
                position=participant.position,
                rotation_rpy_deg=participant.rotation_rpy_deg,
                raw_sensors={},
            )
            for participant_id, participant in participants.items()
        }
        self._body_velocities = {participant_id: (0.0, 0.0, 0.0) for participant_id in participants}
        self._yaw_rates_rad_s = {participant_id: 0.0 for participant_id in participants}

    def spawn_visual_gates(self, visual_gates: Iterable[VisualGate]) -> None:
        count = sum(len(visual_gate.bars) for visual_gate in visual_gates)
        self._camera = SyntheticGateCamera(self.arena.gates)
        if not self._camera.available:
            self._camera = None
            LOGGER.warning(
                "Fallback adapter could not enable the synthetic FrontCamera (numpy missing); "
                "%d debug gate bars are metadata only.",
                count,
            )
            return
        LOGGER.info(
            "Fallback adapter renders %d gate frames through a synthetic FrontCamera "
            "(debug/test substrate; official experiments use the HoloOcean camera).",
            len(self.arena.gates),
        )

    def spawn_obstacles(self, obstacles: Iterable[Obstacle]) -> None:
        count = len(list(obstacles))
        if count:
            LOGGER.info(
                "Fallback adapter keeps %d static obstacles as metadata and uses approximate box collisions.",
                count,
            )

    def get_participant_state(self, participant_id: str) -> AdapterParticipantState:
        try:
            return self._states[participant_id]
        except KeyError as exc:
            raise RaceAdapterError(f"Unknown fallback participant '{participant_id}'.") from exc

    def get_allowed_sensor_data(self, participant_id: str, sensor_profile: Any) -> Dict[str, Any]:
        state = self.get_participant_state(participant_id)
        body_velocity = self._body_velocities.get(participant_id, (0.0, 0.0, 0.0))
        yaw_rate = self._yaw_rates_rad_s.get(participant_id, 0.0)
        raw: Dict[str, Any] = {
            # Same layouts as the HoloOcean onboard sensors.
            "DepthSensor": [state.position[2]],
            "IMUSensor": [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, yaw_rate],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            "DVLSensor": list(body_velocity) + [50.0, 50.0, 50.0, 50.0],
            "CollisionSensor": [False],
        }
        if self._camera is not None:
            frame = self._camera.render(state.position, state.rotation_rpy_deg)
            if frame is not None:
                raw["FrontCamera"] = frame
        participant = self._participants[participant_id]
        return self.filter_sensor_data(
            raw,
            sensor_profile,
            official_mode=self.config.race.official_mode or participant.config.official_sensor_profile,
        )

    def apply_command(self, participant_id: str, command: Mapping[str, Any], control_mode: str) -> None:
        if control_mode == "thrusters" or "thrusters" in command:
            self._commands[participant_id] = {"thrusters": self.clamp_thruster_command(command)}
        else:
            self._commands[participant_id] = self.clamp_high_level_command(
                command, participant_id=participant_id
            )

    def get_collision_state(self, participant_id: str) -> bool:
        self.get_participant_state(participant_id)
        return False

    def get_current_time(self) -> float:
        return self._time_s

    def step(self, dt: float) -> None:
        for participant_id, participant in self._participants.items():
            state = self._states[participant_id]
            command = self._commands.get(participant_id, {})
            current_velocity = self.arena.current_manager.get_current_at(state.position, self._time_s)
            position, rotation, body_velocity, yaw_rate = self._apply_command(
                state.position,
                state.rotation_rpy_deg,
                command,
                dt,
                current_velocity,
                participant.config.control_mode,
            )
            self._states[participant_id] = AdapterParticipantState(
                participant_id=participant_id,
                position=position,
                rotation_rpy_deg=rotation,
                raw_sensors=state.raw_sensors,
            )
            self._body_velocities[participant_id] = body_velocity
            self._yaw_rates_rad_s[participant_id] = yaw_rate
        self._time_s = round(self._time_s + dt, 10)

    def close(self) -> None:
        self._commands.clear()

    def _apply_command(
        self,
        position: Vector3,
        rotation_rpy_deg: Vector3,
        command: Mapping[str, Any],
        dt: float,
        current_velocity: Vector3,
        control_mode: str,
    ) -> tuple[Vector3, Vector3, Vector3, float]:
        yaw_deg = rotation_rpy_deg[2]
        yaw_rad = math.radians(yaw_deg)
        if "thrusters" in command or control_mode == "thrusters":
            surge, sway, heave, yaw_command = self._thruster_fallback(command.get("thrusters", []))
        else:
            safe = self.clamp_high_level_command(command)
            surge = safe["surge"]
            sway = safe["sway"]
            heave = safe["heave"]
            yaw_command = safe["yaw"]

        max_linear_speed_m_s = 1.25
        max_yaw_rate_deg_s = 65.0
        body_vx = surge * max_linear_speed_m_s
        body_vy = sway * max_linear_speed_m_s
        body_vz = heave * max_linear_speed_m_s
        world_vx = math.cos(yaw_rad) * body_vx - math.sin(yaw_rad) * body_vy
        world_vy = math.sin(yaw_rad) * body_vx + math.cos(yaw_rad) * body_vy
        world_vz = body_vz
        velocity = (
            world_vx + current_velocity[0],
            world_vy + current_velocity[1],
            world_vz + current_velocity[2],
        )
        new_position = (
            position[0] + velocity[0] * dt,
            position[1] + velocity[1] * dt,
            position[2] + velocity[2] * dt,
        )
        yaw_rate_rad_s = math.radians(yaw_command * max_yaw_rate_deg_s)
        new_rotation = (
            rotation_rpy_deg[0],
            rotation_rpy_deg[1],
            _wrap_degrees(yaw_deg + yaw_command * max_yaw_rate_deg_s * dt),
        )
        # A DVL measures velocity over ground in the vehicle body frame, so the
        # current-induced drift is part of the measurement.
        dvl_velocity = (
            math.cos(yaw_rad) * velocity[0] + math.sin(yaw_rad) * velocity[1],
            -math.sin(yaw_rad) * velocity[0] + math.cos(yaw_rad) * velocity[1],
            velocity[2],
        )
        return new_position, new_rotation, dvl_velocity, yaw_rate_rad_s

    def _thruster_fallback(self, thrusters: Any) -> tuple[float, float, float, float]:
        values = [float(value) for value in thrusters] if isinstance(thrusters, list) else []
        if not values:
            return (0.0, 0.0, 0.0, 0.0)
        average = sum(values) / len(values)
        yaw = (values[0] - values[-1]) if len(values) >= 2 else 0.0
        return (_clamp(average, -1.0, 1.0), 0.0, 0.0, _clamp(yaw, -1.0, 1.0))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0
