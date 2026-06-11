"""Guarded HoloOcean adapter for BlueROV2 marine races."""

from __future__ import annotations

import importlib
import logging
import math
from typing import Any, Dict, Iterable, Mapping, Optional

from marine_race_arena.adapters.base import AdapterParticipantState, BaseRaceAdapter, RaceAdapterError, RaceAdapterUnavailable
from marine_race_arena.adapters.visual_spawner import HoloOceanVisualSpawner
from marine_race_arena.arena.gate_factory import VisualGate
from marine_race_arena.config.schema import Vector3
from marine_race_arena.participants.participant import RaceParticipant

LOGGER = logging.getLogger(__name__)


class HoloOceanRaceAdapter(BaseRaceAdapter):
    """Adapter that connects the race loop to a HoloOcean BlueROV2 simulation."""

    name = "holoocean"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._holoocean: Any = None
        self.env: Any = None
        self.visual_spawner: Optional[HoloOceanVisualSpawner] = None
        self._participants: Dict[str, RaceParticipant] = {}
        self._states: Dict[str, AdapterParticipantState] = {}
        self._raw_state: Dict[str, Any] = {}
        self._last_actions: Dict[str, list[float]] = {}
        self._time_s = 0.0
        self._active_environment_name: Optional[str] = None
        self._warned_current_coupling = False

    def initialize(self) -> None:
        try:
            self._holoocean = importlib.import_module("holoocean")
        except ImportError as exc:
            raise RaceAdapterUnavailable(
                "The holoocean Python package is not importable in this environment."
            ) from exc

    def reset(self) -> None:
        self._time_s = 0.0
        if self.env is not None and hasattr(self.env, "reset"):
            state = self.env.reset()
            if isinstance(state, dict):
                self._raw_state = state
                self._refresh_states_from_raw()

    def spawn_participants(self, participants: Mapping[str, RaceParticipant]) -> None:
        if self._holoocean is None:
            self.initialize()
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
        self.env = self._make_environment()
        self.visual_spawner = HoloOceanVisualSpawner(self.env)
        self.reset()

    def spawn_visual_gates(self, visual_gates: Iterable[VisualGate]) -> None:
        if self.visual_spawner is None:
            self.visual_spawner = HoloOceanVisualSpawner(self.env)
        bars = [bar for visual_gate in visual_gates for bar in visual_gate.bars]
        self.visual_spawner.spawn_gate_bars(bars)

    def get_participant_state(self, participant_id: str) -> AdapterParticipantState:
        self._refresh_states_from_raw()
        try:
            return self._states[participant_id]
        except KeyError as exc:
            raise RaceAdapterError(f"Unknown HoloOcean participant '{participant_id}'.") from exc

    def get_allowed_sensor_data(self, participant_id: str, sensor_profile: Any) -> Dict[str, Any]:
        state = self.get_participant_state(participant_id)
        raw_sensors = dict(state.raw_sensors)
        current_velocity = self.arena.current_manager.get_current_at(state.position, self._time_s)
        raw_sensors.setdefault("heading_yaw_deg", state.rotation_rpy_deg[2])
        raw_sensors.setdefault("depth_m", -state.position[2])
        raw_sensors["environment_current_m_s"] = current_velocity
        raw_sensors["current_physical_coupling_active"] = False
        participant = self._participants[participant_id]
        raw_sensors["control_mode"] = participant.config.control_mode
        return self.filter_sensor_data(
            raw_sensors,
            sensor_profile,
            official_mode=self.config.race.official_mode or participant.config.official_sensor_profile,
        )

    def apply_command(self, participant_id: str, command: Mapping[str, Any], control_mode: str) -> None:
        if self.env is None:
            raise RaceAdapterError("HoloOcean environment is not initialized.")
        action = self.command_to_bluerov2_thrusters(participant_id, command, control_mode)
        self._last_actions[participant_id] = action
        if not self._warned_current_coupling:
            LOGGER.warning(
                "No supported HoloOcean current-force API was found; configured currents are exposed "
                "to observations/logs but are not physically coupled in the HoloOcean adapter."
            )
            self._warned_current_coupling = True
        self._act(participant_id, action)

    def get_collision_state(self, participant_id: str) -> bool:
        sensors = self._agent_sensors(participant_id)
        for key, value in sensors.items():
            lowered = key.lower()
            if "collision" not in lowered and "contact" not in lowered:
                continue
            if hasattr(value, "any"):
                try:
                    return bool(value.any())
                except Exception:
                    return bool(value)
            return bool(value)
        return False

    def get_current_time(self) -> float:
        return self._time_s

    def step(self, dt: float) -> None:
        if self.env is None:
            raise RaceAdapterError("HoloOcean environment is not initialized.")
        tick = getattr(self.env, "tick", None)
        if callable(tick):
            state = tick()
        else:
            if not self._last_actions:
                raise RaceAdapterError("HoloOcean environment exposes neither tick() nor queued actions.")
            first_action = next(iter(self._last_actions.values()))
            state = self.env.step(first_action)
        if isinstance(state, dict):
            self._raw_state = state
        self._time_s = round(self._time_s + dt, 10)
        self._refresh_states_from_raw()

    def close(self) -> None:
        if self.env is None:
            return
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
        self.env = None

    @property
    def active_environment_name(self) -> Optional[str]:
        return self._active_environment_name

    def _make_environment(self) -> Any:
        failures: list[str] = []
        for environment_name in self._environment_candidates():
            scenario = self._build_scenario(environment_name)
            try:
                LOGGER.info("Trying HoloOcean scenario config for %s.", environment_name)
                env = self._holoocean.make(scenario_cfg=scenario)
                self._active_environment_name = environment_name
                LOGGER.info("Initialized HoloOcean environment %s.", environment_name)
                return env
            except Exception as exc:
                failures.append(f"{environment_name} scenario_cfg failed: {type(exc).__name__}: {exc}")
            try:
                LOGGER.info("Trying prebuilt HoloOcean scenario %s.", environment_name)
                env = self._holoocean.make(environment_name)
                self._active_environment_name = environment_name
                LOGGER.warning(
                    "Using prebuilt HoloOcean scenario %s. Participant start pose and BlueROV2 "
                    "configuration may depend on the installed package scenario.",
                    environment_name,
                )
                return env
            except Exception as exc:
                failures.append(f"{environment_name} prebuilt failed: {type(exc).__name__}: {exc}")
        raise RaceAdapterUnavailable(
            "Could not initialize any configured HoloOcean environment. "
            + " | ".join(failures)
        )

    def _environment_candidates(self) -> list[str]:
        candidates = [
            self.config.world.map,
            self.config.world.preferred_environment,
            self.config.world.fallback_environment,
            "OpenWater-Hovering",
            "PierHarbor-Hovering",
        ]
        unique: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in unique:
                unique.append(candidate)
        return unique

    def _build_scenario(self, environment_name: str) -> Dict[str, Any]:
        return {
            "name": f"{self.config.race.name} Runtime",
            "world": _world_from_environment(environment_name),
            "package_name": self.config.world.package or "Ocean",
            "agents": [
                self._build_agent_config(participant, index == 0)
                for index, participant in enumerate(self._participants.values())
            ],
        }

    def _build_agent_config(self, participant: RaceParticipant, is_main_agent: bool) -> Dict[str, Any]:
        return {
            "agent_name": participant.id,
            "agent_type": participant.config.vehicle or "BlueROV2",
            "location": list(participant.position),
            "rotation": list(participant.rotation_rpy_deg),
            "control_scheme": 0,
            "sensors": self._build_sensor_configs(participant),
            "is_main_agent": is_main_agent,
        }

    def _build_sensor_configs(self, participant: RaceParticipant) -> list[Dict[str, Any]]:
        configured = participant.config.sensors
        if isinstance(configured, Mapping) and isinstance(configured.get("holoocean_sensors"), list):
            sensors = [dict(sensor) for sensor in configured["holoocean_sensors"] if isinstance(sensor, Mapping)]
        else:
            sensors = [
                {"sensor_name": "DepthSensor", "sensor_type": "DepthSensor", "socket": "DepthSocket", "Hz": 20},
                {"sensor_name": "IMUSensor", "sensor_type": "IMUSensor", "socket": "IMUSocket", "Hz": 20},
                {"sensor_name": "DVLSensor", "sensor_type": "DVLSensor", "socket": "DVLSocket", "Hz": 10},
            ]
        internal_sensor_names = {sensor.get("sensor_name") for sensor in sensors}
        if "PoseSensor" not in internal_sensor_names:
            sensors.append({"sensor_name": "PoseSensor", "sensor_type": "PoseSensor", "Hz": 20})
        if "VelocitySensor" not in internal_sensor_names:
            sensors.append({"sensor_name": "VelocitySensor", "sensor_type": "VelocitySensor", "Hz": 20})
        return sensors

    def _act(self, participant_id: str, action: list[float]) -> None:
        act = getattr(self.env, "act", None)
        if callable(act):
            try:
                act(action, participant_id)
                return
            except TypeError:
                act(participant_id, action)
                return
        step = getattr(self.env, "step", None)
        if callable(step) and len(self._participants) == 1:
            return
        raise RaceAdapterError("HoloOcean environment does not expose act(command, agent).")

    def _refresh_states_from_raw(self) -> None:
        for participant_id, participant in self._participants.items():
            sensors = self._agent_sensors(participant_id)
            previous = self._states.get(
                participant_id,
                AdapterParticipantState(participant_id, participant.position, participant.rotation_rpy_deg),
            )
            position, rotation = _extract_pose(sensors, previous.position, previous.rotation_rpy_deg)
            self._states[participant_id] = AdapterParticipantState(
                participant_id=participant_id,
                position=position,
                rotation_rpy_deg=rotation,
                raw_sensors=sensors,
            )

    def _agent_sensors(self, participant_id: str) -> Dict[str, Any]:
        if not isinstance(self._raw_state, dict):
            return {}
        value = self._raw_state.get(participant_id)
        if isinstance(value, dict):
            return value
        # Some single-agent HoloOcean states are returned directly as a sensor dictionary.
        if len(self._participants) == 1:
            return self._raw_state
        return {}


def _world_from_environment(environment_name: str) -> str:
    return environment_name.split("-", 1)[0] if "-" in environment_name else environment_name


def _extract_pose(
    sensors: Mapping[str, Any],
    fallback_position: Vector3,
    fallback_rotation: Vector3,
) -> tuple[Vector3, Vector3]:
    position = fallback_position
    rotation = fallback_rotation
    pose = sensors.get("PoseSensor")
    if pose is not None:
        extracted = _pose_matrix_to_position_rotation(pose)
        if extracted is not None:
            return extracted
    location = sensors.get("LocationSensor")
    if location is not None:
        position = _vector3(location, fallback_position)
    rotation_sensor = sensors.get("RotationSensor")
    if rotation_sensor is not None:
        rotation = _vector3(rotation_sensor, fallback_rotation)
    return position, rotation


def _pose_matrix_to_position_rotation(pose: Any) -> Optional[tuple[Vector3, Vector3]]:
    if hasattr(pose, "tolist"):
        pose = pose.tolist()
    if not isinstance(pose, list) or len(pose) < 3:
        return None
    try:
        position = (float(pose[0][3]), float(pose[1][3]), float(pose[2][3]))
        r00, r01, r02 = float(pose[0][0]), float(pose[0][1]), float(pose[0][2])
        r10, r11, r12 = float(pose[1][0]), float(pose[1][1]), float(pose[1][2])
        r20, r21, r22 = float(pose[2][0]), float(pose[2][1]), float(pose[2][2])
    except (TypeError, ValueError, IndexError):
        return None
    pitch = math.asin(max(-1.0, min(1.0, -r20)))
    if abs(math.cos(pitch)) > 1e-9:
        roll = math.atan2(r21, r22)
        yaw = math.atan2(r10, r00)
    else:
        roll = math.atan2(-r12, r11)
        yaw = 0.0
    rotation = (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))
    return position, rotation


def _vector3(value: Any, fallback: Vector3) -> Vector3:
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError):
        return fallback
