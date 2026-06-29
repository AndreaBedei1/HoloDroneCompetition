"""Guarded HoloOcean adapter for BlueROV2 marine races."""

from __future__ import annotations

import importlib
import logging
import math
from typing import Any, Dict, Iterable, Mapping, Optional

from marine_race_arena.adapters.base import AdapterParticipantState, BaseRaceAdapter, RaceAdapterError, RaceAdapterUnavailable
from marine_race_arena.adapters.visual_spawner import HoloOceanVisualSpawner
from marine_race_arena.arena.gate_factory import VisualGate
from marine_race_arena.arena.obstacle import OBSTACLE_PHYSICS_DYNAMIC, Obstacle
from marine_race_arena.config.schema import Vector3
from marine_race_arena.participants.participant import RaceParticipant

LOGGER = logging.getLogger(__name__)


class HoloOceanRaceAdapter(BaseRaceAdapter):
    """Adapter that connects the race loop to a HoloOcean BlueROV2 simulation."""

    name = "holoocean"
    thruster_limit = 12.0

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
        self.physical_current_coupling_active = False
        self.current_coupling_method = "not_checked"

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
        self.physical_current_coupling_active = callable(getattr(self.env, "set_ocean_currents", None))
        self.current_coupling_method = (
            "env.set_ocean_currents(agent_name, velocity)"
            if self.physical_current_coupling_active
            else "unavailable: environment has no set_ocean_currents method"
        )
        if not self.physical_current_coupling_active:
            LOGGER.warning(
                "This HoloOcean environment does not expose set_ocean_currents; configured currents "
                "will be exposed in observations/logs only."
            )
        self.reset()

    def spawn_visual_gates(self, visual_gates: Iterable[VisualGate]) -> None:
        if self.visual_spawner is None:
            self.visual_spawner = HoloOceanVisualSpawner(self.env)
        bars = [bar for visual_gate in visual_gates for bar in visual_gate.bars]
        self.visual_spawner.spawn_gate_bars(bars)

    def spawn_obstacles(self, obstacles: Iterable[Obstacle]) -> None:
        if self.visual_spawner is None:
            self.visual_spawner = HoloOceanVisualSpawner(self.env)
        obstacle_list = list(obstacles)
        spawned_count = 0
        sim_physics = (
            self.config.obstacle_generation.obstacle_physics.strip().lower()
            == OBSTACLE_PHYSICS_DYNAMIC
        )
        for obstacle in obstacle_list:
            if obstacle.type != "box":
                LOGGER.warning("Skipping unsupported HoloOcean obstacle '%s' of type '%s'.", obstacle.id, obstacle.type)
                continue
            LOGGER.info(
                "Spawning obstacle %s position=%s size=%s sim_physics=%s",
                obstacle.id,
                obstacle.position,
                obstacle.size,
                sim_physics,
            )
            if self.visual_spawner.spawn_physical_box(
                id=obstacle.id,
                position=obstacle.position,
                rotation_rpy_deg=obstacle.rotation_rpy_deg,
                dimensions_m=obstacle.size,
                material="steel",
                sim_physics=sim_physics,
            ):
                spawned_count += 1
        if obstacle_list and spawned_count < len(obstacle_list):
            LOGGER.warning(
                "Only %d of %d configured obstacles were physically spawned; approximate collision checks remain active.",
                spawned_count,
                len(obstacle_list),
            )

    def get_participant_state(self, participant_id: str) -> AdapterParticipantState:
        self._refresh_states_from_raw()
        try:
            return self._states[participant_id]
        except KeyError as exc:
            raise RaceAdapterError(f"Unknown HoloOcean participant '{participant_id}'.") from exc

    def get_allowed_sensor_data(self, participant_id: str, sensor_profile: Any) -> Dict[str, Any]:
        state = self.get_participant_state(participant_id)
        raw_sensors = _normalize_front_camera_alias(dict(state.raw_sensors))
        current_velocity = self.arena.current_manager.get_current_at(state.position, self._time_s)
        raw_sensors.setdefault("heading_yaw_deg", state.rotation_rpy_deg[2])
        raw_sensors.setdefault("depth_m", -state.position[2])
        raw_sensors["environment_current_m_s"] = current_velocity
        raw_sensors["current_physical_coupling_active"] = self.physical_current_coupling_active
        raw_sensors["current_coupling_method"] = self.current_coupling_method
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
        self._act(participant_id, action)

    def teleport_participant(
        self,
        participant_id: str,
        position: Vector3,
        rotation_rpy_deg: Vector3,
    ) -> None:
        """Reposition an already-spawned agent, reusing the running engine.

        Used by the inter-vehicle collision calibration to sweep relative poses
        without relaunching HoloOcean for every sample.
        """
        if self.env is None:
            raise RaceAdapterError("HoloOcean environment is not initialized.")
        agents = getattr(self.env, "agents", None)
        agent = agents.get(participant_id) if isinstance(agents, dict) else None
        if agent is None or not callable(getattr(agent, "teleport", None)):
            raise RaceAdapterError(
                f"HoloOcean agent '{participant_id}' does not support teleport."
            )
        agent.teleport(
            location=[float(position[0]), float(position[1]), float(position[2])],
            rotation=[float(rotation_rpy_deg[0]), float(rotation_rpy_deg[1]), float(rotation_rpy_deg[2])],
        )

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
        self._apply_environment_currents()
        tick = getattr(self.env, "tick", None)
        if callable(tick):
            ticks = max(1, int(round(dt * float(self.config.raw.get("ticks_per_sec", 30)))))
            state = tick(num_ticks=ticks)
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
                env = self._holoocean.make(
                    scenario_cfg=scenario,
                    show_viewport=not self.headless,
                    ticks_per_sec=scenario.get("ticks_per_sec", 30),
                    frames_per_sec=scenario.get("frames_per_sec", True),
                )
                self._active_environment_name = environment_name
                LOGGER.info("Initialized HoloOcean environment %s.", environment_name)
                return env
            except Exception as exc:
                failures.append(f"{environment_name} scenario_cfg failed: {type(exc).__name__}: {exc}")
        raise RaceAdapterUnavailable(
            "Could not initialize a custom BlueROV2 HoloOcean scenario for any configured environment. "
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
            "main_agent": next(iter(self._participants.keys()), "bluerov2_01"),
            "ticks_per_sec": 30,
            "frames_per_sec": True,
            "window_width": 1280,
            "window_height": 720,
            "current": {"vehicle_debugging": False},
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
        }

    def _build_sensor_configs(self, participant: RaceParticipant) -> list[Dict[str, Any]]:
        configured = participant.config.sensors
        if isinstance(configured, Mapping) and isinstance(configured.get("holoocean_sensors"), list):
            sensors = [dict(sensor) for sensor in configured["holoocean_sensors"] if isinstance(sensor, Mapping)]
        else:
            sensors = [
                {"sensor_type": "DepthSensor", "socket": "DepthSocket", "Hz": 30, "configuration": {"Sigma": 0.0}},
                {"sensor_type": "IMUSensor", "socket": "IMUSocket", "Hz": 30, "configuration": {"ReturnBias": True}},
                {
                    "sensor_type": "DVLSensor",
                    "socket": "DVLSocket",
                    "Hz": 15,
                    "configuration": {"Elevation": 22.5, "ReturnRange": True, "MaxRange": 50},
                },
            ]
        sensor_types = {sensor.get("sensor_type") for sensor in sensors}
        sensor_names = {sensor.get("sensor_name") for sensor in sensors}
        if (
            _sensor_profile_requests_front_camera(configured)
            and "FrontCamera" not in sensor_names
            and "RGBCamera" not in sensor_types
        ):
            sensors.append(_front_camera_sensor_config())
        if "PoseSensor" not in sensor_types:
            sensors.append({"sensor_type": "PoseSensor", "socket": "IMUSocket", "Hz": 30})
        if "VelocitySensor" not in sensor_types:
            sensors.append({"sensor_type": "VelocitySensor", "socket": "IMUSocket", "Hz": 30})
        if "CollisionSensor" not in sensor_types:
            sensors.append({"sensor_type": "CollisionSensor", "Hz": 30})
        return sensors

    def _act(self, participant_id: str, action: list[float]) -> None:
        act = getattr(self.env, "act", None)
        if callable(act):
            act(participant_id, action)
            return
        step = getattr(self.env, "step", None)
        if callable(step) and len(self._participants) == 1:
            return
        raise RaceAdapterError("HoloOcean environment does not expose act(agent_name, action).")

    def _apply_environment_currents(self) -> None:
        if self.env is None or not self.physical_current_coupling_active:
            return
        set_currents = getattr(self.env, "set_ocean_currents")
        for participant_id in self._participants:
            state = self.get_participant_state(participant_id)
            current_velocity = self.arena.current_manager.get_current_at(state.position, self._time_s)
            set_currents(participant_id, list(current_velocity))

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

    def diagnose_multi_agent_state(
        self,
        participants: Mapping[str, RaceParticipant],
        stage: str,
    ) -> Dict[str, Any]:
        if len(participants) > 1:
            if not isinstance(self._raw_state, dict):
                raise RaceAdapterError(
                    f"HoloOcean multi-agent state during {stage} was "
                    f"{type(self._raw_state).__name__}, expected a dict keyed by participant id."
                )
            missing = [
                participant_id
                for participant_id in participants
                if not isinstance(self._raw_state.get(participant_id), dict)
            ]
            if missing:
                available = sorted(str(key) for key in self._raw_state.keys())
                raise RaceAdapterError(
                    "HoloOcean multi-agent state is missing participant sensor dictionaries "
                    f"during {stage}. Missing={missing}; available_keys={available}."
                )
        diagnostics = super().diagnose_multi_agent_state(participants, stage)
        if len(participants) > 1 and diagnostics.get("unique_position_count", 0) <= 1:
            raise RaceAdapterError(
                "HoloOcean multi-agent participants do not have distinct state positions "
                f"during {stage}; check spawn offsets and state parsing."
            )
        return diagnostics


def _world_from_environment(environment_name: str) -> str:
    return environment_name.split("-", 1)[0] if "-" in environment_name else environment_name


def _front_camera_sensor_config() -> Dict[str, Any]:
    return {
        "sensor_type": "RGBCamera",
        "sensor_name": "FrontCamera",
        "socket": "CameraSocket",
        "rotation": [0.0, 0.0, 0.0],
        "Hz": 30,
        "configuration": {
            "CaptureWidth": 640,
            "CaptureHeight": 480,
            "FovAngle": 90.0,
        },
    }


def _sensor_profile_requests_front_camera(configured: Any) -> bool:
    if not isinstance(configured, Mapping):
        return False
    if str(configured.get("profile", "")).lower() in {"official_vision_acoustic", "official_vision"}:
        return True
    for key in ("allowed", "allowed_sensors", "sensors"):
        values = configured.get(key)
        if isinstance(values, list) and "FrontCamera" in values:
            return True
    holoocean_sensors = configured.get("holoocean_sensors")
    if isinstance(holoocean_sensors, list):
        return any(
            isinstance(sensor, Mapping)
            and (
                sensor.get("sensor_name") == "FrontCamera"
                or sensor.get("sensor_type") == "RGBCamera"
            )
            for sensor in holoocean_sensors
        )
    return False


def _normalize_front_camera_alias(raw_sensors: Dict[str, Any]) -> Dict[str, Any]:
    if "FrontCamera" in raw_sensors:
        return raw_sensors
    for alias in ("RGBCamera", "FrontRGBCamera", "front_camera"):
        if alias in raw_sensors:
            raw_sensors["FrontCamera"] = raw_sensors[alias]
            break
    return raw_sensors


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
