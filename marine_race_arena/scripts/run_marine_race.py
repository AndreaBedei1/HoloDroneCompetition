"""Run a marine race with the available simulator adapter or fallback kinematics."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marine_race_arena.arena.arena_builder import Arena, ArenaBuilder
from marine_race_arena.config.loader import TrackConfigLoadError, load_track_config
from marine_race_arena.config.schema import RaceConfig, TrackConfig, Vector3
from marine_race_arena.participants.controller_loader import ControllerError, ControllerLoader
from marine_race_arena.participants.participant import RaceParticipant
from marine_race_arena.participants.sensor_profile import build_observation
from marine_race_arena.referee.logger import RaceLogger
from marine_race_arena.referee.race_state import ParticipantStatus
from marine_race_arena.referee.referee import Referee

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, help="Path to track JSON.")
    parser.add_argument(
        "--controller",
        default=None,
        help="Built-in controller alias: oracle, acoustic, student_template. Overrides track config.",
    )
    parser.add_argument(
        "--participant-controller",
        default=None,
        help="External controller module/class, module:Class, or file path. Overrides --controller.",
    )
    parser.add_argument("--duration", type=float, default=None, help="Maximum race duration in seconds.")
    parser.add_argument("--official", action="store_true", help="Force official sensor/timing mode.")
    parser.add_argument("--headless", action="store_true", help="Reserved for HoloOcean adapter use.")
    parser.add_argument("--record", action="store_true", help="Reserved for HoloOcean recording adapter use.")
    parser.add_argument("--log-dir", default="results/marine_race", help="Directory for JSONL and summary logs.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for beacons and fallback runner.")
    parser.add_argument("--dt", type=float, default=0.1, help="Fallback simulation timestep.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        config = load_track_config(args.track)
    except (TrackConfigLoadError, ValueError) as exc:
        print(f"Track validation failed: {exc}", file=sys.stderr)
        return 1

    config = _with_cli_overrides(config, duration_s=args.duration, official=args.official)
    _report_adapter_status(args.headless, args.record)

    arena = ArenaBuilder(config, seed=args.seed).build(visual_spawner=None)
    logger = RaceLogger(args.log_dir, config.race.name, track_file=args.track)
    referee = Referee(config, arena.gate_map, arena.bounds, logger=logger)

    try:
        participants = _load_participants(config, args)
    except ControllerError as exc:
        logger.close()
        print(f"Controller load failed: {exc}", file=sys.stderr)
        return 1

    for participant in participants.values():
        if config.race.official_mode and bool(getattr(participant.controller, "uses_ground_truth", False)):
            logger.close()
            print(
                "Oracle/debug controllers use ground truth and are not allowed in official mode.",
                file=sys.stderr,
            )
            return 1

    referee.register_participants(participants.keys())
    race_info = _race_info(config)
    for participant in participants.values():
        participant.controller.reset(race_info | {"initial_target_gate_id": referee.expected_gate_id(participant.id)})

    try:
        summary = _run_fallback_kinematic_race(
            config=config,
            arena=arena,
            referee=referee,
            participants=participants,
            dt=args.dt,
        )
        logger.log_event("race_summary", config.race.max_duration_s, summary=summary)
        logger.write_summary(summary)
    finally:
        for participant in participants.values():
            try:
                participant.controller.close()
            except Exception as exc:  # pragma: no cover - defensive close path
                LOGGER.warning("Controller '%s' close failed: %s", participant.id, exc)
        logger.close()

    _print_summary(summary)
    print(f"Event log: {logger.event_path}")
    print(f"Summary: {logger.summary_path}")
    return 0


def _with_cli_overrides(config: TrackConfig, duration_s: float | None, official: bool) -> TrackConfig:
    race = config.race
    if duration_s is not None or official:
        race = replace(
            race,
            max_duration_s=float(duration_s) if duration_s is not None else race.max_duration_s,
            official_mode=True if official else race.official_mode,
        )
        config = replace(config, race=race)
    return config


def _report_adapter_status(headless: bool, record: bool) -> None:
    try:
        __import__("holoocean")
    except ImportError:
        LOGGER.warning(
            "HoloOcean is not importable in this Python environment; using fallback kinematic runner."
        )
    else:
        LOGGER.warning(
            "HoloOcean is importable, but no repository-specific spawn/control adapter was found; "
            "using fallback kinematic runner."
        )
    if headless:
        LOGGER.warning("--headless is reserved for a HoloOcean adapter and has no effect in fallback mode.")
    if record:
        LOGGER.warning("--record is reserved for a HoloOcean adapter and has no effect in fallback mode.")


def _load_participants(config: TrackConfig, args: argparse.Namespace) -> Dict[str, RaceParticipant]:
    loader = ControllerLoader()
    participants: Dict[str, RaceParticipant] = {}
    for participant_config in config.participants:
        controller_reference = (
            args.participant_controller
            or args.controller
            or participant_config.controller
        )
        controller_class = participant_config.controller_class
        if args.participant_controller and args.participant_controller.endswith(".py"):
            controller_class = participant_config.controller_class
        controller = loader.load(controller_reference, controller_class=controller_class)
        spawn = participant_config.spawn or {}
        position = _vector3(spawn.get("position", config.start.position))
        rotation = _vector3(spawn.get("rotation_rpy_deg", config.start.rotation_rpy_deg))
        participants[participant_config.id] = RaceParticipant(
            config=participant_config,
            controller=controller,
            position=position,
            rotation_rpy_deg=rotation,
        )
    return participants


def _race_info(config: TrackConfig) -> Dict[str, Any]:
    return {
        "race_name": config.race.name,
        "format": config.race.format,
        "laps": config.race.laps,
        "gates_per_lap": len(config.track.gate_sequence),
        "timing_mode": config.race.timing_mode,
        "official_mode": config.race.official_mode,
        "max_duration_s": config.race.max_duration_s,
        "max_command": 0.95,
        "bounds": {
            "x_min": config.world.bounds.x_min,
            "x_max": config.world.bounds.x_max,
            "y_min": config.world.bounds.y_min,
            "y_max": config.world.bounds.y_max,
            "z_min": config.world.bounds.z_min,
            "z_max": config.world.bounds.z_max,
        },
    }


def _run_fallback_kinematic_race(
    config: TrackConfig,
    arena: Arena,
    referee: Referee,
    participants: Dict[str, RaceParticipant],
    dt: float,
) -> Dict[str, Any]:
    LOGGER.info(
        "Starting fallback kinematic race in %s with %d participant(s).",
        arena.environment_name,
        len(participants),
    )
    time_s = 0.0
    referee.start_race(time_s)
    while time_s <= config.race.max_duration_s:
        all_terminal = True
        for participant in participants.values():
            state = referee.states[participant.id]
            if state.is_terminal:
                continue
            all_terminal = False
            previous_position = participant.position
            target_gate_id = referee.expected_gate_id(participant.id)
            target_gate = arena.gate_map[target_gate_id]
            current_velocity = arena.current_manager.get_current_at(participant.position, time_s)
            observation_mode = _observation_mode(config, participant)
            beacon_observation = arena.beacon_manager.observe(
                receiver_position=participant.position,
                receiver_yaw_deg=participant.rotation_rpy_deg[2],
                target_gate_id=target_gate_id,
                target_sequence_index=state.expected_gate_index,
                observation_mode=observation_mode,
                official_mode=config.race.official_mode,
            )
            sensor_data = {
                "heading_yaw_deg": participant.rotation_rpy_deg[2],
                "depth_m": -participant.position[2],
                "environment_current_m_s": current_velocity,
                "control_mode": participant.config.control_mode,
            }
            debug_ground_truth = None
            if bool(getattr(participant.controller, "uses_ground_truth", False)) and not config.race.official_mode:
                debug_ground_truth = {
                    "own_position": participant.position,
                    "own_rotation_rpy_deg": participant.rotation_rpy_deg,
                    "target_gate_center": target_gate.center,
                    "target_gate_normal": target_gate.normal_vector,
                    "target_gate_right_axis": target_gate.right_axis,
                    "target_gate_up_axis": target_gate.up_axis,
                    "target_gate_inner_width_m": target_gate.inner_width_m,
                    "target_gate_inner_height_m": target_gate.inner_height_m,
                    "bounds": _race_info(config)["bounds"],
                }
            observation = build_observation(
                participant_id=participant.id,
                time_s=time_s,
                sensor_data=sensor_data,
                beacon_observation=beacon_observation,
                race_progress=referee.race_progress(participant.id),
                official_mode=config.race.official_mode,
                debug_ground_truth=debug_ground_truth,
            )

            controller_error = None
            command: Dict[str, Any] = {}
            try:
                command = participant.controller.step(copy.deepcopy(observation))
            except Exception as exc:  # pragma: no cover - exercised by external controllers
                controller_error = f"{type(exc).__name__}: {exc}"

            if controller_error is None:
                participant.position, participant.rotation_rpy_deg = _apply_command(
                    participant.position,
                    participant.rotation_rpy_deg,
                    command,
                    dt,
                    current_velocity,
                    participant.config.control_mode,
                )
            referee.update(
                participant_id=participant.id,
                previous_position=previous_position,
                current_position=participant.position,
                time_s=time_s,
                controller_error=controller_error,
            )
        if all_terminal:
            break
        time_s = round(time_s + dt, 10)

    for participant in participants.values():
        state = referee.states[participant.id]
        if state.status == ParticipantStatus.RUNNING:
            referee.update(
                participant_id=participant.id,
                previous_position=participant.position,
                current_position=participant.position,
                time_s=config.race.max_duration_s + dt,
            )
    return referee.summary()


def _observation_mode(config: TrackConfig, participant: RaceParticipant) -> str:
    if bool(getattr(participant.controller, "uses_ground_truth", False)) and not config.race.official_mode:
        return "oracle"
    if config.race.official_mode or participant.config.official_sensor_profile:
        return "acoustic_noisy"
    if config.beacon.noise_std > 0.0 or config.beacon.dropout_probability > 0.0:
        return "acoustic_noisy"
    return "acoustic_ideal"


def _apply_command(
    position: Vector3,
    rotation_rpy_deg: Vector3,
    command: Dict[str, Any],
    dt: float,
    current_velocity: Vector3,
    control_mode: str,
) -> Tuple[Vector3, Vector3]:
    yaw_deg = rotation_rpy_deg[2]
    yaw_rad = math.radians(yaw_deg)
    if "thrusters" in command or control_mode == "thrusters":
        surge, sway, heave, yaw_command = _thruster_fallback(command.get("thrusters", []))
    else:
        surge = _clamp(float(command.get("surge", 0.0)), -1.0, 1.0)
        sway = _clamp(float(command.get("sway", 0.0)), -1.0, 1.0)
        heave = _clamp(float(command.get("heave", 0.0)), -1.0, 1.0)
        yaw_command = _clamp(float(command.get("yaw", 0.0)), -1.0, 1.0)

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
    new_rotation = (
        rotation_rpy_deg[0],
        rotation_rpy_deg[1],
        _wrap_degrees(yaw_deg + yaw_command * max_yaw_rate_deg_s * dt),
    )
    return new_position, new_rotation


def _thruster_fallback(thrusters: Any) -> Tuple[float, float, float, float]:
    values = [float(value) for value in thrusters] if isinstance(thrusters, list) else []
    if not values:
        return (0.0, 0.0, 0.0, 0.0)
    average = sum(values) / len(values)
    yaw = (values[0] - values[-1]) if len(values) >= 2 else 0.0
    return (_clamp(average, -1.0, 1.0), 0.0, 0.0, _clamp(yaw, -1.0, 1.0))


def _vector3(value: Any) -> Vector3:
    return (float(value[0]), float(value[1]), float(value[2]))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _print_summary(summary: Dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
