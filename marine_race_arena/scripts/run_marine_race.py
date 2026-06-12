"""Run a marine race through a simulator adapter."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marine_race_arena.adapters import (
    AdapterSelectionError,
    FallbackRaceAdapter,
    RaceAdapterError,
    RaceAdapterUnavailable,
    select_adapter,
)
from marine_race_arena.adapters.base import AdapterParticipantState, BaseRaceAdapter
from marine_race_arena.arena.arena_builder import Arena, ArenaBuilder
from marine_race_arena.config.loader import TrackConfigLoadError, load_track_config
from marine_race_arena.config.schema import TrackConfig, Vector3
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
        help=(
            "Built-in controller alias: pygame, pygame_keyboard, keyboard, manual, "
            "oracle, acoustic, student_template. Overrides track config."
        ),
    )
    parser.add_argument(
        "--participant-controller",
        default=None,
        help="External controller module/class, module:Class, or file path. Overrides --controller.",
    )
    parser.add_argument(
        "--adapter",
        choices=("auto", "fallback", "holoocean"),
        default="auto",
        help="Simulator adapter. auto tries HoloOcean and only falls back when --allow-fallback is set.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow fallback kinematics when the HoloOcean adapter cannot initialize.",
    )
    parser.add_argument("--duration", type=float, default=None, help="Maximum race duration in seconds.")
    parser.add_argument("--official", action="store_true", help="Force official sensor/timing mode.")
    parser.add_argument("--headless", action="store_true", help="Request headless HoloOcean mode when supported.")
    parser.add_argument("--record", action="store_true", help="Request HoloOcean recording when supported.")
    parser.add_argument("--log-dir", default="results/marine_race", help="Directory for JSONL and summary logs.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for beacons and adapters.")
    parser.add_argument("--dt", type=float, default=0.1, help="Race loop timestep.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        config = load_track_config(args.track)
    except (TrackConfigLoadError, ValueError) as exc:
        print(f"Track validation failed: {exc}", file=sys.stderr)
        return 1

    config = _with_cli_overrides(config, duration_s=args.duration, official=args.official)
    arena = ArenaBuilder(config, seed=args.seed).build()
    logger = RaceLogger(args.log_dir, config.race.name, track_file=args.track)
    referee = Referee(config, arena.gate_map, arena.bounds, logger=logger)

    try:
        participants = _load_participants(config, args)
        _reject_invalid_official_controllers(config, participants)
        adapter = _prepare_adapter(config, arena, participants, args)
    except (ControllerError, AdapterSelectionError, RaceAdapterError) as exc:
        logger.close()
        print(f"Race setup failed: {exc}", file=sys.stderr)
        return 1

    referee.register_participants(participants.keys())
    race_info = _race_info(config, adapter.name)
    for participant in participants.values():
        participant.controller.reset(
            race_info | {"initial_target_gate_id": referee.expected_gate_id(participant.id)}
        )

    try:
        summary = _run_race_loop(
            config=config,
            arena=arena,
            referee=referee,
            adapter=adapter,
            participants=participants,
            dt=args.dt,
        )
        logger.log_event("race_summary", adapter.get_current_time(), summary=summary)
        logger.write_summary(summary)
    finally:
        for participant in participants.values():
            try:
                participant.controller.close()
            except Exception as exc:  # pragma: no cover - defensive close path
                LOGGER.warning("Controller '%s' close failed: %s", participant.id, exc)
        adapter.close()
        logger.close()

    _print_summary(summary)
    print(f"Adapter: {adapter.name}")
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


def _reject_invalid_official_controllers(
    config: TrackConfig,
    participants: Mapping[str, RaceParticipant],
) -> None:
    for participant in participants.values():
        if config.race.official_mode and bool(getattr(participant.controller, "uses_ground_truth", False)):
            raise ControllerError(
                "Oracle/debug controllers use ground truth and are not allowed in official mode."
            )


def _prepare_adapter(
    config: TrackConfig,
    arena: Arena,
    participants: Mapping[str, RaceParticipant],
    args: argparse.Namespace,
) -> BaseRaceAdapter:
    adapter = select_adapter(
        adapter_name=args.adapter,
        config=config,
        arena=arena,
        allow_fallback=args.allow_fallback,
        headless=args.headless,
        record=args.record,
        seed=args.seed,
    )
    try:
        adapter.spawn_participants(participants)
        adapter.reset()
        adapter.spawn_visual_gates(arena.visual_gates)
        return adapter
    except RaceAdapterUnavailable as exc:
        adapter.close()
        if args.allow_fallback:
            LOGGER.warning(
                "Adapter '%s' failed after initialization; falling back because --allow-fallback is set: %s",
                adapter.name,
                exc,
            )
            fallback = FallbackRaceAdapter(
                config,
                arena,
                seed=args.seed,
                headless=args.headless,
                record=args.record,
            )
            fallback.initialize()
            fallback.spawn_participants(participants)
            fallback.reset()
            fallback.spawn_visual_gates(arena.visual_gates)
            return fallback
        raise AdapterSelectionError(
            "HoloOcean adapter failed during environment setup and fallback is not allowed. "
            "Use --adapter fallback for the kinematic runner or pass --allow-fallback explicitly."
        ) from exc


def _race_info(config: TrackConfig, adapter_name: str) -> Dict[str, Any]:
    return {
        "race_name": config.race.name,
        "format": config.race.format,
        "laps": config.race.laps,
        "gates_per_lap": len(config.track.gate_sequence),
        "timing_mode": config.race.timing_mode,
        "official_mode": config.race.official_mode,
        "max_duration_s": config.race.max_duration_s,
        "adapter": adapter_name,
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


def _run_race_loop(
    config: TrackConfig,
    arena: Arena,
    referee: Referee,
    adapter: BaseRaceAdapter,
    participants: Mapping[str, RaceParticipant],
    dt: float,
) -> Dict[str, Any]:
    LOGGER.info(
        "Starting race '%s' with adapter '%s' in %s.",
        config.race.name,
        adapter.name,
        arena.environment_name,
    )
    referee.start_race(adapter.get_current_time())
    while adapter.get_current_time() <= config.race.max_duration_s:
        all_terminal = True
        previous_states: Dict[str, AdapterParticipantState] = {}
        controller_errors: Dict[str, str] = {}
        for participant in participants.values():
            state = referee.states[participant.id]
            if state.is_terminal:
                continue
            all_terminal = False
            previous_state = adapter.get_participant_state(participant.id)
            previous_states[participant.id] = previous_state
            observation = _build_controller_observation(
                config=config,
                arena=arena,
                referee=referee,
                adapter=adapter,
                participant=participant,
                participant_state=previous_state,
            )
            try:
                command = participant.controller.step(copy.deepcopy(observation))
            except Exception as exc:  # pragma: no cover - exercised by external controllers
                controller_errors[participant.id] = f"{type(exc).__name__}: {exc}"
                command = {}
            adapter.apply_command(participant.id, command, participant.config.control_mode)

        if all_terminal:
            break

        adapter.step(dt)
        time_s = adapter.get_current_time()
        for participant_id, previous_state in previous_states.items():
            state = referee.states[participant_id]
            if state.is_terminal:
                continue
            current_state = adapter.get_participant_state(participant_id)
            referee.update(
                participant_id=participant_id,
                previous_position=previous_state.position,
                current_position=current_state.position,
                time_s=time_s,
                collision=adapter.get_collision_state(participant_id),
                controller_error=controller_errors.get(participant_id),
            )

    for participant in participants.values():
        state = referee.states[participant.id]
        if state.status == ParticipantStatus.RUNNING:
            current_state = adapter.get_participant_state(participant.id)
            referee.update(
                participant_id=participant.id,
                previous_position=current_state.position,
                current_position=current_state.position,
                time_s=config.race.max_duration_s + dt,
            )
    return referee.summary()


def _build_controller_observation(
    config: TrackConfig,
    arena: Arena,
    referee: Referee,
    adapter: BaseRaceAdapter,
    participant: RaceParticipant,
    participant_state: AdapterParticipantState,
) -> Dict[str, Any]:
    target_gate_id = referee.expected_gate_id(participant.id)
    target_gate = arena.gate_map[target_gate_id]
    progress = referee.race_progress(participant.id)
    beacon_observation = arena.beacon_manager.observe(
        receiver_position=participant_state.position,
        receiver_yaw_deg=participant_state.rotation_rpy_deg[2],
        target_gate_id=target_gate_id,
        target_sequence_index=int(progress["target_sequence_index"]),
        observation_mode=_observation_mode(config, participant),
        official_mode=config.race.official_mode,
    )
    sensor_data = adapter.get_allowed_sensor_data(participant.id, participant.config.sensors)
    debug_ground_truth = None
    if bool(getattr(participant.controller, "uses_ground_truth", False)) and not config.race.official_mode:
        debug_ground_truth = {
            "own_position": participant_state.position,
            "own_rotation_rpy_deg": participant_state.rotation_rpy_deg,
            "target_gate_center": target_gate.center,
            "target_gate_normal": target_gate.normal_vector,
            "target_gate_right_axis": target_gate.right_axis,
            "target_gate_up_axis": target_gate.up_axis,
            "target_gate_inner_width_m": target_gate.inner_width_m,
            "target_gate_inner_height_m": target_gate.inner_height_m,
            "bounds": _race_info(config, adapter.name)["bounds"],
        }
    return build_observation(
        participant_id=participant.id,
        time_s=adapter.get_current_time(),
        sensor_data=sensor_data,
        beacon_observation=beacon_observation,
        race_progress=progress,
        official_mode=config.race.official_mode,
        debug_ground_truth=debug_ground_truth,
    )


def _observation_mode(config: TrackConfig, participant: RaceParticipant) -> str:
    if bool(getattr(participant.controller, "uses_ground_truth", False)) and not config.race.official_mode:
        return "oracle"
    if config.race.official_mode or participant.config.official_sensor_profile:
        return "acoustic_noisy"
    if config.beacon.noise_std > 0.0 or config.beacon.dropout_probability > 0.0:
        return "acoustic_noisy"
    return "acoustic_ideal"


def _vector3(value: Any) -> Vector3:
    return (float(value[0]), float(value[1]), float(value[2]))


def _print_summary(summary: Dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
