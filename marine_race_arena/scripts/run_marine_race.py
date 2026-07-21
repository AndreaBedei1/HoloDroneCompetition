"""Run a marine race through a simulator adapter."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
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
from marine_race_arena.arena.obstacle import (
    OBSTACLE_DENSITIES,
    OBSTACLE_MODES,
    OBSTACLE_PHYSICS_MODES,
)
from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_MODES
from marine_race_arena.config.loader import (
    CURRENT_PROFILE_MODES,
    TrackConfigLoadError,
    describe_current_profile,
    load_track_config,
)
from marine_race_arena.config.schema import TrackConfig, Vector3
from marine_race_arena.controllers.motion_compensation import (
    MOTION_COMPENSATION_MODES,
    MOTION_COMPENSATION_NONE,
    make_motion_compensator,
    normalize_motion_compensation_mode,
)
from marine_race_arena.arena.acoustic_comms import AcousticCommsChannel, CommsConfig
from marine_race_arena.participants.controller_interface import ManualStopRequested
from marine_race_arena.participants.controller_loader import ControllerError, ControllerLoader
from marine_race_arena.participants.participant import RaceParticipant
from marine_race_arena.participants.sensor_profile import build_observation
from marine_race_arena.referee.logger import RaceLogger
from marine_race_arena.referee.race_state import ParticipantStatus
from marine_race_arena.referee.referee import INTER_VEHICLE_COLLISION_MODES, Referee

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        config = load_track_config(
            args.track,
            benchmark_task=args.benchmark_task,
            obstacles=args.obstacles,
            obstacle_density=args.obstacle_density,
            obstacle_physics=args.obstacle_physics,
            current_profile=args.current_profile,
            seed=args.seed,
        )
    except (TrackConfigLoadError, ValueError) as exc:
        print(f"Track validation failed: {exc}", file=sys.stderr)
        return 1
    if args.official and args.disable_front_camera:
        print("Race setup failed: --disable-front-camera is not allowed in official mode.", file=sys.stderr)
        return 1

    try:
        config = _with_cli_overrides(config, duration_s=args.duration, official=args.official)
        config = _with_staggered_participants(config, args)
    except ValueError as exc:
        print(f"Race setup failed: {exc}", file=sys.stderr)
        return 1
    print(f"Selected current profile: {_current_profile_label(config)}")
    print(f"Active currents: {len(config.currents)}")
    for line in describe_current_profile(config):
        print(f"  {line}")
    motion_compensation = normalize_motion_compensation_mode(args.motion_compensation)
    print(f"Motion compensation: {motion_compensation}")
    if args.disable_front_camera:
        config = _without_front_camera(config)
    arena = ArenaBuilder(config, seed=args.seed).build()
    logger = RaceLogger(args.log_dir, config.race.name, track_file=args.track)
    referee = Referee(
        config,
        arena.gate_map,
        arena.bounds,
        logger=logger,
        inter_vehicle_collision_mode=args.inter_vehicle_collision_mode,
        inter_vehicle_collision_xy_threshold_m=args.inter_vehicle_collision_xy_threshold_m,
        inter_vehicle_collision_z_threshold_m=args.inter_vehicle_collision_z_threshold_m,
        inter_vehicle_collision_release_threshold_m=args.inter_vehicle_collision_release_threshold_m,
        inter_vehicle_collision_cooldown_s=args.inter_vehicle_collision_cooldown_s,
        team_id=args.team_id,
    )
    comms_channel = None
    if args.comms_enabled:
        comms_channel = AcousticCommsChannel(
            CommsConfig(
                enabled=True,
                sound_speed_m_s=args.comms_sound_speed_m_s,
                max_range_m=args.comms_max_range_m,
                processing_delay_s=args.comms_processing_delay_s,
                packet_loss_prob=args.comms_packet_loss_prob,
                max_payload_bytes=args.comms_max_payload_bytes,
                min_send_interval_s=args.comms_min_send_interval_s,
            ),
            seed=args.seed,
        )
    camera_viewer = FrontCameraViewer(enabled=args.show_front_camera)
    received_beacon_printer = ReceivedBeaconPrinter(
        enabled=args.print_beacons or _env_flag("MARINE_RACE_PRINT_BEACONS")
    )

    try:
        participants = _load_participants(config, args)
        _reject_invalid_official_controllers(config, participants)
        adapter = _prepare_adapter(config, arena, participants, args)
    except (ControllerError, AdapterSelectionError, RaceAdapterError) as exc:
        camera_viewer.close()
        logger.close()
        print(f"Race setup failed: {exc}", file=sys.stderr)
        return 1

    referee.register_participants(participants.keys())
    motion_compensators = {
        participant_id: make_motion_compensator(motion_compensation)
        for participant_id in participants
    }
    for participant in participants.values():
        participant.controller.reset(_mission_info(config, participant.id))
        motion_compensators[participant.id].reset()

    try:
        summary = _run_race_loop(
            config=config,
            arena=arena,
            referee=referee,
            adapter=adapter,
            participants=participants,
            dt=args.dt,
            camera_viewer=camera_viewer,
            received_beacon_printer=received_beacon_printer,
            motion_compensators=motion_compensators,
            gate_timeout_s=args.gate_timeout_s,
            log_participant_states=args.log_participant_states,
            comms_channel=comms_channel,
        )
        summary["motion_compensation"] = motion_compensation
        summary["adapter"] = adapter.name
        summary["fallback_used"] = adapter.name == "fallback"
        summary["physical_current_coupling_active"] = bool(
            getattr(adapter, "physical_current_coupling_active", False)
        )
        summary["current_coupling_method"] = str(
            getattr(adapter, "current_coupling_method", "not_available")
        )
        summary["physical_obstacles_requested"] = int(
            getattr(adapter, "physical_obstacles_requested", len(arena.obstacles))
        )
        summary["physical_obstacles_spawned"] = int(
            getattr(adapter, "physical_obstacles_spawned", len(arena.obstacles))
        )
        summary["physical_obstacle_spawn_complete"] = bool(
            getattr(adapter, "physical_obstacle_spawn_complete", True)
        )
        summary["controller_observation_contract"] = "onboard_only_v1"
        summary["controller_local_progress"] = _controller_local_summary(participants)
        summary["beacon_reception_diagnostics"] = {
            participant_id: dict(counters)
            for participant_id, counters in arena.beacon_manager.diagnostics.items()
        }
        if len(participants) > 1 or args.inter_vehicle_collision_mode != "off":
            summary["inter_vehicle_collision_mode"] = args.inter_vehicle_collision_mode
        if comms_channel is not None:
            summary["comms"] = comms_channel.summary()
        logger.log_event("race_summary", adapter.get_current_time(), summary=summary)
        logger.write_summary(summary)
    finally:
        camera_viewer.close()
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, help="Path to track JSON.")
    parser.add_argument(
        "--controller",
        default=None,
        help=(
            "Built-in controller alias: pygame, pygame_keyboard, keyboard, manual, "
            "oracle, rule_gate_baseline, rule_gate_center_then_commit, "
            "leader_follower, rl_gate_controller, student_template. Overrides track config."
        ),
    )
    parser.add_argument(
        "--participant-controller",
        default=None,
        help="External controller module/class, module:Class, or file path. Overrides --controller.",
    )
    parser.add_argument(
        "--controller-class",
        default=None,
        help="Controller class name when loading from a Python file or module path.",
    )
    parser.add_argument(
        "--controller-model-path",
        default=None,
        help=(
            "Path to a trained model for a learned controller (e.g. rl_gate_controller). "
            "Takes precedence over the MARINE_RACE_RL_MODEL environment variable. Controllers "
            "that do not accept a model path ignore this option."
        ),
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
    parser.add_argument(
        "--benchmark-task",
        choices=BENCHMARK_TASK_MODES,
        default=None,
        help="Validate the track against an explicit benchmark task mode.",
    )
    parser.add_argument(
        "--obstacles",
        choices=OBSTACLE_MODES,
        default=None,
        help="Obstacle mode. none ignores obstacles, fixed uses track JSON obstacles, random generates seeded obstacles.",
    )
    parser.add_argument(
        "--obstacle-density",
        choices=OBSTACLE_DENSITIES,
        default=None,
        help="Density for generated random obstacles.",
    )
    parser.add_argument(
        "--obstacle-physics",
        choices=OBSTACLE_PHYSICS_MODES,
        default=None,
        help="HoloOcean obstacle prop physics. static keeps obstacles suspended; dynamic enables gravity/physics.",
    )
    parser.add_argument(
        "--current-profile",
        choices=CURRENT_PROFILE_MODES,
        default=None,
        help="Current profile override. none disables currents; medium/strong use track current_profiles.",
    )
    parser.add_argument("--official", action="store_true", help="Force official sensor/timing mode.")
    parser.add_argument("--headless", action="store_true", help="Request headless HoloOcean mode when supported.")
    parser.add_argument("--record", action="store_true", help="Request HoloOcean recording when supported.")
    parser.add_argument("--log-dir", default="results/marine_race", help="Directory for JSONL and summary logs.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for beacons and adapters.")
    parser.add_argument("--dt", type=float, default=0.1, help="Race loop timestep.")
    parser.add_argument(
        "--disable-front-camera",
        action="store_true",
        help=(
            "Disable FrontCamera capture for non-official live/debug runs. "
            "Official runs keep the camera enabled."
        ),
    )
    parser.add_argument(
        "--show-front-camera",
        action="store_true",
        help=(
            "Display observation['sensors']['FrontCamera'] in a live viewer. "
            "Press V or Esc in the viewer window to close only the camera viewer."
        ),
    )
    parser.add_argument(
        "--print-beacons",
        action="store_true",
        help="Print de-duplicated received beacon packet diagnostics while the race loop runs.",
    )
    parser.add_argument(
        "--motion-compensation",
        choices=MOTION_COMPENSATION_MODES,
        default=MOTION_COMPENSATION_NONE,
        help="Optional high-level command compensation layer. Only 'none' ships; current compensation is future work.",
    )
    parser.add_argument(
        "--gate-timeout-s",
        type=float,
        default=None,
        help="Optional experiment safety stop: mark STUCK if no new gate is passed within this many seconds.",
    )
    parser.add_argument(
        "--staggered-start",
        action="store_true",
        help="Clone the base participant into multiple staggered-start rovers.",
    )
    parser.add_argument(
        "--num-rovers",
        type=int,
        default=1,
        help="Number of participants to generate when --staggered-start is enabled.",
    )
    parser.add_argument(
        "--start-gap-s",
        type=float,
        default=20.0,
        help="Start delay between generated staggered participants in seconds.",
    )
    parser.add_argument(
        "--staggered-lateral-offset-m",
        type=float,
        default=1.5,
        help="Lateral spacing between generated staggered participants in meters.",
    )
    parser.add_argument(
        "--log-participant-states",
        action="store_true",
        help="Log per-tick participant states for multi-agent diagnostics.",
    )
    parser.add_argument(
        "--inter-vehicle-collision-mode",
        choices=INTER_VEHICLE_COLLISION_MODES,
        default="off",
        help=(
            "Referee-side rover-rover proximity detector. off preserves existing scoring; "
            "diagnostic logs fleet events without penalty; penalize adds one team penalty per pair event."
        ),
    )
    parser.add_argument(
        "--inter-vehicle-collision-xy-threshold-m",
        type=float,
        default=0.8,
        help="Horizontal distance threshold for referee-side inter-vehicle collision detection.",
    )
    parser.add_argument(
        "--inter-vehicle-collision-z-threshold-m",
        type=float,
        default=0.75,
        help="Vertical distance threshold for referee-side inter-vehicle collision detection.",
    )
    parser.add_argument(
        "--inter-vehicle-collision-release-threshold-m",
        type=float,
        default=None,
        help="Horizontal separation required before the same rover pair can count again.",
    )
    parser.add_argument(
        "--inter-vehicle-collision-cooldown-s",
        type=float,
        default=1.0,
        help="Minimum time between inter-vehicle collision events for the same rover pair.",
    )
    parser.add_argument(
        "--team-id",
        default="fleet_01",
        help="Team identifier used in fleet-level summary and inter-vehicle collision events.",
    )
    parser.add_argument(
        "--comms-enabled",
        action="store_true",
        help="Enable the optional inter-rover acoustic communication channel (fleet only).",
    )
    parser.add_argument(
        "--comms-sound-speed-m-s",
        type=float,
        default=1500.0,
        help="Acoustic propagation speed used to compute range-dependent message latency.",
    )
    parser.add_argument(
        "--comms-max-range-m",
        type=float,
        default=100.0,
        help="Maximum range beyond which inter-rover messages are not delivered.",
    )
    parser.add_argument(
        "--comms-processing-delay-s",
        type=float,
        default=0.05,
        help="Fixed per-message processing delay added to the acoustic propagation latency.",
    )
    parser.add_argument(
        "--comms-packet-loss-prob",
        type=float,
        default=0.0,
        help="Per-link probability that a message is dropped (seeded, deterministic).",
    )
    parser.add_argument(
        "--comms-max-payload-bytes",
        type=int,
        default=128,
        help="Maximum JSON-serialized payload size; larger messages are dropped (bandwidth limit).",
    )
    parser.add_argument(
        "--comms-min-send-interval-s",
        type=float,
        default=0.5,
        help="Minimum time between transmissions from the same rover (half-duplex rate limit).",
    )
    return parser


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


def _with_staggered_participants(config: TrackConfig, args: argparse.Namespace) -> TrackConfig:
    if not args.staggered_start:
        if args.num_rovers != 1:
            raise ValueError("--num-rovers requires --staggered-start.")
        return config
    if args.num_rovers < 1:
        raise ValueError("--num-rovers must be at least 1.")
    if args.start_gap_s < 0.0:
        raise ValueError("--start-gap-s must be non-negative.")
    if args.staggered_lateral_offset_m < 0.0:
        raise ValueError("--staggered-lateral-offset-m must be non-negative.")
    if not config.participants:
        raise ValueError("Cannot generate staggered participants because the track has no base participant.")

    base = config.participants[0]
    offsets = _staggered_lateral_offsets(args.num_rovers, spacing_m=float(args.staggered_lateral_offset_m))
    participants = []
    for index in range(args.num_rovers):
        participant_id = f"bluerov2_{index + 1:02d}"
        spawn = dict(base.spawn)
        base_position = _vector3(spawn.get("position", config.start.position))
        base_rotation = _vector3(spawn.get("rotation_rpy_deg", config.start.rotation_rpy_deg))
        spawn["position"] = list(_offset_spawn_position(config, base_position, base_rotation[2], offsets[index]))
        spawn["rotation_rpy_deg"] = list(base_rotation)
        spawn["start_delay_s"] = float(index) * float(args.start_gap_s)
        participants.append(
            replace(
                base,
                id=participant_id,
                spawn=spawn,
                start_delay_s=float(spawn["start_delay_s"]),
            )
        )
    return replace(config, participants=participants)


def _staggered_lateral_offsets(num_rovers: int, spacing_m: float) -> list[float]:
    if num_rovers == 1:
        return [0.0]
    offsets = [0.0]
    for pair_index in range(1, num_rovers):
        magnitude = ((pair_index + 1) // 2) * spacing_m
        sign = -1.0 if pair_index % 2 == 1 else 1.0
        offsets.append(sign * magnitude)
    return offsets


def _offset_spawn_position(
    config: TrackConfig,
    base_position: Vector3,
    yaw_deg: float,
    lateral_offset_m: float,
) -> Vector3:
    yaw_rad = math.radians(yaw_deg)
    right_axis = (-math.sin(yaw_rad), math.cos(yaw_rad))
    bounds = config.world.bounds
    candidate = (
        base_position[0] + lateral_offset_m * right_axis[0],
        base_position[1] + lateral_offset_m * right_axis[1],
        base_position[2],
    )
    if bounds.contains(candidate):
        return candidate
    scale = 0.9
    while scale > 0.0:
        candidate = (
            base_position[0] + lateral_offset_m * scale * right_axis[0],
            base_position[1] + lateral_offset_m * scale * right_axis[1],
            base_position[2],
        )
        if bounds.contains(candidate):
            return candidate
        scale -= 0.1
    return (
        min(max(base_position[0], bounds.x_min), bounds.x_max),
        min(max(base_position[1], bounds.y_min), bounds.y_max),
        min(max(base_position[2], bounds.z_min), bounds.z_max),
    )


def _load_participants(config: TrackConfig, args: argparse.Namespace) -> Dict[str, RaceParticipant]:
    loader = ControllerLoader()
    participants: Dict[str, RaceParticipant] = {}
    for participant_config in config.participants:
        controller_reference = (
            args.participant_controller
            or args.controller
            or participant_config.controller
        )
        controller_class = args.controller_class or participant_config.controller_class
        constructor_kwargs = {"model_path": getattr(args, "controller_model_path", None)}
        controller = loader.load(
            controller_reference,
            controller_class=controller_class,
            constructor_kwargs=constructor_kwargs,
        )
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
        _print_multi_agent_diagnostics(adapter, participants, "after_reset")
        adapter.spawn_visual_gates(arena.visual_gates)
        adapter.spawn_obstacles(arena.obstacles)
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
            _print_multi_agent_diagnostics(fallback, participants, "after_reset")
            fallback.spawn_visual_gates(arena.visual_gates)
            fallback.spawn_obstacles(arena.obstacles)
            return fallback
        raise AdapterSelectionError(
            "HoloOcean adapter failed during environment setup and fallback is not allowed. "
            "Use --adapter fallback for the kinematic runner or pass --allow-fallback explicitly."
        ) from exc


def _print_multi_agent_diagnostics(
    adapter: BaseRaceAdapter,
    participants: Mapping[str, RaceParticipant],
    stage: str,
) -> None:
    if len(participants) <= 1:
        return
    diagnostics = adapter.diagnose_multi_agent_state(participants, stage)
    print(
        f"Multi-agent diagnostics ({adapter.name}, {stage}): "
        f"participant_ids={diagnostics['participant_ids']}"
    )
    participant_details = diagnostics.get("participants", {})
    if isinstance(participant_details, Mapping):
        for participant_id, detail in participant_details.items():
            if not isinstance(detail, Mapping):
                continue
            print(
                "  "
                f"{participant_id}: position={detail.get('position')} "
                f"sensors={detail.get('sensor_keys')}"
            )


def _mission_info(config: TrackConfig, participant_id: str) -> Dict[str, Any]:
    """Minimal static mission information given to a controller at reset.

    Contains only what defines the participant's assigned mission: its own
    identity, the first beacon, the beacon count, the lap count and the
    command envelope. For cooperative fleets it additionally carries the
    statically assigned release order and predecessor identity. It carries no
    referee state, timing mode, environment/current/obstacle metadata, world
    bounds, adapter name or benchmark mode.
    """
    info: Dict[str, Any] = {
        "participant_id": participant_id,
        "initial_beacon_id": "B01",
        "total_beacons": len(config.track.gate_sequence),
        "laps": config.race.laps,
        "command_limits": {
            "surge": [-0.95, 0.95],
            "sway": [-0.95, 0.95],
            "heave": [-0.95, 0.95],
            "yaw": [-0.95, 0.95],
        },
    }
    if len(config.participants) > 1:
        ordered = _participant_release_order(config)
        try:
            position = ordered.index(participant_id)
        except ValueError:
            position = None
        info["fleet"] = {
            "participant_order": ordered,
            "release_index": position,
            "predecessor_id": ordered[position - 1] if position not in (None, 0) else None,
        }
    return info


def _participant_release_order(config: TrackConfig) -> list[str]:
    """Statically assigned release order (start delay, then configured order)."""
    indexed = list(enumerate(config.participants))
    indexed.sort(key=lambda item: (float(getattr(item[1], "start_delay_s", 0.0) or 0.0), item[0]))
    return [participant.id for _, participant in indexed]


def _current_profile_label(config: TrackConfig) -> str:
    if config.selected_current_profile:
        return config.selected_current_profile
    return "track-default"


def _positive_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if converted <= 0.0:
        return None
    return converted


def _participant_start_delay_s(participant: RaceParticipant) -> float:
    configured = getattr(participant.config, "start_delay_s", 0.0)
    try:
        delay = float(configured)
    except (TypeError, ValueError):
        delay = 0.0
    return max(0.0, delay)


def _zero_command() -> dict[str, float]:
    return {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}


def _log_motion_compensation(
    *,
    referee: Referee,
    participant_id: str,
    time_s: float,
    compensator: Any,
    last_log_times: Dict[str, float],
    interval_s: float = 1.0,
) -> None:
    if getattr(compensator, "mode", MOTION_COMPENSATION_NONE) == MOTION_COMPENSATION_NONE:
        return
    logger = referee.logger
    if logger is None:
        return
    last_time = last_log_times.get(participant_id)
    if last_time is not None and time_s - last_time < interval_s:
        return
    diagnostics = compensator.diagnostics()
    last_log_times[participant_id] = time_s
    logger.log_event(
        "motion_compensation",
        time_s,
        participant_id,
        mode=getattr(compensator, "mode", "unknown"),
        **diagnostics.as_event_payload(),
    )


_LOCAL_STATE_LOG_INTERVAL_S = 1.0
_LOCAL_STATE_ADVANCE_COUNTS: Dict[int, int] = {}
_LOCAL_STATE_COORDINATION_STATES: Dict[int, tuple[Any, Any, Any]] = {}
_LOCAL_STATE_PHASES: Dict[int, Any] = {}
_BEACON_DIAGNOSTIC_LOG_INTERVAL_S = 1.0
_COMMAND_LOG_INTERVAL_S = 1.0
_LOCAL_FINISH_TAIL_S = 8.0


def _controller_tracker(controller: Any) -> Any:
    """The controller's LocalCourseTracker, if it exposes one (possibly wrapped)."""
    tracker = getattr(controller, "tracker", None)
    if tracker is not None:
        return tracker
    base = getattr(controller, "base", None)
    if base is not None:
        return getattr(base, "tracker", None)
    return None


def _controller_coordination_diagnostics(controller: Any) -> Dict[str, Any]:
    """Return controller-local coordination diagnostics, when available."""
    diagnostics = getattr(controller, "coordination_diagnostics", None)
    value = diagnostics() if callable(diagnostics) else diagnostics
    return dict(value) if isinstance(value, Mapping) else {}


def _controller_diagnostic_snapshot(controller: Any) -> Dict[str, Any]:
    """Combine local course and coordination estimates for offline logging."""
    snapshot: Dict[str, Any] = {}
    tracker = _controller_tracker(controller)
    if tracker is not None and callable(getattr(tracker, "diagnostics", None)):
        snapshot.update(dict(tracker.diagnostics()))
    coordination = _controller_coordination_diagnostics(controller)
    if coordination:
        snapshot["coordination"] = coordination
    return snapshot


def _needs_local_finish_tail(
    state: Any,
    controller: Any,
    time_s: float,
    *,
    grace_s: float = _LOCAL_FINISH_TAIL_S,
) -> bool:
    """Whether a referee-finished rover still needs onboard confirmation time.

    This scheduler decision never enters the controller observation. Referee
    scoring is already frozen; the bounded physical tail only lets the local
    tracker observe the range rise, rear-beacon packets and persistent
    post-gate disappearance that occur after the referee detects the final
    plane crossing.
    """

    if getattr(state, "status", None) != ParticipantStatus.FINISHED:
        return False
    finish_time_s = getattr(state, "official_finish_time", None)
    if finish_time_s is None:
        return False
    tracker = _controller_tracker(controller)
    if tracker is None or bool(getattr(tracker, "finished", False)):
        return False
    return float(time_s) <= float(finish_time_s) + max(0.0, float(grace_s)) + 1e-9


def _log_controller_local_state(
    *,
    referee: Referee,
    controller: Any,
    participant_id: str,
    time_s: float,
    last_log_times: Dict[str, float],
) -> None:
    """Log the controller's own progression estimate for offline analysis.

    Strictly one-way: the runner reads the tracker's diagnostics for the event
    log so local progression can be compared against referee truth afterwards.
    Nothing is ever written back into the controller or its observation.
    """
    logger = referee.logger
    if logger is None:
        return
    diagnostics = _controller_diagnostic_snapshot(controller)
    if not diagnostics:
        return
    advancements = int(diagnostics.get("advancements", 0) or 0)
    key = id(controller)
    advanced = advancements != _LOCAL_STATE_ADVANCE_COUNTS.get(key)
    phase = diagnostics.get("phase")
    phase_changed = phase != _LOCAL_STATE_PHASES.get(key)
    coordination = diagnostics.get("coordination")
    coordination_signature = None
    if isinstance(coordination, Mapping):
        coordination_signature = (
            coordination.get("is_holding"),
            coordination.get("hold_reason"),
            coordination.get("decision_reason"),
        )
    coordination_changed = (
        coordination_signature is not None
        and coordination_signature != _LOCAL_STATE_COORDINATION_STATES.get(key)
    )
    last_time = last_log_times.get(participant_id)
    if (
        not advanced
        and not phase_changed
        and not coordination_changed
        and last_time is not None
        and time_s - last_time < _LOCAL_STATE_LOG_INTERVAL_S
    ):
        return
    _LOCAL_STATE_ADVANCE_COUNTS[key] = advancements
    _LOCAL_STATE_PHASES[key] = phase
    if coordination_signature is not None:
        _LOCAL_STATE_COORDINATION_STATES[key] = coordination_signature
    last_log_times[participant_id] = time_s
    payload = {
        f"local_{name}" if not name.startswith("local_") else name: value
        for name, value in diagnostics.items()
        if name != "coordination"
    }
    if isinstance(coordination, Mapping):
        payload.update(
            {f"coordination_{name}": value for name, value in coordination.items()}
        )
    logger.log_event(
        "controller_local_state",
        time_s,
        participant_id,
        **payload,
    )


def _controller_local_summary(
    participants: Mapping[str, RaceParticipant],
) -> Dict[str, Dict[str, Any]]:
    """Final controller-local estimates, kept separate from referee truth."""
    result: Dict[str, Dict[str, Any]] = {}
    for participant_id, participant in participants.items():
        diagnostics = _controller_diagnostic_snapshot(participant.controller)
        if diagnostics:
            result[participant_id] = diagnostics
    return result


def _log_comms_deliveries(
    *,
    referee: Referee,
    receiver_id: str,
    inbox: Any,
) -> None:
    """Log acoustic arrival timing offline without copying controller payloads."""
    logger = referee.logger
    if logger is None or not isinstance(inbox, list):
        return
    for message in inbox:
        if not isinstance(message, Mapping):
            continue
        sent_at_s = _safe_float(message.get("sent_at_s"), float("nan"))
        received_at_s = _safe_float(message.get("received_at_s"), float("nan"))
        if not math.isfinite(sent_at_s) or not math.isfinite(received_at_s):
            continue
        logger.log_event(
            "comms_delivery",
            received_at_s,
            receiver_id,
            sender_id=str(message.get("from", "")),
            sent_at_s=sent_at_s,
            received_at_s=received_at_s,
            latency_s=max(0.0, received_at_s - sent_at_s),
        )


def _log_beacon_reception(
    *,
    referee: Referee,
    arena: Arena,
    participant_id: str,
    time_s: float,
    observation: Mapping[str, Any],
    last_log_times: Dict[str, float],
) -> None:
    """Write receiver-side beacon diagnostics without feeding them back.

    The event contains only delivered packet measurements plus cumulative
    transmitter/receiver counters.  It is an offline diagnostic stream and is
    never placed in the official observation.
    """
    logger = referee.logger
    if logger is None:
        return
    last_time = last_log_times.get(participant_id)
    if last_time is not None and time_s - last_time < _BEACON_DIAGNOSTIC_LOG_INTERVAL_S:
        return
    packets = observation.get("beacons")
    packet_list = packets if isinstance(packets, list) else []
    last_log_times[participant_id] = time_s
    logger.log_event(
        "beacon_reception_diagnostics",
        time_s,
        participant_id,
        local_time_s=_safe_float(observation.get("local_time_s"), 0.0),
        received_count=len(packet_list),
        received_beacon_ids=[
            str(packet.get("beacon_id"))
            for packet in packet_list
            if isinstance(packet, Mapping) and packet.get("beacon_id") is not None
        ],
        packets=[dict(packet) for packet in packet_list if isinstance(packet, Mapping)],
        cumulative=dict(arena.beacon_manager.diagnostics.get(participant_id, {})),
    )


def _log_controller_command(
    *,
    referee: Referee,
    participant_id: str,
    time_s: float,
    local_time_s: float,
    command: Mapping[str, Any],
    last_log_times: Dict[str, float],
) -> None:
    """Log the controller output for offline actuator diagnostics."""
    logger = referee.logger
    if logger is None:
        return
    last_time = last_log_times.get(participant_id)
    message = command.get("message")
    has_message = isinstance(message, Mapping)
    if (
        not has_message
        and last_time is not None
        and time_s - last_time < _COMMAND_LOG_INTERVAL_S
    ):
        return
    last_log_times[participant_id] = time_s
    logger.log_event(
        "controller_command",
        time_s,
        participant_id,
        local_time_s=local_time_s,
        command={
            axis: _safe_float(command.get(axis), 0.0)
            for axis in ("surge", "sway", "heave", "yaw")
        },
        message=dict(message) if has_message else None,
    )


def _log_participant_states(
    *,
    referee: Referee,
    adapter: BaseRaceAdapter,
    participants: Mapping[str, RaceParticipant],
    time_s: float,
) -> None:
    logger = referee.logger
    if logger is None:
        return
    for participant_id in participants:
        state = referee.states[participant_id]
        participant_state = adapter.get_participant_state(participant_id)
        target_gate_id = None
        if not state.is_terminal and state.valid_gate_crossings < len(referee.gate_sequence):
            target_gate_id = referee.expected_gate_id(participant_id)
        logger.log_event(
            "participant_state",
            time_s,
            participant_id,
            status=state.status.value,
            start_delay_s=state.start_delay_s,
            release_time_s=state.release_time_s,
            completed_gates=state.valid_gate_crossings,
            target_gate_id=target_gate_id,
            position=participant_state.position,
            rotation_rpy_deg=participant_state.rotation_rpy_deg,
        )


def _run_race_loop(
    config: TrackConfig,
    arena: Arena,
    referee: Referee,
    adapter: BaseRaceAdapter,
    participants: Mapping[str, RaceParticipant],
    dt: float,
    camera_viewer: "FrontCameraViewer | None" = None,
    received_beacon_printer: "ReceivedBeaconPrinter | None" = None,
    motion_compensators: Mapping[str, Any] | None = None,
    gate_timeout_s: float | None = None,
    log_participant_states: bool = False,
    comms_channel: "AcousticCommsChannel | None" = None,
) -> Dict[str, Any]:
    LOGGER.info(
        "Starting race '%s' with adapter '%s' in %s.",
        config.race.name,
        adapter.name,
        arena.environment_name,
    )
    race_start_time = adapter.get_current_time()
    start_delays = {
        participant_id: _participant_start_delay_s(participant)
        for participant_id, participant in participants.items()
    }
    referee.start_race(race_start_time, start_delays=start_delays)
    # Runner-side release clocks used for the participant-local observation
    # time base; scheduling is static configuration, not referee feedback.
    release_times = {
        participant_id: race_start_time + start_delays.get(participant_id, 0.0)
        for participant_id in participants
    }
    motion_compensation_log_times: Dict[str, float] = {}
    local_state_log_times: Dict[str, float] = {}
    beacon_reception_log_times: Dict[str, float] = {}
    controller_command_log_times: Dict[str, float] = {}
    gate_timeout_s = _positive_float_or_none(gate_timeout_s)
    last_gate_counts = {
        participant_id: referee.states[participant_id].valid_gate_crossings
        for participant_id in participants
    }
    last_gate_progress_times = {
        participant_id: adapter.get_current_time() + start_delays.get(participant_id, 0.0)
        for participant_id in participants
    }
    multi_agent_tick_checked = False
    race_deadline_s = float(config.race.max_duration_s)
    tail_hard_deadline_s = race_deadline_s + _LOCAL_FINISH_TAIL_S
    failed_finish_tails: set[str] = set()
    while adapter.get_current_time() <= tail_hard_deadline_s:
        all_terminal = True
        manual_stop_requested = False
        previous_states: Dict[str, AdapterParticipantState] = {}
        controller_errors: Dict[str, str] = {}
        outgoing_messages: Dict[str, Any] = {}
        current_time_s = adapter.get_current_time()
        race_window_open = current_time_s <= race_deadline_s + 1e-9
        for participant in participants.values():
            state = referee.states[participant.id]
            if state.status == ParticipantStatus.NOT_STARTED:
                if not race_window_open:
                    adapter.apply_command(participant.id, _zero_command(), participant.config.control_mode)
                    continue
                release_time_s = release_times[participant.id]
                if current_time_s + 1e-9 >= release_time_s:
                    referee.release_participant(participant.id, current_time_s)
                    release_times[participant.id] = current_time_s
                    last_gate_counts[participant.id] = referee.states[participant.id].valid_gate_crossings
                    last_gate_progress_times[participant.id] = current_time_s
                    state = referee.states[participant.id]
                else:
                    all_terminal = False
                    adapter.apply_command(participant.id, _zero_command(), participant.config.control_mode)
                    continue
            local_finish_tail = (
                participant.id not in failed_finish_tails
                and _needs_local_finish_tail(state, participant.controller, current_time_s)
            )
            if state.is_terminal and not local_finish_tail:
                # Never leave the final pre-terminal command latched in the
                # simulator while another rover is still racing.
                adapter.apply_command(participant.id, _zero_command(), participant.config.control_mode)
                continue
            if not race_window_open and not local_finish_tail:
                adapter.apply_command(participant.id, _zero_command(), participant.config.control_mode)
                continue
            all_terminal = False
            previous_state = adapter.get_participant_state(participant.id)
            previous_states[participant.id] = previous_state
            comms_inbox = (
                comms_channel.deliver(participant.id, current_time_s)
                if comms_channel is not None
                else None
            )
            _log_comms_deliveries(
                referee=referee,
                receiver_id=participant.id,
                inbox=comms_inbox,
            )
            observation = _build_controller_observation(
                config=config,
                arena=arena,
                adapter=adapter,
                participant=participant,
                participant_state=previous_state,
                release_time_s=release_times[participant.id],
                comms_inbox=comms_inbox,
            )
            _log_beacon_reception(
                referee=referee,
                arena=arena,
                participant_id=participant.id,
                time_s=current_time_s,
                observation=observation,
                last_log_times=beacon_reception_log_times,
            )
            if received_beacon_printer is not None:
                received_beacon_printer.update(observation, participant.id)
            if camera_viewer is not None:
                camera_viewer.update(observation, participant.id)
            try:
                command = participant.controller.step(_copy_observation_for_controller(observation))
            except ManualStopRequested:
                manual_stop_requested = True
                break
            except Exception as exc:  # pragma: no cover - exercised by external controllers
                controller_errors[participant.id] = f"{type(exc).__name__}: {exc}"
                if local_finish_tail:
                    failed_finish_tails.add(participant.id)
                command = {}
            if isinstance(command, Mapping):
                _log_controller_command(
                    referee=referee,
                    participant_id=participant.id,
                    time_s=current_time_s,
                    local_time_s=_safe_float(observation.get("local_time_s"), 0.0),
                    command=command,
                    last_log_times=controller_command_log_times,
                )
            _log_controller_local_state(
                referee=referee,
                controller=participant.controller,
                participant_id=participant.id,
                time_s=current_time_s,
                last_log_times=local_state_log_times,
            )
            if comms_channel is not None and isinstance(command, dict) and command.get("message") is not None:
                outgoing_messages[participant.id] = command.get("message")
                command = {key: value for key, value in command.items() if key != "message"}
            compensator = motion_compensators.get(participant.id) if motion_compensators is not None else None
            if compensator is not None:
                command = compensator.compensate(command, observation, dt)
                _log_motion_compensation(
                    referee=referee,
                    participant_id=participant.id,
                    time_s=adapter.get_current_time(),
                    compensator=compensator,
                    last_log_times=motion_compensation_log_times,
                )
            adapter.apply_command(participant.id, command, participant.config.control_mode)

        if manual_stop_requested:
            print("Manual stop requested.")
            referee.manual_stop(participants.keys(), adapter.get_current_time())
            break

        if all_terminal:
            break

        if comms_channel is not None and outgoing_messages:
            sender_positions = {
                participant_id: state.position
                for participant_id, state in previous_states.items()
            }
            for sender_id in sorted(outgoing_messages):
                sender_position = sender_positions.get(sender_id)
                if sender_position is None:
                    continue
                receiver_positions = {
                    rid: position
                    for rid, position in sender_positions.items()
                    if rid != sender_id
                }
                comms_channel.send(
                    sender_id=sender_id,
                    payload=outgoing_messages[sender_id],
                    send_time_s=current_time_s,
                    sender_position=sender_position,
                    receiver_positions=receiver_positions,
                )

        adapter.step(dt)
        if not multi_agent_tick_checked:
            _print_multi_agent_diagnostics(adapter, participants, "after_first_tick")
            multi_agent_tick_checked = True
        time_s = adapter.get_current_time()
        for participant_id, previous_state in previous_states.items():
            state = referee.states[participant_id]
            if state.is_terminal:
                continue
            current_state = adapter.get_participant_state(participant_id)
            obstacle_collisions = adapter.get_obstacle_collision_events(
                participant_id,
                previous_position=previous_state.position,
                current_position=current_state.position,
            )
            collision = adapter.get_collision_state(participant_id)
            referee.update(
                participant_id=participant_id,
                previous_position=previous_state.position,
                current_position=current_state.position,
                time_s=time_s,
                collision=collision and not obstacle_collisions,
                obstacle_collisions=obstacle_collisions,
                controller_error=controller_errors.get(participant_id),
            )
            if state.is_terminal:
                continue
            completed_gates = state.valid_gate_crossings
            if completed_gates > last_gate_counts.get(participant_id, 0):
                last_gate_counts[participant_id] = completed_gates
                last_gate_progress_times[participant_id] = time_s
            elif (
                gate_timeout_s is not None
                and time_s - last_gate_progress_times.get(participant_id, time_s) >= gate_timeout_s
            ):
                referee.gate_timeout_stuck(participant_id, time_s, gate_timeout_s)
        if len(participants) > 1 and time_s <= race_deadline_s + 1e-9:
            participant_positions = {
                participant_id: adapter.get_participant_state(participant_id).position
                for participant_id in participants
            }
            referee.detect_inter_vehicle_collisions(time_s, participant_positions)
        if log_participant_states:
            _log_participant_states(
                referee=referee,
                adapter=adapter,
                participants=participants,
                time_s=time_s,
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


class FrontCameraViewer:
    """Optional live display for controller-safe FrontCamera observations."""

    window_name = "Marine Race FrontCamera"

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.closed = False
        self._backend: str | None = None
        self._cv2: Any = None
        self._pygame: Any = None
        self._screen: Any = None
        self._frame_count = 0
        self._start_time = time.monotonic()
        self._missing_warning_printed = False
        self._invalid_warning_printed = False
        self._backend_warning_printed = False

    def update(self, observation: Mapping[str, Any], participant_id: str) -> None:
        if not self.enabled or self.closed:
            return
        sensors = observation.get("sensors", {})
        if not isinstance(sensors, Mapping):
            self._warn_missing([])
            return
        image = sensors.get("FrontCamera")
        if image is None:
            self._warn_missing(sensors.keys())
            return
        frame = self._image_to_uint8_array(image)
        if frame is None:
            if not self._invalid_warning_printed:
                print(
                    "FrontCamera viewer warning: FrontCamera exists but could not be converted "
                    f"for display. Type: {type(image).__name__}",
                    file=sys.stderr,
                )
                self._invalid_warning_printed = True
            return
        if self._backend is None and not self._initialize_backend(frame):
            return
        self._frame_count += 1
        fps = self._frame_count / max(1e-6, time.monotonic() - self._start_time)
        if self._backend == "opencv":
            self._show_with_opencv(frame, participant_id, fps)
        elif self._backend == "pygame":
            self._show_with_pygame(frame, participant_id, fps)

    def close(self) -> None:
        if self._backend == "opencv" and self._cv2 is not None:
            try:
                self._cv2.destroyWindow(self.window_name)
            except Exception:
                pass
        if self._backend == "pygame" and self._pygame is not None:
            try:
                self._pygame.display.quit()
            except Exception:
                pass
        self.closed = True

    def _warn_missing(self, sensor_keys: Any) -> None:
        if self._missing_warning_printed:
            return
        keys = sorted(str(key) for key in sensor_keys)
        print(
            "FrontCamera viewer warning: observation['sensors']['FrontCamera'] is missing. "
            f"Available sensor keys: {keys}",
            file=sys.stderr,
        )
        self._missing_warning_printed = True

    def _initialize_backend(self, frame: Any) -> bool:
        if self._try_initialize_opencv():
            return True
        return self._try_initialize_pygame(frame)

    def _try_initialize_opencv(self) -> bool:
        try:
            import cv2
        except ImportError:
            return False
        self._cv2 = cv2
        self._backend = "opencv"
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except Exception:
            pass
        return True

    def _try_initialize_pygame(self, frame: Any) -> bool:
        try:
            import pygame
        except ImportError:
            if not self._backend_warning_printed:
                print(
                    "FrontCamera viewer warning: OpenCV is not installed and pygame is not "
                    "available for fallback display. Install opencv-python in the ocean "
                    "environment or run without --show-front-camera.",
                    file=sys.stderr,
                )
                self._backend_warning_printed = True
            self.closed = True
            return False
        if pygame.get_init() and pygame.display.get_surface() is not None:
            if not self._backend_warning_printed:
                print(
                    "FrontCamera viewer warning: OpenCV is not installed and pygame already "
                    "has an active controller window. The fallback pygame camera viewer would "
                    "take over that display, so the camera viewer is disabled.",
                    file=sys.stderr,
                )
                self._backend_warning_printed = True
            self.closed = True
            return False
        pygame.init()
        height, width = int(frame.shape[0]), int(frame.shape[1])
        self._screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(self.window_name)
        self._pygame = pygame
        self._backend = "pygame"
        return True

    def _show_with_opencv(self, frame: Any, participant_id: str, fps: float) -> None:
        cv2 = self._cv2
        display = self._frame_for_opencv(frame)
        label = f"{participant_id}  frame={self._frame_count}  fps={fps:.1f}  V/Esc closes viewer"
        try:
            cv2.putText(
                display,
                label,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if hasattr(cv2, "setWindowTitle"):
                cv2.setWindowTitle(self.window_name, f"{self.window_name} | {label}")
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("v"), ord("V")):
                self.close()
        except Exception as exc:
            print(f"FrontCamera viewer warning: OpenCV display failed: {exc}", file=sys.stderr)
            self.close()

    def _show_with_pygame(self, frame: Any, participant_id: str, fps: float) -> None:
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_v):
                self.close()
                return
        rgb = self._frame_for_pygame(frame)
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        self._screen.blit(surface, (0, 0))
        pygame.display.set_caption(
            f"{self.window_name} | {participant_id} frame={self._frame_count} fps={fps:.1f} V/Esc closes"
        )
        pygame.display.flip()

    def _image_to_uint8_array(self, image: Any) -> Any:
        try:
            import numpy as np
        except ImportError:
            if not self._backend_warning_printed:
                print(
                    "FrontCamera viewer warning: numpy is required to display FrontCamera frames.",
                    file=sys.stderr,
                )
                self._backend_warning_printed = True
            self.closed = True
            return None
        try:
            frame = np.asarray(image)
        except Exception:
            return None
        if frame.ndim < 2:
            return None
        if frame.ndim == 2:
            frame = frame[:, :, None]
        if frame.ndim != 3 or frame.shape[2] not in (1, 3, 4):
            return None
        if frame.dtype != np.uint8:
            frame = frame.astype(np.float32, copy=False)
            max_value = float(np.nanmax(frame)) if frame.size else 0.0
            if max_value <= 1.0:
                frame = frame * 255.0
            frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
            frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
        return frame

    def _frame_for_opencv(self, frame: Any) -> Any:
        cv2 = self._cv2
        channels = frame.shape[2]
        if channels == 1:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # The local HoloOcean 2.3.0 FrontCamera runtime returns BGRA/BGR buffers
        # even though the Python docstring describes RGBA.
        return frame[:, :, :3].copy()

    def _frame_for_pygame(self, frame: Any) -> Any:
        if frame.shape[2] == 1:
            return frame.repeat(3, axis=2)
        if frame.shape[2] == 3:
            return frame[:, :, [2, 1, 0]]
        return frame[:, :, [2, 1, 0, 3]][:, :, :3]


class ReceivedBeaconPrinter:
    """De-duplicated received-packet diagnostics for manual debugging.

    Prints what the controller physically received (the ``beacons`` packet
    list); it has no access to referee targets because the observation no
    longer carries any.
    """

    def __init__(
        self,
        enabled: bool,
        periodic_interval_s: float = 2.0,
        stream: Any = None,
    ) -> None:
        self.enabled = enabled
        self.periodic_interval_s = max(0.0, float(periodic_interval_s))
        self.stream = stream if stream is not None else sys.stdout
        self._last_signature_by_participant: dict[str, tuple[object, ...]] = {}
        self._last_print_time_by_participant: dict[str, float] = {}

    def update(self, observation: Mapping[str, Any], participant_id: str) -> bool:
        if not self.enabled:
            return False
        time_s = _safe_float(observation.get("local_time_s"), 0.0)
        beacons = observation.get("beacons")
        packets = [packet for packet in beacons or [] if isinstance(packet, Mapping)]
        signature = tuple(sorted(str(packet.get("beacon_id")) for packet in packets))
        previous_signature = self._last_signature_by_participant.get(participant_id)
        previous_time = self._last_print_time_by_participant.get(participant_id)
        periodic_due = (
            previous_time is None
            or self.periodic_interval_s > 0.0
            and (time_s - previous_time) >= self.periodic_interval_s
        )
        if signature == previous_signature and not periodic_due:
            return False

        if packets:
            strongest = max(packets, key=lambda packet: _safe_float(packet.get("signal_strength"), 0.0))
            line = (
                f"[BEACONS] local_t={time_s:.1f}s participant={participant_id} "
                f"received={len(packets)} ids={','.join(signature)} "
                f"strongest={_display_value(strongest.get('beacon_id'))} "
                f"range={_display_number(strongest.get('range_m'))} "
                f"bearing={_display_number(strongest.get('bearing_deg'))} "
                f"elevation={_display_number(strongest.get('elevation_deg'))}"
            )
        else:
            line = f"[BEACONS] local_t={time_s:.1f}s participant={participant_id} no packets received"
        print(line, file=self.stream)
        self._last_signature_by_participant[participant_id] = signature
        self._last_print_time_by_participant[participant_id] = time_s
        return True


def _build_controller_observation(
    config: TrackConfig,
    arena: Arena,
    adapter: BaseRaceAdapter,
    participant: RaceParticipant,
    participant_state: AdapterParticipantState,
    release_time_s: float,
    comms_inbox: "list[Dict[str, Any]] | None" = None,
) -> Dict[str, Any]:
    """Build the official controller observation from onboard information only.

    The builder never receives (or reads) the referee: beacons are the packets
    physically received from the independent transmitters, the clock is the
    participant's own elapsed time since its release, and sensors are the
    adapter's allowed onboard sensor set.
    """
    time_s = adapter.get_current_time()
    local_time_s = max(0.0, time_s - release_time_s)
    beacon_packets = arena.beacon_manager.receive(
        receiver_id=participant.id,
        receiver_position=participant_state.position,
        receiver_yaw_deg=participant_state.rotation_rpy_deg[2],
        time_s=time_s,
        received_at_s=local_time_s,
    )
    sensor_data = adapter.get_allowed_sensor_data(participant.id, participant.config.sensors)
    debug_ground_truth = None
    if bool(getattr(participant.controller, "uses_ground_truth", False)) and not config.race.official_mode:
        debug_ground_truth = {
            "own_position": participant_state.position,
            "own_rotation_rpy_deg": participant_state.rotation_rpy_deg,
            "gates": [
                {
                    "gate_id": gate.id,
                    "center": gate.center,
                    "normal": gate.normal_vector,
                    "right_axis": gate.right_axis,
                    "up_axis": gate.up_axis,
                    "inner_width_m": gate.inner_width_m,
                    "inner_height_m": gate.inner_height_m,
                }
                for gate in (arena.gate_map[gate_id] for gate_id in config.track.gate_sequence)
            ],
            "bounds": {
                "x_min": config.world.bounds.x_min,
                "x_max": config.world.bounds.x_max,
                "y_min": config.world.bounds.y_min,
                "y_max": config.world.bounds.y_max,
                "z_min": config.world.bounds.z_min,
                "z_max": config.world.bounds.z_max,
            },
        }
    inbox = None
    if comms_inbox is not None:
        inbox = [_localized_inbox_message(message, release_time_s) for message in comms_inbox]
    return build_observation(
        local_time_s=local_time_s,
        sensor_data=sensor_data,
        beacon_packets=beacon_packets,
        official_mode=config.race.official_mode,
        comms_inbox=inbox,
        debug_ground_truth=debug_ground_truth,
    )


def _localized_inbox_message(message: Mapping[str, Any], release_time_s: float) -> Dict[str, Any]:
    """Re-stamp a delivered message onto the receiver's local clock.

    The channel works on the simulator clock; the controller only ever sees
    its own local time, so the delivery timestamp is converted and the
    sender's transmit timestamp (a global-clock value) is dropped.
    """
    received_global = message.get("received_at_s")
    local = None
    try:
        local = max(0.0, float(received_global) - release_time_s)
    except (TypeError, ValueError):
        local = 0.0
    return {
        "from": message.get("from"),
        "payload": message.get("payload"),
        "received_at_s": local,
    }


def _vector3(value: Any) -> Vector3:
    return (float(value[0]), float(value[1]), float(value[2]))


def _without_front_camera(config: TrackConfig) -> TrackConfig:
    participants = [
        replace(participant, sensors=_strip_front_camera_from_sensors(participant.sensors))
        for participant in config.participants
    ]
    return replace(config, participants=participants)


def _strip_front_camera_from_sensors(sensors: Any) -> Any:
    if not isinstance(sensors, Mapping):
        return sensors
    stripped = copy.deepcopy(dict(sensors))
    for key in ("allowed", "allowed_sensors", "sensors"):
        values = stripped.get(key)
        if isinstance(values, list):
            stripped[key] = [value for value in values if value != "FrontCamera"]
    holoocean_sensors = stripped.get("holoocean_sensors")
    if isinstance(holoocean_sensors, list):
        stripped["holoocean_sensors"] = [
            sensor
            for sensor in holoocean_sensors
            if not (
                isinstance(sensor, Mapping)
                and (
                    sensor.get("sensor_name") == "FrontCamera"
                    or sensor.get("sensor_type") == "RGBCamera"
                )
            )
        ]
    if stripped.get("profile") == "official_vision_acoustic":
        stripped["profile"] = "official_acoustic_no_front_camera"
    return stripped


def _copy_observation_for_controller(value: Any) -> Any:
    if _looks_like_image_array(value):
        return value
    if isinstance(value, dict):
        return {key: _copy_observation_for_controller(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_observation_for_controller(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_observation_for_controller(item) for item in value)
    return copy.deepcopy(value)


def _looks_like_image_array(value: Any) -> bool:
    shape = getattr(value, "shape", None)
    return shape is not None and len(shape) >= 2


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _display_number(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "NA"


def _display_value(value: Any) -> str:
    if value is None:
        return "NA"
    return str(value)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _print_summary(summary: Dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
