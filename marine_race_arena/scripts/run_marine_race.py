"""Run a marine race through a simulator adapter."""

from __future__ import annotations

import argparse
import copy
import json
import logging
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
    effective_obstacle_mode,
)
from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_MODES
from marine_race_arena.config.loader import TrackConfigLoadError, load_track_config
from marine_race_arena.config.schema import TrackConfig, Vector3
from marine_race_arena.participants.controller_interface import ManualStopRequested
from marine_race_arena.participants.controller_loader import ControllerError, ControllerLoader
from marine_race_arena.participants.participant import RaceParticipant
from marine_race_arena.participants.sensor_profile import build_observation
from marine_race_arena.referee.logger import RaceLogger
from marine_race_arena.referee.race_state import ParticipantStatus
from marine_race_arena.referee.referee import Referee

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
            seed=args.seed,
        )
    except (TrackConfigLoadError, ValueError) as exc:
        print(f"Track validation failed: {exc}", file=sys.stderr)
        return 1
    if args.official and args.disable_front_camera:
        print("Race setup failed: --disable-front-camera is not allowed in official mode.", file=sys.stderr)
        return 1

    config = _with_cli_overrides(config, duration_s=args.duration, official=args.official)
    if args.disable_front_camera:
        config = _without_front_camera(config)
    arena = ArenaBuilder(config, seed=args.seed).build()
    logger = RaceLogger(args.log_dir, config.race.name, track_file=args.track)
    referee = Referee(config, arena.gate_map, arena.bounds, logger=logger)
    camera_viewer = FrontCameraViewer(enabled=args.show_front_camera)
    beacon_target_printer = BeaconTargetPrinter(
        enabled=args.print_beacon_targets or _env_flag("MARINE_RACE_PRINT_BEACON_TARGETS")
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
            camera_viewer=camera_viewer,
            beacon_target_printer=beacon_target_printer,
        )
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
            "oracle, acoustic, acoustic_baseline, acoustic_vision_baseline, "
            "rule_gate_baseline, vision_gate_baseline, student_template. Overrides track config."
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
        "--print-beacon-targets",
        action="store_true",
        help="Print de-duplicated beacon/race target diagnostics while the race loop runs.",
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
            fallback.spawn_visual_gates(arena.visual_gates)
            fallback.spawn_obstacles(arena.obstacles)
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
        "benchmark_task": config.benchmark_task.mode,
        "obstacle_mode": effective_obstacle_mode(config),
        "obstacle_density": config.obstacle_generation.density,
        "obstacle_physics": config.obstacle_generation.obstacle_physics,
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
    camera_viewer: "FrontCameraViewer | None" = None,
    beacon_target_printer: "BeaconTargetPrinter | None" = None,
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
        manual_stop_requested = False
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
            if beacon_target_printer is not None:
                beacon_target_printer.update(observation, participant.id)
            if camera_viewer is not None:
                camera_viewer.update(observation, participant.id)
            try:
                command = participant.controller.step(_copy_observation_for_controller(observation))
            except ManualStopRequested:
                manual_stop_requested = True
                break
            except Exception as exc:  # pragma: no cover - exercised by external controllers
                controller_errors[participant.id] = f"{type(exc).__name__}: {exc}"
                command = {}
            adapter.apply_command(participant.id, command, participant.config.control_mode)

        if manual_stop_requested:
            print("Manual stop requested.")
            referee.manual_stop(participants.keys(), adapter.get_current_time())
            break

        if all_terminal:
            break

        adapter.step(dt)
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


class BeaconTargetPrinter:
    """De-duplicated beacon/race target diagnostics for manual debugging."""

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
        time_s = _safe_float(observation.get("time_s"), 0.0)
        beacon = observation.get("beacon")
        race = observation.get("race")
        beacon_map = beacon if isinstance(beacon, Mapping) else {}
        race_map = race if isinstance(race, Mapping) else {}
        available = bool(beacon_map.get("valid")) and bool(
            beacon_map.get("target_gate_id") or race_map.get("target_gate_id")
        )
        target = str(race_map.get("target_gate_id") or beacon_map.get("target_gate_id") or "")
        index = race_map.get("target_sequence_index", beacon_map.get("sequence_index"))
        status = race_map.get("status")
        completed = race_map.get("completed_gates")
        signature = (available, target if available else None, index if available else None, status, completed)
        previous_signature = self._last_signature_by_participant.get(participant_id)
        previous_time = self._last_print_time_by_participant.get(participant_id)
        periodic_due = (
            previous_time is None
            or self.periodic_interval_s > 0.0
            and (time_s - previous_time) >= self.periodic_interval_s
        )
        if signature == previous_signature and not periodic_due:
            return False

        if available:
            line = (
                f"[BEACON] t={time_s:.1f}s participant={participant_id} "
                f"status={_display_value(status)} target={target} index={_display_value(index)} "
                f"range={_display_number(beacon_map.get('range_m'))} "
                f"bearing={_display_number(beacon_map.get('bearing_deg'))} "
                f"elevation={_display_number(beacon_map.get('elevation_deg'))} "
                f"completed={_display_value(completed)}"
            )
        else:
            line = f"[BEACON] t={time_s:.1f}s participant={participant_id} target unavailable"
        print(line, file=self.stream)
        self._last_signature_by_participant[participant_id] = signature
        self._last_print_time_by_participant[participant_id] = time_s
        return True


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
