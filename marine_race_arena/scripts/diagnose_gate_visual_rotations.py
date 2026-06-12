"""Spawn visual gate rotation test grids in HoloOcean and save screenshots."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image

from marine_race_arena.adapters.holoocean_adapter import HoloOceanRaceAdapter
from marine_race_arena.adapters.visual_spawner import HoloOceanVisualSpawner
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.arena.gate import Gate
from marine_race_arena.arena.gate_factory import GateBar, GateFactory
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.participants.participant import RaceParticipant


Vector3 = Tuple[float, float, float]


ROTATION_MAPPINGS: Dict[str, Callable[[Vector3], Vector3]] = {
    "rpy": lambda rpy: (rpy[0], rpy[1], rpy[2]),
    "pitch_yaw_roll": lambda rpy: (rpy[1], rpy[2], rpy[0]),
    "yaw_pitch_roll": lambda rpy: (rpy[2], rpy[1], rpy[0]),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, help="Track JSON used for the HoloOcean world.")
    parser.add_argument(
        "--output-dir",
        default="diagnostics/gate_rotation_tests",
        help="Directory for PNG screenshots and transform metadata.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Viewport capture width.")
    parser.add_argument("--height", type=int, default=720, help="Viewport capture height.")
    parser.add_argument("--ticks", type=int, default=8, help="Ticks to render after spawning props.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Hide the viewport if HoloOcean supports hidden rendering.",
    )
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Spawn only the selected library rotation mapping instead of comparison rows.",
    )
    parser.add_argument(
        "--single-long-only",
        action="store_true",
        help="Save one screenshot per yaw/pitch case using four single long box props per gate.",
    )
    parser.add_argument(
        "--fixed-front-camera",
        action="store_true",
        help="Use a fixed front camera for single-long screenshots instead of rotating with each gate normal.",
    )
    parser.add_argument(
        "--rotation-mapping",
        choices=sorted(ROTATION_MAPPINGS),
        default="yaw_pitch_roll",
        help="Rotation mapping to use for --single-long-only screenshots.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import holoocean

    config = load_track_config(args.track)
    arena = ArenaBuilder(config).build()
    participants = _participants_from_config(config)
    participant_id = next(iter(participants))

    adapter = HoloOceanRaceAdapter(config, arena, headless=args.headless)
    adapter.initialize()
    adapter._participants = participants
    adapter._states = {}
    environment_name = adapter._environment_candidates()[0]
    scenario = adapter._build_scenario(environment_name)
    scenario["window_width"] = args.width
    scenario["window_height"] = args.height
    scenario["agents"][0]["sensors"].append(
        {
            "sensor_type": "ViewportCapture",
            "Hz": 30,
            "configuration": {"CaptureWidth": args.width, "CaptureHeight": args.height},
        }
    )

    env = holoocean.make(
        scenario_cfg=scenario,
        show_viewport=not args.headless,
        ticks_per_sec=30,
        frames_per_sec=True,
    )
    try:
        state = env.reset()
        print(f"HoloOcean: {getattr(holoocean, '__version__', 'unknown')}")
        print(f"Environment: {environment_name}")
        print(f"Initial state keys: {list(state.keys()) if isinstance(state, dict) else type(state)}")

        bars, metadata = _build_rotation_test_bars(config)
        if args.single_long_only:
            saved_images = _save_single_long_bar_case_images(
                env,
                bars,
                metadata,
                participant_id,
                output_dir,
                args.ticks,
                fixed_front_camera=args.fixed_front_camera,
                rotation_mapping=args.rotation_mapping,
            )
            metadata_path = output_dir / "single_long_bar_transforms.json"
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "track": str(args.track),
                        "environment": environment_name,
                        "saved_images": saved_images,
                        "bars": metadata,
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
            print(f"Saved {metadata_path}")
            return 0

        _spawn_bars_for_all_mappings(env, bars, metadata, selected_only=args.selected_only)

        saved_images: List[str] = []
        default_state = env.tick(num_ticks=max(1, args.ticks))
        default_image = _extract_viewport_image(default_state, participant_id)
        if default_image is None:
            print("WARN: ViewportCapture missing for default agent view.")
        else:
            image_path = output_dir / "gate_rotation_agent_default.png"
            Image.fromarray(default_image).save(image_path)
            saved_images.append(str(image_path))
            print(f"Saved {image_path}")

        # Keep the camera far enough to see the whole test grid. These poses are
        # saved independently because HoloOcean viewport rotation behavior can
        # differ between builds.
        camera_poses = [
            ("front_oblique", (-8.0, -22.0, -0.8), (0.0, -12.0, 55.0)),
            ("top_oblique", (10.0, -20.0, 8.0), (-25.0, -35.0, 60.0)),
            ("side", (-14.0, 4.0, -1.0), (0.0, -8.0, 0.0)),
        ]

        for pose_name, location, rotation in camera_poses:
            move_viewport = getattr(env, "move_viewport", None)
            if callable(move_viewport):
                move_viewport(list(location), list(rotation))
            frame_state = env.tick(num_ticks=max(1, args.ticks))
            image = _extract_viewport_image(frame_state, participant_id)
            if image is None:
                print(f"WARN: ViewportCapture missing for pose {pose_name}.")
                continue
            image_path = output_dir / f"gate_rotation_{pose_name}.png"
            Image.fromarray(image).save(image_path)
            saved_images.append(str(image_path))
            print(f"Saved {image_path}")

        metadata_path = output_dir / "gate_rotation_transforms.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "track": str(args.track),
                    "environment": environment_name,
                    "rotation_mappings": list(ROTATION_MAPPINGS),
                    "saved_images": saved_images,
                    "bars": metadata,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        print(f"Saved {metadata_path}")
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return 0


def _participants_from_config(config: Any) -> Dict[str, RaceParticipant]:
    participants: Dict[str, RaceParticipant] = {}
    for participant_config in config.participants:
        spawn = participant_config.spawn or {}
        position = tuple(float(value) for value in spawn.get("position", config.start.position))
        rotation = tuple(float(value) for value in spawn.get("rotation_rpy_deg", config.start.rotation_rpy_deg))
        participants[participant_config.id] = RaceParticipant(
            config=participant_config,
            controller=object(),
            position=position,
            rotation_rpy_deg=rotation,
        )
    return participants


def _build_rotation_test_bars(config: Any) -> tuple[list[GateBar], list[dict[str, Any]]]:
    factory = GateFactory(config)
    rotations = [
        ("yaw_0_pitch_0", 0.0, 0.0),
        ("yaw_45_pitch_0", 45.0, 0.0),
        ("yaw_90_pitch_0", 90.0, 0.0),
        ("yaw_45_pitch_20", 45.0, 20.0),
        ("yaw_135_pitch_minus_20", 135.0, -20.0),
    ]
    bars: list[GateBar] = []
    metadata: list[dict[str, Any]] = []
    base_x = -2.0
    for index, (case_id, yaw_deg, pitch_deg) in enumerate(rotations):
        center = (base_x + index * 4.5, 0.0, -4.0)
        direction = _direction_from_yaw_pitch(yaw_deg, pitch_deg)
        gate = Gate(
            id=case_id,
            type="single",
            center=center,
            rotation_rpy_deg=(0.0, pitch_deg, yaw_deg),
            inner_width_m=1.5,
            inner_height_m=1.5,
            bar_thickness_m=0.18,
            color="white",
            passage_direction=direction,
        )
        visual_gate = factory.build_visual_gate(gate)
        bars.extend(visual_gate.bars)
        for bar in visual_gate.bars:
            metadata.append(
                {
                    "case_id": case_id,
                    "gate_id": gate.id,
                    "part": bar.part,
                    "position": list(bar.position),
                    "center": list(center),
                    "case_index": index,
                    "rotation_rpy_deg": list(bar.rotation_rpy_deg),
                    "dimensions_m": list(bar.dimensions_m),
                    "passage_direction": list(direction),
                    "yaw_deg": yaw_deg,
                    "pitch_deg": pitch_deg,
                }
            )
    return bars, metadata


def _spawn_bars_for_all_mappings(
    env: Any,
    bars: Iterable[GateBar],
    metadata: list[dict[str, Any]],
    selected_only: bool = False,
) -> None:
    material_by_mapping = {
        "rpy": "white",
        "pitch_yaw_roll": "gold",
        "yaw_pitch_roll": "steel",
    }
    y_offsets = {
        "rpy": -5.0,
        "pitch_yaw_roll": 0.0,
        "yaw_pitch_roll": 5.0,
    }
    spawn_prop = getattr(env, "spawn_prop")
    source_bars = list(bars)
    source_metadata = list(metadata)
    metadata.clear()
    if selected_only:
        moved_bars: list[GateBar] = []
        for bar, original in zip(source_bars, source_metadata):
            original_center = original["center"]
            target_center = (2.0, -8.0 + float(original["case_index"]) * 4.0, -4.0)
            offset = (
                bar.position[0] - original_center[0],
                bar.position[1] - original_center[1],
                bar.position[2] - original_center[2],
            )
            position = (
                target_center[0] + offset[0],
                target_center[1] + offset[1],
                target_center[2] + offset[2],
            )
            moved_bars.append(
                GateBar(
                    id=bar.id,
                    gate_id=bar.gate_id,
                    part=bar.part,
                    position=position,
                    rotation_rpy_deg=bar.rotation_rpy_deg,
                    dimensions_m=bar.dimensions_m,
                    color=bar.color,
                )
            )
        spawner = HoloOceanVisualSpawner(env)
        spawner.spawn_gate_bars(moved_bars)
        metadata.extend(spawner.spawned_props)
        return

    mappings = {"rpy": ROTATION_MAPPINGS["rpy"]} if selected_only else ROTATION_MAPPINGS
    for mapping_name, mapper in mappings.items():
        for bar, original in zip(source_bars, source_metadata):
            if selected_only:
                original_center = original["center"]
                target_center = (2.0, -8.0 + float(original["case_index"]) * 4.0, -4.0)
                offset = (
                    bar.position[0] - original_center[0],
                    bar.position[1] - original_center[1],
                    bar.position[2] - original_center[2],
                )
                position = (
                    target_center[0] + offset[0],
                    target_center[1] + offset[1],
                    target_center[2] + offset[2],
                )
            else:
                position = (bar.position[0], bar.position[1] + y_offsets[mapping_name], bar.position[2])
            rotation = mapper(bar.rotation_rpy_deg)
            tag = f"{mapping_name}_{bar.id}"
            spawn_prop(
                "box",
                location=list(position),
                rotation=list(rotation),
                scale=list(bar.dimensions_m),
                sim_physics=False,
                material=material_by_mapping[mapping_name],
                tag=tag,
            )
            item = dict(original)
            item.update(
                {
                    "mapping": mapping_name,
                    "tag": tag,
                    "spawn_position": list(position),
                    "spawn_rotation_deg": list(rotation),
                    "material": material_by_mapping[mapping_name],
                }
            )
            metadata.append(item)


def _save_single_long_bar_case_images(
    env: Any,
    bars: Iterable[GateBar],
    metadata: list[dict[str, Any]],
    participant_id: str,
    output_dir: Path,
    ticks: int,
    fixed_front_camera: bool = False,
    rotation_mapping: str = "rpy",
) -> list[str]:
    grouped: dict[str, list[tuple[GateBar, dict[str, Any]]]] = {}
    for bar, item in zip(bars, metadata):
        grouped.setdefault(str(item["case_id"]), []).append((bar, item))

    spawn_prop = getattr(env, "spawn_prop")
    saved_images: list[str] = []
    updated_metadata: list[dict[str, Any]] = []
    for case_index, (case_id, case_bars) in enumerate(grouped.items()):
        env.reset()
        target_center = (2.0, 0.0, -4.0)
        for bar, item in case_bars:
            original_center = item["center"]
            offset = (
                bar.position[0] - original_center[0],
                bar.position[1] - original_center[1],
                bar.position[2] - original_center[2],
            )
            position = (
                target_center[0] + offset[0],
                target_center[1] + offset[1],
                target_center[2] + offset[2],
            )
            rotation = ROTATION_MAPPINGS[rotation_mapping](tuple(float(value) for value in bar.rotation_rpy_deg))
            tag = f"single_long_{case_id}_{bar.part}"
            spawn_prop(
                "box",
                location=list(position),
                rotation=list(rotation),
                scale=list(bar.dimensions_m),
                sim_physics=False,
                material="white",
                tag=tag,
            )
            updated = dict(item)
            updated.update(
                {
                    "mapping": f"single_long_{rotation_mapping}",
                    "rotation_mapping": rotation_mapping,
                    "tag": tag,
                    "spawn_position": list(position),
                    "spawn_rotation_deg": list(rotation),
                    "spawn_scale": list(bar.dimensions_m),
                }
            )
            updated_metadata.append(updated)

        if fixed_front_camera:
            camera_location = (-5.5, 0.0, -4.0)
            camera_rotation = (0.0, 0.0, 0.0)
        else:
            first_item = case_bars[0][1]
            normal = _normalize(tuple(float(value) for value in first_item["passage_direction"]))
            camera_location = (
                target_center[0] - normal[0] * 6.0,
                target_center[1] - normal[1] * 6.0,
                target_center[2] - normal[2] * 6.0,
            )
            camera_rotation = (
                0.0,
                -math.degrees(math.asin(max(-1.0, min(1.0, normal[2])))),
                math.degrees(math.atan2(normal[1], normal[0])),
            )
        move_viewport = getattr(env, "move_viewport", None)
        if callable(move_viewport):
            move_viewport(list(camera_location), list(camera_rotation))
        frame_state = env.tick(num_ticks=max(1, ticks))
        image = _extract_viewport_image(frame_state, participant_id)
        if image is not None:
            image_path = output_dir / f"single_long_{case_index:02d}_{case_id}.png"
            Image.fromarray(image).save(image_path)
            saved_images.append(str(image_path))
            print(f"Saved {image_path}")

    metadata.clear()
    metadata.extend(updated_metadata)
    return saved_images


def _direction_from_yaw_pitch(yaw_deg: float, pitch_deg: float) -> Vector3:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    return (
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
    )


def _normalize(vector: Vector3) -> Vector3:
    length = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
    if length <= 1e-12:
        return (1.0, 0.0, 0.0)
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _extract_viewport_image(state: Any, participant_id: str) -> Any:
    if not isinstance(state, dict):
        return None
    sensors = state.get(participant_id)
    if not isinstance(sensors, dict):
        sensors = state
    image = sensors.get("ViewportCapture")
    if image is None:
        return None
    if hasattr(image, "shape") and len(image.shape) == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    return image


if __name__ == "__main__":
    raise SystemExit(main())
