"""CLI for fast geometry previews and real rendered transition evaluations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from marine_race_arena.learning.longrun_checkpoint import atomic_write_json
from marine_race_arena.learning.transition_curriculum import (
    DIFFICULTY_LEVELS,
    FULL_SEQUENCE_LENGTHS,
)
from marine_race_arena.learning.transition_track_visualization import (
    holoocean_flythrough,
    render_geometry_track,
    rendered_long_sequence_evaluation,
    sample_geometries,
    snapshot_json,
    track_summary,
    write_sampled_tracks,
)


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in str(value).split(",") if item.strip()]


def _difficulties(value: str) -> list[str]:
    rows = list(DIFFICULTY_LEVELS) if value == "all" else [item.strip() for item in value.split(",")]
    unknown = sorted(set(rows) - set(DIFFICULTY_LEVELS))
    if unknown:
        raise ValueError(f"unknown difficulties {unknown}")
    return rows


def _episode_types(value: str) -> list[str]:
    rows = ["transition_focus", "full_sequence"] if value == "all" else [item.strip() for item in value.split(",")]
    unknown = sorted(set(rows) - {"transition_focus", "full_sequence"})
    if unknown:
        raise ValueError(f"unknown episode types {unknown}")
    return rows


def _record_path(base: Path, index: int, backend: str) -> Path:
    suffix = ".mp4"
    return base / f"preview_{index:05d}_{backend}{suffix}"


def _display_track(
    track_path: Path,
    *,
    backend: str,
    speed: float,
    loop: bool,
    pause: float,
    show_gate_frames: bool,
    show_camera_frustum: bool,
    record: bool,
    output_dir: Path,
    index: int,
    live: bool,
    camera: str,
    show: bool,
) -> dict[str, Any]:
    if backend == "geometry":
        return render_geometry_track(
            snapshot_json(track_path), speed=speed, loop=loop,
            pause_between_tracks=pause, show_gate_frames=show_gate_frames,
            show_camera_frustum=show_camera_frustum,
            record=_record_path(output_dir, index, backend) if record else None,
            show=show,
        )
    return holoocean_flythrough(
        track_path, speed=speed, live=live,
        output_video=_record_path(output_dir, index, backend) if record else None,
        camera=camera,
    )


def sampler_command(args: argparse.Namespace) -> int:
    # Loading the config validates that this preview describes a real training
    # contract, but geometry generation reuses the canonical sampler directly.
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("observation_version") != "onboard_local_transition_v1":
        raise ValueError("preview config is not a local-transition run")
    output = Path(args.output_dir)
    geometries = sample_geometries(
        seed_start=args.seed_start,
        num_tracks=args.num_tracks,
        difficulties=_difficulties(args.difficulty),
        episode_types=_episode_types(args.episode_type),
        lengths=_csv_ints(args.lengths),
    )
    paths = write_sampled_tracks(
        geometries, output / "tracks",
        frames_per_sec=config.get("holoocean_frames_per_sec", False),
    )
    summaries = []
    for index, path in enumerate(paths):
        summary = _display_track(
            path, backend=args.backend, speed=args.speed, loop=args.loop,
            pause=args.pause_between_tracks,
            show_gate_frames=args.show_gate_frames,
            show_camera_frustum=args.show_camera_frustum,
            record=args.record, output_dir=output / "videos", index=index,
            live=args.live, camera=args.camera, show=not args.no_show,
        )
        summaries.append(summary)
    atomic_write_json(output / "preview_manifest.json", {
        "schema_version": "transition_track_preview_v1",
        "backend": args.backend,
        "tracks": summaries,
    })
    print(json.dumps({"tracks": len(paths), "output": str(output)}, indent=2))
    return 0


def _active_paths(run: Path, workers: str) -> list[Path]:
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    identities = list(status.get("worker_identities") or [])
    selected = None if workers == "all" else set(_csv_ints(workers))
    paths = []
    for identity in identities:
        worker_id = int(identity["worker_id"])
        if selected is not None and worker_id not in selected:
            continue
        # Resolve from the run directory, never from a possibly foreign cwd.
        path = run / "workers" / f"worker_{worker_id:02d}" / "generated_tracks" / "active_episode.json"
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
    if not paths:
        raise ValueError("no active worker tracks selected")
    return paths


def active_command(args: argparse.Namespace) -> int:
    run = Path(args.run).resolve()
    output = Path(args.output_dir)
    summaries = []
    for index, source in enumerate(_active_paths(run, args.workers)):
        snapshot = snapshot_json(source)
        summary = track_summary(snapshot)
        if args.backend == "geometry":
            result = render_geometry_track(
                snapshot, speed=args.speed, loop=args.loop,
                pause_between_tracks=args.pause_between_tracks,
                show_gate_frames=args.show_gate_frames,
                show_camera_frustum=args.show_camera_frustum,
                record=_record_path(output / "videos", index, args.backend) if args.record else None,
                show=not args.no_show,
            )
        else:
            # The worker may atomically replace active_episode.json at any time;
            # HoloOcean always receives the immutable snapshot copied here.
            snapshot_path = output / "snapshots" / f"worker_{index:02d}.json"
            atomic_write_json(snapshot_path, snapshot)
            result = holoocean_flythrough(
                snapshot_path, speed=args.speed, live=args.live,
                output_video=_record_path(output / "videos", index, args.backend) if args.record else None,
                camera=args.camera,
            )
        summaries.append({"source": str(source), **summary, "render": result})
    atomic_write_json(output / "active_preview_manifest.json", {
        "schema_version": "active_transition_track_preview_v1",
        "run": str(run), "tracks": summaries,
    })
    print(json.dumps({"run": str(run), "tracks": summaries}, indent=2))
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    report = rendered_long_sequence_evaluation(
        algorithm=args.algorithm,
        run=args.run,
        checkpoint=args.checkpoint,
        lengths=_csv_ints(args.lengths),
        episodes_per_length=args.episodes_per_length,
        output_dir=args.output_dir,
        difficulty=args.difficulty,
        seed=args.seed,
        live=args.live,
        record=args.record,
        camera=args.camera,
    )
    print(json.dumps({
        "output": str(Path(args.output_dir) / "evaluation.json"),
        "metrics": report["metrics"],
    }, indent=2))
    return 0


def _common_preview_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=("geometry", "holoocean"), default="geometry")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--pause-between-tracks", type=float, default=0.0)
    parser.add_argument("--show-gate-frames", action="store_true")
    parser.add_argument("--show-camera-frustum", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--camera", choices=("spectator", "chase", "first_person"), default="spectator")
    parser.add_argument("--no-show", action="store_true", help=argparse.SUPPRESS)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    sampler = commands.add_parser("sampler")
    sampler.add_argument("--config", required=True)
    sampler.add_argument("--difficulty", default="all")
    sampler.add_argument("--episode-type", default="all")
    sampler.add_argument("--lengths", default=",".join(map(str, FULL_SEQUENCE_LENGTHS)))
    sampler.add_argument("--seed-start", type=int, default=43001)
    sampler.add_argument("--num-tracks", type=int, default=100)
    sampler.add_argument("--output-dir", default="results/rl/universal_transition/previews/sampler")
    _common_preview_options(sampler)

    active = commands.add_parser("active")
    active.add_argument("--run", required=True)
    active.add_argument("--workers", default="all")
    active.add_argument("--output-dir", default="results/rl/universal_transition/previews/active")
    _common_preview_options(active)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--algorithm", choices=("ppo", "sac"), required=True)
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--checkpoint", default="best_universal_transition")
    evaluate.add_argument("--lengths", default=",".join(map(str, FULL_SEQUENCE_LENGTHS)))
    evaluate.add_argument("--episodes-per-length", type=int, default=2)
    evaluate.add_argument("--backend", choices=("holoocean",), default="holoocean")
    evaluate.add_argument("--camera", choices=("spectator", "chase", "first_person"), default="chase")
    evaluate.add_argument("--live", action="store_true")
    evaluate.add_argument("--record", action="store_true")
    evaluate.add_argument("--difficulty", default="G6")
    evaluate.add_argument("--seed", type=int, default=88001)
    evaluate.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "sampler":
        return sampler_command(args)
    if args.command == "active":
        return active_command(args)
    return evaluate_command(args)


if __name__ == "__main__":
    raise SystemExit(main())

