"""Validate a marine race arena JSON track file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marine_race_arena.arena.obstacle import (
    OBSTACLE_DENSITIES,
    OBSTACLE_MODES,
    ObstacleConfigError,
    resolve_active_obstacles,
)
from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_MODES
from marine_race_arena.config.loader import TrackConfigLoadError, load_track_config
from marine_race_arena.config.validation import compute_declared_path_length_m, validate_track_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, help="Path to a marine race track JSON file.")
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
        help="Obstacle mode for validation.",
    )
    parser.add_argument(
        "--obstacle-density",
        choices=OBSTACLE_DENSITIES,
        default=None,
        help="Density for generated random obstacles.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic random obstacles.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print validation errors but return success for exploratory debugging.",
    )
    args = parser.parse_args(argv)

    try:
        config = load_track_config(
            args.track,
            debug=True,
            benchmark_task=args.benchmark_task,
            obstacles=args.obstacles,
            obstacle_density=args.obstacle_density,
            seed=args.seed,
        )
    except TrackConfigLoadError as exc:
        print(f"Track parse failed: {exc}", file=sys.stderr)
        return 1

    result = validate_track_config(config)
    computed_length = compute_declared_path_length_m(config)
    print(f"Track: {config.race.name}")
    print(f"Environment: {config.world.map}")
    print(f"Benchmark task: {config.benchmark_task.mode or 'custom'}")
    try:
        active_obstacle_count = len(resolve_active_obstacles(config))
    except ObstacleConfigError:
        active_obstacle_count = 0
    print(f"Active obstacles: {active_obstacle_count}")
    print(f"Gates per lap: {len(config.track.gate_sequence)}")
    print(f"Laps: {config.race.laps}")
    print(f"Declared path length: {config.track.declared_length_m:.2f} m")
    print(f"Computed path length: {computed_length:.2f} m")

    for warning in result.warnings:
        print(f"WARNING: {warning}")
    for error in result.errors:
        print(f"ERROR: {error}", file=sys.stderr)

    if result.errors and not args.debug:
        return 1
    print("Validation passed." if not result.errors else "Validation completed in debug mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
