"""Compact deterministic audit of training/evaluation track geometry."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np

from marine_race_arena.learning.longrun_checkpoint import atomic_write_json
from marine_race_arena.learning.provenance import now_utc
from marine_race_arena.learning.transition_curriculum import (
    DIFFICULTY_LEVELS,
    TransitionGeometry,
    TransitionGeometrySampler,
)


def _row(geometry: TransitionGeometry, *, sampler: str) -> Dict[str, Any]:
    turns = np.abs(np.asarray(geometry.turn_deltas_deg, dtype=float))
    signed_turns = np.asarray(geometry.turn_deltas_deg, dtype=float)
    elevation = np.abs(np.asarray(geometry.vertical_deltas_m, dtype=float))
    signed_elevation = np.asarray(geometry.vertical_deltas_m, dtype=float)
    spacings = np.asarray(geometry.spacings_m, dtype=float)
    visibility_loss = int((turns > 45.0).sum())
    return {
        "sampler": sampler,
        "difficulty": geometry.difficulty,
        "episode_type": geometry.episode_type,
        "pattern": geometry.pattern,
        "seed": geometry.seed,
        "gate_count": geometry.gate_count,
        "mean_turn_angle_deg": float(turns.mean()) if turns.size else 0.0,
        "maximum_turn_angle_deg": float(turns.max()) if turns.size else 0.0,
        "mean_elevation_change_m": float(elevation.mean()) if elevation.size else 0.0,
        "maximum_elevation_change_m": float(elevation.max()) if elevation.size else 0.0,
        "mean_gate_spacing_m": float(spacings.mean()) if spacings.size else 0.0,
        "minimum_gate_spacing_m": float(spacings.min()) if spacings.size else 0.0,
        "maximum_gate_spacing_m": float(spacings.max()) if spacings.size else 0.0,
        "initial_lateral_offset_m": geometry.initial_lateral_offset_m,
        "initial_orientation_difference_deg": abs(geometry.initial_yaw_error_deg),
        "visibility_loss_transitions": visibility_loss,
        "expected_off_camera_target_acquisitions": visibility_loss,
        "positive_turns": int((signed_turns > 0).sum()),
        "negative_turns": int((signed_turns < 0).sum()),
        "positive_elevation_changes": int((signed_elevation > 0).sum()),
        "negative_elevation_changes": int((signed_elevation < 0).sum()),
    }


def generate_rows(*, tracks_per_difficulty: int, seed: int, sampler: str) -> list[Dict[str, Any]]:
    rows = []
    for index, difficulty in enumerate(DIFFICULTY_LEVELS):
        generator = TransitionGeometrySampler(
            seed=int(seed) + index * 1_000_003,
            difficulty=difficulty,
            transition_focus_fraction=0.70 if sampler == "training" else 1.0,
        )
        for sample_index in range(int(tracks_per_difficulty)):
            force = None
            if sampler == "evaluation":
                # Match the 100 transition + 12 full-sequence intermediate mix.
                force = (
                    "transition_focus"
                    if sample_index % 112 < 100 else "full_sequence"
                )
            rows.append(_row(generator.sample(force_episode_type=force), sampler=sampler))
    return rows


def _mean(rows: Iterable[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return statistics.fmean(values) if values else 0.0


def summarize(rows: list[Mapping[str, Any]]) -> Dict[str, Any]:
    count = max(1, len(rows))
    difficulty = Counter(str(row["difficulty"]) for row in rows)
    episode_type = Counter(str(row["episode_type"]) for row in rows)
    patterns = Counter(str(row["pattern"]) for row in rows)
    positions = Counter()
    for row in rows:
        for position in range(1, int(row["gate_count"])):
            positions[position] += 1
    positive_turns = sum(int(row["positive_turns"]) for row in rows)
    negative_turns = sum(int(row["negative_turns"]) for row in rows)
    positive_elevation = sum(int(row["positive_elevation_changes"]) for row in rows)
    negative_elevation = sum(int(row["negative_elevation_changes"]) for row in rows)
    extreme_fraction = sum(
        float(row["maximum_turn_angle_deg"]) >= 0.9 * 55.0
        or float(row["maximum_elevation_change_m"]) >= 0.9 * 2.0
        for row in rows
    ) / count
    return {
        "tracks": len(rows),
        "mean_gate_count": _mean(rows, "gate_count"),
        "mean_turn_angle_deg": _mean(rows, "mean_turn_angle_deg"),
        "maximum_turn_angle_deg": max(float(row["maximum_turn_angle_deg"]) for row in rows),
        "mean_elevation_change_m": _mean(rows, "mean_elevation_change_m"),
        "maximum_elevation_change_m": max(float(row["maximum_elevation_change_m"]) for row in rows),
        "mean_gate_spacing_m": _mean(rows, "mean_gate_spacing_m"),
        "minimum_gate_spacing_m": min(float(row["minimum_gate_spacing_m"]) for row in rows),
        "maximum_gate_spacing_m": max(float(row["maximum_gate_spacing_m"]) for row in rows),
        "mean_absolute_initial_offset_m": statistics.fmean(
            abs(float(row["initial_lateral_offset_m"])) for row in rows
        ),
        "mean_initial_orientation_difference_deg": _mean(
            rows, "initial_orientation_difference_deg"
        ),
        "visibility_loss_transitions": sum(int(row["visibility_loss_transitions"]) for row in rows),
        "expected_off_camera_target_acquisitions": sum(
            int(row["expected_off_camera_target_acquisitions"]) for row in rows
        ),
        "difficulty_percent": {key: 100.0 * difficulty[key] / count for key in DIFFICULTY_LEVELS},
        "episode_type_percent": {
            key: 100.0 * value / count for key, value in sorted(episode_type.items())
        },
        "pattern_percent": {key: 100.0 * value / count for key, value in sorted(patterns.items())},
        "transition_position_counts": dict(sorted(positions.items())),
        "direction_balance": {
            "positive_turns": positive_turns,
            "negative_turns": negative_turns,
            "positive_elevation_changes": positive_elevation,
            "negative_elevation_changes": negative_elevation,
        },
        "extreme_geometry_fraction": extreme_fraction,
        "diagnostics": {
            "too_easy_warning": _mean(rows, "mean_turn_angle_deg") < 3.0,
            "extreme_dominated_warning": extreme_fraction > 0.35,
            "turn_direction_imbalance_warning": abs(positive_turns - negative_turns) / max(1, positive_turns + negative_turns) > 0.15,
            "elevation_direction_imbalance_warning": abs(positive_elevation - negative_elevation) / max(1, positive_elevation + negative_elevation) > 0.15,
            "later_transition_underrepresentation": (
                positions.get(2, 0) < 0.25 * max(1, positions.get(1, 0))
            ),
        },
    }


def _figures(rows: list[Mapping[str, Any]], output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("turn_angle_histogram.png", "maximum_turn_angle_deg", "Maximum turn [deg]"),
        ("elevation_histogram.png", "maximum_elevation_change_m", "Maximum elevation change [m]"),
        ("gate_count_histogram.png", "gate_count", "Gate count"),
    )
    paths = []
    for name, key, label in specifications:
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.hist([float(row[key]) for row in rows], bins=20, color="tab:blue", alpha=0.8)
        axis.set_xlabel(label)
        axis.set_ylabel("Tracks")
        axis.grid(alpha=0.2)
        target = output / name
        figure.tight_layout()
        figure.savefig(target, dpi=140)
        plt.close(figure)
        paths.append(str(target))
    return paths


def _markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Universal-transition track distribution", ""]
    for name in ("training", "evaluation"):
        summary = report[name]
        lines += [
            f"## {name.title()}", "",
            f"Tracks: {summary['tracks']}",
            f"Mean gate count: {summary['mean_gate_count']:.2f}",
            f"Mean/max turn: {summary['mean_turn_angle_deg']:.2f}/{summary['maximum_turn_angle_deg']:.2f} deg",
            f"Mean/max elevation change: {summary['mean_elevation_change_m']:.3f}/{summary['maximum_elevation_change_m']:.3f} m",
            f"Episode mix: {summary['episode_type_percent']}",
            f"Diagnostics: {summary['diagnostics']}", "",
        ]
    lines.append("This report is diagnostic only; it does not automatically simplify the sampler.")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks-per-difficulty", type=int, default=500)
    parser.add_argument("--training-seed", type=int, default=43001)
    parser.add_argument("--evaluation-seed", type=int, default=88001)
    parser.add_argument(
        "--output-dir",
        default="results/rl_public/universal_transition_track_distribution",
    )
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    training_rows = generate_rows(
        tracks_per_difficulty=args.tracks_per_difficulty,
        seed=args.training_seed, sampler="training",
    )
    evaluation_rows = generate_rows(
        tracks_per_difficulty=args.tracks_per_difficulty,
        seed=args.evaluation_seed, sampler="evaluation",
    )
    all_rows = training_rows + evaluation_rows
    report = {
        "schema_version": "universal_transition_track_distribution_v1",
        "generated_utc": now_utc(),
        "tracks_per_difficulty": args.tracks_per_difficulty,
        "training": summarize(training_rows),
        "evaluation": summarize(evaluation_rows),
        "figures": _figures(all_rows, output / "figures"),
        "sampler_changed": False,
    }
    atomic_write_json(output / "distribution.json", report)
    markdown = output / "distribution.md"
    temporary = markdown.with_suffix(".md.tmp")
    temporary.write_text(_markdown(report), encoding="utf-8")
    temporary.replace(markdown)
    print(json.dumps({"output": str(output), "training": report["training"], "evaluation": report["evaluation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
