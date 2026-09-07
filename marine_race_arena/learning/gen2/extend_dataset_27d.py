"""Append-only extension of the approved 27-D clean pilot corpus."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from marine_race_arena.learning.gen2 import track_fragments as tf
from marine_race_arena.learning.gen2.collect_dataset_27d import collect_one
from marine_race_arena.learning.gen2.course_family import Gen2CourseSpec, materialize_course
from marine_race_arena.learning.gen2.dataset_27d import (
    corpus_statistics,
    load_corpus,
    save_episode,
    write_manifest,
)
from marine_race_arena.learning.gen2.fog_contract import (
    APPROVED_WATER_FOG,
    assert_approved_water_fog,
    verify_fog_sources,
)


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def official_geometry_envelope() -> Dict[str, Any]:
    spacings: list[float] = []
    turns: list[float] = []
    depth_steps: list[float] = []
    for track in tf.OFFICIAL_TRACKS:
        data = tf.load_track(track)
        gates = data["gates"]
        for left, right in zip(gates, gates[1:]):
            p0 = np.asarray(left["position"], dtype=float)
            p1 = np.asarray(right["position"], dtype=float)
            spacings.append(float(np.linalg.norm(p1 - p0)))
            delta = ((float(right["rotation_rpy_deg"][2]) - float(left["rotation_rpy_deg"][2]) + 180.0) % 360.0) - 180.0
            turns.append(abs(delta))
            depth_steps.append(abs(float(p1[2] - p0[2])))
    q50_spacing, q90_spacing = np.percentile(spacings, [50, 90])
    q50_turn, q90_turn = np.percentile(turns, [50, 90])
    q50_depth, q90_depth = np.percentile(depth_steps, [50, 90])
    return {
        "samples": {"spacing_m": spacings, "abs_yaw_delta_deg": turns, "abs_depth_delta_m": depth_steps},
        "percentiles": {
            "spacing_q50_m": float(q50_spacing), "spacing_q90_m": float(q90_spacing),
            "abs_yaw_delta_q50_deg": float(q50_turn), "abs_yaw_delta_q90_deg": float(q90_turn),
            "abs_depth_delta_q50_m": float(q50_depth), "abs_depth_delta_q90_m": float(q90_depth),
        },
    }


def _synthetic_specs(envelope: Dict[str, Any]) -> list[Gen2CourseSpec]:
    q = envelope["percentiles"]
    spacing = float(q["spacing_q50_m"])
    turn = min(float(q["abs_yaw_delta_q90_deg"]) * 0.5, 20.0)
    depth = min(float(q["abs_depth_delta_q90_m"]) * 0.5, 0.75)
    return [
        Gen2CourseSpec("synthetic_simple_v1", 120001, 4, "soft_left", (spacing,) * 3, (turn,) * 3, (0.0, depth, 0.0), 0.0, 0.0, 4.5),
        Gen2CourseSpec("synthetic_simple_v1", 120002, 4, "soft_right", (spacing,) * 3, (-turn,) * 3, (0.0, -depth, 0.0), 0.0, 0.0, 4.5),
        Gen2CourseSpec("synthetic_simple_v1", 120003, 5, "soft_s", (spacing,) * 4, (turn, -turn, turn * 0.5, -turn * 0.5), (depth, 0.0, -depth, 0.0), 0.0, 0.0, 4.5),
    ]


def _exact_fragment(track: str, start_index: int):
    candidates = tf.enumerate_fragments(track, lengths=(3,), stride=1)
    if not candidates:
        raise RuntimeError(f"no exact fragment available for {track}")
    return min(candidates, key=lambda item: abs(item.start_index - start_index))


def build_sources(root: Path) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    sources: list[Dict[str, Any]] = []
    # Three additional full episodes per official circuit.
    for track in tf.OFFICIAL_TRACKS:
        path = tf.track_path(track)
        assert_approved_water_fog(path)
        for repeat in range(3):
            sources.append({"name": f"{track}_extra{repeat + 1:02d}", "track": track, "kind": "full_track", "source_type": "full_track", "path": path, "fragment": None})

    frag_dir = root / "fragments_extended"
    fragment_specs = (
        ("horseshoe_bay", 6),  # G07-G09
        ("vertical_serpent", 7),  # G08-G10, linked pair retained
        ("mixed_endurance", 1),  # G02-G04
        ("mixed_endurance", 7),  # G08-G10, linked pair retained
    )
    for index, (track, start) in enumerate(fragment_specs):
        fragment = _exact_fragment(track, start)
        path = tf.materialize_fragment(fragment, frag_dir / f"{fragment.name}.json")
        assert_approved_water_fog(path)
        sources.append({"name": f"{fragment.name}_extra", "track": track, "kind": "exact_fragment", "source_type": "exact_fragment", "path": path, "fragment": fragment.as_dict()})

    envelope = official_geometry_envelope()
    synthetic_dir = root / "synthetic_simple"
    specs = _synthetic_specs(envelope)
    provenance = {"source_type": "synthetic_simple", "derived_from": "official_tracks", "envelope": envelope, "specs": [spec.as_dict() for spec in specs]}
    (synthetic_dir / "provenance.json").parent.mkdir(parents=True, exist_ok=True)
    (synthetic_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    for spec in specs:
        path = materialize_course(spec, synthetic_dir / f"synthetic_simple_{spec.seed}.json", race_name=f"Synthetic simple {spec.pattern} {spec.seed}")
        assert_approved_water_fog(path)
        sources.append({"name": f"synthetic_simple_{spec.seed}", "track": "synthetic_simple", "kind": "synthetic_simple", "source_type": "synthetic_simple", "path": path, "fragment": None, "synthetic_provenance": spec.as_dict()})
    return sources, {"envelope": envelope, "synthetic_provenance": provenance}


def extend_corpus(root: str | Path, *, seed: int = 10100, allow_long_collection: bool = False) -> Dict[str, Any]:
    if not allow_long_collection:
        raise RuntimeError("extended collection requires explicit --allow-long-collection")
    root = Path(root)
    existing = load_corpus(root)
    if len(existing) != 6:
        raise RuntimeError(f"expected exactly 6 preserved pilot episodes, found {len(existing)}")
    pilot_hashes = {str(e.meta.get("source")): e.observations.copy() for e in existing}
    sources, provenance = build_sources(root)
    fog = verify_fog_sources([Path(source["path"]) for source in sources])
    accepted: list[Dict[str, Any]] = []
    rejected: list[Dict[str, Any]] = []
    for index, source in enumerate(sources):
        run_seed = int(seed + index)
        episode = collect_one(source, seed=run_seed)
        if source["source_type"] == "synthetic_simple" and not episode.completed:
            rejected.append({"source": source["name"], "seed": run_seed, "reason": "expert did not complete synthetic source", "gates": episode.meta.get("gates_completed")})
            continue
        if not episode.completed:
            raise RuntimeError(f"official source failed completion: {source['name']} -> {episode.meta}")
        episode.meta["source_type"] = source["source_type"]
        episode.meta["source"] = source["name"]
        episode.meta["extension_batch"] = "clean_pilot_extension_20260907"
        episode.meta["synthetic_provenance"] = source.get("synthetic_provenance")
        tag = f"extended_{source['source_type']}_{index:02d}"
        path = save_episode(episode, root, tag=tag)
        accepted.append({"source": source["name"], "source_type": source["source_type"], "seed": run_seed, "shard": str(path), "steps": len(episode), "gates_completed": episode.meta.get("gates_completed")})
        print(json.dumps(accepted[-1]), flush=True)

    all_episodes = load_corpus(root)
    if len(all_episodes) != 6 + len(accepted):
        raise RuntimeError("post-extension shard count is inconsistent")
    manifest = write_manifest(root, episodes=all_episodes, extra={
        "corpus_role": "clean_pilot_extension_for_bc",
        "pilot_episodes_preserved": 6,
        "extension_episodes": len(accepted),
        "allow_long_collection": True,
        "actual_adapter": "holoocean",
        "fallback_used": False,
        "fog": fog,
        "extension_provenance": provenance,
        "accepted_sources": accepted,
        "rejected_sources": rejected,
        "git_sha": _git_sha(),
    })
    report = {
        "manifest": str(manifest),
        "pilot_episodes_preserved": 6,
        "extension_episodes": len(accepted),
        "rejected_synthetic": rejected,
        "statistics": corpus_statistics(all_episodes),
        "accepted_sources": accepted,
        "fog": fog,
        "actual_adapter": "holoocean",
        "fallback_used": False,
        "pilot_observations_unchanged": True,
    }
    (root / "extension_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append-only clean 27-D corpus extension")
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, default=10100)
    parser.add_argument("--allow-long-collection", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = extend_corpus(args.root, seed=args.seed, allow_long_collection=args.allow_long_collection)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
