"""Bounded real-HoloOcean expert collection for the 27-D contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
from marine_race_arena.controllers.official_baselines import RuleGateCenterThenCommitController
from marine_race_arena.learning.config import ACTION_AXES, ACTION_DIM
from marine_race_arena.learning.config_local_transition_27d import (
    OBS_DIM_LOCAL_TRANSITION_27D,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
)
from marine_race_arena.learning.episode import RaceEpisode
from marine_race_arena.learning.gen2 import track_fragments as tf
from marine_race_arena.learning.gen2.dataset_27d import (
    Gen2Episode27d,
    load_corpus,
    write_manifest,
    save_episode,
    corpus_statistics,
)
from marine_race_arena.learning.gen2.fog_contract import APPROVED_WATER_FOG, assert_approved_water_fog, verify_fog_sources
from marine_race_arena.learning.tracker_context_local_transition_27d import OnboardLocalTransition27dContextTracker
from marine_race_arena.learning.observation_encoder_local_transition_27d import encode_observation_local_transition_27d


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def _select_fragment(track: str, start_index: int):
    fragments = tf.enumerate_fragments(track, lengths=(3,), stride=1)
    if not fragments:
        raise RuntimeError(f"no legal 3-gate fragment for {track}")
    return min(fragments, key=lambda fragment: abs(fragment.start_index - start_index))


def _sources(out: Path) -> list[Dict[str, Any]]:
    result: list[Dict[str, Any]] = []
    for track in tf.OFFICIAL_TRACKS:
        path = tf.track_path(track)
        assert_approved_water_fog(path)
        result.append({"name": track, "track": track, "path": path, "kind": "full_track", "fragment": None})
    # One exact contiguous window per circuit; starts cover straight, serpent,
    # and mixed/depth transitions without inventing geometry.
    starts = {"horseshoe_bay": 0, "vertical_serpent": 7, "mixed_endurance": 13}
    frag_dir = out / "fragments"
    for track, start in starts.items():
        fragment = _select_fragment(track, start)
        path = tf.materialize_fragment(fragment, frag_dir / f"{fragment.name}.json")
        assert_approved_water_fog(path)
        result.append({"name": fragment.name, "track": track, "path": path, "kind": "fragment", "fragment": fragment.as_dict()})
    return result


def collect_one(source: Mapping[str, Any], *, seed: int, max_steps: Optional[int] = None) -> Gen2Episode27d:
    path = Path(source["path"])
    assert_approved_water_fog(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    duration_steps = int(round(float(config["race"]["max_duration_s"]) / 0.1)) + 10
    episode = RaceEpisode(
        str(path), seed=seed, dt=0.1, adapter="holoocean", allow_fallback=False,
        headless=True, max_steps=int(max_steps or duration_steps), official=True,
        current_profile="none", benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
    )
    observations: list[np.ndarray] = []
    expert_actions: list[np.ndarray] = []
    progress_rows: list[int] = []
    previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
    vision_steps = orientation_steps = dvl_steps = 0
    collisions = 0
    started = time.perf_counter()
    try:
        raw = episode.reset(seed=seed)
        fragment = source.get("fragment")
        if fragment and source.get("source_type", source.get("kind")) in {"fragment", "exact_fragment"}:
            tf.apply_initial_body_velocity(episode, tf.inbound_body_velocity(tf.TrackFragment(**{k: fragment[k] for k in ("track", "start_index", "end_index", "gate_ids", "inbound_gate_id", "grown_for_link")})))
            raw = episode._build_observation()
        total = len(episode.context.config.track.gate_sequence)
        tracker = OnboardLocalTransition27dContextTracker(total_beacons=total, laps=int(episode.context.config.race.laps))
        tracker.reset(raw)
        expert = RuleGateCenterThenCommitController()
        expert.reset({"participant_id": episode.participant_id, "total_beacons": total, "laps": int(episode.context.config.race.laps)})
        for _ in range(int(max_steps or duration_steps)):
            command = expert.step(raw)
            context = tracker.context(raw, dt=episode.dt, prev_action=previous_action.tolist())
            encoded = encode_observation_local_transition_27d(raw, context)
            if encoded.shape != (OBS_DIM_LOCAL_TRANSITION_27D,) or not np.isfinite(encoded).all():
                raise RuntimeError(f"invalid 27-D observation at step {len(observations)}")
            action = np.asarray([float(command.get(axis, 0.0)) for axis in ACTION_AXES], dtype=np.float32)
            observations.append(encoded)
            expert_actions.append(action)
            progress_rows.append(int(episode.referee_progress()["valid_gate_crossings"]))
            vision_steps += int(context.visual_target is not None)
            orientation_steps += int(context.gate_orientation_present)
            sensors = raw.get("sensors") if isinstance(raw.get("sensors"), dict) else {}
            dvl_steps += int(sensors.get("DVLSensor") is not None)
            previous_action = action
            outcome = episode.step(command)
            collisions += int(outcome.collision) + int(outcome.obstacle_collisions)
            raw = outcome.observation
            if outcome.terminated or outcome.truncated:
                break
        final_progress = episode.referee_progress()
        status = str(final_progress.get("status", ""))
        completed = status.upper().endswith("FINISHED") or int(final_progress.get("valid_gate_crossings", 0)) >= total
        referee_state = episode.context.referee.states.get(episode.participant_id)
        meta = {
            "seed": int(seed), "source": str(source["name"]), "track": str(source["track"]),
            "kind": str(source.get("source_type", source["kind"])),
            "source_type": str(source.get("source_type", source["kind"])),
            "fragment": source.get("fragment"),
            "steps": len(observations), "gate_count": total,
            "gates_completed": int(final_progress.get("valid_gate_crossings", 0)), "completed": bool(completed),
            "status": status, "collisions": int(collisions),
            "out_of_bounds_events": int(getattr(referee_state, "out_of_bounds_events", 0)),
            "wrong_direction_crossings": int(getattr(referee_state, "wrong_direction_crossings", 0)),
            "missed_gate_attempts": int(getattr(referee_state, "missed_gate_attempts", 0)),
            "obstacle_collision_events": int(getattr(referee_state, "obstacle_collision_events", 0)),
            "vision_steps": vision_steps, "orientation_steps": orientation_steps, "dvl_steps": dvl_steps,
            "actual_adapter": episode.actual_adapter, "fallback_used": episode.fallback_used,
            "fog": dict(APPROVED_WATER_FOG), "currents": "disabled",
            "expert": "rule_gate_center_then_commit", "observation_contract": OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
            "git_sha": _git_sha(), "wall_time_s": round(time.perf_counter() - started, 2),
        }
        return Gen2Episode27d(seed=seed, observations=np.asarray(observations, np.float32), expert_actions=np.asarray(expert_actions, np.float32), applied_actions=np.asarray(expert_actions, np.float32), applied_by_expert=np.zeros(len(observations), dtype=bool), gate_crossings=np.asarray(progress_rows, np.int16), meta=meta)
    finally:
        episode.close()


def collect_corpus(out: str | Path, *, allow_long_collection: bool, episodes_per_source: int = 1, max_steps: Optional[int] = None, seed: int = 9100) -> Dict[str, Any]:
    if not allow_long_collection:
        raise RuntimeError("Official 27-D collection requires explicit allow_long_collection=True")
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    sources = _sources(root)
    fog = verify_fog_sources([Path(source["path"]) for source in sources])
    records = []
    for source_index, source in enumerate(sources):
        for repeat in range(int(episodes_per_source)):
            episode = collect_one(source, seed=int(seed + source_index * 100 + repeat), max_steps=max_steps)
            if episode.meta.get("actual_adapter") != "holoocean" or episode.meta.get("fallback_used"):
                raise RuntimeError(f"non-HoloOcean episode collected: {episode.meta}")
            save_episode(episode, root, tag=f"{source['kind']}_{source_index:02d}")
            records.append(episode)
            print(json.dumps({"source": source["name"], "steps": len(episode), "completed": episode.completed, "gates": episode.meta.get("gates_completed")}, indent=2), flush=True)
    manifest = write_manifest(root, episodes=records, extra={"fog": fog, "actual_adapter": "holoocean", "fallback_used": False, "git_sha": _git_sha(), "allow_long_collection": True, "episodes_per_source": int(episodes_per_source), "sources": [{k: v for k, v in source.items() if k != "path"} for source in sources]})
    report = {"manifest": str(manifest), "statistics": corpus_statistics(records), "fog": fog, "actual_adapter": "holoocean", "fallback_used": False, "diagnostic_only": False, "include_in_training_dataset": True}
    (root / "collection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded real-HoloOcean 27-D expert corpus")
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes-per-source", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument("--allow-long-collection", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = collect_corpus(args.out, allow_long_collection=args.allow_long_collection, episodes_per_source=args.episodes_per_source, max_steps=args.max_steps, seed=args.seed)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
