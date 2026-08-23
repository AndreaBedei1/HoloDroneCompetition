"""Evaluate the learned controller on track fragments and on full circuits.

Two measurements, one purpose:

``evaluate_fragments``
    Runs every extracted window and returns a PASS/FAIL matrix indexed by
    track, fragment length and start gate.  This is the curriculum: a failing
    window names an exact transition to train on.

``evaluate_circuit``
    Runs a complete circuit end to end.  This is the campaign's primary
    progress metric -- **full track completion** -- and it does not wait on any
    procedural benchmark.

The failure map turns repeated circuit failures into fragment targets.  If
Mixed Endurance keeps dying at gate 7, the map says so, and the next collection
round weights G04-G08, G05-G09 and G06-G10 accordingly.  No geometry is
invented; the targets are windows of the real course.

Everything here drives the learned controller only::

    action = learned_policy(observation, recurrent_state)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.gen2 import seeds as gen2_seeds
from marine_race_arena.learning.gen2 import track_fragments as tf

#: Rule-controller reference times on the three circuits (seconds).
RULE_BASELINE_TIME_S: Dict[str, float] = {
    "horseshoe_bay": 225.9,
    "vertical_serpent": 290.6,
    "mixed_endurance": 472.9,
}

#: Fragment-competence targets used to decide whether to move on.
FRAGMENT_TARGETS: Dict[int, float] = {2: 1.00, 3: 1.00, 4: 0.95, 5: 0.95}

STEPS_PER_GATE = 900


def _load_controller(checkpoint: str | Path):
    from sb3_contrib import RecurrentPPO

    from marine_race_arena.learning.gen2.recurrent_policy import Gen2RecurrentController

    return Gen2RecurrentController(
        RecurrentPPO.load(str(checkpoint), device="cpu"), deterministic=True
    )


@dataclass
class FragmentOutcome:
    track: str
    fragment: str
    length: int
    start_gate: int          # 1-based gate number in the original circuit
    end_gate: int
    trial: int
    seed: int
    gates_completed: int
    gate_count: int
    completed: bool
    status: str
    failure: Optional[str]
    steps: int
    collisions: int
    out_of_bounds: int
    wrong_direction: int
    time_s: float
    path_length_m: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _classify_failure(episode) -> Optional[str]:
    """One label per failed episode, in priority order."""
    if episode.succeeded:
        return None
    if int(episode.out_of_bounds_events) > 0:
        return "out_of_bounds"
    if int(episode.collision_events) > 0:
        return "collision"
    if int(episode.wrong_direction_crossings) > 0:
        return "wrong_direction"
    if int(episode.missed_gate_attempts) > 0:
        return "missed_gate"
    if bool(episode.timeout):
        return "timeout"
    return "unknown"


def evaluate_fragments(
    checkpoint: str | Path,
    *,
    out_dir: str | Path,
    lengths: Sequence[int] = (2, 3, 4, 5),
    tracks: Sequence[str] = tuple(tf.OFFICIAL_TRACKS),
    trials: int = 1,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
    fragments: Optional[Sequence[tf.TrackFragment]] = None,
    shard: int = 0,
    shards: int = 1,
) -> Dict[str, Any]:
    """Run the learned controller on every fragment; return the PASS/FAIL matrix.

    ``shard``/``shards`` split the window list across concurrent processes.
    The split is by index, so every shard sees the same seeds it would have
    seen sequentially and the shards can simply be concatenated.
    """
    from marine_race_arena.learning.gen2.evaluation import run_policy_episode

    out_dir = Path(out_dir)
    track_dir = out_dir / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)
    controller = _load_controller(checkpoint)
    windows = list(fragments) if fragments is not None else tf.all_fragments(
        lengths, tracks=tracks
    )
    pool = gen2_seeds.GEN2_TRACK_EVAL_SEEDS
    rows: List[FragmentOutcome] = []
    started = time.perf_counter()
    suffix = "" if int(shards) <= 1 else f"_shard{int(shard):02d}"

    for index, fragment in enumerate(windows):
        if int(shards) > 1 and index % int(shards) != int(shard):
            continue
        for trial in range(int(trials)):
            seed = pool[(index * max(1, int(trials)) + trial) % len(pool)]
            path = tf.materialize_fragment(fragment, track_dir / f"{fragment.name}.json")
            episode = run_policy_episode(
                controller, path, seed=int(seed), spec=None,
                adapter=adapter, allow_fallback=allow_fallback,
                max_steps=max(600, fragment.length * STEPS_PER_GATE),
                initial_body_velocity=tf.inbound_body_velocity(fragment),
            )
            rows.append(FragmentOutcome(
                track=fragment.track, fragment=fragment.name, length=fragment.length,
                start_gate=fragment.start_index + 1, end_gate=fragment.end_index + 1,
                trial=trial, seed=int(seed),
                gates_completed=episode.gates_completed, gate_count=episode.gate_count,
                completed=episode.succeeded, status=episode.status,
                failure=_classify_failure(episode), steps=episode.steps,
                collisions=episode.collision_events,
                out_of_bounds=episode.out_of_bounds_events,
                wrong_direction=episode.wrong_direction_crossings,
                time_s=episode.completion_time_s,
                path_length_m=episode.path_length_m,
            ))
        (out_dir / f"fragment_matrix{suffix}.json").write_text(
            json.dumps([r.as_dict() for r in rows], indent=2), encoding="utf-8"
        )

    report = summarize_fragments(rows)
    report["wall_time_s"] = round(time.perf_counter() - started, 1)
    report["checkpoint"] = str(checkpoint)
    (out_dir / f"fragment_report{suffix}.json").write_text(
        json.dumps({"summary": report, "rows": [r.as_dict() for r in rows]}, indent=2),
        encoding="utf-8",
    )
    return report


def collect_fragment_shards(out_dir: str | Path) -> Dict[str, Any]:
    """Merge ``fragment_matrix_shard*.json`` into one report."""
    out_dir = Path(out_dir)
    rows: List[FragmentOutcome] = []
    for path in sorted(out_dir.glob("fragment_matrix*.json")):
        if path.name == "fragment_matrix.json" and list(out_dir.glob("fragment_matrix_shard*.json")):
            continue
        for payload in json.loads(path.read_text(encoding="utf-8")):
            rows.append(FragmentOutcome(**payload))
    report = summarize_fragments(rows)
    (out_dir / "fragment_report.json").write_text(
        json.dumps({"summary": report, "rows": [r.as_dict() for r in rows]}, indent=2),
        encoding="utf-8",
    )
    return report


def summarize_fragments(rows: Sequence[FragmentOutcome]) -> Dict[str, Any]:
    if not rows:
        return {"fragments": 0}
    by_track_length: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        bucket = by_track_length.setdefault(row.track, {}).setdefault(
            str(row.length), {"trials": 0, "passed": 0}
        )
        bucket["trials"] += 1
        bucket["passed"] += int(row.completed)
    for track in by_track_length.values():
        for bucket in track.values():
            bucket["rate"] = round(bucket["passed"] / max(1, bucket["trials"]), 4)

    failing = [r for r in rows if not r.completed]
    failure_kinds: Dict[str, int] = {}
    for row in failing:
        key = row.failure or "unknown"
        failure_kinds[key] = failure_kinds.get(key, 0) + 1

    by_length: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        bucket = by_length.setdefault(str(row.length), {"trials": 0, "passed": 0})
        bucket["trials"] += 1
        bucket["passed"] += int(row.completed)
    for length, bucket in by_length.items():
        bucket["rate"] = round(bucket["passed"] / max(1, bucket["trials"]), 4)
        target = FRAGMENT_TARGETS.get(int(length))
        bucket["target"] = target
        bucket["meets_target"] = None if target is None else bucket["rate"] >= target

    return {
        "fragments": len({r.fragment for r in rows}),
        "trials": len(rows),
        "overall_pass_rate": round(sum(r.completed for r in rows) / len(rows), 4),
        "by_length": dict(sorted(by_length.items(), key=lambda kv: int(kv[0]))),
        "by_track_and_length": {
            track: dict(sorted(lengths.items(), key=lambda kv: int(kv[0])))
            for track, lengths in sorted(by_track_length.items())
        },
        "failure_kinds": dict(sorted(failure_kinds.items())),
        "failing_fragments": [
            {"fragment": r.fragment, "track": r.track, "start_gate": r.start_gate,
             "gates": f"{r.gates_completed}/{r.gate_count}", "failure": r.failure}
            for r in failing
        ],
    }


@dataclass
class CircuitOutcome:
    track: str
    trial: int
    seed: int
    completed: bool
    gates_completed: int
    gate_count: int
    status: str
    failure: Optional[str]
    failure_gate: Optional[int]
    steps: int
    time_s: float
    path_length_m: float
    mean_action_jerk: float
    collisions: int
    out_of_bounds: int
    wrong_direction: int
    missed_gate: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_circuit(
    checkpoint: str | Path,
    track: str,
    *,
    trials: int = 10,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
    controller=None,
    root: Optional[str | Path] = None,
) -> List[CircuitOutcome]:
    """Run the complete circuit ``trials`` times with the learned controller."""
    from marine_race_arena.learning.gen2.evaluation import run_policy_episode

    controller = controller or _load_controller(checkpoint)
    path = tf.track_path(track, root)
    data = tf.load_track(track, root)
    gate_count = len(data["track"]["gate_sequence"])
    pool = gen2_seeds.GEN2_FULL_CIRCUIT_SEEDS
    offset = sorted(tf.OFFICIAL_TRACKS).index(track) * 100
    rows: List[CircuitOutcome] = []
    for trial in range(int(trials)):
        seed = pool[(offset + trial) % len(pool)]
        episode = run_policy_episode(
            controller, path, seed=int(seed), spec=None,
            adapter=adapter, allow_fallback=allow_fallback,
            max_steps=max(2000, gate_count * STEPS_PER_GATE),
        )
        failure = _classify_failure(episode)
        rows.append(CircuitOutcome(
            track=track, trial=trial, seed=int(seed),
            completed=episode.succeeded,
            gates_completed=episode.gates_completed, gate_count=gate_count,
            status=episode.status, failure=failure,
            failure_gate=None if episode.succeeded else episode.gates_completed + 1,
            steps=episode.steps, time_s=episode.completion_time_s,
            path_length_m=episode.path_length_m,
            mean_action_jerk=episode.mean_action_jerk,
            collisions=episode.collision_events,
            out_of_bounds=episode.out_of_bounds_events,
            wrong_direction=episode.wrong_direction_crossings,
            missed_gate=episode.missed_gate_attempts,
        ))
    return rows


def summarize_circuit(rows: Sequence[CircuitOutcome]) -> Dict[str, Any]:
    if not rows:
        return {"trials": 0}
    track = rows[0].track
    completed = [r for r in rows if r.completed]
    times = [r.time_s for r in completed]
    paths = [r.path_length_m for r in completed]
    failure_gates: Dict[str, int] = {}
    failure_kinds: Dict[str, int] = {}
    for row in rows:
        if row.completed:
            continue
        failure_gates[str(row.failure_gate)] = failure_gates.get(str(row.failure_gate), 0) + 1
        failure_kinds[row.failure or "unknown"] = failure_kinds.get(row.failure or "unknown", 0) + 1
    baseline = RULE_BASELINE_TIME_S.get(track)
    mean_time = round(float(np.mean(times)), 2) if times else None
    return {
        "track": track,
        "trials": len(rows),
        "completed": len(completed),
        "completion_rate": round(len(completed) / len(rows), 4),
        "gate_count": rows[0].gate_count,
        "mean_gates_completed": round(float(np.mean([r.gates_completed for r in rows])), 2),
        "best_gates_completed": max(r.gates_completed for r in rows),
        "mean_time_s": mean_time,
        "best_time_s": round(min(times), 2) if times else None,
        "rule_baseline_time_s": baseline,
        "time_delta_vs_rules_s": (
            round(mean_time - baseline, 2) if (mean_time and baseline) else None
        ),
        "mean_path_length_m": round(float(np.mean(paths)), 2) if paths else None,
        "mean_action_jerk": round(float(np.mean([r.mean_action_jerk for r in rows])), 6),
        "collision_episodes": sum(1 for r in rows if r.collisions > 0),
        "out_of_bounds_episodes": sum(1 for r in rows if r.out_of_bounds > 0),
        "wrong_direction_episodes": sum(1 for r in rows if r.wrong_direction > 0),
        "failure_gates": dict(sorted(failure_gates.items(), key=lambda kv: int(kv[0]))),
        "failure_kinds": dict(sorted(failure_kinds.items())),
    }


def failure_map(rows: Sequence[CircuitOutcome]) -> Dict[str, Any]:
    """Where each circuit actually breaks, ranked by frequency."""
    per_track: Dict[str, Dict[int, int]] = {}
    kinds: Dict[str, Dict[str, int]] = {}
    for row in rows:
        if row.completed or row.failure_gate is None:
            continue
        per_track.setdefault(row.track, {})
        per_track[row.track][row.failure_gate] = per_track[row.track].get(row.failure_gate, 0) + 1
        kinds.setdefault(row.track, {})
        key = row.failure or "unknown"
        kinds[row.track][key] = kinds[row.track].get(key, 0) + 1
    ranked = {
        track: sorted(
            ({"gate": gate, "failures": count} for gate, count in gates.items()),
            key=lambda item: (-item["failures"], item["gate"]),
        )
        for track, gates in per_track.items()
    }
    return {
        "weak_gates": ranked,
        "failure_kinds": kinds,
        "worst": {
            track: entries[0]["gate"] if entries else None
            for track, entries in ranked.items()
        },
    }


def targeted_fragments(
    failures: Mapping[str, Any],
    *,
    window: int = 2,
    lengths: Sequence[int] = (3, 4, 5),
    root: Optional[str | Path] = None,
) -> List[tf.TrackFragment]:
    """Fragments that straddle each observed weak gate.

    A circuit that dies entering gate *g* needs practice on the windows that
    *contain* the g-1 -> g transition, so the selection keeps any fragment
    whose span covers gate g with at least one gate of run-up.
    """
    out: List[tf.TrackFragment] = []
    for track, entries in (failures.get("weak_gates") or {}).items():
        weak = {int(entry["gate"]) for entry in entries}
        if not weak:
            continue
        for fragment in tf.enumerate_fragments(track, lengths, root=root):
            span = range(fragment.start_index + 1, fragment.end_index + 2)
            for gate in weak:
                if gate in span and gate - fragment.start_index >= min(2, window):
                    out.append(fragment)
                    break
    # Stable, de-duplicated.
    seen: set = set()
    unique: List[tf.TrackFragment] = []
    for fragment in out:
        if fragment.name in seen:
            continue
        seen.add(fragment.name)
        unique.append(fragment)
    return unique


def evaluate_all_circuits(
    checkpoint: str | Path,
    *,
    out_dir: str | Path,
    trials: int = 10,
    tracks: Sequence[str] = tuple(tf.OFFICIAL_TRACKS),
    adapter: str = "holoocean",
    allow_fallback: bool = False,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    controller = _load_controller(checkpoint)
    started = time.perf_counter()
    all_rows: List[CircuitOutcome] = []
    summaries: Dict[str, Any] = {}
    for track in tracks:
        rows = evaluate_circuit(
            checkpoint, track, trials=trials, adapter=adapter,
            allow_fallback=allow_fallback, controller=controller,
        )
        all_rows.extend(rows)
        summaries[track] = summarize_circuit(rows)
        (out_dir / "circuit_report.json").write_text(json.dumps({
            "checkpoint": str(checkpoint),
            "summaries": summaries,
            "rows": [r.as_dict() for r in all_rows],
        }, indent=2), encoding="utf-8")

    report = {
        "protocol": "track_specific_gen2_v1",
        "checkpoint": str(checkpoint),
        "trials_per_track": int(trials),
        "summaries": summaries,
        "failure_map": failure_map(all_rows),
        "tracks_completed_at_least_once": [
            track for track, s in summaries.items() if s.get("completed", 0) > 0
        ],
        "all_three_completed_at_least_once": all(
            summaries.get(t, {}).get("completed", 0) > 0 for t in tf.OFFICIAL_TRACKS
        ),
        "wall_time_s": round(time.perf_counter() - started, 1),
    }
    (out_dir / "circuit_report.json").write_text(json.dumps({
        **report, "rows": [r.as_dict() for r in all_rows],
    }, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate on track fragments and full circuits")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", default="both", choices=("fragments", "circuits", "both"))
    parser.add_argument("--lengths", default="2,3,4,5")
    parser.add_argument("--tracks", default=",".join(tf.OFFICIAL_TRACKS))
    parser.add_argument("--fragment-trials", type=int, default=1)
    parser.add_argument("--circuit-trials", type=int, default=3)
    parser.add_argument("--adapter", default="holoocean", choices=("holoocean", "fallback"))
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--collect-shards", action="store_true",
                        help="merge fragment shard files already on disk and exit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    tracks = [v.strip() for v in args.tracks.split(",") if v.strip()]
    if args.collect_shards:
        report = collect_fragment_shards(Path(args.out) / "fragments")
        print(json.dumps({
            "overall_pass_rate": report["overall_pass_rate"],
            "by_length": report["by_length"],
            "by_track_and_length": report["by_track_and_length"],
            "failure_kinds": report["failure_kinds"],
            "failing_fragments": report["failing_fragments"][:25],
        }, indent=2), flush=True)
        return 0
    if args.mode in ("fragments", "both"):
        report = evaluate_fragments(
            args.checkpoint, out_dir=Path(args.out) / "fragments",
            lengths=[int(v) for v in args.lengths.split(",") if v.strip()],
            tracks=tracks, trials=args.fragment_trials,
            adapter=args.adapter, allow_fallback=args.allow_fallback,
            shard=args.shard, shards=args.shards,
        )
        print(json.dumps({
            "overall_pass_rate": report["overall_pass_rate"],
            "by_length": report["by_length"],
            "by_track_and_length": report["by_track_and_length"],
            "failure_kinds": report["failure_kinds"],
        }, indent=2), flush=True)
    if args.mode in ("circuits", "both"):
        report = evaluate_all_circuits(
            args.checkpoint, out_dir=Path(args.out) / "circuits",
            trials=args.circuit_trials, tracks=tracks,
            adapter=args.adapter, allow_fallback=args.allow_fallback,
        )
        print(json.dumps({
            "summaries": report["summaries"],
            "failure_map": report["failure_map"],
            "all_three_completed_at_least_once": report["all_three_completed_at_least_once"],
        }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
