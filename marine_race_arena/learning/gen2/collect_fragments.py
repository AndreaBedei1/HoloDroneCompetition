"""Collect expert demonstrations on exact fragments of the three circuits.

    python -m marine_race_arena.learning.gen2.collect_fragments \\
        --out results/rl/gen2/track_corpus --lengths 2,3,4,5 --trials 1 --workers 5

Every episode is the frozen rule controller driving one unmodified window of a
real circuit, entered from the pose the vehicle would actually hold there.  The
shard metadata records the track, the fragment bounds and the gate ids so the
corpus can be balanced and so failures can be traced back to an exact
transition -- none of it reaches the network, which still sees only the 35
onboard features.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from marine_race_arena.learning.gen2 import seeds as gen2_seeds
from marine_race_arena.learning.gen2 import track_fragments as tf
from marine_race_arena.learning.gen2.collect_expert import choose_workers
from marine_race_arena.learning.gen2.dataset import (
    load_corpus,
    save_episode,
    shard_path,
    write_manifest,
)
from marine_race_arena.learning.gen2.expert_rollout import run_gen2_episode

#: Generous per-gate step ceiling for a fragment episode.
STEPS_PER_GATE = 900
MIN_STEPS = 600


@dataclass
class FragmentJob:
    fragment: tf.TrackFragment
    trial: int
    seed: int

    @property
    def tag(self) -> str:
        return f"{self.fragment.name}_t{self.trial}"


def build_jobs(
    lengths: Sequence[int] = (2, 3, 4, 5),
    *,
    trials: int = 1,
    tracks: Sequence[str] = tuple(tf.OFFICIAL_TRACKS),
    root: Optional[str | Path] = None,
    band: Optional[Sequence[int]] = None,
) -> List[FragmentJob]:
    """One job per (fragment, trial), seeded inside ``band``.

    DAgger rollouts take the DAgger band rather than the demonstration band, so
    a round can never be mistaken for, or collide with, the corpus it
    aggregates onto.
    """
    pool = list(band) if band is not None else gen2_seeds.GEN2_TRACK_FRAGMENT_SEEDS
    jobs: List[FragmentJob] = []
    used: set = set()
    for index, fragment in enumerate(tf.all_fragments(lengths, tracks=tracks, root=root)):
        for trial in range(int(trials)):
            seed = pool[(index * max(1, int(trials)) + trial) % len(pool)]
            # Walk to the next free slot so every episode keeps its own shard.
            while seed in used:
                seed = pool[(pool.index(seed) + 1) % len(pool)]
            used.add(seed)
            jobs.append(FragmentJob(fragment=fragment, trial=trial, seed=seed))
    return jobs


@dataclass
class WorkerArgs:
    worker_id: int
    workers: int
    jobs: List[FragmentJob]
    out_dir: str
    adapter: str
    allow_fallback: bool
    dt: float
    #: When set, the LEARNER drives and the expert only labels (targeted DAgger).
    dagger_from: Optional[str] = None


def _load_learner(checkpoint: str):
    from sb3_contrib import RecurrentPPO

    from marine_race_arena.learning.gen2.recurrent_policy import Gen2RecurrentController

    return Gen2RecurrentController(
        RecurrentPPO.load(checkpoint, device="cpu"), deterministic=True
    )


def _run_worker(args: WorkerArgs) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    track_dir = out_dir / "workers" / f"worker_{args.worker_id:02d}" / "tracks"
    track_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "workers" / f"worker_{args.worker_id:02d}" / "progress.json"
    collected = skipped = 0
    failures: List[Dict[str, Any]] = []
    learner_completed = 0
    started = time.perf_counter()
    learner = _load_learner(args.dagger_from) if args.dagger_from else None
    mode = "dagger" if learner is not None else "expert"

    for index, job in enumerate(args.jobs):
        if index % args.workers != args.worker_id:
            continue
        target = shard_path(out_dir, job.seed)
        if target.exists():
            skipped += 1
            continue
        gen2_seeds.assert_expert_labelling_seed(
            job.seed, context="track fragment collection"
        )
        try:
            path = tf.materialize_fragment(job.fragment, track_dir / f"{job.tag}.json")
            record = run_gen2_episode(
                path, seed=job.seed, spec=None, mode=mode, learner=learner,
                adapter=args.adapter, allow_fallback=args.allow_fallback, dt=args.dt,
                max_steps=max(MIN_STEPS, job.fragment.length * STEPS_PER_GATE),
                safety_takeover=None,   # pure learner rollout; no blending
                initial_body_velocity=tf.inbound_body_velocity(job.fragment),
            )
            # Fragment identity is diagnostics only; it never enters the network.
            record.course = {
                "track": job.fragment.track,
                "fragment": job.fragment.name,
                "start_index": job.fragment.start_index,
                "end_index": job.fragment.end_index,
                "gate_ids": list(job.fragment.gate_ids),
                "fragment_length": job.fragment.length,
                "is_prefix": job.fragment.is_prefix,
                "trial": job.trial,
            }
            save_episode(record, out_dir)
            collected += 1
            learner_completed += int(record.completed)
        except Exception as exc:
            failures.append({
                "fragment": job.fragment.name, "seed": job.seed,
                "error": f"{type(exc).__name__}: {exc}",
            })
        progress.write_text(json.dumps({
            "worker_id": args.worker_id, "pid": os.getpid(),
            "collected": collected, "skipped": skipped, "failures": len(failures),
            "mode": mode, "learner_completed": learner_completed,
            "elapsed_s": round(time.perf_counter() - started, 1),
        }, indent=2), encoding="utf-8")

    return {
        "worker_id": args.worker_id, "collected": collected,
        "skipped": skipped, "failures": failures, "mode": mode,
        "learner_completed": learner_completed,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }


def track_composition(out_dir: str | Path) -> Dict[str, Any]:
    """Transitions and completion per circuit -- the balance check."""
    episodes = load_corpus(out_dir)
    per_track: Dict[str, Dict[str, Any]] = {}
    per_length: Dict[str, int] = {}
    for episode in episodes:
        track = str(episode.meta.get("course_track", "unknown"))
        bucket = per_track.setdefault(track, {
            "episodes": 0, "transitions": 0, "completed": 0,
            "gates_crossed": 0, "gates_possible": 0,
        })
        bucket["episodes"] += 1
        bucket["transitions"] += len(episode)
        bucket["completed"] += int(episode.completed)
        bucket["gates_crossed"] += int(episode.meta.get("gates_completed", 0))
        bucket["gates_possible"] += episode.gate_count
        key = str(episode.meta.get("course_fragment_length", episode.gate_count))
        per_length[key] = per_length.get(key, 0) + 1
    total = sum(b["transitions"] for b in per_track.values()) or 1
    for bucket in per_track.values():
        bucket["completion_rate"] = round(bucket["completed"] / max(1, bucket["episodes"]), 4)
        bucket["transition_share"] = round(bucket["transitions"] / total, 4)
    return {
        "episodes": len(episodes),
        "transitions": total,
        "by_track": dict(sorted(per_track.items())),
        "by_fragment_length": dict(sorted(per_length.items(), key=lambda kv: int(kv[0]))),
    }


def collect(
    out_dir: str | Path,
    *,
    lengths: Sequence[int] = (2, 3, 4, 5),
    trials: int = 1,
    tracks: Sequence[str] = tuple(tf.OFFICIAL_TRACKS),
    workers: int = 1,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
    dt: float = 0.1,
    jobs: Optional[Sequence[FragmentJob]] = None,
    dagger_from: Optional[str] = None,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = list(jobs) if jobs is not None else build_jobs(
        lengths, trials=trials, tracks=tracks,
        band=(gen2_seeds.GEN2_TRACK_DAGGER_SEEDS if dagger_from else None),
    )
    effective = choose_workers(workers, adapter=adapter)
    started = time.perf_counter()

    (out_dir / "collection_config.json").write_text(json.dumps({
        "schema_version": "gen2_track_fragment_collection_v1",
        "protocol": "track_specific_gen2_v1",
        "lengths": list(lengths), "trials": int(trials),
        "tracks": list(tracks), "episodes": len(plan),
        "workers_requested": int(workers), "workers_effective": effective,
        "adapter": adapter,
        "mode": "dagger" if dagger_from else "expert",
        "dagger_from": dagger_from,
        "note": (
            "Fragments are exact contiguous windows of the three evaluation "
            "circuits. This corpus makes no unseen-track claim."
        ),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2), encoding="utf-8")

    worker_args = [
        WorkerArgs(index, effective, plan, str(out_dir), adapter, allow_fallback, dt,
                   dagger_from)
        for index in range(effective)
    ]
    if effective == 1:
        results = [_run_worker(worker_args[0])]
    else:
        with mp.get_context("spawn").Pool(processes=effective) as pool:
            results = pool.map(_run_worker, worker_args)

    episodes = load_corpus(out_dir)
    manifest_path = write_manifest(out_dir, episodes=episodes, extra={
        "protocol": "track_specific_gen2_v1",
        "track_composition": track_composition(out_dir),
        "workers": results,
    })
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {
        "manifest_path": str(manifest_path),
        "corpus_sha256": manifest["corpus_sha256"],
        "statistics": manifest["statistics"],
        "track_composition": manifest["track_composition"],
        "workers": results,
        "wall_time_s": round(time.perf_counter() - started, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect expert data on exact track fragments")
    parser.add_argument("--out", required=True)
    parser.add_argument("--lengths", default="2,3,4,5")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--tracks", default=",".join(tf.OFFICIAL_TRACKS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--adapter", default="holoocean", choices=("holoocean", "fallback"))
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--dagger-from", default=None,
        help="learner checkpoint; when set the learner drives and the expert "
             "only labels the states it visits (targeted DAgger, no blending)",
    )
    parser.add_argument(
        "--weak-gates", default=None,
        help="JSON failure-map path; restricts fragments to windows containing "
             "the observed weak gates with run-up",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    lengths = [int(v) for v in args.lengths.split(",") if v.strip()]
    tracks = [v.strip() for v in args.tracks.split(",") if v.strip()]
    jobs = None
    if args.weak_gates:
        from marine_race_arena.learning.gen2.track_eval import targeted_fragments

        failures = json.loads(Path(args.weak_gates).read_text(encoding="utf-8"))
        picked = targeted_fragments(failures.get("failure_map", failures), lengths=lengths)
        if not picked:
            raise SystemExit("the failure map selected no fragments")
        jobs = []
        used: set = set()
        for fragment in picked:
            for trial in range(int(args.trials)):
                seed = tf.fragment_seed(fragment, 1000 + trial)
                while seed in used:
                    band = gen2_seeds.GEN2_TRACK_DAGGER_SEEDS
                    seed = band[(band.index(seed) + 1) % len(band)] if seed in band else band[0]
                used.add(seed)
                jobs.append(FragmentJob(fragment=fragment, trial=trial, seed=seed))
        print(f"[gen2] targeting {len(picked)} fragments around the observed weak gates",
              flush=True)
    summary = collect(
        args.out, lengths=lengths, trials=args.trials, tracks=tracks,
        workers=args.workers, adapter=args.adapter,
        allow_fallback=args.allow_fallback, dt=args.dt,
        jobs=jobs, dagger_from=args.dagger_from,
    )
    print(json.dumps(summary["track_composition"], indent=2), flush=True)
    print(f"[gen2] corpus_sha256={summary['corpus_sha256']}", flush=True)
    print(f"[gen2] wall_time_s={summary['wall_time_s']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
