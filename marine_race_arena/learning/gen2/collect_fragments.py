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
) -> List[FragmentJob]:
    jobs: List[FragmentJob] = []
    used: set = set()
    for fragment in tf.all_fragments(lengths, tracks=tracks, root=root):
        for trial in range(int(trials)):
            seed = tf.fragment_seed(fragment, trial)
            # Derived seeds can collide inside the band; walk to the next free
            # one so every episode keeps its own shard.
            while seed in used:
                seed = gen2_seeds.GEN2_TRACK_FRAGMENT_SEEDS[
                    (gen2_seeds.GEN2_TRACK_FRAGMENT_SEEDS.index(seed) + 1)
                    % len(gen2_seeds.GEN2_TRACK_FRAGMENT_SEEDS)
                ]
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


def _run_worker(args: WorkerArgs) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    track_dir = out_dir / "workers" / f"worker_{args.worker_id:02d}" / "tracks"
    track_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "workers" / f"worker_{args.worker_id:02d}" / "progress.json"
    collected = skipped = 0
    failures: List[Dict[str, Any]] = []
    started = time.perf_counter()

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
                path, seed=job.seed, spec=None, mode="expert",
                adapter=args.adapter, allow_fallback=args.allow_fallback, dt=args.dt,
                max_steps=max(MIN_STEPS, job.fragment.length * STEPS_PER_GATE),
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
        except Exception as exc:
            failures.append({
                "fragment": job.fragment.name, "seed": job.seed,
                "error": f"{type(exc).__name__}: {exc}",
            })
        progress.write_text(json.dumps({
            "worker_id": args.worker_id, "pid": os.getpid(),
            "collected": collected, "skipped": skipped, "failures": len(failures),
            "elapsed_s": round(time.perf_counter() - started, 1),
        }, indent=2), encoding="utf-8")

    return {
        "worker_id": args.worker_id, "collected": collected,
        "skipped": skipped, "failures": failures,
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
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = list(jobs) if jobs is not None else build_jobs(
        lengths, trials=trials, tracks=tracks
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
        "note": (
            "Fragments are exact contiguous windows of the three evaluation "
            "circuits. This corpus makes no unseen-track claim."
        ),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2), encoding="utf-8")

    worker_args = [
        WorkerArgs(index, effective, plan, str(out_dir), adapter, allow_fallback, dt)
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
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = collect(
        args.out,
        lengths=[int(v) for v in args.lengths.split(",") if v.strip()],
        trials=args.trials,
        tracks=[v.strip() for v in args.tracks.split(",") if v.strip()],
        workers=args.workers, adapter=args.adapter,
        allow_fallback=args.allow_fallback, dt=args.dt,
    )
    print(json.dumps(summary["track_composition"], indent=2), flush=True)
    print(f"[gen2] corpus_sha256={summary['corpus_sha256']}", flush=True)
    print(f"[gen2] wall_time_s={summary['wall_time_s']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
