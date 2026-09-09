"""Collect the Gen-2 expert demonstration corpus, resumably and in parallel.

Usage::

    python -m marine_race_arena.learning.gen2.collect_expert \\
        --out results/rl/gen2/expert_corpus_v1 \\
        --episodes 400 --workers 4

Design notes:

* **Seed-addressed, therefore resumable.**  Episode ``i`` is always seed
  ``GEN2_EXPERT_DEMO_SEEDS[i]`` and always the same course, so a run that dies
  half way is restarted by re-issuing the same command; existing shards are
  skipped.
* **Round-robin sharding.**  Worker ``w`` takes every ``W``-th seed, so each
  worker sees the same length mixture and one slow worker cannot end up owning
  all the 22-gate courses.
* **Engine safety is inherited, not reimplemented.**  Episodes go through
  ``RaceEpisode`` -> the HoloOcean adapter, which already serialises engine
  birth behind ``engine_start_slot`` and proves health with
  ``qualify_engine_start``.  This module only decides *how many* workers to
  run, using the live :func:`capacity_snapshot`, and reaps the engines its own
  children owned.
* **TRAIN-only.**  Every seed is checked with
  :func:`~marine_race_arena.learning.gen2.seeds.assert_expert_labelling_seed`
  before an episode starts, so this collector physically cannot label a
  validation, test or sealed-holdout state.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from marine_race_arena.learning.gen2 import course_family as cf
from marine_race_arena.learning.gen2 import seeds as gen2_seeds
from marine_race_arena.learning.gen2.dataset import (
    iter_shards,
    load_corpus,
    save_episode,
    shard_path,
    write_manifest,
)
from marine_race_arena.learning.gen2.expert_rollout import run_gen2_episode

#: Episode-length mixture for the first behaviour-cloning corpus.
#: Short courses dominate the *episode* count so the corpus buys many distinct
#: geometries per simulator-hour, while the long courses still contribute the
#: majority of the *samples* (a 12-gate episode is ~10x a 2-gate one).
BC_STAGE1_GATE_WEIGHTS: Dict[int, float] = {
    2: 0.30, 3: 0.26, 5: 0.22, 8: 0.13, 12: 0.06, 17: 0.02, 22: 0.01,
}

#: Uniform coverage across every defined length, for the later corpora.
BC_BALANCED_GATE_WEIGHTS: Dict[int, float] = {
    2: 0.16, 3: 0.16, 5: 0.16, 8: 0.16, 12: 0.14, 17: 0.12, 22: 0.10,
}

GATE_WEIGHT_PRESETS: Dict[str, Dict[int, float]] = {
    "stage1": BC_STAGE1_GATE_WEIGHTS,
    "balanced": BC_BALANCED_GATE_WEIGHTS,
    "demo": cf.GEN2_DEMO_GATE_WEIGHTS,
}

#: Generous per-gate step ceiling: the expert needs ~350 steps per gate, so
#: this only bites on an episode that has already gone wrong.
STEPS_PER_GATE_BUDGET = 900
MIN_EPISODE_STEPS = 600


def episode_seeds(count: int, role: str = "gen2_expert_demonstrations") -> List[int]:
    pool = gen2_seeds.GEN2_ROLE_SEEDS[role]
    if count > len(pool):
        raise ValueError(f"role {role!r} only has {len(pool)} seeds, asked for {count}")
    return list(pool[:count])


def max_steps_for(gate_count: int) -> int:
    return max(MIN_EPISODE_STEPS, int(gate_count) * STEPS_PER_GATE_BUDGET)


@dataclass
class WorkerArgs:
    worker_id: int
    workers: int
    seeds: List[int]
    out_dir: str
    adapter: str
    allow_fallback: bool
    gate_weights: Dict[int, float]
    dt: float
    #: When set, gate counts cycle through this list instead of being sampled.
    #: Used by validation batches that must cover every length exactly.
    gate_count_cycle: Optional[List[int]] = None


def _worker_track_dir(out_dir: str | Path, worker_id: int) -> Path:
    path = Path(out_dir) / "workers" / f"worker_{worker_id:02d}" / "tracks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_worker(args: WorkerArgs) -> Dict[str, Any]:
    """Run one worker's slice of the seed list.  Executes in its own process."""
    out_dir = Path(args.out_dir)
    track_dir = _worker_track_dir(out_dir, args.worker_id)
    progress_path = out_dir / "workers" / f"worker_{args.worker_id:02d}" / "progress.json"
    done = 0
    skipped = 0
    failures: List[Dict[str, Any]] = []
    started = time.perf_counter()

    for index, seed in enumerate(args.seeds):
        if index % args.workers != args.worker_id:
            continue
        target = shard_path(out_dir, seed)
        if target.exists():
            skipped += 1
            continue
        # Hard guard: this process may only ever label a TRAIN state.
        gen2_seeds.assert_expert_labelling_seed(seed, context="expert corpus collection")
        forced = (
            args.gate_count_cycle[index % len(args.gate_count_cycle)]
            if args.gate_count_cycle else None
        )
        spec = cf.sample_course(
            seed, gate_count=forced, gate_count_weights=args.gate_weights
        )
        try:
            track = cf.materialize_course(spec, track_dir / f"course_{seed:06d}.json")
            record = run_gen2_episode(
                track,
                seed=seed,
                spec=spec,
                mode="expert",
                adapter=args.adapter,
                allow_fallback=args.allow_fallback,
                dt=args.dt,
                max_steps=max_steps_for(spec.gate_count),
            )
            save_episode(record, out_dir)
            done += 1
        except Exception as exc:  # keep collecting; a bad episode is data, not a crash
            failures.append({
                "seed": int(seed),
                "gate_count": int(spec.gate_count),
                "error": f"{type(exc).__name__}: {exc}",
            })
        progress_path.write_text(
            json.dumps({
                "worker_id": args.worker_id,
                "pid": os.getpid(),
                "collected": done,
                "skipped": skipped,
                "failures": len(failures),
                "elapsed_s": round(time.perf_counter() - started, 1),
                "last_seed": int(seed),
            }, indent=2),
            encoding="utf-8",
        )
    return {
        "worker_id": args.worker_id,
        "pid": os.getpid(),
        "collected": done,
        "skipped": skipped,
        "failures": failures,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }


def choose_workers(requested: int, *, adapter: str) -> int:
    """Clamp the worker count to what this machine can actually sustain now.

    The box is shared with another HoloOcean project, so capacity is measured
    at launch rather than assumed from a previous session's benchmark.
    """
    if adapter != "holoocean":
        return max(1, int(requested))
    try:
        from marine_race_arena.learning.holoocean_capacity import capacity_snapshot

        snapshot = capacity_snapshot()
        available = int(snapshot.get("available", 0))
    except Exception:
        return max(1, int(requested))
    if available <= 0:
        raise RuntimeError(
            f"no HoloOcean engine slots available: {snapshot.get('active')} active, "
            f"{snapshot.get('reserved')} reserved, ceiling {snapshot.get('maximum')}"
        )
    return max(1, min(int(requested), available))


def collect(
    out_dir: str | Path,
    *,
    episodes: int,
    workers: int = 1,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
    gate_weights: Optional[Mapping[int, float]] = None,
    role: str = "gen2_expert_demonstrations",
    dt: float = 0.1,
    gate_count_cycle: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Collect ``episodes`` expert demonstrations into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = dict(gate_weights or BC_STAGE1_GATE_WEIGHTS)
    seeds = episode_seeds(episodes, role)
    effective_workers = choose_workers(workers, adapter=adapter)
    started = time.perf_counter()

    run_config = {
        "schema_version": "gen2_expert_collection_v1",
        "role": role,
        "episodes_requested": int(episodes),
        "seed_first": seeds[0],
        "seed_last": seeds[-1],
        "workers_requested": int(workers),
        "workers_effective": int(effective_workers),
        "adapter": adapter,
        "allow_fallback": bool(allow_fallback),
        "dt": float(dt),
        "gate_weights": {str(k): v for k, v in sorted(weights.items())},
        "gate_count_cycle": list(gate_count_cycle) if gate_count_cycle else None,
        "course_family_version": cf.GEN2_COURSE_FAMILY_VERSION,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "collection_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    jobs = [
        WorkerArgs(
            worker_id=index,
            workers=effective_workers,
            seeds=seeds,
            out_dir=str(out_dir),
            adapter=adapter,
            allow_fallback=allow_fallback,
            gate_weights=weights,
            dt=dt,
            gate_count_cycle=list(gate_count_cycle) if gate_count_cycle else None,
        )
        for index in range(effective_workers)
    ]

    if effective_workers == 1:
        results = [_run_worker(jobs[0])]
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=effective_workers) as pool:
            results = pool.map(_run_worker, jobs)

    episodes_loaded = load_corpus(out_dir)
    manifest_path = write_manifest(
        out_dir,
        episodes=episodes_loaded,
        extra={
            "collection": run_config,
            "workers": results,
            "wall_time_s": round(time.perf_counter() - started, 1),
        },
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {
        "manifest_path": str(manifest_path),
        "corpus_sha256": manifest["corpus_sha256"],
        "statistics": manifest["statistics"],
        "workers": results,
        "wall_time_s": round(time.perf_counter() - started, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect the Gen-2 expert corpus")
    parser.add_argument("--out", required=True, help="corpus directory")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--adapter", default="holoocean", choices=("holoocean", "fallback"))
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--weights", default="stage1", choices=sorted(GATE_WEIGHT_PRESETS))
    parser.add_argument(
        "--role", default="gen2_expert_demonstrations",
        choices=sorted(gen2_seeds.EXPERT_LABELLING_ALLOWED_ROLES),
        help="seed role; only expert-labelling roles are accepted",
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--gate-count-cycle", default=None,
        help="comma-separated gate counts to cycle through instead of sampling "
             "(e.g. 2,3,5,8,12,17,22 for a coverage batch)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = collect(
        args.out,
        episodes=args.episodes,
        workers=args.workers,
        adapter=args.adapter,
        allow_fallback=args.allow_fallback,
        gate_weights=GATE_WEIGHT_PRESETS[args.weights],
        role=args.role,
        dt=args.dt,
        gate_count_cycle=(
            [int(v) for v in args.gate_count_cycle.split(",")]
            if args.gate_count_cycle else None
        ),
    )
    print(json.dumps(summary["statistics"], indent=2), flush=True)
    print(f"[gen2] corpus_sha256={summary['corpus_sha256']}", flush=True)
    print(f"[gen2] manifest={summary['manifest_path']}", flush=True)
    print(f"[gen2] wall_time_s={summary['wall_time_s']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
