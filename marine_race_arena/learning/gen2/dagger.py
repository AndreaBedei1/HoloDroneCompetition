"""DAgger: aggregate expert labels on states the *learner* actually visits.

This is the part of Gen-2 that attacks compounding error directly.  Behaviour
cloning only ever sees the expert's own state distribution, so the first time
the learner drifts it is off-distribution and has never been told what to do
there.  DAgger closes that loop: the learner drives, and the frozen expert
labels every state it reaches, including the ones the expert would never have
produced.

The semantics that matter, and which
``tests/learning/gen2/test_gen2_dagger.py`` pins:

* **The learner drives.**  The applied command is exactly the learner's output.
  The expert is stepped in shadow on the same raw observation and its command
  is stored as the label only.
* **No blending.**  There is no convex combination, no beta schedule, no
  "expert for the first k steps".  Mixed control would make the recorded state
  distribution neither the learner's nor the expert's.
* **The learner owns its recurrent state.**  Hidden state is carried across the
  episode and zeroed only at the episode boundary, exactly as at inference.
* **Safety takeovers are logged and excludable.**  A takeover step is marked in
  ``applied_by_expert``; any claim about autonomous learner completion loads
  the corpus with ``exclude_expert_takeover=True``.  The default is no takeover
  at all.
* **TRAIN seeds only.**  Every rollout seed passes
  ``assert_expert_labelling_seed`` before an episode starts.

Rounds advance on measured validation competence, not on a loop counter, and
every round retains shorter sequences so the policy cannot forget what it
already composes.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.gen2 import course_family as cf
from marine_race_arena.learning.gen2 import seeds as gen2_seeds
from marine_race_arena.learning.gen2.collect_expert import choose_workers, max_steps_for
from marine_race_arena.learning.gen2.dataset import load_corpus, save_episode, shard_path, write_manifest
from marine_race_arena.learning.gen2.expert_rollout import run_gen2_episode

#: Sequence-length curriculum.  Each round keeps a tail of shorter courses so
#: the policy is never trained only on the hardest thing it currently sees.
DAGGER_CURRICULUM: Tuple[Dict[str, Any], ...] = (
    {"round": 1, "lengths": (3, 5), "retention": (2, 3)},
    {"round": 2, "lengths": (5, 8), "retention": (2, 3, 5)},
    {"round": 3, "lengths": (8, 12), "retention": (3, 5, 8)},
    {"round": 4, "lengths": (12, 17), "retention": (3, 5, 8, 12)},
    {"round": 5, "lengths": (17, 22), "retention": (3, 5, 8, 12, 17)},
)

#: Fraction of a round's episodes drawn from the retention (shorter) lengths.
RETENTION_FRACTION = 0.35


def round_gate_counts(round_index: int, episodes: int) -> List[int]:
    """Interleave the round's target lengths with its retention lengths."""
    plan = next((r for r in DAGGER_CURRICULUM if r["round"] == int(round_index)), None)
    if plan is None:
        raise ValueError(f"no DAgger curriculum entry for round {round_index}")
    lengths = list(plan["lengths"])
    retention = list(plan["retention"])
    counts: List[int] = []
    retain_every = max(2, int(round(1.0 / RETENTION_FRACTION)))
    for index in range(int(episodes)):
        if retention and index % retain_every == 0:
            counts.append(retention[(index // retain_every) % len(retention)])
        else:
            counts.append(lengths[index % len(lengths)])
    return counts


@dataclass
class DaggerWorkerArgs:
    worker_id: int
    workers: int
    seeds: List[int]
    gate_counts: List[int]
    out_dir: str
    checkpoint_path: str
    adapter: str
    allow_fallback: bool
    dt: float
    deterministic: bool


def _load_controller(checkpoint_path: str, *, deterministic: bool = True):
    """Load the learner in this process.  Import-local so spawn stays cheap."""
    from sb3_contrib import RecurrentPPO

    from marine_race_arena.learning.gen2.recurrent_policy import Gen2RecurrentController

    model = RecurrentPPO.load(checkpoint_path, device="cpu")
    return Gen2RecurrentController(model, deterministic=deterministic)


def _run_dagger_worker(args: DaggerWorkerArgs) -> Dict[str, Any]:
    out_dir = Path(args.out_dir)
    track_dir = out_dir / "workers" / f"worker_{args.worker_id:02d}" / "tracks"
    track_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "workers" / f"worker_{args.worker_id:02d}" / "progress.json"
    controller = _load_controller(args.checkpoint_path, deterministic=args.deterministic)

    collected = 0
    skipped = 0
    failures: List[Dict[str, Any]] = []
    learner_completed = 0
    learner_gates = 0
    course_gates = 0
    started = time.perf_counter()

    for index, seed in enumerate(args.seeds):
        if index % args.workers != args.worker_id:
            continue
        target = shard_path(out_dir, seed)
        if target.exists():
            skipped += 1
            continue
        gen2_seeds.assert_expert_labelling_seed(seed, context="DAgger aggregation")
        gate_count = int(args.gate_counts[index % len(args.gate_counts)])
        spec = cf.sample_course(seed, gate_count=gate_count)
        try:
            track = cf.materialize_course(spec, track_dir / f"course_{seed:06d}.json")
            record = run_gen2_episode(
                track,
                seed=seed,
                spec=spec,
                mode="dagger",              # learner drives, expert labels
                learner=controller,
                adapter=args.adapter,
                allow_fallback=args.allow_fallback,
                dt=args.dt,
                max_steps=max_steps_for(spec.gate_count),
                safety_takeover=None,       # pure learner rollout
            )
            save_episode(record, out_dir)
            collected += 1
            learner_completed += int(record.completed)
            learner_gates += int(record.gates_completed)
            course_gates += int(record.gate_count)
        except Exception as exc:
            failures.append({
                "seed": int(seed), "gate_count": gate_count,
                "error": f"{type(exc).__name__}: {exc}",
            })
        progress_path.write_text(json.dumps({
            "worker_id": args.worker_id, "pid": os.getpid(),
            "collected": collected, "skipped": skipped, "failures": len(failures),
            "learner_completed": learner_completed,
            "elapsed_s": round(time.perf_counter() - started, 1),
        }, indent=2), encoding="utf-8")

    return {
        "worker_id": args.worker_id,
        "collected": collected,
        "skipped": skipped,
        "failures": failures,
        "learner_completed": learner_completed,
        "learner_gates": learner_gates,
        "course_gates": course_gates,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }


@dataclass
class DaggerRoundResult:
    round_index: int
    out_dir: str
    episodes_requested: int
    episodes_collected: int
    seeds: Tuple[int, int]
    gate_counts: Dict[str, int]
    learner_completion_rate: Optional[float]
    learner_gate_rate: Optional[float]
    aggregated_transitions: int
    corpus_sha256: str
    wall_time_s: float
    workers: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_dagger_round(
    checkpoint_path: str | Path,
    *,
    round_index: int,
    episodes: int,
    out_dir: str | Path,
    workers: int = 1,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
    dt: float = 0.1,
    deterministic: bool = True,
    gate_counts: Optional[Sequence[int]] = None,
) -> DaggerRoundResult:
    """Roll out the learner on TRAIN courses and store expert labels.

    ``deterministic`` keeps the learner on its mean action.  Sampling would add
    exploration noise that the expert then labels away, which teaches the
    policy to correct its own noise rather than its own mistakes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = gen2_seeds.dagger_round_seeds(int(round_index), count=max(1, int(episodes)))
    seeds = list(pool[: int(episodes)])
    counts = list(gate_counts) if gate_counts else round_gate_counts(round_index, len(seeds))
    effective_workers = choose_workers(workers, adapter=adapter)
    started = time.perf_counter()

    (out_dir / "round_config.json").write_text(json.dumps({
        "schema_version": "gen2_dagger_round_v1",
        "round": int(round_index),
        "episodes": len(seeds),
        "seed_first": seeds[0], "seed_last": seeds[-1],
        "checkpoint": str(checkpoint_path),
        "workers": effective_workers,
        "adapter": adapter,
        "deterministic_learner": bool(deterministic),
        "gate_count_plan": counts,
        "semantics": (
            "learner drives, frozen expert labels the learner-visited state; "
            "no blending, no safety takeover"
        ),
    }, indent=2), encoding="utf-8")

    jobs = [
        DaggerWorkerArgs(
            worker_id=index, workers=effective_workers, seeds=seeds,
            gate_counts=counts, out_dir=str(out_dir),
            checkpoint_path=str(checkpoint_path), adapter=adapter,
            allow_fallback=allow_fallback, dt=dt, deterministic=deterministic,
        )
        for index in range(effective_workers)
    ]
    if effective_workers == 1:
        results = [_run_dagger_worker(jobs[0])]
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=effective_workers) as pool_ctx:
            results = pool_ctx.map(_run_dagger_worker, jobs)

    episodes_loaded = load_corpus(out_dir)
    manifest_path = write_manifest(out_dir, episodes=episodes_loaded, extra={
        "dagger_round": int(round_index),
        "checkpoint": str(checkpoint_path),
        "workers": results,
    })
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    collected = sum(int(r["collected"]) for r in results)
    completed = sum(int(r["learner_completed"]) for r in results)
    gates = sum(int(r["learner_gates"]) for r in results)
    possible = sum(int(r["course_gates"]) for r in results)
    histogram: Dict[str, int] = {}
    for value in counts[: len(seeds)]:
        histogram[str(value)] = histogram.get(str(value), 0) + 1

    return DaggerRoundResult(
        round_index=int(round_index),
        out_dir=str(out_dir),
        episodes_requested=len(seeds),
        episodes_collected=collected,
        seeds=(seeds[0], seeds[-1]),
        gate_counts=dict(sorted(histogram.items(), key=lambda kv: int(kv[0]))),
        learner_completion_rate=round(completed / collected, 4) if collected else None,
        learner_gate_rate=round(gates / possible, 4) if possible else None,
        aggregated_transitions=int(sum(len(e) for e in episodes_loaded)),
        corpus_sha256=manifest["corpus_sha256"],
        wall_time_s=round(time.perf_counter() - started, 1),
        workers=results,
    )


def aggregated_corpus_roots(
    base: str | Path, *, through_round: int, include_bc: bool = True
) -> List[Path]:
    """Every corpus a retrain at ``through_round`` should learn from.

    DAgger *aggregates*: round r trains on the BC corpus plus rounds 1..r, not
    on round r alone.  Training only on the newest round is the classic way to
    make each iteration forget the last one.
    """
    base = Path(base)
    roots: List[Path] = []
    if include_bc:
        bc = base / "expert_corpus"
        if bc.is_dir():
            roots.append(bc)
    for index in range(1, int(through_round) + 1):
        candidate = base / f"dagger_round_{index:02d}"
        if candidate.is_dir():
            roots.append(candidate)
    return roots


def verify_no_blending(episodes: Sequence[Any]) -> Dict[str, Any]:
    """Confirm the recorded rollouts really were pure learner control.

    An episode where ``applied_actions`` equals ``expert_actions`` on a step
    that is not marked as a takeover would mean the expert was driving without
    being recorded as doing so.  Reported rather than asserted, so a legitimate
    coincidence (both agree exactly) is visible instead of fatal.
    """
    takeover_steps = 0
    total_steps = 0
    exact_matches = 0
    for episode in episodes:
        applied = np.asarray(episode.applied_actions, dtype=np.float32)
        expert = np.asarray(episode.expert_actions, dtype=np.float32)
        marked = np.asarray(episode.applied_by_expert, dtype=bool)
        total_steps += len(applied)
        takeover_steps += int(marked.sum())
        same = np.all(np.isclose(applied, expert, atol=1e-6), axis=1)
        exact_matches += int(np.count_nonzero(same & ~marked))
    return {
        "episodes": len(episodes),
        "total_steps": total_steps,
        "takeover_steps": takeover_steps,
        "takeover_fraction": round(takeover_steps / total_steps, 6) if total_steps else 0.0,
        "unmarked_exact_expert_matches": exact_matches,
        "pure_learner_rollout": takeover_steps == 0,
    }
