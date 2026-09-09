"""Drive the Gen-2 DAgger campaign: rollout, aggregate, retrain, evaluate.

Usage::

    python -m marine_race_arena.learning.gen2.run_dagger \\
        --base results/rl/gen2 --bc results/rl/gen2/bc_v1/bc_policy.zip \\
        --rounds 1,2,3 --episodes-per-round 120

One round is:

1. the learner drives ``episodes-per-round`` TRAIN courses at the round's
   curriculum lengths, with the frozen expert labelling every visited state;
2. the round's shards are aggregated with the BC corpus and every earlier
   round -- DAgger aggregates, it does not replace;
3. the policy is retrained from the **BC initialization**, not from the
   previous round's weights.  Restarting from BC each time keeps the result a
   function of the aggregated dataset rather than of the optimization path, so
   a bad round cannot permanently damage the policy;
4. the retrained policy is measured on untouched VALIDATION courses.

A round only advances when validation competence supports it -- reaching an
iteration number is not a reason to make the courses longer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from marine_race_arena.learning.gen2 import seeds as gen2_seeds
from marine_race_arena.learning.gen2.bc_recurrent import BCConfig, calibrate_action_std, train_recurrent_bc
from marine_race_arena.learning.gen2.dagger import (
    DAGGER_CURRICULUM,
    aggregated_corpus_roots,
    run_dagger_round,
    verify_no_blending,
)
from marine_race_arena.learning.gen2.dataset import corpus_statistics, load_corpus, observation_statistics

#: Minimum validation completion at the round's own lengths before advancing.
ROUND_ADVANCE_COMPLETION: Dict[int, float] = {1: 0.80, 2: 0.75, 3: 0.65, 4: 0.55, 5: 0.40}

#: Validation episodes per round.
ROUND_VALIDATION_EPISODES = 40  # <= GEN2_VALIDATION_BLOCK


@dataclass
class RoundReport:
    round_index: int
    lengths: List[int]
    rollout: Dict[str, Any]
    blending_audit: Dict[str, Any]
    aggregated_corpus: Dict[str, Any]
    bc: Dict[str, Any]
    validation: Dict[str, Any]
    advance: bool
    advance_reason: str
    checkpoint: str
    wall_time_s: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validation_seeds(round_index: int, episodes: int) -> List[int]:
    """Disjoint VALIDATION slices per round, so rounds are not compared on reruns."""
    chosen = gen2_seeds.dagger_validation_seeds(int(round_index), int(episodes))
    for seed in chosen:
        if gen2_seeds.band_of(seed) != "VALIDATION":
            raise PermissionError(f"round {round_index} drew a non-validation seed {seed}")
    return chosen


def retrain_on_aggregate(
    base: str | Path,
    *,
    through_round: int,
    out_dir: str | Path,
    config: Optional[BCConfig] = None,
) -> Dict[str, Any]:
    """Retrain from a fresh BC initialization on the aggregated corpus."""
    from marine_race_arena.learning.gen2.recurrent_policy import (
        build_gen2_policy_for_training,
        write_policy_manifest,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = aggregated_corpus_roots(base, through_round=through_round)
    episodes = load_corpus(roots)
    if not episodes:
        raise ValueError(f"aggregated corpus is empty: {roots}")
    statistics = corpus_statistics(episodes)
    mean, std = observation_statistics(episodes)

    config = config or BCConfig()
    model = build_gen2_policy_for_training(
        seed=config.seed, obs_mean=mean, obs_std=std, device=config.device
    )
    result = train_recurrent_bc(
        model, episodes, config, progress_path=out_dir / "bc_progress.json"
    )
    action_std = calibrate_action_std(model, episodes, config)
    checkpoint = out_dir / "dagger_policy.zip"
    model.save(checkpoint)
    write_policy_manifest(
        model, out_dir / "policy_manifest.json",
        training_method=f"recurrent_bc_on_dagger_aggregate_through_round_{through_round}",
        corpus=[str(r) for r in roots],
        corpus_statistics=statistics,
        bc_result={k: v for k, v in result.as_dict().items() if k != "history"},
        action_std=[round(float(v), 5) for v in action_std],
    )
    return {
        "checkpoint": str(checkpoint),
        "corpus_roots": [str(r) for r in roots],
        "corpus_statistics": statistics,
        "bc": result.as_dict(),
        "action_std": [round(float(v), 5) for v in action_std],
    }


def evaluate_round(
    checkpoint: str | Path,
    *,
    round_index: int,
    episodes: int,
    track_dir: str | Path,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
) -> Dict[str, Any]:
    """Measure the retrained policy on untouched VALIDATION courses."""
    from sb3_contrib import RecurrentPPO

    from marine_race_arena.learning.gen2.evaluation import evaluate_policy
    from marine_race_arena.learning.gen2.recurrent_policy import Gen2RecurrentController

    plan = next(r for r in DAGGER_CURRICULUM if r["round"] == int(round_index))
    model = RecurrentPPO.load(str(checkpoint), device="cpu")
    controller = Gen2RecurrentController(model, deterministic=True)
    seeds = _validation_seeds(round_index, episodes)
    result = evaluate_policy(
        controller, seeds,
        gate_counts=list(plan["lengths"]),
        track_dir=track_dir, adapter=adapter, allow_fallback=allow_fallback,
    )
    return result


def run_campaign(
    base: str | Path,
    bc_checkpoint: str | Path,
    *,
    rounds: Sequence[int] = (1, 2, 3),
    episodes_per_round: int = 120,
    validation_episodes: int = ROUND_VALIDATION_EPISODES,
    workers: int = 1,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
    config: Optional[BCConfig] = None,
    stop_on_stall: bool = True,
) -> Dict[str, Any]:
    """Run DAgger rounds, advancing only on measured validation competence."""
    base = Path(base)
    reports: List[RoundReport] = []
    current_checkpoint = Path(bc_checkpoint)

    for round_index in rounds:
        started = time.perf_counter()
        round_dir = base / f"dagger_round_{int(round_index):02d}"
        rollout = run_dagger_round(
            current_checkpoint,
            round_index=int(round_index),
            episodes=int(episodes_per_round),
            out_dir=round_dir,
            workers=workers,
            adapter=adapter,
            allow_fallback=allow_fallback,
        )
        audit = verify_no_blending(load_corpus(round_dir))
        retrain_dir = base / f"dagger_policy_r{int(round_index):02d}"
        retrained = retrain_on_aggregate(
            base, through_round=int(round_index), out_dir=retrain_dir, config=config
        )
        validation = evaluate_round(
            retrained["checkpoint"], round_index=int(round_index),
            episodes=validation_episodes,
            track_dir=retrain_dir / "eval_tracks",
            adapter=adapter, allow_fallback=allow_fallback,
        )
        completion = validation["metrics"].get("overall_completion_rate")
        bar = ROUND_ADVANCE_COMPLETION.get(int(round_index), 0.5)
        advance = completion is not None and completion >= bar
        reason = (
            f"validation completion {completion} >= {bar} at round {round_index} lengths"
            if advance else
            f"validation completion {completion} < {bar}; the next round's longer "
            f"courses would train on states the policy cannot yet reach"
        )
        report = RoundReport(
            round_index=int(round_index),
            lengths=list(next(r for r in DAGGER_CURRICULUM if r["round"] == int(round_index))["lengths"]),
            rollout=rollout.as_dict(),
            blending_audit=audit,
            aggregated_corpus=retrained["corpus_statistics"],
            bc={k: v for k, v in retrained["bc"].items() if k != "history"},
            validation=validation["metrics"],
            advance=advance,
            advance_reason=reason,
            checkpoint=retrained["checkpoint"],
            wall_time_s=round(time.perf_counter() - started, 1),
        )
        reports.append(report)
        (base / "dagger_campaign.json").write_text(
            json.dumps([r.as_dict() for r in reports], indent=2), encoding="utf-8"
        )
        current_checkpoint = Path(retrained["checkpoint"])
        if not advance and stop_on_stall:
            break

    best = reports[-1].checkpoint if reports else str(bc_checkpoint)
    if reports:
        target = base / "best_dagger.zip"
        shutil.copyfile(best, target)
        best = str(target)
    return {
        "schema_version": "gen2_dagger_campaign_v1",
        "rounds": [r.as_dict() for r in reports],
        "best_dagger": best,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Gen-2 DAgger campaign")
    parser.add_argument("--base", required=True)
    parser.add_argument("--bc", required=True, help="BC checkpoint to start from")
    parser.add_argument("--rounds", default="1,2,3")
    parser.add_argument("--episodes-per-round", type=int, default=120)
    parser.add_argument("--validation-episodes", type=int, default=ROUND_VALIDATION_EPISODES)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--adapter", default="holoocean", choices=("holoocean", "fallback"))
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-stop-on-stall", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_campaign(
        args.base, args.bc,
        rounds=[int(v) for v in args.rounds.split(",") if v.strip()],
        episodes_per_round=args.episodes_per_round,
        validation_episodes=args.validation_episodes,
        workers=args.workers, adapter=args.adapter,
        allow_fallback=args.allow_fallback,
        config=BCConfig(epochs=args.epochs, seed=args.seed),
        stop_on_stall=not args.no_stop_on_stall,
    )
    for report in summary["rounds"]:
        print(json.dumps({
            "round": report["round_index"],
            "lengths": report["lengths"],
            "learner_completion_during_rollout": report["rollout"]["learner_completion_rate"],
            "pure_learner_rollout": report["blending_audit"]["pure_learner_rollout"],
            "aggregated_transitions": report["aggregated_corpus"]["transitions"],
            "validation_completion": report["validation"].get("overall_completion_rate"),
            "gate1_to_gate2": report["validation"].get("gate1_to_gate2_transition_rate"),
            "advance": report["advance"],
            "reason": report["advance_reason"],
        }, indent=2), flush=True)
    print(f"[gen2] best_dagger={summary['best_dagger']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
