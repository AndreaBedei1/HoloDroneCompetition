"""Train the Gen-2 recurrent BC policy and evaluate it closed-loop by stage.

Usage::

    python -m marine_race_arena.learning.gen2.train_bc \\
        --corpus results/rl/gen2/expert_corpus \\
        --out results/rl/gen2/bc_v1 \\
        --stages A,B,C --episodes-per-stage 30

Progression is staged on purpose (the brief's stages A..E):

===== ======  ===================
Stage Gates   Progression target
===== ======  ===================
A     2       0.95 completion
B     3       0.90
C     5       0.80
D     8       --  (diagnostic)
E     12      --  (diagnostic)
===== ======  ===================

These are *progression* criteria for deciding whether to keep cloning or move
to DAgger -- they are not paper claims and they are not the readiness gate.

The decision rule that matters: **closed-loop behaviour beats supervised MSE.**
A tiny action MSE with poor rollout completion means the corpus does not cover
the states the learner actually reaches, and no amount of further cloning fixes
that -- only DAgger does.  :func:`recommend_next_phase` encodes that rule so it
is applied consistently rather than by eyeballing a loss curve.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from marine_race_arena.learning.gen2 import seeds as gen2_seeds
from marine_race_arena.learning.gen2.bc_recurrent import (
    BCConfig,
    calibrate_action_std,
    train_recurrent_bc,
)
from marine_race_arena.learning.gen2.dataset import corpus_statistics, load_corpus, observation_statistics

#: Stage -> (gate count, progression target completion rate).
BC_STAGES: Dict[str, Any] = {
    "A": {"gates": 2, "target": 0.95},
    "B": {"gates": 3, "target": 0.90},
    "C": {"gates": 5, "target": 0.80},
    "D": {"gates": 8, "target": None},
    "E": {"gates": 12, "target": None},
}


@dataclass
class StageResult:
    stage: str
    gates: int
    target: Optional[float]
    episodes: int
    completion_rate: Optional[float]
    first_gate_rate: Optional[float]
    gate1_to_gate2_rate: Optional[float]
    collision_rate: Optional[float]
    out_of_bounds_rate: Optional[float]
    passed: Optional[bool]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def stage_seeds(stage: str, episodes: int) -> List[int]:
    """Disjoint VALIDATION slices, so stages never share courses."""
    chosen = gen2_seeds.bc_stage_seeds(list(BC_STAGES).index(stage), int(episodes))
    for seed in chosen:
        if gen2_seeds.band_of(seed) != "VALIDATION":
            raise PermissionError(f"stage {stage} drew a non-validation seed {seed}")
    return chosen


def evaluate_stage(
    controller,
    stage: str,
    *,
    episodes: int,
    track_dir: str | Path,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
) -> StageResult:
    """Closed-loop rollout of the learned controller on one stage."""
    from marine_race_arena.learning.gen2.evaluation import evaluate_policy

    plan = BC_STAGES[stage]
    seeds = stage_seeds(stage, episodes)
    result = evaluate_policy(
        controller, seeds,
        gate_counts=[int(plan["gates"])],
        track_dir=Path(track_dir) / f"stage_{stage}",
        adapter=adapter, allow_fallback=allow_fallback,
    )
    metrics = result["metrics"]
    completion = metrics.get("overall_completion_rate")
    target = plan["target"]
    return StageResult(
        stage=stage,
        gates=int(plan["gates"]),
        target=target,
        episodes=len(seeds),
        completion_rate=completion,
        first_gate_rate=metrics.get("first_gate_crossing_rate"),
        gate1_to_gate2_rate=metrics.get("gate1_to_gate2_transition_rate"),
        collision_rate=metrics.get("collision_episode_rate"),
        out_of_bounds_rate=metrics.get("out_of_bounds_episode_rate"),
        passed=None if target is None or completion is None else completion >= target,
    )


def recommend_next_phase(
    stages: Sequence[StageResult], validation_mse: Optional[float]
) -> Dict[str, Any]:
    """Decide whether to keep cloning or move to DAgger, and say why.

    Cloning more only helps while the supervised fit is still the limiting
    factor.  Once the fit is good and the rollouts are not, the gap is state
    distribution, not capacity -- that is DAgger's job.
    """
    graded = [s for s in stages if s.passed is not None]
    if not graded:
        return {"phase": "evaluate", "reason": "no graded stage was measured"}
    failed = [s for s in graded if not s.passed]
    if not failed:
        return {
            "phase": "dagger",
            "reason": (
                "every graded BC stage met its progression target; DAgger should "
                "now extend competence to longer sequences"
            ),
        }
    fit_is_good = validation_mse is not None and validation_mse < 5e-3
    if fit_is_good:
        return {
            "phase": "dagger",
            "reason": (
                f"supervised fit is already tight (validation MSE {validation_mse:.5f}) "
                f"but stages {[s.stage for s in failed]} miss their closed-loop target: "
                "the gap is off-expert state distribution, which more cloning cannot "
                "close"
            ),
        }
    return {
        "phase": "more_bc",
        "reason": (
            f"stages {[s.stage for s in failed]} miss their target and the supervised "
            f"fit is still loose (validation MSE {validation_mse}); more cloning is "
            "still the cheaper improvement"
        ),
    }


def train(
    corpus: Sequence[str | Path],
    out_dir: str | Path,
    *,
    config: Optional[BCConfig] = None,
    stages: Sequence[str] = ("A", "B", "C"),
    episodes_per_stage: int = 30,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
    completed_only: bool = False,
    evaluate: bool = True,
) -> Dict[str, Any]:
    """Clone the expert corpus, then measure the result closed-loop."""
    from marine_race_arena.learning.gen2.recurrent_policy import (
        Gen2RecurrentController,
        build_gen2_policy_for_training,
        policy_parameter_count,
        write_policy_manifest,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    episodes = load_corpus(list(corpus), completed_only=completed_only)
    if not episodes:
        raise ValueError(f"no episodes found under {list(corpus)}")
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

    checkpoint = out_dir / "bc_policy.zip"
    model.save(checkpoint)
    write_policy_manifest(
        model, out_dir / "policy_manifest.json",
        training_method="recurrent_behavior_cloning",
        corpus=list(str(c) for c in corpus),
        corpus_statistics=statistics,
        bc_result={k: v for k, v in result.as_dict().items() if k != "history"},
        action_std=[round(float(v), 5) for v in action_std],
    )

    stage_results: List[StageResult] = []
    if evaluate:
        controller = Gen2RecurrentController(model, deterministic=True)
        for stage in stages:
            outcome = evaluate_stage(
                controller, stage, episodes=episodes_per_stage,
                track_dir=out_dir / "eval_tracks",
                adapter=adapter, allow_fallback=allow_fallback,
            )
            stage_results.append(outcome)
            (out_dir / "stage_results.json").write_text(
                json.dumps([s.as_dict() for s in stage_results], indent=2), encoding="utf-8"
            )
            # Do not spend simulator time on a longer stage once a shorter one
            # has already failed its progression target.
            if outcome.passed is False:
                break

    summary = {
        "schema_version": "gen2_bc_run_v1",
        "checkpoint": str(checkpoint),
        "corpus": [str(c) for c in corpus],
        "corpus_statistics": statistics,
        "parameters": policy_parameter_count(model),
        "bc": result.as_dict(),
        "action_std": [round(float(v), 5) for v in action_std],
        "stages": [s.as_dict() for s in stage_results],
        "recommendation": recommend_next_phase(
            stage_results, result.validation.get("mse")
        ),
        "wall_time_s": round(time.perf_counter() - started, 1),
    }
    (out_dir / "bc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Gen-2 recurrent BC policy")
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--chunk-length", type=int, default=64)
    parser.add_argument("--batch-episodes", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stages", default="A,B,C")
    parser.add_argument("--episodes-per-stage", type=int, default=30)
    parser.add_argument("--adapter", default="holoocean", choices=("holoocean", "fallback"))
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--completed-only", action="store_true",
                        help="clone only episodes where the expert finished the course")
    parser.add_argument("--no-eval", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = train(
        args.corpus, args.out,
        config=BCConfig(
            epochs=args.epochs, chunk_length=args.chunk_length,
            batch_episodes=args.batch_episodes, learning_rate=args.learning_rate,
            patience=args.patience, seed=args.seed, device=args.device,
        ),
        stages=tuple(s.strip().upper() for s in args.stages.split(",") if s.strip()),
        episodes_per_stage=args.episodes_per_stage,
        adapter=args.adapter, allow_fallback=args.allow_fallback,
        completed_only=args.completed_only, evaluate=not args.no_eval,
    )
    print(json.dumps({
        "train_mse": summary["bc"]["train"]["mse"],
        "validation_mse": summary["bc"]["validation"]["mse"],
        "per_axis_correlation": summary["bc"]["validation"]["per_axis_correlation"],
        "stages": summary["stages"],
        "recommendation": summary["recommendation"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
