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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.gen2 import seeds as gen2_seeds
from marine_race_arena.learning.gen2.bc_recurrent import (
    BCConfig,
    calibrate_action_std,
    split_episodes,
    train_recurrent_bc,
)
from marine_race_arena.learning.gen2.dataset import (
    corpus_statistics,
    load_corpus,
    load_corpus_cached,
    observation_statistics,
)

#: Learning rate for warm-started fine-tuning.
#:
#: The from-scratch default of 3e-4 destroyed a converged policy in a single
#: epoch: starting from train MSE 8.3e-05, the first epoch on a corpus that was
#: one-third learner-visited DAgger states landed at 2.65e-03 -- 32x worse --
#: and eight epochs never recovered. SB3's ``load`` restores the weights but not
#: the optimizer moments, so a fresh Adam takes large unconditioned steps away
#: from a minimum. An order of magnitude lower keeps the warm start.
FINETUNE_LEARNING_RATE = 3e-5

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
    from marine_race_arena.learning.gen2.evaluation import (
        evaluate_policy,
        failures_by_pattern,
    )

    plan = BC_STAGES[stage]
    seeds = stage_seeds(stage, episodes)
    result = evaluate_policy(
        controller, seeds,
        gate_counts=[int(plan["gates"])],
        track_dir=Path(track_dir) / f"stage_{stage}",
        adapter=adapter, allow_fallback=allow_fallback,
    )
    metrics = result["metrics"]
    # Persist the per-episode rows. Keeping only the aggregate is what left the
    # first out-of-bounds finding undiagnosable: a rate of 0.100 says nothing
    # about which geometry produced it.
    rows_path = Path(track_dir).parent / f"stage_{stage}_episodes.json"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows_path.write_text(json.dumps({
        "stage": stage,
        "gates": int(plan["gates"]),
        "metrics": metrics,
        "diagnosis": failures_by_pattern(result["episodes"]),
        "episodes": result["episodes"],
    }, indent=2), encoding="utf-8")
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


def load_weighted_corpus(
    corpus: Sequence[str | Path],
    weights: Optional[Sequence[int]] = None,
    *,
    completed_only: bool = False,
    validation_fraction: float = 0.0,
    seed: int = 0,
) -> Tuple[List, List]:
    """Load each corpus root and return ``(train, validation)`` episode lists.

    Repetition is how a small corpus is given enough weight to change
    behaviour.  Repeating whole episodes keeps every sequence intact, which
    matters for a recurrent policy; reweighting individual steps would not.

    **The split happens before replication, and that ordering is the point.**
    Replicating first and splitting afterwards puts byte-identical copies of
    the same trajectory on both sides of the split: a x3 fragment corpus mixed
    into a x1 general corpus leaked 35.1% of validation transitions into
    training, and the resulting validation curve reported an improvement that
    did not exist.  Splitting first means a held-out episode has no duplicate
    anywhere in training, whatever the weights are.
    """
    roots = list(corpus)
    counts = list(weights) if weights else [1] * len(roots)
    if len(counts) != len(roots):
        raise ValueError("one weight per corpus root is required")
    train: List = []
    validation: List = []
    for index, (root, repeat) in enumerate(zip(roots, counts)):
        loaded = load_corpus_cached(root, completed_only=completed_only)
        if not loaded:
            raise ValueError(f"corpus root {root} contributed no episodes")
        # Split each root independently so the validation set keeps the same
        # corpus mixture as training rather than being dominated by whichever
        # root happens to be larger.
        if validation_fraction > 0.0:
            kept, held = split_episodes(loaded, validation_fraction, seed + index)
        else:
            kept, held = list(loaded), []
        for _ in range(max(1, int(repeat))):
            train.extend(kept)
        validation.extend(held)
    return train, validation


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
    init_from: Optional[str | Path] = None,
    corpus_weights: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Clone the expert corpus, then measure the result closed-loop.

    ``init_from`` warm-starts from an existing checkpoint instead of a random
    network.  When it is given the observation normalization is **kept** rather
    than re-latched: the incoming weights were fitted against those statistics,
    and rescaling the input under them would discard the warm start it is the
    whole point of.
    """
    from marine_race_arena.learning.gen2.recurrent_policy import (
        Gen2RecurrentController,
        build_gen2_policy_for_training,
        policy_parameter_count,
        write_policy_manifest,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    config = config or BCConfig()
    train_episodes, validation_episodes = load_weighted_corpus(
        list(corpus), corpus_weights, completed_only=completed_only,
        validation_fraction=config.validation_fraction, seed=config.seed,
    )
    if not train_episodes:
        raise ValueError(f"no episodes found under {list(corpus)}")
    episodes = train_episodes
    statistics = corpus_statistics(train_episodes + validation_episodes)
    mean, std = observation_statistics(train_episodes)

    if init_from is not None:
        from sb3_contrib import RecurrentPPO

        model = RecurrentPPO.load(str(init_from), device=config.device)
        if config.learning_rate >= BCConfig().learning_rate:
            # Caller left the from-scratch default in place; a warm start needs
            # a conservative rate or it walks straight out of the minimum.
            config = BCConfig(**{
                **config.as_dict(), "learning_rate": FINETUNE_LEARNING_RATE
            })
    else:
        model = build_gen2_policy_for_training(
            seed=config.seed, obs_mean=mean, obs_std=std, device=config.device
        )
    result = train_recurrent_bc(
        model, episodes, config,
        latch_normalization=init_from is None,
        progress_path=out_dir / "bc_progress.json",
        validation_episodes=validation_episodes,
    )
    action_std = calibrate_action_std(model, episodes, config)

    checkpoint = out_dir / "bc_policy.zip"
    warm_start = str(init_from) if init_from is not None else None
    model.save(checkpoint)
    write_policy_manifest(
        model, out_dir / "policy_manifest.json",
        training_method=(
            "recurrent_behavior_cloning" if warm_start is None
            else "recurrent_bc_finetune_from_checkpoint"
        ),
        warm_start=warm_start,
        corpus=list(str(c) for c in corpus),
        corpus_weights=list(corpus_weights) if corpus_weights else None,
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
            # A narrow miss is still worth profiling: knowing the shape of the
            # decay across 3/5/8 gates is what tells you whether the deficit is
            # transition quality or long-horizon drift. Only abandon the longer
            # stages when the policy has actually collapsed, where the extra
            # simulator hours would buy a row of zeros.
            if outcome.completion_rate is not None and outcome.completion_rate < 0.5:
                break

    summary = {
        "schema_version": "gen2_bc_run_v1",
        "checkpoint": str(checkpoint),
        "warm_start": warm_start,
        "corpus": [str(c) for c in corpus],
        "corpus_weights": list(corpus_weights) if corpus_weights else None,
        "corpus_statistics": statistics,
        "parameters": policy_parameter_count(model),
        "bc": result.as_dict(),
        "improved_on_parent": result.improved_on_parent,
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
    parser.add_argument("--init-from", default=None,
                        help="warm-start from this checkpoint instead of random init")
    parser.add_argument("--corpus-weights", default=None,
                        help="comma-separated repeat count per --corpus root")
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
        init_from=args.init_from,
        corpus_weights=(
            [int(v) for v in args.corpus_weights.split(",")]
            if args.corpus_weights else None
        ),
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
