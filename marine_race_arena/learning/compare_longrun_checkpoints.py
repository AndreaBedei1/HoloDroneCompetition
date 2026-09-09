"""Deterministic HoloOcean comparison for long-run reliability checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from marine_race_arena.learning.longrun_checkpoint import atomic_write_json
from marine_race_arena.learning.longrun_config import LongRunConfig
from marine_race_arena.learning.longrun_evaluation import (
    evaluate_longrun_policy,
)
from marine_race_arena.learning.reliability_first import ReliabilityFirstPPO
from marine_race_arena.learning.reward_v3 import MultiGateRewardConfig
from marine_race_arena.learning.seed_registry import (
    MULTIGATE_RELIABILITY_DEV_EVAL_SEEDS,
)


def _checkpoint_timesteps(path: Path) -> int:
    match = re.search(r"ppo_(\d+)_steps", path.stem)
    if not match:
        raise ValueError(f"cannot infer checkpoint timestep from {path}")
    return int(match.group(1))


def _metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: report.get(key)
        for key in (
            "n_eval",
            "completions",
            "completion_rate",
            "collision_events",
            "out_of_bounds_events",
            "wrong_direction_count",
            "previous_gate_returns",
            "mean_action_jerk",
            "mean_successful_time_s",
            "mean_penalized_time_s",
            "left_completion_rate",
            "right_completion_rate",
            "straight_completion_rate",
            "single_gate_completion_rate",
            "all_actions_finite",
        )
    }


def _trace_summary(row: Dict[str, Any], *, radius: int = 6) -> Dict[str, Any]:
    trace = list(row.get("trace", []))
    collision_indices = [
        index
        for index, step in enumerate(trace)
        if step.get("collision_contact_frame")
    ]
    if not collision_indices:
        return {
            "collision_step": None,
            "trajectory_phase": None,
            "action_window": [],
        }
    collision_index = collision_indices[0]
    collision = trace[collision_index]
    start = max(0, collision_index - int(radius))
    end = min(len(trace), collision_index + int(radius) + 1)
    return {
        "collision_step": collision.get("step"),
        "collision_time_s": collision.get("time_s"),
        "trajectory_phase": collision.get("phase"),
        "collision_position": collision.get("position"),
        "collision_rotation_rpy_deg": collision.get("rotation_rpy_deg"),
        "action_window": trace[start:end],
    }


def compare_checkpoints(
    *,
    run_dir: str | Path,
    checkpoint_a: str | Path,
    checkpoint_b: Optional[str | Path],
    failed_evaluation: str | Path,
    output_dir: str | Path,
) -> Dict[str, Any]:
    run_path = Path(run_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = LongRunConfig.load(run_path / "config.json")
    failed = json.loads(Path(failed_evaluation).read_text(encoding="utf-8"))
    failed_rows = [
        row
        for row in failed.get("rows", [])
        if int(row.get("collision_events", 0)) > 0
    ]
    if len(failed_rows) != 1:
        raise ValueError("failed evaluation must identify exactly one collision case")
    failed_row = failed_rows[0]
    failed_seed = int(failed_row["seed"])
    failed_index = next(
        index
        for index, row in enumerate(failed["rows"])
        if int(row["seed"]) == failed_seed
    )
    phase = config.reliability.reward_phase
    reward = MultiGateRewardConfig(
        reward_phase="reliability",
        reliability_time_cost=phase.reliability_time_cost,
        efficiency_time_cost=phase.efficiency_time_cost,
        efficiency_detour_penalty=phase.efficiency_detour_penalty,
        efficiency_jerk_penalty=phase.efficiency_jerk_penalty,
        efficiency_energy_penalty=phase.efficiency_energy_penalty,
        per_step_efficiency_penalty_cap=phase.per_step_efficiency_penalty_cap,
        action_change_penalty=phase.action_change_penalty,
    )
    env_kwargs = {
        "adapter": "holoocean",
        "allow_fallback": False,
        "current_profile": config.current_profile,
        "max_steps": config.max_episode_steps,
        "observation_encoding_version": config.observation_version,
    }
    seeds = MULTIGATE_RELIABILITY_DEV_EVAL_SEEDS[
        : config.evaluation.full_episodes + 4
    ]
    results: Dict[str, Any] = {}
    checkpoints = [("checkpoint_a", Path(checkpoint_a))]
    if checkpoint_b is not None:
        checkpoints.append(("checkpoint_b", Path(checkpoint_b)))
    for label, checkpoint in checkpoints:
        timesteps = _checkpoint_timesteps(checkpoint)
        model = ReliabilityFirstPPO.load(str(checkpoint), device="cpu")

        def progress(phase_name: str):
            def update(value: Dict[str, Any]) -> None:
                atomic_write_json(
                    out / "progress.json",
                    {
                        "checkpoint": label,
                        "timesteps": timesteps,
                        "phase": phase_name,
                        "completed": int(value["completed"]),
                        "total": int(value["total"]),
                    },
                )

            return update

        scenario = evaluate_longrun_policy(
            model,
            stage="C3",
            mode="full",
            seeds=seeds,
            output_dir=out / label / "failed_scenario",
            env_kwargs=env_kwargs,
            reward_config=reward,
            timesteps=timesteps,
            policy_mode=config.policy_mode,
            frame_stack=config.frame_stack,
            trace_case_seeds=(failed_seed,),
            case_indices=(failed_index,),
            progress_callback=progress("failed_scenario"),
        )
        suite = evaluate_longrun_policy(
            model,
            stage="C3",
            mode="full",
            seeds=seeds,
            output_dir=out / label / "full_suite",
            env_kwargs=env_kwargs,
            reward_config=reward,
            timesteps=timesteps,
            policy_mode=config.policy_mode,
            frame_stack=config.frame_stack,
            progress_callback=progress("full_suite"),
        )
        scenario_row = scenario["rows"][0]
        results[label] = {
            "checkpoint": str(checkpoint),
            "timesteps": timesteps,
            "failed_scenario": {
                **_metrics(scenario),
                "category": scenario_row.get("category"),
                "direction": scenario_row.get("direction"),
                "seed": scenario_row.get("seed"),
                "geometry": scenario_row.get("geometry"),
                "raw_time_s": scenario_row.get("raw_time_s"),
                "penalized_time_s": scenario_row.get("penalized_time_s"),
                "action_jerk": scenario_row.get("action_jerk"),
                "trace_summary": _trace_summary(scenario_row),
            },
            "full_suite": _metrics(suite),
            "scenario_report": str(
                out
                / label
                / "failed_scenario"
                / f"full_{timesteps:09d}.json"
            ),
            "suite_report": str(
                out / label / "full_suite" / f"full_{timesteps:09d}.json"
            ),
        }
    comparison = {
        "schema_version": "multigate_checkpoint_comparison_v1",
        "failed_evaluation": str(failed_evaluation),
        "failed_case_index": failed_index,
        "failed_seed": failed_seed,
        "results": results,
    }
    atomic_write_json(out / "comparison.json", comparison)
    return comparison


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", default=None)
    parser.add_argument("--failed-evaluation", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = compare_checkpoints(
        run_dir=args.run_dir,
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        failed_evaluation=args.failed_evaluation,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
