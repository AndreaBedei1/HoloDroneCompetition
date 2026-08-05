"""Fair equal-environment-budget PPO versus SAC transition comparison."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from marine_race_arena.learning.longrun_checkpoint import atomic_write_json
from marine_race_arena.learning.provenance import now_utc


DEFAULT_BUDGETS = (25_000, 50_000, 75_000, 100_000, 150_000, 200_000)


def _json_lines(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except (TypeError, ValueError):
            continue
    return rows


def _evaluations(run: Path, cutoff: Optional[int]) -> list[Dict[str, Any]]:
    rows = []
    for path in sorted((run / "evaluations").glob("unseen_*/evaluation.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        steps = int(report.get("timesteps") or path.parent.name.rsplit("_", 1)[-1])
        if cutoff is not None and steps > int(cutoff):
            continue
        rows.append({
            "timesteps": steps,
            "nominal_timestep": int(report.get("nominal_timestep", steps)),
            "metrics": dict(report.get("metrics") or {}),
            "competence": report.get("competence"),
            "path": str(path),
        })
    return sorted(rows, key=lambda row: row["timesteps"])


def _at_budget(rows: Iterable[Mapping[str, Any]], budget: int) -> Optional[Dict[str, Any]]:
    values = list(rows)
    if not values:
        return None
    after = [row for row in values if int(row["timesteps"]) >= int(budget)]
    selected = min(after, key=lambda row: int(row["timesteps"])) if after else None
    if selected is None:
        return None
    return dict(selected)


def _timing(run: Path, algorithm: str) -> Dict[str, Any]:
    progress = _json_lines(run / "logs" / "progress.jsonl")
    timing = _json_lines(run / "logs" / "timing.jsonl")
    rates = [
        float(row["environment_transitions_per_second"])
        for row in progress if row.get("environment_transitions_per_second") is not None
    ]
    update_rates = [
        float(row["learner_updates_per_second"])
        for row in progress if row.get("learner_updates_per_second") is not None
    ]
    status = {}
    try:
        status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        pass
    workers = int(status.get("n_envs", 0) or 0)
    mean_rate = None if not rates else statistics.fmean(rates)
    return {
        "algorithm": algorithm,
        "progress_samples": len(progress),
        "mean_environment_transitions_per_second": mean_rate,
        "mean_learner_updates_per_second": (
            None if not update_rates else statistics.fmean(update_rates)
        ),
        "reported_evaluation_wall_time_s": sum(
            float(row.get("evaluation_wall_time_s", 0.0) or 0.0) for row in timing
        ),
        "reported_checkpoint_wall_time_s": sum(
            float(row.get("checkpoint_wall_time_s", 0.0) or 0.0) for row in timing
        ),
        "reported_replay_sampling_wall_time_s": sum(
            float(row.get("replay_sampling_wall_time_s", 0.0) or 0.0)
            for row in progress
        ),
        "rollout_workers": workers or None,
        "rollout_engine_seconds_per_1000_transitions": (
            None if not workers or not mean_rate
            else 1000.0 * workers / mean_rate
        ),
        "resource_cost_scope": (
            "actual rollout throughput and configured rollout engines; CPU/GPU/"
            "RAM/VRAM are reported by the separate concurrent capacity benchmark"
        ),
    }


def build_comparison(
    ppo_run: str | Path,
    sac_run: str | Path,
    *,
    cutoff: Optional[int] = None,
    budgets: Iterable[int] = DEFAULT_BUDGETS,
) -> Dict[str, Any]:
    ppo = Path(ppo_run)
    sac = Path(sac_run)
    ppo_rows = _evaluations(ppo, cutoff)
    sac_rows = _evaluations(sac, cutoff)
    equal = []
    for budget in budgets:
        ppo_value = _at_budget(ppo_rows, int(budget))
        sac_value = _at_budget(sac_rows, int(budget))
        equal.append({
            "budget": int(budget),
            "ppo": ppo_value,
            "sac": sac_value,
            "comparable": ppo_value is not None and sac_value is not None,
        })
    return {
        "schema_version": "universal_transition_ppo_sac_comparison_v1",
        "generated_utc": now_utc(),
        "ppo_run": str(ppo),
        "sac_run": str(sac),
        "cutoff": cutoff,
        "task_contract": {
            "observation_version": "onboard_local_transition_v1",
            "observation_dim": 35,
            "action_version": "surge_sway_heave_yaw_pm1_v1",
            "evaluation_seed": 88001,
        },
        "equal_environment_transition_budgets": equal,
        "wall_clock": {
            "ppo": _timing(ppo, "ppo"),
            "sac": _timing(sac, "sac"),
        },
        "conclusion_allowed": any(row["comparable"] for row in equal),
        "caveat": (
            "Do not rank algorithms until both have an unseen evaluation at "
            "the same new-environment-transition budget. Training reward is not used."
        ),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# PPO versus multi-step SAC: universal transition",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        "| Budget | PPO actual | PPO success | SAC actual | SAC success | Comparable |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["equal_environment_transition_budgets"]:
        ppo = row["ppo"]
        sac = row["sac"]
        ppo_success = None if ppo is None else ppo["metrics"].get("universal_transition_success_rate")
        sac_success = None if sac is None else sac["metrics"].get("universal_transition_success_rate")
        lines.append(
            f"| {row['budget']} | {'n/a' if ppo is None else ppo['timesteps']} | "
            f"{'n/a' if ppo_success is None else f'{100*float(ppo_success):.1f}%'} | "
            f"{'n/a' if sac is None else sac['timesteps']} | "
            f"{'n/a' if sac_success is None else f'{100*float(sac_success):.1f}%'} | "
            f"{row['comparable']} |"
        )
    lines += ["", str(report["caveat"]), ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-run", required=True)
    parser.add_argument("--sac-run", required=True)
    parser.add_argument("--cutoff", type=int)
    parser.add_argument(
        "--output-dir",
        default="results/rl_public/universal_transition_ppo_vs_sac",
    )
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    report = build_comparison(args.ppo_run, args.sac_run, cutoff=args.cutoff)
    atomic_write_json(output / "comparison.json", report)
    markdown = output / "comparison.md"
    temporary = markdown.with_suffix(".md.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(_markdown(report), encoding="utf-8")
    temporary.replace(markdown)
    print(json.dumps({"json": str(output / 'comparison.json'), "markdown": str(markdown)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
