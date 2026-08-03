"""Run and rank the controlled warm-start versus scratch PPO comparison."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

from marine_race_arena.learning.longrun_checkpoint import latest_valid_checkpoint
from marine_race_arena.learning.provenance import now_utc
from marine_race_arena.learning.transition_evaluation import (
    evaluate_universal_transition_benchmark,
    transition_checkpoint_rank_key,
)


def _load(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run_dir(config: Mapping[str, Any]) -> Path:
    return Path(config["output_root"]) / str(config["run_name"])


def _comparison_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    value = copy.deepcopy(dict(config))
    value.pop("run_name", None)
    initialization = value.pop("initialization", {})
    value["initialization_source_excluded"] = sorted(initialization)
    return value


def validate_pair(warm: Mapping[str, Any], scratch: Mapping[str, Any]) -> None:
    if warm["initialization"]["mode"] != "selective_warm_start":
        raise ValueError("warm config is not selective_warm_start")
    if scratch["initialization"]["mode"] != "scratch":
        raise ValueError("scratch config is not scratch")
    if _comparison_contract(warm) != _comparison_contract(scratch):
        raise ValueError("A/B configs differ outside run name and initialization")
    steps = int(warm["new_environment_steps"])
    if not 30_000 <= steps <= 50_000:
        raise ValueError("A/B comparison must use 30,000--50,000 transitions")
    if int(warm["evaluation"]["dedicated_transition_cases"]) < 1000:
        raise ValueError("dedicated unseen comparison requires at least 1000 transitions")


def _completed(config: Mapping[str, Any]) -> bool:
    status = _run_dir(config) / "status.json"
    if not status.exists():
        return False
    value = json.loads(status.read_text(encoding="utf-8"))
    return (
        value.get("state") == "completed"
        and int(value.get("total_environment_transitions", 0))
        >= int(config["new_environment_steps"])
    )


def _train(config_path: Path, config: Mapping[str, Any], allow_dirty: bool) -> Dict[str, Any]:
    run_dir = _run_dir(config)
    if _completed(config):
        return {"reused_completed_run": True, "command": None}
    command = [
        sys.executable,
        "-m",
        "marine_race_arena.learning.train_ppo_transition",
        "train",
        "--config",
        str(config_path),
    ]
    if any((run_dir / "checkpoints").glob("*.manifest.json")):
        command.append("--resume")
    if allow_dirty:
        command.append("--allow-dirty-smoke")
    subprocess.run(command, check=True)
    if not _completed(config):
        raise RuntimeError(f"comparison run did not reach its target: {run_dir}")
    return {"reused_completed_run": False, "command": command}


def _dedicated_benchmark(config: Mapping[str, Any]) -> Dict[str, Any]:
    run_dir = _run_dir(config)
    output = run_dir / "evaluations" / "dedicated_unseen_1000"
    report_path = output / "evaluation.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected = int(config["evaluation"]["dedicated_transition_cases"])
        if int(report.get("transition_cases", 0)) == expected:
            return report
    checkpoint = latest_valid_checkpoint(run_dir, safe_only=False)
    if checkpoint is None:
        raise RuntimeError(f"no valid checkpoint in {run_dir}")
    from stable_baselines3 import PPO

    model = PPO.load(str(checkpoint.model_path), device="cpu")
    report = evaluate_universal_transition_benchmark(
        model,
        output_dir=output,
        seed=int(config["evaluation"]["seed"]),
        difficulty="G6",
        transition_cases=int(config["evaluation"]["dedicated_transition_cases"]),
        full_cases_per_length=int(config["evaluation"]["full_cases_per_length"]),
        adapter=str(config["adapter"]),
        frames_per_sec=config["holoocean_frames_per_sec"],
        max_steps=int(config["max_episode_steps"]),
    )
    report["checkpoint"] = str(checkpoint.model_path)
    return report


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Selective warm-start versus scratch",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        "Selection order: safety, unseen transition success, long-sequence completion, action jerk, acquisition time.",
        "",
        "| Initialization | Safety clean | Safety events | Transition success | Long completion | Mean jerk | Mean acquisition (s) |",
        "|---|:---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("selective_warm_start", "scratch"):
        metrics = report["runs"][name]["benchmark"]["metrics"]
        safety_events = sum(int(metrics.get(key, 0) or 0) for key in (
            "collision_episodes", "out_of_bounds_episodes", "wrong_direction_events",
            "previous_gate_returns", "missed_gate_dnf", "acquisition_timeouts",
        ))
        lines.append(
            f"| {name} | {metrics['safety_clean']} | {safety_events} | "
            f"{float(metrics.get('universal_transition_success_rate') or 0.0):.4f} | "
            f"{float(metrics.get('long_sequence_completion_score') or 0.0):.4f} | "
            f"{metrics.get('mean_action_jerk')} | {metrics.get('mean_acquisition_time_s')} |"
        )
    lines += ["", f"Selected initialization: **{report['selected_initialization']}**.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-config", required=True)
    parser.add_argument("--scratch-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    warm_path = Path(args.warm_config)
    scratch_path = Path(args.scratch_config)
    warm = _load(warm_path)
    scratch = _load(scratch_path)
    validate_pair(warm, scratch)

    runs = {}
    for name, path, config in (
        ("selective_warm_start", warm_path, warm),
        ("scratch", scratch_path, scratch),
    ):
        training = _train(path, config, args.allow_dirty)
        benchmark = _dedicated_benchmark(config)
        runs[name] = {
            "run_dir": str(_run_dir(config)),
            "training": training,
            "benchmark": benchmark,
            "rank_key": list(transition_checkpoint_rank_key(benchmark["metrics"])),
        }
    selected = max(
        runs,
        key=lambda name: transition_checkpoint_rank_key(runs[name]["benchmark"]["metrics"]),
    )
    report = {
        "schema_version": "universal_transition_ab_comparison_v1",
        "generated_utc": now_utc(),
        "controlled_fields": _comparison_contract(warm),
        "selection_priority": [
            "safety", "universal_transition_success", "long_sequence_completion",
            "action_jerk", "acquisition_time",
        ],
        "runs": runs,
        "selected_initialization": selected,
    }
    output = Path(args.output_dir)
    _atomic_write(output / "ab_comparison.json", json.dumps(report, indent=2))
    _atomic_write(output / "ab_comparison.md", _markdown(report))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
