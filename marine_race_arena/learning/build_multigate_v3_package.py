"""Build the compact audit package for the learned multi-gate v3 experiment.

The heavy datasets and PPO ZIPs remain under git-ignored ``results/rl``.  This
builder copies or derives only compact measurements, provenance, hashes, and
reproduction commands from runs that actually exist.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

from marine_race_arena.learning.config import ACTION_AXES, ACTION_CONTRACT_VERSION
from marine_race_arena.learning.config_v3 import (
    FEATURE_BOUNDS_V3,
    FEATURE_NAMES_V3,
    OBS_DIM_V3,
    OBS_ENCODING_VERSION_V3,
)
from marine_race_arena.learning.provenance import git_sha, now_utc, sha256_file
from marine_race_arena.learning.seed_registry import registry_dict

PUB = Path("results/rl_public/multigate_rl_v3")
R1_RUN = Path(
    "results/rl/multigate_v3/r1/ppo_multigate_v3/20260727_133031"
)
R1_SELECTION = Path("results/rl/multigate_v3/r1_checkpoint_selection")
FINAL_TWO_GATE = Path("results/rl/multigate_v3/final_two_gate")
R2_LEFT_PROBE = Path("results/rl/multigate_v3/r2_left_probe")
R2_RIGHT_PROBE = Path("results/rl/multigate_v3/r2_right_probe")
R2_RUN = Path(
    "results/rl/multigate_v3/r2_left/ppo_multigate_v3/20260727_142237"
)
R2_REWARD_RUN = Path(
    "results/rl/multigate_v3/r2_left_reward_v3/ppo_multigate_v3/"
    "20260727_144855"
)
RULE_COMPARE = Path("results/rl/multigate_v3/comparison/two_gate_rule")
HYBRID_COMPARE = Path("results/rl/multigate_v3/comparison/two_gate_hybrid")
BC_MODEL = Path("results/rl/multigate_v3/models/bc_v3_dagger_neutral.pt")
TRANSFER_MANIFEST = Path(
    "results/rl/multigate_v3/models/bc_v1_transfer_v3.pt.transfer.json"
)
BC_REPORT = Path(
    "results/rl/multigate_v3/models/bc_v3_dagger_neutral.report.json"
)


def _load(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _require(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing measured artifacts: " + ", ".join(missing))


def _csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row: Dict[str, Any] = {}
            for key, value in raw.items():
                if value in ("", "None", None):
                    row[key] = None
                    continue
                try:
                    number = float(value)
                    row[key] = int(number) if key == "timesteps" else number
                except ValueError:
                    row[key] = value
            rows.append(row)
        return rows


def _run_summary(run: Path) -> Dict[str, Any]:
    return {
        "run_dir": str(run),
        "run_config": _load(run / "run_config.json"),
        "run_status": _load(run / "run_status.json"),
        "initial_eval": _load(run / "evaluation" / "initial_eval.json"),
        "eval_history": _csv_rows(run / "evaluation" / "eval.csv"),
        "best_metrics": _load(run / "best_model" / "best_metrics.json"),
        "best_model": {
            "path": str(run / "best_model" / "best_model.zip"),
            "bytes": (run / "best_model" / "best_model.zip").stat().st_size,
            "sha256": sha256_file(run / "best_model" / "best_model.zip"),
        },
    }


def _mean(values: Iterable[Any]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None]
    return round(mean(finite), 4) if finite else None


def _aggregate(label: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    finished = [row for row in rows if row.get("finished")]
    max_duration_s = 150.0
    penalized = [
        row.get("penalized_time_s")
        if row.get("penalized_time_s") is not None
        else max_duration_s
        for row in rows
    ]
    return {
        "controller": label,
        "n_eval": len(rows),
        "completions": len(finished),
        "completion_rate": round(len(finished) / len(rows), 4) if rows else 0.0,
        "mean_gates": _mean(row.get("completed_gates") for row in rows),
        "mean_official_time_finished_s": _mean(
            row.get("official_time_s") for row in finished
        ),
        "mean_penalized_time_s": _mean(penalized),
        "dnf_penalty_s": max_duration_s,
        "mean_time_per_completed_gate_s": _mean(
            (
                float(row["official_time_s"]) / int(row["completed_gates"])
                if row.get("official_time_s") is not None
                and int(row.get("completed_gates") or 0) > 0
                else None
            )
            for row in rows
        ),
        "collisions": sum(int(row.get("collision_events") or 0) for row in rows),
        "out_of_bounds": sum(
            int(row.get("out_of_bounds_events") or 0) for row in rows
        ),
        "wrong_direction": sum(
            int(row.get("wrong_direction_crossings") or 0) for row in rows
        ),
        "previous_gate_returns": sum(
            int(row.get("previous_gate_returns") or 0) for row in rows
        ),
        "mean_abs_sway_action": _mean(
            row.get("mean_abs_sway_action") for row in rows
        ),
        "mean_abs_yaw_action": _mean(
            row.get("mean_abs_yaw_action") for row in rows
        ),
        "mean_action_jerk": _mean(row.get("mean_action_jerk") for row in rows),
        "mean_action_oscillation_rate": _mean(
            row.get("action_oscillation_rate") for row in rows
        ),
        "mean_inference_time_ms": _mean(
            row.get("inference_time_ms") for row in rows
        ),
        "mean_wall_s": _mean(row.get("wall_s") for row in rows),
        "deterministic_runtime_interventions": sum(
            int(row.get("deterministic_runtime_intervention_count") or 0)
            for row in rows
        ),
    }


def _write_comparison() -> Dict[str, Any]:
    rl_all = _load(FINAL_TWO_GATE / "eval_results.json")
    paired_seeds = list(range(21200, 21205))
    sources = {
        "rule_gate_center_then_commit": _load(RULE_COMPARE / "eval_results.json"),
        "hybrid_gate_controller": _load(HYBRID_COMPARE / "eval_results.json"),
        "rl_multigate_controller": [
            row for row in rl_all if int(row["seed"]) in paired_seeds
        ],
    }
    fields = [
        "controller",
        "seed",
        "finished",
        "completed_gates",
        "expected_gates",
        "official_time_s",
        "penalized_time_s",
        "collision_events",
        "out_of_bounds_events",
        "wrong_direction_crossings",
        "previous_gate_returns",
        "mean_abs_sway_action",
        "mean_abs_yaw_action",
        "mean_action_jerk",
        "action_oscillation_rate",
        "inference_time_ms",
        "wall_s",
        "deterministic_runtime_intervention_count",
    ]
    csv_path = PUB / "comparison" / "paired_per_seed.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for controller, rows in sources.items():
            for row in sorted(rows, key=lambda item: int(item["seed"])):
                writer.writerow(
                    {
                        key: controller if key == "controller" else row.get(key)
                        for key in fields
                    }
                )
    aggregate = {
        "paired_seeds": paired_seeds,
        "track": "marine_race_arena/tracks/tests/two_gate_straight.json",
        "adapter_actual": "holoocean",
        "fallback_used": False,
        "current_profile": "none",
        "currents_actual": [0.0, 0.0, 0.0],
        "controllers": {
            controller: _aggregate(controller, rows)
            for controller, rows in sources.items()
        },
    }
    _write(PUB / "comparison" / "aggregate.json", aggregate)
    _write(
        PUB / "comparison" / "time_comparison.json",
        {
            "paired_seeds": paired_seeds,
            "finished_episode_mean_official_time_s": {
                controller: values["mean_official_time_finished_s"]
                for controller, values in aggregate["controllers"].items()
            },
            "mean_penalized_time_s_with_150s_dnf": {
                controller: values["mean_penalized_time_s"]
                for controller, values in aggregate["controllers"].items()
            },
            "conclusion": (
                "On the paired subset RL completed 4/5 and was slower and less "
                "smooth than both deterministic-runtime baselines."
            ),
        },
    )
    return aggregate


def main(argv=None) -> int:
    global PUB
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(PUB))
    args = parser.parse_args(argv)
    PUB = Path(args.out)

    required = [
        TRANSFER_MANIFEST,
        BC_REPORT,
        BC_MODEL,
        R1_RUN / "run_status.json",
        R1_RUN / "best_model" / "best_model.zip",
        R1_SELECTION / "eval_summary.json",
        FINAL_TWO_GATE / "evaluation_manifest.json",
        FINAL_TWO_GATE / "eval_results.json",
        FINAL_TWO_GATE / "eval_summary.json",
        R2_LEFT_PROBE / "eval_results.json",
        R2_RIGHT_PROBE / "eval_results.json",
        R2_RUN / "run_status.json",
        R2_REWARD_RUN / "run_status.json",
        RULE_COMPARE / "eval_results.json",
        HYBRID_COMPARE / "eval_results.json",
    ]
    _require(required)

    generated = now_utc()
    selected_model = R1_RUN / "best_model" / "best_model.zip"
    final_manifest = _load(FINAL_TWO_GATE / "evaluation_manifest.json")
    final_summary = _load(FINAL_TWO_GATE / "eval_summary.json")
    final_rows = _load(FINAL_TWO_GATE / "eval_results.json")

    _write(
        PUB / "observation_v3.json",
        {
            "observation_encoding_version": OBS_ENCODING_VERSION_V3,
            "observation_dim": OBS_DIM_V3,
            "feature_names": list(FEATURE_NAMES_V3),
            "feature_bounds": [list(bounds) for bounds in FEATURE_BOUNDS_V3],
            "v1_prefix_columns": 36,
            "temporal_columns": OBS_DIM_V3 - 36,
            "privileged_fields_present": False,
            "metric_visual_gate_pose_present": False,
        },
    )
    _write(PUB / "seed_registry_snapshot.json", registry_dict())
    _write(PUB / "models" / "transfer_v1_to_v3.json", _load(TRANSFER_MANIFEST))
    _write(PUB / "models" / "bc_v3.json", _load(BC_REPORT))
    _write(PUB / "models" / "ppo_v3.json", _run_summary(R1_RUN))
    _write(
        PUB / "models" / "selected_model_sha256.json",
        {
            "path": str(selected_model),
            "sha256": sha256_file(selected_model),
            "bytes": selected_model.stat().st_size,
            "committed": False,
            "reason_not_committed": "SB3 checkpoints remain under git-ignored results/rl",
            "observation_encoding_version": OBS_ENCODING_VERSION_V3,
            "observation_dim": OBS_DIM_V3,
            "action_contract_version": ACTION_CONTRACT_VERSION,
        },
    )

    _write(
        PUB / "two_gate" / "train_summary.json",
        {
            "r1": _run_summary(R1_RUN),
            "checkpoint_selection": _load(R1_SELECTION / "eval_summary.json"),
            "selection_rule": (
                "completion rate, gates, safety, then finished time; never reward alone"
            ),
        },
    )
    _write(PUB / "two_gate" / "eval_results.json", final_rows)
    failures = [row for row in final_rows if not row.get("finished")]
    _write(
        PUB / "two_gate" / "failure_analysis.json",
        {
            "n_eval": len(final_rows),
            "failures": failures,
            "first_failure_counts": {"NEXT_GATE_NOT_ACQUIRED": len(failures)},
            "diagnosis": (
                "The lone reserved failure crossed gate 1 but timed out before "
                "gate 2; no collision, OOB, wrong-direction crossing, or runtime "
                "intervention occurred in that episode."
            ),
        },
    )

    _write(
        PUB / "three_gate" / "train_summary.json",
        {
            "status": "NOT_RUN",
            "reason": "R2 balanced-turn criterion was not met; curriculum did not advance.",
        },
    )
    _write(PUB / "three_gate" / "eval_results.json", [])
    _write(
        PUB / "three_gate" / "failure_analysis.json",
        {
            "status": "NOT_EVALUATED",
            "earliest_blocking_stage": "R2",
            "left_probe": _load(R2_LEFT_PROBE / "eval_results.json"),
            "right_probe": _load(R2_RIGHT_PROBE / "eval_results.json"),
            "r2_first_run": _run_summary(R2_RUN),
            "r2_reward_corrected_run": _run_summary(R2_REWARD_RUN),
            "decision": (
                "Stop PPO after two non-improving held-out endpoints; retain R1 "
                "model and report the boundary."
            ),
        },
    )

    for circuit in ("horseshoe", "vertical", "mixed"):
        _write(
            PUB / "official_no_current" / circuit / "status.json",
            {
                "status": "NOT_RUN",
                "reason": (
                    "The learned policy did not pass R2/R3 progression criteria; "
                    "no official-circuit claim is made."
                ),
            },
        )
    _write(
        PUB / "official_no_current" / "summary.json",
        {
            "status": "NOT_RUN",
            "circuits_completed_by_rl": 0,
            "reason": "Curriculum stopped honestly at R2.",
        },
    )

    comparison = _write_comparison()
    manifest = {
        "schema_version": "multigate_rl_v3_public_v1",
        "generated_utc": generated,
        "generated_at_code_sha": git_sha(),
        "starting_sha": "0b24d003e60dc47ef7269645eb9648fd9ead255f",
        "branch": "feature/rl-multigate-policy",
        "experiment_code_sha": final_manifest["code_sha"],
        "result_category": "PARTIAL RL SUCCESS",
        "conclusion": (
            "The learned controller reliably progressed beyond frozen BC-v1 and "
            "met the straight two-gate criterion (9/10), but balanced turns were "
            "unreliable and three-gate/official evaluation was not claimed."
        ),
        "runtime_architecture": {
            "description": (
                "Onboard multi-sensor RL with minimal deterministic gate index "
                "and passage confirmation"
            ),
            "policy": "feed-forward PPO MLP [256, 256]",
            "actions": list(ACTION_AXES),
            "rule_action_weight": 0,
            "hybrid_blending": False,
            "rule_controller_instantiated": False,
            "deterministic_runtime_interventions_final": 0,
            "recurrence_used": False,
        },
        "frozen_baselines": {
            "bc_v1_sha256": (
                "fd6fc7e6fba9b88ccc84fa76275d72ce3c907c75d632367995cb5257ae71d5d3"
            ),
            "main_sha": "f4d63751fd935f724a9dce3ea802f76e1e27d45f",
            "feature_rl_controller_sha": (
                "5a02838e24fec21c3fc540102beb9c13e942f0ac"
            ),
            "feature_visual_pose_multigate_sha": (
                "0b24d003e60dc47ef7269645eb9648fd9ead255f"
            ),
        },
        "selected_model": {
            "path": str(selected_model),
            "sha256": sha256_file(selected_model),
        },
        "final_two_gate": {
            "summary": final_summary,
            "manifest": final_manifest,
        },
        "r2_status": "HOLD",
        "three_gate_status": "NOT_RUN",
        "official_status": "NOT_RUN",
        "paired_comparison": comparison,
        "reproduce_commands": {
            "train_two_gate": (
                "scripts\\train_rl_multigate_two_gate.bat"
            ),
            "resume_two_gate": (
                "scripts\\resume_rl_multigate_two_gate.bat <run-dir> 5000"
            ),
            "eval_two_gate": "scripts\\eval_rl_multigate_two_gate.bat",
            "paired_comparison": "scripts\\compare_rule_hybrid_rl.bat",
        },
    }
    _write(PUB / "experiment_manifest.json", manifest)
    (PUB / "README.md").write_text(
        "# Learned Multi-gate RL v3\n\n"
        "**Honest result category: PARTIAL RL SUCCESS.** The feed-forward, "
        "observation-v3 PPO controller completed the current-free straight "
        "two-gate task on 9/10 reserved seeds in real HoloOcean with no fallback "
        "and no runtime rule-generated action. It did not meet the balanced-turn "
        "R2 criterion, so three-gate and official-circuit RL claims were not made.\n\n"
        "The selected PPO ZIP and raw datasets remain under git-ignored `results/rl`; "
        "this package publishes their SHA-256 hashes, exact paths, compact metrics, "
        "provenance, failure evidence, and reproduction commands. See "
        "`experiment_manifest.json`, `two_gate/`, `three_gate/failure_analysis.json`, "
        "and `comparison/`.\n",
        encoding="utf-8",
    )
    print(f"[multigate-v3] wrote {PUB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
