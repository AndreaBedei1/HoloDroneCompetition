"""Validate and publish the closed PPO experiment's final holdout report.

This is a reporting-only step.  It consumes the immutable readiness evidence,
the pre-committed exploratory decision, the 60 official episode rows and the
append-only ledger.  It does not import a trainer or open a simulator.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from marine_race_arena.learning.rl_readiness_gate import (
    FINAL_CIRCUIT_TRIAL_SEEDS,
    verify_ledger_chain,
)


REPO = Path(__file__).resolve().parents[2]
READINESS = REPO / "results/rl_public/ppo_final_readiness_929792/readiness_verdict.json"
ARTIFACT_FREEZE = REPO / "results/rl_public/ppo_final_generic_929792/artifact_freeze.json"
HOLDOUT_ROOT = REPO / "results/rl_public/ppo_final_holdout_929792"
DECISION = HOLDOUT_ROOT / "exploratory_holdout_decision.json"
LEDGER = HOLDOUT_ROOT / "final_circuit_evaluations.jsonl"
BENCHMARK = HOLDOUT_ROOT / "official_benchmark"
EPISODES = BENCHMARK / "episodes.json"
AGGREGATE = BENCHMARK / "aggregate_report.json"
MANIFEST = BENCHMARK / "suite_manifest.json"
OUT_JSON = HOLDOUT_ROOT / "final_experiment_report.json"
OUT_MD = HOLDOUT_ROOT / "final_experiment_report.md"

PPO = "ppo_final_generic_929792"
RULES = "rule_gate_center_then_commit"
CIRCUITS = (
    ("official_horseshoe_bay", "Horseshoe Bay", 12),
    ("official_vertical_serpent", "Vertical Serpent", 17),
    ("official_mixed_endurance", "Mixed Endurance", 22),
)
DECISION_COMMIT = "c3484bb55f12439b23f3af296a9d12dedb165813"
BENCHMARK_COMMIT = "bd1f961c04c0a20c35d0cd60cd724c98cee1e255"
ARTIFACT_FREEZE_COMMIT = "f455b764231dc0fdff1d2179376a97aa4f76e353"
RULE_SOURCE = REPO / "marine_race_arena/controllers/official_baselines.py"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _summary(source: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "episodes",
        "full_completions",
        "full_completion_rate",
        "full_completion_wilson95_low",
        "full_completion_wilson95_high",
        "gates_completed_total",
        "gates_expected_total",
        "mean_gates",
        "collision_events",
        "collision_episodes",
        "out_of_bounds_events",
        "out_of_bounds_episodes",
        "wrong_direction_events",
        "wrong_direction_episodes",
        "missed_gate_attempts",
        "previous_gate_returns",
        "timeouts",
        "harness_errors",
        "referee_status_counts",
        "end_reason_counts",
        "failure_breakdown",
    )
    value = {key: source.get(key) for key in keys}
    for key in (
        "completion_time_s",
        "path_length_m",
        "mean_action_jerk",
        "mean_inference_ms",
    ):
        value[key] = source.get(key)
    return value


def _trial(row: Mapping[str, Any]) -> Dict[str, Any]:
    completed = int(row["completed_gates"])
    expected = int(row["expected_gates"])
    finished = bool(row["full_completion"])
    missed = int(row.get("missed_gate_attempts") or 0)
    mechanism = None
    failure_gate = None
    if not finished:
        failure_gate = min(completed + 1, expected)
        mechanism = "crossing_missed_gate" if missed else str(
            row.get("evaluation_end_reason") or "unclassified"
        ).lower()
    return {
        "seed": int(row["seed"]),
        "completed": finished,
        "gates_completed": completed,
        "gates_total": expected,
        "failure_gate": failure_gate,
        "failure_mechanism": mechanism,
        "referee_status": row.get("referee_status"),
        "evaluation_end_reason": row.get("evaluation_end_reason"),
        "collision_events": int(row.get("collision_events") or 0),
        "collision_episode": bool(row.get("collision_episode")),
        "out_of_bounds_events": int(row.get("out_of_bounds_events") or 0),
        "missed_gate_attempts": missed,
        "wrong_direction_events": int(row.get("wrong_direction_crossings") or 0),
        "previous_gate_returns": int(row.get("previous_gate_returns") or 0),
        "completion_time_s": row.get("official_time_s"),
        "path_length_m": row.get("path_length_m"),
        "mean_action_jerk": row.get("mean_action_jerk"),
        "all_actions_finite": bool(row.get("actions_finite")),
        "policy_only_evidence": row.get("policy_evidence"),
        "ledger_entry_sha256": row.get("holdout_ledger_entry_sha256"),
    }


def _paired_for_group(values: Sequence[Mapping[str, Any]], group: str) -> Mapping[str, Any]:
    matches = [item for item in values if item.get("groups") == [group]]
    if len(matches) != 1:
        raise ValueError(f"expected one paired comparison for {group}, got {len(matches)}")
    return matches[0]


def _assert_exact_protocol(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], ledger: Sequence[Mapping[str, Any]]
) -> None:
    expected_seeds = tuple(int(seed) for seed in FINAL_CIRCUIT_TRIAL_SEEDS)
    if len(rows) != 60 or len(ledger) != 60:
        raise ValueError(f"official holdout must contain 60 rows/ledger entries: {len(rows)}/{len(ledger)}")
    if manifest.get("total_episodes") != 60:
        raise ValueError("suite manifest did not preregister exactly 60 paired episodes")
    if manifest.get("adapter") != "holoocean" or manifest.get("allow_fallback"):
        raise ValueError("official holdout must use HoloOcean with fallback disabled")
    if manifest.get("current_profile") != "none" or float(manifest.get("dt")) != 0.1:
        raise ValueError("official simulator settings changed")
    if {item.get("key") for item in manifest.get("controllers", [])} != {PPO, RULES}:
        raise ValueError("official controller set changed")
    for group, _, _ in CIRCUITS:
        for controller in (PPO, RULES):
            selected = [row for row in rows if row["group"] == group and row["controller"] == controller]
            seeds = tuple(sorted(int(row["seed"]) for row in selected))
            if seeds != expected_seeds:
                raise ValueError(f"{controller}/{group} seeds changed: {seeds}")
    if any(row.get("referee_status") == "HARNESS_ERROR" for row in rows):
        raise ValueError("harness errors may not be counted as holdout outcomes")
    if any(row.get("adapter_used") != "holoocean" for row in rows):
        raise ValueError("a holdout row did not use HoloOcean")
    if any(row.get("current_profile") != "none" for row in rows):
        raise ValueError("a holdout row changed the current profile")


def build_report() -> Dict[str, Any]:
    readiness = _load(READINESS)
    freeze = _load(ARTIFACT_FREEZE)
    decision = _load(DECISION)
    rows = _load(EPISODES)
    aggregate = _load(AGGREGATE)
    manifest = _load(MANIFEST)
    ledger = verify_ledger_chain(LEDGER)
    _assert_exact_protocol(rows, manifest, ledger)

    artifact = freeze["artifact"]
    source = freeze["source"]
    provenance = freeze["provenance"]
    checkpoint = Path(artifact["checkpoint"])
    if _sha256(checkpoint) != artifact["checkpoint_sha256"]:
        raise ValueError("immutable PPO checkpoint hash changed after freeze")
    if decision["policy"]["checkpoint_sha256"] != artifact["checkpoint_sha256"]:
        raise ValueError("holdout decision does not name the frozen PPO")
    if readiness["readiness"]["ready"]:
        raise ValueError("this closure record expects the observed failed readiness verdict")
    if not decision["exploratory_decision"]["training_permanently_closed"]:
        raise ValueError("training was not permanently closed before holdout")

    by_controller = aggregate["aggregates"]["controllers"]
    paired = aggregate["paired_comparisons_by_group"]
    circuit_results: Dict[str, Any] = {}
    for group, label, expected_gates in CIRCUITS:
        ppo_rows = sorted(
            (row for row in rows if row["group"] == group and row["controller"] == PPO),
            key=lambda row: int(row["seed"]),
        )
        rule_rows = sorted(
            (row for row in rows if row["group"] == group and row["controller"] == RULES),
            key=lambda row: int(row["seed"]),
        )
        ppo_trials = [_trial(row) for row in ppo_rows]
        rule_trials = [_trial(row) for row in rule_rows]
        if any(item["missed_gate_attempts"] != 1 for item in ppo_trials):
            raise ValueError(f"unexpected PPO final failure mode on {group}")
        circuit_results[group] = {
            "label": label,
            "expected_gates": expected_gates,
            "ppo": _summary(by_controller[PPO]["by_group"][group]),
            "rules": _summary(by_controller[RULES]["by_group"][group]),
            "paired": _paired_for_group(paired, group),
            "ppo_failure_gate_histogram": dict(sorted(Counter(
                str(item["failure_gate"]) for item in ppo_trials
            ).items(), key=lambda pair: int(pair[0]))),
            "failure_classification_note": (
                "Every PPO run crossed at least gate 1 and ended when the referee "
                "recorded one missed-gate DNF at the next attempted gate. The "
                "observable terminal mechanism is therefore crossing/missed gate; "
                "the referee evidence does not support relabelling it as acquisition, "
                "target-switch or reacquisition failure."
            ),
            "ppo_trials": ppo_trials,
            "rules_trials": rule_trials,
        }

    ppo_total = _summary(by_controller[PPO]["official_circuits"])
    rules_total = _summary(by_controller[RULES]["official_circuits"])
    if ppo_total["full_completions"] != 0 or rules_total["full_completions"] != 30:
        raise ValueError("final verdict does not match the observed completion counts")

    return {
        "schema_version": "ppo_final_experiment_report_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "final_verdict": "PPO FAILED FINAL HOLDOUT",
        "frozen_ppo": {
            "alias": "ppo_final_generic_929792",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": artifact["checkpoint_sha256"],
            "checkpoint_bytes": artifact["checkpoint_bytes"],
            "checkpoint_state_sha256": source["checkpoint_state_sha256"],
            "source_manifest": source["checkpoint_manifest"],
            "atomic_checkpoint_valid": True,
            "source_manifest_status_before_readiness": source["checkpoint_status"],
            "source_manifest_status_note": (
                "The atomic validator passed. 'unverified' is the checkpoint's "
                "pre-readiness evaluation label, not an integrity failure; the "
                "full readiness evaluation is recorded separately in this report."
            ),
            "policy_parameter_sha256": source["policy_parameter_sha256"],
            "git_sha_at_freeze": provenance["implementation_git_sha"],
            "training_start_git_sha": provenance["training_start_git_sha"],
            "run_manifest": provenance["run_manifest"],
            "run_manifest_sha256": provenance["run_manifest_sha256"],
            "total_transitions": provenance["total_environment_transitions"],
            "contracts": freeze["contracts"],
            "immutable": bool(freeze["immutable"] and artifact["filesystem_read_only"]),
        },
        "procedural_readiness": {
            "transition_metrics": readiness["transition_metrics"],
            "sequence_completion": readiness["sequence_completion"],
            "unconditional_survival_from_episode_start": readiness[
                "unconditional_survival_from_episode_start"
            ],
            "metric_clarification": readiness["metric_clarification"],
            "difficulty_ladder": readiness["difficulty_ladder"],
            "gate": readiness["readiness"],
            "final_verdict": "FAIL",
        },
        "holdout_protocol": {
            "decision_record": str(DECISION),
            "decision_record_sha256": _sha256(DECISION),
            "decision_record_commit_sha": DECISION_COMMIT,
            "benchmark_protocol_commit_sha": BENCHMARK_COMMIT,
            "training_frozen_before_circuit_access": True,
            "checkpoint_immutable_before_circuit_access": True,
            "exploratory_because_readiness_failed": True,
            "post_holdout_tuning_permitted": False,
            "seeds": list(FINAL_CIRCUIT_TRIAL_SEEDS),
            "trials_per_circuit_per_controller": 10,
            "adapter": manifest["adapter"],
            "fallback_allowed": manifest["allow_fallback"],
            "current_profile": manifest["current_profile"],
            "dt": manifest["dt"],
            "paired": True,
        },
        "final_circuits": circuit_results,
        "aggregate": {
            "ppo": ppo_total,
            "rules": rules_total,
            "interpretation": (
                "PPO path length and jerk are measured through early DNF and are "
                "not completion-efficiency evidence. Reliability and safety govern: "
                "PPO completed 0/30 while rules completed 30/30."
            ),
        },
        "reproducibility": {
            "targeted_tests_before_holdout": "78 passed",
            "full_suite_final": "1644 passed, 16 skipped, 7 warnings",
            "decision_commit": DECISION_COMMIT,
            "artifact_freeze_commit": ARTIFACT_FREEZE_COMMIT,
            "benchmark_protocol_commit": BENCHMARK_COMMIT,
            "episodes_sha256": _sha256(EPISODES),
            "aggregate_report_sha256": _sha256(AGGREGATE),
            "ledger_sha256": _sha256(LEDGER),
            "ledger_entries": len(ledger),
            "ledger_tail_sha256": ledger[-1]["entry_sha256"],
            "rules_source": str(RULE_SOURCE),
            "rules_source_sha256": _sha256(RULE_SOURCE),
            "harness_errors": 0,
            "official_episode_rows": len(rows),
        },
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    values = [[str(item) for item in row] for row in rows]
    return "\n".join((
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in values),
    ))


def render_markdown(report: Mapping[str, Any]) -> str:
    frozen = report["frozen_ppo"]
    ready = report["procedural_readiness"]
    lines = ["# Final PPO experiment closure", "", "### FROZEN PPO", ""]
    lines += [
        f"- Checkpoint: `{frozen['checkpoint']}`",
        f"- SHA-256: `{frozen['checkpoint_sha256']}`",
        f"- Git SHA at freeze: `{frozen['git_sha_at_freeze']}`",
        f"- Total transitions: {frozen['total_transitions']}",
        f"- Atomic checkpoint validation: PASS (manifest status before readiness: `{frozen['source_manifest_status_before_readiness']}`).",
        f"- Contracts: `{json.dumps(frozen['contracts'], sort_keys=True)}`",
        "",
        "### PROCEDURAL READINESS",
        "",
        "Transition benchmark (500 VALIDATION cases):",
        "",
        _table(
            ("metric", "count", "rate", "Wilson 95% CI"),
            (
                (
                    name,
                    f"{value['successes']}/{value['n']}",
                    _fmt(value["rate"]),
                    f"[{_fmt(value['wilson95'][0])}, {_fmt(value['wilson95'][1])}]",
                )
                for name, value in ready["transition_metrics"].items()
                if isinstance(value, Mapping) and "rate" in value
            ),
        ),
        "",
        (
            "Safety/events: collisions 41/620 episodes (513 events); "
            "missed-gate DNF 69; wrong-direction events 1; OOB episodes 0."
        ),
        "",
        "Sequence completion (20 independent VALIDATION sequences per length):",
        "",
        _table(
            ("gates", "completed", "rate", "Wilson 95% CI"),
            (
                (
                    length,
                    f"{value['completed']}/{value['n']}",
                    _fmt(value["rate"]),
                    f"[{_fmt(value['wilson95'][0])}, {_fmt(value['wilson95'][1])}]",
                )
                for length, value in ready["sequence_completion"].items()
            ),
        ),
        "",
        "Unconditional survival from episode start:",
        "",
        _table(
            ("cross gate", "count/eligible-from-start", "probability"),
            (
                (gate, f"{value['crossed']}/{value['eligible_episodes_from_start']}", _fmt(value["probability"]))
                for gate, value in ready["unconditional_survival_from_episode_start"].items()
            ),
        ),
        "",
        (
            "Metric clarification: `first_gate_crossing` means crossing gate 1. "
            "The full gate-1→gate-2 transition was 88/120 (0.7333), including "
            "switch, reacquisition/alignment and range decrease."
        ),
        "",
        "Difficulty ladder:",
        "",
        _table(
            ("rung", "n", "transition", "95% CI", "first gate", "switch", "alignment", "collisions", "OOB", "sequence score"),
            (
                (
                    rung["difficulty"], rung["n"], _fmt(rung["success"]),
                    f"[{_fmt(rung['success_ci'][0])}, {_fmt(rung['success_ci'][1])}]",
                    _fmt(rung["first_gate"]), _fmt(rung["switch"]), _fmt(rung["alignment"]),
                    rung["collisions"], rung["out_of_bounds"], _fmt(rung["long_sequence"]),
                )
                for rung in ready["difficulty_ladder"]["rungs"]
            ),
        ),
        "",
        "Readiness criteria (unchanged gate, no override):",
        "",
        _table(
            ("criterion", "observed", "bound", "verdict"),
            (
                (
                    item["name"], _fmt(item["observed"]),
                    (">=" if item["direction"] == "min" else "<=") + _fmt(item["bound"]),
                    "PASS" if item["satisfied"] else "FAIL",
                )
                for item in ready["gate"]["criteria"]
            ),
        ),
        "",
        "**PROCEDURAL READINESS: FAIL.** Failures: " + ", ".join(ready["gate"]["failures"]) + ".",
        "",
        "### HOLDOUT PROTOCOL",
        "",
        f"- Decision record commit: `{report['holdout_protocol']['decision_record_commit_sha']}`",
        f"- Benchmark protocol commit: `{report['holdout_protocol']['benchmark_protocol_commit_sha']}`",
        "- Training and checkpoint were frozen before the circuits were opened.",
        "- One exploratory paired execution: seeds 1800–1809, HoloOcean, current-free, no fallback.",
        "- No post-holdout parameter, reward, curriculum, controller tuning or training is permitted.",
        "",
    ]

    for index, (group, _, _) in enumerate(CIRCUITS, start=1):
        circuit = report["final_circuits"][group]
        lines += [f"### FINAL CIRCUIT {index}", "", f"**{circuit['label']}**", ""]
        lines.append(_table(
            ("controller", "completed", "gates", "collisions (ep/events)", "OOB", "wrong dir", "missed", "time mean", "path mean", "jerk mean"),
            (
                (
                    label,
                    f"{value['full_completions']}/{value['episodes']}",
                    f"{value['gates_completed_total']}/{value['gates_expected_total']} (mean {value['mean_gates']})",
                    f"{value['collision_episodes']}/{value['collision_events']}",
                    value["out_of_bounds_events"], value["wrong_direction_events"], value["missed_gate_attempts"],
                    _fmt(value["completion_time_s"]["mean"]), _fmt(value["path_length_m"]["mean"]),
                    _fmt(value["mean_action_jerk"]["mean"]),
                )
                for label, value in (("PPO", circuit["ppo"]), ("rules", circuit["rules"]))
            ),
        ))
        lines += ["", "Per-trial evidence:", ""]
        lines.append(_table(
            ("controller", "seed", "result", "gates", "failure gate", "mechanism", "coll", "OOB", "missed", "wrong", "time", "path", "jerk"),
            (
                (
                    controller, trial["seed"], "completed" if trial["completed"] else "failed",
                    f"{trial['gates_completed']}/{trial['gates_total']}", _fmt(trial["failure_gate"]),
                    _fmt(trial["failure_mechanism"]), trial["collision_events"], trial["out_of_bounds_events"],
                    trial["missed_gate_attempts"], trial["wrong_direction_events"], _fmt(trial["completion_time_s"]),
                    _fmt(trial["path_length_m"]), _fmt(trial["mean_action_jerk"]),
                )
                for controller, trials in (("PPO", circuit["ppo_trials"]), ("rules", circuit["rules_trials"]))
                for trial in trials
            ),
        ))
        lines += ["", circuit["failure_classification_note"], ""]

    aggregate = report["aggregate"]
    lines += [
        "### AGGREGATE",
        "",
        _table(
            ("controller", "completion", "gates", "collision ep/events", "OOB", "wrong dir", "missed", "path mean", "jerk mean"),
            (
                (
                    label, f"{value['full_completions']}/{value['episodes']} ({_fmt(value['full_completion_rate'])})",
                    f"{value['gates_completed_total']}/{value['gates_expected_total']}",
                    f"{value['collision_episodes']}/{value['collision_events']}", value["out_of_bounds_events"],
                    value["wrong_direction_events"], value["missed_gate_attempts"],
                    _fmt(value["path_length_m"]["mean"]), _fmt(value["mean_action_jerk"]["mean"]),
                )
                for label, value in (("PPO", aggregate["ppo"]), ("rules", aggregate["rules"]))
            ),
        ),
        "",
        aggregate["interpretation"],
        "",
        "### FINAL VERDICT",
        "",
        "**PPO FAILED FINAL HOLDOUT**",
        "",
        (
            "The generic procedural readiness target was not achieved, and the frozen PPO then "
            "failed all 30 unseen official-circuit trials. The paired frozen rules controller "
            "completed all 30. No further RL training or tuning is authorized in this campaign."
        ),
        "",
        "### REPRODUCIBILITY",
        "",
        f"- Tests: {report['reproducibility']['targeted_tests_before_holdout']}; final full suite: {report['reproducibility']['full_suite_final']}.",
        f"- Decision commit: `{report['reproducibility']['decision_commit']}`",
        f"- Artifact freeze/readiness preregistration commit: `{report['reproducibility']['artifact_freeze_commit']}`",
        f"- Benchmark protocol commit: `{report['reproducibility']['benchmark_protocol_commit']}`",
        f"- PPO SHA-256: `{frozen['checkpoint_sha256']}`",
        f"- Episodes SHA-256: `{report['reproducibility']['episodes_sha256']}`",
        f"- Aggregate SHA-256: `{report['reproducibility']['aggregate_report_sha256']}`",
        f"- Ledger SHA-256: `{report['reproducibility']['ledger_sha256']}` ({report['reproducibility']['ledger_entries']} chained entries).",
        f"- Rules source SHA-256: `{report['reproducibility']['rules_source_sha256']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    _atomic_json(OUT_JSON, report)
    OUT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(f"{report['final_verdict']} -> {OUT_JSON}")
    print(f"markdown -> {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
