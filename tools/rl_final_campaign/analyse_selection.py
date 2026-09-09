"""Turn the matched benchmark into a decision: parent, CIs, failure families.

Reports intervals rather than point estimates throughout, because the campaign's
whole history is a warning about reading small samples literally: a policy whose
weights moved 0.006% scored anywhere from 0.51 to 0.87 across separate n=112
evaluations, and the run's own checkpoint aliases were assigned from that noise.

Survivor bias is handled explicitly.  P(reach gate k) is reported from episode
start, not conditioned on having reached k-1; the conditional form is reported
beside it under a different name so the two can never be confused.

Usage (marine_race_rl env, from the repository root)::

    python -m tools.rl_final_campaign.analyse_selection
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from marine_race_arena.learning.rl_failure_taxonomy import (
    assert_no_seed_memorisation,
    classify_many,
    failure_position_distribution,
    mine_failure_families,
    survival_curve,
)
from marine_race_arena.learning.rl_matched_benchmark import (
    align_outcomes,
    equivalent_within_noise,
    paired_difference,
    wilson_interval,
)

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "results/rl_public/ppo_final_matched_benchmark"


def _episodes(result: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    for path in result.get("report_paths", {}).values():
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.extend(raw.get("episodes") or [])
    return rows


def _counts(rows: List[Mapping[str, Any]], key: str, kind: str) -> tuple:
    selected = [row for row in rows if row.get("episode_type") == kind]
    hits = sum(1 for row in selected if row.get(key))
    return hits, len(selected)


def summarise_candidate(name: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    rows = _episodes(result)
    transition_hits, transition_n = _counts(
        rows, "universal_transition_success", "transition_focus"
    )
    sequence_hits, sequence_n = _counts(
        rows, "full_sequence_completion", "full_sequence"
    )
    by_length: Dict[str, Any] = {}
    for row in rows:
        if row.get("episode_type") != "full_sequence":
            continue
        key = str(row.get("gate_count"))
        entry = by_length.setdefault(key, {"n": 0, "completed": 0})
        entry["n"] += 1
        entry["completed"] += bool(row.get("full_sequence_completion"))
    for entry in by_length.values():
        low, high = wilson_interval(entry["completed"], entry["n"])
        entry["rate"] = entry["completed"] / entry["n"] if entry["n"] else None
        entry["ci"] = [low, high]

    survival = survival_curve(rows, episode_types=("full_sequence",))
    return {
        "candidate": name,
        "checkpoint": result.get("checkpoint"),
        "sha256": result.get("sha256"),
        "transition": {
            "successes": transition_hits, "n": transition_n,
            "rate": transition_hits / transition_n if transition_n else None,
            "ci": list(wilson_interval(transition_hits, transition_n)),
        },
        "sequence": {
            "completed": sequence_hits, "n": sequence_n,
            "rate": sequence_hits / sequence_n if sequence_n else None,
            "ci": list(wilson_interval(sequence_hits, sequence_n)),
            "by_length": dict(sorted(by_length.items(), key=lambda kv: int(kv[0]))),
        },
        "survival": survival,
        "failure_positions": failure_position_distribution(
            rows, episode_types=("full_sequence",)
        ),
        "safety": {
            key: result["metrics"].get(key) for key in (
                "collision_episodes", "out_of_bounds_episodes", "missed_gate_dnf",
                "wrong_direction_events", "previous_gate_returns",
            )
        },
        "activity": {
            key: result["metrics"].get(key) for key in (
                "mean_absolute_action", "nontrivial_action_fraction",
                "mean_action_jerk",
            )
        },
    }


def paired_table(results: Mapping[str, Any], reference: str) -> List[Dict[str, Any]]:
    """Every candidate against the selected parent, on the same cases."""

    rows: List[Dict[str, Any]] = []
    base = results[reference].get("case_outcomes") or {}
    for name, result in results.items():
        if name == reference:
            continue
        other = result.get("case_outcomes") or {}
        if not base or not other:
            continue
        a, b, _keys = align_outcomes(base, other)
        if not a:
            continue
        diff = paired_difference(a, b)
        rows.append({
            "candidate": name,
            "versus": reference,
            "paired_n": diff["n_pairs"],
            "mean_parent": diff["mean_a"],
            "mean_candidate": diff["mean_b"],
            "mean_difference": diff["mean_difference"],
            "ci": diff["difference_ci"],
            "n_discordant": diff["n_discordant"],
            "sign_test_p": diff["sign_test_p_value"],
            "equivalent_within_noise": bool(diff["equivalent_within_noise"]),
        })
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default=str(BENCH / "matched_benchmark.json"))
    parser.add_argument("--output", default=str(BENCH / "selection_analysis.json"))
    args = parser.parse_args(argv)

    path = Path(args.benchmark)
    if not path.is_file():
        raise SystemExit(
            f"{path} not found; the matched benchmark has not finished yet"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    results = report["results"]
    selection = report["selection"]
    selected = selection.get("selected")

    summaries = {
        name: summarise_candidate(name, result) for name, result in results.items()
    }
    paired = paired_table(results, selected) if selected else []

    # Families are mined from the SELECTED parent only: the continuation trains
    # from it, so its weaknesses are the ones worth targeting.
    mining: Optional[Dict[str, Any]] = None
    if selected:
        directories = [
            Path(value).parent
            for value in results[selected].get("report_paths", {}).values()
        ]
        if directories:
            mining = mine_failure_families(directories[0], top_k=8)
            assert_no_seed_memorisation(mining)

    analysis = {
        "schema_version": "ppo_selection_analysis_v1",
        "benchmark": str(path),
        "plan_fingerprint": report["plan"].get("plan_fingerprint"),
        "selection_rule_fingerprint": selection.get("selection_rule_fingerprint"),
        "final_ppo_parent": selected,
        "selection_reason": selection.get("reason"),
        "candidates": summaries,
        "paired_against_parent": paired,
        "failure_mining": mining,
    }
    Path(args.output).write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    print(f"FINAL_PPO_PARENT = {selected}")
    print(f"  reason: {selection.get('reason')}\n")
    print(f"{'candidate':<14} {'transition (95% CI)':<26} {'sequence (95% CI)':<26} coll")
    for name, summary in summaries.items():
        transition = summary["transition"]
        sequence = summary["sequence"]
        mark = "*" if name == selected else " "
        print(
            f"{mark}{name:<13} "
            f"{transition['rate']!s:<6} "
            f"[{transition['ci'][0]:.3f},{transition['ci'][1]:.3f}] n={transition['n']:<5} "
            f"{sequence['rate']!s:<6} "
            f"[{sequence['ci'][0]:.3f},{sequence['ci'][1]:.3f}] n={sequence['n']:<4} "
            f"{summary['safety']['collision_episodes']}"
        )
    if paired:
        print("\npaired differences vs the parent (same cases):")
        for row in paired:
            ci = row["ci"] or [float("nan"), float("nan")]
            verdict = "tie" if row["equivalent_within_noise"] else "DIFFERENT"
            print(
                f"  {row['candidate']:<13} diff={row['mean_difference']:+.4f} "
                f"CI=[{ci[0]:+.4f},{ci[1]:+.4f}] discordant={row['n_discordant']} "
                f"p={row['sign_test_p']:.4f}  {verdict}"
            )
    if mining:
        print("\ntop failure families (selected parent):")
        for entry in (mining.get("families") or [])[:8]:
            print(f"  {entry}")
    print(f"\nwritten -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
