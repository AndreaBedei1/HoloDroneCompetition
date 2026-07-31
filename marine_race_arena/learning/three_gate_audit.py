"""Audit whether a long run ever evaluated its three-gate (and six-gate) cases.

Answers, from the run's own stored evaluations, the question a ``0.0`` success
rate cannot: were sequence cases *absent from the suite*, or were they *run and
failed*?  Every evaluation report is classified, and the audit states which of
the two the status value represented.

Usage::

    python -m marine_race_arena.learning.three_gate_audit \
        results/rl/multigate_reliability_first/<run> --out <run>/diagnostics/three_gate_audit.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from marine_race_arena.learning.longrun_evaluation import (
    CATEGORY_FIRST_STAGE_INDEX,
    category_not_evaluated_reason,
)

AUDIT_SCHEMA_VERSION = "three_gate_audit_v1"


def _category_counts(report: Dict[str, Any], category: str) -> Dict[str, Any]:
    """Count a category's episodes from the stored rows, not from the summary.

    Reading the rows is what makes the audit trustworthy: it does not rely on the
    same summary field whose ambiguity is under investigation.
    """
    rows = report.get("rows") or []
    subset = [row for row in rows if row.get("category") == category]
    successes = sum(1 for row in subset if row.get("finished"))
    return {
        "n": len(subset),
        "successes": successes,
        "measured_rate": (successes / len(subset)) if subset else None,
        "reported_rate": report.get(f"{category}_completion_rate"),
    }


def audit_run(run_dir: str | Path, *, category: str = "three_gate") -> Dict[str, Any]:
    run = Path(run_dir)
    evaluations = sorted((run / "evaluations").glob("*.json"))
    entries: List[Dict[str, Any]] = []
    for path in evaluations:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict) or "rows" not in report:
            continue
        counts = _category_counts(report, category)
        entries.append(
            {
                "evaluation": path.name,
                "timesteps": report.get("timesteps"),
                "stage": report.get("stage"),
                "mode": report.get("mode"),
                "n_eval": report.get("n_eval"),
                "categories_present": sorted(
                    {str(row.get("category")) for row in report.get("rows") or []}
                ),
                **counts,
            }
        )

    evaluated = [e for e in entries if e["n"] > 0]
    failures = [e for e in evaluated if (e["measured_rate"] or 0.0) < 1.0]
    status_path = run / "status.json"
    status: Dict[str, Any] = {}
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = {}
    final_stage = status.get("curriculum_stage")
    stages_reached = sorted({str(e["stage"]) for e in entries if e.get("stage")})
    verdict = (
        "evaluated_and_failed"
        if evaluated and failures
        else "evaluated_and_passed"
        if evaluated
        else "never_evaluated"
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "run_dir": str(run.as_posix()),
        "category": category,
        "first_stage_in_suite": f"C{CATEGORY_FIRST_STAGE_INDEX.get(category, 0)}",
        "final_curriculum_stage": final_stage,
        "stages_reached": stages_reached,
        "evaluations_scanned": len(entries),
        "evaluations_containing_category": len(evaluated),
        "episodes_of_category": sum(e["n"] for e in entries),
        "verdict": verdict,
        "reason": category_not_evaluated_reason(category, final_stage)
        if verdict == "never_evaluated"
        else None,
        "reported_status_value": status.get(f"{category}_success"),
        "reported_status_was_misleading": bool(
            verdict == "never_evaluated"
            and status.get(f"{category}_success") is not None
        ),
        "mode_counts": dict(Counter(str(e["mode"]) for e in entries)),
        "stage_counts": dict(Counter(str(e["stage"]) for e in entries)),
        "evaluations": entries,
    }


def render(audit: Dict[str, Any]) -> str:
    lines = [
        f"category                : {audit['category']}",
        f"run                     : {audit['run_dir']}",
        f"evaluations scanned     : {audit['evaluations_scanned']}",
        f"evaluations with cases  : {audit['evaluations_containing_category']}",
        f"episodes of category    : {audit['episodes_of_category']}",
        f"stages reached          : {', '.join(audit['stages_reached'])}",
        f"final curriculum stage  : {audit['final_curriculum_stage']}",
        f"category enters suite at: {audit['first_stage_in_suite']}",
        f"reported status value   : {audit['reported_status_value']}",
        f"VERDICT                 : {audit['verdict']}",
    ]
    if audit.get("reason"):
        lines.append(f"reason                  : {audit['reason']}")
    if audit["reported_status_was_misleading"]:
        lines.append(
            "NOTE                    : the reported value was NOT a measured failure; "
            "no episode of this category was ever run."
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--category", default="three_gate")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    audit = audit_run(args.run_dir, category=args.category)
    print(render(audit))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(f"\n[audit] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
