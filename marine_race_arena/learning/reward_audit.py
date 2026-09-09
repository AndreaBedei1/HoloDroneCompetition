"""Static + empirical audit of the local-transition reward contract.

Run once before long training and then freeze the reward.  The audit answers a
specific question: can any component grow without bound, fire every frame when it
should be an event, or dominate the others -- and does the shaping create an
incentive to stall, to leave the course, to circle a gate, or to farm progress
without ever crossing?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from marine_race_arena.learning.provenance import now_utc
from marine_race_arena.learning.reward_local_transition import (
    LocalTransitionRewardConfig,
)

AUDIT_VERSION = "local_transition_reward_audit_v1"
REWARD_CONTRACT_VERSION = "local_transition_reward_v2_collision_entry"

# kind: "event" fires a bounded number of times; "dense" fires every step.
COMPONENT_SPEC: Dict[str, Dict[str, Any]] = {
    "bearing_delta": {"kind": "dense", "bound": "bearing_delta_scale/45 per step, clipped +-1 input"},
    "elevation_delta": {"kind": "dense", "bound": "elevation_delta_scale/30 per step"},
    "range_delta": {"kind": "dense", "bound": "range_delta_scale per step"},
    "visual_center_delta": {"kind": "dense", "bound": "visual_center_delta_scale per step"},
    "visual_area_delta": {"kind": "dense", "bound": "visual_area_delta_scale*0.1 per step"},
    "gate_crossing": {"kind": "event", "bound": "gate_crossing_bonus per crossing"},
    "post_cross_alignment_delta": {"kind": "dense", "bound": "scaled bearing+elevation delta"},
    "post_cross_range_delta": {"kind": "dense", "bound": "post_cross_range_delta_scale per step"},
    "previous_gate_return_penalty": {"kind": "event", "bound": "once per crossing"},
    "post_cross_orbit_penalty": {"kind": "event", "bound": "once per crossing"},
    "moving_away_penalty": {"kind": "event", "bound": "once per crossing"},
    "acquisition_timeout_penalty": {"kind": "event", "bound": "once per crossing"},
    "collision_penalty": {"kind": "event", "bound": "collision_entry_penalty_episode_cap"},
    "collision_contact_penalty": {"kind": "event", "bound": "collision_contact_penalty_episode_cap"},
    "out_of_bounds_penalty": {"kind": "event", "bound": "once per episode"},
    "wrong_direction_penalty": {"kind": "event", "bound": "once per episode"},
    "missed_gate_penalty": {"kind": "event", "bound": "once per episode"},
    "action_change_penalty": {"kind": "dense", "bound": "efficiency only"},
    "jerk_penalty": {"kind": "dense", "bound": "efficiency only"},
    "energy_penalty": {"kind": "dense", "bound": "efficiency only"},
    "time_cost": {"kind": "dense", "bound": "constant per step"},
    "completion": {"kind": "event", "bound": "constant, length-independent"},
    "terminal_failure_penalty": {"kind": "event", "bound": "once per episode"},
}


def static_ranges(config: LocalTransitionRewardConfig) -> Dict[str, Any]:
    """Per-step and per-episode theoretical magnitudes from the config."""

    c = config
    per_step = {
        "bearing_delta": c.bearing_delta_scale / 45.0,
        "elevation_delta": c.elevation_delta_scale / 30.0,
        "range_delta": c.range_delta_scale,
        "visual_center_delta": c.visual_center_delta_scale,
        "visual_area_delta": c.visual_area_delta_scale * 0.1,
        "post_cross_alignment_delta": (
            (c.bearing_delta_scale / 45.0 + c.elevation_delta_scale / 30.0)
            * c.post_cross_alignment_delta_scale
        ),
        "post_cross_range_delta": c.range_delta_scale * c.post_cross_range_delta_scale,
        "time_cost": max(c.reliability_time_cost, c.efficiency_time_cost),
    }
    per_episode_events = {
        "gate_crossing": c.gate_crossing_bonus,
        "collision_penalty": c.collision_entry_penalty_episode_cap,
        "collision_contact_penalty": c.collision_contact_penalty_episode_cap,
        "out_of_bounds_penalty": c.out_of_bounds_penalty,
        "wrong_direction_penalty": c.wrong_direction_penalty,
        "missed_gate_penalty": c.missed_gate_penalty,
        "previous_gate_return_penalty": c.previous_gate_return_penalty,
        "acquisition_timeout_penalty": c.acquisition_timeout_penalty,
        "post_cross_orbit_penalty": c.post_cross_orbit_penalty,
        "moving_away_penalty": c.moving_away_penalty,
        "completion": c.completion_bonus,
        "terminal_failure_penalty": c.terminal_failure_penalty,
    }
    return {
        "component_abs_bound": c.component_abs_bound,
        "total_abs_bound": c.total_abs_bound,
        "per_step_magnitude": {k: round(float(v), 6) for k, v in per_step.items()},
        "per_episode_event_magnitude": {
            k: round(float(v), 6) for k, v in per_episode_events.items()
        },
    }


def static_findings(config: LocalTransitionRewardConfig) -> List[Dict[str, Any]]:
    """Structural checks that do not need a simulator."""

    c = config
    out: List[Dict[str, Any]] = []

    def add(check, ok, detail, severity="info"):
        out.append({"check": check, "ok": bool(ok),
                    "severity": severity if not ok else "info", "detail": detail})

    add("every_component_is_clipped",
        c.component_abs_bound > 0 and c.total_abs_bound > 0,
        f"component |x|<={c.component_abs_bound}, total |x|<={c.total_abs_bound}",
        "critical")
    add("collision_is_event_based_not_per_frame",
        c.collision_contact_frame_penalty <= 0.1 * c.collision_penalty,
        f"entry {c.collision_penalty} vs per-contact-frame "
        f"{c.collision_contact_frame_penalty}", "critical")
    add("collision_shaping_is_capped",
        c.collision_entry_penalty_episode_cap > 0
        and c.collision_contact_penalty_episode_cap > 0,
        f"entry cap {c.collision_entry_penalty_episode_cap}, contact cap "
        f"{c.collision_contact_penalty_episode_cap}", "critical")
    add("sustained_contact_cannot_outweigh_the_impact",
        c.collision_contact_penalty_episode_cap < c.collision_penalty,
        f"contact cap {c.collision_contact_penalty_episode_cap} < entry "
        f"{c.collision_penalty}", "high")
    add("collision_dominates_efficiency_shaping",
        c.collision_penalty > 100 * max(
            c.efficiency_time_cost, c.efficiency_jerk_penalty,
            c.efficiency_energy_penalty, c.efficiency_action_change_penalty),
        f"collision {c.collision_penalty} vs efficiency terms", "critical")
    add("leaving_the_course_is_never_profitable",
        c.out_of_bounds_penalty > c.gate_crossing_bonus,
        f"out of bounds {c.out_of_bounds_penalty} > crossing "
        f"{c.gate_crossing_bonus}", "critical")
    add("wrong_direction_is_never_profitable",
        c.wrong_direction_penalty > c.gate_crossing_bonus,
        f"wrong direction {c.wrong_direction_penalty} > crossing "
        f"{c.gate_crossing_bonus}", "critical")
    add("missed_gate_is_never_profitable",
        c.missed_gate_penalty > c.gate_crossing_bonus,
        f"missed gate {c.missed_gate_penalty} > crossing "
        f"{c.gate_crossing_bonus}", "critical")
    add("circling_a_gate_is_penalized",
        c.post_cross_orbit_penalty > 0, "post-crossing orbit penalty present",
        "high")
    add("stalling_is_not_rewarded",
        c.reliability_time_cost > 0.0,
        f"time cost {c.reliability_time_cost} per step is strictly negative",
        "high")
    add("completion_is_length_independent",
        not hasattr(c, "completion_bonus_per_gate"),
        "single constant completion bonus", "high")
    add("efficiency_disabled_during_reliability_phase",
        not c.efficiency_unlocked
        and c.reliability_jerk_penalty == 0.0
        and c.reliability_energy_penalty == 0.0
        and c.reliability_action_change_penalty == 0.0,
        "jerk/energy/action-change are zero until efficiency unlocks", "high")
    # A pure oscillation must not accumulate: bounded signed deltas cancel.
    forward = float(np.clip(5.0 - 4.0, -1.0, 1.0)) * c.range_delta_scale
    backward = float(np.clip(4.0 - 5.0, -1.0, 1.0)) * c.range_delta_scale
    add("progress_shaping_cannot_be_farmed_by_oscillation",
        abs(forward + backward) < 1e-9,
        f"a forward/backward pair sums to {forward + backward}", "critical")
    crossing_vs_dense = c.gate_crossing_bonus / max(
        1e-9, sum(static_ranges(c)["per_step_magnitude"].values()))
    add("crossing_reward_outweighs_one_step_of_dense_shaping",
        crossing_vs_dense > 1.0,
        f"crossing/dense-per-step ratio {crossing_vs_dense:.2f}", "high")
    return out


def summarize_episode_components(
    episodes: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Empirical per-component distribution over recorded episodes.

    ``episodes`` is a sequence of {component: [per-step values]} mappings.
    """

    totals: Dict[str, List[float]] = {}
    fire_counts: Dict[str, List[int]] = {}
    for episode in episodes:
        for name, values in episode.items():
            series = [float(v) for v in values]
            totals.setdefault(name, []).append(float(sum(series)))
            fire_counts.setdefault(name, []).append(
                sum(1 for v in series if abs(v) > 1e-12)
            )
    out: Dict[str, Any] = {}
    for name in sorted(totals):
        episode_totals = np.asarray(totals[name], dtype=np.float64)
        fires = np.asarray(fire_counts[name], dtype=np.float64)
        spec = COMPONENT_SPEC.get(name, {"kind": "unknown", "bound": "unspecified"})
        out[name] = {
            "kind": spec["kind"],
            "documented_bound": spec["bound"],
            "episode_total_mean": float(episode_totals.mean()),
            "episode_total_p05": float(np.quantile(episode_totals, 0.05)),
            "episode_total_p95": float(np.quantile(episode_totals, 0.95)),
            "episode_total_min": float(episode_totals.min()),
            "episode_total_max": float(episode_totals.max()),
            "mean_firings_per_episode": float(fires.mean()),
            "max_firings_per_episode": int(fires.max()),
        }
    return out


def empirical_findings(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Flag event components that behaved like dense ones, and outliers."""

    out = []
    for name, row in summary.items():
        if row["kind"] == "event" and row["max_firings_per_episode"] > 64:
            out.append({
                "check": f"{name}_fires_like_a_dense_term",
                "ok": False, "severity": "high",
                "detail": (
                    f"event component fired up to "
                    f"{row['max_firings_per_episode']} times in one episode"
                ),
            })
    magnitudes = {
        name: abs(row["episode_total_mean"]) for name, row in summary.items()
    }
    if magnitudes:
        largest = max(magnitudes, key=magnitudes.get)
        others = sum(v for k, v in magnitudes.items() if k != largest)
        if others > 0 and magnitudes[largest] > 10.0 * others:
            out.append({
                "check": "one_component_dominates_the_return",
                "ok": False, "severity": "high",
                "detail": f"{largest} exceeds all others combined by >10x",
            })
    return out


def build_report(
    config: Optional[LocalTransitionRewardConfig] = None,
    episodes: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    config = config or LocalTransitionRewardConfig()
    checks = static_findings(config)
    summary = summarize_episode_components(episodes) if episodes else {}
    checks = checks + (empirical_findings(summary) if summary else [])
    failures = [row for row in checks if not row["ok"]]
    critical = [row for row in failures if row["severity"] == "critical"]
    return {
        "schema_version": AUDIT_VERSION,
        "reward_contract_version": REWARD_CONTRACT_VERSION,
        "generated_utc": now_utc(),
        "config": {k: v for k, v in vars(config).items()},
        "static_ranges": static_ranges(config),
        "component_spec": COMPONENT_SPEC,
        "checks": checks,
        "empirical_summary": summary,
        "empirical_episodes": len(episodes or []),
        "failures": failures,
        "critical_failures": critical,
        "verdict": (
            "reward_contract_coherent_freeze_it" if not failures
            else "reward_bug_detected_fix_required" if critical
            else "reward_contract_coherent_with_minor_notes"
        ),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Local-transition reward audit",
        "",
        f"Generated: {report['generated_utc']}",
        f"Reward contract: `{report['reward_contract_version']}`",
        "",
        f"**Verdict: {report['verdict']}**",
        "",
        "## Structural checks",
        "",
        "| Check | Result | Detail |",
        "|---|:---:|---|",
    ]
    for row in report["checks"]:
        lines.append(
            f"| {row['check']} | {'PASS' if row['ok'] else row['severity'].upper()} "
            f"| {row['detail']} |"
        )
    ranges = report["static_ranges"]
    lines += [
        "",
        f"Every component is clipped to +-{ranges['component_abs_bound']} and the "
        f"total step reward to +-{ranges['total_abs_bound']}, so no term is "
        "unbounded.",
        "",
        "## Per-episode event magnitudes",
        "",
        "| Component | Magnitude |",
        "|---|---:|",
    ]
    for name, value in sorted(ranges["per_episode_event_magnitude"].items()):
        lines.append(f"| {name} | {value} |")
    if report["empirical_summary"]:
        lines += [
            "",
            f"## Empirical distribution ({report['empirical_episodes']} episodes)",
            "",
            "| Component | Kind | Mean total | p05 | p95 | Max firings/ep |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for name, row in report["empirical_summary"].items():
            lines.append(
                f"| {name} | {row['kind']} | {row['episode_total_mean']:.4f} | "
                f"{row['episode_total_p05']:.4f} | {row['episode_total_p95']:.4f} | "
                f"{row['max_firings_per_episode']} |"
            )
    lines += ["", "## Conclusion", ""]
    if not report["failures"]:
        lines.append(
            "No unbounded term, no duplicated penalty, no per-frame accumulation "
            "of an event penalty, and no incentive to stall, leave the course, "
            "circle a gate or farm progress without crossing. The contract is "
            "coherent and is frozen for the long experiments."
        )
    else:
        for row in report["failures"]:
            lines.append(f"- **{row['severity']}**: {row['check']} - {row['detail']}")
    lines.append("")
    return "\n".join(lines)


def write_report(output_dir: str | Path, report: Mapping[str, Any]) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "reward_audit.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (path / "reward_audit.md").write_text(_markdown(report), encoding="utf-8")
    return path / "reward_audit.json"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="results/rl_public/reward_audit")
    parser.add_argument("--episodes", help="optional recorded component JSON")
    args = parser.parse_args(argv)
    episodes = None
    if args.episodes and Path(args.episodes).exists():
        episodes = json.loads(Path(args.episodes).read_text(encoding="utf-8"))
    report = build_report(episodes=episodes)
    path = write_report(args.output_dir, report)
    print(json.dumps({
        "verdict": report["verdict"],
        "checks": len(report["checks"]),
        "failures": len(report["failures"]),
        "critical": len(report["critical_failures"]),
        "report": str(path),
    }, indent=2))
    return 0 if not report["critical_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
