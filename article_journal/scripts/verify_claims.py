"""Verify every headline quantitative claim of the journal manuscript.

Read-only: opens frozen artifacts and recomputes the values stated in the text
and tables. It launches nothing and modifies nothing. Exits non-zero on any
mismatch.

Usage:
    python article_journal/scripts/verify_claims.py
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The raw 78-run matrix is git-ignored; it is read from the main checkout when
# present. Its absence downgrades those rows to SKIP rather than failing.
MAIN_MATRIX = Path(
    os.environ.get(
        "MRA_MATRIX_ROOT",
        r"C:\Users\andrea.bedei3\Desktop\HoloDroneCompetition"
        r"\results\onboard_only_validation\final_20260715",
    )
)

results: list = []


def check(claim: str, got, want, tol: float = 0.0) -> None:
    if isinstance(got, (int, float)) and isinstance(want, (int, float)):
        ok = abs(got - want) <= max(tol, 0.0)
    else:
        ok = got == want
    results.append(("YES" if ok else "NO", claim, got, want))


def skip(claim: str, why: str) -> None:
    results.append(("SKIP", claim, why, ""))


def load(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def main() -> int:
    # --- RQ1: current-free completion -------------------------------------
    data = load("results/rl_public/visual_pose_v2/official_no_current/summary.json")
    circuits = data["circuits"]
    check(
        "current-free completions",
        "{}/{}".format(sum(c["completions"] for c in circuits),
                       sum(c["n_eval"] for c in circuits)),
        "9/9",
    )
    check("current-free expected gates", sum(c["expected_gates"] for c in circuits), 51)
    check("current-free collisions", sum(c["total_collisions"] for c in circuits), 0)
    check("current-free out-of-bounds", sum(c["total_out_of_bounds"] for c in circuits), 0)
    check("current-free wrong-direction", sum(c["total_wrong_direction_crossings"] for c in circuits), 0)
    for circuit in circuits:
        name = circuit["circuit"]
        check("current-free adapter [{}]".format(name), circuit["adapter_actual"], "holoocean")
        check("current-free fallback off [{}]".format(name), circuit["fallback_allowed"], False)
        check("current-free zero currents [{}]".format(name), circuit["currents_are_zero"], True)

    timing = [
        ("Horseshoe Bay", "circuit_horseshoe", 236.0, 6.9),
        ("Vertical Serpent", "circuit_vertical", 308.6, 6.5),
        ("Mixed Endurance", "circuit_mixed", 480.4, 15.7),
    ]
    for name, sub, mean, sd in timing:
        path = ROOT / "results/rl_public/visual_pose_v2/official_no_current" / sub / "eval_results.csv"
        times = [float(row["official_time_s"]) for row in csv.DictReader(path.open(encoding="utf-8"))]
        computed_mean = sum(times) / len(times)
        computed_sd = math.sqrt(sum((t - computed_mean) ** 2 for t in times) / (len(times) - 1))
        check("current-free mean time [{}]".format(name), round(computed_mean, 1), mean, 0.05)
        check("current-free sd time [{}]".format(name), round(computed_sd, 1), sd, 0.05)

    # --- Readiness gate and multi-gate survival ---------------------------
    verdict = load("results/rl_public/ppo_final_readiness_929792/readiness_verdict.json")
    survival = verdict["unconditional_survival_from_episode_start"]
    check("survival gate 1",
          (survival["1"]["crossed"], survival["1"]["eligible_episodes_from_start"]), (120, 120))
    check("survival gate 2", round(survival["2"]["probability"], 4), 0.7333, 1e-4)
    check("survival gate 8", round(survival["8"]["probability"], 3), 0.575, 1e-3)
    check("survival gate 22", round(survival["22"]["probability"], 3), 0.300, 1e-3)
    check("universal transition success",
          verdict["transition_metrics"]["universal_transition_success"]["rate"], 0.82)
    check("transition cases", verdict["sample_contract"]["transition_cases"], 500)
    check("no manual override", verdict["manual_override_applied"], False)

    # --- RQ5: matched 474-episode benchmark -------------------------------
    rows = list(csv.DictReader(
        (ROOT / "results/rl_public/final_benchmark/aggregate_by_group.csv").open(encoding="utf-8")))
    lut = {(row["controller"], row["group"]): row for row in rows}
    expected = {
        ("rule_gate_center_then_commit", "ALL_official"): 93.3,
        ("hybrid", "ALL_official"): 86.7,
        ("ppo_900462", "ALL_official"): 40.0,
        ("ppo_1000814", "ALL_official"): 26.7,
        ("ppo_525678", "ALL_official"): 20.0,
        ("bc_v3", "ALL_official"): 0.0,
        ("rule_gate_center_then_commit", "ALL_non_official"): 100.0,
        ("hybrid", "ALL_non_official"): 100.0,
        ("ppo_525678", "ALL_non_official"): 100.0,
        ("ppo_1000814", "ALL_non_official"): 96.9,
        ("ppo_900462", "ALL_non_official"): 90.6,
        ("bc_v3", "ALL_non_official"): 89.1,
    }
    for key, want in expected.items():
        check("completion {} / {}".format(*key),
              round(float(lut[key]["success_rate"]) * 100, 1), want, 0.05)

    manifest = load("results/rl_public/final_benchmark/package_manifest.json")
    check("benchmark episodes run", manifest["episodes_run"], 474)
    check("benchmark episodes planned", manifest["episodes_planned"], 474)
    check("benchmark control timestep", manifest["dt"], 0.1)
    check("benchmark adapter", manifest["adapter"], "holoocean")
    check("benchmark current profile", manifest["current_profile"], "none")

    episodes = list(csv.DictReader(
        (ROOT / "results/rl_public/final_benchmark/episodes.csv").open(encoding="utf-8")))
    official = {"official_horseshoe_bay", "official_vertical_serpent", "official_mixed_endurance"}
    ppo = {"ppo_525678", "ppo_900462", "ppo_1000814"}
    failures = [e for e in episodes
                if e["group"] in official and e["controller"] in ppo
                and e["finished"].lower() != "true"]
    dnf = sum(1 for e in failures if e["referee_status"] == "DNF")
    check("PPO official failures", len(failures), 32)
    check("PPO failures that are missed-gate DNF", dnf, 29)
    check("PPO missed-gate share (%)", round(100 * dnf / len(failures), 1), 90.6, 0.05)
    fractions = sorted(int(e["completed_gates"]) / int(e["expected_gates"]) for e in failures)
    middle = len(fractions) // 2
    median = (fractions[middle] if len(fractions) % 2
              else 0.5 * (fractions[middle - 1] + fractions[middle]))
    check("PPO median failure course fraction (%)", round(100 * median, 1), 23.9, 0.05)

    # --- Perception audit --------------------------------------------------
    smoke = load("artifacts_gen2/visual_collection_smoke/visual_smoke_report.json")
    by_track = {t["track"]: t["metrics"] for t in smoke["tracks"]}
    audit = [
        ("horseshoe_bay", 0.9, 0.8167, 0.5167, 0.018),
        ("vertical_serpent", 0.8833, 0.7833, 0.4833, 1.129),
        ("mixed_endurance", 0.9167, 0.8333, 0.5333, 0.064),
    ]
    for track, detection, quad, pnp, median_error in audit:
        metrics = by_track[track]
        check("detection availability [{}]".format(track), metrics["detection_availability"], detection, 1e-4)
        check("four-corner availability [{}]".format(track), metrics["valid_four_corner_availability"], quad, 1e-4)
        check("metric PnP availability [{}]".format(track), metrics["metric_pnp_availability"], pnp, 1e-4)
        check("median orientation error [{}]".format(track), metrics["median_orientation_error_deg"], median_error, 1e-3)
        check("gross orientation error [{}]".format(track), metrics["gross_orientation_error_rate_gt_30deg"], 0.0)
        check("association correct [{}]".format(track), metrics["multi_candidate_association_within_0_25"], 1.0)
        check("online target switches [{}]".format(track), metrics["target_switches_online"], 0)

    pose = load("results/rl_public/visual_pose_v2/vision_pose/pose_metrics.json")["overall"]
    check("oblique rig pose availability", pose["pose_availability_rate"], 0.117, 1e-3)
    check("oblique rig median yaw error", pose["median_yaw_error_deg"], 48.16, 1e-2)
    check("oblique rig p90 yaw error", pose["p90_yaw_error_deg"], 50.344, 1e-2)
    check("oblique rig orientation accuracy", pose["orientation_class_accuracy"], 0.0)
    check("oblique rig sign accuracy", pose["yaw_sign_accuracy"], 0.0)
    check("oblique rig median distance error", pose["median_distance_error_m"], 1.732, 1e-3)
    check("oblique rig frames", pose["n_frames"], 60)

    # --- Information boundary, from the raw 78-run matrix ------------------
    matrix = MAIN_MATRIX / "complete_experiment_manifest.json"
    if matrix.exists():
        summary = json.loads(matrix.read_text(encoding="utf-8"))
        overall = summary["local_vs_referee"]["overall"]
        check("matrix run count", summary["run_count"], 78)
        check("matrix complete", summary["complete_matrix"], True)
        check("referee advancements", overall["referee_advancements"], 1474)
        check("local advancements", overall["local_advancements"], 1472)
        check("matched advancements", overall["matched_advancements"], 1467)
        check("false local advancements", overall["false_local_advancements"], 11)
        check("missed local advancements", overall["missed_local_advancements"], 7)
        check("median local delay", overall["advancement_delay_s"]["median"], 0.858, 1e-3)
        check("p95 local delay", overall["advancement_delay_s"]["p95"], 1.353, 1e-3)
        check("event consistency errors", overall["event_consistency_errors"], 0)
        check("finish-order inversions", overall["finish_order_inversions"], 0)
        check("local finish before referee", overall["local_finish_before_referee"], 0)
    else:
        skip("78-run matrix rows", "raw matrix not present at {}".format(MAIN_MATRIX))

    # --- Derived quantities stated in prose --------------------------------
    check("strong set-flow magnitude", round(math.hypot(0.75, 1.05), 2), 1.29, 5e-3)
    check("strong / clean speed ratio", round(math.hypot(0.75, 1.05) / (93.8 / 194.5), 1), 2.7, 0.05)
    check("medium constant component", round(0.5 * math.hypot(0.75, 1.05), 2), 0.65, 5e-3)
    check("clean Horseshoe course speed", round(93.8 / 194.5, 2), 0.48, 5e-3)
    check("22-gate compounding at 0.95 per gate", round(0.95 ** 22, 2), 0.32, 5e-3)
    check("Center-then-Commit clean penalty (%)", round(100 * (197.7 - 194.5) / 194.5, 1), 1.6, 0.15)
    check("medium-current time gap (s)", round(337.9 - 217.3, 1), 120.6, 0.05)

    width = max(len(row[1]) for row in results)
    mismatched = sum(1 for status, *_ in results if status == "NO")
    skipped = sum(1 for status, *_ in results if status == "SKIP")
    for status, claim, got, want in results:
        detail = ("got={!r}".format(got) if status == "SKIP"
                  else "got={!r} expected={!r}".format(got, want))
        print("{:4s} {:<{w}}  {}".format(status, claim, detail, w=width))
    print("\n{} checks: {} verified, {} mismatched, {} skipped".format(
        len(results), len(results) - mismatched - skipped, mismatched, skipped))
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
