"""Verify every quantitative claim of the Marine Race Arena manuscript.

Read-only: opens the released evidence package under ``artifacts/paper/`` and
recomputes the values stated in the text and in the tables. It launches nothing,
modifies nothing and uses no path outside the repository. Exits non-zero on any
mismatch.

Usage:
    python article_journal/scripts/verify_claims.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "artifacts" / "paper"
TRACKS = ROOT / "marine_race_arena" / "tracks"

SERVO = "rule_gate_baseline"
CTC = "rule_gate_center_then_commit"
TRACK_FULL = {
    "horseshoe": "Marine Race Horseshoe Bay",
    "vertical": "Marine Race Vertical Serpent",
    "mixed": "Marine Race Mixed Endurance",
}
TRACK_FILE = {
    "horseshoe": "marine_race_horseshoe_bay.json",
    "vertical": "marine_race_vertical_serpent.json",
    "mixed": "marine_race_mixed_endurance.json",
}

results: list[tuple[str, str, object, object]] = []


def check(claim: str, got, want, tol: float = 0.0) -> None:
    if isinstance(got, (int, float)) and isinstance(want, (int, float)):
        ok = abs(got - want) <= max(tol, 0.0)
    else:
        ok = got == want
    results.append(("YES" if ok else "NO", claim, got, want))


def load(rel: str):
    return json.loads((PAPER / rel).read_text(encoding="utf-8"))


def rows_csv(rel: str) -> list[dict]:
    with (PAPER / rel).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def num(value):
    return None if value in (None, "") else float(value)


def mean_sd(values) -> tuple[float, float | None]:
    values = [v for v in values if v is not None]
    mean = statistics.fmean(values)
    return mean, (statistics.stdev(values) if len(values) > 1 else None)


# --------------------------------------------------------------------------- #
def rq1_current_free() -> None:
    data = load("current_free/summary.json")
    circuits = data["circuits"]
    check("current-free completions",
          "{}/{}".format(sum(c["completions"] for c in circuits),
                         sum(c["n_eval"] for c in circuits)), "9/9")
    check("current-free expected gates", sum(c["expected_gates"] for c in circuits), 51)
    check("current-free collisions", sum(c["total_collisions"] for c in circuits), 0)
    check("current-free out-of-bounds", sum(c["total_out_of_bounds"] for c in circuits), 0)
    check("current-free wrong-direction",
          sum(c["total_wrong_direction_crossings"] for c in circuits), 0)
    check("current-free controller", data["controller"], "rule_gate_center_then_commit")
    for circuit in circuits:
        name = circuit["circuit"]
        check("current-free adapter [{}]".format(name), circuit["adapter_actual"], "holoocean")
        check("current-free fallback off [{}]".format(name), circuit["fallback_allowed"], False)
        check("current-free zero currents [{}]".format(name), circuit["currents_are_zero"], True)
        check("current-free runs [{}]".format(name), circuit["n_eval"], 3)

    stated = [("horseshoe", "Horseshoe Bay", 236.0, 6.9, 0.40),
              ("vertical", "Vertical Serpent", 308.6, 6.5, 0.38),
              ("mixed", "Mixed Endurance", 480.4, 15.7, 0.43)]
    for key, label, want_mean, want_sd, want_rate in stated:
        path = PAPER / "current_free" / ("circuit_" + key) / "eval_results.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            times = [float(row["official_time_s"]) for row in csv.DictReader(stream)]
        mean, sd = mean_sd(times)
        check("current-free mean time [{}]".format(label), round(mean, 1), want_mean, 0.05)
        check("current-free sd time [{}]".format(label), round(sd, 1), want_sd, 0.05)
        length = json.loads((TRACKS / TRACK_FILE[key]).read_text(encoding="utf-8"))
        declared = length["track"]["declared_length_m"]
        check("current-free course progress [{}] (m/s)".format(label),
              round(declared / mean, 2), want_rate, 5e-3)


# --------------------------------------------------------------------------- #
def _clean(rows, track, controller) -> list[dict]:
    return [r for r in rows if r["experiment"] == "clean"
            and r["track"] == TRACK_FULL[track] and r["controller"] == controller
            and r["current_profile"] == "none"]


def rq2_clean_tracks(rows) -> None:
    stated = {
        ("horseshoe", SERVO): (5, 12.0, 194.5, 4.8, 0.0),
        ("horseshoe", CTC): (5, 12.0, 197.7, 7.1, 0.0),
        ("vertical", SERVO): (5, 17.0, 236.9, 6.4, 0.0),
        ("vertical", CTC): (4, 14.8, 241.1, 1.8, 0.0),
        ("mixed", SERVO): (2, 17.0, 394.7, 3.9, 0.0),
        ("mixed", CTC): (5, 22.0, 410.7, 8.8, 0.0),
    }
    for (track, controller), (fin, gates, mean, sd, coll) in stated.items():
        label = "{}/{}".format(track, "servo" if controller == SERVO else "ctc")
        sel = _clean(rows, track, controller)
        finished = [r for r in sel if r["status"] == "FINISHED"]
        check("clean seeds [{}]".format(label), len(sel), 5)
        check("clean finished [{}]".format(label), len(finished), fin)
        check("clean mean gates [{}]".format(label),
              round(statistics.fmean(num(r["completed_gates"]) for r in sel), 1), gates, 0.05)
        got_mean, got_sd = mean_sd([num(r["official_time_s"]) for r in finished])
        check("clean mean time [{}]".format(label), round(got_mean, 1), mean, 0.05)
        check("clean sd time [{}]".format(label), round(got_sd, 1), sd, 0.05)
        check("clean mean collisions [{}]".format(label),
              round(statistics.fmean(num(r["gate_world_collisions"]) for r in sel), 1),
              coll, 0.05)

    servo, _ = mean_sd([num(r["official_time_s"]) for r in _clean(rows, "horseshoe", SERVO)
                        if r["status"] == "FINISHED"])
    ctc, _ = mean_sd([num(r["official_time_s"]) for r in _clean(rows, "horseshoe", CTC)
                      if r["status"] == "FINISHED"])
    check("clean Horseshoe staged delay (s)", round(ctc - servo, 1), 3.2, 0.05)
    check("clean Horseshoe staged penalty (%)", round(100 * (ctc - servo) / servo, 1), 1.7, 0.05)


# --------------------------------------------------------------------------- #
def rq3_currents(rows) -> None:
    stated = {SERVO: (2, 8.4, 337.9, 19.2, 692.9, 75.7, 43.4),
              CTC: (3, 8.8, 217.3, 1.5, 223.9, 1.8, 25.2)}
    means = {}
    for controller, (fin, gates, off, off_sd, pen, pen_sd, coll) in stated.items():
        label = "servo" if controller == SERVO else "ctc"
        sel = [r for r in rows if r["experiment"] == "currents"
               and r["current_profile"] == "medium" and r["controller"] == controller]
        finished = [r for r in sel if r["status"] == "FINISHED"]
        check("medium seeds [{}]".format(label), len(sel), 5)
        check("medium finished [{}]".format(label), len(finished), fin)
        check("medium mean gates [{}]".format(label),
              round(statistics.fmean(num(r["completed_gates"]) for r in sel), 1), gates, 0.05)
        got, got_sd = mean_sd([num(r["official_time_s"]) for r in finished])
        means[controller] = got
        check("medium official mean [{}]".format(label), round(got, 1), off, 0.05)
        check("medium official sd [{}]".format(label), round(got_sd, 1), off_sd, 0.05)
        got_p, got_p_sd = mean_sd([num(r["penalized_time_s"]) for r in finished])
        check("medium penalized mean [{}]".format(label), round(got_p, 1), pen, 0.05)
        check("medium penalized sd [{}]".format(label), round(got_p_sd, 1), pen_sd, 0.05)
        check("medium mean collisions [{}]".format(label),
              round(statistics.fmean(num(r["gate_world_collisions"]) for r in sel), 1),
              coll, 0.05)
    check("medium-current time gap (s)",
          round(round(means[SERVO], 1) - round(means[CTC], 1), 1), 120.6, 0.05)
    check("strong-current rows excluded from the release",
          sum(1 for r in rows if r["current_profile"] == "strong"), 0)


# --------------------------------------------------------------------------- #
def _team(sel) -> dict:
    finished = [r for r in sel if r["all_rovers_finished"] == "True"]
    mean, sd = mean_sd([num(r["team_elapsed_time_s"]) for r in finished]) if finished else (None, None)
    pen_mean, pen_sd = (mean_sd([num(r["team_penalized_time_s"]) for r in finished])
                        if finished else (None, None))
    return {
        "n": len(sel),
        "fin": len(finished),
        "gates": round(statistics.fmean(num(r["completed_gates"]) for r in sel), 1),
        "time": None if mean is None else round(mean, 1),
        "time_sd": None if sd is None else round(sd, 1),
        "pen": None if pen_mean is None else round(pen_mean, 1),
        "pen_sd": None if pen_sd is None else round(pen_sd, 1),
        "gw": round(statistics.fmean(num(r["gate_world_collisions"]) for r in sel), 1),
        "iv": round(statistics.fmean(num(r["proximity_events"]) for r in sel), 1),
        "stuck": round(statistics.fmean(num(r["stuck_events"]) for r in sel), 1),
        "oob": round(statistics.fmean(num(r["out_of_bounds_events"]) for r in sel), 1),
    }


def rq4_fleet(rows) -> None:
    for controller, label, time, sd in ((SERVO, "servo", 303.1, 3.2),
                                        (CTC, "ctc", 308.0, 5.8)):
        pair = "{}; {}".format(controller, controller)
        team = _team([r for r in rows if r["experiment"] == "fleet_gap90"
                      and r["controller"] == pair])
        check("fleet seeds [{}]".format(label), team["n"], 5)
        check("fleet finished [{}]".format(label), team["fin"], 5)
        check("fleet team gates [{}]".format(label), team["gates"], 24.0, 0.05)
        check("fleet team time [{}]".format(label), team["time"], time, 0.05)
        check("fleet team time sd [{}]".format(label), team["time_sd"], sd, 0.05)
        check("fleet penalized equals elapsed [{}]".format(label), team["pen"], time, 0.05)
        check("fleet gate/world collisions [{}]".format(label), team["gw"], 0.0)
        check("fleet proximity events [{}]".format(label), team["iv"], 0.0)
        check("fleet out-of-bounds [{}]".format(label), team["oob"], 0.0)
        check("fleet stuck events [{}]".format(label), team["stuck"], 0.0)

    # gap, condition, min_gate_gap, label, finished, gates, time, sd, gw, iv, stuck
    stated = [
        ("0.0", "no_coordination", "2", "gap0/none", 3, 36.0, 219.7, 2.5, 9.3, 2.3, 0.0),
        ("0.0", "leader_follower", "1", "gap0/LF1", 3, 36.0, 262.3, 7.7, 0.0, 0.0, 0.0),
        ("0.0", "leader_follower", "2", "gap0/LF2", 3, 36.0, 288.3, 1.3, 0.0, 0.0, 1.0),
        ("8.0", "no_coordination", "2", "gap8/none", 2, 34.3, 254.0, 24.5, 127.3, 2.0, 0.0),
        ("8.0", "leader_follower", "1", "gap8/LF1", 3, 36.0, 266.0, 8.3, 0.0, 0.0, 0.0),
        ("8.0", "leader_follower", "2", "gap8/LF2", 3, 36.0, 285.7, 3.8, 0.0, 0.0, 0.0),
    ]
    for gap, condition, min_gap, label, fin, gates, time, sd, gw, iv, stuck in stated:
        team = _team([r for r in rows if r["experiment"] == "coordination"
                      and r["start_gap_s"] == gap and r["condition"] == condition
                      and r["min_gate_gap_configured"] == min_gap])
        check("convoy seeds [{}]".format(label), team["n"], 3)
        check("convoy finished [{}]".format(label), team["fin"], fin)
        check("convoy team gates [{}]".format(label), team["gates"], gates, 0.05)
        check("convoy team time [{}]".format(label), team["time"], time, 0.05)
        check("convoy team time sd [{}]".format(label), team["time_sd"], sd, 0.05)
        check("convoy gate/world collisions [{}]".format(label), team["gw"], gw, 0.05)
        check("convoy proximity events [{}]".format(label), team["iv"], iv, 0.05)
        check("convoy stuck events [{}]".format(label), team["stuck"], stuck, 0.05)
        check("convoy out-of-bounds [{}]".format(label), team["oob"], 0.0)

    lf2_gap0 = _team([r for r in rows if r["experiment"] == "coordination"
                      and r["start_gap_s"] == "0.0" and r["condition"] == "leader_follower"
                      and r["min_gate_gap_configured"] == "2"])
    check("convoy LF(2) gap0 penalized team time", lf2_gap0["pen"], 303.3, 0.05)
    check("convoy LF(2) gap0 penalized sd", lf2_gap0["pen_sd"], 1.3, 0.05)


# --------------------------------------------------------------------------- #
def rq5_learning() -> None:
    provenance = load("ppo/provenance.json")
    model = PAPER / "ppo" / "model" / "policy_recurrent_ppo_27d.zip"
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    check("PPO model sha256", digest, provenance["model_sha256"])
    check("PPO model bytes", model.stat().st_size, provenance["model_bytes"])

    episodes = []
    for path in sorted((PAPER / "ppo" / "validation").glob("*_attempt*.result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("technical_status") == "VALID":
            episodes.append(row)
    check("PPO valid episodes", len(episodes), 9)
    check("PPO finished episodes", sum(bool(r["finished"]) for r in episodes), 9)
    check("PPO gates completed", sum(r["gates_completed"] for r in episodes), 153)
    check("PPO collisions", sum(r["collisions"] for r in episodes), 1)
    check("PPO out-of-bounds", sum(r["out_of_bounds_events"] for r in episodes), 0)
    check("PPO observation contract", {r["observation_contract"] for r in episodes},
          {provenance["observation_contract"]})
    check("PPO adapter", {r["adapter"] for r in episodes}, {"holoocean"})
    check("PPO fallback", {r["fallback_used"] for r in episodes}, {False})

    by_track: dict[str, list[dict]] = {}
    for row in episodes:
        by_track.setdefault(row["track"], []).append(row)
    for track, label, count, mean_time, coll in (
            ("horseshoe_bay", "Horseshoe Bay", 3, 192.5, 0),
            ("vertical_serpent", "Vertical Serpent", 3, 239.1, 1),
            ("mixed_endurance", "Mixed Endurance", 3, 365.6, 0)):
        rows_ = by_track[track]
        check("PPO episodes [{}]".format(label), len(rows_), count)
        check("PPO finished [{}]".format(label), sum(bool(r["finished"]) for r in rows_), count)
        check("PPO mean time [{}]".format(label),
              round(statistics.fmean(r["completion_time_s"] for r in rows_), 1),
              mean_time, 0.05)
        check("PPO collisions [{}]".format(label), sum(r["collisions"] for r in rows_), coll)

    campaign = load("ppo/campaign_config.json")
    check("PPO observation dimension", campaign["observation_dim"], 27)
    check("PPO observation contract (config)", campaign["observation_contract"],
          provenance["observation_contract"])
    check("PPO learning rate", campaign["ppo"]["learning_rate"], 0.000006, 1e-12)
    check("PPO clip range", campaign["ppo"]["clip_range"], 0.07, 1e-9)


# --------------------------------------------------------------------------- #
def perception() -> None:
    report = load("perception/visual_smoke_report.json")
    metrics = {t["track"]: t["metrics"] for t in report["tracks"]}
    audit = [("horseshoe_bay", 0.9, 0.8167, 0.5167, 0.018, 7),
             ("vertical_serpent", 0.8833, 0.7833, 0.4833, 1.129, 4),
             ("mixed_endurance", 0.9167, 0.8333, 0.5333, 0.064, 11)]
    for track, detection, quad, pnp, median_error, candidates in audit:
        m = metrics[track]
        check("detection availability [{}]".format(track), m["detection_availability"],
              detection, 1e-4)
        check("four-corner availability [{}]".format(track),
              m["valid_four_corner_availability"], quad, 1e-4)
        check("metric PnP availability [{}]".format(track), m["metric_pnp_availability"],
              pnp, 1e-4)
        check("median orientation error [{}]".format(track),
              m["median_orientation_error_deg"], median_error, 1e-3)
        check("gross orientation error [{}]".format(track),
              m["gross_orientation_error_rate_gt_30deg"], 0.0)
        check("multi-candidate frames [{}]".format(track), m["multi_candidate_frames"],
              candidates)
        check("association correct [{}]".format(track),
              m["multi_candidate_association_within_0_25"], 1.0)
        check("online target switches [{}]".format(track), m["target_switches_online"], 0)

    pose = load("perception/pose/pose_metrics.json")["overall"]
    check("oblique rig pose availability", pose["pose_availability_rate"], 0.117, 1e-3)
    check("oblique rig pose availability (%)",
          round(100 * pose["pose_availability_rate"], 1), 11.7, 0.05)
    check("oblique rig median yaw error", pose["median_yaw_error_deg"], 48.16, 1e-2)
    check("oblique rig median yaw error (stated)",
          round(pose["median_yaw_error_deg"], 1), 48.2, 0.05)
    check("oblique rig p90 yaw error", pose["p90_yaw_error_deg"], 50.344, 1e-2)
    check("oblique rig orientation accuracy", pose["orientation_class_accuracy"], 0.0)
    check("oblique rig sign accuracy", pose["yaw_sign_accuracy"], 0.0)
    check("oblique rig median distance error", pose["median_distance_error_m"], 1.732, 1e-3)
    check("oblique rig frames", pose["n_frames"], 60)


# --------------------------------------------------------------------------- #
def information_boundary() -> None:
    data = load("benchmark/local_vs_referee.json")
    overall = data["overall"]
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
    check("local agreement above 99%",
          overall["matched_advancements"] / overall["referee_advancements"] > 0.99, True)

    stated = {"Marine Race Horseshoe Bay": (1120, 1120, 1116, 10, 4, 0.924, 1.353),
              "Marine Race Vertical Serpent": (159, 160, 159, 1, 0, 0.627, 1.155),
              "Marine Race Mixed Endurance": (195, 192, 192, 0, 3, 0.462, 1.287)}
    for track, (ref, local, matched, false, missed, median, p95) in stated.items():
        block = data["by_track"][track]
        label = track.replace("Marine Race ", "")
        check("referee advancements [{}]".format(label), block["referee_advancements"], ref)
        check("local advancements [{}]".format(label), block["local_advancements"], local)
        check("matched advancements [{}]".format(label), block["matched_advancements"], matched)
        check("false local [{}]".format(label), block["false_local_advancements"], false)
        check("missed local [{}]".format(label), block["missed_local_advancements"], missed)
        check("median delay [{}]".format(label),
              block["advancement_delay_s"]["median"], median, 1e-3)
        check("p95 delay [{}]".format(label), block["advancement_delay_s"]["p95"], p95, 1e-3)


# --------------------------------------------------------------------------- #
def released_package() -> None:
    """The manifest must describe exactly the files on disk, byte for byte."""
    manifest = load("manifest.json")
    listed = {entry["path"]: entry for entry in manifest["files"]}
    on_disk = {path.relative_to(PAPER).as_posix()
               for path in PAPER.rglob("*")
               if path.is_file() and path.name != "manifest.json"}
    check("manifest file count", manifest["file_count"], len(listed))
    check("manifest lists every released file", sorted(on_disk - set(listed)), [])
    check("manifest lists no missing file", sorted(set(listed) - on_disk), [])
    mismatched = []
    for rel, entry in sorted(listed.items()):
        data = (PAPER / rel).read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            mismatched.append(rel)
    check("manifest hashes ({} files)".format(len(listed)), mismatched, [])
    check("manifest total bytes", manifest["total_bytes"],
          sum(entry["bytes"] for entry in listed.values()))


def penalty_identity(rows) -> None:
    problems = []
    checked = 0
    for row in rows:
        if row["team_elapsed_time_s"]:
            if row["all_rovers_finished"] != "True":
                continue
            base, done = num(row["team_elapsed_time_s"]), num(row["team_penalized_time_s"])
        else:
            if row["status"] != "FINISHED":
                continue
            base, done = num(row["official_time_s"]), num(row["penalized_time_s"])
        if base is None or done is None:
            continue
        checked += 1
        if abs(base + (num(row["penalties_s"]) or 0.0) - done) > 0.05:
            problems.append(row["run_id"])
    check("penalty identity discrepancies ({} runs)".format(checked), len(problems), 0)


def main() -> int:
    if not PAPER.is_dir():
        print("missing evidence package: {}".format(PAPER))
        return 2
    rows = rows_csv("benchmark/runs.csv")
    check("released benchmark runs", len(rows), 68)

    released_package()
    rq1_current_free()
    rq2_clean_tracks(rows)
    rq3_currents(rows)
    rq4_fleet(rows)
    rq5_learning()
    perception()
    information_boundary()
    penalty_identity(rows)

    width = max(len(row[1]) for row in results)
    mismatched = sum(1 for status, *_ in results if status == "NO")
    for status, claim, got, want in results:
        print("{:4s} {:<{w}}  got={!r} expected={!r}".format(
            status, claim, got, want, w=width))
    print("\n{} checks: {} verified, {} mismatched".format(
        len(results), len(results) - mismatched, mismatched))
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
