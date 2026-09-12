"""Regenerate the data-driven result tables of the manuscript.

Post-processing only: this reads the released evidence package under
``artifacts/paper/`` and rewrites the seven data-driven table bodies in
``article_journal/tables/``. It never launches the simulator and never modifies
an artifact.

Numbers come entirely from the artifacts. Emphasis (``\\best``) is an editorial
choice declared in ``EMPHASIS`` below, so a regenerated table is byte-identical
to the committed one.

Conventions (manuscript policy):

* sample standard deviation (ddof=1), omitted when a cell has fewer than two
  finished runs;
* one decimal place for times, mean gates and mean event counts;
* integer finish counts and exact gate totals;
* time statistics use finished runs only; gate and event means use all seeds.

Usage:
    python article_journal/scripts/regenerate_tables.py            # rewrite the .tex files
    python article_journal/scripts/regenerate_tables.py --check    # print, do not write
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "artifacts" / "paper"
TABLES = ROOT / "article_journal" / "tables"

# Tables are written with LF on every platform so a checkout is byte-identical
# everywhere (see .gitattributes).
LF = chr(10)

TRACK_FULL = {
    "horseshoe": "Marine Race Horseshoe Bay",
    "vertical": "Marine Race Vertical Serpent",
    "mixed": "Marine Race Mixed Endurance",
}
TRACK_LABEL = {
    "horseshoe": "Horseshoe Bay",
    "vertical": "Vertical Serpent",
    "mixed": "Mixed Endurance",
}
EXPECTED_GATES = {"horseshoe": 12, "vertical": 17, "mixed": 22}
SERVO = "rule_gate_baseline"
CTC = "rule_gate_center_then_commit"

# Editorial emphasis. Each entry names a cell that carries \best in the
# manuscript. Numbers are never chosen here, only which cell is highlighted.
#
# clean_tracks / currents: the better completion outcome where the two reference
#   controllers differ.
# fleet: the recommended coordination policy, LF(1). At gap 0 s the uncoordinated
#   convoy is nominally faster but contacts the gates, so the emphasis marks the
#   fastest *contact-free* policy rather than the fastest policy.
EMPHASIS = {
    "clean_tracks": {("vertical", SERVO): ("finished", "gates"),
                     ("mixed", CTC): ("finished", "gates")},
    "currents": {("medium", CTC): ("finished", "gates", "official", "penalized", "collisions")},
    "fleet": {("gap_0", "lf1"): ("time", "events"),
              ("gap_8", "lf1"): ("finished", "gates", "time", "events")},
}


# --------------------------------------------------------------------------- #
# artifact readers
# --------------------------------------------------------------------------- #
def _runs() -> list[dict]:
    path = PAPER / "benchmark" / "runs.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _num(value):
    if value is None or value == "":
        return None
    return float(value)


def _single(rows: list[dict], **filters) -> list[dict]:
    out = []
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            out.append(row)
    if not out:
        raise SystemExit("no rows for {}".format(filters))
    return out


# --------------------------------------------------------------------------- #
# statistics helpers
# --------------------------------------------------------------------------- #
def _mean1(values) -> str:
    values = [v for v in values if v is not None]
    return "\\textemdash" if not values else "{:.1f}".format(statistics.fmean(values))


def _pm(values, places: int = 1) -> str:
    values = [v for v in values if v is not None]
    if not values:
        return "\\textemdash"
    mean = statistics.fmean(values)
    if len(values) < 2:
        return "${:.{p}f}$".format(mean, p=places)
    return "${:.{p}f}\\pm{:.{p}f}$".format(mean, statistics.stdev(values), p=places)


def _best(text: str, on: bool = True) -> str:
    return "\\best{{{}}}".format(text) if on else text


# --------------------------------------------------------------------------- #
# tab:clean_tracks
# --------------------------------------------------------------------------- #
def clean_tracks(rows: list[dict]) -> str:
    def cells(track: str, controller: str) -> tuple[str, str, str, str, str]:
        sel = _single(rows, experiment="clean", track=TRACK_FULL[track],
                      controller=controller, current_profile="none")
        finished = [r for r in sel if r["status"] == "FINISHED"]
        marks = EMPHASIS["clean_tracks"].get((track, controller), ())
        n = len(sel)
        fin = _best("{}/{}".format(len(finished), n), "finished" in marks)
        gates = _best("{}/{}".format(_mean1([_num(r["completed_gates"]) for r in sel]),
                                     EXPECTED_GATES[track]), "gates" in marks)
        time = _pm([_num(r["official_time_s"]) for r in finished])
        coll = _mean1([_num(r["gate_world_collisions"]) for r in sel])
        return str(n), fin, gates, time, coll

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Controller comparison on the three official circuits under clean water.}",
        r"\label{tab:clean_tracks}",
        r"\mratablestyle",
        r"\begin{tabular}{@{}llrlrrr@{}}",
        r"\toprule",
        r"Track & Controller & Seeds & Finished & Gates $\uparrow$ & Official time (s) $\downarrow$ & Collisions $\downarrow$ \\",
        r"\midrule",
    ]
    for index, track in enumerate(("horseshoe", "vertical", "mixed")):
        if index:
            lines.append(r"\midrule")
        lines.append(r"\multirow{2}{*}{" + TRACK_LABEL[track] + "}")
        for controller, label in ((SERVO, "Continuous Servo   "),
                                  (CTC, "Center-then-Commit ")):
            n, fin, gates, time, coll = cells(track, controller)
            lines.append(" & {} & {} & {} & {} & {} & {} \\\\".format(
                label, n, fin, gates, time, coll))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\mratablenote",
        r"Collisions combine gate and world events; bold marks the better completion result",
        r"where the controllers differ.",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# tab:currents
# --------------------------------------------------------------------------- #
def currents(rows: list[dict]) -> str:
    def cells(profile: str, controller: str):
        if profile == "none":
            sel = _single(rows, experiment="clean", track=TRACK_FULL["horseshoe"],
                          controller=controller, current_profile="none")
        else:
            sel = _single(rows, experiment="currents", track=TRACK_FULL["horseshoe"],
                          controller=controller, current_profile=profile)
        finished = [r for r in sel if r["status"] == "FINISHED"]
        marks = EMPHASIS["currents"].get((profile, controller), ())
        return {
            "n": str(len(sel)),
            "finished": _best("{}/{}".format(len(finished), len(sel)), "finished" in marks),
            "gates": _best("{}/12".format(_mean1([_num(r["completed_gates"]) for r in sel])),
                           "gates" in marks),
            "official": _best(_pm([_num(r["official_time_s"]) for r in finished]),
                              "official" in marks),
            "penalized": _best(_pm([_num(r["penalized_time_s"]) for r in finished]),
                               "penalized" in marks),
            "collisions": _best(_mean1([_num(r["gate_world_collisions"]) for r in sel]),
                                "collisions" in marks),
        }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Controller performance on Horseshoe Bay with and without current.}",
        r"\label{tab:currents}",
        r"\mratablestyle",
        r"\begin{tabular}{@{}llrlrrrr@{}}",
        r"\toprule",
        r"Profile & Controller & Seeds & Finished & Gates $\uparrow$ & Official (s) & Penalized (s) & Collisions $\downarrow$ \\",
        r"\midrule",
    ]
    blocks = (("none", r"\multirow{2}{*}{Clean}"),
              ("medium", r"\multirow{2}{*}{\texttt{medium}}"))
    for index, (profile, header) in enumerate(blocks):
        if index:
            lines.append(r"\midrule")
        lines.append(header)
        for controller, label in ((SERVO, "Continuous Servo  "),
                                  (CTC, "Center-then-Commit")):
            c = cells(profile, controller)
            lines.append(" & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                label, c["n"], c["finished"], c["gates"], c["official"],
                c["penalized"], c["collisions"]))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\mratablenote",
        r"Times are mean $\pm$ sample standard deviation; collisions combine gate and world",
        r"events.",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# tab:fleet
# --------------------------------------------------------------------------- #
def fleet(rows: list[dict]) -> str:
    def team(sel: list[dict], marks: tuple = ()) -> str:
        finished = [r for r in sel if r["all_rovers_finished"] == "True"]
        gates_total = 24 if sel[0]["experiment"] == "fleet_gap90" else 36
        fin = _best("{}/{}".format(len(finished), len(sel)), "finished" in marks)
        gates = _best("{}/{}".format(_mean1([_num(r["completed_gates"]) for r in sel]),
                                     gates_total), "gates" in marks)
        time = _best(_pm([_num(r["team_elapsed_time_s"]) for r in finished]), "time" in marks)
        events = _best("{} / {} / {}".format(
            _mean1([_num(r["gate_world_collisions"]) for r in sel]),
            _mean1([_num(r["proximity_events"]) for r in sel]),
            _mean1([_num(r["stuck_events"]) for r in sel])), "events" in marks)
        return "{} & {} & {} & {} & {}".format(len(sel), fin, gates, time, events)

    def homogeneous(controller: str) -> list[dict]:
        pair = "{}; {}".format(controller, controller)
        return _single(rows, experiment="fleet_gap90", controller=pair)

    def convoy(gap: str, condition: str, min_gap: str) -> list[dict]:
        return _single(rows, experiment="coordination", start_gap_s=gap,
                       condition=condition, min_gate_gap_configured=min_gap)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Multi-vehicle evaluation on clean Horseshoe Bay.}",
        r"\label{tab:fleet}",
        r"\mratablestyle",
        r"\begin{tabular}{@{}llrlrrr@{}}",
        r"\toprule",
        r"Setting & Policy & Seeds & Team finish & Team gates $\uparrow$ & Team time (s) $\downarrow$ & GW / IV / S $\downarrow$ \\",
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\emph{Two vehicles, homogeneous, gap $90$\,s}}\\",
        " & Continuous Servo   & " + team(homogeneous(SERVO)) + r" \\",
        " & Center-then-Commit & " + team(homogeneous(CTC)) + r" \\",
    ]
    for gap_value, gap_key, gap_text in (("0.0", "gap_0", "0"), ("8.0", "gap_8", "8")):
        lines += [
            r"\midrule",
            r"\multicolumn{7}{@{}l}{\emph{Three vehicles, heterogeneous convoy, gap $"
            + gap_text + r"$\,s}}\\",
            " & No coordination & "
            + team(convoy(gap_value, "no_coordination", "2")) + r" \\",
            " & LF(1)           & "
            + team(convoy(gap_value, "leader_follower", "1"),
                   EMPHASIS["fleet"].get((gap_key, "lf1"), ())) + r" \\",
            " & LF(2)           & "
            + team(convoy(gap_value, "leader_follower", "2")) + r" \\",
        ]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\mratablenote",
        r"GW / IV / S denotes gate and world collisions, inter-vehicle proximity and stuck",
        r"events. LF($\Delta g$) is the leader--follower policy of \eqref{eq:yield}.",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# tab:current_free
# --------------------------------------------------------------------------- #
def current_free() -> str:
    summary = json.loads((PAPER / "current_free" / "summary.json").read_text(encoding="utf-8"))
    by_circuit = {c["circuit"]: c for c in summary["circuits"]}
    labels = (("horseshoe", "Horseshoe Bay   "),
              ("vertical", "Vertical Serpent"),
              ("mixed", "Mixed Endurance "))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Current-free completion on the official circuits.}",
        r"\label{tab:current_free}",
        r"\mratablestyle",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"Circuit & Gates & Runs & Completed & Mean gates & Official time (s) & Coll. & OOB / WD \\",
        r"\midrule",
    ]
    total_gates = total_runs = total_done = total_coll = total_oob = total_wd = 0
    widths = []
    body = []
    for key, label in labels:
        circuit = by_circuit[key]
        path = PAPER / "current_free" / ("circuit_" + key) / "eval_results.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            times = [float(r["official_time_s"]) for r in csv.DictReader(stream)]
        expected = circuit["expected_gates"]
        total_gates += expected
        total_runs += circuit["n_eval"]
        total_done += circuit["completions"]
        total_coll += circuit["total_collisions"]
        total_oob += circuit["total_out_of_bounds"]
        total_wd += circuit["total_wrong_direction_crossings"]
        time = _pm(times)
        widths.append(len(time))
        body.append((label, expected, circuit["n_eval"], circuit["completions"],
                     circuit["mean_gates"], time, circuit["total_collisions"],
                     circuit["total_out_of_bounds"],
                     circuit["total_wrong_direction_crossings"]))
    pad = max(widths)
    for (label, expected, n, done, mean_gates, time, coll, oob, wd) in body:
        lines.append(
            " & ".join([label, str(expected), str(n), "{}/{}".format(done, n),
                        "{:.1f}/{}".format(mean_gates, expected), time.ljust(pad),
                        str(coll), "{} / {}".format(oob, wd)]) + r" \\")
    lines += [
        r"\midrule",
        " & ".join([r"\textbf{Total}  ", _best(str(total_gates)), _best(str(total_runs)),
                    _best("{}/{}".format(total_done, total_runs)),
                    _best("{:.1f}/{}".format(float(total_gates), total_gates)),
                    r"\textemdash", _best(str(total_coll)),
                    _best("{} / {}".format(total_oob, total_wd))]) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\mratablenote",
        r"Times are mean $\pm$ sample standard deviation; Coll. denotes collision events,",
        r"and OOB / WD denotes out-of-bounds and wrong-direction events.",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# tab:learning
# --------------------------------------------------------------------------- #
def _ppo_episodes() -> list[dict]:
    episodes = []
    for path in sorted((PAPER / "ppo" / "validation").glob("*_attempt*.result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("technical_status") == "VALID":
            episodes.append(row)
    return episodes


def learning() -> str:
    episodes = _ppo_episodes()
    by_track = {}
    for row in episodes:
        by_track.setdefault(row["track"], []).append(row)
    labels = (("horseshoe_bay", "Horseshoe Bay", 12),
              ("vertical_serpent", "Vertical Serpent", 17),
              ("mixed_endurance", "Mixed Endurance", 22))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Recurrent PPO validation results.}",
        r"\label{tab:learning}",
        r"\mratablestyle",
        r"\begin{tabular}{@{}l r r c c r r@{}}",
        r"\toprule",
        r"Circuit & Gates & Episodes & Finished & Mean time (s) & Collisions & OOB \\",
        r"\midrule",
    ]
    total_coll = total_oob = total_n = 0
    for key, label, gates in labels:
        rows = by_track[key]
        finished = [r for r in rows if r["finished"]]
        mean_time = statistics.fmean(r["completion_time_s"] for r in finished)
        coll = sum(r["collisions"] for r in rows)
        oob = sum(r["out_of_bounds_events"] for r in rows)
        total_coll += coll
        total_oob += oob
        total_n += len(rows)
        lines.append("{} & {} & {} & {}/{} & {:.1f} & {} & {} \\\\".format(
            label, gates, len(rows), len(finished), len(rows), mean_time, coll, oob))
    lines += [
        r"\midrule",
        "All circuits & \\textemdash & {} & {}/{} & \\textemdash & {} & {} \\\\".format(
            total_n, sum(1 for r in episodes if r["finished"]), total_n,
            total_coll, total_oob),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# tab:local_vs_referee
# --------------------------------------------------------------------------- #
def local_vs_referee() -> str:
    data = json.loads((PAPER / "benchmark" / "local_vs_referee.json").read_text(encoding="utf-8"))
    labels = (("Marine Race Horseshoe Bay", "Horseshoe Bay   "),
              ("Marine Race Vertical Serpent", "Vertical Serpent"),
              ("Marine Race Mixed Endurance", "Mixed Endurance "))

    def row(block: dict) -> tuple:
        delay = block["advancement_delay_s"]
        return (block["referee_advancements"], block["local_advancements"],
                block["matched_advancements"], block["false_local_advancements"],
                block["missed_local_advancements"], delay["median"], delay["p95"])

    body = [(label,) + row(data["by_track"][key]) for key, label in labels]
    widths = [max(len(str(r[i])) for r in body) for i in range(1, 8)]
    total = ("\\textbf{All}    ",) + row(data["overall"])
    widths = [max(w, len(str(total[i + 1]))) for i, w in enumerate(widths)]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Controller-local and referee course progression.}",
        r"\label{tab:local_vs_referee}",
        r"\mratablestyle",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Gate advancements} & \multicolumn{2}{c}{Mismatches} & \multicolumn{2}{c}{Local delay (s)} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
        r"Circuit & referee & local & matched & false & missed & median & p95 \\",
        r"\midrule",
    ]
    for entry in body:
        cells = [str(v).rjust(widths[i]) for i, v in enumerate(entry[1:])]
        lines.append(entry[0] + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    cells = [_best(str(v)) for v in total[1:]]
    lines.append(total[0] + " & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\mratablenote",
        r"False denotes a local advancement not matched by the corresponding referee",
        r"crossing; missed denotes a referee crossing without the corresponding local",
        r"advancement.",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# tab:perception_audit
# --------------------------------------------------------------------------- #
def perception_audit() -> str:
    report = json.loads((PAPER / "perception" / "visual_smoke_report.json")
                        .read_text(encoding="utf-8"))
    metrics = {t["track"]: t["metrics"] for t in report["tracks"]}
    labels = (("horseshoe_bay", "Horseshoe Bay   "),
              ("vertical_serpent", "Vertical Serpent"),
              ("mixed_endurance", "Mixed Endurance "))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Perception audit on the official circuits.}",
        r"\label{tab:perception_audit}",
        r"\mratablestyle",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Circuit & Detection & 4-corner & Pose avail. & Multi-candidate association \\",
        r"\midrule",
    ]
    for key, label in labels:
        m = metrics[key]
        frames = m["multi_candidate_frames"]
        matched = round(m["multi_candidate_association_within_0_25"] * frames)
        lines.append("{} & {:.1f}\\% & {:.1f}\\% & {:.1f}\\% & {}/{} \\\\".format(
            label, 100 * m["detection_availability"],
            100 * m["valid_four_corner_availability"],
            100 * m["metric_pnp_availability"], matched, frames))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# penalty identity, checked on the released rows
# --------------------------------------------------------------------------- #
def penalty_identity(rows: list[dict]) -> list[str]:
    problems = []
    for row in rows:
        if row["kind"] != "benchmark":
            continue
        if row["team_elapsed_time_s"]:
            if row["all_rovers_finished"] != "True":
                continue
            base = _num(row["team_elapsed_time_s"])
            done = _num(row["team_penalized_time_s"])
        else:
            if row["status"] != "FINISHED":
                continue
            base = _num(row["official_time_s"])
            done = _num(row["penalized_time_s"])
        extra = _num(row["penalties_s"]) or 0.0
        if base is None or done is None:
            continue
        if abs((base + extra) - done) > 0.05:
            problems.append("{}: base+penalties={:.3f} != penalized={:.3f}".format(
                row["run_id"], base + extra, done))
    return problems


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="print the tables and the penalty check, write nothing")
    args = parser.parse_args()

    if not PAPER.is_dir():
        raise SystemExit("missing evidence package: {}".format(PAPER))

    rows = _runs()
    problems = penalty_identity(rows)
    print("[penalty identity] {} finished runs checked, {} discrepancies".format(
        sum(1 for r in rows if r["status"] == "FINISHED"
            or r["all_rovers_finished"] == "True"), len(problems)))
    for problem in problems:
        print("   ", problem)

    generated = {
        "clean_tracks": clean_tracks(rows),
        "currents": currents(rows),
        "fleet": fleet(rows),
        "current_free": current_free(),
        "learning": learning(),
        "local_vs_referee": local_vs_referee(),
        "perception_audit": perception_audit(),
    }

    if args.check:
        changed = []
        for name, body in generated.items():
            target = TABLES / (name + ".tex")
            current = target.read_text(encoding="utf-8") if target.exists() else ""
            if current.replace("\r\n", "\n") != body.replace("\r\n", "\n"):
                changed.append(name)
        for name, body in generated.items():
            print("\n===== {}.tex =====\n".format(name))
            print(body)
        print("tables differing from the committed version: {}".format(changed or "none"))
        return 1 if (changed or problems) else 0

    for name, body in generated.items():
        target = TABLES / (name + ".tex")
        with target.open("w", encoding="utf-8", newline=LF) as stream:
            stream.write(body)
        print("wrote {}".format(target.relative_to(ROOT).as_posix()))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
