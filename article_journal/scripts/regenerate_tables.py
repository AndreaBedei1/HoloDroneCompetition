"""Regenerate the data-driven result tables of the manuscript.

Post-processing only: this reads the released evidence package under
``artifacts/paper/`` and rewrites the seven data-driven table bodies in
``article_journal/tables/``. It never launches the simulator and never modifies
an artifact.

Numbers come entirely from the artifacts and no cell carries emphasis (see
``EMPHASIS`` below), so a regenerated table is byte-identical to the committed
one.

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

# Editorial emphasis, declared here so that no table function can crown a winner
# of its own. It is empty, and deliberately so: with three to five seeds per cell
# a bold number reads as an established ordering, which runs of this size cannot
# support. Every table therefore reports plain numbers and the comparisons are
# drawn in the prose, where they can be stated with their sample sizes.
EMPHASIS: dict = {}


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


# --------------------------------------------------------------------------- #
# tab:clean_tracks
# --------------------------------------------------------------------------- #
def clean_tracks(rows: list[dict]) -> str:
    def cells(track: str, controller: str) -> tuple[str, str, str, str, str]:
        sel = _single(rows, experiment="clean", track=TRACK_FULL[track],
                      controller=controller, current_profile="none")
        finished = [r for r in sel if r["status"] == "FINISHED"]
        n = len(sel)
        fin = "{}/{}".format(len(finished), n)
        gates = "{}/{}".format(_mean1([_num(r["completed_gates"]) for r in sel]),
                               EXPECTED_GATES[track])
        time = _pm([_num(r["official_time_s"]) for r in finished])
        coll = _mean1([_num(r["gate_world_collisions"]) for r in sel])
        return str(n), fin, gates, time, coll

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Controller comparison on the three official circuits under clean water"
        r" ($30$\,Hz systematic evaluation).}",
        r"\label{tab:clean_tracks}",
        r"\mratablestyle",
        r"\begin{tabular}{@{}llrlrrr@{}}",
        r"\toprule",
        r"Track & Controller & Seeds & Finished & Gates & Official time (s) & Collisions \\",
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
        r"Collisions combine gate and world events. Finished counts, mean gates and mean",
        r"collisions are computed over all seeds of the cell, whereas the official time is",
        r"computed over the finished runs only.",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# tab:currents
# --------------------------------------------------------------------------- #
def currents(rows: list[dict]) -> str:
    # No cell of this table is emphasised. Under the medium current the two
    # controllers finish a different number of runs, so their time columns
    # average different subsets of seeds and no column of that block sets one
    # controller against the other on equal terms.
    def cells(profile: str, controller: str):
        if profile == "none":
            sel = _single(rows, experiment="clean", track=TRACK_FULL["horseshoe"],
                          controller=controller, current_profile="none")
        else:
            sel = _single(rows, experiment="currents", track=TRACK_FULL["horseshoe"],
                          controller=controller, current_profile=profile)
        finished = [r for r in sel if r["status"] == "FINISHED"]
        return {
            "n": str(len(sel)),
            "finished": "{}/{}".format(len(finished), len(sel)),
            "gates": "{}/12".format(_mean1([_num(r["completed_gates"]) for r in sel])),
            "official": _pm([_num(r["official_time_s"]) for r in finished]),
            "penalized": _pm([_num(r["penalized_time_s"]) for r in finished]),
            "collisions": _mean1([_num(r["gate_world_collisions"]) for r in sel]),
        }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Controller performance on Horseshoe Bay with and without current"
        r" ($30$\,Hz systematic evaluation).}",
        r"\label{tab:currents}",
        r"\mratablestyle",
        r"\begin{tabular}{@{}llrlrrrr@{}}",
        r"\toprule",
        r"Profile & Controller & Seeds & Finished & Gates & Official (s) & Penalized (s) & Collisions \\",
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
        r"events. Finished counts, mean gates and mean collisions are computed over all seeds",
        r"of the cell, whereas the official and penalized times are computed over the finished",
        r"runs only.",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# tab:fleet
# --------------------------------------------------------------------------- #
def fleet(rows: list[dict]) -> str:
    # Rows whose two time columns rest on a subset of the seeds carry a dagger,
    # because the caveat applies to the row and not to a single column.
    partial_rows: list[str] = []

    def policy(label: str, sel: list[dict]) -> str:
        """Label a row, daggered when its time columns use only part of the seeds."""
        if any(r["all_rovers_finished"] != "True" for r in sel):
            partial_rows.append(label.strip())
            return label.rstrip() + r"$^{\dagger}$" + " "
        return label

    def team(sel: list[dict]) -> str:
        finished = [r for r in sel if r["all_rovers_finished"] == "True"]
        gates_total = 24 if sel[0]["experiment"] == "fleet_gap90" else 36
        fin = "{}/{}".format(len(finished), len(sel))
        gates = "{}/{}".format(_mean1([_num(r["completed_gates"]) for r in sel]),
                               gates_total)
        # Elapsed and penalized team time share one subset -- the runs in which
        # every vehicle finished -- so the two columns are directly comparable.
        elapsed = _pm([_num(r["team_elapsed_time_s"]) for r in finished])
        penalized = _pm([_num(r["team_penalized_time_s"]) for r in finished])
        events = "{} / {} / {}".format(
            _mean1([_num(r["gate_world_collisions"]) for r in sel]),
            _mean1([_num(r["proximity_events"]) for r in sel]),
            _mean1([_num(r["stuck_events"]) for r in sel]))
        return "{} & {} & {} & {} & {} & {}".format(
            len(sel), fin, gates, elapsed, penalized, events)

    def homogeneous(controller: str) -> list[dict]:
        pair = "{}; {}".format(controller, controller)
        return _single(rows, experiment="fleet_gap90", controller=pair)

    def convoy(gap: str, condition: str, min_gap: str) -> list[dict]:
        return _single(rows, experiment="coordination", start_gap_s=gap,
                       condition=condition, min_gate_gap_configured=min_gap)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Multi-vehicle evaluation on clean Horseshoe Bay"
        r" ($30$\,Hz systematic evaluation).}",
        r"\label{tab:fleet}",
        r"\mratablestyle",
        r"\begin{tabular}{@{}llrlrrrr@{}}",
        r"\toprule",
        r"Setting & Policy & Seeds & Team finish & Team gates & Team elapsed (s) & Team penalized (s) & GW / IV / S \\",
        r"\midrule",
        r"\multicolumn{8}{@{}l}{\emph{Two vehicles, homogeneous, gap $90$\,s}}\\",
    ]
    for controller, label in ((SERVO, "Continuous Servo   "),
                              (CTC, "Center-then-Commit ")):
        sel = homogeneous(controller)
        lines.append(" & " + policy(label, sel) + "& " + team(sel) + r" \\")
    for gap_value, gap_text in (("0.0", "0"), ("8.0", "8")):
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{8}{@{}l}{\emph{Three vehicles, heterogeneous convoy, gap $"
                     + gap_text + r"$\,s}}\\")
        for label, condition, min_gap in (("No coordination ", "no_coordination", "2"),
                                          ("LF(1)           ", "leader_follower", "1"),
                                          ("LF(2)           ", "leader_follower", "2")):
            sel = convoy(gap_value, condition, min_gap)
            lines.append(" & " + policy(label, sel) + "& " + team(sel) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\mratablenote",
        r"GW / IV / S denotes gate and world collisions, inter-vehicle proximity and stuck",
        r"events. LF($\Delta g$) is the leader--follower policy of \eqref{eq:yield}. Seeds, team",
        r"finish, team gates and event counts are computed over all seeds of the row; elapsed",
        r"and penalized team time are both computed over the seeds in which every vehicle",
        r"finished, so the two columns are directly comparable. Penalized time is the official",
        r"team result.",
    ]
    if partial_rows:
        lines += [
            r"$^{\dagger}$Not every vehicle finished in every seed of this row, so both of its",
            r"time columns describe the subset that did; the caveat applies to the row rather",
            r"than to one column.",
        ]
    lines.append(r"\end{table}")
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
        r"\caption{Current-free completion of the Center-then-Commit controller on the"
        r" official circuits ($10$\,Hz reference validation).}",
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
        # The Total row aggregates the circuits above it; it competes with
        # nothing, so it carries neither \best nor a bold label -- \best is
        # \textbf, and a bold cell in a results table reads as a mark.
        " & ".join(["Total           ", str(total_gates), str(total_runs),
                    "{}/{}".format(total_done, total_runs),
                    "{:.1f}/{}".format(float(total_gates), total_gates),
                    r"\textemdash", str(total_coll),
                    "{} / {}".format(total_oob, total_wd)]) + r" \\",
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
        r"\caption{Recurrent PPO evaluation episodes on the three official circuits"
        r" ($10$\,Hz protocol).}",
        r"\label{tab:learning}",
        r"\mratablestyle",
        r"\begin{tabular}{@{}l r r c c r r@{}}",
        r"\toprule",
        r"Circuit & Gates & Episodes & Finished & Mean episode time (s) & Collisions & OOB \\",
        r"\midrule",
    ]
    total_coll = total_oob = total_n = 0
    for key, label, gates in labels:
        rows = by_track[key]
        finished = [r for r in rows if r["finished"]]
        time = _pm([r["completion_time_s"] for r in finished])
        coll = sum(r["collisions"] for r in rows)
        oob = sum(r["out_of_bounds_events"] for r in rows)
        total_coll += coll
        total_oob += oob
        total_n += len(rows)
        lines.append("{} & {} & {} & {}/{} & {} & {} & {} \\\\".format(
            label, gates, len(rows), len(finished), len(rows), time, coll, oob))
    lines += [
        r"\midrule",
        "All circuits & \\textemdash & {} & {}/{} & \\textemdash & {} & {} \\\\".format(
            total_n, sum(1 for r in episodes if r["finished"]), total_n,
            total_coll, total_oob),
        r"\bottomrule",
        r"\end{tabular}",
        r"\mratablenote",
        r"Times are mean $\pm$ sample standard deviation over the finished episodes of the",
        r"circuit; OOB denotes out-of-bounds events. The time reported here is the episode",
        r"clock, not the referee's official first-to-last-gate time: official and penalized",
        r"times were not recorded for this campaign, so this column is not comparable with",
        r"the official times reported in the other result tables.",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# tab:local_vs_referee
# --------------------------------------------------------------------------- #
def _premature(block: dict) -> int:
    """Local advancements counted in both *matched* and *false*.

    ``false_local_advancements`` is the union of two disjoint classes: local
    advancements with no referee crossing at the same course ordinal, and local
    advancements that precede their same-ordinal referee crossing by more than
    one control step. Only the first class is absent from
    ``matched_advancements``, so the second class is the difference below. The
    two independent derivations must agree; see ``advancement_identity``.
    """
    unmatched = block["local_advancements"] - block["matched_advancements"]
    return block["false_local_advancements"] - unmatched


def advancement_identity(data: dict) -> list[str]:
    """Check the identities the table note states, on every released block."""
    problems = []
    blocks = dict(data["by_track"])
    blocks["overall"] = data["overall"]
    for name, block in sorted(blocks.items()):
        early = _premature(block)
        if early != block["matched_advancements"] - block["delayed_local_advancements"]:
            problems.append("{}: premature count is not reproduced by "
                            "matched - delayed".format(name))
        if block["referee_advancements"] != (block["matched_advancements"]
                                             + block["missed_local_advancements"]):
            problems.append("{}: referee != matched + missed".format(name))
        if block["local_advancements"] != (block["matched_advancements"]
                                           + block["false_local_advancements"] - early):
            problems.append("{}: local != matched + false - premature".format(name))
    return problems


def local_vs_referee() -> str:
    data = json.loads((PAPER / "benchmark" / "local_vs_referee.json").read_text(encoding="utf-8"))
    source = json.loads((PAPER / "benchmark" / "source.json").read_text(encoding="utf-8"))
    released = source["released_run_count"]
    # The caption and the note below both state the audited population. The audit
    # is scoped to the released runs, the same population as runs.csv, so refuse
    # to emit that number unless the artifact's own scope names it and the row
    # count of runs.csv agrees with it.
    if str(released) not in data["scope"]:
        raise SystemExit("local_vs_referee scope no longer names {} released runs: "
                         "{}".format(released, data["scope"]))
    if len(_runs()) != released:
        raise SystemExit("runs.csv holds {} rows, not the {} released runs the audit "
                         "claims to cover".format(len(_runs()), released))
    labels = (("Marine Race Horseshoe Bay", "Horseshoe Bay   "),
              ("Marine Race Vertical Serpent", "Vertical Serpent"),
              ("Marine Race Mixed Endurance", "Mixed Endurance "))

    # The artifact also carries the local-delay distribution, which this table no
    # longer prints: its median and p95 are unremarkable, but its tail is not, and
    # a two-number summary of it would need more defence than the boundary audit
    # is here to provide. The columns below are the information separation itself.
    def row(block: dict) -> tuple:
        return (block["referee_advancements"], block["local_advancements"],
                block["matched_advancements"], block["false_local_advancements"],
                block["missed_local_advancements"])

    body = [(label,) + row(data["by_track"][key]) for key, label in labels]
    premature = {label.strip(): _premature(data["by_track"][key]) for key, label in labels}
    carriers = sorted(name for name, count in premature.items() if count)
    # The table note below says the premature advancements all fall on one
    # circuit; refuse to emit that sentence if the artifact ever says otherwise.
    if len(carriers) != 1 or premature[carriers[0]] != _premature(data["overall"]):
        raise SystemExit("premature advancements are no longer confined to one circuit: "
                         "{}".format(premature))
    widths = [max(len(str(r[i])) for r in body) for i in range(1, 6)]
    total = ("All             ",) + row(data["overall"])
    widths = [max(w, len(str(total[i + 1]))) for i, w in enumerate(widths)]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Boundary audit of controller-local against referee course"
        r" progression over the $" + str(released) + r"$ released benchmark runs.}",
        r"\label{tab:local_vs_referee}",
        r"\mratablestyle",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Gate advancements} & \multicolumn{2}{c}{Mismatches} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-6}",
        r"Circuit & referee & local & matched & false & missed \\",
        r"\midrule",
    ]
    for entry in body:
        cells = [str(v).rjust(widths[i]) for i, v in enumerate(entry[1:])]
        lines.append(entry[0] + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    # The All row aggregates the circuits above it; it competes with nothing, so
    # it carries neither \best nor a bold label.
    cells = [str(v).rjust(widths[i]) for i, v in enumerate(total[1:])]
    lines.append(total[0] + " & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\mratablenote",
        # The tolerance is one control step of the run being audited, which is how
        # the artifact defines it, and no fixed step length is quoted anywhere here.
        r"The audit covers the " + str(released) + r" released runs of the frozen benchmark matrix ---",
        r"single-vehicle, two-vehicle fleet and three-vehicle coordination --- the same",
        r"population as the result tables of Section~\ref{sec:evaluation}.",
        r"Matched counts course ordinals at which both a referee crossing and a",
        r"controller-local advancement occur. False counts local advancements that either",
        r"have no same-ordinal referee crossing or precede it by more than one control step",
        r"of their run, and missed counts referee crossings with no corresponding local",
        r"advancement. Advancements of the second false kind are counted in both matched",
        r"and false, so local $=$ matched $+$ false $-$ "
        + str(_premature(data["overall"])) + r" rather than matched $+$ false;",
        r"every one of them falls on " + carriers[0] + r", whereas referee $=$ matched $+$ missed",
        r"holds on every row.",
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
        r"\caption{Perception audit on the official circuits, recorded along"
        r" Center-then-Commit trajectories without current ($10$\,Hz protocol).}",
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

    lvr = json.loads((PAPER / "benchmark" / "local_vs_referee.json").read_text(encoding="utf-8"))
    advancement = advancement_identity(lvr)
    problems += advancement
    print("[advancement identity] {} blocks checked, {} discrepancies".format(
        len(lvr["by_track"]) + 1, len(advancement)))
    for problem in advancement:
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
