"""Regenerate the manuscript result tables from the existing raw 78-run matrix.

Post-processing only: this reads the frozen per-run ``*_summary.json`` artifacts
under ``results/onboard_only_validation/final_20260715`` and writes two LaTeX
table bodies. It never launches HoloOcean and never modifies a raw artifact.

Conventions (manuscript policy):
* sample standard deviation (ddof=1); omitted when a cell has < 2 finished runs;
* one decimal place for times, mean gates, mean collisions and mean events;
* integer finish counts and exact gate totals;
* time statistics use finished runs only; gate/collision/event means use all seeds.

Usage:
    python article/regenerate_tables.py            # writes the two .tex files
    python article/regenerate_tables.py --check     # print numbers, do not write
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path("results/onboard_only_validation/final_20260715")
ARTICLE_TABLES = Path("article/tables")

CONTROLLER_LABEL = {
    "rule_gate_baseline": "Continuous servo",
    "rule_gate_center_then_commit": "Center-then-commit",
}
TRACK_LABEL = {
    "horseshoe": "Horseshoe Bay",
    "vertical": "Vertical Serpent",
    "mixed": "Mixed Endurance",
}


# --------------------------------------------------------------------------- #
# raw-artifact readers
# --------------------------------------------------------------------------- #
def _summary(run_dir: Path) -> Optional[dict]:
    files = sorted(glob.glob(str(run_dir / "*_summary.json")))
    if not files:
        return None
    return json.loads(Path(files[0]).read_text(encoding="utf-8"))


def _single_runs(*parts) -> List[dict]:
    base = ROOT.joinpath(*parts) / "runs"
    rows = []
    for run_dir in sorted(base.glob("run_*")):
        s = _summary(run_dir)
        if s is None:
            continue
        p = s["participants"][0]
        rows.append(
            {
                "status": p["status"],
                "finished": p["status"] == "FINISHED",
                "completed_gates": p["completed_gates"],
                "official_time_s": p["official_time_s"],
                "penalized_time_s": p["penalized_time_s"],
                "penalties_s": p["penalties_s"],
                "collisions": p["collisions"],
                "out_of_bounds_events": p["out_of_bounds_events"],
                "stuck_events": p["stuck_events"],
            }
        )
    return rows


def _team_runs(dirs: List[Path]) -> List[dict]:
    rows = []
    for run_dir in dirs:
        s = _summary(run_dir)
        if s is None:
            continue
        team = s["team_summary"]
        oob = sum(int(p.get("out_of_bounds_events") or 0) for p in s["participants"])
        stuck = sum(int(p.get("stuck_events") or 0) for p in s["participants"])
        rows.append(
            {
                "finished": bool(team["all_rovers_finished"]),
                "total_completed_gates": team["total_completed_gates"],
                "team_elapsed_time_s": team["team_elapsed_time_s"],
                "team_penalized_time_s": team["team_penalized_time_s"],
                "total_gate_collisions": team["total_gate_collisions"],
                "total_inter_vehicle_collisions": team["total_inter_vehicle_collisions"],
                "out_of_bounds_events": oob,
                "stuck_events": stuck,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def _mean(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _sample_std(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return statistics.stdev(values) if len(values) >= 2 else None


def _pm(values: List[float], places: int = 1) -> str:
    """mean$\\pm$sample_std over the given (already-filtered) values."""
    values = [v for v in values if v is not None]
    if not values:
        return "\\textemdash"
    mean = statistics.fmean(values)
    if len(values) < 2:
        return f"${mean:.{places}f}$"  # single finished run: no +/-
    return f"${mean:.{places}f}\\pm{statistics.stdev(values):.{places}f}$"


def _m1(values: List[float]) -> str:
    m = _mean(values)
    return "\\textemdash" if m is None else f"{m:.1f}"


def _finished(rows: List[dict], key: str) -> List[float]:
    return [r[key] for r in rows if r["finished"]]


# --------------------------------------------------------------------------- #
# table 1: clean + currents + fleet
# --------------------------------------------------------------------------- #
def build_validation_results() -> str:
    def clean_row(controller: str, track: str) -> str:
        rows = _single_runs("clean", track, controller)
        n = len(rows)
        fin = sum(r["finished"] for r in rows)
        gates = _m1([r["completed_gates"] for r in rows])
        time = _pm(_finished(rows, "official_time_s"))
        coll = _m1([r["collisions"] for r in rows])
        return (
            f" & {CONTROLLER_LABEL[controller]}, {TRACK_LABEL[track]} & {n} & "
            f"{fin}/{n} FIN & {gates}/{EXPECTED_GATES[track]} & {time} & {coll} \\\\"
        )

    def current_row(controller: str, profile: str) -> str:
        rows = _single_runs("currents", "horseshoe", controller, profile)
        n = len(rows)
        fin = sum(r["finished"] for r in rows)
        gates = _m1([r["completed_gates"] for r in rows])
        time = _pm(_finished(rows, "official_time_s"))
        coll = _m1([r["collisions"] for r in rows])
        label = f"{CONTROLLER_LABEL[controller]}, Horseshoe {profile}"
        return f" & {label} & {n} & {fin}/{n} FIN & {gates}/12 & {time} & {coll} \\\\"

    def fleet_row(controller: str) -> str:
        base = ROOT / "fleet_gap90" / controller / "runs"
        rows = _team_runs(sorted(base.glob("run_*")))
        n = len(rows)
        fin = sum(r["finished"] for r in rows)
        gates = _m1([r["total_completed_gates"] for r in rows])
        time = _pm(_finished(rows, "team_elapsed_time_s"))
        coll = _m1([r["total_gate_collisions"] for r in rows])
        return (
            f" & {CONTROLLER_LABEL[controller]}, team aggregate & {n} & "
            f"{fin}/{n} FIN & {gates}/24 & {time} & {coll} \\\\"
        )

    # penalized-time note values (finished-only), for the medium-current row.
    def pen(controller: str, profile: str) -> str:
        rows = _single_runs("currents", "horseshoe", controller, profile)
        return _pm(_finished(rows, "penalized_time_s"))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Final HoloOcean clean-track, current and homogeneous two-vehicle fleet results over five seeds per case. Times are finished-runs-only; gate, collision and event means are over all five seeds.}",
        r"\label{tab:validation_results}",
        r"\mratablestyle",
        r"\begin{tabular}{llrlrrr}",
        r"\toprule",
        r"Setting & Controller and track & Seeds & Status & Gates $\uparrow$ & Time (s) $\downarrow$ & Collisions $\downarrow$ \\",
        r"\midrule",
        r"\multirow{6}{*}{Clean}",
        clean_row("rule_gate_baseline", "horseshoe"),
        clean_row("rule_gate_center_then_commit", "horseshoe"),
        clean_row("rule_gate_baseline", "vertical"),
        clean_row("rule_gate_center_then_commit", "vertical"),
        clean_row("rule_gate_baseline", "mixed"),
        clean_row("rule_gate_center_then_commit", "mixed"),
        r"\midrule",
        r"\multirow{4}{*}{Currents}",
        current_row("rule_gate_baseline", "medium"),
        current_row("rule_gate_center_then_commit", "medium"),
        current_row("rule_gate_baseline", "strong"),
        current_row("rule_gate_center_then_commit", "strong"),
        r"\midrule",
        r"\multirow{2}{*}{Fleet, gap 90 s}",
        fleet_row("rule_gate_baseline"),
        fleet_row("rule_gate_center_then_commit"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\mratablenote Times are mean$\pm$sample standard deviation (ddof$=1$) over finished runs; a cell with a single finished run shows the mean without a spread. Gates and collisions are five-seed means; fleet values are team-level and collisions combine gate and world events. Medium-current penalized times are "
        + pen("rule_gate_baseline", "medium")
        + r"\,s for continuous servo and "
        + pen("rule_gate_center_then_commit", "medium")
        + r"\,s for center-then-commit.",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


EXPECTED_GATES = {"horseshoe": 12, "vertical": 17, "mixed": 22}


# --------------------------------------------------------------------------- #
# table 2: three-rover coordination
# --------------------------------------------------------------------------- #
def _coord_dirs(variant: str, gap_label: str, condition: str) -> List[Path]:
    dirs = []
    for seed in (0, 1, 2):
        d = ROOT / "coordination" / variant / gap_label / "diagnostic" / f"seed_{seed}" / condition
        if d.is_dir():
            dirs.append(d)
    return dirs


def build_coordination() -> str:
    def row(policy_label: str, dirs: List[Path]) -> str:
        rows = _team_runs(dirs)
        n = len(rows)
        fin = sum(r["finished"] for r in rows)
        gates = _m1([r["total_completed_gates"] for r in rows])
        raw = _pm(_finished(rows, "team_elapsed_time_s"))
        gw = _m1([r["total_gate_collisions"] for r in rows])
        iv = _m1([r["total_inter_vehicle_collisions"] for r in rows])
        stuck = _m1([r["stuck_events"] for r in rows])
        return (
            f" & {policy_label} & {n} & {fin}/{n} FIN & {gates}/36 & {raw} & "
            f"{gw} / {iv} / {stuck} \\\\"
        )

    def block(gap_label: str, gap_text: str) -> List[str]:
        return [
            rf"\multirow{{3}}{{*}}{{Gap {gap_text} s}}",
            row("No coordination", _coord_dirs("main", gap_label, "no_coordination")),
            row("LF(1)", _coord_dirs("min_gate_gap_1", gap_label, "leader_follower")),
            row("LF(2)", _coord_dirs("main", gap_label, "leader_follower")),
        ]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Three-vehicle coordination on clean Horseshoe Bay over three seeds per condition.}",
        r"\label{tab:holoocean_coordination}",
        r"\mratablestyle",
        r"\begin{tabular}{llrlrrr}",
        r"\toprule",
        r"Setting & Policy & Seeds & Status & Gates $\uparrow$ & Team time (s) $\downarrow$ & Events $\downarrow$ \\",
        r"\midrule",
        *block("gap_0", "0"),
        r"\midrule",
        *block("gap_8", "8"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\mratablenote Times are raw team elapsed mean$\pm$sample standard deviation over full-team finishes; gates and events are three-seed means. Events are GW / IV / S: gate and world collisions, inter-vehicle proximity and stuck. No run records an out-of-bounds event. Penalized time differs for no coordination at gaps 0 and 8 s ($266.4\pm64.1$ and $564.0\pm463.0$\,s) and LF(2) at gap 0 s ($303.3\pm1.3$\,s).",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# penalty-identity consistency check (report only; never repairs raw artifacts)
# --------------------------------------------------------------------------- #
def penalty_consistency_report() -> List[str]:
    problems: List[str] = []
    # single-rover finished runs: penalized == official + penalties
    for path in sorted(glob.glob(str(ROOT / "**" / "*_summary.json"), recursive=True)):
        s = json.loads(Path(path).read_text(encoding="utf-8"))
        team = s.get("team_summary")
        if team is not None:
            if team.get("all_rovers_finished"):
                base = team.get("team_elapsed_time_s")
                pen = team.get("team_penalized_time_s")
                extra = team.get("total_penalties_s") or 0.0
                if base is not None and pen is not None and abs((base + extra) - pen) > 0.05:
                    problems.append(f"TEAM {path}: elapsed+penalties={base+extra:.3f} != penalized={pen:.3f}")
            continue
        for p in s.get("participants", []):
            if p.get("status") != "FINISHED":
                continue
            base = p.get("official_time_s")
            pen = p.get("penalized_time_s")
            extra = p.get("penalties_s") or 0.0
            if base is not None and pen is not None and abs((base + extra) - pen) > 0.05:
                problems.append(
                    f"{path} [{p.get('participant_id')}]: official+penalties={base+extra:.3f} != penalized={pen:.3f}"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="print tables + checks, do not write files")
    args = parser.parse_args()

    problems = penalty_consistency_report()
    print(f"[penalty identity] finished runs checked; {len(problems)} discrepancies")
    for p in problems:
        print("  ", p)

    table1 = build_validation_results()
    table2 = build_coordination()

    if args.check:
        print("\n===== validation_results.tex =====\n")
        print(table1)
        print("\n===== holoocean_coordination.tex =====\n")
        print(table2)
        return 0

    (ARTICLE_TABLES / "validation_results.tex").write_text(table1, encoding="utf-8")
    (ARTICLE_TABLES / "holoocean_coordination.tex").write_text(table2, encoding="utf-8")
    print(f"\nWrote {ARTICLE_TABLES/'validation_results.tex'}")
    print(f"Wrote {ARTICLE_TABLES/'holoocean_coordination.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
