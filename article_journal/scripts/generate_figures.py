"""Generate the data-driven manuscript figures from track files and artifacts.

Post-processing only. Reads the official track JSON files and the released
evidence package under ``artifacts/paper/``; writes two vector PDFs into
``article_journal/figures/generated/``:

* ``tracks_layout.pdf``         -- top-down layouts of the three official
                                   circuits (ordered gates, sequence path,
                                   start, finish);
* ``controller_comparison.pdf`` -- clean and medium-current completion of the
                                   two reference controllers.

No simulator run is launched and no artifact is modified. A provenance JSON is
written alongside the figures recording the inputs and asserting this.

Usage:
    python article_journal/scripts/generate_figures.py
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TRACKS = ROOT / "marine_race_arena" / "tracks"
RUNS_CSV = ROOT / "artifacts" / "paper" / "benchmark" / "runs.csv"
OUT = ROOT / "article_journal" / "figures" / "generated"

C_BASE = "#2f6db0"   # continuous servo
C_CTC = "#e07b39"    # center-then-commit
INK = "#12263a"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.6,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

OFFICIAL = [
    ("marine_race_horseshoe_bay.json", "Horseshoe Bay"),
    ("marine_race_vertical_serpent.json", "Vertical Serpent"),
    ("marine_race_mixed_endurance.json", "Mixed Endurance"),
]
TRACK_NAMES = {
    "horseshoe": "Marine Race Horseshoe Bay",
    "vertical": "Marine Race Vertical Serpent",
    "mixed": "Marine Race Mixed Endurance",
}


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    order = data["track"]["gate_sequence"]
    by_id = {gate["id"]: gate for gate in data["gates"]}
    return {
        "name": data["race"]["name"],
        "length": data["track"]["declared_length_m"],
        "start": data["start"]["position"],
        "gates": [by_id[gid] for gid in order],
    }


def tracks_layout() -> None:
    # Top-down (x, y) layouts so the course shapes are visible. No depth
    # encoding -- a single gate colour keeps the paths clean; true aspect
    # preserves shape.
    figure, axes = plt.subplots(1, 3, figsize=(7.1, 2.5))
    for axis, (fname, short) in zip(axes, OFFICIAL):
        track = _load(TRACKS / fname)
        xs = [gate["position"][0] for gate in track["gates"]]
        ys = [gate["position"][1] for gate in track["gates"]]
        sx, sy = track["start"][0], track["start"][1]
        axis.plot([sx] + xs, [sy] + ys, "-", color="#9bb4cc", lw=1.0, zorder=1)
        axis.scatter(xs, ys, s=16, color=C_BASE, zorder=3,
                     edgecolors="white", linewidths=0.4)
        # number every gate on the short track, every second on the denser ones
        step = 1 if len(xs) <= 12 else 2
        for index, (x, y) in enumerate(zip(xs, ys), start=1):
            if index == 1 or index == len(xs) or index % step == 0:
                axis.annotate(str(index), (x, y), fontsize=5.0, color=INK,
                              ha="center", va="bottom", xytext=(0, 3),
                              textcoords="offset points", zorder=4)
        axis.scatter([sx], [sy], marker="s", s=30, color="#3aa03a",
                     edgecolors="white", linewidths=0.4, zorder=5, label="start")
        axis.scatter([xs[-1]], [ys[-1]], marker="*", s=70, color="#d24d4d",
                     edgecolors="white", linewidths=0.4, zorder=5, label="finish")
        axis.set_title("{} ({} gates, {:.0f} m)".format(short, len(xs), track["length"]))
        axis.set_xlabel("$x$ (m)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.tick_params(length=2)
        axis.margins(0.12)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("$y$ (m)")
    axes[0].legend(loc="best", frameon=False, fontsize=6, handletextpad=0.3)
    figure.tight_layout()
    figure.savefig(OUT / "tracks_layout.pdf")
    plt.close(figure)
    print("wrote tracks_layout.pdf")


def _condition(experiment: str, track_key: str, controller: str,
               current_profile: str) -> dict:
    with RUNS_CSV.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["kind"] == "benchmark"
                and row["experiment"] == experiment
                and row["track"] == TRACK_NAMES[track_key]
                and row["controller"] == controller
                and row["current_profile"] == current_profile]
    if not rows:
        raise SystemExit("no rows for experiment={}, track={}, controller={}, "
                         "current_profile={}".format(experiment, track_key,
                                                     controller, current_profile))
    expected = int(rows[0]["expected_gates"])
    finished = sum(row["status"] == "FINISHED" for row in rows)
    return {
        "finish_rate": finished / len(rows),
        "gate_frac": statistics.fmean(int(row["completed_gates"]) / expected
                                      for row in rows),
        "fin": finished,
        "n": len(rows),
    }


def controller_comparison() -> None:
    conditions = [
        ("Horseshoe\n(clean)", ("clean", "horseshoe", "none")),
        ("Vertical\n(clean)", ("clean", "vertical", "none")),
        ("Mixed\n(clean)", ("clean", "mixed", "none")),
        ("Horseshoe\n(medium cur.)", ("currents", "horseshoe", "medium")),
    ]
    base = [_condition(e, t, "rule_gate_baseline", c) for _, (e, t, c) in conditions]
    ctc = [_condition(e, t, "rule_gate_center_then_commit", c)
           for _, (e, t, c) in conditions]

    # Guard against a silently changed artifact: these are the completion counts
    # reported in Tables tab:clean_tracks and tab:currents.
    expected_base = [(5, 5), (5, 5), (2, 5), (2, 5)]
    expected_ctc = [(5, 5), (4, 5), (5, 5), (3, 5)]
    actual_base = [(d["fin"], d["n"]) for d in base]
    actual_ctc = [(d["fin"], d["n"]) for d in ctc]
    if actual_base != expected_base:
        raise SystemExit("unexpected Continuous Servo completion counts: "
                         "{}".format(actual_base))
    if actual_ctc != expected_ctc:
        raise SystemExit("unexpected Center-then-Commit completion counts: "
                         "{}".format(actual_ctc))

    figure, axis = plt.subplots(figsize=(7.1, 2.6))
    x = range(len(conditions))
    width = 0.30
    bars_base = axis.bar([i - width / 2 - 0.02 for i in x],
                         [d["finish_rate"] for d in base], width,
                         color=C_BASE, label="Continuous servo",
                         edgecolor="white", linewidth=0.4)
    bars_ctc = axis.bar([i + width / 2 + 0.02 for i in x],
                        [d["finish_rate"] for d in ctc], width,
                        color=C_CTC, label="Center-then-commit",
                        edgecolor="white", linewidth=0.4)

    # The label and the bar height encode the same finished-runs fraction.
    for bars, data, color in ((bars_base, base, C_BASE), (bars_ctc, ctc, C_CTC)):
        for rect, entry in zip(bars, data):
            height = rect.get_height()
            inside = height >= 0.12
            axis.annotate("{}/{}".format(entry["fin"], entry["n"]),
                          (rect.get_x() + rect.get_width() / 2,
                           height - 0.03 if inside else 0.02),
                          ha="center", va="top" if inside else "bottom",
                          fontsize=5.4, color="white" if inside else color)

    axis.set_ylim(0, 1.05)
    axis.set_ylabel("finished-run fraction")
    axis.set_xticks(list(x))
    axis.set_xticklabels([label for label, _ in conditions])
    axis.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.2)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.tick_params(length=2)
    figure.tight_layout()
    figure.savefig(OUT / "controller_comparison.pdf")
    plt.close(figure)
    print("wrote controller_comparison.pdf")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tracks_layout()
    controller_comparison()
    provenance = {
        "figures": ["tracks_layout.pdf", "controller_comparison.pdf"],
        "script": "article_journal/scripts/generate_figures.py",
        "track_sources": [("marine_race_arena/tracks/" + name) for name, _ in OFFICIAL],
        "result_source": "artifacts/paper/benchmark/runs.csv",
        "holoocean_launched_by_this_script": False,
        "artifacts_modified_by_this_script": False,
    }
    with (OUT / "figures.provenance.json").open("w", encoding="utf-8",
                                                newline=chr(10)) as stream:
        stream.write(json.dumps(provenance, indent=2) + "\n")
    print("wrote figures.provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
