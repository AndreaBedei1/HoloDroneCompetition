"""Generate manuscript figures from existing track JSON and the raw 78-run matrix.

Produces three vector PDFs in this directory:

* ``tracks_layout.pdf``  -- top-down layouts of the three official tracks
                            (ordered gates, sequence path, start, finish, depth);
* ``controller_comparison.pdf`` -- clean vs current performance of the two
                            reference controllers, from the frozen artifacts;
* ``task_teaser.pdf``    -- schematic of the racing task (vehicle passing an
                            ordered gate with its beacon on Horseshoe Bay).

Data only comes from the repository's track JSON and the existing result
artifacts; no HoloOcean run is launched and no artifact is modified. The teaser
is an honest schematic drawn from the real gate geometry, not a HoloOcean
render (a photo-real render can be captured with
``marine_race_arena/scripts/capture_environment_screenshot.py`` on a machine
with HoloOcean installed).
"""

from __future__ import annotations

import glob
import json
import math
import statistics
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Polygon

HERE = Path(__file__).resolve().parent
TRACKS = Path("marine_race_arena/tracks")
MATRIX = Path("results/onboard_only_validation/final_20260715")

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


def _load(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    order = d["track"]["gate_sequence"]
    by_id = {g["id"]: g for g in d["gates"]}
    gates = [by_id[gid] for gid in order]
    return {
        "name": d["race"]["name"],
        "length": d["track"]["declared_length_m"],
        "start": d["start"]["position"],
        "finish": d["finish"]["gate_id"],
        "gates": gates,
        "order": order,
    }


# --------------------------------------------------------------------------- #
def tracks_layout() -> None:
    # Top-down (x, y) layouts so the course shapes are visible. No depth encoding
    # -- a single gate colour keeps the paths clean; true aspect preserves shape.
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5))
    for ax, (fname, short) in zip(axes, OFFICIAL):
        t = _load(TRACKS / fname)
        xs = [g["position"][0] for g in t["gates"]]
        ys = [g["position"][1] for g in t["gates"]]
        sx, sy = t["start"][0], t["start"][1]
        # ordered path: start -> gates
        ax.plot([sx] + xs, [sy] + ys, "-", color="#9bb4cc", lw=1.0, zorder=1)
        ax.scatter(xs, ys, s=16, color="#2f6db0", zorder=3,
                   edgecolors="white", linewidths=0.4)
        # number every gate on the short track, every second on the denser ones
        step = 1 if len(xs) <= 12 else 2
        for i, (x, y) in enumerate(zip(xs, ys), start=1):
            if i == 1 or i == len(xs) or i % step == 0:
                ax.annotate(str(i), (x, y), fontsize=5.0, color=INK,
                            ha="center", va="bottom", xytext=(0, 3),
                            textcoords="offset points", zorder=4)
        ax.scatter([sx], [sy], marker="s", s=30, color="#3aa03a",
                   edgecolors="white", linewidths=0.4, zorder=5, label="start")
        ax.scatter([xs[-1]], [ys[-1]], marker="*", s=70, color="#d24d4d",
                   edgecolors="white", linewidths=0.4, zorder=5, label="finish")
        ax.set_title(f"{short} ({len(xs)} gates, {t['length']:.0f} m)")
        ax.set_xlabel("$x$ (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(length=2)
        ax.margins(0.12)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("$y$ (m)")
    axes[0].legend(loc="best", frameon=False, fontsize=6, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "tracks_layout.pdf")
    plt.close(fig)
    print("wrote tracks_layout.pdf")


# --------------------------------------------------------------------------- #
def _single_condition(*parts) -> Dict[str, float]:
    base = MATRIX.joinpath(*parts) / "runs"
    fin, n, fracs = 0, 0, []
    exp = {"horseshoe": 12, "vertical": 17, "mixed": 22}[parts[1]]
    for run_dir in sorted(base.glob("run_*")):
        files = sorted(glob.glob(str(run_dir / "*_summary.json")))
        if not files:
            continue
        p = json.loads(Path(files[0]).read_text())["participants"][0]
        n += 1
        fin += p["status"] == "FINISHED"
        fracs.append(p["completed_gates"] / exp)
    return {"finish_rate": fin / n if n else 0.0, "gate_frac": statistics.fmean(fracs) if fracs else 0.0, "fin": fin, "n": n}


def controller_comparison() -> None:
    conditions = [
        ("Horseshoe\n(clean)", ("clean", "horseshoe")),
        ("Vertical\n(clean)", ("clean", "vertical")),
        ("Mixed\n(clean)", ("clean", "mixed")),
        ("Horseshoe\n(medium cur.)", ("currents", "horseshoe", None, "medium")),
        ("Horseshoe\n(strong cur.)", ("currents", "horseshoe", None, "strong")),
    ]

    def get(controller, parts):
        if parts[0] == "clean":
            return _single_condition("clean", parts[1], controller)
        return _single_condition("currents", parts[1], controller, parts[3])

    base = [get("rule_gate_baseline", p) for _, p in conditions]
    ctc = [get("rule_gate_center_then_commit", p) for _, p in conditions]

    fig, ax = plt.subplots(figsize=(7.1, 2.6))
    x = range(len(conditions))
    w = 0.30
    b1 = ax.bar([i - w / 2 - 0.02 for i in x], [d["finish_rate"] for d in base], w,
                color=C_BASE, label="Continuous servo", edgecolor="white", linewidth=0.4)
    b2 = ax.bar([i + w / 2 + 0.02 for i in x], [d["finish_rate"] for d in ctc], w,
                color=C_CTC, label="Center-then-commit", edgecolor="white", linewidth=0.4)

    # The label and bar height encode the same finished-runs fraction.
    for bars, data, color in ((b1, base, C_BASE), (b2, ctc, C_CTC)):
        for rect, d in zip(bars, data):
            height = rect.get_height()
            inside = height >= 0.12
            ax.annotate(f"{d['fin']}/{d['n']}",
                        (rect.get_x() + rect.get_width() / 2,
                         height - 0.03 if inside else 0.02),
                        ha="center", va="top" if inside else "bottom",
                        fontsize=5.4, color="white" if inside else color)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("finished-run fraction")
    ax.set_xticks(list(x))
    ax.set_xticklabels([c for c, _ in conditions])
    ax.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2)
    fig.tight_layout()
    fig.savefig(HERE / "controller_comparison.pdf")
    plt.close(fig)
    print("wrote controller_comparison.pdf")


# --------------------------------------------------------------------------- #
def task_teaser() -> None:
    t = _load(TRACKS / "marine_race_horseshoe_bay.json")
    gates = t["gates"]
    xs = [g["position"][0] for g in gates]
    ys = [g["position"][1] for g in gates]

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    # subtle current field (medium set-flow direction), background
    gx = [i for i in range(-40, 15, 8)]
    gy = [i for i in range(-20, 25, 8)]
    for X in gx:
        for Y in gy:
            ax.add_patch(FancyArrow(X, Y, 2.2, 3.1, width=0.05, head_width=0.9,
                                    head_length=0.9, color="#cfe0ef", zorder=0,
                                    length_includes_head=True))

    # ordered gate path
    ax.plot(xs, ys, "-", color=INK, lw=1.0, alpha=0.4, zorder=1)

    # draw gates as apertures (two posts perpendicular to passage direction)
    for i, g in enumerate(gates):
        cx, cy, _ = g["position"]
        n = g["passage_direction"]
        nn = math.hypot(n[0], n[1]) or 1.0
        # right axis (perpendicular, in-plane)
        rx, ry = -n[1] / nn, n[0] / nn
        half = 0.9
        ax.plot([cx - rx * half, cx + rx * half], [cy - ry * half, cy + ry * half],
                color="#2b8a5b", lw=2.2, solid_capstyle="round", zorder=3)
        ax.plot([cx - rx * half, cx + rx * half], [cy - ry * half, cy + ry * half],
                color="#2b8a5b", lw=2.2, alpha=0.25, zorder=2)
        # beacon dot above gate centre
        ax.scatter([cx], [cy], s=8, color="#2b8a5b", zorder=3)

    # start marker
    sx, sy = t["start"][0], t["start"][1]
    ax.scatter([sx], [sy], marker="s", s=34, color="#3aa03a",
               edgecolors="white", linewidths=0.5, zorder=5)
    ax.annotate("start", (sx, sy), textcoords="offset points", xytext=(2, -9),
                fontsize=6, color="#2c6b2c")

    # vehicle passing gate ~#5 with a heading arrow along the passage direction
    k = 4
    cx, cy, _ = gates[k]["position"]
    n = gates[k]["passage_direction"]
    nn = math.hypot(n[0], n[1]) or 1.0
    hx, hy = n[0] / nn, n[1] / nn
    # AUV body as a small oriented triangle just before the gate
    bx, by = cx - hx * 2.6, cy - hy * 2.6
    left = (-hy, hx)
    body = Polygon([
        (bx + hx * 1.4, by + hy * 1.4),
        (bx - hx * 0.8 + left[0] * 0.7, by - hy * 0.8 + left[1] * 0.7),
        (bx - hx * 0.8 - left[0] * 0.7, by - hy * 0.8 - left[1] * 0.7),
    ], closed=True, facecolor="#d24d4d", edgecolor="white", linewidth=0.5, zorder=6)
    ax.add_patch(body)
    ax.add_patch(FancyArrow(bx + hx * 1.5, by + hy * 1.5, hx * 2.3, hy * 2.3,
                            width=0.06, head_width=1.0, head_length=1.0,
                            color="#d24d4d", zorder=6, length_includes_head=True))

    # gate index labels for a few gates
    for idx in (0, 4, 11):
        ax.annotate(f"g{idx + 1}", (xs[idx], ys[idx]), textcoords="offset points",
                    xytext=(3, 3), fontsize=6, color=INK)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title("Underwater gate racing: an ordered, beacon-marked course",
                 fontsize=7.5)
    fig.tight_layout()
    fig.savefig(HERE / "task_teaser.pdf")
    plt.close(fig)
    print("wrote task_teaser.pdf")


if __name__ == "__main__":
    tracks_layout()
    controller_comparison()
    task_teaser()
    print("done")
