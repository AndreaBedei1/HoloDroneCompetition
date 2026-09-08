"""Generate the learning-extension figures from frozen artifacts.

Post-processing only: no HoloOcean run is launched, no GPU is used, and no
artifact under ``results/`` is modified. Two vector PDFs are written into
``article_journal/figures/generated/``:

* ``learning_survival.pdf`` -- unconditional multi-gate survival from episode
  start for the frozen Generation-1 PPO policy, with Wilson 95 % intervals,
  read from ``results/rl_public/ppo_final_readiness_929792/readiness_verdict.json``;
* ``learning_group_success.pdf`` -- completion rate per test group for the six
  controllers of the 474-episode matched benchmark, read from
  ``results/rl_public/final_benchmark/aggregate_by_group.csv``.

Usage:
    python article_journal/scripts/make_learning_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "article_journal" / "figures" / "generated"
VERDICT = ROOT / "results/rl_public/ppo_final_readiness_929792/readiness_verdict.json"
GROUPS = ROOT / "results/rl_public/final_benchmark/aggregate_by_group.csv"

INK = "#12263a"
C_RULE = "#2f6db0"
C_HYBRID = "#7a9e3f"
C_PPO = "#e07b39"
C_BC = "#9a6cb4"

plt.rcParams.update({
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.5,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def survival_figure() -> None:
    data = json.loads(VERDICT.read_text(encoding="utf-8"))
    survival = data["unconditional_survival_from_episode_start"]
    completion = data["sequence_completion"]

    gates = sorted(int(k) for k in survival)
    probability = [survival[str(g)]["probability"] for g in gates]
    low = [probability[i] - survival[str(g)]["wilson95"][0] for i, g in enumerate(gates)]
    high = [survival[str(g)]["wilson95"][1] - probability[i] for i, g in enumerate(gates)]

    lengths = sorted(int(k) for k in completion)
    rate = [completion[str(n)]["rate"] for n in lengths]
    clow = [rate[i] - completion[str(n)]["wilson95"][0] for i, n in enumerate(lengths)]
    chigh = [completion[str(n)]["wilson95"][1] - rate[i] for i, n in enumerate(lengths)]

    figure, axis = plt.subplots(figsize=(3.35, 2.25))
    axis.errorbar(gates, probability, yerr=[low, high], marker="o", markersize=3.4,
                  linewidth=1.3, capsize=2.0, color=C_PPO,
                  label="reaching gate $k$ from episode start")
    axis.errorbar(lengths, rate, yerr=[clow, chigh], marker="s", markersize=3.2,
                  linewidth=1.1, capsize=2.0, color=C_RULE, linestyle="--",
                  label="completing a $k$-gate sequence")
    axis.axhline(1.0, color="#9aa7b4", linewidth=0.7, linestyle=":")
    axis.annotate("first gate: 120/120", xy=(1, 1.0), xytext=(2.4, 1.045),
                  fontsize=6.2, color=INK,
                  arrowprops=dict(arrowstyle="-", color="#9aa7b4", linewidth=0.6))
    axis.set_xlabel("gate index $k$ / sequence length $k$")
    axis.set_ylabel("probability")
    axis.set_ylim(0.0, 1.14)
    axis.set_xticks(gates)
    axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.grid(axis="y", color="#e3e8ee", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc="lower left", frameon=False)
    figure.tight_layout(pad=0.35)
    figure.savefig(OUT / "learning_survival.pdf")
    plt.close(figure)


def group_figure() -> None:
    rows = list(csv.DictReader(GROUPS.open(encoding="utf-8")))
    order = [
        ("single_gate_retention", "single\ngate"),
        ("two_gate_straight", "two gate\nstraight"),
        ("two_gate_left", "two gate\nleft"),
        ("two_gate_right", "two gate\nright"),
        ("vertical_low_to_high", "vertical\nup"),
        ("vertical_high_to_low", "vertical\ndown"),
        ("three_gate_sequence", "three\ngate"),
        ("three_gate_s_shape", "three gate\nS-shape"),
        ("official_horseshoe_bay", "Horseshoe\nBay"),
        ("official_vertical_serpent", "Vertical\nSerpent"),
        ("official_mixed_endurance", "Mixed\nEndurance"),
    ]
    series = [
        ("rule_gate_center_then_commit", "Center-then-commit (rule)", C_RULE),
        ("hybrid", "Hybrid (rule + learned servo)", C_HYBRID),
        ("ppo_900462", "PPO 900k (best PPO)", C_PPO),
        ("bc_v3", "Behaviour cloning (BC-v3)", C_BC),
    ]
    lookup = {(r["controller"], r["group"]): r for r in rows}

    figure, axis = plt.subplots(figsize=(7.5, 2.4))
    width = 0.20
    positions = range(len(order))
    for index, (key, label, colour) in enumerate(series):
        values = [float(lookup[(key, group)]["success_rate"]) * 100.0 for group, _ in order]
        offsets = [p + (index - 1.5) * width for p in positions]
        axis.bar(offsets, values, width=width, color=colour, label=label,
                 edgecolor="white", linewidth=0.4)
    axis.axvline(7.5, color="#9aa7b4", linewidth=0.8, linestyle="--")
    axis.text(3.5, 110, "generated geometries", ha="center", fontsize=6.6, color="#5a6a7a")
    axis.text(9.5, 110, "official circuits", ha="center", fontsize=6.6, color="#5a6a7a")
    axis.set_xticks(list(positions))
    axis.set_xticklabels([label for _, label in order])
    axis.set_ylabel("completion rate (\\%)")
    axis.set_ylim(0, 118)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.grid(axis="y", color="#e3e8ee", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.legend(loc="lower left", bbox_to_anchor=(0.0, -0.42), ncol=4, frameon=False)
    figure.tight_layout(pad=0.35)
    figure.savefig(OUT / "learning_group_success.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (VERDICT, GROUPS):
        if not path.exists():
            raise SystemExit(f"missing frozen artifact: {path}")
    survival_figure()
    group_figure()
    provenance = {
        "figures": ["learning_survival.pdf", "learning_group_success.pdf"],
        "script": "article_journal/scripts/make_learning_figures.py",
        "sources": [
            "results/rl_public/ppo_final_readiness_929792/readiness_verdict.json",
            "results/rl_public/final_benchmark/aggregate_by_group.csv",
        ],
        "holoocean_launched_by_this_script": False,
        "artifacts_modified": False,
    }
    (OUT / "learning_figures.provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print("wrote learning_survival.pdf and learning_group_success.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
