"""Prepare-only tooling for extreme-corner expert demonstrations (``bc_extreme_corners_v2``).

Scans candidate seeds, computes each seed's Stage-2 body-frame offsets, and selects those
landing in the *extreme corner* of the randomization envelope (``|lateral| in [0.8, 1.0] m``
and ``|yaw| in [12, 15] deg``), covering all four sign combinations. This is where the
frozen BC evaluation failed.

It only PREPARES a demonstration plan (a seed list + the collect command); it does NOT
collect demonstrations or train a model. Use it as a fallback only if the Stage-2 PPO
diagnostic fails to improve extreme-corner robustness (DAgger-style / corrective demos ->
a BC-v2 model -> an optional BC-v2 -> PPO warm-start).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

from marine_race_arena.learning.curriculum import STAGE2_RANDOMIZATION
from marine_race_arena.learning.randomization import sample_offsets

LAT_MIN, LAT_MAX = 0.8, 1.0
YAW_MIN, YAW_MAX = 12.0, 15.0
# Scan range disjoint from every evaluation/test seed (see seed_registry).
DEFAULT_SEED_START, DEFAULT_SEED_LIMIT = 20000, 60000


def _quadrant(lat: float, yaw: float):
    return (lat >= 0, yaw >= 0)


def select_extreme_corner_seeds(*, n_per_quadrant: int = 5, seed_start: int = DEFAULT_SEED_START,
                                seed_limit: int = DEFAULT_SEED_LIMIT, spec=None) -> Dict:
    """Select seeds whose Stage-2 offsets fall in the extreme corner, balanced across the
    four (lateral-sign, yaw-sign) quadrants. Returns a prepare-only plan (no collection)."""
    spec = spec or STAGE2_RANDOMIZATION
    buckets: Dict = {(True, True): [], (True, False): [], (False, True): [], (False, False): []}
    for seed in range(seed_start, seed_limit):
        o = sample_offsets(spec, seed)
        lat, yaw = o["lateral_offset_m"], o["yaw_offset_deg"]
        if LAT_MIN <= abs(lat) <= LAT_MAX and YAW_MIN <= abs(yaw) <= YAW_MAX:
            key = _quadrant(lat, yaw)
            if len(buckets[key]) < n_per_quadrant:
                buckets[key].append({"seed": seed, "lateral_offset_m": round(lat, 4),
                                     "yaw_offset_deg": round(yaw, 4),
                                     "longitudinal_offset_m": round(o["longitudinal_offset_m"], 4),
                                     "depth_offset_m": round(o["depth_offset_m"], 4)})
        if all(len(v) >= n_per_quadrant for v in buckets.values()):
            break
    selected = [row for v in buckets.values() for row in v]
    selected_seeds = sorted(r["seed"] for r in selected)
    return {
        "experiment": "bc_extreme_corners_v2",
        "purpose": ("PREPARE-ONLY plan for corrective/extreme-corner expert demonstrations. Do NOT "
                    "collect or train until the Stage-2 PPO diagnostic is complete and only if it "
                    "fails to improve extreme-corner robustness. The frozen BC model is never modified."),
        "criterion": {"lateral_m": [LAT_MIN, LAT_MAX], "yaw_deg": [YAW_MIN, YAW_MAX],
                      "all_sign_combinations": True, "depth_and_longitudinal": "still randomized by the spec"},
        "spec": asdict(spec),
        "n_per_quadrant": n_per_quadrant,
        "seed_scan_range": [seed_start, seed_limit],
        "quadrant_counts": {f"lat{'+' if k[0] else '-'}_yaw{'+' if k[1] else '-'}": len(v)
                            for k, v in buckets.items()},
        "selected_seeds": selected_seeds,
        "selected": selected,
        "prepared_collect_command": (
            "python -m marine_race_arena.learning.collect_demos "
            "--track marine_race_arena/tracks/training/stage1_single_gate.json "
            f"--seeds {','.join(str(s) for s in selected_seeds)} "
            "--out results/rl/stage2/demos_extreme_v2 --adapter holoocean --randomize   "
            "# RUN ONLY after the Stage-2 diagnostic and if corrective demos are warranted"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/rl_public/stage2/bc_extreme_corners_v2_plan.json")
    parser.add_argument("--n-per-quadrant", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seed-limit", type=int, default=DEFAULT_SEED_LIMIT)
    args = parser.parse_args(argv)
    plan = select_extreme_corner_seeds(n_per_quadrant=args.n_per_quadrant,
                                       seed_start=args.seed_start, seed_limit=args.seed_limit)
    out: Optional[Path] = Path(args.out) if args.out else None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"[extreme-corner] wrote PREPARE-ONLY plan: {out} ({len(plan['selected_seeds'])} seeds, "
              f"quadrants {plan['quadrant_counts']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
