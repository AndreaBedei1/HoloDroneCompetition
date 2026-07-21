"""Closed-loop evaluation of a learned controller under the unchanged referee.

Runs a controller (BC or PPO model, or a rule baseline) on held-out seeds through
the real race runner + independent referee, saving each seed's result incrementally
so a long real-HoloOcean evaluation is crash-safe and resumable (already-evaluated
seeds are skipped). Reports completion rate, gates, collisions, wrong-direction and
out-of-bounds events, and mean inference time.

Usage (marine_race_rl env):
    python -m marine_race_arena.learning.closed_loop_eval --track <path> \
        --seeds 300-319 --model results/rl/stage1/bc/best_model.pt \
        --out results/rl/stage1/eval_bc --adapter holoocean
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np

from marine_race_arena.learning.evaluate_policy import evaluate_controller
from marine_race_arena.participants.controller_loader import ControllerLoader


def _parse_seeds(spec: str) -> List[int]:
    seeds: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=None, help="model path; omit for a rule controller")
    parser.add_argument("--controller", default="rl_gate_controller")
    parser.add_argument("--adapter", default="holoocean")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--randomize", action="store_true", help="Apply Stage-2 start randomization (held-out seeds).")
    args = parser.parse_args(argv)

    start_randomization = None
    if args.randomize:
        from marine_race_arena.learning.curriculum import STAGE2_RANDOMIZATION

        start_randomization = STAGE2_RANDOMIZATION

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "eval_results.json"

    existing = {}
    if results_path.exists():
        existing = {int(r["seed"]): r for r in json.loads(results_path.read_text(encoding="utf-8"))}

    def factory():
        return ControllerLoader().load(
            args.controller, constructor_kwargs={"model_path": args.model} if args.model else None
        )

    seeds = _parse_seeds(args.seeds)
    rows = list(existing.values())
    for seed in seeds:
        if seed in existing:
            print(f"[eval] seed={seed} already done ({existing[seed]['status']}) -- skip")
            continue
        t0 = time.time()
        report = evaluate_controller(
            args.track,
            factory,
            seeds=[seed],
            label=args.controller,
            adapter=args.adapter,
            allow_fallback=args.allow_fallback,
            duration_s=args.duration,
            dt=args.dt,
            start_randomization=start_randomization,
        )
        r = report.results[0]
        wall = time.time() - t0
        row = {
            "seed": int(seed),
            "status": r.status,
            "finished": r.finished,
            "completed_gates": r.completed_gates,
            "expected_gates": r.expected_gates,
            "collision_events": r.collision_events,
            "obstacle_collision_events": r.obstacle_collision_events,
            "out_of_bounds_events": r.out_of_bounds_events,
            "wrong_direction": r.missed_gate_attempts,
            "official_time_s": r.official_time_s,
            "wall_s": round(wall, 1),
        }
        rows.append(row)
        print(f"[eval] seed={seed:>3} status={r.status:<9} gates={r.completed_gates}/{r.expected_gates} "
              f"coll={r.collision_events} oob={r.out_of_bounds_events} wall={wall:5.1f}s")
        rows_sorted = sorted(rows, key=lambda x: x["seed"])
        results_path.write_text(json.dumps(rows_sorted, indent=2), encoding="utf-8")

    rows = sorted(rows, key=lambda x: x["seed"])
    evaluated = [r for r in rows if r["seed"] in set(seeds)]
    n = len(evaluated)
    finished = [r for r in evaluated if r["finished"]]
    summary = {
        "controller": args.controller,
        "model": args.model,
        "track": args.track,
        "adapter": args.adapter,
        "n_eval": n,
        "completion_rate": round(len(finished) / n, 4) if n else 0.0,
        "mean_gates": round(float(np.mean([r["completed_gates"] for r in evaluated])), 3) if evaluated else 0.0,
        "total_collisions": int(sum(r["collision_events"] for r in evaluated)),
        "total_out_of_bounds": int(sum(r["out_of_bounds_events"] for r in evaluated)),
        "total_wrong_direction": int(sum(r["wrong_direction"] for r in evaluated)),
        "seeds": sorted(r["seed"] for r in evaluated),
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[eval] SUMMARY:", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
