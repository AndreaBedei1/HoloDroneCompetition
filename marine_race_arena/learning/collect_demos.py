"""Collect real-HoloOcean expert demonstrations for behavioral cloning.

Records one episode per seed with the chosen expert on the real HoloOcean adapter
(no fallback), saving the growing :class:`BCDataset` after every episode so a long
collection is crash-safe. Prints per-episode completion/steps and a final quality
summary. Intended to be run from the ``marine_race_rl`` environment.

Usage:
    python -m marine_race_arena.learning.collect_demos --track <path> \
        --seeds 0-29 --out results/rl/stage1/demos --adapter holoocean
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np

from marine_race_arena.learning.dataset import BCDataset
from marine_race_arena.learning.trajectory_recorder import EpisodeRecord, record_episode


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
    parser.add_argument("--seeds", required=True, help="e.g. 0-29 or 0,1,2")
    parser.add_argument("--out", required=True)
    parser.add_argument("--controller", default="rule_gate_center_then_commit")
    parser.add_argument("--adapter", default="holoocean")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Apply the Stage-2 seeded start-pose/beacon-noise randomization for diversity.",
    )
    args = parser.parse_args(argv)

    start_randomization = None
    if args.randomize:
        from marine_race_arena.learning.curriculum import STAGE2_RANDOMIZATION

        start_randomization = STAGE2_RANDOMIZATION

    seeds = _parse_seeds(args.seeds)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "stage1_demos.npz"

    records: List[EpisodeRecord] = []
    per_episode = []
    print(f"[collect] {len(seeds)} demos, track={args.track}, adapter={args.adapter}, fallback={args.allow_fallback}")
    for episode_id, seed in enumerate(seeds):
        t0 = time.time()
        rec = record_episode(
            args.track,
            args.controller,
            seed=int(seed),
            dt=args.dt,
            adapter=args.adapter,
            allow_fallback=args.allow_fallback,
            max_steps=args.max_steps,
            official=True,
            episode_id=episode_id,
            start_randomization=start_randomization,
        )
        wall = time.time() - t0
        records.append(rec)
        finished = rec.final_status == "FINISHED"
        row = {
            "episode_id": episode_id,
            "seed": int(seed),
            "steps": rec.length,
            "final_status": rec.final_status,
            "gate_crossings": rec.gate_crossings,
            "finished": finished,
            "wall_s": round(wall, 1),
        }
        per_episode.append(row)
        print(f"[collect] seed={seed:>3} status={rec.final_status:<9} gates={rec.gate_crossings} "
              f"steps={rec.length:>4} wall={wall:5.1f}s")

        # Crash-safe incremental save of the whole dataset so far.
        try:
            dataset = BCDataset.from_records(records)
            dataset.save(dataset_path)
        except Exception as exc:  # pragma: no cover
            print(f"[collect] WARNING could not save incremental dataset: {exc}")
        (out_dir / "collection_log.json").write_text(json.dumps(per_episode, indent=2), encoding="utf-8")

    # Final quality summary.
    dataset = BCDataset.from_records(records)
    dataset.check_integrity()
    dataset.save(dataset_path)
    finished = [r for r in per_episode if r["finished"]]
    all_actions = dataset.actions
    summary = {
        "demos": len(records),
        "expert_completion_rate": len(finished) / len(records) if records else 0.0,
        "total_steps": int(len(dataset)),
        "mean_steps_finished": float(np.mean([r["steps"] for r in finished])) if finished else 0.0,
        "action_mean": [round(float(x), 4) for x in all_actions.mean(axis=0)],
        "action_min": [round(float(x), 4) for x in all_actions.min(axis=0)],
        "action_max": [round(float(x), 4) for x in all_actions.max(axis=0)],
        "action_saturation_frac": float(np.mean(np.abs(all_actions) > 0.98)),
        "obs_finite": bool(np.all(np.isfinite(dataset.observations))),
        "dataset_path": str(dataset_path),
    }
    (out_dir / "quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[collect] SUMMARY:", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
