"""Closed-loop HoloOcean comparison: transferred parent versus recurrent BC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from sb3_contrib import RecurrentPPO

from marine_race_arena.learning.gen2.evaluation import aggregate, run_policy_episode
from marine_race_arena.learning.gen2.recurrent_policy import Gen2RecurrentController
from marine_race_arena.learning.gen2 import track_fragments as tf
from marine_race_arena.learning.gen2.fog_contract import assert_approved_water_fog


def run(*, parent: str | Path, bc: str | Path, output: str | Path, seed: int = 28001) -> dict:
    rows = []
    for label, checkpoint in (("parent_warmstart_27d", parent), ("bc", bc)):
        model = RecurrentPPO.load(str(checkpoint), device="cpu")
        controller = Gen2RecurrentController(model, deterministic=True)
        for index, track in enumerate(tf.OFFICIAL_TRACKS):
            path = tf.track_path(track)
            assert_approved_water_fog(path)
            result = run_policy_episode(
                controller, path, seed=int(seed + index), adapter="holoocean",
                allow_fallback=False, max_steps=7000,
            )
            row = {"policy": label, "track": track, **result.as_row()}
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "parent": str(parent), "bc": str(bc), "rows": rows,
        "aggregate_by_policy": {
            label: aggregate([row for row in rows if row["policy"] == label])
            for label in ("parent_warmstart_27d", "bc")
        },
        "ppo_started": False,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate 27-D parent and BC checkpoints on real HoloOcean tracks")
    parser.add_argument("--parent", required=True)
    parser.add_argument("--bc", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=28001)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run(parent=args.parent, bc=args.bc, output=args.output, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
