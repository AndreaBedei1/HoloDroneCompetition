"""Collect expert data on the exact transitions where the parent loses runs.

    python -m marine_race_arena.learning.gen2.collect_hotspots \\
        --failure-map results/rl/gen2/n10_best_completion \\
        --out results/rl/gen2/hotspot_corpus --trials 2 --workers 5

The windows come from the measured per-transition table, not from a guess: a
transition qualifies as a hotspot only if it was attempted enough times to be
distinguishable from noise and still fell below the rate threshold.  Each one
becomes exact contiguous windows G(k-2)..G(k+2) of the real circuit, so the
learner practises the transition it actually fails, entered the way the circuit
enters it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from marine_race_arena.learning.gen2 import collect_fragments as cf
from marine_race_arena.learning.gen2 import seeds as gen2_seeds
from marine_race_arena.learning.gen2 import track_fragments as tf
from marine_race_arena.learning.gen2.track_eval import (
    CircuitOutcome,
    hotspot_fragments,
    transition_hotspots,
)

#: Offset into the fragment seed band, so hotspot episodes never collide with
#: the broad fragment corpus collected earlier.
HOTSPOT_SEED_OFFSET = 2000


def load_circuit_rows(report_dir: str | Path) -> List[CircuitOutcome]:
    """Read every per-track circuit report under ``report_dir``."""
    base = Path(report_dir)
    rows: List[CircuitOutcome] = []
    for track in tf.OFFICIAL_TRACKS:
        for candidate in (
            base / track / "circuits" / "circuit_report.json",
            base / f"eval_{track}" / "circuits" / "circuit_report.json",
        ):
            if candidate.exists():
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                rows += [CircuitOutcome(**r) for r in payload.get("rows", [])]
                break
    return rows


def build_hotspot_jobs(
    rows: Sequence[CircuitOutcome],
    *,
    trials: int = 2,
    radius: int = 2,
    min_attempts: int = 5,
    max_rate: float = 0.92,
) -> tuple[List[cf.FragmentJob], List[Dict[str, Any]], List[tf.TrackFragment]]:
    hotspots = transition_hotspots(rows, min_attempts=min_attempts, max_rate=max_rate)
    fragments = hotspot_fragments(hotspots, radius=radius)
    band = gen2_seeds.GEN2_TRACK_FRAGMENT_SEEDS
    jobs: List[cf.FragmentJob] = []
    used: set = set()
    for index, fragment in enumerate(fragments):
        for trial in range(int(trials)):
            seed = band[(HOTSPOT_SEED_OFFSET + index * int(trials) + trial) % len(band)]
            while seed in used:
                seed = band[(band.index(seed) + 1) % len(band)]
            used.add(seed)
            jobs.append(cf.FragmentJob(fragment=fragment, trial=trial, seed=seed))
    return jobs, hotspots, fragments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect expert data on failure hotspots")
    parser.add_argument("--failure-map", required=True,
                        help="directory holding the per-track circuit reports")
    parser.add_argument("--out", required=True)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--min-attempts", type=int, default=5)
    parser.add_argument("--max-rate", type=float, default=0.92)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--adapter", default="holoocean", choices=("holoocean", "fallback"))
    parser.add_argument("--allow-fallback", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_circuit_rows(args.failure_map)
    if not rows:
        raise SystemExit(f"no circuit rows found under {args.failure_map}")
    jobs, hotspots, fragments = build_hotspot_jobs(
        rows, trials=args.trials, radius=args.radius,
        min_attempts=args.min_attempts, max_rate=args.max_rate,
    )
    print(f"[hot] {len(rows)} run analizzati -> {len(hotspots)} hotspot "
          f"-> {len(fragments)} frammenti esatti -> {len(jobs)} episodi", flush=True)
    for entry in hotspots[:8]:
        print("[hot]   %-18s %-10s rate=%.2f attempts=%d %s" % (
            entry["track"], entry["transition"], entry["rate"],
            entry["attempts"], entry["kinds"]), flush=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "hotspots.json").write_text(
        json.dumps({"hotspots": hotspots,
                    "fragments": [f.as_dict() for f in fragments]}, indent=2),
        encoding="utf-8",
    )
    summary = cf.collect(
        args.out, workers=args.workers, adapter=args.adapter,
        allow_fallback=args.allow_fallback, jobs=jobs,
    )
    print("[hot] " + json.dumps(summary["track_composition"], indent=2), flush=True)
    print(f"[hot] corpus_sha256={summary['corpus_sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
