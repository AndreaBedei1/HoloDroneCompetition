"""Append-only record of what the agents actually trained on.

``active_episode.json`` is overwritten every episode, so after days of training
there is no way to answer "what distribution did the policy actually see?".  The
previous audit could only sample four surviving track files across two runs.

One compact row per episode, no frame data.  Enough to reconstruct the realised
curriculum, verify the dataset split was respected, and check geometric
diversity long after the fact.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

EPISODE_LOG_VERSION = "rl_episode_composition_v1"
EPISODE_LOG_NAME = "episode_composition.jsonl"


def episode_row(
    *,
    utc: str,
    algorithm: str,
    run: str,
    worker_id: int,
    geometry: Any,
    outcome: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one compact row from a sampled geometry plus its result."""

    spacings = [float(v) for v in (getattr(geometry, "spacings_m", ()) or ())]
    turns = [float(v) for v in (getattr(geometry, "turn_deltas_deg", ()) or ())]
    vertical = [float(v) for v in (getattr(geometry, "vertical_deltas_m", ()) or ())]
    result = dict(outcome or {})
    row = {
        "schema_version": EPISODE_LOG_VERSION,
        "utc": str(utc),
        "algorithm": str(algorithm),
        "run": str(run),
        "worker_id": int(worker_id),
        "episode_seed": int(getattr(geometry, "seed", -1)),
        "dataset_split": getattr(geometry, "dataset_split", None),
        "curriculum_stage": getattr(geometry, "curriculum_stage", None),
        "sequence_category": getattr(geometry, "sequence_bucket", None),
        "episode_type": getattr(geometry, "episode_type", None),
        "gate_count": int(getattr(geometry, "gate_count", 0)),
        "difficulty": getattr(geometry, "difficulty", None),
        "pattern": getattr(geometry, "pattern", None),
        "spacing_mean_m": round(mean(spacings), 4) if spacings else None,
        "spacing_min_m": round(min(spacings), 4) if spacings else None,
        "spacing_max_m": round(max(spacings), 4) if spacings else None,
        "turn_abs_mean_deg": round(mean(abs(v) for v in turns), 4) if turns else None,
        "turn_abs_max_deg": round(max(abs(v) for v in turns), 4) if turns else None,
        "turn_sign_changes": sum(
            1 for a, b in zip(turns, turns[1:]) if a * b < 0
        ) if len(turns) > 1 else 0,
        "vertical_abs_mean_m": round(mean(abs(v) for v in vertical), 4) if vertical else None,
        "vertical_abs_max_m": round(max(abs(v) for v in vertical), 4) if vertical else None,
        "initial_yaw_error_deg": round(float(getattr(geometry, "initial_yaw_error_deg", 0.0)), 4),
        "initial_lateral_offset_m": round(float(getattr(geometry, "initial_lateral_offset_m", 0.0)), 4),
    }
    for key in (
        "gates_completed", "collision", "missed_gate", "wrong_direction",
        "out_of_bounds", "timeout", "outcome", "steps",
    ):
        if key in result:
            row[key] = result[key]
    return row


def append_episode(run_dir: str | Path, row: Mapping[str, Any]) -> None:
    """Append one row; never let logging break training."""

    try:
        path = Path(run_dir) / "logs" / EPISODE_LOG_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        return


def read_episodes(run_dir: str | Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = Path(run_dir) / "logs" / EPISODE_LOG_NAME
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows[-int(limit):] if limit else rows


def _quantiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        return {"min": None, "p50": None, "max": None, "mean": None}
    return {
        "min": round(clean[0], 4),
        "p50": round(median(clean), 4),
        "max": round(clean[-1], 4),
        "mean": round(mean(clean), 4),
    }


def summarize_episodes(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """What the agent actually saw: category shares and geometry spread."""

    rows = list(rows)
    total = len(rows)
    if not total:
        return {"schema_version": EPISODE_LOG_VERSION, "episodes": 0}
    categories = Counter(str(r.get("sequence_category")) for r in rows)
    splits = Counter(str(r.get("dataset_split")) for r in rows)
    stages = Counter(str(r.get("curriculum_stage")) for r in rows)
    gate_counts = Counter(int(r.get("gate_count", 0)) for r in rows)
    return {
        "schema_version": EPISODE_LOG_VERSION,
        "episodes": total,
        "category_share": {
            key: round(categories.get(key, 0) / total, 4)
            for key in ("focus", "short", "medium", "long")
        },
        "dataset_split_share": {k: round(v / total, 4) for k, v in splits.items()},
        "curriculum_stage_share": {k: round(v / total, 4) for k, v in stages.items()},
        "distinct_gate_counts": sorted(gate_counts),
        "distinct_episode_seeds": len({r.get("episode_seed") for r in rows}),
        "spacing_m": _quantiles([r.get("spacing_mean_m") for r in rows]),
        "turn_abs_deg": _quantiles([r.get("turn_abs_mean_deg") for r in rows]),
        "vertical_abs_m": _quantiles([r.get("vertical_abs_mean_m") for r in rows]),
        "turn_sign_changes": _quantiles([r.get("turn_sign_changes") for r in rows]),
        "patterns": dict(Counter(str(r.get("pattern")) for r in rows)),
    }


def report(run_dir: str | Path, limit: Optional[int] = None) -> Dict[str, Any]:
    return summarize_episodes(read_episodes(run_dir, limit))


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(report(args.run_dir, args.limit), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
