"""Closed-loop evaluation of a learned controller under the unchanged referee.

Runs a controller (BC or PPO model, or a rule baseline) on held-out seeds through
the real race runner + independent referee, saving each seed's result incrementally
so a long real-HoloOcean evaluation is crash-safe and resumable. Every result row
records the referee's own participant status *and* the reason the evaluation runner
stopped (see :func:`evaluate_policy.derive_evaluation_end_reason`).

Resume is guarded by an ``evaluation_manifest.json``: a resumed run must match the
recorded experiment (same model hash, track, adapter/fallback, randomization, dt and
encoding) or it is refused. Combining rows from different experiments is never done
silently; start a new output directory, or pass ``--force-new`` (which backs the old
directory up to a timestamped copy — it never deletes).

Usage (marine_race_rl env):
    python -m marine_race_arena.learning.closed_loop_eval --track <path> \
        --seeds 300-319 --model results/rl/stage1/bc/best_model.pt \
        --out results/rl/stage1/eval_bc --adapter holoocean
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from marine_race_arena.learning.config import ACTION_CONTRACT_VERSION, OBS_ENCODING_VERSION
from marine_race_arena.learning.evaluate_policy import evaluate_controller
from marine_race_arena.learning.provenance import git_sha, now_utc, sha256_file
from marine_race_arena.participants.controller_loader import ControllerLoader

MANIFEST_SCHEMA_VERSION = "eval_manifest_v1"

# Fields whose disagreement means two runs are NOT the same experiment and must not
# be merged. The requested seed set is intentionally excluded (a run may add seeds).
_IDENTITY_FIELDS = (
    "controller_name",
    "model_sha256",
    "track_sha256",
    "adapter_requested",
    "fallback_allowed",
    "randomization_enabled",
    "randomization_spec",
    "dt",
    "duration_s",
    "max_steps",
    "observation_encoding_version",
    "action_contract_version",
)


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


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_csv(path: Path, rows) -> None:
    import csv

    if not rows:
        return
    fields = [k for k in rows[0].keys() if k != "applied_randomization"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
    tmp.replace(path)


def _wilson(p: float, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def build_requested_config(args, *, model_sha256, randomization_spec) -> Dict:
    """The experiment-identity + provenance fields captured in the manifest."""
    return {
        "controller_name": args.controller,
        "model_path": args.model,
        "model_sha256": model_sha256,
        "track_path": args.track,
        "track_sha256": sha256_file(args.track),
        "adapter_requested": args.adapter,
        "fallback_allowed": bool(args.allow_fallback),
        "randomization_enabled": bool(args.randomize),
        "randomization_spec": randomization_spec,
        "dt": float(args.dt),
        "duration_s": (float(args.duration) if args.duration is not None else None),
        "max_steps": None,  # the race runner is time-deadline based, not step-capped
        "observation_encoding_version": OBS_ENCODING_VERSION,
        "action_contract_version": ACTION_CONTRACT_VERSION,
    }


def manifest_incompatibilities(existing: Dict, requested: Dict) -> List[str]:
    """List the identity fields on which an existing manifest disagrees with the request."""
    out: List[str] = []
    for key in _IDENTITY_FIELDS:
        if existing.get(key) != requested.get(key):
            out.append(f"{key} ({existing.get(key)!r} != {requested.get(key)!r})")
    return out


def _backup_dir(out_dir: Path) -> Path:
    """Move an existing output directory aside to a timestamped backup (never delete)."""
    stamp = now_utc().replace(":", "").replace("-", "").replace("T", "_").rstrip("Z")
    backup = out_dir.parent / f"{out_dir.name}_backup_{stamp}"
    shutil.move(str(out_dir), str(backup))
    return backup


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
    parser.add_argument("--force-new", action="store_true",
                        help="Start a fresh experiment; an existing output directory is moved to a timestamped backup.")
    args = parser.parse_args(argv)

    start_randomization = None
    randomization_spec = None
    if args.randomize:
        from marine_race_arena.learning.curriculum import STAGE2_RANDOMIZATION

        start_randomization = STAGE2_RANDOMIZATION
        randomization_spec = asdict(STAGE2_RANDOMIZATION)

    model_sha256 = sha256_file(args.model) if args.model else None
    requested = build_requested_config(args, model_sha256=model_sha256, randomization_spec=randomization_spec)

    out_dir = Path(args.out)
    results_path = out_dir / "eval_results.json"
    manifest_path = out_dir / "evaluation_manifest.json"

    # --- Resume safety: verify the existing directory is the same experiment ---
    existing_manifest: Optional[Dict] = None
    if out_dir.exists() and any(out_dir.iterdir()):
        if manifest_path.exists():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            issues = manifest_incompatibilities(existing_manifest, requested)
            if issues and not args.force_new:
                print("[eval] REFUSING to resume: existing results were produced by a different experiment:")
                for issue in issues:
                    print("   -", issue)
                print("[eval] Use a new --out directory, or --force-new to archive the old results and start fresh.")
                return 2
            if issues and args.force_new:
                backup = _backup_dir(out_dir)
                print(f"[eval] --force-new: archived incompatible results to {backup}")
                existing_manifest = None
        elif results_path.exists():
            # Legacy directory without a manifest: cannot verify compatibility.
            if not args.force_new:
                print(f"[eval] REFUSING to resume: {results_path} has no evaluation_manifest.json to verify "
                      "compatibility. Use a new --out directory, or --force-new to archive and start fresh.")
                return 2
            backup = _backup_dir(out_dir)
            print(f"[eval] --force-new: archived unverifiable results to {backup}")

    out_dir.mkdir(parents=True, exist_ok=True)

    existing = {}
    if results_path.exists():
        existing = {int(r["seed"]): r for r in json.loads(results_path.read_text(encoding="utf-8"))}

    # Write / refresh the manifest (created_utc preserved across resume).
    created_utc = (existing_manifest or {}).get("created_utc") or now_utc()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "git_sha": git_sha(),
        **requested,
        "model_bytes": (Path(args.model).stat().st_size if args.model and Path(args.model).exists() else None),
        "adapter_actual": (existing_manifest or {}).get("adapter_actual"),
        "requested_seeds": sorted(set(_parse_seeds(args.seeds)) | set((existing_manifest or {}).get("requested_seeds", []))),
        "completed_seeds": sorted(existing.keys()),
        "created_utc": created_utc,
        "updated_utc": now_utc(),
    }
    _atomic_write(manifest_path, json.dumps(manifest, indent=2))

    def factory():
        return ControllerLoader().load(
            args.controller, constructor_kwargs={"model_path": args.model} if args.model else None
        )

    seeds = _parse_seeds(args.seeds)
    rows = list(existing.values())
    for seed in seeds:
        if seed in existing:
            print(f"[eval] seed={seed} already done ({existing[seed].get('referee_status', existing[seed].get('status'))}) -- skip")
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
            "referee_status": r.referee_status,
            "evaluation_end_reason": r.evaluation_end_reason,
            "finished": r.finished,
            "completed_gates": r.completed_gates,
            "expected_gates": r.expected_gates,
            "official_time_s": r.official_time_s,
            "penalized_time_s": r.penalized_time_s,
            "collision_events": r.collision_events,
            "obstacle_collision_events": r.obstacle_collision_events,
            "out_of_bounds_events": r.out_of_bounds_events,
            "stuck_events": r.stuck_events,
            "missed_gate_attempts": r.missed_gate_attempts,
            "wrong_direction_crossings": r.wrong_direction_crossings,
            "inference_time_ms": r.inference_time_ms,
            "wall_s": round(wall, 3),
            "adapter_used": r.adapter_used,
            "applied_randomization": r.applied_randomization,
        }
        rows.append(row)
        print(f"[eval] seed={seed:>3} referee={r.referee_status:<9} end={r.evaluation_end_reason:<16} "
              f"gates={r.completed_gates}/{r.expected_gates} coll={r.collision_events} "
              f"oob={r.out_of_bounds_events} wrongdir={r.wrong_direction_crossings} wall={wall:5.1f}s")
        rows_sorted = sorted(rows, key=lambda x: x["seed"])
        _atomic_write(results_path, json.dumps(rows_sorted, indent=2))
        _write_csv(out_dir / "eval_results.csv", rows_sorted)
        # Keep the manifest's completed set + adapter-actual current after every seed.
        manifest["completed_seeds"] = [x["seed"] for x in rows_sorted]
        manifest["adapter_actual"] = r.adapter_used
        manifest["updated_utc"] = now_utc()
        _atomic_write(manifest_path, json.dumps(manifest, indent=2))

    rows = sorted(rows, key=lambda x: x["seed"])
    evaluated = [r for r in rows if r["seed"] in set(seeds)]
    n = len(evaluated)
    finished = [r for r in evaluated if r["finished"]]
    rate = len(finished) / n if n else 0.0
    ci = _wilson(rate, n)
    end_reason_counts: Dict[str, int] = {}
    referee_status_counts: Dict[str, int] = {}
    for r in evaluated:
        end_reason_counts[r.get("evaluation_end_reason", "UNKNOWN")] = end_reason_counts.get(r.get("evaluation_end_reason", "UNKNOWN"), 0) + 1
        rs = r.get("referee_status", r.get("status"))
        referee_status_counts[rs] = referee_status_counts.get(rs, 0) + 1
    summary = {
        "controller": args.controller,
        "model": args.model,
        "track": args.track,
        "adapter": args.adapter,
        "randomized": bool(args.randomize),
        "n_eval": n,
        "completions": len(finished),
        "completion_rate": round(rate, 4),
        "completion_rate_wilson95_low": round(ci[0], 4),
        "completion_rate_wilson95_high": round(ci[1], 4),
        "mean_gates": round(float(np.mean([r["completed_gates"] for r in evaluated])), 3) if evaluated else 0.0,
        "total_collisions": int(sum(r["collision_events"] for r in evaluated)),
        "total_out_of_bounds": int(sum(r["out_of_bounds_events"] for r in evaluated)),
        "total_stuck": int(sum(r["stuck_events"] for r in evaluated)),
        "total_missed_gate_attempts": int(sum(r["missed_gate_attempts"] for r in evaluated)),
        "total_wrong_direction_crossings": int(sum(r["wrong_direction_crossings"] for r in evaluated)),
        "mean_inference_time_ms": round(float(np.mean([r["inference_time_ms"] for r in evaluated if r["inference_time_ms"] is not None])), 4) if any(r["inference_time_ms"] is not None for r in evaluated) else None,
        "end_reason_counts": end_reason_counts,
        "referee_status_counts": referee_status_counts,
        "seeds": sorted(r["seed"] for r in evaluated),
    }
    _atomic_write(out_dir / "eval_summary.json", json.dumps(summary, indent=2))
    print("[eval] SUMMARY:", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
