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
    "current_profile",
    "benchmark_task_override",
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


def currents_summary(track: str, current_profile: Optional[str],
                     benchmark_task: Optional[str] = None) -> Dict:
    """Describe the currents actually applied after resolving ``current_profile``.

    Returns the resolved net current vector and per-current descriptors so a run manifest can
    prove ``--current-profile none`` produced zero applied current (``currents_actual`` all 0),
    without modifying the official JSON track (a runtime override only). ``benchmark_task`` may
    be overridden (e.g. a ``current_gate`` circuit run current-free becomes ``clean_gate``);
    the geometry, gate order, laps and referee are unchanged by this validation-mode override.
    """
    from marine_race_arena.config.loader import describe_current_profile, load_track_config

    config = load_track_config(track, current_profile=current_profile, benchmark_task=benchmark_task)
    net = [0.0, 0.0, 0.0]
    for current in config.currents:
        if current.type == "constant":
            vel = current.params.get("velocity", [0.0, 0.0, 0.0])
            for i in range(3):
                net[i] += float(vel[i]) if i < len(vel) else 0.0
    return {
        "current_profile_requested": current_profile,
        "selected_current_profile": config.selected_current_profile,
        "currents_actual": [round(v, 6) for v in net],
        "n_currents": len(config.currents),
        "current_descriptors": describe_current_profile(config),
        "currents_are_zero": (len(config.currents) == 0 and net == [0.0, 0.0, 0.0]),
    }


def build_requested_config(args, *, model_sha256, randomization_spec) -> Dict:
    """The experiment-identity + provenance fields captured in the manifest."""
    observation_version = OBS_ENCODING_VERSION
    if args.controller == "rl_multigate_controller":
        from marine_race_arena.learning.config_v3 import OBS_ENCODING_VERSION_V3

        observation_version = OBS_ENCODING_VERSION_V3
    return {
        "controller_name": args.controller,
        "model_path": args.model,
        "model_sha256": model_sha256,
        "track_path": args.track,
        "track_sha256": sha256_file(args.track),
        "adapter_requested": args.adapter,
        "fallback_allowed": bool(args.allow_fallback),
        "current_profile": args.current_profile,
        "benchmark_task_override": args.benchmark_task,
        "randomization_enabled": bool(args.randomize),
        "randomization_spec": randomization_spec,
        "dt": float(args.dt),
        "duration_s": (float(args.duration) if args.duration is not None else None),
        "max_steps": None,  # the race runner is time-deadline based, not step-capped
        "observation_encoding_version": observation_version,
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
    parser.add_argument("--current-profile", default=None,
                        help="Runtime current override (none|medium|strong). 'none' disables all "
                             "currents; the manifest records currents_actual and asserts they are zero.")
    parser.add_argument("--benchmark-task", default=None,
                        help="Runtime benchmark-task validation override (e.g. clean_gate). Needed to "
                             "run a current_gate circuit current-free; geometry/gates/laps/referee unchanged.")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--randomize", action="store_true", help="Apply Stage-2 start randomization (held-out seeds).")
    parser.add_argument("--force-new", action="store_true",
                        help="Start a fresh experiment; an existing output directory is moved to a timestamped backup.")
    args = parser.parse_args(argv)

    if args.controller == "rl_multigate_controller":
        if not args.model:
            parser.error("--controller rl_multigate_controller requires --model")
        from marine_race_arena.learning.model_contract_v3 import validate_v3_model

        validate_v3_model(args.model)

    start_randomization = None
    randomization_spec = None
    if args.randomize:
        from marine_race_arena.learning.curriculum import STAGE2_RANDOMIZATION

        start_randomization = STAGE2_RANDOMIZATION
        randomization_spec = asdict(STAGE2_RANDOMIZATION)

    model_sha256 = sha256_file(args.model) if args.model else None
    requested = build_requested_config(args, model_sha256=model_sha256, randomization_spec=randomization_spec)

    # Resolve + verify the applied currents. A current-free run MUST prove currents_actual == 0.
    currents = currents_summary(args.track, args.current_profile, args.benchmark_task)
    if args.current_profile is not None and str(args.current_profile).strip().lower() == "none" \
            and not currents["currents_are_zero"]:
        print(f"[eval] ABORT: --current-profile none did not disable currents: {currents}")
        return 3
    print(f"[eval] currents: profile={args.current_profile!r} -> actual={currents['currents_actual']} "
          f"(n={currents['n_currents']}, zero={currents['currents_are_zero']})")

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
        "code_sha": git_sha(),
        **requested,
        "model_bytes": (Path(args.model).stat().st_size if args.model and Path(args.model).exists() else None),
        "currents": currents,
        "currents_actual": currents["currents_actual"],
        "adapter_actual": (existing_manifest or {}).get("adapter_actual"),
        "fallback_used": (existing_manifest or {}).get("fallback_used"),
        "runtime_constraints": {
            "rule_action_weight": (
                0 if args.controller == "rl_multigate_controller" else None
            ),
            "hybrid_blending": (
                False if args.controller == "rl_multigate_controller" else None
            ),
            "rule_controller_not_instantiated": (
                True if args.controller == "rl_multigate_controller" else None
            ),
        },
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
            current_profile=args.current_profile,
            benchmark_task=args.benchmark_task,
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
            "mean_abs_sway_action": r.mean_abs_sway_action,
            "mean_abs_yaw_action": r.mean_abs_yaw_action,
            "mean_action_jerk": r.mean_action_jerk,
            "action_oscillation_rate": r.action_oscillation_rate,
            "deterministic_runtime_intervention_count": (
                r.deterministic_runtime_intervention_count
            ),
            "previous_gate_returns": r.previous_gate_returns,
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
        manifest["fallback_used"] = (
            args.adapter != "fallback" and r.adapter_used == "fallback"
        )
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
        "mean_abs_sway_action": round(
            float(np.mean([r.get("mean_abs_sway_action", 0.0) for r in evaluated])), 4
        ) if evaluated else 0.0,
        "mean_abs_yaw_action": round(
            float(np.mean([r.get("mean_abs_yaw_action", 0.0) for r in evaluated])), 4
        ) if evaluated else 0.0,
        "mean_action_jerk": round(
            float(np.mean([r.get("mean_action_jerk", 0.0) for r in evaluated])), 4
        ) if evaluated else 0.0,
        "mean_action_oscillation_rate": round(
            float(np.mean([r.get("action_oscillation_rate", 0.0) for r in evaluated])), 4
        ) if evaluated else 0.0,
        "total_deterministic_runtime_interventions": int(
            sum(
                    r.get("deterministic_runtime_intervention_count") or 0
                for r in evaluated
            )
        ),
        "total_previous_gate_returns": int(
            sum(r.get("previous_gate_returns") or 0 for r in evaluated)
        ),
        "end_reason_counts": end_reason_counts,
        "referee_status_counts": referee_status_counts,
        "seeds": sorted(r["seed"] for r in evaluated),
    }
    _atomic_write(out_dir / "eval_summary.json", json.dumps(summary, indent=2))
    print("[eval] SUMMARY:", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
