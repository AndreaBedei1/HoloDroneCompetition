"""Aggregate the matched Gen-2 PPO-vs-rules benchmark into article artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stats(rows: list[dict[str, Any]], track: str, controller: str) -> dict[str, Any]:
    subset = [r for r in rows if r["track"] == track and r["controller"] == controller]
    finished = [r for r in subset if r["succeeded"]]
    times = [float(r["completion_time_s"]) for r in finished]
    paths = [float(r["path_length_m"]) for r in finished]
    total_gates = sum(int(r["gate_count"]) for r in subset)
    gates = sum(int(r["gates_completed"]) for r in subset)
    return {
        "track": track, "controller": controller, "runs": len(subset),
        "success": f"{len(finished)}/{len(subset)}",
        "success_rate": round(len(finished) / max(1, len(subset)), 4),
        "gates": f"{gates}/{total_gates}",
        "gate_completion_rate": round(gates / max(1, total_gates), 4),
        "collision_free": f"{sum(int(int(r['collision_events']) == 0) for r in subset)}/{len(subset)}",
        "collision_free_rate": round(sum(int(int(r['collision_events']) == 0) for r in subset) / max(1, len(subset)), 4),
        "out_of_bounds": sum(int(r["out_of_bounds_events"]) for r in subset),
        "wrong_direction": sum(int(r["wrong_direction_crossings"]) for r in subset),
        "missed_gate": sum(int(r["missed_gate_attempts"]) for r in subset),
        "timeout": sum(int(bool(r["timeout"])) for r in subset),
        "time_mean_s": round(statistics.mean(times), 3) if times else None,
        "time_std_s": round(statistics.stdev(times), 3) if len(times) > 1 else (0.0 if times else None),
        "time_median_s": round(statistics.median(times), 3) if times else None,
        "time_min_s": round(min(times), 3) if times else None,
        "time_max_s": round(max(times), 3) if times else None,
        "path_mean_m": round(statistics.mean(paths), 3) if paths else None,
        "path_std_m": round(statistics.stdev(paths), 3) if len(paths) > 1 else (0.0 if paths else None),
        "vision_availability_mean": round(statistics.mean(float(r["vision_availability"]) for r in subset), 4) if subset else None,
        "orientation_availability_mean": round(statistics.mean(float(r["orientation_availability"]) for r in subset), 4) if subset else None,
        "mean_action_jerk": round(statistics.mean(float(r["mean_action_jerk"]) for r in subset), 6) if subset else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    args = p.parse_args()
    rows: list[dict[str, Any]] = []
    for path in sorted(args.raw.glob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")).get("rows", []))
    tracks = ["horseshoe_bay", "vertical_serpent", "mixed_endurance"]
    controllers = ["rule_gate_center_then_commit", "ppo_25k"]
    table = [stats(rows, t, c) for t in tracks for c in controllers]
    all_stats = []
    for c in controllers:
        subset = [r for r in rows if r["controller"] == c]
        finished = [r for r in subset if r["succeeded"]]
        times = [float(r["completion_time_s"]) for r in finished]
        paths = [float(r["path_length_m"]) for r in finished]
        all_stats.append({
            "controller": c, "runs": len(subset),
            "completion_rate": round(len(finished) / max(1, len(subset)), 4),
            "gate_completion_rate": round(sum(int(r["gates_completed"]) for r in subset) / max(1, sum(int(r["gate_count"]) for r in subset)), 4),
            "collision_free_runs": sum(int(int(r["collision_events"]) == 0) for r in subset),
            "mean_completion_time_s": round(statistics.mean(times), 3) if times else None,
            "median_completion_time_s": round(statistics.median(times), 3) if times else None,
            "mean_path_length_m": round(statistics.mean(paths), 3) if paths else None,
        })
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "final_benchmark_raw.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    (args.out / "final_summary.json").write_text(json.dumps({"by_track_controller": table, "overall": all_stats}, indent=2), encoding="utf-8")
    with (args.out / "final_benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader(); writer.writerows(table)
    md = ["Track | Controller | Success | Gates | Collision-free | Time mean ± std (s) | Path mean ± std (m)", "---|---|---:|---:|---:|---:|---:"]
    for r in table:
        tm = "n/a" if r["time_mean_s"] is None else f"{r['time_mean_s']:.1f} ± {r['time_std_s']:.1f}"
        pm = "n/a" if r["path_mean_m"] is None else f"{r['path_mean_m']:.1f} ± {r['path_std_m']:.1f}"
        md.append(f"{r['track']} | {r['controller']} | {r['success']} | {r['gates']} | {r['collision_free']} | {tm} | {pm}")
    (args.out / "final_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    tex = [r"\begin{tabular}{llrrrr}", r"Track & Controller & Success & Gates & Collision-free & Time (s) \\", r"\hline"]
    for r in table:
        tm = "--" if r["time_mean_s"] is None else f"{r['time_mean_s']:.1f} $\\pm$ {r['time_std_s']:.1f}"
        tex.append(f"{r['track']} & {r['controller']} & {r['success']} & {r['gates']} & {r['collision_free']} & {tm} \\\\")
    tex.append(r"\end{tabular}")
    (args.out / "final_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    manifest = {
        "schema": "final_27d_controller_manifest_v1",
        "selected_controller": "ppo_25k",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        "rejected_checkpoints": {"ppo_50k": {"path": str(args.checkpoint.parent / "candidate_50k.zip"), "reason": "closed-loop regression: 36/51 gates in matched evaluation; repeat Vertical passed but did not remove H/V regression"}},
        "parent_warmstart": "artifacts_gen2/bc_27d_clean_20260907/parent_warmstart_27d.zip",
        "observation_contract": "onboard_local_transition_27d_v1",
        "controllers": ["ppo_25k", "rule_gate_center_then_commit"],
        "tracks": tracks, "seeds": [66001, 66002, 66003],
        "adapter": "holoocean", "allow_fallback": False, "currents": "disabled",
        "fog": {"enabled": True, "density": 5.0, "start_distance_m": 1.0, "color_rgb": [0.4, 0.6, 1.0]},
        "inference_privileged_state": False,
        "selection_criterion": ["completion", "gates", "safety", "time"],
    }
    (args.out / "final_controller_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
