"""Derive the compact, auditable public Stage-2 PPO-diagnostic package from run dirs.

Reads the (git-ignored) heavy Stage-2 run dirs and the KL-calibration runs and writes only
compact, audit-relevant files under ``results/rl_public/stage2/ppo_diagnostic_5k/``. It
fabricates nothing. Large SB3 model ZIPs are NOT committed -- only hashes + reproduce.txt.

    python -m marine_race_arena.learning.build_stage2_package \
        --bcinit  <run> --scratch-controlled <run> --scratch-default <run> \
        --calibration <cal_run> [<cal_run> ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from marine_race_arena.learning.provenance import git_sha, now_utc, sha256_file
from marine_race_arena.learning.seed_registry import registry_dict

PUB = Path("results/rl_public/stage2/ppo_diagnostic_5k")


def _load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else None


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _read_csv(path: Path) -> List[Dict]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as h:
        return list(csv.DictReader(h))


def _model_hashes(run_dir: Path) -> Dict:
    out: Dict = {"note": "SB3 model ZIPs stay under results/rl/ (git-ignored); hashes + reproduce.txt published."}
    for rel in ("best_model/best_model.zip", "final_model.zip"):
        p = run_dir / rel
        if p.exists():
            out[Path(rel).name] = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
    return out


def summarize_arm(run_dir: Path, out_sub: Path) -> Dict:
    run_dir = Path(run_dir)
    rc = _load(run_dir / "run_config.json") or {}
    initial = _load(run_dir / "evaluation" / "initial_eval.json") or {}
    best = _load(run_dir / "best_model" / "best_metrics.json") or {}
    env = _load(run_dir / "environment.json") or {}
    status = _load(run_dir / "run_status.json") or {}
    action_std = _load(run_dir / "action_std.json")
    eval_rows = _read_csv(run_dir / "evaluation" / "eval.csv")
    kl_rows = _read_csv(run_dir / "training" / "ppo_update_metrics.csv")

    _write(out_sub / "run_config.json", rc)
    # Drop the (potentially large) per-seed detail from the published initial_eval.
    _write(out_sub / "initial_eval.json", {k: v for k, v in initial.items() if k != "per_seed"})
    _write(out_sub / "best_eval.json", best)
    if eval_rows:
        shutil.copyfile(run_dir / "evaluation" / "eval.csv", out_sub / "eval_history.csv")
        _write(out_sub / "final_eval.json", {"source": "last row of evaluation/eval.csv", **eval_rows[-1]})
    if kl_rows:
        shutil.copyfile(run_dir / "training" / "ppo_update_metrics.csv", out_sub / "kl_metrics.csv")
    _write(out_sub / "action_statistics.json", {
        "action_std_config": action_std,
        "final_policy_std": status.get("kl_summary", {}).get("final_policy_std"),
        "final_action_saturation": status.get("kl_summary", {}).get("final_action_saturation"),
        "initial_action_saturation": initial.get("mean_action_saturation"),
        "initial_action_smoothness": initial.get("mean_action_smoothness"),
    })
    _write(out_sub / "model_hashes.json", _model_hashes(run_dir))
    if (run_dir / "reproduce.txt").exists():
        shutil.copyfile(run_dir / "reproduce.txt", out_sub / "reproduce.txt")

    return {
        "arm": rc.get("arm"), "algorithm": rc.get("algorithm"), "run_dir": str(run_dir),
        "run_status": status.get("run_status"), "total_timesteps": env.get("final_num_timesteps"),
        "wall_clock_s": env.get("wall_clock_s"), "adapter_actual": env.get("adapter_actual"),
        "initial_completion": initial.get("completion_rate"),
        "initial_interior_completion": initial.get("interior_completion"),
        "initial_extreme_completion": initial.get("extreme_completion"),
        "best_completion": best.get("completion_rate"),
        "best_interior_completion": best.get("interior_completion"),
        "best_extreme_completion": best.get("extreme_completion"),
        "final_completion": (float(eval_rows[-1]["completion_rate"]) if eval_rows and eval_rows[-1].get("completion_rate") not in (None, "", "None") else None),
        "max_approx_kl": status.get("kl_summary", {}).get("max_approx_kl"),
        "final_action_saturation": status.get("kl_summary", {}).get("final_action_saturation"),
    }


def build_calibration_section(cal_dirs: List[Path]) -> Dict:
    rows = []
    for d in cal_dirs:
        d = Path(d)
        rc = _load(d / "run_config.json") or {}
        initial = _load(d / "evaluation" / "initial_eval.json") or {}
        best = _load(d / "best_model" / "best_metrics.json") or {}
        status = _load(d / "run_status.json") or {}
        eval_rows = _read_csv(d / "evaluation" / "eval.csv")
        ppo = rc.get("ppo_kwargs", {})
        kl = status.get("kl_summary", {})
        rows.append({
            "run_dir": str(d), "config_preset": (rc.get("bc_action_std_config") or {}),
            "learning_rate": ppo.get("learning_rate"), "n_epochs": ppo.get("n_epochs"),
            "clip_range": ppo.get("clip_range"), "action_std": (rc.get("action_std") or {}).get("std_per_axis"),
            "run_status": status.get("run_status"), "max_approx_kl": kl.get("max_approx_kl"),
            "final_approx_kl": kl.get("final_approx_kl"), "final_clip_fraction": kl.get("final_clip_fraction"),
            "final_action_saturation": kl.get("final_action_saturation"),
            "initial_completion": initial.get("completion_rate"),
            "final_completion": (float(eval_rows[-1]["completion_rate"]) if eval_rows and eval_rows[-1].get("completion_rate") not in (None, "", "None") else None),
            "best_at_timestep_zero": (best.get("timesteps") == 0),
        })
    # A config passes if COMPLETED, KL <= its max, low saturation, completion stayed high.
    def _passed(r):
        return (r["run_status"] == "COMPLETED" and (r["max_approx_kl"] or 1) <= 0.02
                and (r["final_action_saturation"] or 0) < 0.05 and (r["final_completion"] or 0) >= 1.0)
    for r in rows:
        r["passed_calibration"] = _passed(r)
    selected = next((r for r in rows if r["passed_calibration"]), None)
    if rows:
        with (PUB / "calibration" / "compared_configs.csv").open("w", newline="", encoding="utf-8") as h:
            fields = list(rows[0].keys())
            w = csv.DictWriter(h, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
    _write(PUB / "calibration" / "selected_config.json", {
        "selected": selected, "criterion": "COMPLETED, max approx_kl <= 0.02, action saturation < 5%, completion 100% on the calibration set",
        "note": "Select the SAFEST passing configuration, not the lowest training loss.",
    })
    return {"configs": rows, "selected": selected}


def build_comparison(arms: Dict[str, Dict]) -> None:
    _write(PUB / "comparison" / "overall_comparison.json", {
        "generated_utc": now_utc(), "commit": git_sha(),
        "primary_comparison": "bcinit_controlled vs scratch_controlled (same exploration std; only the weights differ)",
        "scratch_default_role": "exploration-variance diagnostic only (SB3 default ~1.0 std); NOT a proof about BC weights",
        "arms": arms,
    })
    fields = ["arm", "run_status", "initial_completion", "best_completion", "final_completion",
              "initial_extreme_completion", "best_extreme_completion", "max_approx_kl", "final_action_saturation", "wall_clock_s"]
    with (PUB / "comparison" / "overall_comparison.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        for a in arms.values():
            w.writerow({k: a.get(k) for k in fields})
    _write(PUB / "comparison" / "interior_vs_extreme.json", {
        "note": "Completion split by randomization region; the frozen BC failures were extreme-corner.",
        "arms": {name: {"initial_interior": a.get("initial_interior_completion"),
                        "initial_extreme": a.get("initial_extreme_completion"),
                        "best_interior": a.get("best_interior_completion"),
                        "best_extreme": a.get("best_extreme_completion")} for name, a in arms.items()},
    })
    bc = arms.get("bcinit_controlled", {})
    _write(PUB / "comparison" / "failure_analysis.json", {
        "note": "Where each arm still fails after 5,000 steps (development seeds).",
        "bcinit_controlled": {"best_extreme_completion": bc.get("best_extreme_completion"),
                              "best_completion": bc.get("best_completion")},
        "extreme_corner_definition": {"lateral_m": ">= 0.8", "yaw_deg": ">= 12"},
        "corrective_demo_tooling": "marine_race_arena.learning.extreme_corner_demos (prepare-only)",
    })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcinit", required=True)
    parser.add_argument("--scratch-controlled", required=True)
    parser.add_argument("--scratch-default", default=None)
    parser.add_argument("--calibration", nargs="*", default=[])
    args = parser.parse_args(argv)

    PUB.mkdir(parents=True, exist_ok=True)
    arms: Dict[str, Dict] = {}
    arms["bcinit_controlled"] = summarize_arm(Path(args.bcinit), PUB / "bcinit_controlled")
    arms["scratch_controlled"] = summarize_arm(Path(args.scratch_controlled), PUB / "scratch_controlled")
    if args.scratch_default:
        arms["scratch_default"] = summarize_arm(Path(args.scratch_default), PUB / "scratch_default")

    cal = build_calibration_section([Path(d) for d in args.calibration]) if args.calibration else {"configs": [], "selected": None}
    build_comparison(arms)
    _write(PUB / "seed_registry_snapshot.json", registry_dict())
    _write(PUB / "experiment_manifest.json", {
        "generated_utc": now_utc(), "commit": git_sha(),
        "title": "Stage-2 randomized PPO diagnostic (5,000 steps) -- BC-init vs scratch, KL-safe",
        "disclaimers": [
            "These are 5,000-step DIAGNOSTICS, not final scientific results.",
            "Development seeds (1410-1419) were used for checkpoint selection.",
            "The reserved final seed ranges (1500-1549 fixed, 1550-1599 randomized) remain untouched.",
            "PPO improvement over BC is NOT established until a new held-out evaluation shows it.",
        ],
        "calibration_selected": cal.get("selected"),
        "arms": {name: {"run_status": a.get("run_status"), "best_completion": a.get("best_completion"),
                        "best_extreme_completion": a.get("best_extreme_completion")} for name, a in arms.items()},
    })
    (PUB / "README.md").write_text(
        "# Stage-2 Randomized PPO Diagnostic (5,000 steps)\n\n"
        "**These are 5,000-step diagnostics, not final scientific results.** Development seeds "
        "1410-1419 were used for checkpoint selection; the reserved final ranges (1500-1549, "
        "1550-1599) remain untouched. PPO improvement over BC is not established until a new "
        "held-out evaluation shows it.\n\n"
        "- **Primary comparison:** `bcinit_controlled` vs `scratch_controlled` (identical exploration "
        "std and hyperparameters; only the initial weights differ).\n"
        "- **`scratch_default`** is an exploration-variance diagnostic (SB3's ~1.0 default std), not a "
        "proof about BC weights.\n"
        "- Completion is split into interior vs **extreme-corner** (|lateral| >= 0.8 m and |yaw| >= 12 "
        "deg) -- where the frozen BC evaluation failed.\n\n"
        "See `calibration/` (KL-safe config selection), each arm's folder (run config, timestep-zero + "
        "best + final eval, KL metrics, action statistics, model hashes, reproduce command) and "
        "`comparison/` (overall + interior-vs-extreme + failure analysis). SB3 model ZIPs are not "
        "committed (hashes + reproduce.txt only).\n",
        encoding="utf-8",
    )
    print("[stage2] wrote", PUB)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
