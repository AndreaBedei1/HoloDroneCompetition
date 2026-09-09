"""Inspect a demonstration dataset and train a Stage-1 BC policy.

Loads the recorded :class:`BCDataset`, runs pre-training inspection (integrity,
per-axis action stats, feature saturation, mask coverage), trains BC with an
episode-level split, and saves the policy + a training report. No HoloOcean needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from marine_race_arena.learning.bc_train import BCConfig, save_policy, train_bc
from marine_race_arena.learning.config import FEATURE_NAMES
from marine_race_arena.learning.dataset import BCDataset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, nargs="+", help="one or more BCDataset .npz files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = BCDataset.load(args.dataset[0]) if len(args.dataset) == 1 else BCDataset.load_many(args.dataset)
    dataset.check_integrity()

    # --- Pre-training inspection ------------------------------------------------
    obs, act = dataset.observations, dataset.actions
    present_features = [f for f in FEATURE_NAMES if f.endswith("_present")]
    mask_coverage = {
        name: round(float(obs[:, FEATURE_NAMES.index(name)].mean()), 3) for name in present_features
    }
    inspection = {
        "episodes": dataset.num_episodes,
        "steps": len(dataset),
        "obs_finite": bool(np.all(np.isfinite(obs))),
        "act_finite": bool(np.all(np.isfinite(act))),
        "act_in_bounds": bool(np.all(np.abs(act) <= 1.0 + 1e-6)),
        "action_mean": [round(float(x), 4) for x in act.mean(axis=0)],
        "action_std": [round(float(x), 4) for x in act.std(axis=0)],
        "action_saturation_frac": round(float(np.mean(np.abs(act) > 0.98)), 4),
        "obs_feature_saturation_frac": round(float(np.mean(np.abs(obs) > 0.999)), 4),
        "mask_coverage": mask_coverage,
        "seeds": sorted({m.seed for m in dataset.episodes}),
    }
    (out_dir / "dataset_inspection.json").write_text(json.dumps(inspection, indent=2), encoding="utf-8")
    print("[bc] inspection:", json.dumps(inspection, indent=2))

    # --- Train ------------------------------------------------------------------
    config = BCConfig(
        hidden_sizes=tuple(args.hidden),
        max_epochs=args.epochs,
        patience=40,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    policy, history = train_bc(dataset, config, log_csv=str(out_dir / "bc_training_log.csv"))
    best = min(history, key=lambda h: h["val_mse"])
    model_path = out_dir / "best_model.pt"
    save_policy(policy, model_path)

    report = {
        "model_path": str(model_path),
        "epochs_run": len(history),
        "best_epoch": best["epoch"],
        "best_val_mse": round(best["val_mse"], 6),
        "final_train_mse": round(history[-1]["train_mse"], 6),
        "best_val_mse_per_axis": {
            axis: round(best[f"val_mse_{axis}"], 6) for axis in ("surge", "sway", "heave", "yaw")
        },
        "hidden_sizes": list(args.hidden),
        "val_fraction": args.val_fraction,
    }
    (out_dir / "bc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[bc] report:", json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
