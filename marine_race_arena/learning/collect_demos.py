"""Collect real-HoloOcean expert demonstrations for behavioral cloning.

Crash-safe and truly resumable: each episode is written to its own
``episodes/ep_<seed>.npz`` (atomic temp-then-rename), and the combined
:class:`BCDataset` plus a ``collection_manifest.json`` are rebuilt after every
episode. Re-running the same command loads the manifest, verifies the run is
compatible (track/controller/adapter/randomization/observation-encoding), skips
already-recorded seeds and appends only new ones. An incompatible resume is
refused unless ``--force-new`` (or a new output directory) is given. Intended to be
run from the ``marine_race_rl`` environment.

Usage:
    python -m marine_race_arena.learning.collect_demos --track <path> \
        --seeds 0-29 --out results/rl/stage1/demos --adapter holoocean [--randomize]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from marine_race_arena.learning.config import OBS_ENCODING_VERSION
from marine_race_arena.learning.dataset import BCDataset
from marine_race_arena.learning.provenance import git_sha, now_utc, package_versions, sha256_file
from marine_race_arena.learning.trajectory_recorder import EpisodeRecord, record_episode

MANIFEST_NAME = "collection_manifest.json"
DATASET_NAME = "stage1_demos.npz"
EPISODES_DIR = "episodes"


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


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def _run_signature(args, track_hash: str, randomization) -> Dict:
    return {
        "track": args.track,
        "track_sha256": track_hash,
        "controller": args.controller,
        "adapter": args.adapter,
        "allow_fallback": bool(args.allow_fallback),
        "randomized": bool(args.randomize),
        "randomization_spec": (dataclasses.asdict(randomization) if randomization is not None else None),
        "obs_encoding_version": OBS_ENCODING_VERSION,
    }


def _incompatibilities(existing: Dict, current: Dict) -> List[str]:
    keys = ["track", "track_sha256", "controller", "adapter", "allow_fallback",
            "randomized", "randomization_spec", "obs_encoding_version"]
    return [k for k in keys if existing.get(k) != current.get(k)]


def _load_existing_records(episodes_dir: Path) -> Dict[int, EpisodeRecord]:
    records: Dict[int, EpisodeRecord] = {}
    if not episodes_dir.exists():
        return records
    for ep_file in sorted(episodes_dir.glob("ep_*.npz")):
        try:
            rec = EpisodeRecord.load_npz(ep_file)
            records[int(rec.seed)] = rec
        except Exception:  # pragma: no cover - skip corrupt file, will be re-recorded
            continue
    return records


def _rebuild_dataset(records: Dict[int, EpisodeRecord], dataset_path: Path) -> BCDataset:
    ordered = [records[s] for s in sorted(records)]
    # Re-assign episode_id in seed order so group ids are unique and stable.
    for new_id, rec in enumerate(ordered):
        rec.episode_id = new_id
    dataset = BCDataset.from_records(ordered)
    dataset.check_integrity()
    # Atomic save: temp name ends in .npz (np.savez_compressed appends .npz otherwise).
    tmp = dataset_path.with_name("_tmp_" + dataset_path.name)
    dataset.save(tmp)
    tmp.replace(dataset_path)
    return dataset


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
    parser.add_argument("--randomize", action="store_true",
                        help="Apply the Stage-2 seeded start-pose/beacon-noise randomization for diversity.")
    parser.add_argument("--force-new", action="store_true",
                        help="Overwrite an incompatible existing collection in --out instead of refusing.")
    args = parser.parse_args(argv)

    start_randomization = None
    if args.randomize:
        from marine_race_arena.learning.curriculum import STAGE2_RANDOMIZATION

        start_randomization = STAGE2_RANDOMIZATION

    out_dir = Path(args.out)
    episodes_dir = out_dir / EPISODES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_NAME
    dataset_path = out_dir / DATASET_NAME

    track_hash = sha256_file(args.track)
    signature = _run_signature(args, track_hash, start_randomization)

    # --- Resume / compatibility --------------------------------------------------
    records = _load_existing_records(episodes_dir)
    failed_seeds: List[int] = []
    created_utc = now_utc()
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing_manifest = {}
        created_utc = existing_manifest.get("created_utc") or created_utc
        existing_sig = existing_manifest.get("signature", {})
        incompat = _incompatibilities(existing_sig, signature) if existing_sig else []
        if incompat and not args.force_new:
            print(f"[collect] ERROR incompatible resume in {out_dir}: differs in {incompat}. "
                  f"Use --force-new or a new --out.")
            return 2
        if incompat and args.force_new:
            print(f"[collect] --force-new: discarding incompatible existing collection ({incompat}).")
            records = {}
            for ep_file in episodes_dir.glob("ep_*.npz"):
                ep_file.unlink()
        failed_seeds = list(existing_manifest.get("failed_seeds", []))

    done_seeds = set(records)
    requested = _parse_seeds(args.seeds)
    print(f"[collect] {len(requested)} requested, {len(done_seeds)} already recorded; "
          f"track={args.track} adapter={args.adapter} randomize={args.randomize}")

    versions = package_versions()
    commit = git_sha()

    def write_manifest(dataset: Optional[BCDataset]):
        manifest = {
            "created_utc": created_utc,
            "updated_utc": now_utc(),
            "git_sha": commit,
            "signature": signature,
            "track": args.track,
            "track_sha256": track_hash,
            "controller": args.controller,
            "adapter_requested": args.adapter,
            "allow_fallback": bool(args.allow_fallback),
            "randomization_enabled": bool(args.randomize),
            "randomization_spec": signature["randomization_spec"],
            "obs_encoding_version": OBS_ENCODING_VERSION,
            "requested_seeds": requested,
            "completed_seeds": sorted(records),
            "failed_seeds": sorted(set(failed_seeds)),
            "total_episodes": len(records),
            "total_steps": int(len(dataset)) if dataset is not None else 0,
            "dataset_path": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path) if dataset_path.exists() else None,
            "python_packages": versions,
            "holoocean_version": versions.get("holoocean"),
        }
        _atomic_write_json(manifest_path, manifest)

    dataset = _rebuild_dataset(records, dataset_path) if records else None
    write_manifest(dataset)

    for seed in requested:
        if seed in done_seeds:
            print(f"[collect] seed={seed} already recorded -- skip")
            continue
        t0 = time.time()
        try:
            rec = record_episode(
                args.track, args.controller, seed=int(seed), dt=args.dt, adapter=args.adapter,
                allow_fallback=args.allow_fallback, max_steps=args.max_steps, official=True,
                episode_id=len(records), start_randomization=start_randomization,
            )
        except Exception as exc:  # pragma: no cover - engine/adapter failure path
            failed_seeds.append(int(seed))
            print(f"[collect] seed={seed} FAILED: {type(exc).__name__}: {exc}")
            write_manifest(dataset)
            continue
        wall = time.time() - t0
        # Persist the episode first (atomic), then rebuild the combined dataset.
        rec.save_npz(episodes_dir / f"ep_{int(seed):05d}.npz")
        records[int(seed)] = rec
        done_seeds.add(int(seed))
        dataset = _rebuild_dataset(records, dataset_path)
        write_manifest(dataset)
        print(f"[collect] seed={seed:>3} status={rec.final_status:<9} gates={rec.gate_crossings} "
              f"steps={rec.length:>4} wall={wall:5.1f}s")

    # --- Final quality summary ---------------------------------------------------
    if not records:
        print("[collect] no episodes recorded.")
        return 0
    dataset = _rebuild_dataset(records, dataset_path)
    write_manifest(dataset)
    finished = [r for r in records.values() if r.final_status == "FINISHED"]
    all_actions = dataset.actions
    summary = {
        "demos": len(records),
        "expert_completion_rate": len(finished) / len(records),
        "total_steps": int(len(dataset)),
        "action_saturation_frac": float(np.mean(np.abs(all_actions) > 0.98)),
        "obs_finite": bool(np.all(np.isfinite(dataset.observations))),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
    }
    (out_dir / "quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[collect] SUMMARY:", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
