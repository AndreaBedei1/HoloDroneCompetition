"""Build the compact, auditable package for the final common benchmark.

The heavy artifacts (per-episode trajectories, onboard-camera videos, PPO ZIPs)
stay under git-ignored ``results/rl``.  This builder copies only the compact
measurements, the report, the plots and the provenance needed to audit the
comparison and the checkpoint recommendation.

Usage::

    python -m marine_race_arena.learning.build_final_benchmark_package \
        --run results/rl/final_benchmark/<name>
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from marine_race_arena.learning.provenance import git_sha, now_utc, sha256_file

PUB = Path("results/rl_public/final_benchmark")

# Compact artifacts copied verbatim. Videos and per-episode trajectories are
# deliberately excluded: the repository does not track binaries or bulk traces.
_COPY_FILES = (
    "episodes.csv",
    "episodes.json",
    "aggregate_by_group.csv",
    "paired_comparisons.csv",
    "aggregate_report.json",
    "final_benchmark_report.md",
    "suite_manifest.json",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def build(run_dir: str | Path, *, out_dir: str | Path = PUB) -> Path:
    run = Path(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []
    for name in _COPY_FILES:
        source = run / name
        if source.exists():
            shutil.copy2(source, out / name)
            copied.append(name)

    plots_source = run / "plots"
    plots: List[str] = []
    if plots_source.exists():
        target = out / "plots"
        target.mkdir(parents=True, exist_ok=True)
        for png in sorted(plots_source.glob("*.png")):
            shutil.copy2(png, target / png.name)
            plots.append(png.name)

    tracks_source = run / "generated_tracks"
    tracks: List[str] = []
    if tracks_source.exists():
        target = out / "generated_tracks"
        target.mkdir(parents=True, exist_ok=True)
        for track in sorted(tracks_source.glob("*.json")):
            shutil.copy2(track, target / track.name)
            tracks.append(track.name)

    manifest = _load(run / "suite_manifest.json") if (run / "suite_manifest.json").exists() else {}
    report_path = run / "aggregate_report.json"
    report = _load(report_path) if report_path.exists() else {}
    videos = sorted(p.name for p in (run / "artifacts").glob("*.mp4")) if (run / "artifacts").exists() else []
    trajectories = (
        len(list((run / "artifacts").glob("*.trajectory.json")))
        if (run / "artifacts").exists()
        else 0
    )

    provenance = {
        "schema_version": "final_benchmark_package_v1",
        "created_utc": now_utc(),
        "git_sha": git_sha(),
        "source_run": str(run.as_posix()),
        "suite_git_sha": manifest.get("git_sha"),
        "adapter": manifest.get("adapter"),
        "current_profile": manifest.get("current_profile"),
        "dt": manifest.get("dt"),
        "episodes_planned": manifest.get("total_episodes"),
        "episodes_run": report.get("suite", {}).get("total_episodes_run"),
        "controllers": [
            {
                "key": entry.get("key"),
                "kind": entry.get("kind"),
                "policy_only": entry.get("policy_only"),
                "separate_reporting": entry.get("separate_reporting"),
                "model": entry.get("model"),
                "model_sha256": entry.get("model_sha256"),
            }
            for entry in manifest.get("controllers", [])
        ],
        "seeds": {
            "all": manifest.get("all_seeds"),
            "reused": manifest.get("reused_seeds"),
            "holdout": manifest.get("holdout_seeds"),
        },
        "recommendation": report.get("recommendation"),
        "copied_files": copied,
        "file_sha256": {
            name: sha256_file(str(out / name)) for name in copied
        },
        "plots": plots,
        "generated_tracks": tracks,
        "excluded_from_package": {
            "videos": videos,
            "video_location": str((run / "artifacts").as_posix()),
            "per_episode_trajectories": trajectories,
            "reason": (
                "the repository does not track binaries or bulk per-step traces; "
                "they stay in the git-ignored run directory"
            ),
        },
        "reproduce": (
            "conda run -n marine_race_rl python -m "
            "marine_race_arena.learning.final_benchmark run --out "
            f"{run.as_posix()} --workers 6 --episodes-per-case 8 "
            "--official-episodes 5 --adapter holoocean --current-profile none "
            "--video --video-stride 5"
        ),
    }
    _write(out / "package_manifest.json", provenance)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default=str(PUB))
    args = parser.parse_args(argv)
    out = build(args.run, out_dir=args.out)
    print(f"[package] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
