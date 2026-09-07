"""Gen-2 PPO pipeline runner with 27-D contract, warm-start, and rare snapshot monitor.

SAFETY NOTICE:
By default, this module operates in DRY-RUN / PREFLIGHT mode.
Execution stops before any training updates occur unless authorized.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
from marine_race_arena.learning.config_local_transition_27d import (
    FEATURE_NAMES_LOCAL_TRANSITION_27D,
    OBS_DIM_LOCAL_TRANSITION_27D,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
)
from marine_race_arena.learning.gen2 import track_fragments as tf
from marine_race_arena.learning.gen2.fog_contract import (
    APPROVED_WATER_FOG,
    assert_approved_water_fog,
    verify_fog_sources,
)
from marine_race_arena.learning.gen2.monitor_snapshot import Gen2MonitorSnapshotWrapper
from marine_race_arena.learning.gen2.recurrent_policy import GEN2_27D_ARCH
from marine_race_arena.learning.gen2.training_status import Gen2TrainingStatusCallback
from marine_race_arena.learning.gen2.transfer_27d import (
    sha256_file,
    transfer_parent_35d_to_27d,
    verify_transfer_27d_tensors,
)
from marine_race_arena.learning.gym_env import MarineRaceGymEnv


SAFE_ENV_COUNT = 6
PARENT_CHECKPOINT_DEFAULT = "results/rl/gen2/best/best_completion_policy.zip"


def make_gen2_27d_env(
    track: str | Path,
    *,
    seed: int = 0,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
    **kwargs: Any,
) -> MarineRaceGymEnv:
    """Production Gymnasium factory for Gen-2 27-D RL.

    Mandatorily enforces adapter='holoocean' and allow_fallback=False.
    """
    if allow_fallback:
        raise ValueError("Production 27-D env factory strictly forbids allow_fallback=True.")
    if adapter != "holoocean":
        raise ValueError(
            f"Production 27-D env factory strictly requires adapter='holoocean'. Got {adapter!r}."
        )
    return MarineRaceGymEnv(
        str(track),
        seed=seed,
        adapter="holoocean",
        allow_fallback=False,
        observation_encoding_version=OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
        **kwargs,
    )


@dataclass(frozen=True)
class TrainingSource27d:
    name: str
    track: str
    kind: str
    path: str
    gate_count: int
    max_steps: int
    initial_body_velocity: Optional[tuple[float, float, float]] = None


def materialize_sources_27d(run_dir: str | Path) -> Tuple[List[TrainingSource27d], Dict[str, Any]]:
    """Build sources: 3 official tracks + 3 contiguous fragments."""
    run_dir = Path(run_dir)
    sources: List[TrainingSource27d] = []
    source_paths: List[Path] = []

    for track in tf.OFFICIAL_TRACKS:
        p = tf.track_path(track)
        assert_approved_water_fog(p)
        data = tf.load_track(track)
        gate_count = len(data["track"]["gate_sequence"])
        sources.append(
            TrainingSource27d(
                name=f"full_{track}",
                track=track,
                kind="full_circuit",
                path=str(p),
                gate_count=gate_count,
                max_steps=max(2000, gate_count * 900),
            )
        )
        source_paths.append(p)

    frag_dir = run_dir / "training_tracks"
    frag_dir.mkdir(parents=True, exist_ok=True)
    weak_gates = {"horseshoe_bay": 9, "vertical_serpent": 10, "mixed_endurance": 4}

    for track in tf.OFFICIAL_TRACKS:
        candidates = [
            f for f in tf.enumerate_fragments(track, lengths=(3, 4))
            if f.start_index < weak_gates[track] - 1 <= f.end_index
        ]
        chosen = candidates[0] if candidates else tf.enumerate_fragments(track, lengths=(3,))[0]
        frag_path = tf.materialize_fragment(chosen, frag_dir / f"{chosen.name}.json")
        assert_approved_water_fog(frag_path)
        sources.append(
            TrainingSource27d(
                name=chosen.name,
                track=track,
                kind="exact_fragment",
                path=str(frag_path),
                gate_count=chosen.length,
                max_steps=max(600, chosen.length * 900),
                initial_body_velocity=tf.inbound_body_velocity(chosen),
            )
        )
        source_paths.append(frag_path)

    fog_report = verify_fog_sources(source_paths)
    return sources, {"fog": fog_report, "source_count": len(sources)}


def prepare_and_verify_pipeline(
    *,
    parent_path: str | Path = PARENT_CHECKPOINT_DEFAULT,
    run_dir: str | Path,
    seed: int = 8400,
) -> Dict[str, Any]:
    """Execute complete preparation and validation of the 27-D pipeline without training."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = run_dir / "monitor_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    parent_path = Path(parent_path)
    parent_hash_before = sha256_file(parent_path)

    # 1. Materialize sources and verify fog
    sources, source_audit = materialize_sources_27d(run_dir)

    # 2. Transfer 35-D -> 27-D warm-start model
    model = transfer_parent_35d_to_27d(parent_path, seed=seed)
    transfer_report = verify_transfer_27d_tensors(parent_path, model)

    if not transfer_report["verified"]:
        raise RuntimeError(f"Warm-start transfer verification failed: {transfer_report}")

    # 3. Save initialized 27-D checkpoint
    init_ckpt = run_dir / "initialized_27d_policy.zip"
    model.save(init_ckpt)

    # 4. Verify parent immutability
    parent_hash_after = sha256_file(parent_path)
    if parent_hash_after != parent_hash_before:
        raise RuntimeError("Parent checkpoint hash altered!")

    pipeline_manifest = {
        "schema_version": "gen2_ppo_27d_preparation_v1",
        "parent_checkpoint": str(parent_path),
        "parent_sha256": parent_hash_before,
        "parent_immutable": True,
        "observation_contract": OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
        "observation_dim": OBS_DIM_LOCAL_TRANSITION_27D,
        "features": list(FEATURE_NAMES_LOCAL_TRANSITION_27D),
        "sources": [asdict(s) for s in sources],
        "source_audit": source_audit,
        "transfer_verification": transfer_report,
        "initialized_checkpoint": str(init_ckpt),
        "snapshots_dir": str(snapshots_dir),
        "training_updates_executed": 0,
        "training_authorized": False,
        "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_file = run_dir / "pipeline_manifest.json"
    manifest_file.write_text(json.dumps(pipeline_manifest, indent=2), encoding="utf-8")
    return pipeline_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and verify the Gen-2 27-D PPO pipeline")
    parser.add_argument("--parent", default=PARENT_CHECKPOINT_DEFAULT)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=8400)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Attempt training. HARD BLOCKED by default.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.train:
        print("[SAFETY ERROR] Training is strictly disabled during pipeline preparation.", file=sys.stderr)
        return 1

    report = prepare_and_verify_pipeline(
        parent_path=args.parent,
        run_dir=args.out,
        seed=args.seed,
    )
    print(json.dumps({
        "status": "prepared",
        "manifest": str(Path(args.out) / "pipeline_manifest.json"),
        "parent_sha256": report["parent_sha256"],
        "training_updates": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
