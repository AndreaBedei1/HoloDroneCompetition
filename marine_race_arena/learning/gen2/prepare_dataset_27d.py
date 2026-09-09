"""Expert demonstration dataset collection pipeline for the Gen-2 27-D contract.

Collects expert demonstrations using ONLY:
- Mandatory approved underwater fog (density=5.0, start=1.0m, color=[0.4, 0.6, 1.0]);
- Updated underwater vision + TemporalVisionTracker;
- Correct gate aperture center and corner detection;
- Correct gate plane yaw from :mod:`gate_pose`;
- Onboard-only 27-D observation contract;
- Expert: :class:`~marine_race_arena.controllers.official_baselines.RuleGateCenterThenCommitController`;
- Tracks: Horseshoe Bay, Vertical Serpent, Mixed Endurance, and exact contiguous fragments.

CRITICAL GUARD:
Long collection campaigns are blocked by default.
Only short micro-smokes (<= 5 steps) are permitted for wiring verification.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
from marine_race_arena.controllers.official_baselines import (
    RuleGateCenterThenCommitController,
)
from marine_race_arena.learning.config import ACTION_AXES, ACTION_DIM
from marine_race_arena.learning.config_local_transition_27d import (
    FEATURE_NAMES_LOCAL_TRANSITION_27D,
    OBS_DIM_LOCAL_TRANSITION_27D,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
)
from marine_race_arena.learning.episode import RaceEpisode
from marine_race_arena.learning.gen2 import track_fragments as tf
from marine_race_arena.learning.gen2.fog_contract import (
    APPROVED_WATER_FOG,
    assert_approved_water_fog,
    verify_fog_sources,
)
from marine_race_arena.learning.observation_encoder_local_transition_27d import (
    encode_observation_local_transition_27d,
)
from marine_race_arena.learning.tracker_context_local_transition_27d import (
    OnboardLocalTransition27dContextTracker,
)


MAX_SMOKE_STEPS = 10


def collect_expert_27d_rollout(
    track_path: str | Path,
    *,
    seed: int = 0,
    adapter: str = "fallback",
    allow_fallback: bool = True,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
    max_steps: int = MAX_SMOKE_STEPS,
    allow_long_collection: bool = False,
    initial_body_velocity: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """Run one expert rollout recording 27-D observations and expert actions."""
    if max_steps > MAX_SMOKE_STEPS and not allow_long_collection:
        raise RuntimeError(
            f"Long dataset collection is strictly blocked during preparation phase "
            f"(requested {max_steps} steps > MAX_SMOKE_STEPS={MAX_SMOKE_STEPS}). "
            f"Pass allow_long_collection=True only when authorized."
        )

    if not allow_fallback and adapter != "holoocean":
        raise ValueError(
            f"Production 27-D dataset collection strictly requires adapter='holoocean' and allow_fallback=False. Got {adapter!r}."
        )

    track_path = Path(track_path)
    assert_approved_water_fog(track_path)

    episode = RaceEpisode(
        str(track_path),
        seed=int(seed),
        adapter=adapter,
        allow_fallback=allow_fallback,
        max_steps=int(max_steps),
        official=True,
        current_profile="none",
        benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
    )

    try:
        raw_obs = episode.reset()
        actual_adapter = episode.actual_adapter
        fallback_used = episode.fallback_used
        if not allow_fallback and fallback_used:
            raise RuntimeError("Fallback adapter was used when allow_fallback is False!")

        if initial_body_velocity is not None:
            tf.apply_initial_body_velocity(episode, initial_body_velocity)
            raw_obs = episode._build_observation()

        ctx_cfg = episode.context.config
        total_beacons = max(1, len(ctx_cfg.track.gate_sequence))
        laps = max(1, int(ctx_cfg.race.laps))

        tracker = OnboardLocalTransition27dContextTracker(
            total_beacons=total_beacons, laps=laps
        )
        tracker.reset(raw_obs)

        expert = RuleGateCenterThenCommitController()
        expert.reset(
            {
                "participant_id": episode.participant_id,
                "total_beacons": total_beacons,
                "laps": laps,
            }
        )

        obs_list: List[np.ndarray] = []
        action_list: List[np.ndarray] = []
        prev_action = np.zeros(ACTION_DIM, dtype=np.float32)

        for step_idx in range(max_steps):
            command = expert.step(raw_obs)
            action_vec = np.asarray(
                [float(command.get(axis, 0.0)) for axis in ACTION_AXES],
                dtype=np.float32,
            )

            # Build 27-D observation
            context = tracker.context(
                raw_obs, dt=episode.dt, prev_action=prev_action.tolist()
            )
            encoded = encode_observation_local_transition_27d(raw_obs, context)

            if encoded.shape != (OBS_DIM_LOCAL_TRANSITION_27D,):
                raise ValueError(
                    f"encoded shape {encoded.shape} != ({OBS_DIM_LOCAL_TRANSITION_27D},)"
                )
            if not np.isfinite(encoded).all():
                raise ValueError("encoded observation contains NaN or inf")

            obs_list.append(encoded)
            action_list.append(action_vec)
            prev_action = action_vec

            step_res = episode.step(command)
            raw_obs = step_res.observation

            if step_res.terminated or step_res.truncated:
                break

        return {
            "track": str(track_path),
            "requested_adapter": adapter,
            "actual_adapter": actual_adapter,
            "fallback_used": fallback_used,
            "steps": len(obs_list),
            "observations_shape": (len(obs_list), OBS_DIM_LOCAL_TRANSITION_27D),
            "actions_shape": (len(action_list), ACTION_DIM),
            "fog_verified": True,
            "contract": OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
        }
    finally:
        episode.close()


def prepare_dataset_sources(output_dir: str | Path) -> Dict[str, Any]:
    """Prepare source configs for the 3 official tracks plus real fragments."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sources = []
    source_paths = []

    # Official circuits
    for track_name in tf.OFFICIAL_TRACKS:
        p = tf.track_path(track_name)
        assert_approved_water_fog(p)
        sources.append({"name": track_name, "path": str(p), "kind": "full_circuit"})
        source_paths.append(p)

    # Real contiguous fragments
    frag_dir = out / "fragments"
    frag_dir.mkdir(parents=True, exist_ok=True)
    for track_name in tf.OFFICIAL_TRACKS:
        frags = tf.enumerate_fragments(track_name, lengths=(3, 4))
        if frags:
            sample_frag = frags[0]
            frag_path = tf.materialize_fragment(
                sample_frag, frag_dir / f"{sample_frag.name}.json"
            )
            assert_approved_water_fog(frag_path)
            sources.append({
                "name": sample_frag.name,
                "path": str(frag_path),
                "kind": "fragment",
                "fragment": sample_frag.as_dict(),
            })
            source_paths.append(frag_path)

    fog_audit = verify_fog_sources(source_paths)
    manifest = {
        "schema_version": "gen2_dataset_preparation_27d_v1",
        "observation_contract": OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
        "observation_dim": OBS_DIM_LOCAL_TRANSITION_27D,
        "expert": "rule_gate_center_then_commit",
        "fog": fog_audit,
        "sources": sources,
    }
    manifest_path = out / "dataset_sources_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare 27-D dataset sources or collect expert micro-rollout")
    parser.add_argument("--manifest-out", default=None, help="Directory to materialize fragment sources and manifest")
    parser.add_argument("--track", default=None, help="Track name or path to collect expert rollout on")
    parser.add_argument("--steps", type=int, default=MAX_SMOKE_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter", default="holoocean", choices=["holoocean", "fallback"])
    parser.add_argument("--allow-fallback", action="store_true", default=False)
    parser.add_argument("--allow-long-collection", action="store_true", default=False)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.manifest_out:
        manifest = prepare_dataset_sources(args.manifest_out)
        print(json.dumps({"manifest": str(Path(args.manifest_out) / "dataset_sources_manifest.json")}, indent=2))
        return 0

    if args.track:
        track_path = tf.track_path(args.track) if args.track in tf.OFFICIAL_TRACKS else Path(args.track)
        res = collect_expert_27d_rollout(
            track_path,
            seed=args.seed,
            adapter=args.adapter,
            allow_fallback=args.allow_fallback,
            max_steps=args.steps,
            allow_long_collection=args.allow_long_collection,
        )
        print(json.dumps(res, indent=2))
        return 0

    print("Please specify --manifest-out or --track.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
