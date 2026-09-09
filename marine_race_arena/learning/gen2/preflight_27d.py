"""No-learning preflight and perception audit for the Gen-2 27-D pipeline.

Runs a short audit (zero learning updates) checking:
- Mandatory approved underwater fog on all official tracks;
- 27-D observation shape, finiteness, and bounds;
- Gate detector and gate plane orientation availability;
- Sensor and derivative validity;
- Parent checkpoint immutability.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
from marine_race_arena.learning.config import ACTION_AXES, ACTION_DIM
from marine_race_arena.learning.config_local_transition_27d import (
    FEATURE_BOUNDS_LOCAL_TRANSITION_27D,
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
from marine_race_arena.learning.gen2.recurrent_policy import Gen2RecurrentController
from marine_race_arena.learning.gen2.transfer_27d import (
    sha256_file,
    transfer_parent_35d_to_27d,
)
from marine_race_arena.learning.observation_encoder_local_transition_27d import (
    encode_observation_local_transition_27d,
)
from marine_race_arena.learning.tracker_context_local_transition_27d import (
    OnboardLocalTransition27dContextTracker,
)


PARENT_CHECKPOINT_DEFAULT = "results/rl/gen2/best/best_completion_policy.zip"


def audit_circuit_27d(
    track: str,
    *,
    controller,
    output_dir: Path,
    adapter: str = "fallback",
    allow_fallback: bool = True,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
    seed: int = 8300,
    steps: int = 150,
) -> Dict[str, Any]:
    """Audit one circuit with 27-D contract without learning."""
    track_path = tf.track_path(track)
    assert_approved_water_fog(track_path)

    if not allow_fallback and adapter != "holoocean":
        raise ValueError(
            f"Production 27-D audit strictly requires adapter='holoocean' and allow_fallback=False. Got {adapter!r}."
        )

    episode = RaceEpisode(
        str(track_path),
        seed=int(seed),
        adapter=adapter,
        allow_fallback=allow_fallback,
        max_steps=int(steps),
        official=True,
        current_profile="none",
        benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
    )

    out = output_dir / track
    out.mkdir(parents=True, exist_ok=True)

    try:
        raw = episode.reset()
        actual_adapter = episode.actual_adapter
        fallback_used = episode.fallback_used
        if not allow_fallback and fallback_used:
            raise RuntimeError(
                f"Production 27-D audit strictly forbids fallback adapter! Got {actual_adapter} on {track}."
            )

        ctx_cfg = episode.context.config
        tracker = OnboardLocalTransition27dContextTracker(
            total_beacons=len(ctx_cfg.track.gate_sequence),
            laps=int(ctx_cfg.race.laps),
        )
        tracker.reset(raw)

        prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        valid_steps = 0
        orient_present_steps = 0
        vision_present_steps = 0
        all_finite = True
        all_within_bounds = True

        lows = np.array([b[0] for b in FEATURE_BOUNDS_LOCAL_TRANSITION_27D], dtype=np.float32)
        highs = np.array([b[1] for b in FEATURE_BOUNDS_LOCAL_TRANSITION_27D], dtype=np.float32)

        for s in range(steps):
            context = tracker.context(
                raw, dt=episode.dt, prev_action=prev_action.tolist()
            )
            encoded = encode_observation_local_transition_27d(raw, context)

            if encoded.shape != (27,):
                raise ValueError(f"step {s}: invalid observation shape {encoded.shape}")
            if not np.isfinite(encoded).all():
                all_finite = False
            if np.any(encoded < lows - 1e-4) or np.any(encoded > highs + 1e-4):
                all_within_bounds = False

            valid_steps += 1
            if context.gate_orientation_present:
                orient_present_steps += 1
            if context.visual_target is not None:
                vision_present_steps += 1

            # Act with controller
            action_vec = controller.act(encoded, first_step=(s == 0))
            prev_action = action_vec
            command = {axis: float(action_vec[i]) for i, axis in enumerate(ACTION_AXES)}

            step_res = episode.step(command)
            raw = step_res.observation

            if step_res.terminated or step_res.truncated:
                break

        summary = {
            "track": track,
            "adapter": adapter,
            "requested_adapter": adapter,
            "actual_adapter": actual_adapter,
            "fallback_used": fallback_used,
            "steps": valid_steps,
            "all_finite": all_finite,
            "all_within_bounds": all_within_bounds,
            "vision_fraction": round(vision_present_steps / max(1, valid_steps), 4),
            "orientation_fraction": round(orient_present_steps / max(1, valid_steps), 4),
            "pass": all_finite and all_within_bounds and (valid_steps > 0),
            "pass": all_finite and all_within_bounds and (valid_steps > 0) and (not fallback_used if not allow_fallback else True),
        }
        (out / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        episode.close()


def run_preflight_27d(
    *,
    parent_path: str | Path = PARENT_CHECKPOINT_DEFAULT,
    output_dir: str | Path,
    adapter: str = "fallback",
    allow_fallback: bool = True,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
    steps: int = 100,
    seed: int = 8300,
) -> Dict[str, Any]:
    """Execute complete no-learning preflight over the 3 official tracks."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    parent_path = Path(parent_path)
    hash_before = sha256_file(parent_path)

    # Verify fog single source of truth across official circuits
    paths = [tf.track_path(t) for t in tf.OFFICIAL_TRACKS]
    fog_report = verify_fog_sources(paths)

    # Initialize 27-D model via warm start
    child_model = transfer_parent_35d_to_27d(parent_path, seed=seed)
    controller = Gen2RecurrentController(child_model, deterministic=True)

    summaries = []
    for offset, track in enumerate(tf.OFFICIAL_TRACKS):
        res = audit_circuit_27d(
            track,
            controller=controller,
            output_dir=out,
            adapter=adapter,
            allow_fallback=allow_fallback,
            seed=seed + offset,
            steps=steps,
        )
        summaries.append(res)

    hash_after = sha256_file(parent_path)
    if hash_after != hash_before:
        raise RuntimeError("Parent checkpoint modified during preflight!")

    all_passed = all(s["pass"] for s in summaries)
    actual_adapter_overall = "holoocean" if all(s.get("actual_adapter") == "holoocean" for s in summaries) else "fallback"
    fallback_used_overall = any(s.get("fallback_used", False) for s in summaries)
    report = {
        "schema_version": "gen2_preflight_27d_v1",
        "actual_adapter": actual_adapter_overall,
        "fallback_used": fallback_used_overall,
        "parent_checkpoint": str(parent_path),
        "parent_sha256": hash_before,
        "parent_immutable": True,
        "contract": OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
        "learning_updates": 0,
        "fog": fog_report,
        "tracks": summaries,
        "pass": all_passed,
        "pass": all_passed and (not fallback_used_overall if not allow_fallback else True),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    report_file = out / "preflight_report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_file)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="No-learning 27-D preflight audit")
    parser.add_argument("--parent", default=PARENT_CHECKPOINT_DEFAULT)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=8300)
    parser.add_argument("--adapter", default="fallback")
    parser.add_argument("--allow-fallback", action="store_true", default=True)
    parser.add_argument("--adapter", default="holoocean", choices=["holoocean", "fallback"])
    parser.add_argument("--allow-fallback", action="store_true", default=False)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_preflight_27d(
        parent_path=args.parent,
        output_dir=args.out,
        adapter=args.adapter,
        allow_fallback=args.allow_fallback,
        steps=args.steps,
        seed=args.seed,
    )
    print(json.dumps({"pass": report["pass"], "report": report["report_path"]}, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
