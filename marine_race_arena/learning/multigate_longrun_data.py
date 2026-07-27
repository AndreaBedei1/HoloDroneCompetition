"""Balanced expert and policy-state DAgger collection for multi-gate BC-v3."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from marine_race_arena.learning.config import (
    ACTION_AXES,
    ACTION_DIM,
    ACTION_CONTRACT_VERSION,
)
from marine_race_arena.learning.config_v3 import (
    OBS_DIM_V3,
    OBS_ENCODING_VERSION_V3,
)
from marine_race_arena.learning.dataset import BCDataset
from marine_race_arena.learning.episode import RaceEpisode
from marine_race_arena.learning.longrun_checkpoint import atomic_write_json
from marine_race_arena.learning.model_contract_v3 import assert_clean_worktree
from marine_race_arena.learning.observation_encoder_v3 import encode_observation_v3
from marine_race_arena.learning.parametric_curriculum import (
    ANGLE_BANDS_DEG,
    TransitionGeometry,
    generate_two_gate_track,
)
from marine_race_arena.learning.provenance import git_sha, now_utc, sha256_file
from marine_race_arena.learning.seed_registry import (
    MULTIGATE_LONGRUN_DEMONSTRATION_SEEDS,
)
from marine_race_arena.learning.tracker_context_v3 import (
    OnboardMultiGateContextTracker,
)
from marine_race_arena.learning.trajectory_recorder import (
    EpisodeRecord,
    record_episode,
)
from marine_race_arena.participants.controller_loader import ControllerLoader
from marine_race_arena.scripts.run_marine_race import _mission_info


@dataclass(frozen=True)
class CollectionPlanRow:
    episode_id: int
    seed: int
    split: str
    geometry: TransitionGeometry


def _stable_split(geometry_group: str) -> str:
    bucket = int(hashlib.sha256(geometry_group.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket in (1, 2):
        return "validation"
    return "train"


def build_balanced_plan(
    *,
    straight: int = 30,
    left: int = 50,
    right: int = 50,
    first_seed: int = 26000,
    seed: int = 26000,
) -> List[CollectionPlanRow]:
    """Cover the requested angle bands with balanced geometry-level splits."""
    rng = np.random.default_rng(seed)
    directions = ["straight"] * straight + ["left"] * left + ["right"] * right
    turn_bands = (10.0, 15.0, 20.0, 30.0, 40.0, 45.0)
    rows: List[CollectionPlanRow] = []
    direction_counts = {"straight": 0, "left": 0, "right": 0}
    for episode_id, direction in enumerate(directions):
        local_index = direction_counts[direction]
        direction_counts[direction] += 1
        if direction == "straight":
            angle = (0.0, 5.0, -5.0)[local_index % 3]
        else:
            band = turn_bands[local_index % len(turn_bands)]
            angle = band if direction == "left" else -band
        geometry = TransitionGeometry(
            signed_turn_deg=float(angle),
            gate_separation_m=float((3.5, 4.5, 5.5, 6.5)[local_index % 4]),
            lateral_displacement_m=float((-0.8, -0.3, 0.3, 0.8)[(local_index // 2) % 4]),
            vertical_displacement_m=float((-0.6, 0.0, 0.6)[(local_index // 3) % 3]),
            starting_yaw_error_deg=float((-15.0, -8.0, 0.0, 8.0, 15.0)[local_index % 5]),
            initial_lateral_offset_m=float((-1.0, -0.5, 0.0, 0.5, 1.0)[(local_index // 2) % 5]),
            beacon_bearing_noise_std_deg=float(rng.uniform(0.0, 1.0)),
            beacon_range_noise_std_m=float(rng.uniform(0.0, 0.08)),
            beacon_dropout_probability=float(rng.uniform(0.0, 0.02)),
            visibility_scale=float(rng.uniform(0.85, 1.0)),
            source="balanced_expert",
            stage="C4" if abs(angle) > 30 else "C3",
        )
        rows.append(
            CollectionPlanRow(
                episode_id=episode_id,
                seed=first_seed + episode_id,
                split=_stable_split(geometry.geometry_group),
                geometry=geometry,
            )
        )
    return rows


def write_collection_plan(plan: Sequence[CollectionPlanRow], path: str | Path) -> None:
    counts: Dict[str, int] = {}
    split_counts: Dict[str, int] = {}
    for row in plan:
        counts[row.geometry.direction] = counts.get(row.geometry.direction, 0) + 1
        split_counts[row.split] = split_counts.get(row.split, 0) + 1
    atomic_write_json(
        path,
        {
            "schema_version": "multigate_balanced_collection_plan_v1",
            "created_utc": now_utc(),
            "counts": counts,
            "split_counts": split_counts,
            "angle_bands_deg": list(ANGLE_BANDS_DEG),
            "rows": [
                {
                    "episode_id": row.episode_id,
                    "seed": row.seed,
                    "split": row.split,
                    "geometry_group": row.geometry.geometry_group,
                    "geometry": asdict(row.geometry),
                }
                for row in plan
            ],
        },
    )


def _command_vector(command: Any) -> np.ndarray:
    if not isinstance(command, dict):
        return np.zeros(ACTION_DIM, dtype=np.float32)
    return np.asarray(
        [float(command.get(axis, 0.0)) for axis in ACTION_AXES],
        dtype=np.float32,
    )


def record_dagger_correction_episode(
    track: str,
    model_path: str,
    *,
    seed: int,
    episode_id: int,
    adapter: str = "holoocean",
    max_steps: int = 1200,
) -> EpisodeRecord:
    """Apply only learned actions; retain shadow-expert labels on visited states."""
    episode = RaceEpisode(
        track,
        seed=seed,
        adapter=adapter,
        allow_fallback=False,
        max_steps=max_steps,
        official=True,
        current_profile="none",
    )
    obs = episode.reset()
    cfg = episode.context.config
    mission = _mission_info(cfg, episode.participant_id)
    learned = ControllerLoader().load(
        "rl_multigate_controller", constructor_kwargs={"model_path": model_path}
    )
    expert = ControllerLoader().load("rule_gate_center_then_commit")
    learned.reset(mission)
    expert.reset(mission)
    context = OnboardMultiGateContextTracker(
        total_beacons=max(1, len(cfg.track.gate_sequence)),
        laps=max(1, int(cfg.race.laps)),
    )
    context.reset(obs)
    observations: List[np.ndarray] = []
    expert_labels: List[np.ndarray] = []
    learned_actions: List[np.ndarray] = []
    done_flags: List[bool] = []
    trunc_flags: List[bool] = []
    positions: List[List[float]] = []
    crossing_counts: List[int] = []
    previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
    try:
        while True:
            onboard = context.context(
                obs, dt=episode.dt, prev_action=previous_action.tolist()
            )
            observations.append(encode_observation_v3(obs, onboard))
            learned_command = learned.step(copy.deepcopy(obs))
            expert_command = expert.step(copy.deepcopy(obs))
            learned_action = np.clip(_command_vector(learned_command), -1.0, 1.0)
            expert_action = np.clip(_command_vector(expert_command), -1.0, 1.0)
            step = episode.step(learned_command)
            learned_actions.append(learned_action)
            expert_labels.append(expert_action)
            done_flags.append(bool(step.terminated))
            trunc_flags.append(bool(step.truncated))
            positions.append([float(v) for v in step.current_state.position])
            crossing_counts.append(
                int(episode.referee_progress()["valid_gate_crossings"])
            )
            previous_action = learned_action
            obs = step.observation
            if step.terminated or step.truncated:
                break
        progress = episode.referee_progress()
        adapter_actual = episode.context.adapter.name
    finally:
        learned.close()
        expert.close()
        episode.close()

    crossings = np.asarray(crossing_counts, dtype=np.int64)
    crossed_indices = np.flatnonzero(crossings >= 1)
    if len(crossed_indices):
        start = max(0, int(crossed_indices[0]) - 40)
    else:
        start = max(0, len(observations) - 160)
    selected = np.arange(start, len(observations), dtype=np.int64)
    dones = np.asarray(done_flags, dtype=bool)[selected]
    truncs = np.asarray(trunc_flags, dtype=bool)[selected]
    if len(selected) and not (dones[-1] or truncs[-1]):
        truncs[-1] = True
    return EpisodeRecord(
        episode_id=episode_id,
        seed=seed,
        track=track,
        controller="dagger_shadow_expert_labels_on_learned_rollout",
        observations=np.asarray(observations, dtype=np.float32)[selected].reshape(
            -1, OBS_DIM_V3
        ),
        expert_actions_raw=np.asarray(expert_labels, dtype=np.float32)[selected],
        actions=np.asarray(expert_labels, dtype=np.float32)[selected],
        dones=dones,
        truncated=truncs,
        step_ids=np.arange(len(selected), dtype=np.int64),
        phase_ids=np.full(len(selected), -1, dtype=np.int64),
        final_status=str(progress["status"]),
        gate_crossings=int(progress["valid_gate_crossings"]),
        diagnostics={
            "positions": np.asarray(positions, dtype=np.float32)[selected],
            "gate_crossings": crossings[selected],
            "learned_actions": np.asarray(learned_actions, dtype=np.float32)[
                selected
            ],
        },
        metadata={
            "track_sha256": sha256_file(track),
            "adapter_requested": adapter,
            "adapter_actual": adapter_actual,
            "fallback_used": False,
            "obs_encoding_version": OBS_ENCODING_VERSION_V3,
            "action_contract_version": ACTION_CONTRACT_VERSION,
            "collection_git_sha": git_sha(),
            "runtime_actions_from_learned_policy_only": True,
            "expert_labels_applied_to_environment": False,
            "selection_start_step": int(start),
        },
    )


def _rebuild(records: Sequence[EpisodeRecord], path: Path) -> BCDataset:
    for episode_id, record in enumerate(records):
        record.episode_id = episode_id
    dataset = BCDataset.from_records(records)
    dataset.check_integrity()
    tmp = path.with_name("_tmp_" + path.name)
    dataset.save(tmp)
    tmp.replace(path)
    return dataset


def collect_balanced(
    output_dir: str | Path,
    *,
    plan: Sequence[CollectionPlanRow],
    adapter: str = "holoocean",
    max_steps: int = 1200,
    dagger_model: Optional[str] = None,
) -> Dict[str, Any]:
    out = Path(output_dir)
    episodes_dir = out / "episodes"
    tracks_dir = out / "tracks"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)
    existing: Dict[int, EpisodeRecord] = {}
    for path in episodes_dir.glob("ep_*.npz"):
        try:
            record = EpisodeRecord.load_npz(path)
            existing[record.seed] = record
        except Exception:
            continue
    for row in plan:
        if row.seed in existing:
            continue
        track_path = tracks_dir / f"transition_{row.episode_id:04d}.json"
        generate_two_gate_track(row.geometry, track_path)
        if dagger_model:
            record = record_dagger_correction_episode(
                str(track_path),
                dagger_model,
                seed=row.seed,
                episode_id=row.episode_id,
                adapter=adapter,
                max_steps=max_steps,
            )
        else:
            record = record_episode(
                str(track_path),
                "rule_gate_center_then_commit",
                seed=row.seed,
                adapter=adapter,
                allow_fallback=False,
                max_steps=max_steps,
                official=True,
                episode_id=row.episode_id,
                current_profile="none",
                observation_encoding_version=OBS_ENCODING_VERSION_V3,
            )
        record.metadata["geometry"] = asdict(row.geometry)
        record.metadata["geometry_group"] = row.geometry.geometry_group
        record.metadata["dataset_split"] = row.split
        record.save_npz(episodes_dir / f"ep_{row.seed:05d}.npz")
        existing[row.seed] = record
        _rebuild(list(existing.values()), out / "dataset.npz")

    records = [existing[seed] for seed in sorted(existing)]
    dataset = _rebuild(records, out / "dataset.npz")
    plan_by_seed = {row.seed: row for row in plan}
    directions: Dict[str, Dict[str, int]] = {}
    split_counts: Dict[str, int] = {}
    correction_samples = 0
    episode_rows = []
    for record in records:
        plan_row = plan_by_seed[record.seed]
        direction = plan_row.geometry.direction
        counters = directions.setdefault(
            direction, {"episodes": 0, "successful": 0, "steps": 0}
        )
        counters["episodes"] += 1
        counters["successful"] += int(record.final_status == "FINISHED")
        counters["steps"] += record.length
        split_counts[plan_row.split] = split_counts.get(plan_row.split, 0) + 1
        if dagger_model:
            correction_samples += record.length
        episode_path = episodes_dir / f"ep_{record.seed:05d}.npz"
        episode_rows.append(
            {
                "seed": record.seed,
                "split": plan_row.split,
                "geometry_group": plan_row.geometry.geometry_group,
                "direction": direction,
                "angle_deg": plan_row.geometry.signed_turn_deg,
                "status": record.final_status,
                "gates": record.gate_crossings,
                "steps": record.length,
                "episode_sha256": sha256_file(episode_path),
            }
        )
    manifest = {
        "schema_version": "multigate_dataset_manifest_v1",
        "created_utc": now_utc(),
        "code_sha": git_sha(),
        "adapter": adapter,
        "fallback_allowed": False,
        "controller": (
            "learned rollout + shadow expert labels"
            if dagger_model
            else "training-only deterministic expert"
        ),
        "dagger_model": dagger_model,
        "runtime_expert_allowed": False,
        "observation_version": OBS_ENCODING_VERSION_V3,
        "action_version": ACTION_CONTRACT_VERSION,
        "directions": directions,
        "split_counts": split_counts,
        "geometry_groups_disjoint_by_split": True,
        "dagger_correction_samples": correction_samples,
        "dataset_path": str(out / "dataset.npz"),
        "dataset_sha256": sha256_file(out / "dataset.npz"),
        "episodes": dataset.num_episodes,
        "steps": len(dataset),
        "episode_rows": episode_rows,
    }
    atomic_write_json(out / "manifest.json", manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument(
        "--out",
        default="datasets/multigate_v3/r2_balanced_turns_v1/collection_plan.json",
    )
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument(
        "--out", default="datasets/multigate_v3/r2_balanced_turns_v1"
    )
    collect_parser.add_argument("--straight", type=int, default=30)
    collect_parser.add_argument("--left", type=int, default=50)
    collect_parser.add_argument("--right", type=int, default=50)
    collect_parser.add_argument("--first-seed", type=int, default=26000)
    collect_parser.add_argument("--adapter", default="holoocean")
    collect_parser.add_argument("--max-steps", type=int, default=1200)
    collect_parser.add_argument("--dagger-model", default=None)
    args = parser.parse_args(argv)
    plan = build_balanced_plan(
        straight=getattr(args, "straight", 30),
        left=getattr(args, "left", 50),
        right=getattr(args, "right", 50),
        first_seed=getattr(args, "first_seed", 26000),
    )
    if args.command == "plan":
        write_collection_plan(plan, args.out)
        result = {"plan": args.out, "episodes": len(plan)}
    else:
        assert_clean_worktree()
        if args.adapter != "holoocean":
            raise ValueError("published balanced data must use real HoloOcean")
        if not {row.seed for row in plan} <= set(
            MULTIGATE_LONGRUN_DEMONSTRATION_SEEDS
        ):
            raise ValueError("collection plan uses seeds outside the long-run demo range")
        result = collect_balanced(
            args.out,
            plan=plan,
            adapter=args.adapter,
            max_steps=args.max_steps,
            dagger_model=args.dagger_model,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
