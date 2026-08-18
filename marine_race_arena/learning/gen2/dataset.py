"""Versioned, reproducible storage for the Gen-2 expert / DAgger corpus.

One episode is one ``.npz`` shard, so collection is resumable and a single
corrupt episode can never invalidate the corpus.  A ``manifest.json`` records
the SHA-256 of every shard plus the corpus composition, which is what makes the
dataset citable: the manifest hash pins the exact bytes every downstream model
was trained on.

The stored arrays are only what the learner is allowed to see:

``observations``      (T, 35) float32 -- ``onboard_local_transition_v1``
``expert_actions``    (T, 4)  float32 -- the supervision target
``applied_actions``   (T, 4)  float32 -- what actually drove the vehicle
``applied_by_expert`` (T,)    bool    -- audit trail for DAgger takeovers
``gate_crossings``    (T,)    int16   -- diagnostics; NEVER a network input

Course geometry lives in the per-shard metadata for diagnostics only.
:func:`load_corpus` does not return it, so it cannot leak into training.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.config import ACTION_DIM
from marine_race_arena.learning.config_local_transition import (
    FEATURE_NAMES_LOCAL_TRANSITION,
    OBS_DIM_LOCAL_TRANSITION,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION,
)
from marine_race_arena.learning.gen2 import GEN2_ACTION_CONTRACT, GEN2_EXPERT_ID

GEN2_DATASET_SCHEMA = "gen2_expert_corpus_v1"

#: Arrays that must never appear in a shard: their presence is a leak.
FORBIDDEN_ARRAY_NAMES = frozenset({
    "position", "positions", "pose", "poses", "rotation", "yaw_deg",
    "gate_positions", "gate_geometry", "circuit_id", "track_id",
    "expected_gate_id", "referee_state",
})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Gen2Episode:
    """One episode in memory: observations, expert labels, and diagnostics."""

    seed: int
    observations: np.ndarray
    expert_actions: np.ndarray
    applied_actions: np.ndarray
    applied_by_expert: np.ndarray
    gate_crossings: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    @property
    def gate_count(self) -> int:
        return int(self.meta.get("gate_count", 0))

    @property
    def completed(self) -> bool:
        return bool(self.meta.get("completed", False))

    def validate(self) -> None:
        if self.observations.ndim != 2 or self.observations.shape[1] != OBS_DIM_LOCAL_TRANSITION:
            raise ValueError(
                f"episode {self.seed}: observations must be (T, {OBS_DIM_LOCAL_TRANSITION}), "
                f"got {self.observations.shape}"
            )
        if self.expert_actions.shape != (len(self), ACTION_DIM):
            raise ValueError(f"episode {self.seed}: expert actions have shape {self.expert_actions.shape}")
        if not np.isfinite(self.observations).all():
            raise ValueError(f"episode {self.seed}: observations contain non-finite values")
        if not np.isfinite(self.expert_actions).all():
            raise ValueError(f"episode {self.seed}: expert actions contain non-finite values")
        if np.abs(self.expert_actions).max(initial=0.0) > 1.0 + 1e-6:
            raise ValueError(f"episode {self.seed}: expert actions escape the [-1, 1] contract")


def _episode_from_record(record) -> Gen2Episode:
    return Gen2Episode(
        seed=int(record.seed),
        observations=np.asarray(record.observations, dtype=np.float32),
        expert_actions=np.asarray(record.expert_actions, dtype=np.float32),
        applied_actions=np.asarray(record.applied_actions, dtype=np.float32),
        applied_by_expert=np.asarray(record.applied_by_expert, dtype=bool),
        gate_crossings=np.asarray(record.gate_crossings, dtype=np.int16),
        meta=record.diagnostics(),
    )


def shard_path(root: str | Path, seed: int, *, tag: str = "ep") -> Path:
    return Path(root) / "episodes" / f"{tag}_{int(seed):06d}.npz"


def save_episode(record, root: str | Path, *, tag: str = "ep") -> Path:
    """Write one episode shard atomically and return its path."""
    episode = record if isinstance(record, Gen2Episode) else _episode_from_record(record)
    episode.validate()
    target = shard_path(root, episode.seed, tag=tag)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.stem + ".partial.npz")
    np.savez_compressed(
        tmp,
        observations=episode.observations,
        expert_actions=episode.expert_actions,
        applied_actions=episode.applied_actions,
        applied_by_expert=episode.applied_by_expert,
        gate_crossings=episode.gate_crossings,
        meta=np.asarray([json.dumps(episode.meta, sort_keys=True)]),
    )
    tmp.replace(target)
    return target


def load_episode(path: str | Path) -> Gen2Episode:
    """Read one shard and re-validate it against the contract."""
    with np.load(path, allow_pickle=False) as payload:
        names = set(payload.files)
        leaked = names & FORBIDDEN_ARRAY_NAMES
        if leaked:
            raise ValueError(f"{path}: shard carries privileged arrays {sorted(leaked)}")
        meta = json.loads(str(payload["meta"][0]))
        episode = Gen2Episode(
            seed=int(meta.get("seed", -1)),
            observations=np.asarray(payload["observations"], dtype=np.float32),
            expert_actions=np.asarray(payload["expert_actions"], dtype=np.float32),
            applied_actions=np.asarray(payload["applied_actions"], dtype=np.float32),
            applied_by_expert=np.asarray(payload["applied_by_expert"], dtype=bool),
            gate_crossings=np.asarray(payload["gate_crossings"], dtype=np.int16),
            meta=meta,
        )
    episode.validate()
    return episode


def iter_shards(root: str | Path) -> Iterator[Path]:
    directory = Path(root) / "episodes"
    if not directory.is_dir():
        return iter(())
    return iter(sorted(directory.glob("*.npz")))


def load_corpus(
    roots: str | Path | Sequence[str | Path],
    *,
    completed_only: bool = False,
    exclude_expert_takeover: bool = False,
    min_steps: int = 2,
) -> List[Gen2Episode]:
    """Load every shard under ``roots``.

    ``exclude_expert_takeover`` drops any episode where a safety takeover put
    the expert on the stick, so a claim about autonomous learner competence is
    never contaminated by an intervened episode.
    """
    if isinstance(roots, (str, Path)):
        roots = [roots]
    episodes: List[Gen2Episode] = []
    for root in roots:
        for path in iter_shards(root):
            episode = load_episode(path)
            if len(episode) < int(min_steps):
                continue
            if completed_only and not episode.completed:
                continue
            if exclude_expert_takeover and bool(episode.applied_by_expert.any()):
                continue
            episodes.append(episode)
    return episodes


def corpus_statistics(episodes: Sequence[Gen2Episode]) -> Dict[str, Any]:
    """Composition and quality numbers reported for the dataset."""
    if not episodes:
        return {"episodes": 0, "transitions": 0}
    by_gate_count: Dict[str, Dict[str, Any]] = {}
    for episode in episodes:
        key = str(episode.gate_count)
        bucket = by_gate_count.setdefault(
            key, {"episodes": 0, "transitions": 0, "completed": 0, "gates_crossed": 0, "gates_possible": 0}
        )
        bucket["episodes"] += 1
        bucket["transitions"] += len(episode)
        bucket["completed"] += int(episode.completed)
        bucket["gates_crossed"] += int(episode.meta.get("gates_completed", 0))
        bucket["gates_possible"] += episode.gate_count
    for bucket in by_gate_count.values():
        bucket["completion_rate"] = round(bucket["completed"] / max(1, bucket["episodes"]), 4)
        bucket["gate_success_rate"] = round(
            bucket["gates_crossed"] / max(1, bucket["gates_possible"]), 4
        )

    lengths = np.asarray([len(e) for e in episodes])
    patterns: Dict[str, int] = {}
    for episode in episodes:
        pattern = str(episode.meta.get("course_pattern", "unknown"))
        patterns[pattern] = patterns.get(pattern, 0) + 1

    total_transitions = int(lengths.sum())
    return {
        "episodes": len(episodes),
        "transitions": total_transitions,
        "unique_courses": len({e.seed for e in episodes}),
        "completed_episodes": int(sum(e.completed for e in episodes)),
        "completion_rate": round(sum(e.completed for e in episodes) / len(episodes), 4),
        "gate_success_rate": round(
            sum(int(e.meta.get("gates_completed", 0)) for e in episodes)
            / max(1, sum(e.gate_count for e in episodes)),
            4,
        ),
        "collision_episodes": int(sum(int(e.meta.get("collisions", 0)) > 0 for e in episodes)),
        "out_of_bounds_episodes": int(sum(int(e.meta.get("out_of_bounds", 0)) > 0 for e in episodes)),
        "wrong_direction_episodes": int(sum(int(e.meta.get("wrong_direction", 0)) > 0 for e in episodes)),
        "expert_takeover_episodes": int(sum(bool(e.applied_by_expert.any()) for e in episodes)),
        "episode_length_steps": {
            "min": int(lengths.min()), "max": int(lengths.max()),
            "mean": round(float(lengths.mean()), 1),
            "median": int(np.median(lengths)),
        },
        "by_gate_count": dict(sorted(by_gate_count.items(), key=lambda kv: int(kv[0]))),
        "patterns": dict(sorted(patterns.items())),
    }


def observation_statistics(episodes: Sequence[Gen2Episode]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and standard deviation over the whole corpus.

    Computed in one streaming pass so a large corpus never has to be
    concatenated into a single array.
    """
    count = 0
    total = np.zeros(OBS_DIM_LOCAL_TRANSITION, dtype=np.float64)
    total_sq = np.zeros(OBS_DIM_LOCAL_TRANSITION, dtype=np.float64)
    for episode in episodes:
        values = episode.observations.astype(np.float64)
        count += values.shape[0]
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
    if count == 0:
        return (
            np.zeros(OBS_DIM_LOCAL_TRANSITION, np.float32),
            np.ones(OBS_DIM_LOCAL_TRANSITION, np.float32),
        )
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    std = np.sqrt(variance)
    # A constant feature (e.g. a permanently-set presence flag) would divide by
    # zero; floor it so normalization leaves it untouched instead of exploding.
    std = np.where(std < 1e-3, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def write_manifest(
    root: str | Path,
    *,
    episodes: Optional[Sequence[Gen2Episode]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Hash every shard and record the corpus composition."""
    root = Path(root)
    shards = []
    for path in iter_shards(root):
        shards.append({
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if episodes is None:
        episodes = load_corpus(root)
    combined = hashlib.sha256()
    for shard in shards:
        combined.update(shard["sha256"].encode("ascii"))
    payload: Dict[str, Any] = {
        "schema_version": GEN2_DATASET_SCHEMA,
        "observation_contract": OBS_ENCODING_VERSION_LOCAL_TRANSITION,
        "observation_dim": OBS_DIM_LOCAL_TRANSITION,
        "feature_names": list(FEATURE_NAMES_LOCAL_TRANSITION),
        "action_contract": GEN2_ACTION_CONTRACT,
        "action_dim": ACTION_DIM,
        "expert": GEN2_EXPERT_ID,
        "shard_count": len(shards),
        "corpus_sha256": combined.hexdigest(),
        "statistics": corpus_statistics(episodes),
        "shards": shards,
        **dict(extra or {}),
    }
    target = root / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.partial")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(target)
    return target


def verify_manifest(root: str | Path) -> Dict[str, Any]:
    """Re-hash every shard and report any drift from the recorded manifest."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    recorded = {entry["file"]: entry["sha256"] for entry in manifest.get("shards", [])}
    present = {path.name: sha256_file(path) for path in iter_shards(root)}
    changed = sorted(k for k in recorded.keys() & present.keys() if recorded[k] != present[k])
    return {
        "manifest": str(root / "manifest.json"),
        "recorded_shards": len(recorded),
        "present_shards": len(present),
        "missing": sorted(recorded.keys() - present.keys()),
        "unrecorded": sorted(present.keys() - recorded.keys()),
        "changed": changed,
        "intact": not changed and not (recorded.keys() - present.keys()),
    }
