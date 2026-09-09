"""Versioned, onboard-only shards for the approved 27-D expert corpus."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.config import ACTION_DIM
from marine_race_arena.learning.config_local_transition_27d import (
    FEATURE_NAMES_LOCAL_TRANSITION_27D,
    OBS_DIM_LOCAL_TRANSITION_27D,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
)
from marine_race_arena.learning.gen2.dataset import FORBIDDEN_ARRAY_NAMES

GEN2_DATASET_27D_SCHEMA = "gen2_expert_corpus_27d_v1"


@dataclass
class Gen2Episode27d:
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
        if self.observations.ndim != 2 or self.observations.shape[1] != OBS_DIM_LOCAL_TRANSITION_27D:
            raise ValueError(f"episode {self.seed}: observations must be (T, 27), got {self.observations.shape}")
        if self.expert_actions.shape != (len(self), ACTION_DIM):
            raise ValueError(f"episode {self.seed}: expert actions have shape {self.expert_actions.shape}")
        if self.applied_actions.shape != (len(self), ACTION_DIM):
            raise ValueError(f"episode {self.seed}: applied actions have shape {self.applied_actions.shape}")
        if self.applied_by_expert.shape != (len(self),):
            raise ValueError(f"episode {self.seed}: applied_by_expert has shape {self.applied_by_expert.shape}")
        if not np.isfinite(self.observations).all() or not np.isfinite(self.expert_actions).all():
            raise ValueError(f"episode {self.seed}: non-finite learner-visible arrays")
        if np.abs(self.expert_actions).max(initial=0.0) > 1.0 + 1e-6:
            raise ValueError(f"episode {self.seed}: expert actions escape [-1, 1]")


def shard_path(root: str | Path, seed: int, *, tag: str = "ep") -> Path:
    return Path(root) / "episodes" / f"{tag}_{int(seed):06d}.npz"


def save_episode(record: Gen2Episode27d, root: str | Path, *, tag: str = "ep") -> Path:
    record.validate()
    target = shard_path(root, record.seed, tag=tag)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.stem}.partial.{os.getpid()}.npz")
    np.savez_compressed(
        tmp,
        observations=record.observations.astype(np.float32),
        expert_actions=record.expert_actions.astype(np.float32),
        applied_actions=record.applied_actions.astype(np.float32),
        applied_by_expert=record.applied_by_expert.astype(bool),
        gate_crossings=record.gate_crossings.astype(np.int16),
        meta=np.asarray([json.dumps(record.meta, sort_keys=True)]),
    )
    tmp.replace(target)
    return target


def iter_shards(root: str | Path) -> Iterator[Path]:
    directory = Path(root) / "episodes"
    if not directory.is_dir():
        return iter(())
    return iter(sorted(p for p in directory.glob("*.npz") if ".partial." not in p.name))


def load_episode(path: str | Path) -> Gen2Episode27d:
    with np.load(path, allow_pickle=False) as payload:
        leaked = set(payload.files) & FORBIDDEN_ARRAY_NAMES
        if leaked:
            raise ValueError(f"{path}: privileged arrays present: {sorted(leaked)}")
        meta = json.loads(str(payload["meta"][0]))
        episode = Gen2Episode27d(
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


def load_corpus(root: str | Path, *, completed_only: bool = False) -> list[Gen2Episode27d]:
    episodes = [load_episode(path) for path in iter_shards(root)]
    return [episode for episode in episodes if episode.completed] if completed_only else episodes


def observation_statistics(episodes: Sequence[Gen2Episode27d]) -> Tuple[np.ndarray, np.ndarray]:
    if not episodes:
        return np.zeros(OBS_DIM_LOCAL_TRANSITION_27D, np.float32), np.ones(OBS_DIM_LOCAL_TRANSITION_27D, np.float32)
    values = np.concatenate([episode.observations for episode in episodes], axis=0).astype(np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-3] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def corpus_statistics(episodes: Sequence[Gen2Episode27d]) -> Dict[str, Any]:
    if not episodes:
        return {"episodes": 0, "transitions": 0}
    lengths = np.asarray([len(episode) for episode in episodes], dtype=np.int64)
    total = max(1, int(lengths.sum()))
    def _episode_source_type(episode: Gen2Episode27d) -> str:
        explicit = episode.meta.get("source_type")
        if explicit:
            return str(explicit)
        # Pilot shards predate the explicit source_type field.
        kind = str(episode.meta.get("kind", "unknown"))
        return "exact_fragment" if kind == "fragment" else kind

    return {
        "episodes": len(episodes),
        "transitions": int(lengths.sum()),
        "completed_episodes": int(sum(episode.completed for episode in episodes)),
        "completion_rate": round(sum(episode.completed for episode in episodes) / len(episodes), 4),
        "gate_success_rate": round(sum(int(e.meta.get("gates_completed", 0)) for e in episodes) / max(1, sum(e.gate_count for e in episodes)), 4),
        "vision_availability": round(sum(int(e.meta.get("vision_steps", 0)) for e in episodes) / total, 4),
        "orientation_availability": round(sum(int(e.meta.get("orientation_steps", 0)) for e in episodes) / total, 4),
        "dvl_availability": round(sum(int(e.meta.get("dvl_steps", 0)) for e in episodes) / total, 4),
        "episode_length_steps": {"min": int(lengths.min()), "max": int(lengths.max()), "mean": round(float(lengths.mean()), 1), "median": int(np.median(lengths))},
        "by_source": {str(source): sum(1 for e in episodes if e.meta.get("source") == source) for source in sorted({str(e.meta.get("source")) for e in episodes})},
        "by_source_type": {
            stype: sum(
                1 for e in episodes
                if _episode_source_type(e) == stype
            )
            for stype in sorted({
                _episode_source_type(e)
                for e in episodes
            })
        },
        "transitions_by_source_type": {
            stype: int(sum(
                len(e) for e in episodes
                if _episode_source_type(e) == stype
            ))
            for stype in sorted({
                _episode_source_type(e)
                for e in episodes
            })
        },
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root: str | Path, *, episodes: Optional[Sequence[Gen2Episode27d]] = None, extra: Optional[Mapping[str, Any]] = None) -> Path:
    root = Path(root)
    shards = [{"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in iter_shards(root)]
    if episodes is None:
        episodes = load_corpus(root)
    digest = hashlib.sha256()
    for shard in shards:
        digest.update(shard["sha256"].encode("ascii"))
    manifest = {
        "schema_version": GEN2_DATASET_27D_SCHEMA,
        "observation_contract": OBS_ENCODING_VERSION_LOCAL_TRANSITION_27D,
        "observation_dim": OBS_DIM_LOCAL_TRANSITION_27D,
        "feature_names": list(FEATURE_NAMES_LOCAL_TRANSITION_27D),
        "action_dim": ACTION_DIM,
        "expert": "rule_gate_center_then_commit",
        "shard_count": len(shards),
        "corpus_sha256": digest.hexdigest(),
        "statistics": corpus_statistics(episodes),
        "shards": shards,
        **dict(extra or {}),
    }
    target = root / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target
