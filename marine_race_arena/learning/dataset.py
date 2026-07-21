"""Behavioral-cloning dataset: assemble, validate, persist and split.

Consumes :class:`EpisodeRecord`s (from :mod:`trajectory_recorder`) into flat
``(observation, action)`` arrays for supervised behavioral cloning. Only the
policy-legal fields are kept; the recorder's privileged ``diagnostics`` are
dropped here and can never leak into training.

Integrity is checked explicitly (finite, correct dims, action bounds, unique
episode identity, per-episode completeness). The train/validation split is by
*episode* (a whole episode is entirely in one split), so no step leaks across
the split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.config import ACTION_DIM, OBS_DIM


class DatasetIntegrityError(ValueError):
    """Raised when a behavioral-cloning dataset fails an integrity check."""


@dataclass
class EpisodeMeta:
    group_id: int
    episode_id: int
    seed: int
    track: str
    controller: str
    length: int
    final_status: str
    gate_crossings: int


class BCDataset:
    """Flat (observation, action) dataset with per-episode grouping."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        group_ids: np.ndarray,
        seeds: np.ndarray,
        episode_ids: np.ndarray,
        step_ids: np.ndarray,
        dones: np.ndarray,
        truncated: np.ndarray,
        episodes: Sequence[EpisodeMeta],
    ) -> None:
        self.observations = np.asarray(observations, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.group_ids = np.asarray(group_ids, dtype=np.int64)
        self.seeds = np.asarray(seeds, dtype=np.int64)
        self.episode_ids = np.asarray(episode_ids, dtype=np.int64)
        self.step_ids = np.asarray(step_ids, dtype=np.int64)
        self.dones = np.asarray(dones, dtype=bool)
        self.truncated = np.asarray(truncated, dtype=bool)
        self.episodes = list(episodes)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_records(cls, records: Sequence["EpisodeRecordLike"]) -> "BCDataset":
        obs, act, gid, seeds, eids, sids, dones, truncs = [], [], [], [], [], [], [], []
        episodes: List[EpisodeMeta] = []
        for group_id, rec in enumerate(records):
            n = int(rec.observations.shape[0])
            if n == 0:
                continue
            obs.append(np.asarray(rec.observations, dtype=np.float32))
            act.append(np.asarray(rec.actions, dtype=np.float32))
            gid.append(np.full(n, group_id, dtype=np.int64))
            seeds.append(np.full(n, int(rec.seed), dtype=np.int64))
            eids.append(np.full(n, int(rec.episode_id), dtype=np.int64))
            sids.append(np.asarray(rec.step_ids, dtype=np.int64))
            dones.append(np.asarray(rec.dones, dtype=bool))
            truncs.append(np.asarray(rec.truncated, dtype=bool))
            episodes.append(
                EpisodeMeta(
                    group_id=group_id,
                    episode_id=int(rec.episode_id),
                    seed=int(rec.seed),
                    track=str(rec.track),
                    controller=str(rec.controller),
                    length=n,
                    final_status=str(rec.final_status),
                    gate_crossings=int(rec.gate_crossings),
                )
            )
        if not obs:
            raise DatasetIntegrityError("no non-empty episodes to build a dataset from")
        return cls(
            np.concatenate(obs),
            np.concatenate(act),
            np.concatenate(gid),
            np.concatenate(seeds),
            np.concatenate(eids),
            np.concatenate(sids),
            np.concatenate(dones),
            np.concatenate(truncs),
            episodes,
        )

    # ------------------------------------------------------------------ props
    def __len__(self) -> int:
        return int(self.observations.shape[0])

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    # ------------------------------------------------------------------ checks
    def check_integrity(self) -> None:
        n = len(self)
        if n == 0:
            raise DatasetIntegrityError("dataset is empty")
        if self.observations.shape[1] != OBS_DIM:
            raise DatasetIntegrityError(f"observation dim {self.observations.shape[1]} != {OBS_DIM}")
        if self.actions.shape[1] != ACTION_DIM:
            raise DatasetIntegrityError(f"action dim {self.actions.shape[1]} != {ACTION_DIM}")
        for name, arr in (("observations", self.observations), ("actions", self.actions)):
            if not np.all(np.isfinite(arr)):
                raise DatasetIntegrityError(f"{name} contains non-finite values")
        if np.any(self.actions < -1.0 - 1e-5) or np.any(self.actions > 1.0 + 1e-5):
            raise DatasetIntegrityError("actions outside [-1, 1]")
        lengths = {a.shape[0] for a in (self.actions, self.group_ids, self.seeds, self.episode_ids, self.step_ids, self.dones, self.truncated)}
        if lengths != {n}:
            raise DatasetIntegrityError("array length mismatch across dataset columns")
        # Unique episode identity and per-episode completeness.
        identities = set()
        for meta in self.episodes:
            key = (meta.seed, meta.episode_id, meta.track, meta.controller)
            if key in identities:
                raise DatasetIntegrityError(f"duplicate episode identity {key}")
            identities.add(key)
            mask = self.group_ids == meta.group_id
            if int(mask.sum()) != meta.length:
                raise DatasetIntegrityError(f"episode {meta.group_id} length mismatch")
            last = np.argmax(self.step_ids[mask])
            if not (self.dones[mask][last] or self.truncated[mask][last]):
                raise DatasetIntegrityError(f"episode {meta.group_id} does not end with done/truncated")
        # No-leakage: the only feature array is `observations` of size OBS_DIM.
        if hasattr(self, "positions") or hasattr(self, "diagnostics"):  # pragma: no cover
            raise DatasetIntegrityError("privileged diagnostics leaked into the dataset")

    # ------------------------------------------------------------------ split
    def train_val_split(self, val_fraction: float = 0.2, seed: int = 0) -> Tuple["BCDataset", "BCDataset"]:
        """Split by episode: a whole episode goes to exactly one side."""
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        group_ids = np.array([m.group_id for m in self.episodes], dtype=np.int64)
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(group_ids))
        n_val = max(1, int(round(len(group_ids) * val_fraction)))
        val_groups = set(int(group_ids[i]) for i in order[:n_val])
        train_groups = set(int(group_ids[i]) for i in order[n_val:]) or {int(group_ids[order[-1]])}
        assert not (val_groups & train_groups), "train/val episode overlap"
        return self._subset(train_groups), self._subset(val_groups)

    def _subset(self, groups) -> "BCDataset":
        mask = np.isin(self.group_ids, np.array(sorted(groups), dtype=np.int64))
        episodes = [m for m in self.episodes if m.group_id in groups]
        return BCDataset(
            self.observations[mask],
            self.actions[mask],
            self.group_ids[mask],
            self.seeds[mask],
            self.episode_ids[mask],
            self.step_ids[mask],
            self.dones[mask],
            self.truncated[mask],
            episodes,
        )

    def normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-feature mean and (floored) std of the observations."""
        mean = self.observations.mean(axis=0)
        std = self.observations.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)

    # ------------------------------------------------------------------ io
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = [vars(m) for m in self.episodes]
        np.savez_compressed(
            path,
            observations=self.observations,
            actions=self.actions,
            group_ids=self.group_ids,
            seeds=self.seeds,
            episode_ids=self.episode_ids,
            step_ids=self.step_ids,
            dones=self.dones,
            truncated=self.truncated,
            episodes_json=np.array(json.dumps(meta)),
        )

    @classmethod
    def load_many(cls, paths) -> "BCDataset":
        """Load and concatenate several saved datasets, re-indexing episode groups."""
        datasets = [cls.load(p) for p in paths]
        datasets = [d for d in datasets if len(d) > 0]
        if not datasets:
            raise DatasetIntegrityError("no non-empty datasets to combine")
        obs, act, gid, seeds, eids, sids, dones, truncs = [], [], [], [], [], [], [], []
        episodes: List[EpisodeMeta] = []
        offset = 0
        for ds in datasets:
            local_groups = sorted({int(g) for g in ds.group_ids})
            remap = {g: offset + i for i, g in enumerate(local_groups)}
            obs.append(ds.observations)
            act.append(ds.actions)
            gid.append(np.array([remap[int(g)] for g in ds.group_ids], dtype=np.int64))
            seeds.append(ds.seeds)
            eids.append(ds.episode_ids)
            sids.append(ds.step_ids)
            dones.append(ds.dones)
            truncs.append(ds.truncated)
            for m in ds.episodes:
                episodes.append(
                    EpisodeMeta(
                        group_id=remap[m.group_id],
                        episode_id=m.episode_id,
                        seed=m.seed,
                        track=m.track,
                        controller=m.controller,
                        length=m.length,
                        final_status=m.final_status,
                        gate_crossings=m.gate_crossings,
                    )
                )
            offset += len(local_groups)
        return cls(
            np.concatenate(obs), np.concatenate(act), np.concatenate(gid), np.concatenate(seeds),
            np.concatenate(eids), np.concatenate(sids), np.concatenate(dones), np.concatenate(truncs), episodes,
        )

    @classmethod
    def load(cls, path) -> "BCDataset":
        data = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(data["episodes_json"]))
        episodes = [EpisodeMeta(**m) for m in meta]
        return cls(
            data["observations"],
            data["actions"],
            data["group_ids"],
            data["seeds"],
            data["episode_ids"],
            data["step_ids"],
            data["dones"],
            data["truncated"],
            episodes,
        )


# Structural typing hint for from_records (avoids importing the recorder here).
class EpisodeRecordLike:  # pragma: no cover - documentation only
    observations: np.ndarray
    actions: np.ndarray
    step_ids: np.ndarray
    dones: np.ndarray
    truncated: np.ndarray
    seed: int
    episode_id: int
    track: str
    controller: str
    final_status: str
    gate_crossings: int
