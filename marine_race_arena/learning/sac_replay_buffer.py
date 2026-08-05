"""Per-worker n-step accumulation and deterministic stratified SAC replay."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from marine_race_arena.learning.config import ACTION_DIM
from marine_race_arena.learning.config_local_transition import (
    FEATURE_NAMES_LOCAL_TRANSITION,
    OBS_DIM_LOCAL_TRANSITION,
)


EVENT_CATEGORIES = (
    "general",
    "pre_crossing",
    "crossing",
    "post_target_switch",
    "successful_transition",
    "collision_entry",
    "missed_gate",
    "acquisition_timeout",
    "wrong_direction",
    "out_of_bounds",
)
EVENT_TO_CODE = {name: index for index, name in enumerate(EVENT_CATEGORIES)}
CODE_TO_EVENT = dict(enumerate(EVENT_CATEGORIES))
REPLAY_GROUPS = {
    "general": ("general",),
    "transition": ("pre_crossing", "crossing", "post_target_switch"),
    "success": ("successful_transition",),
    "safety": (
        "collision_entry",
        "missed_gate",
        "acquisition_timeout",
        "wrong_direction",
        "out_of_bounds",
    ),
}
DEFAULT_BATCH_COMPOSITION = {
    "general": 0.35,
    "transition": 0.30,
    "success": 0.20,
    "safety": 0.15,
}
TARGET_CHANGED_INDEX = FEATURE_NAMES_LOCAL_TRANSITION.index("target_changed_recently")


@dataclass
class RawTransition:
    observation: list[float]
    action: list[float]
    reward: float
    next_observation: list[float]
    terminated: bool
    truncated: bool
    bootstrap_allowed: bool
    event_category: str


@dataclass
class NStepTransition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    discount: float
    event_category: str


def classify_replay_event(
    info: Mapping[str, Any],
    observation: np.ndarray,
    next_observation: np.ndarray,
) -> str:
    """Training-only event label; never appended to the actor observation."""

    components = dict(info.get("reward_components") or {})
    if float(components.get("collision_penalty", 0.0)) < 0.0:
        return "collision_entry"
    if float(components.get("missed_gate_penalty", 0.0)) < 0.0 or int(
        info.get("episode_missed_gate_dnf", 0) or 0
    ) > 0:
        return "missed_gate"
    if float(components.get("acquisition_timeout_penalty", 0.0)) < 0.0 or bool(
        info.get("episode_acquisition_timeout", False)
    ):
        return "acquisition_timeout"
    if float(components.get("wrong_direction_penalty", 0.0)) < 0.0 or int(
        info.get("episode_wrong_direction", 0) or 0
    ) > 0:
        return "wrong_direction"
    if float(components.get("out_of_bounds_penalty", 0.0)) < 0.0 or int(
        info.get("episode_out_of_bounds", 0) or 0
    ) > 0:
        return "out_of_bounds"
    if bool(info.get("episode_full_sequence_completion", False)) or (
        bool(info.get("transition_focus_window_complete", False))
        and not bool(info.get("episode_acquisition_timeout", False))
    ):
        return "successful_transition"
    if float(components.get("gate_crossing", 0.0)) > 0.0:
        return "crossing"
    next_array = np.asarray(next_observation, dtype=np.float32)
    if next_array.shape == (OBS_DIM_LOCAL_TRANSITION,) and next_array[TARGET_CHANGED_INDEX] > 0.5:
        return "post_target_switch"
    current = np.asarray(observation, dtype=np.float32)
    # A visible, near target is a useful pre-crossing event.  The indices are
    # existing legal policy features, not privileged replay labels.
    if current.shape == (OBS_DIM_LOCAL_TRANSITION,) and current[0] > 0.5 and current[4] < 0.18:
        return "pre_crossing"
    return "general"


def is_legal_bootstrap_truncation(info: Mapping[str, Any]) -> bool:
    """Bootstrap time-limit truncations, never domain safety failures."""

    failures = (
        bool(info.get("episode_collision", False)),
        int(info.get("episode_missed_gate_dnf", 0) or 0) > 0,
        int(info.get("episode_out_of_bounds", 0) or 0) > 0,
        int(info.get("episode_wrong_direction", 0) or 0) > 0,
        bool(info.get("episode_acquisition_timeout", False)),
    )
    return bool(info.get("TimeLimit.truncated", False)) and not any(failures)


class MultiWorkerNStepAccumulator:
    def __init__(self, *, n_step: int = 3, gamma: float = 0.995, n_workers: int = 1) -> None:
        if int(n_step) < 1 or int(n_workers) < 1:
            raise ValueError("n_step and n_workers must be positive")
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.queues = [deque() for _ in range(int(n_workers))]

    def _emit(self, worker_id: int, length: int) -> NStepTransition:
        queue = self.queues[int(worker_id)]
        rows = list(queue)[: int(length)]
        first = rows[0]
        reward = sum((self.gamma ** index) * row.reward for index, row in enumerate(rows))
        final = rows[-1]
        discount = self.gamma ** len(rows) if final.bootstrap_allowed else 0.0
        # Preserve the rarest/most informative event in the accumulated span.
        priority = {name: index for index, name in enumerate(EVENT_CATEGORIES)}
        category = max((row.event_category for row in rows), key=priority.get)
        queue.popleft()
        return NStepTransition(
            observation=np.asarray(first.observation, dtype=np.float32),
            action=np.asarray(first.action, dtype=np.float32),
            reward=float(reward),
            next_observation=np.asarray(final.next_observation, dtype=np.float32),
            discount=float(discount),
            event_category=category,
        )

    def append(
        self,
        worker_id: int,
        *,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
        bootstrap_allowed: bool,
        event_category: str,
    ) -> list[NStepTransition]:
        if event_category not in EVENT_TO_CODE:
            raise ValueError(event_category)
        queue = self.queues[int(worker_id)]
        queue.append(RawTransition(
            observation=np.asarray(observation, dtype=np.float32).tolist(),
            action=np.asarray(action, dtype=np.float32).tolist(),
            reward=float(reward),
            next_observation=np.asarray(next_observation, dtype=np.float32).tolist(),
            terminated=bool(terminated),
            truncated=bool(truncated),
            bootstrap_allowed=bool(bootstrap_allowed and not terminated),
            event_category=event_category,
        ))
        emitted = []
        if len(queue) >= self.n_step:
            emitted.append(self._emit(worker_id, self.n_step))
        if terminated or truncated:
            while queue:
                emitted.append(self._emit(worker_id, min(self.n_step, len(queue))))
        return emitted

    def flush_worker(self, worker_id: int, *, bootstrap_allowed: bool = False) -> list[NStepTransition]:
        queue = self.queues[int(worker_id)]
        if queue:
            final = queue[-1]
            final.bootstrap_allowed = bool(bootstrap_allowed)
        emitted = []
        while queue:
            emitted.append(self._emit(worker_id, min(self.n_step, len(queue))))
        return emitted

    def flush_all(self, *, bootstrap_allowed: bool = False) -> list[NStepTransition]:
        rows = []
        for worker_id in range(len(self.queues)):
            rows.extend(self.flush_worker(worker_id, bootstrap_allowed=bootstrap_allowed))
        return rows

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "multiworker_n_step_v1",
            "n_step": self.n_step,
            "gamma": self.gamma,
            "queues": [[asdict(row) for row in queue] for queue in self.queues],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != "multiworker_n_step_v1":
            raise ValueError("unsupported n-step state")
        if int(state["n_step"]) != self.n_step or not np.isclose(float(state["gamma"]), self.gamma):
            raise ValueError("n-step configuration changed")
        queues = list(state.get("queues") or [])
        if len(queues) != len(self.queues):
            raise ValueError("n-step worker count changed")
        self.queues = [deque(RawTransition(**dict(row)) for row in queue) for queue in queues]


class StratifiedReplayBuffer:
    def __init__(
        self,
        *,
        capacity: int = 1_000_000,
        observation_dim: int = OBS_DIM_LOCAL_TRANSITION,
        action_dim: int = ACTION_DIM,
        seed: int = 23001,
        composition: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.observations = np.empty((self.capacity, self.observation_dim), dtype=np.float32)
        self.actions = np.empty((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.next_observations = np.empty((self.capacity, self.observation_dim), dtype=np.float32)
        self.discounts = np.empty(self.capacity, dtype=np.float32)
        self.categories = np.empty(self.capacity, dtype=np.uint8)
        self.position = 0
        self.size = 0
        self.total_added = 0
        self.rng = np.random.default_rng(int(seed))
        self.composition = dict(DEFAULT_BATCH_COMPOSITION if composition is None else composition)
        if set(self.composition) != set(DEFAULT_BATCH_COMPOSITION):
            raise ValueError("replay composition groups changed")
        if not np.isclose(sum(self.composition.values()), 1.0):
            raise ValueError("replay composition must sum to one")
        if float(self.composition["general"]) < 0.25:
            raise ValueError("replay batches require a substantial general component")
        self.sampled_category_counts = {name: 0 for name in EVENT_CATEGORIES}
        self.sampled_group_counts = {name: 0 for name in REPLAY_GROUPS}
        self.sampled_total = 0

    def add(self, transition: NStepTransition) -> None:
        index = self.position
        self.observations[index] = transition.observation
        self.actions[index] = transition.action
        self.rewards[index] = transition.reward
        self.next_observations[index] = transition.next_observation
        self.discounts[index] = transition.discount
        self.categories[index] = EVENT_TO_CODE[transition.event_category]
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1

    def extend(self, transitions: Iterable[NStepTransition]) -> None:
        for transition in transitions:
            self.add(transition)

    def _pool(self, group: str) -> np.ndarray:
        codes = np.asarray([EVENT_TO_CODE[name] for name in REPLAY_GROUPS[group]], dtype=np.uint8)
        return np.flatnonzero(np.isin(self.categories[: self.size], codes))

    @staticmethod
    def _quotas(batch_size: int, composition: Mapping[str, float]) -> Dict[str, int]:
        groups = list(DEFAULT_BATCH_COMPOSITION)
        quotas = {name: int(math_floor(batch_size * float(composition[name]))) for name in groups}
        for name in groups[: int(batch_size) - sum(quotas.values())]:
            quotas[name] += 1
        return quotas

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self.size < 1:
            raise ValueError("cannot sample an empty replay buffer")
        quotas = self._quotas(int(batch_size), self.composition)
        selections = []
        all_indices = np.arange(self.size)
        for group, count in quotas.items():
            pool = self._pool(group)
            if pool.size == 0:
                # Early in a run a rare stratum may be empty.  Falling back to
                # the whole replay preserves batch size without inventing data.
                pool = all_indices
            selections.append(self.rng.choice(pool, size=count, replace=pool.size < count))
        indices = np.concatenate(selections)
        self.rng.shuffle(indices)
        sampled_codes = self.categories[indices]
        for code in sampled_codes:
            self.sampled_category_counts[CODE_TO_EVENT[int(code)]] += 1
        for group, names in REPLAY_GROUPS.items():
            codes = [EVENT_TO_CODE[name] for name in names]
            self.sampled_group_counts[group] += int(np.isin(sampled_codes, codes).sum())
        self.sampled_total += len(indices)
        return {
            "observations": self.observations[indices].copy(),
            "actions": self.actions[indices].copy(),
            "rewards": self.rewards[indices].copy(),
            "next_observations": self.next_observations[indices].copy(),
            "discounts": self.discounts[indices].copy(),
            "event_categories": sampled_codes.copy(),
            "indices": indices,
        }

    def sampling_report(self) -> Dict[str, Any]:
        total = max(1, self.sampled_total)
        return {
            "configured_composition": dict(self.composition),
            "sampled_total": self.sampled_total,
            "actual_group_proportions": {
                name: count / total for name, count in self.sampled_group_counts.items()
            },
            "actual_category_proportions": {
                name: count / total for name, count in self.sampled_category_counts.items()
            },
        }

    def metadata_state(self) -> Dict[str, Any]:
        return {
            "schema_version": "stratified_sac_replay_v1",
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "position": self.position,
            "size": self.size,
            "total_added": self.total_added,
            "composition": dict(self.composition),
            "rng_state": self.rng.bit_generator.state,
            "sampled_category_counts": dict(self.sampled_category_counts),
            "sampled_group_counts": dict(self.sampled_group_counts),
            "sampled_total": self.sampled_total,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            np.savez(
                handle,
                observations=self.observations[: self.size],
                actions=self.actions[: self.size],
                rewards=self.rewards[: self.size],
                next_observations=self.next_observations[: self.size],
                discounts=self.discounts[: self.size],
                categories=self.categories[: self.size],
                metadata=np.asarray(json.dumps(self.metadata_state())),
            )

    @classmethod
    def load(cls, path: str | Path):
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            replay = cls(
                capacity=int(metadata["capacity"]),
                observation_dim=int(metadata["observation_dim"]),
                action_dim=int(metadata["action_dim"]),
                composition=metadata["composition"],
            )
            size = int(metadata["size"])
            replay.observations[:size] = archive["observations"]
            replay.actions[:size] = archive["actions"]
            replay.rewards[:size] = archive["rewards"]
            replay.next_observations[:size] = archive["next_observations"]
            replay.discounts[:size] = archive["discounts"]
            replay.categories[:size] = archive["categories"]
        replay.size = size
        replay.position = int(metadata["position"])
        replay.total_added = int(metadata["total_added"])
        replay.rng.bit_generator.state = metadata["rng_state"]
        replay.sampled_category_counts = dict(metadata["sampled_category_counts"])
        replay.sampled_group_counts = dict(metadata["sampled_group_counts"])
        replay.sampled_total = int(metadata["sampled_total"])
        return replay


def math_floor(value: float) -> int:
    # Local helper keeps NumPy scalar rounding out of persisted quota semantics.
    return int(np.floor(float(value)))

