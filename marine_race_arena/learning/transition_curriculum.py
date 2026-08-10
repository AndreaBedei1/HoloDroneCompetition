"""Geometry-only curriculum for universal local gate transitions."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from marine_race_arena.learning.generic_sequence_curriculum import (
    GenericSequenceCurriculum,
)

from marine_race_arena.learning.sequence_curriculum import (
    BASE_TRACK,
    SequenceGeometry,
    _atomic_json,
    generate_sequence_track,
)


DIFFICULTY_LEVELS = ("G1", "G2", "G3", "G4", "G5", "G6")
FULL_SEQUENCE_LENGTHS = (3, 5, 8, 12, 17, 22)


@dataclass(frozen=True)
class GeometricDifficulty:
    key: str
    max_turn_deg: float
    max_vertical_step_m: float
    min_spacing_m: float
    max_spacing_m: float
    max_initial_yaw_error_deg: float
    max_lateral_offset_m: float
    max_initial_speed_m_s: float
    minimum_transitions: int


DIFFICULTIES: Tuple[GeometricDifficulty, ...] = (
    GeometricDifficulty("G1", 8.0, 0.20, 4.5, 6.0, 8.0, 0.35, 0.15, 40_000),
    GeometricDifficulty("G2", 25.0, 0.35, 4.0, 7.0, 15.0, 0.55, 0.25, 60_000),
    GeometricDifficulty("G3", 18.0, 1.20, 4.0, 7.5, 15.0, 0.65, 0.30, 80_000),
    GeometricDifficulty("G4", 35.0, 1.35, 3.5, 8.0, 20.0, 0.80, 0.40, 100_000),
    GeometricDifficulty("G5", 45.0, 1.65, 3.2, 8.5, 25.0, 1.00, 0.55, 125_000),
    GeometricDifficulty("G6", 55.0, 2.00, 3.0, 10.0, 30.0, 1.20, 0.70, 150_000),
)
DIFFICULTY_BY_KEY = {item.key: item for item in DIFFICULTIES}


@dataclass(frozen=True)
class TransitionGeometry:
    difficulty: str
    episode_type: str
    pattern: str
    seed: int
    gate_count: int
    spacings_m: Tuple[float, ...]
    turn_deltas_deg: Tuple[float, ...]
    vertical_deltas_m: Tuple[float, ...]
    initial_yaw_error_deg: float
    initial_lateral_offset_m: float
    start_distance_m: float
    initial_body_velocity_m_s: Tuple[float, float, float]
    #: Provenance, set by the sampler.  Never observable by the policy.
    dataset_split: Optional[str] = None
    sequence_bucket: Optional[str] = None
    curriculum_stage: Optional[str] = None

    @property
    def geometry_group(self) -> str:
        return (
            f"{self.difficulty}:{self.episode_type}:{self.gate_count}:"
            f"{self.pattern}"
        )


@dataclass
class TransitionWorkerState:
    difficulty: str = "G1"
    total_environment_transitions: int = 0
    sample_counts: Dict[str, int] = field(default_factory=dict)
    bucket_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class TransitionCurriculumState:
    difficulty: str = "G1"
    total_environment_transitions: int = 0
    difficulty_entry_transitions: int = 0
    consecutive_qualifying_evaluations: int = 0
    efficiency_reward_active: bool = False
    evaluation_history: list[Dict[str, Any]] = field(default_factory=list)
    difficulty_history: list[Dict[str, Any]] = field(default_factory=list)


def _deltas(
    rng: np.random.Generator,
    difficulty: GeometricDifficulty,
    gate_count: int,
    pattern: str,
) -> tuple[Tuple[float, ...], Tuple[float, ...]]:
    n = max(0, gate_count - 1)
    if pattern == "aligned":
        turns = rng.uniform(-min(5.0, difficulty.max_turn_deg), min(5.0, difficulty.max_turn_deg), n)
        vertical = rng.uniform(-min(0.12, difficulty.max_vertical_step_m), min(0.12, difficulty.max_vertical_step_m), n)
    elif pattern == "yaw":
        signs = rng.choice((-1.0, 1.0), n)
        turns = signs * rng.uniform(0.45, 1.0, n) * difficulty.max_turn_deg
        vertical = rng.uniform(-0.15, 0.15, n)
    elif pattern == "elevation":
        turns = rng.uniform(-0.3, 0.3, n) * difficulty.max_turn_deg
        signs = rng.choice((-1.0, 1.0), n)
        vertical = signs * rng.uniform(0.45, 1.0, n) * difficulty.max_vertical_step_m
    elif pattern == "combined":
        turns = rng.uniform(-1.0, 1.0, n) * difficulty.max_turn_deg
        vertical = rng.uniform(-1.0, 1.0, n) * difficulty.max_vertical_step_m
    elif pattern == "official_serpent":
        turns = np.asarray([
            difficulty.max_turn_deg * (1.0 if index % 2 == 0 else -1.0)
            for index in range(n)
        ])
        vertical = np.asarray([
            difficulty.max_vertical_step_m * (-1.0 if index % 3 == 0 else 0.6)
            for index in range(n)
        ])
    else:
        turns = rng.uniform(-1.0, 1.0, n) * difficulty.max_turn_deg
        vertical = rng.uniform(-1.0, 1.0, n) * difficulty.max_vertical_step_m
    return tuple(map(float, turns)), tuple(map(float, vertical))


class TransitionGeometrySampler:
    """Independent deterministic sampler used by exactly one worker."""

    def __init__(
        self,
        *,
        seed: int,
        difficulty: str = "G1",
        transition_focus_fraction: float = 0.70,
        sequence_curriculum: Any = None,
        dataset_split: str = "train",
    ) -> None:
        if difficulty not in DIFFICULTY_BY_KEY:
            raise ValueError(f"unknown transition difficulty {difficulty!r}")
        if not 0.0 <= float(transition_focus_fraction) <= 1.0:
            raise ValueError("transition focus fraction must be in [0, 1]")
        self.seed = int(seed)
        self.transition_focus_fraction = float(transition_focus_fraction)
        # When a generic sequence curriculum is attached it *replaces* the
        # binary focus/full-sequence coin flip: episode length is drawn from the
        # current stage mixture instead.  This is the difference between the
        # curriculum existing and the curriculum running.
        self.sequence_curriculum = sequence_curriculum
        self.dataset_split = str(dataset_split)
        self.rng = np.random.default_rng(self.seed)
        self.state = TransitionWorkerState(difficulty=difficulty)

    @property
    def difficulty(self) -> str:
        return self.state.difficulty

    def set_difficulty(self, difficulty: str) -> None:
        if difficulty not in DIFFICULTY_BY_KEY:
            raise ValueError(difficulty)
        self.state.difficulty = difficulty

    def set_total_environment_transitions(self, value: int) -> None:
        self.state.total_environment_transitions = max(
            self.state.total_environment_transitions, int(value)
        )

    def _episode_seed(self) -> int:
        """Draw an episode seed from this sampler's dataset band.

        Enforcing the band here rather than by convention is what makes
        TRAIN/VALIDATION/TEST separation real: a rollout worker physically
        cannot emit a validation or test seed.
        """

        from marine_race_arena.learning.rl_holdout_policy import seed_group

        group = seed_group(self.dataset_split)
        span = int(group.end) - int(group.start)
        return int(group.start) + int(self.rng.integers(0, span))

    def bucket_counts(self) -> Dict[str, int]:
        """Observed episode counts per length bucket for this worker."""

        return dict(self.state.bucket_counts)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "transition_worker_sampler_v1",
            "seed": self.seed,
            "dataset_split": self.dataset_split,
            "transition_focus_fraction": self.transition_focus_fraction,
            "rng_state": self.rng.bit_generator.state,
            "state": asdict(self.state),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if value.get("schema_version") != "transition_worker_sampler_v1":
            raise ValueError("unsupported transition worker state")
        if int(value["seed"]) != self.seed:
            raise ValueError("transition worker seed changed")
        if not np.isclose(
            float(value["transition_focus_fraction"]),
            self.transition_focus_fraction,
        ):
            raise ValueError("transition focus fraction changed")
        self.state = TransitionWorkerState(**dict(value["state"]))
        self.rng.bit_generator.state = value["rng_state"]

    def sample(self, *, force_episode_type: Optional[str] = None) -> TransitionGeometry:
        difficulty = DIFFICULTY_BY_KEY[self.difficulty]
        if force_episode_type is not None:
            if force_episode_type not in {"transition_focus", "full_sequence"}:
                raise ValueError(force_episode_type)
            episode_type = force_episode_type
            gate_count = 2 if episode_type == "transition_focus" else int(
                self.rng.choice(FULL_SEQUENCE_LENGTHS)
            )
            bucket = "focus" if gate_count <= 2 else None
        elif self.sequence_curriculum is not None:
            gate_count = int(self.sequence_curriculum.sample_gate_count(self.rng))
            bucket = self.sequence_curriculum.bucket_of_gate_count(gate_count)
            episode_type = "transition_focus" if gate_count <= 2 else "full_sequence"
        else:
            episode_type = (
                "transition_focus"
                if self.rng.random() < self.transition_focus_fraction
                else "full_sequence"
            )
            gate_count = 2 if episode_type == "transition_focus" else int(
                self.rng.choice(FULL_SEQUENCE_LENGTHS)
            )
            bucket = "focus" if gate_count <= 2 else None
        allowed_patterns = ["aligned", "yaw", "elevation", "combined", "varied"]
        if self.difficulty in {"G5", "G6"}:
            allowed_patterns += ["official_serpent", "combined", "varied"]
        pattern = str(self.rng.choice(allowed_patterns))
        turns, vertical = _deltas(
            self.rng, difficulty, gate_count, pattern
        )
        spacings = tuple(map(float, self.rng.uniform(
            difficulty.min_spacing_m,
            difficulty.max_spacing_m,
            max(0, gate_count - 1),
        )))
        velocity_cap = difficulty.max_initial_speed_m_s
        geometry = TransitionGeometry(
            difficulty=self.difficulty,
            episode_type=episode_type,
            pattern=pattern,
            seed=self._episode_seed(),
            gate_count=gate_count,
            spacings_m=spacings,
            turn_deltas_deg=turns,
            vertical_deltas_m=vertical,
            initial_yaw_error_deg=float(self.rng.uniform(
                -difficulty.max_initial_yaw_error_deg,
                difficulty.max_initial_yaw_error_deg,
            )),
            initial_lateral_offset_m=float(self.rng.uniform(
                -difficulty.max_lateral_offset_m,
                difficulty.max_lateral_offset_m,
            )),
            start_distance_m=float(self.rng.uniform(
                0.8, 1.8
            ) if episode_type == "transition_focus" else 4.5),
            initial_body_velocity_m_s=(
                float(self.rng.uniform(0.0, velocity_cap)),
                float(self.rng.uniform(-0.5 * velocity_cap, 0.5 * velocity_cap)),
                float(self.rng.uniform(-0.35 * velocity_cap, 0.35 * velocity_cap)),
            ),
            # Provenance travels with the episode so the composition log and the
            # split assertions read exactly what the policy trained on.  None of
            # it is observable by the policy.
            dataset_split=self.dataset_split,
            sequence_bucket=(
                bucket if bucket
                else GenericSequenceCurriculum.bucket_of_gate_count(gate_count)
            ),
            curriculum_stage=(
                self.sequence_curriculum.stage["name"]
                if self.sequence_curriculum is not None else None
            ),
        )
        key = geometry.geometry_group
        self.state.sample_counts[key] = self.state.sample_counts.get(key, 0) + 1
        realised = geometry.sequence_bucket
        self.state.bucket_counts[realised] = (
            self.state.bucket_counts.get(realised, 0) + 1
        )
        return geometry


class TransitionCurriculumController:
    """Central promotion state; sequence length never affects difficulty."""

    def __init__(self, *, initial_difficulty: str = "G1", maximum_difficulty: str = "G6") -> None:
        if initial_difficulty not in DIFFICULTY_BY_KEY or maximum_difficulty not in DIFFICULTY_BY_KEY:
            raise ValueError("unknown geometric difficulty")
        self.maximum_difficulty = maximum_difficulty
        self.state = TransitionCurriculumState(
            difficulty=initial_difficulty,
            difficulty_history=[{
                "total_environment_transitions": 0,
                "from": None,
                "to": initial_difficulty,
                "reason": "initialized",
            }],
        )

    @property
    def difficulty(self) -> str:
        return self.state.difficulty

    def state_dict(self, worker_states: Optional[list[Mapping[str, Any]]] = None) -> Dict[str, Any]:
        return {
            "schema_version": "universal_transition_curriculum_v1",
            "maximum_difficulty": self.maximum_difficulty,
            "state": asdict(self.state),
            "worker_states": list(worker_states or []),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        if value.get("schema_version") != "universal_transition_curriculum_v1":
            raise ValueError("unsupported transition curriculum state")
        if value.get("maximum_difficulty") != self.maximum_difficulty:
            raise ValueError("maximum geometric difficulty changed")
        self.state = TransitionCurriculumState(**dict(value["state"]))
        return list(value.get("worker_states") or [])

    def set_total_environment_transitions(self, value: int) -> None:
        self.state.total_environment_transitions = max(
            self.state.total_environment_transitions, int(value)
        )

    def observe_evaluation(
        self,
        metrics: Mapping[str, Any],
        total_transitions: int,
        *,
        thresholds: Optional[Any] = None,
    ) -> bool:
        """Record an evaluation and decide whether geometry may be promoted.

        Promotion additionally requires the competence gate, so a policy that
        produced no safety events by remaining inactive cannot advance the
        geometric difficulty.  Sequence length is never a curriculum variable.
        """

        from marine_race_arena.learning.transition_selection import (
            DEFAULT_COMPETENCE_THRESHOLDS,
            evaluate_competence_gate,
        )

        self.set_total_environment_transitions(total_transitions)
        rate = float(metrics.get("universal_transition_success_rate", 0.0) or 0.0)
        safety_clean = all(int(metrics.get(key, 0) or 0) == 0 for key in (
            "collision_episodes", "out_of_bounds_episodes", "wrong_direction_events",
            "previous_gate_returns", "missed_gate_dnf", "acquisition_timeouts",
        ))
        verdict = evaluate_competence_gate(
            metrics, thresholds or DEFAULT_COMPETENCE_THRESHOLDS
        )
        qualifies = verdict.passed and safety_clean and rate >= 0.99
        self.state.consecutive_qualifying_evaluations = (
            self.state.consecutive_qualifying_evaluations + 1 if qualifies else 0
        )
        self.state.evaluation_history.append({
            "total_environment_transitions": int(total_transitions),
            "difficulty": self.difficulty,
            "qualifies": qualifies,
            "competence_classification": verdict.classification,
            "metrics": dict(metrics),
        })
        stage = DIFFICULTY_BY_KEY[self.difficulty]
        elapsed = int(total_transitions) - self.state.difficulty_entry_transitions
        index = DIFFICULTY_LEVELS.index(self.difficulty)
        max_index = DIFFICULTY_LEVELS.index(self.maximum_difficulty)
        if (
            qualifies
            and self.state.consecutive_qualifying_evaluations >= 2
            and elapsed >= stage.minimum_transitions
            and index < max_index
        ):
            previous = self.difficulty
            self.state.difficulty = DIFFICULTY_LEVELS[index + 1]
            self.state.difficulty_entry_transitions = int(total_transitions)
            self.state.consecutive_qualifying_evaluations = 0
            self.state.difficulty_history.append({
                "total_environment_transitions": int(total_transitions),
                "from": previous,
                "to": self.difficulty,
                "reason": "two_safety_clean_99pct_transition_evaluations",
            })
            return True
        if (
            self.difficulty == self.maximum_difficulty
            and qualifies
            and self.state.consecutive_qualifying_evaluations >= 3
        ):
            self.state.efficiency_reward_active = True
        return False


def generate_transition_track(
    geometry: TransitionGeometry,
    output_path: str | Path,
    *,
    frames_per_sec: bool | int = False,
    base_track: str | Path = BASE_TRACK,
) -> Path:
    """Generate a current-free track; privileged reset metadata stays off-policy."""

    sequence = SequenceGeometry(
        stage=geometry.difficulty,
        source=geometry.episode_type,
        pattern=geometry.pattern,
        seed=geometry.seed,
        gate_count=geometry.gate_count,
        spacings_m=geometry.spacings_m,
        turn_deltas_deg=geometry.turn_deltas_deg,
        vertical_deltas_m=geometry.vertical_deltas_m,
        initial_yaw_error_deg=geometry.initial_yaw_error_deg,
        initial_lateral_offset_m=geometry.initial_lateral_offset_m,
    )
    target = generate_sequence_track(sequence, output_path, base_track=base_track)
    data = json.loads(target.read_text(encoding="utf-8"))
    first = data["gates"][0]
    direction = first["passage_direction"]
    start = [
        float(first["position"][0]) - geometry.start_distance_m * float(direction[0]),
        float(first["position"][1]) - geometry.start_distance_m * float(direction[1])
        + geometry.initial_lateral_offset_m,
        float(first["position"][2]),
    ]
    data["start"]["position"] = [round(value, 5) for value in start]
    data["participants"][0]["spawn"] = copy.deepcopy(data["start"])
    path_positions = [
        data["start"]["position"],
        *(gate["position"] for gate in data["gates"]),
    ]
    data["track"]["declared_length_m"] = round(sum(
        math.dist(previous, current)
        for previous, current in zip(path_positions, path_positions[1:])
    ), 3)
    data["race"]["name"] = f"Universal transition {geometry.geometry_group}"
    if geometry.episode_type == "transition_focus":
        data["race"]["max_duration_s"] = 30
    data["holoocean_frames_per_sec"] = frames_per_sec
    data["universal_transition"] = asdict(geometry)
    _atomic_json(target, data)
    return target
