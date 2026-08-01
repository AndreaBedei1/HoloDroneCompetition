"""Procedural S1--S5 curriculum for PPO-only ordered gate sequences."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


SEQUENCE_STAGES = ("S1", "S2", "S3", "S4", "S5")
BASE_TRACK = Path("marine_race_arena/tracks/training/stage3_three_gates.json")


@dataclass(frozen=True)
class SequenceStage:
    key: str
    min_gates: int
    max_gates: int
    max_turn_deg: float
    max_vertical_step_m: float
    min_spacing_m: float
    max_spacing_m: float
    completion_threshold: float
    minimum_timesteps: int


STAGES: Tuple[SequenceStage, ...] = (
    SequenceStage("S1", 3, 3, 25.0, 1.0, 4.0, 6.0, 0.95, 50_000),
    SequenceStage("S2", 4, 5, 35.0, 1.2, 3.5, 7.0, 0.95, 75_000),
    SequenceStage("S3", 6, 8, 42.0, 1.4, 3.5, 7.5, 0.90, 125_000),
    SequenceStage("S4", 9, 12, 45.0, 1.5, 3.5, 8.0, 0.90, 175_000),
    SequenceStage("S5", 13, 22, 45.0, 1.6, 3.5, 8.5, 0.90, 250_000),
)
STAGE_BY_KEY = {stage.key: stage for stage in STAGES}


@dataclass(frozen=True)
class SequenceGeometry:
    stage: str
    source: str
    pattern: str
    seed: int
    gate_count: int
    spacings_m: Tuple[float, ...]
    turn_deltas_deg: Tuple[float, ...]
    vertical_deltas_m: Tuple[float, ...]
    initial_yaw_error_deg: float
    initial_lateral_offset_m: float

    @property
    def geometry_group(self) -> str:
        return f"{self.stage}:{self.gate_count}:{self.pattern}:{self.source}"


@dataclass
class SequenceCurriculumState:
    current_stage: str = "S1"
    training_timesteps: int = 0
    stage_entry_timesteps: int = 0
    consecutive_qualifying_evaluations: int = 0
    efficiency_reward_active: bool = False
    replay_mixture: Dict[str, float] = field(default_factory=dict)
    sample_counts: Dict[str, int] = field(default_factory=dict)
    evaluation_history: list[Dict[str, Any]] = field(default_factory=list)
    curriculum_stage_history: list[Dict[str, Any]] = field(default_factory=list)
    targeted_failures: list[Dict[str, Any]] = field(default_factory=list)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def _pattern_deltas(
    rng: np.random.Generator,
    *,
    stage: SequenceStage,
    gate_count: int,
    pattern: str,
    difficulty: float,
) -> tuple[Tuple[float, ...], Tuple[float, ...]]:
    n = gate_count - 1
    turn_cap = max(5.0, stage.max_turn_deg * difficulty)
    vertical_cap = stage.max_vertical_step_m * difficulty
    if pattern == "straight":
        turns = rng.uniform(-3.0, 3.0, n)
        vertical = rng.uniform(-0.08, 0.08, n)
    elif pattern == "s_curve_left":
        turns = np.asarray([turn_cap * (1 if i % 2 == 0 else -1) for i in range(n)])
        vertical = rng.uniform(-vertical_cap, vertical_cap, n)
    elif pattern == "s_curve_right":
        turns = np.asarray([-turn_cap * (1 if i % 2 == 0 else -1) for i in range(n)])
        vertical = rng.uniform(-vertical_cap, vertical_cap, n)
    elif pattern == "vertical_serpent":
        turns = rng.uniform(-0.6 * turn_cap, 0.6 * turn_cap, n)
        vertical = np.asarray([
            vertical_cap * (1 if i % 2 == 0 else -1) for i in range(n)
        ])
    elif pattern == "high_to_low":
        turns = rng.uniform(-0.5 * turn_cap, 0.5 * turn_cap, n)
        vertical = np.full(n, vertical_cap)
    elif pattern == "low_to_high":
        turns = rng.uniform(-0.5 * turn_cap, 0.5 * turn_cap, n)
        vertical = np.full(n, -vertical_cap)
    else:
        turns = rng.uniform(-turn_cap, turn_cap, n)
        # Preserve occasional near-straight recovery segments in long paths.
        if n >= 5:
            turns[rng.choice(n, size=max(1, n // 5), replace=False)] *= 0.12
        vertical = rng.uniform(-vertical_cap, vertical_cap, n)
    return tuple(float(v) for v in turns), tuple(float(v) for v in vertical)


def generate_sequence_track(
    geometry: SequenceGeometry,
    output_path: str | Path,
    *,
    base_track: str | Path = BASE_TRACK,
) -> Path:
    """Generate current-free geometry without embedding future layout in observations."""
    data = json.loads(Path(base_track).read_text(encoding="utf-8"))
    yaw = 0.0
    x = y = 0.0
    z = -4.0
    gates = []
    positions = []
    for index in range(geometry.gate_count):
        if index:
            yaw += geometry.turn_deltas_deg[index - 1]
            spacing = geometry.spacings_m[index - 1]
            x += spacing * math.cos(math.radians(yaw))
            y += spacing * math.sin(math.radians(yaw))
            z = float(np.clip(z + geometry.vertical_deltas_m[index - 1], -8.0, -2.0))
        direction = [math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0.0]
        gate_id = f"G{index + 1:02d}"
        positions.append((x, y, z))
        gates.append({
            "id": gate_id,
            "type": "single",
            "position": [round(x, 5), round(y, 5), round(z, 5)],
            "rotation_rpy_deg": [0.0, 0.0, round(yaw, 5)],
            "color": "#00ff88",
            "passage_direction": [round(v, 7) for v in direction],
        })

    first_direction = gates[0]["passage_direction"]
    start = [
        -4.5 * first_direction[0],
        -4.5 * first_direction[1] + geometry.initial_lateral_offset_m,
        -4.0,
    ]
    data["race"].update({
        "name": f"PPO sequence {geometry.geometry_group}",
        "official_mode": False,
        "expected_gates_per_lap": geometry.gate_count,
        "max_duration_s": max(240, geometry.gate_count * 45),
    })
    # Sequence PPO is deliberately current-free.  Do not inherit a task such
    # as ``current_gate`` when callers provide a different base track: removing
    # currents while retaining that task produces an invalid track contract.
    data["benchmark_task"] = {"mode": "clean_gate"}
    data["track"]["gate_sequence"] = [gate["id"] for gate in gates]
    data["track"]["declared_length_m"] = round(4.5 + sum(geometry.spacings_m), 3)
    data["track"]["length_tolerance_m"] = max(2.0, geometry.gate_count * 0.5)
    data["gates"] = gates
    data["finish"]["gate_id"] = gates[-1]["id"]
    data["start"]["position"] = [round(v, 5) for v in start]
    data["start"]["rotation_rpy_deg"] = [0.0, 0.0, geometry.initial_yaw_error_deg]
    data["participants"][0]["spawn"] = copy.deepcopy(data["start"])
    data["participants"][0]["controller"] = "rl_sequence_ppo"
    data["participants"][0]["controller_class"] = None
    data["currents"] = []
    data["obstacles"] = []
    xs = [start[0], *(item[0] for item in positions)]
    ys = [start[1], *(item[1] for item in positions)]
    zs = [start[2], *(item[2] for item in positions)]
    data["world"]["bounds"] = {
        "x_min": min(xs) - 8.0, "x_max": max(xs) + 8.0,
        "y_min": min(ys) - 8.0, "y_max": max(ys) + 8.0,
        "z_min": min(zs) - 3.0, "z_max": max(zs) + 3.0,
    }
    data["sequence_curriculum"] = asdict(geometry)
    target = Path(output_path)
    _atomic_json(target, data)
    return target


class SequenceCurriculumSampler:
    """Deterministic, resumable sequence sampler with gradual stage difficulty."""

    DEFAULT_MIXTURE = {
        "current_stage": 0.65,
        "three_gate": 0.15,
        "previous_stage": 0.10,
        "short_retention": 0.08,
        "targeted_failure": 0.02,
    }

    def __init__(
        self,
        *,
        seed: int,
        initial_stage: str = "S1",
        maximum_stage: str = "S5",
        replay_mixture: Optional[Mapping[str, float]] = None,
    ) -> None:
        if initial_stage not in STAGE_BY_KEY or maximum_stage not in STAGE_BY_KEY:
            raise ValueError("unknown sequence stage")
        if SEQUENCE_STAGES.index(maximum_stage) < SEQUENCE_STAGES.index(initial_stage):
            raise ValueError("maximum stage precedes initial stage")
        mixture = dict(replay_mixture or self.DEFAULT_MIXTURE)
        if set(mixture) != set(self.DEFAULT_MIXTURE) or not np.isclose(sum(mixture.values()), 1.0):
            raise ValueError("invalid sequence replay mixture")
        self.seed = int(seed)
        self.maximum_stage = maximum_stage
        self.rng = np.random.default_rng(seed)
        self.state = SequenceCurriculumState(
            current_stage=initial_stage,
            replay_mixture={key: float(value) for key, value in mixture.items()},
            curriculum_stage_history=[{
                "timesteps": 0,
                "from": None,
                "to": initial_stage,
                "reason": "initialized",
            }],
        )

    @property
    def current_stage(self) -> str:
        return self.state.current_stage

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "ppo_sequence_curriculum_v1",
            "seed": self.seed,
            "maximum_stage": self.maximum_stage,
            "rng_state": self.rng.bit_generator.state,
            "state": asdict(self.state),
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if value.get("schema_version") != "ppo_sequence_curriculum_v1":
            raise ValueError("unsupported sequence curriculum state")
        if int(value["seed"]) != self.seed or value["maximum_stage"] != self.maximum_stage:
            raise ValueError("sequence curriculum contract changed")
        self.state = SequenceCurriculumState(**dict(value["state"]))
        self.rng.bit_generator.state = value["rng_state"]

    def set_timesteps(self, timesteps: int) -> None:
        self.state.training_timesteps = max(self.state.training_timesteps, int(timesteps))

    def add_failure_case(self, geometry: SequenceGeometry, reason: str) -> None:
        self.state.targeted_failures.append({"geometry": asdict(geometry), "reason": reason})
        self.state.targeted_failures = self.state.targeted_failures[-100:]

    def sample(self) -> SequenceGeometry:
        source = str(self.rng.choice(
            list(self.state.replay_mixture),
            p=list(self.state.replay_mixture.values()),
        ))
        current_index = SEQUENCE_STAGES.index(self.current_stage)
        if source == "targeted_failure" and self.state.targeted_failures:
            saved = dict(self.rng.choice(self.state.targeted_failures)["geometry"])
            saved["seed"] = int(self.rng.integers(0, 2**31 - 1))
            saved["source"] = source
            geometry = SequenceGeometry(**saved)
        else:
            if source == "short_retention":
                stage = STAGE_BY_KEY["S1"]
                gate_count = int(self.rng.integers(1, 3))
            elif source == "three_gate":
                stage = STAGE_BY_KEY["S1"]
                gate_count = 3
            elif source == "previous_stage":
                stage = STAGES[max(0, current_index - 1)]
                gate_count = int(self.rng.integers(stage.min_gates, stage.max_gates + 1))
            else:
                stage = STAGE_BY_KEY[self.current_stage]
                gate_count = int(self.rng.integers(stage.min_gates, stage.max_gates + 1))
            elapsed = self.state.training_timesteps - self.state.stage_entry_timesteps
            difficulty = float(np.clip(0.35 + 0.65 * elapsed / max(1, stage.minimum_timesteps), 0.35, 1.0))
            patterns = ["straight", "s_curve_left", "s_curve_right", "mixed", "high_to_low", "low_to_high"]
            if stage.key in {"S3", "S4", "S5"}:
                patterns += ["vertical_serpent", "mixed", "mixed"]
            pattern = str(self.rng.choice(patterns))
            turns, vertical = _pattern_deltas(
                self.rng, stage=stage, gate_count=gate_count,
                pattern=pattern, difficulty=difficulty,
            )
            spacings = tuple(float(v) for v in self.rng.uniform(
                stage.min_spacing_m,
                stage.min_spacing_m + (stage.max_spacing_m - stage.min_spacing_m) * difficulty,
                max(0, gate_count - 1),
            ))
            geometry = SequenceGeometry(
                stage=self.current_stage,
                source=source,
                pattern=pattern,
                seed=int(self.rng.integers(0, 2**31 - 1)),
                gate_count=gate_count,
                spacings_m=spacings,
                turn_deltas_deg=turns,
                vertical_deltas_m=vertical,
                initial_yaw_error_deg=float(self.rng.uniform(-20.0, 20.0) * difficulty),
                initial_lateral_offset_m=float(self.rng.uniform(-1.0, 1.0) * difficulty),
            )
        key = geometry.geometry_group
        self.state.sample_counts[key] = self.state.sample_counts.get(key, 0) + 1
        return geometry

    def observe_full_evaluation(self, metrics: Mapping[str, Any], timesteps: int) -> bool:
        stage = STAGE_BY_KEY[self.current_stage]
        safety_clean = all(int(metrics.get(key, 0) or 0) == 0 for key in (
            "collision_episodes", "out_of_bounds_episodes", "wrong_direction_events",
            "previous_gate_returns", "missed_gate_dnf",
        ))
        rate = metrics.get("full_sequence_completion_rate")
        if self.current_stage == "S1":
            measured = [
                metrics.get("three_gate_completion_rate"),
                metrics.get("current_stage_completion_rate"),
            ]
            measured = [float(value) for value in measured if value is not None]
            if measured:
                rate = min(measured)
        elif metrics.get("current_stage_completion_rate") is not None:
            rate = metrics.get("current_stage_completion_rate")
        qualifies = rate is not None and float(rate) >= stage.completion_threshold and safety_clean
        self.state.consecutive_qualifying_evaluations = (
            self.state.consecutive_qualifying_evaluations + 1 if qualifies else 0
        )
        self.state.efficiency_reward_active = self.state.consecutive_qualifying_evaluations >= 2
        row = {"timesteps": int(timesteps), "stage": self.current_stage,
               "qualifies": qualifies, "metrics": dict(metrics)}
        self.state.evaluation_history.append(row)
        elapsed = int(timesteps) - self.state.stage_entry_timesteps
        if (qualifies and self.state.consecutive_qualifying_evaluations >= 2
                and elapsed >= stage.minimum_timesteps
                and self.current_stage != self.maximum_stage):
            old = self.current_stage
            new = SEQUENCE_STAGES[SEQUENCE_STAGES.index(old) + 1]
            self.state.current_stage = new
            self.state.stage_entry_timesteps = int(timesteps)
            self.state.consecutive_qualifying_evaluations = 0
            self.state.efficiency_reward_active = False
            self.state.curriculum_stage_history.append({
                "timesteps": int(timesteps), "from": old, "to": new,
                "reason": "two_safety_clean_full_evaluations",
            })
            return True
        return False
