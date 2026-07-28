"""Parametric, replay-mixed curriculum for multi-gate turning transitions."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.longrun_config import CURRICULUM_STAGES
from marine_race_arena.learning.multigate_curriculum import (
    HORSESHOE,
    MIXED,
    SINGLE_GATE,
    SIX_GATE,
    THREE_GATE_S,
    TWO_GATE_STRAIGHT,
    VERTICAL,
)

BASE_TWO_GATE_TRACK = Path(
    "marine_race_arena/tracks/tests/two_gate_straight.json"
)
ANGLE_BANDS_DEG = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 45.0)


@dataclass(frozen=True)
class ParametricStage:
    key: str
    max_abs_turn_deg: float
    min_separation_m: float
    max_separation_m: float
    max_lateral_offset_m: float
    max_vertical_step_m: float
    max_start_yaw_error_deg: float
    max_start_lateral_offset_m: float
    gate_count: int = 2
    official_tracks: Tuple[str, ...] = ()


PARAMETRIC_STAGES: Tuple[ParametricStage, ...] = (
    ParametricStage("C0", 5.0, 3.5, 5.0, 0.30, 0.20, 5.0, 0.30),
    ParametricStage("C1", 10.0, 3.5, 6.0, 0.50, 0.30, 8.0, 0.50),
    ParametricStage("C2", 20.0, 3.0, 6.5, 0.75, 0.50, 12.0, 0.75),
    ParametricStage("C3", 30.0, 3.0, 7.0, 1.00, 0.70, 15.0, 1.00),
    ParametricStage("C4", 45.0, 2.8, 7.5, 1.25, 0.90, 20.0, 1.20),
    ParametricStage("C5", 45.0, 3.0, 7.5, 1.25, 1.20, 20.0, 1.20, gate_count=3),
    ParametricStage("C6", 45.0, 3.0, 8.0, 1.50, 1.50, 20.0, 1.20, gate_count=6),
    ParametricStage(
        "C7",
        45.0,
        3.0,
        8.0,
        1.50,
        1.50,
        20.0,
        1.20,
        gate_count=6,
        official_tracks=(HORSESHOE, VERTICAL, MIXED),
    ),
)
STAGE_BY_KEY = {stage.key: stage for stage in PARAMETRIC_STAGES}


@dataclass(frozen=True)
class TransitionGeometry:
    signed_turn_deg: float
    gate_separation_m: float
    lateral_displacement_m: float
    vertical_displacement_m: float
    starting_yaw_error_deg: float
    initial_lateral_offset_m: float
    beacon_bearing_noise_std_deg: float = 0.0
    beacon_range_noise_std_m: float = 0.0
    beacon_dropout_probability: float = 0.0
    visibility_scale: float = 1.0
    source: str = "current_stage"
    stage: str = "C0"

    @property
    def direction(self) -> str:
        # The retention/straight band intentionally includes small +/-5 degree
        # perturbations; turns start at the +/-10 degree band.
        if abs(self.signed_turn_deg) <= 5.0 + 1e-6:
            return "straight"
        if self.signed_turn_deg < 0:
            return "right"
        if self.signed_turn_deg > 0:
            return "left"
        return "straight"

    @property
    def angle_band(self) -> float:
        return min(ANGLE_BANDS_DEG, key=lambda value: abs(value - abs(self.signed_turn_deg)))

    @property
    def geometry_group(self) -> str:
        separation_bucket = round(self.gate_separation_m)
        vertical_bucket = round(self.vertical_displacement_m * 2.0) / 2.0
        return (
            f"{self.direction}:a{self.angle_band:g}:"
            f"d{separation_bucket:g}:z{vertical_bucket:+g}"
        )


@dataclass
class CurriculumState:
    current_stage: str = "C0"
    training_timesteps: int = 0
    stage_entry_timesteps: int = 0
    consecutive_full_passes: int = 0
    reliable_full_streak: int = 0
    efficiency_phase_active: bool = False
    sample_counts: Dict[str, int] = field(default_factory=dict)
    evaluation_history: List[Dict[str, Any]] = field(default_factory=list)
    stage_changes: List[Dict[str, Any]] = field(default_factory=list)
    targeted_failures: List[Dict[str, Any]] = field(default_factory=list)
    rng_state: Optional[Dict[str, Any]] = None


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def generate_two_gate_track(
    geometry: TransitionGeometry,
    output_path: str | Path,
    *,
    base_track: str | Path = BASE_TWO_GATE_TRACK,
) -> Path:
    """Write one current-free training track without changing referee semantics."""
    data = json.loads(Path(base_track).read_text(encoding="utf-8"))
    angle_rad = math.radians(float(geometry.signed_turn_deg))
    separation = float(geometry.gate_separation_m)
    gate2_x = separation * math.cos(angle_rad)
    gate2_y = (
        separation * math.sin(angle_rad)
        + float(geometry.lateral_displacement_m)
    )
    gate2_z = -4.0 + float(geometry.vertical_displacement_m)
    direction = [math.cos(angle_rad), math.sin(angle_rad), 0.0]

    data["race"]["name"] = (
        f"Parametric {geometry.stage} {geometry.signed_turn_deg:+.1f} degree"
    )
    data["race"]["official_mode"] = False
    data["race"]["max_duration_s"] = 180
    data["track"]["declared_length_m"] = round(4.0 + separation, 3)
    data["track"]["length_tolerance_m"] = 2.0
    data["start"]["position"] = [
        -4.0,
        float(geometry.initial_lateral_offset_m),
        -4.0,
    ]
    data["start"]["rotation_rpy_deg"] = [
        0.0,
        0.0,
        float(geometry.starting_yaw_error_deg),
    ]
    data["participants"][0]["spawn"] = copy.deepcopy(data["start"])
    data["gates"][0]["position"] = [0.0, 0.0, -4.0]
    data["gates"][0]["rotation_rpy_deg"] = [0.0, 0.0, 0.0]
    data["gates"][0]["passage_direction"] = [1.0, 0.0, 0.0]
    data["gates"][1]["position"] = [gate2_x, gate2_y, gate2_z]
    data["gates"][1]["rotation_rpy_deg"] = [
        0.0,
        0.0,
        float(geometry.signed_turn_deg),
    ]
    data["gates"][1]["passage_direction"] = direction
    data["beacon"]["angular_noise_std_deg"] = float(
        geometry.beacon_bearing_noise_std_deg
    )
    data["beacon"]["range_noise_std_m"] = float(
        geometry.beacon_range_noise_std_m
    )
    data["beacon"]["dropout_probability"] = float(
        geometry.beacon_dropout_probability
    )
    for sensor in data["participants"][0]["sensors"].get(
        "holoocean_sensors", []
    ):
        if sensor.get("sensor_name") == "FrontCamera":
            config = sensor.setdefault("configuration", {})
            config["FovAngle"] = float(
                np.clip(90.0 * geometry.visibility_scale, 70.0, 90.0)
            )
    data["currents"] = []
    # Training geometry is kept inside generous bounds; the referee implementation
    # and all gate-validation rules are copied unchanged from the base track.
    margin = 8.0
    xs = [-4.0, 0.0, gate2_x]
    ys = [geometry.initial_lateral_offset_m, 0.0, gate2_y]
    zs = [-4.0, gate2_z]
    bounds = data["world"]["bounds"]
    bounds.update(
        {
            "x_min": min(xs) - margin,
            "x_max": max(xs) + margin,
            "y_min": min(ys) - margin,
            "y_max": max(ys) + margin,
            "z_min": min(zs) - 3.0,
            "z_max": max(zs) + 3.0,
        }
    )
    data["training_geometry"] = asdict(geometry)
    target = Path(output_path)
    _atomic_json(target, data)
    return target


class CurriculumSampler:
    """Deterministic replay mixture with serializable RNG and promotion state."""

    def __init__(
        self,
        *,
        seed: int,
        initial_stage: str = "C0",
        maximum_stage: str = "C4",
        mixture: Sequence[float] = (0.20, 0.20, 0.30, 0.20, 0.10),
        early_mixture: Optional[Sequence[float]] = None,
        early_until_timesteps: int = 0,
        promotion_config: Optional[Any] = None,
        sensor_noise: bool = True,
    ) -> None:
        if initial_stage not in STAGE_BY_KEY or maximum_stage not in STAGE_BY_KEY:
            raise ValueError("unknown curriculum stage")
        if CURRICULUM_STAGES.index(maximum_stage) < CURRICULUM_STAGES.index(
            initial_stage
        ):
            raise ValueError("maximum stage precedes initial stage")
        if len(mixture) != 5 or abs(sum(mixture) - 1.0) > 1e-9:
            raise ValueError("mixture must contain five weights summing to one")
        if early_mixture is None:
            early_mixture = mixture
        if len(early_mixture) != 5 or abs(sum(early_mixture) - 1.0) > 1e-9:
            raise ValueError("early_mixture must contain five weights summing to one")
        self.seed = int(seed)
        self.maximum_stage = maximum_stage
        self.mixture = tuple(float(v) for v in mixture)
        self.early_mixture = tuple(float(v) for v in early_mixture)
        self.early_until_timesteps = int(early_until_timesteps)
        self.promotion_config = promotion_config
        self.sensor_noise = bool(sensor_noise)
        self.rng = np.random.default_rng(self.seed)
        self.state = CurriculumState(current_stage=initial_stage)

    @property
    def current_stage(self) -> str:
        return self.state.current_stage

    def state_dict(self) -> Dict[str, Any]:
        state = asdict(self.state)
        state["rng_state"] = self.rng.bit_generator.state
        return {
            "seed": self.seed,
            "maximum_stage": self.maximum_stage,
            "mixture": list(self.mixture),
            "early_mixture": list(self.early_mixture),
            "early_until_timesteps": self.early_until_timesteps,
            "sensor_noise": self.sensor_noise,
            "state": state,
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if int(value["seed"]) != self.seed:
            raise ValueError("curriculum seed mismatch")
        if str(value["maximum_stage"]) != self.maximum_stage:
            raise ValueError("curriculum maximum-stage mismatch")
        stored_early = tuple(value.get("early_mixture", value["mixture"]))
        if stored_early != self.early_mixture:
            raise ValueError("curriculum early-mixture mismatch")
        if int(value.get("early_until_timesteps", 0)) != self.early_until_timesteps:
            raise ValueError("curriculum early replay boundary mismatch")
        state = dict(value["state"])
        rng_state = state.pop("rng_state", None)
        self.state = CurriculumState(**state)
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

    def add_failure_case(self, geometry: TransitionGeometry, reason: str) -> None:
        row = {"geometry": asdict(geometry), "reason": str(reason)}
        self.state.targeted_failures.append(row)
        self.state.targeted_failures = self.state.targeted_failures[-100:]

    @property
    def active_mixture(self) -> Tuple[float, ...]:
        if self.state.training_timesteps < self.early_until_timesteps:
            return self.early_mixture
        return self.mixture

    def set_timesteps(self, timesteps: int) -> None:
        self.state.training_timesteps = max(
            self.state.training_timesteps, int(timesteps)
        )

    def force_stage(self, stage: str, *, timesteps: int, reason: str) -> Dict[str, Any]:
        if stage not in CURRICULUM_STAGES:
            raise ValueError(f"unknown forced curriculum stage {stage!r}")
        old = self.current_stage
        self.state.current_stage = stage
        self.state.stage_entry_timesteps = int(timesteps)
        self.state.consecutive_full_passes = 0
        change = {
            "timesteps": int(timesteps),
            "from": old,
            "to": stage,
            "reason": str(reason),
        }
        self.state.stage_changes.append(change)
        return change

    def _choose_source(self) -> str:
        sources = (
            "retention",
            "straight",
            "current_stage",
            "previous_stage",
            "targeted_failure",
        )
        source = str(self.rng.choice(sources, p=self.active_mixture))
        if source == "targeted_failure" and not self.state.targeted_failures:
            source = "current_stage"
        return source

    def sample(self) -> TransitionGeometry:
        source = self._choose_source()
        current_index = CURRICULUM_STAGES.index(self.current_stage)
        if source == "retention":
            stage_key = "C0"
            max_angle = 3.0
        elif source == "straight":
            stage_key = "C0"
            max_angle = 0.0
        elif source == "previous_stage":
            stage_key = CURRICULUM_STAGES[max(0, current_index - 1)]
            max_angle = STAGE_BY_KEY[stage_key].max_abs_turn_deg
        elif source == "targeted_failure":
            chosen = self.state.targeted_failures[
                int(self.rng.integers(0, len(self.state.targeted_failures)))
            ]
            geometry = TransitionGeometry(**chosen["geometry"])
            jittered = TransitionGeometry(
                **{
                    **asdict(geometry),
                    "signed_turn_deg": float(
                        np.clip(
                            geometry.signed_turn_deg
                            + self.rng.uniform(-2.0, 2.0),
                            -45.0,
                            45.0,
                        )
                    ),
                    "initial_lateral_offset_m": float(
                        geometry.initial_lateral_offset_m
                        + self.rng.uniform(-0.15, 0.15)
                    ),
                    "source": source,
                    "stage": self.current_stage,
                }
            )
            self._record_sample(jittered)
            return jittered
        else:
            stage_key = self.current_stage
            max_angle = STAGE_BY_KEY[stage_key].max_abs_turn_deg

        limits = STAGE_BY_KEY[stage_key]
        turn = 0.0 if max_angle == 0 else float(self.rng.uniform(-max_angle, max_angle))
        # Once real turns are introduced, prevent a transient success/failure
        # sequence from creating a persistent left/right replay imbalance.
        if max_angle > 5.0 and current_index >= 2 and abs(turn) > 5.0:
            left_count = self.state.sample_counts.get("left", 0)
            right_count = self.state.sample_counts.get("right", 0)
            magnitude = abs(turn)
            if left_count > right_count + 1:
                turn = -magnitude
            elif right_count > left_count + 1:
                turn = magnitude
        noise_scale = 1.0 if self.sensor_noise else 0.0
        geometry = TransitionGeometry(
            signed_turn_deg=turn,
            gate_separation_m=float(
                self.rng.uniform(limits.min_separation_m, limits.max_separation_m)
            ),
            lateral_displacement_m=float(
                self.rng.uniform(
                    -limits.max_lateral_offset_m, limits.max_lateral_offset_m
                )
            ),
            vertical_displacement_m=float(
                self.rng.uniform(-limits.max_vertical_step_m, limits.max_vertical_step_m)
            ),
            starting_yaw_error_deg=float(
                self.rng.uniform(
                    -limits.max_start_yaw_error_deg,
                    limits.max_start_yaw_error_deg,
                )
            ),
            initial_lateral_offset_m=float(
                self.rng.uniform(
                    -limits.max_start_lateral_offset_m,
                    limits.max_start_lateral_offset_m,
                )
            ),
            beacon_bearing_noise_std_deg=float(self.rng.uniform(0.0, 1.0))
            * noise_scale,
            beacon_range_noise_std_m=float(self.rng.uniform(0.0, 0.08))
            * noise_scale,
            beacon_dropout_probability=float(self.rng.uniform(0.0, 0.02))
            * noise_scale,
            visibility_scale=float(self.rng.uniform(0.85, 1.0)),
            source=source,
            stage=self.current_stage,
        )
        self._record_sample(geometry)
        return geometry

    def _record_sample(self, geometry: TransitionGeometry) -> None:
        for key in (geometry.source, geometry.direction, geometry.stage):
            self.state.sample_counts[key] = self.state.sample_counts.get(key, 0) + 1

    def record_evaluation(
        self,
        metrics: Mapping[str, Any],
        *,
        timesteps: int,
        auto_promote: bool = True,
        allow_demotion: bool = True,
    ) -> Optional[Dict[str, Any]]:
        row = {"timesteps": int(timesteps), "stage": self.current_stage, **dict(metrics)}
        self.state.evaluation_history.append(row)
        self.set_timesteps(timesteps)
        if self.promotion_config is not None:
            passed = reliability_requirements_met(metrics, self.promotion_config)
            self.state.consecutive_full_passes = (
                self.state.consecutive_full_passes + 1 if passed else 0
            )
            self.state.reliable_full_streak = (
                self.state.reliable_full_streak + 1 if passed else 0
            )
            required = int(
                self.promotion_config.required_consecutive_full_evaluations
            )
            self.state.efficiency_phase_active = (
                self.state.reliable_full_streak >= required
            )
        decision = curriculum_stage_decision(
            self.current_stage,
            metrics,
            maximum_stage=self.maximum_stage,
            allow_demotion=allow_demotion,
            promotion_config=self.promotion_config,
            consecutive_full_passes=self.state.consecutive_full_passes,
            stage_timesteps=int(timesteps) - self.state.stage_entry_timesteps,
        )
        if decision and auto_promote:
            old = self.current_stage
            self.state.current_stage = decision
            self.state.stage_entry_timesteps = int(timesteps)
            self.state.consecutive_full_passes = 0
            change = {
                "timesteps": int(timesteps),
                "from": old,
                "to": decision,
                "metrics": dict(metrics),
            }
            self.state.stage_changes.append(change)
            return change
        return None


def curriculum_stage_decision(
    current_stage: str,
    metrics: Mapping[str, Any],
    *,
    maximum_stage: str = "C7",
    allow_demotion: bool = True,
    promotion_config: Optional[Any] = None,
    consecutive_full_passes: int = 1,
    stage_timesteps: int = 0,
) -> Optional[str]:
    """Return the next stage only when the documented multi-episode gate passes."""
    current_index = CURRICULUM_STAGES.index(current_stage)
    maximum_index = CURRICULUM_STAGES.index(maximum_stage)
    straight = float(metrics.get("straight_completion_rate", 1.0))
    total = int(metrics.get("completions", 0))
    n_eval = max(1, int(metrics.get("n_eval", 0)))
    left_successes = int(metrics.get("left_successes", 0))
    left_n = max(1, int(metrics.get("left_n", 0)))
    right_successes = int(metrics.get("right_successes", 0))
    right_n = max(1, int(metrics.get("right_n", 0)))
    previous_returns = int(metrics.get("previous_gate_returns", 0))
    safety = int(metrics.get("safety_events", 0))
    three_gate = float(metrics.get("three_gate_completion_rate", 0.0))
    six_gate = float(metrics.get("six_gate_completion_rate", 0.0))

    if allow_demotion and current_index > 0 and straight < 0.7:
        return CURRICULUM_STAGES[current_index - 1]
    if current_index >= maximum_index:
        return None
    if promotion_config is not None:
        required = int(promotion_config.required_consecutive_full_evaluations)
        minimum = int(
            promotion_config.minimum_stage_timesteps.get(current_stage, 0)
        )
        if (
            consecutive_full_passes >= required
            and stage_timesteps >= minimum
            and reliability_requirements_met(metrics, promotion_config)
        ):
            return CURRICULUM_STAGES[current_index + 1]
        return None
    promote = False
    if current_stage == "C0":
        promote = total / n_eval >= 0.8 and straight >= 0.8
    elif current_stage == "C1":
        promote = (
            total >= 16
            and n_eval >= 20
            and left_successes >= 7
            and left_n >= 10
            and right_successes >= 7
            and right_n >= 10
            and previous_returns == 0
        )
    elif current_stage == "C2":
        promote = (
            total >= 16
            and n_eval >= 20
            and left_successes >= 8
            and left_n >= 10
            and right_successes >= 8
            and right_n >= 10
            and safety == 0
        )
    elif current_stage == "C3":
        promote = (
            total >= 16
            and n_eval >= 20
            and min(left_successes / left_n, right_successes / right_n) >= 0.8
            and straight >= 0.9
        )
    elif current_stage == "C4":
        promote = total / n_eval >= 0.8 and straight >= 0.9
    elif current_stage == "C5":
        promote = three_gate >= 0.8
    elif current_stage == "C6":
        promote = six_gate >= 0.8
    if promote:
        return CURRICULUM_STAGES[current_index + 1]
    return None


def reliability_requirements_met(
    metrics: Mapping[str, Any], promotion_config: Any
) -> bool:
    """Return whether one full suite satisfies the reliability-first gate."""
    if str(metrics.get("mode", "full")) != "full":
        return False
    return (
        float(metrics.get("completion_rate", 0.0))
        >= float(promotion_config.overall_completion_rate)
        and float(metrics.get("single_gate_completion_rate", 0.0))
        >= float(promotion_config.single_gate_completion_rate)
        and float(metrics.get("straight_completion_rate", 0.0))
        >= float(promotion_config.straight_completion_rate)
        and float(metrics.get("left_completion_rate", 0.0))
        >= float(promotion_config.left_completion_rate)
        and float(metrics.get("right_completion_rate", 0.0))
        >= float(promotion_config.right_completion_rate)
        and int(metrics.get("episodes_with_collision", 0))
        <= int(promotion_config.maximum_collision_episodes)
        and int(metrics.get("episodes_with_out_of_bounds", 0))
        <= int(promotion_config.maximum_out_of_bounds_episodes)
        and int(metrics.get("episodes_with_wrong_direction", 0))
        <= int(promotion_config.maximum_wrong_direction_episodes)
        and int(metrics.get("previous_gate_returns", 0))
        <= int(promotion_config.maximum_previous_gate_returns)
    )


def fixed_evaluation_tracks(stage_key: str) -> Dict[str, str]:
    """Static retention/sequence tracks used alongside generated turn tracks."""
    if stage_key == "C0":
        return {"retention": SINGLE_GATE, "straight": TWO_GATE_STRAIGHT}
    tracks = {"retention": SINGLE_GATE, "straight": TWO_GATE_STRAIGHT}
    if CURRICULUM_STAGES.index(stage_key) >= CURRICULUM_STAGES.index("C5"):
        tracks["three_gate"] = THREE_GATE_S
    if CURRICULUM_STAGES.index(stage_key) >= CURRICULUM_STAGES.index("C6"):
        tracks["six_gate"] = SIX_GATE
    return tracks
