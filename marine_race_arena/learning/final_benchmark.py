"""Final common benchmark: one identical test suite for every deployable controller.

Every controller (PPO checkpoints, BC-v3, the deterministic rule baseline and the
hybrid baseline) is run through *exactly* the same test groups, the same generated
and official track geometries, the same seeds, the same adapter and the same
current-free simulator configuration.  Nothing about the referee, the official
tracks, the gate order or the scoring is modified: circuits that declare currents
are run current-free through the documented runtime override only.

Learned controllers stay entirely policy-driven: no runtime rule action, no expert
correction, no fallback control and no hybrid blending.  ``_assert_policy_only``
verifies this per episode from the controller's own declared contract *and* from
its runtime intervention counter, and the episode row records the evidence.

The suite is deterministic: given the same suite parameters, ``build_suite``
produces the same cases, the same generated geometries and the same seed lists, so
the whole benchmark reproduces from one command.  Episodes are written
incrementally to a JSONL shard, so a crashed or interrupted benchmark resumes
without re-running finished episodes.

Usage (marine_race_rl env)::

    python -m marine_race_arena.learning.final_benchmark run \
        --out results/rl/final_benchmark/<name> --workers 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.config import ACTION_AXES, ACTION_CONTRACT_VERSION
from marine_race_arena.learning.multigate_curriculum import (
    HORSESHOE,
    MIXED,
    SINGLE_GATE,
    THREE_GATE_S,
    THREE_GATE_TRAINING,
    TWO_GATE_LEFT,
    TWO_GATE_RIGHT,
    TWO_GATE_STRAIGHT,
    VERTICAL,
)
from marine_race_arena.learning.parametric_curriculum import (
    TransitionGeometry,
    generate_two_gate_track,
)
from marine_race_arena.learning.provenance import git_sha, now_utc, sha256_file
from marine_race_arena.learning.seed_registry import (
    MULTIGATE_FINAL_BENCHMARK_HOLDOUT_SEEDS,
    MULTIGATE_V3_FINAL_THREE_GATE_SEEDS,
    MULTIGATE_V3_FINAL_TWO_GATE_SEEDS,
    RESERVED_FINAL_MULTIGATE_SEEDS,
)

SUITE_SCHEMA_VERSION = "final_benchmark_suite_v1"
EPISODE_SCHEMA_VERSION = "final_benchmark_episode_v1"

RUN_ROOT = "results/rl/multigate_reliability_first/r2_reliability_first_seed23001_c3_recovery_476525"
BC_V3_MODEL = "results/rl/multigate_longrun/bc_v3_balanced_v2_20260728/bc_v3.pt"
HYBRID_MODEL = "results/rl_public/stage1/bc/model/best_model.pt"


# --------------------------------------------------------------------------- #
# Controllers under test
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ControllerSpec:
    """One controller under test.

    ``policy_only`` marks controllers whose every runtime command must come from
    the learned policy.  ``separate_reporting`` marks controllers that are not
    fully policy-based and therefore must not be mixed into the learned-controller
    ranking (the hybrid baseline).
    """

    key: str
    label: str
    controller: str
    model: Optional[str]
    kind: str
    policy_only: bool = False
    separate_reporting: bool = False
    note: str = ""


def default_controllers() -> Tuple[ControllerSpec, ...]:
    checkpoints = Path(RUN_ROOT) / "checkpoints"
    return (
        ControllerSpec(
            "ppo_525678",
            "PPO 525,678 steps (training best_reliable / best_fast_reliable)",
            "rl_multigate_controller",
            str(checkpoints / "ppo_525678_steps.zip"),
            "ppo",
            policy_only=True,
            note="training-metadata best_reliable and best_fast_reliable alias",
        ),
        ControllerSpec(
            "ppo_900462",
            "PPO 900,462 steps (latest known safe checkpoint during C4)",
            "rl_multigate_controller",
            str(checkpoints / "ppo_900462_steps.zip"),
            "ppo",
            policy_only=True,
            note="latest safe checkpoint recorded while the run was in stage C4",
        ),
        ControllerSpec(
            "ppo_1000814",
            "PPO 1,000,814 steps (final safe checkpoint after rollback to C3)",
            "rl_multigate_controller",
            str(checkpoints / "ppo_1000814_steps.zip"),
            "ppo",
            policy_only=True,
            note="final checkpoint of the completed reliability-first run",
        ),
        ControllerSpec(
            "bc_v3",
            "BC-v3 balanced warm start (learned baseline)",
            "rl_multigate_controller",
            BC_V3_MODEL,
            "bc",
            policy_only=True,
            note="behaviour-cloning reference the PPO retention term regularises towards",
        ),
        ControllerSpec(
            "rule_gate_center_then_commit",
            "Deterministic rule baseline (gate-center then commit)",
            "rule_gate_center_then_commit",
            None,
            "rule",
            note="no learned component",
        ),
        ControllerSpec(
            "hybrid",
            "Hybrid rule backbone + BC-v1 visual servo (NOT fully policy-based)",
            "hybrid_gate_controller",
            HYBRID_MODEL,
            "hybrid",
            separate_reporting=True,
            note="reported separately: deterministic backbone blended with a learned servo",
        ),
    )


CONTROLLERS_BY_KEY: Dict[str, ControllerSpec] = {
    spec.key: spec for spec in default_controllers()
}


# --------------------------------------------------------------------------- #
# Test groups
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CaseTemplate:
    """One geometry inside a test group.

    ``geometry`` is non-``None`` for parametrically generated two-gate transitions;
    those tracks are written once into the benchmark output directory so every
    controller sees byte-identical geometry.
    """

    case_id: str
    track: Optional[str] = None
    geometry: Optional[TransitionGeometry] = None
    benchmark_task: Optional[str] = None


@dataclass(frozen=True)
class GroupTemplate:
    group: str
    description: str
    cases: Tuple[CaseTemplate, ...]
    seed_pool: str
    episodes_per_case: int
    official: bool = False


def _vertical_geometry(stage: str, signed_turn: float, vertical: float, index: int) -> TransitionGeometry:
    """Two-gate transition with a deliberate depth change between the gates.

    Positive ``vertical`` raises the second gate (low-to-high), negative lowers it
    (high-to-low); z is negative-down in the track frame.
    """
    return TransitionGeometry(
        signed_turn_deg=float(signed_turn),
        gate_separation_m=5.0 + 0.5 * (index % 3),
        lateral_displacement_m=0.0,
        vertical_displacement_m=float(vertical),
        starting_yaw_error_deg=-4.0 + 4.0 * (index % 3),
        initial_lateral_offset_m=-0.3 + 0.3 * (index % 3),
        source="final_benchmark",
        stage=stage,
    )


def default_groups(*, episodes_per_case: int, official_episodes: int) -> Tuple[GroupTemplate, ...]:
    half = max(1, episodes_per_case // 2)
    return (
        GroupTemplate(
            "single_gate_retention",
            "Single-gate retention: the competence every later stage must not lose.",
            (CaseTemplate("single_gate", track=SINGLE_GATE),),
            "two_gate",
            episodes_per_case,
        ),
        GroupTemplate(
            "two_gate_straight",
            "Two aligned gates: cross, clear and reacquire without turning.",
            (CaseTemplate("straight", track=TWO_GATE_STRAIGHT),),
            "two_gate",
            episodes_per_case,
        ),
        GroupTemplate(
            "two_gate_left",
            "Two-gate left transitions at two turn magnitudes.",
            (
                CaseTemplate("left_37deg_fixed", track=TWO_GATE_LEFT),
                CaseTemplate(
                    "left_20deg_generated",
                    geometry=_vertical_geometry("C3", 20.0, 0.0, 1),
                ),
            ),
            "two_gate",
            half,
        ),
        GroupTemplate(
            "two_gate_right",
            "Two-gate right transitions at two turn magnitudes.",
            (
                CaseTemplate("right_37deg_fixed", track=TWO_GATE_RIGHT),
                CaseTemplate(
                    "right_20deg_generated",
                    geometry=_vertical_geometry("C3", -20.0, 0.0, 2),
                ),
            ),
            "two_gate",
            half,
        ),
        GroupTemplate(
            "vertical_low_to_high",
            "Ascending gate transition (second gate shallower than the first).",
            (
                CaseTemplate(
                    "up_0p7m_straight",
                    geometry=_vertical_geometry("C3", 0.0, 0.7, 0),
                ),
                CaseTemplate(
                    "up_1p2m_left15",
                    geometry=_vertical_geometry("C4", 15.0, 1.2, 1),
                ),
            ),
            "two_gate",
            half,
        ),
        GroupTemplate(
            "vertical_high_to_low",
            "Descending gate transition (second gate deeper than the first).",
            (
                CaseTemplate(
                    "down_0p7m_straight",
                    geometry=_vertical_geometry("C3", 0.0, -0.7, 0),
                ),
                CaseTemplate(
                    "down_1p2m_right15",
                    geometry=_vertical_geometry("C4", -15.0, -1.2, 1),
                ),
            ),
            "two_gate",
            half,
        ),
        GroupTemplate(
            "three_gate_sequence",
            "Three-gate sequence with two consecutive reacquisitions.",
            (CaseTemplate("three_gate_training", track=THREE_GATE_TRAINING),),
            "three_gate",
            episodes_per_case,
        ),
        GroupTemplate(
            "three_gate_s_shape",
            "S-shaped three-gate sequence: a left transition immediately followed by a right one.",
            (CaseTemplate("three_gate_s_curve", track=THREE_GATE_S),),
            "three_gate",
            episodes_per_case,
        ),
        GroupTemplate(
            "official_horseshoe_bay",
            "Official Horseshoe Bay circuit, current-free (12 gates).",
            (CaseTemplate("horseshoe_bay", track=HORSESHOE),),
            "official",
            official_episodes,
            official=True,
        ),
        GroupTemplate(
            "official_vertical_serpent",
            "Official Vertical Serpent circuit, current-free (17 gates).",
            (CaseTemplate("vertical_serpent", track=VERTICAL),),
            "official",
            official_episodes,
            official=True,
        ),
        GroupTemplate(
            "official_mixed_endurance",
            "Official Mixed Endurance circuit, current-free (22 gates, clean_gate override).",
            (
                CaseTemplate(
                    "mixed_endurance",
                    track=MIXED,
                    benchmark_task="clean_gate",
                ),
            ),
            "official",
            official_episodes,
            official=True,
        ),
    )


_SEED_POOLS: Dict[str, Sequence[int]] = {
    "two_gate": MULTIGATE_V3_FINAL_TWO_GATE_SEEDS,
    "three_gate": MULTIGATE_V3_FINAL_THREE_GATE_SEEDS,
    "official": RESERVED_FINAL_MULTIGATE_SEEDS,
}

# Pools whose reused seeds are the *same* for every case, so the official-circuit
# results stay directly comparable with the already-published current-free runs
# (which used seeds 1800-1804 on each of the three circuits).
_SHARED_SEED_POOLS = frozenset({"official"})


@dataclass(frozen=True)
class BenchmarkCase:
    group: str
    case_id: str
    track: str
    seeds: Tuple[int, ...]
    reused_seeds: Tuple[int, ...]
    holdout_seeds: Tuple[int, ...]
    official: bool
    benchmark_task: Optional[str]
    geometry: Optional[Dict[str, Any]]
    expected_gates: int
    max_duration_s: float

    @property
    def uid(self) -> str:
        return f"{self.group}/{self.case_id}"


def _materialize_track(geometry: TransitionGeometry, track_path: Path) -> Path:
    """Write a generated track once, safely under concurrent shards.

    Every shard rebuilds the suite so it can resolve its own cases, and on Windows
    two shards renaming the same temporary file over the same target raises
    ``PermissionError``. The track is content-addressed instead: generate into a
    process-unique file, keep the existing target when it already matches, and
    only then move the new content into place.
    """
    staging = track_path.with_name(f"{track_path.stem}.{os.getpid()}.staging.json")
    generate_two_gate_track(geometry, staging)
    desired = staging.read_text(encoding="utf-8")
    try:
        if track_path.exists() and track_path.read_text(encoding="utf-8") == desired:
            return track_path
        for attempt in range(5):
            try:
                os.replace(staging, track_path)
                return track_path
            except PermissionError:
                # Another shard is publishing the identical file; re-check and wait.
                time.sleep(0.2 * (attempt + 1))
                if (
                    track_path.exists()
                    and track_path.read_text(encoding="utf-8") == desired
                ):
                    return track_path
        raise RuntimeError(f"could not publish generated track {track_path}")
    finally:
        if staging.exists():
            staging.unlink(missing_ok=True)


def _track_facts(track: str, benchmark_task: Optional[str]) -> Tuple[int, float]:
    """Expected gates and race deadline, read from the resolved track config."""
    from marine_race_arena.config.loader import load_track_config

    config = load_track_config(
        track, benchmark_task=benchmark_task, current_profile="none"
    )
    expected = len(config.track.gate_sequence) * int(config.race.laps)
    return int(expected), float(config.race.max_duration_s)


def build_suite(
    output_dir: str | Path,
    *,
    episodes_per_case: int = 8,
    official_episodes: int = 5,
    groups: Optional[Sequence[str]] = None,
) -> List[BenchmarkCase]:
    """Deterministically build the benchmark cases and their generated tracks.

    Half of every case's seeds are reused from the already-allocated final
    evaluation ranges (so results are comparable with earlier official runs) and
    half come from the never-used final-benchmark holdout range.
    """
    out = Path(output_dir)
    track_dir = out / "generated_tracks"
    track_dir.mkdir(parents=True, exist_ok=True)
    templates = default_groups(
        episodes_per_case=episodes_per_case, official_episodes=official_episodes
    )
    if groups:
        wanted = set(groups)
        templates = tuple(t for t in templates if t.group in wanted)
        unknown = wanted - {t.group for t in default_groups(
            episodes_per_case=episodes_per_case, official_episodes=official_episodes
        )}
        if unknown:
            raise ValueError(f"unknown benchmark groups: {sorted(unknown)}")

    cases: List[BenchmarkCase] = []
    pool_cursor: Dict[str, int] = {name: 0 for name in _SEED_POOLS}
    holdout_cursor = 0
    for template in templates:
        for case in template.cases:
            n = int(template.episodes_per_case)
            n_reused = n // 2 if n > 1 else 1
            n_holdout = n - n_reused
            pool = _SEED_POOLS[template.seed_pool]
            shared = template.seed_pool in _SHARED_SEED_POOLS
            start = 0 if shared else pool_cursor[template.seed_pool]
            if start + n_reused > len(pool):
                raise ValueError(
                    f"seed pool {template.seed_pool!r} exhausted for {template.group}"
                )
            reused = tuple(int(s) for s in pool[start : start + n_reused])
            if not shared:
                pool_cursor[template.seed_pool] = start + n_reused
            holdout = tuple(
                int(s)
                for s in MULTIGATE_FINAL_BENCHMARK_HOLDOUT_SEEDS[
                    holdout_cursor : holdout_cursor + n_holdout
                ]
            )
            holdout_cursor += n_holdout

            if case.geometry is not None:
                track_path = track_dir / f"{template.group}__{case.case_id}.json"
                track = str(_materialize_track(case.geometry, track_path).as_posix())
                geometry = asdict(case.geometry)
            else:
                track = str(case.track)
                geometry = None
            expected_gates, max_duration = _track_facts(track, case.benchmark_task)
            cases.append(
                BenchmarkCase(
                    group=template.group,
                    case_id=case.case_id,
                    track=track,
                    seeds=tuple(sorted(reused + holdout)),
                    reused_seeds=reused,
                    holdout_seeds=holdout,
                    official=template.official,
                    benchmark_task=case.benchmark_task,
                    geometry=geometry,
                    expected_gates=expected_gates,
                    max_duration_s=max_duration,
                )
            )
    return cases


def group_descriptions() -> Dict[str, str]:
    return {t.group: t.description for t in default_groups(episodes_per_case=2, official_episodes=2)}


# --------------------------------------------------------------------------- #
# Instrumented episode
# --------------------------------------------------------------------------- #
class _EpisodeProbe:
    """Wrap a controller's ``step`` to record per-frame evidence, unchanged behavior.

    The probe never touches the command: it times inference, stores the produced
    action, and samples the adapter state, the referee counters, the collision
    sensor and (optionally) the onboard camera *after* the controller has already
    decided.  Nothing it reads is fed back to the controller.
    """

    def __init__(
        self,
        controller: Any,
        *,
        capture_video: bool = False,
        video_stride: int = 5,
        trajectory_stride: int = 1,
    ) -> None:
        self._controller = controller
        self._original = controller.step
        self._capture_video = bool(capture_video)
        self._video_stride = max(1, int(video_stride))
        self._trajectory_stride = max(1, int(trajectory_stride))
        self._ctx = None
        self._pid: Optional[str] = None

        self.inference_s = 0.0
        self.steps = 0
        self.actions: List[np.ndarray] = []
        self.trajectory: List[Dict[str, Any]] = []
        self.frames: List[np.ndarray] = []
        self.collision_frames = 0
        self.out_of_bounds_frames = 0
        self.path_length_m = 0.0
        self._last_position: Optional[Tuple[float, float, float]] = None
        self._bounds: Optional[Tuple[float, float, float, float, float, float]] = None

        def probed_step(observation):
            started = time.perf_counter()
            command = self._original(observation)
            self.inference_s += time.perf_counter() - started
            self.steps += 1
            self._record(observation, command)
            return command

        controller.step = probed_step  # type: ignore[method-assign]

    def bind(self, ctx: Any, participant_id: str) -> None:
        self._ctx = ctx
        self._pid = participant_id
        bounds = ctx.config.world.bounds
        self._bounds = (
            float(bounds.x_min),
            float(bounds.x_max),
            float(bounds.y_min),
            float(bounds.y_max),
            float(bounds.z_min),
            float(bounds.z_max),
        )

    # -- recording ---------------------------------------------------------- #
    def _record(self, observation: Mapping[str, Any], command: Any) -> None:
        action = np.asarray(
            [
                float(command.get(axis, 0.0)) if isinstance(command, Mapping) else 0.0
                for axis in ACTION_AXES
            ],
            dtype=np.float32,
        )
        self.actions.append(action)
        sensors = observation.get("sensors", {}) if isinstance(observation, Mapping) else {}
        if _collision_sensor_active(sensors):
            self.collision_frames += 1
        if self._capture_video and (self.steps - 1) % self._video_stride == 0:
            frame = _camera_frame(sensors)
            if frame is not None:
                self.frames.append(frame)
        if self._ctx is None or self._pid is None:
            return
        try:
            state = self._ctx.adapter.get_participant_state(self._pid)
            referee_state = self._ctx.referee.states[self._pid]
        except Exception:  # pragma: no cover - defensive sampling path
            return
        position = tuple(float(v) for v in state.position)
        if self._last_position is not None:
            self.path_length_m += float(
                math.dist(position, self._last_position)
            )
        self._last_position = position
        if self._bounds is not None and _outside(position, self._bounds):
            self.out_of_bounds_frames += 1
        if (self.steps - 1) % self._trajectory_stride == 0:
            self.trajectory.append(
                {
                    "step": int(self.steps),
                    "time_s": float(observation.get("local_time_s", 0.0))
                    if isinstance(observation, Mapping)
                    else 0.0,
                    "position": list(position),
                    "yaw_deg": float(state.rotation_rpy_deg[2]),
                    "gates": int(referee_state.valid_gate_crossings),
                    "action": [round(float(v), 5) for v in action],
                }
            )

    # -- derived metrics ---------------------------------------------------- #
    def action_metrics(self) -> Dict[str, float]:
        if not self.actions:
            return {
                "mean_abs_sway_action": 0.0,
                "mean_abs_yaw_action": 0.0,
                "mean_action_jerk": 0.0,
                "action_oscillation_rate": 0.0,
                "action_saturation": 0.0,
                "actions_finite": True,
            }
        actions = np.stack(self.actions)
        changes = np.diff(actions, axis=0)
        oscillations = 0
        opportunities = 0
        for axis in (1, 3):
            values = actions[:, axis]
            if len(values) > 1:
                active = (np.abs(values[1:]) > 0.05) & (np.abs(values[:-1]) > 0.05)
                oscillations += int(
                    np.sum(active & (np.sign(values[1:]) != np.sign(values[:-1])))
                )
                opportunities += int(np.sum(active))
        return {
            "mean_abs_sway_action": float(np.mean(np.abs(actions[:, 1]))),
            "mean_abs_yaw_action": float(np.mean(np.abs(actions[:, 3]))),
            "mean_action_jerk": (
                float(np.mean(np.linalg.norm(changes, axis=1))) if len(changes) else 0.0
            ),
            "action_oscillation_rate": (
                float(oscillations / opportunities) if opportunities else 0.0
            ),
            "action_saturation": float(np.mean(np.abs(actions) > 0.98)),
            "actions_finite": bool(np.all(np.isfinite(actions))),
        }

    def mean_inference_ms(self) -> Optional[float]:
        if self.steps == 0:
            return None
        return round(1000.0 * self.inference_s / self.steps, 4)


def _collision_sensor_active(sensors: Mapping[str, Any]) -> bool:
    value = sensors.get("CollisionSensor") if isinstance(sensors, Mapping) else None
    if value is None:
        return False
    try:
        return bool(np.any(np.asarray(value) != 0))
    except Exception:  # pragma: no cover - unusual sensor payloads
        return bool(value)


def _camera_frame(sensors: Mapping[str, Any]):
    value = sensors.get("FrontCamera") if isinstance(sensors, Mapping) else None
    if value is None:
        return None
    frame = np.asarray(value)
    if frame.ndim != 3 or frame.shape[0] < 2:
        return None
    if frame.shape[2] >= 3:
        frame = frame[:, :, :3]
    return np.ascontiguousarray(frame[::2, ::2, ::-1].astype(np.uint8))


def _outside(position: Sequence[float], bounds: Sequence[float]) -> bool:
    x, y, z = position[0], position[1], position[2]
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    return not (
        x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max
    )


def _assert_policy_only(spec: ControllerSpec, controller: Any) -> Dict[str, Any]:
    """Prove a learned controller produced every action itself.

    Raises if the controller declares any rule weight, hybrid blending or a
    runtime rule-controller instance, or if it reports runtime interventions.
    """
    evidence = {
        "rule_action_weight": getattr(controller, "rule_action_weight", None),
        "hybrid_blending": getattr(controller, "hybrid_blending", None),
        "rule_controller_instantiated": getattr(
            controller, "rule_controller_instantiated", None
        ),
        "deterministic_runtime_intervention_count": getattr(
            controller, "deterministic_runtime_intervention_count", None
        ),
    }
    if not spec.policy_only:
        return evidence
    problems = []
    if float(evidence["rule_action_weight"] or 0.0) != 0.0:
        problems.append(f"rule_action_weight={evidence['rule_action_weight']}")
    if bool(evidence["hybrid_blending"]):
        problems.append("hybrid_blending=True")
    if bool(evidence["rule_controller_instantiated"]):
        problems.append("rule_controller_instantiated=True")
    if int(evidence["deterministic_runtime_intervention_count"] or 0) != 0:
        problems.append(
            "deterministic_runtime_intervention_count="
            f"{evidence['deterministic_runtime_intervention_count']}"
        )
    if problems:
        raise RuntimeError(
            f"controller {spec.key!r} is declared policy-only but reported: "
            + ", ".join(problems)
        )
    return evidence


def _gate_frames(ctx: Any) -> List[Dict[str, Any]]:
    gates = []
    for gate_id in ctx.config.track.gate_sequence:
        gate = ctx.arena.gate_map[gate_id]
        gates.append(
            {
                "gate_id": gate_id,
                "center": [float(v) for v in gate.center],
                "normal": [float(v) for v in gate.normal_vector],
            }
        )
    return gates


def _previous_gate_returns(
    trajectory: Sequence[Mapping[str, Any]], gates: Sequence[Mapping[str, Any]]
) -> int:
    """Controller-agnostic count of returns behind an already-passed gate.

    After the vehicle has cleared gate ``k`` by more than ``clearance``, dropping
    back behind that gate's plane counts one return event; it is re-armed only
    after clearing the gate again.
    """
    clearance = 1.0
    returns = 0
    armed: Dict[int, bool] = {}
    for point in trajectory:
        passed = int(point.get("gates", 0))
        if passed <= 0 or passed > len(gates):
            continue
        index = passed - 1
        gate = gates[index]
        centre = np.asarray(gate["center"], dtype=float)
        normal = np.asarray(gate["normal"], dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            continue
        normal = normal / norm
        signed = float(np.dot(np.asarray(point["position"], dtype=float) - centre, normal))
        if signed > clearance:
            armed[index] = True
        elif signed < 0.0 and armed.get(index):
            returns += 1
            armed[index] = False
    return returns


def run_benchmark_episode(
    spec: ControllerSpec,
    case: BenchmarkCase,
    seed: int,
    *,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
    dt: float = 0.1,
    current_profile: str = "none",
    capture_video: bool = False,
    video_stride: int = 5,
    trajectory_stride: int = 1,
    artifact_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run one instrumented episode through the unchanged runner and referee."""
    from marine_race_arena.learning.episode import build_single_vehicle_race
    from marine_race_arena.learning.evaluate_policy import derive_evaluation_end_reason
    from marine_race_arena.participants.controller_loader import ControllerLoader
    from marine_race_arena.scripts.run_marine_race import _mission_info, _run_race_loop

    controller = ControllerLoader().load(
        spec.controller,
        constructor_kwargs={"model_path": spec.model} if spec.model else None,
    )
    probe = _EpisodeProbe(
        controller,
        capture_video=capture_video,
        video_stride=video_stride,
        trajectory_stride=trajectory_stride,
    )
    wall_start = time.time()
    ctx = build_single_vehicle_race(
        case.track,
        seed=int(seed),
        adapter=adapter,
        allow_fallback=allow_fallback,
        official=True,
        current_profile=current_profile,
        benchmark_task=case.benchmark_task,
        controller=controller,
    )
    pid = ctx.participant.id
    probe.bind(ctx, pid)
    try:
        controller.reset(_mission_info(ctx.config, pid))
        _run_race_loop(
            config=ctx.config,
            arena=ctx.arena,
            referee=ctx.referee,
            adapter=ctx.adapter,
            participants={pid: ctx.participant},
            dt=dt,
        )
        state = ctx.referee.states[pid]
        status = state.status.value if hasattr(state.status, "value") else str(state.status)
        end_reason = derive_evaluation_end_reason(status, truncated_by_max_steps=False)
        gates = _gate_frames(ctx)
        policy_evidence = _assert_policy_only(spec, controller)
        summary = _referee_times(ctx.referee, pid)
        finished = status == "FINISHED"
        completed_gates = int(state.valid_gate_crossings)
        official_time = summary.get("official_time_s")
        row: Dict[str, Any] = {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "controller": spec.key,
            "controller_kind": spec.kind,
            "controller_alias": spec.controller,
            "model": spec.model,
            "policy_only": spec.policy_only,
            "separate_reporting": spec.separate_reporting,
            "group": case.group,
            "case_id": case.case_id,
            "case_uid": case.uid,
            "track": case.track,
            "official_circuit": case.official,
            "benchmark_task": case.benchmark_task,
            "seed": int(seed),
            "seed_role": "reused" if int(seed) in case.reused_seeds else "holdout",
            "referee_status": status,
            "evaluation_end_reason": end_reason,
            "finished": finished,
            "completed_gates": completed_gates,
            "expected_gates": int(case.expected_gates),
            "full_completion": bool(finished and completed_gates >= case.expected_gates),
            "official_time_s": official_time,
            "penalized_time_s": summary.get("penalized_time_s"),
            "penalties_s": float(state.penalties_s),
            "time_per_gate_s": (
                float(official_time) / completed_gates
                if official_time is not None and completed_gates > 0
                else None
            ),
            "collision_events": int(state.collision_events),
            "obstacle_collision_events": int(state.obstacle_collision_events),
            "collision_frames": int(probe.collision_frames),
            "collision_episode": bool(state.collision_events > 0),
            "out_of_bounds_events": int(state.out_of_bounds_events),
            "out_of_bounds_frames": int(probe.out_of_bounds_frames),
            "out_of_bounds_episode": bool(state.out_of_bounds_events > 0),
            "wrong_direction_crossings": int(state.wrong_direction_crossings),
            "wrong_direction_episode": bool(state.wrong_direction_crossings > 0),
            "missed_gate_attempts": int(state.missed_gate_attempts),
            "stuck_events": int(state.stuck_events),
            "any_safety_event": bool(
                state.collision_events
                or state.out_of_bounds_events
                or state.wrong_direction_crossings
            ),
            "previous_gate_returns": _previous_gate_returns(probe.trajectory, gates),
            "previous_gate_returns_controller": getattr(
                controller, "previous_gate_return_count", None
            ),
            "timeout": end_reason in {"TIME_LIMIT", "MAX_STEPS"},
            "path_length_m": round(float(probe.path_length_m), 4),
            "steps": int(probe.steps),
            "mean_inference_ms": probe.mean_inference_ms(),
            "wall_s": round(time.time() - wall_start, 3),
            "adapter_used": ctx.adapter.name,
            "dt": float(dt),
            "current_profile": current_profile,
            "max_duration_s": float(case.max_duration_s),
            "policy_evidence": policy_evidence,
            **probe.action_metrics(),
        }
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{spec.key}__{case.group}__{case.case_id}__seed{seed}"
            trajectory_path = artifact_dir / f"{stem}.trajectory.json"
            _atomic_write_json(
                trajectory_path,
                {
                    "schema_version": "final_benchmark_trajectory_v1",
                    "controller": spec.key,
                    "group": case.group,
                    "case_id": case.case_id,
                    "seed": int(seed),
                    "track": case.track,
                    "referee_status": status,
                    "finished": finished,
                    "completed_gates": completed_gates,
                    "expected_gates": int(case.expected_gates),
                    "gates": gates,
                    "bounds": list(probe._bounds or ()),
                    "trajectory": probe.trajectory,
                },
            )
            row["trajectory_path"] = str(trajectory_path.as_posix())
            if capture_video and probe.frames:
                video_path = artifact_dir / f"{stem}.mp4"
                written = _write_video(video_path, probe.frames, fps=10)
                if written is not None:
                    row["video_path"] = str(written.as_posix())
                    row["video_frames"] = len(probe.frames)
        return row
    finally:
        try:
            controller.close()
        except Exception:  # pragma: no cover - defensive close path
            pass
        ctx.adapter.close()


def _referee_times(referee: Any, pid: str) -> Dict[str, Optional[float]]:
    try:
        summary = referee.summary()
    except Exception:  # pragma: no cover
        return {"official_time_s": None, "penalized_time_s": None}
    for entry in (summary.get("participants") or []) if isinstance(summary, dict) else []:
        if isinstance(entry, dict) and entry.get("participant_id") == pid:
            return {
                "official_time_s": (
                    float(entry["official_time_s"])
                    if isinstance(entry.get("official_time_s"), (int, float))
                    else None
                ),
                "penalized_time_s": (
                    float(entry["penalized_time_s"])
                    if isinstance(entry.get("penalized_time_s"), (int, float))
                    else None
                ),
            }
    return {"official_time_s": None, "penalized_time_s": None}


def _write_video(path: Path, frames: Sequence[np.ndarray], *, fps: int = 10) -> Optional[Path]:
    try:
        import imageio.v2 as imageio
    except Exception:  # pragma: no cover - imageio is optional
        return None
    try:
        with imageio.get_writer(str(path), fps=fps, macro_block_size=None) as writer:
            for frame in frames:
                writer.append_data(frame)
    except Exception:  # pragma: no cover - encoder unavailable
        try:
            gif = path.with_suffix(".gif")
            imageio.mimsave(str(gif), list(frames), duration=1.0 / max(1, fps))
            return gif
        except Exception:
            return None
    return path


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Episode planning, sharding and execution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlannedEpisode:
    index: int
    controller: str
    group: str
    case_id: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.controller}|{self.group}|{self.case_id}|{self.seed}"


def plan_episodes(
    cases: Sequence[BenchmarkCase], controllers: Sequence[ControllerSpec]
) -> List[PlannedEpisode]:
    """Deterministic episode order: official circuits last, cheapest groups first."""
    episodes: List[PlannedEpisode] = []
    ordered = sorted(cases, key=lambda c: (c.official, c.group, c.case_id))
    index = 0
    for case in ordered:
        for seed in case.seeds:
            for spec in controllers:
                episodes.append(
                    PlannedEpisode(index, spec.key, case.group, case.case_id, int(seed))
                )
                index += 1
    return episodes


def _completed_keys(out_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    for path in sorted(Path(out_dir).glob("episodes.shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # pragma: no cover - torn final line
                continue
            key = (
                f"{row.get('controller')}|{row.get('group')}|"
                f"{row.get('case_id')}|{row.get('seed')}"
            )
            done[key] = row
    return done


def _suite_manifest(
    out_dir: Path,
    cases: Sequence[BenchmarkCase],
    controllers: Sequence[ControllerSpec],
    args: Any,
) -> Dict[str, Any]:
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "created_utc": now_utc(),
        "git_sha": git_sha(),
        "run_root": RUN_ROOT,
        "adapter": args.adapter,
        "allow_fallback": bool(args.allow_fallback),
        "current_profile": args.current_profile,
        "dt": float(args.dt),
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "episodes_per_case": int(args.episodes_per_case),
        "official_episodes": int(args.official_episodes),
        "group_descriptions": group_descriptions(),
        "controllers": [
            {
                **asdict(spec),
                "model_sha256": sha256_file(spec.model) if spec.model else None,
                "model_exists": bool(spec.model and Path(spec.model).exists()),
            }
            for spec in controllers
        ],
        "cases": [asdict(case) for case in cases],
        "all_seeds": sorted({int(s) for case in cases for s in case.seeds}),
        "reused_seeds": sorted({int(s) for case in cases for s in case.reused_seeds}),
        "holdout_seeds": sorted({int(s) for case in cases for s in case.holdout_seeds}),
        "total_episodes": sum(len(case.seeds) for case in cases) * len(controllers),
    }


def _select_controllers(keys: Optional[Sequence[str]]) -> List[ControllerSpec]:
    specs = list(default_controllers())
    if not keys:
        return specs
    wanted = list(keys)
    unknown = [k for k in wanted if k not in CONTROLLERS_BY_KEY]
    if unknown:
        raise ValueError(f"unknown controllers: {unknown}")
    return [CONTROLLERS_BY_KEY[k] for k in wanted]


def _video_seed(case: BenchmarkCase) -> Optional[int]:
    return int(case.seeds[0]) if case.official and case.seeds else None


def run_shard(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    cases = build_suite(
        out_dir,
        episodes_per_case=args.episodes_per_case,
        official_episodes=args.official_episodes,
        groups=args.groups,
    )
    case_by_uid = {case.uid: case for case in cases}
    controllers = _select_controllers(args.controllers)
    episodes = plan_episodes(cases, controllers)
    done = _completed_keys(out_dir)
    suffix = "r" if args.reverse else ""
    shard_path = out_dir / f"episodes.shard{args.shard:02d}{suffix}.jsonl"
    artifact_dir = out_dir / "artifacts"
    mine = [e for e in episodes if e.index % args.shards == args.shard]
    todo = [e for e in mine if e.key not in done]
    if args.reverse:
        # A helper drains the same queue from the tail. The expensive official
        # circuits are last, so a reverse helper starts on them immediately while
        # the forward shard is still working through the cheap groups; the two
        # only overlap once the queue is nearly empty, and the merge is keyed by
        # episode so a duplicate is harmless.
        todo = list(reversed(todo))
    print(
        f"[bench:{args.shard}] {len(todo)} episodes to run "
        f"({len(mine) - len(todo)} already complete)",
        flush=True,
    )
    for episode in todo:
        case = case_by_uid[f"{episode.group}/{episode.case_id}"]
        spec = CONTROLLERS_BY_KEY[episode.controller]
        capture_video = bool(
            args.video and case.official and episode.seed == _video_seed(case)
        )
        started = time.time()
        row = None
        exc: Optional[BaseException] = None
        # A simulator launch can time out when several instances start at once.
        # That is an infrastructure failure, not a controller failure, so retry
        # before recording anything: a recorded HARNESS_ERROR would otherwise be
        # indistinguishable from the policy losing the episode.
        for attempt in range(1, max(1, args.retries) + 1):
            try:
                row = run_benchmark_episode(
                    spec,
                    case,
                    episode.seed,
                    adapter=args.adapter,
                    allow_fallback=args.allow_fallback,
                    dt=args.dt,
                    current_profile=args.current_profile,
                    capture_video=capture_video,
                    video_stride=args.video_stride,
                    trajectory_stride=args.trajectory_stride,
                    artifact_dir=artifact_dir,
                )
                exc = None
                break
            except Exception as error:  # pragma: no cover - simulator/runtime failure
                exc = error
                print(
                    f"[bench:{args.shard}] attempt {attempt}/{args.retries} failed for "
                    f"{episode.key}: {type(error).__name__}: {error}",
                    flush=True,
                )
                if attempt < args.retries:
                    time.sleep(15.0 * attempt)
        if row is None:
            row = {
                "schema_version": EPISODE_SCHEMA_VERSION,
                "controller": spec.key,
                "controller_kind": spec.kind,
                "model": spec.model,
                "policy_only": spec.policy_only,
                "separate_reporting": spec.separate_reporting,
                "group": case.group,
                "case_id": case.case_id,
                "case_uid": case.uid,
                "track": case.track,
                "official_circuit": case.official,
                "seed": int(episode.seed),
                "seed_role": "reused"
                if int(episode.seed) in case.reused_seeds
                else "holdout",
                "referee_status": "HARNESS_ERROR",
                "evaluation_end_reason": "HARNESS_ERROR",
                "finished": False,
                "completed_gates": 0,
                "expected_gates": int(case.expected_gates),
                "full_completion": False,
                "error": f"{type(exc).__name__}: {exc}",
                "attempts": int(args.retries),
                "wall_s": round(time.time() - started, 3),
            }
            print(f"[bench:{args.shard}] ERROR {episode.key}: {row['error']}", flush=True)
        with shard_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(
            f"[bench:{args.shard}] {episode.controller:<28} {episode.group:<26} "
            f"{episode.case_id:<22} seed={episode.seed:<6} "
            f"{row.get('referee_status','?'):<9} "
            f"gates={row.get('completed_gates')}/{row.get('expected_gates')} "
            f"t={row.get('official_time_s')} wall={row.get('wall_s')}s",
            flush=True,
        )
    return 0


def drop_harness_errors(out_dir: str | Path) -> int:
    """Forget recorded harness errors so a re-run re-attempts those episodes.

    A ``HARNESS_ERROR`` row is a simulator launch failure, never a controller
    result. Keeping it would both freeze the episode as "done" and count it as a
    failed episode, so the repair pass removes it and lets the benchmark re-run it.
    """
    out = Path(out_dir)
    removed = 0
    for path in sorted(out.glob("episodes.shard*.jsonl")):
        kept: List[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if row.get("referee_status") == "HARNESS_ERROR":
                removed += 1
                continue
            kept.append(stripped)
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed


def run_all(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "repair", False):
        removed = drop_harness_errors(out_dir)
        print(f"[bench] repair: dropped {removed} harness-error episode(s)", flush=True)
    cases = build_suite(
        out_dir,
        episodes_per_case=args.episodes_per_case,
        official_episodes=args.official_episodes,
        groups=args.groups,
    )
    controllers = _select_controllers(args.controllers)
    missing = [
        spec.key
        for spec in controllers
        if spec.model and not Path(spec.model).exists()
    ]
    if missing:
        print(f"[bench] ABORT: missing model files for {missing}")
        return 2
    manifest = _suite_manifest(out_dir, cases, controllers, args)
    _atomic_write_json(out_dir / "suite_manifest.json", manifest)
    print(
        f"[bench] {manifest['total_episodes']} episodes = "
        f"{len(cases)} cases x {len(controllers)} controllers; "
        f"workers={args.workers}",
        flush=True,
    )
    if args.plan_only:
        return 0

    processes: List[subprocess.Popen] = []
    for shard in range(args.workers):
        command = [
            sys.executable,
            "-m",
            "marine_race_arena.learning.final_benchmark",
            "shard",
            "--out",
            str(out_dir),
            "--shard",
            str(shard),
            "--shards",
            str(args.workers),
            "--adapter",
            args.adapter,
            "--current-profile",
            args.current_profile,
            "--dt",
            str(args.dt),
            "--episodes-per-case",
            str(args.episodes_per_case),
            "--official-episodes",
            str(args.official_episodes),
            "--video-stride",
            str(args.video_stride),
            "--trajectory-stride",
            str(args.trajectory_stride),
            "--retries",
            str(args.retries),
        ]
        if args.allow_fallback:
            command.append("--allow-fallback")
        if args.video:
            command.append("--video")
        if args.groups:
            command += ["--groups", *args.groups]
        if args.controllers:
            command += ["--controllers", *args.controllers]
        log_path = out_dir / f"shard{shard:02d}.log"
        handle = log_path.open("a", encoding="utf-8")
        processes.append(
            subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        )
        print(f"[bench] shard {shard} -> {log_path}", flush=True)

    codes = [process.wait() for process in processes]
    print(f"[bench] shard exit codes: {codes}", flush=True)
    merged = merge_episodes(out_dir)
    print(f"[bench] merged {len(merged)} episodes -> {out_dir/'episodes.json'}", flush=True)
    return 0 if all(code == 0 for code in codes) else 1


def merge_episodes(out_dir: str | Path) -> List[Dict[str, Any]]:
    """Merge shard JSONL files into a stable, machine-readable episode table."""
    out = Path(out_dir)
    rows = list(_completed_keys(out).values())
    rows.sort(
        key=lambda r: (
            bool(r.get("official_circuit")),
            str(r.get("group")),
            str(r.get("case_id")),
            int(r.get("seed", 0)),
            str(r.get("controller")),
        )
    )
    _atomic_write_json(out / "episodes.json", rows)
    _write_episode_csv(out / "episodes.csv", rows)
    return rows


_CSV_EXCLUDE = {"policy_evidence"}


def _write_episode_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields and key not in _CSV_EXCLUDE:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    tmp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--out", required=True)
        sp.add_argument("--adapter", default="holoocean")
        sp.add_argument("--allow-fallback", action="store_true")
        sp.add_argument("--current-profile", default="none")
        sp.add_argument("--dt", type=float, default=0.1)
        sp.add_argument("--episodes-per-case", type=int, default=8)
        sp.add_argument("--official-episodes", type=int, default=5)
        sp.add_argument("--groups", nargs="*", default=None)
        sp.add_argument("--controllers", nargs="*", default=None)
        sp.add_argument("--video", action="store_true")
        sp.add_argument("--video-stride", type=int, default=5)
        sp.add_argument("--trajectory-stride", type=int, default=1)
        sp.add_argument(
            "--retries",
            type=int,
            default=3,
            help="attempts per episode before recording a harness error; a simulator "
                 "launch timeout is infrastructure, not a controller failure",
        )

    run = sub.add_parser("run", help="plan and execute the whole benchmark")
    common(run)
    run.add_argument("--workers", type=int, default=5)
    run.add_argument("--plan-only", action="store_true")
    run.add_argument(
        "--repair",
        action="store_true",
        help="drop previously recorded harness errors first so they are re-run",
    )

    shard = sub.add_parser("shard", help="execute one shard (used by run)")
    common(shard)
    shard.add_argument("--shard", type=int, required=True)
    shard.add_argument("--shards", type=int, required=True)
    shard.add_argument(
        "--reverse",
        action="store_true",
        help="drain this shard's queue from the tail; run alongside the forward "
             "shard to start the expensive official circuits sooner",
    )

    merge = sub.add_parser("merge", help="merge shard results into episodes.json/csv")
    merge.add_argument("--out", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return run_all(args)
    if args.command == "shard":
        return run_shard(args)
    if args.command == "merge":
        merge_episodes(args.out)
        return 0
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
