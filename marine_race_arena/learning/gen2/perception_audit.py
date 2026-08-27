"""Offline audit of the rover's OWN onboard perception.

The controller is onboard-only and stays that way::

    action = learned_policy(onboard_observation_35, recurrent_state)

Nothing here changes that.  Simulator ground truth is used here as an external
**measuring instrument**, exactly the way a bench technician uses a reference
sensor: it is read after the fact to decide whether the rover's own estimates
are correct.  It is never concatenated to the observation, never given to a
policy, never stored in recurrent state, never used to correct or select
anything, and it does not exist at inference.

The separation is structural rather than promised.  One function produces the
policy input, a different function produces the reference, they return
different objects, and :func:`assert_audit_is_offline_only` reads this module
and the inference path to confirm no reference value can reach an actor.

What the audit answers, per step, on a real circuit:

* does the rover's estimated bearing/elevation/range to its expected gate agree
  with the true geometry, in sign, in units, and in magnitude;
* does the expected target switch after a crossing, and how many steps late;
* are frames stale, are sensors missing, are features saturating at their
  clip bounds;
* is the observation/action timestep aligned -- does ``previous_*`` really
  carry the action applied on the step before.

If perception is wrong, the fix belongs in the perception pipeline.  Feeding
the policy simulator state instead would make the number better and the
controller invalid.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.config import (
    ACTION_AXES,
    ACTION_DIM,
    ELEVATION_SCALE_DEG,
    RANGE_SCALE_M,
    VELOCITY_SCALE_MPS,
)
from marine_race_arena.learning.config_local_transition import (
    FEATURE_BOUNDS_LOCAL_TRANSITION,
    FEATURE_NAMES_LOCAL_TRANSITION,
    OBS_DIM_LOCAL_TRANSITION,
)

F = {name: index for index, name in enumerate(FEATURE_NAMES_LOCAL_TRANSITION)}

#: Steps after a crossing within which the expected target must switch.
TARGET_SWITCH_DEADLINE_STEPS = 30


@dataclass
class PerceptionSample:
    """One step: what the rover believed, and what was true.

    The two halves are kept in separate fields on purpose.  ``sensed_*`` is
    decoded from the 35-feature observation the policy receives; ``true_*``
    comes from the simulator and exists only inside this dataclass.
    """

    step: int
    # --- what the rover sensed (decoded from its own observation) -----------
    sensed_present: bool
    sensed_bearing_deg: Optional[float]
    sensed_elevation_deg: Optional[float]
    sensed_range_m: Optional[float]
    sensed_expected_beacon: Optional[str]
    # --- what was actually true (offline reference only) --------------------
    # Two references, because the first run of this audit proved they answer
    # different questions.  ``true_*`` aims at the beacon of the gate the
    # referee is waiting for and therefore measures *targeting*: is the rover
    # pointed at the right gate?  ``tracked_*`` aims at the beacon the rover
    # itself has decided to chase and therefore measures *perception*: given
    # its own choice of target, is its bearing/elevation/range right?  Reading
    # only the first conflates a wrong target with a wrong measurement.
    true_bearing_deg: Optional[float]
    true_elevation_deg: Optional[float]
    true_range_m: Optional[float]
    true_expected_gate: Optional[str]
    tracked_bearing_deg: Optional[float]
    tracked_elevation_deg: Optional[float]
    tracked_range_m: Optional[float]
    tracked_gate: Optional[str]
    gates_crossed: int
    # --- pipeline health ----------------------------------------------------
    beacon_age_norm: float
    vision_present: bool
    dvl_present: bool
    imu_present: bool
    depth_present: bool
    saturated_features: int
    observation_repeated: bool

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def decode_sensed(observation: np.ndarray) -> Dict[str, Any]:
    """Invert the encoder's scaling to read the rover's estimate in real units.

    This is the only honest way to compare the rover against a reference: the
    policy sees normalized features, so the audit must undo exactly the scaling
    the encoder applied rather than compare against raw sensor values the
    policy never sees.
    """
    obs = np.asarray(observation, dtype=np.float64).reshape(-1)
    present = bool(obs[F["beacon_present"]] > 0.5)
    if not present:
        return {"present": False, "bearing_deg": None,
                "elevation_deg": None, "range_m": None}
    bearing = math.degrees(math.atan2(
        obs[F["beacon_bearing_sin"]], obs[F["beacon_bearing_cos"]]
    ))
    return {
        "present": True,
        "bearing_deg": bearing,
        "elevation_deg": float(obs[F["beacon_elevation_norm"]]) * ELEVATION_SCALE_DEG,
        "range_m": float(obs[F["beacon_range_norm"]]) * RANGE_SCALE_M,
    }


def true_geometry(
    rover_position: Sequence[float],
    rover_yaw_deg: float,
    target_position: Sequence[float],
) -> Dict[str, float]:
    """Reference bearing/elevation/range, body frame.  OFFLINE USE ONLY.

    ``target_position`` must be the point the sensor actually measures, which
    is the beacon and *not* the gate centre: the beacon sits 0.35 m up the
    gate's own up-axis.  Aiming this reference at the centre instead made the
    first run of this audit report near-chance elevation sign agreement, which
    was the instrument's error and not the rover's -- at gate depth the true
    elevation to the centre is ~0, so a 0.35 m offset decides its sign.

    Kept deliberately separate from anything the policy touches; the caller is
    responsible for never letting the result leave the diagnostics channel.
    """
    delta = np.asarray(target_position, dtype=np.float64) - np.asarray(
        rover_position, dtype=np.float64
    )
    yaw = math.radians(float(rover_yaw_deg))
    forward = delta[0] * math.cos(yaw) + delta[1] * math.sin(yaw)
    left = -delta[0] * math.sin(yaw) + delta[1] * math.cos(yaw)
    horizontal = math.hypot(forward, left)
    distance = float(np.linalg.norm(delta))
    return {
        "bearing_deg": math.degrees(math.atan2(left, forward)),
        "elevation_deg": math.degrees(math.atan2(delta[2], horizontal)) if horizontal > 1e-9 else 0.0,
        "range_m": distance,
    }


def _gate_for_beacon(
    beacon_id: Optional[str], gate_sequence: Sequence[str]
) -> Optional[str]:
    """Which gate the rover's own expected beacon refers to.  OFFLINE ONLY.

    Beacon ids are assigned by sequence position (``B01`` is the first gate in
    ``track.gate_sequence``), so this is a lookup and not an inference.
    """
    text = str(beacon_id or "").strip().upper()
    if not text.startswith("B"):
        return None
    try:
        index = int(text[1:]) - 1
    except ValueError:
        return None
    if 0 <= index < len(gate_sequence):
        return str(gate_sequence[index])
    return None


def _beacon_position(beacon_manager, gate_id: str, gate_map) -> Optional[Sequence[float]]:
    """The transmitter's own position, falling back to the gate centre."""
    for beacon in getattr(beacon_manager, "beacons", []):
        if getattr(beacon, "gate_id", None) == gate_id:
            return beacon.position
    gate = gate_map.get(gate_id)
    return None if gate is None else gate.center


def _saturated(observation: np.ndarray) -> int:
    """How many features sit exactly on a clip bound this step."""
    obs = np.asarray(observation, dtype=np.float64).reshape(-1)
    count = 0
    for index, (low, high) in enumerate(FEATURE_BOUNDS_LOCAL_TRANSITION):
        value = obs[index]
        if abs(value - low) < 1e-9 or abs(value - high) < 1e-9:
            count += 1
    return count


def audit_episode(
    track_path: str | Path,
    *,
    seed: int,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
    dt: float = 0.1,
    max_steps: int = 4000,
    initial_body_velocity: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Drive one expert episode and record sensed-versus-true per step."""
    from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
    from marine_race_arena.controllers.official_baselines import (
        RuleGateCenterThenCommitController,
    )
    from marine_race_arena.learning.episode import RaceEpisode
    from marine_race_arena.learning.gen2.expert_rollout import (
        action_to_command,
        command_to_action,
    )
    from marine_race_arena.learning.observation_encoder_local_transition import (
        encode_observation_local_transition,
    )
    from marine_race_arena.learning.tracker_context_local_transition import (
        OnboardLocalTransitionContextTracker,
    )
    from marine_race_arena.scripts.run_marine_race import _mission_info

    episode = RaceEpisode(
        str(track_path), seed=int(seed), dt=float(dt), adapter=adapter,
        allow_fallback=allow_fallback, max_steps=int(max_steps),
        official=True, current_profile="none",
        benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
    )
    expert = RuleGateCenterThenCommitController()
    samples: List[PerceptionSample] = []
    applied_actions: List[np.ndarray] = []
    observations: List[np.ndarray] = []

    try:
        raw = episode.reset(seed=int(seed))
        if initial_body_velocity is not None:
            from marine_race_arena.learning.gen2.track_fragments import (
                apply_initial_body_velocity,
            )

            apply_initial_body_velocity(episode, initial_body_velocity)
            raw = episode._build_observation()
        config = episode.context.config
        gate_map = episode.context.arena.gate_map
        gate_sequence = list(config.track.gate_sequence)
        beacons = episode.context.arena.beacon_manager
        context_source = OnboardLocalTransitionContextTracker(
            total_beacons=max(1, len(config.track.gate_sequence)),
            laps=max(1, int(config.race.laps)),
        )
        context_source.reset(raw)
        expert.reset(dict(_mission_info(config, episode.context.participant.id)))

        previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        previous_obs: Optional[np.ndarray] = None
        step_index = 0

        while step_index < max_steps:
            context = context_source.context(
                raw, dt=float(dt), prev_action=previous_action.tolist()
            )
            encoded = encode_observation_local_transition(raw, context)
            observations.append(encoded)
            sensed = decode_sensed(encoded)

            # --- OFFLINE REFERENCE. Read after the observation is already
            # --- built, so it cannot influence what the policy would see.
            state = episode.context.adapter.get_participant_state(episode.participant_id)
            expected_gate = episode.expected_gate_id()
            tracked_gate = _gate_for_beacon(
                getattr(context, "expected_beacon_id", None), gate_sequence
            )

            def reference(gate_id):
                if gate_id is None:
                    return None
                target = _beacon_position(beacons, gate_id, gate_map)
                if target is None:
                    return None
                return true_geometry(
                    state.position, state.rotation_rpy_deg[2], target
                )

            truth = reference(expected_gate)
            tracked = reference(tracked_gate)

            crossings = int(episode.referee_progress()["valid_gate_crossings"])
            repeated = bool(
                previous_obs is not None and np.array_equal(previous_obs, encoded)
            )
            samples.append(PerceptionSample(
                step=step_index,
                sensed_present=sensed["present"],
                sensed_bearing_deg=sensed["bearing_deg"],
                sensed_elevation_deg=sensed["elevation_deg"],
                sensed_range_m=sensed["range_m"],
                sensed_expected_beacon=getattr(context, "expected_beacon_id", None),
                true_bearing_deg=None if truth is None else truth["bearing_deg"],
                true_elevation_deg=None if truth is None else truth["elevation_deg"],
                true_range_m=None if truth is None else truth["range_m"],
                true_expected_gate=expected_gate,
                tracked_bearing_deg=None if tracked is None else tracked["bearing_deg"],
                tracked_elevation_deg=None if tracked is None else tracked["elevation_deg"],
                tracked_range_m=None if tracked is None else tracked["range_m"],
                tracked_gate=tracked_gate,
                gates_crossed=crossings,
                beacon_age_norm=float(encoded[F["beacon_age_norm"]]),
                vision_present=bool(encoded[F["vision_present"]] > 0.5),
                dvl_present=bool(encoded[F["dvl_present"]] > 0.5),
                imu_present=bool(encoded[F["imu_present"]] > 0.5),
                depth_present=bool(encoded[F["depth_present"]] > 0.5),
                saturated_features=_saturated(encoded),
                observation_repeated=repeated,
            ))

            command = expert.step(dict(raw))
            action = command_to_action(command)
            applied_actions.append(action)
            step = episode.step(action_to_command(action))
            raw = step.observation
            previous_obs = encoded
            previous_action = action
            step_index += 1
            if step.terminated or step.truncated:
                break

        report = summarize(samples, observations, applied_actions)
        report["track"] = str(track_path)
        report["seed"] = int(seed)
        report["steps"] = step_index
        report["gates_crossed"] = samples[-1].gates_crossed if samples else 0
        return report
    finally:
        try:
            expert.close()
        finally:
            episode.close()


def _angle_error(sensed: float, true: float) -> float:
    """Shortest-arc difference, so 179 versus -179 is 2 degrees, not 358."""
    return (float(sensed) - float(true) + 180.0) % 360.0 - 180.0


def summarize(
    samples: Sequence[PerceptionSample],
    observations: Sequence[np.ndarray],
    actions: Sequence[np.ndarray],
) -> Dict[str, Any]:
    """Turn per-step records into the questions the audit is asking."""
    if not samples:
        return {"samples": 0}

    paired = [
        s for s in samples
        if s.sensed_present and s.true_bearing_deg is not None and s.sensed_range_m
    ]
    # Against the rover's OWN target: this is the perception question.  A large
    # error here means the sensing or the encoding is wrong.
    tracked = [
        s for s in samples
        if s.sensed_present and s.tracked_bearing_deg is not None and s.sensed_range_m
    ]
    bearing_err = np.asarray(
        [_angle_error(s.sensed_bearing_deg, s.true_bearing_deg) for s in paired]
    ) if paired else np.zeros(0)
    elevation_err = np.asarray(
        [_angle_error(s.sensed_elevation_deg, s.true_elevation_deg) for s in paired]
    ) if paired else np.zeros(0)
    range_err = np.asarray(
        [s.sensed_range_m - s.true_range_m for s in paired]
    ) if paired else np.zeros(0)
    tracked_bearing_err = np.asarray(
        [_angle_error(s.sensed_bearing_deg, s.tracked_bearing_deg) for s in tracked]
    ) if tracked else np.zeros(0)
    tracked_elevation_err = np.asarray(
        [_angle_error(s.sensed_elevation_deg, s.tracked_elevation_deg) for s in tracked]
    ) if tracked else np.zeros(0)
    tracked_range_err = np.asarray(
        [s.sensed_range_m - s.tracked_range_m for s in tracked]
    ) if tracked else np.zeros(0)
    # And the targeting question, kept as a plain count so the two never blur.
    aimed = [s for s in samples if s.true_expected_gate and s.tracked_gate]
    on_target = sum(1 for s in aimed if s.true_expected_gate == s.tracked_gate)

    def stats(values: np.ndarray) -> Dict[str, Any]:
        if values.size == 0:
            return {"n": 0}
        return {
            "n": int(values.size),
            "mean": round(float(values.mean()), 4),
            "median": round(float(np.median(values)), 4),
            "abs_mean": round(float(np.abs(values).mean()), 4),
            "p95_abs": round(float(np.percentile(np.abs(values), 95)), 4),
            "max_abs": round(float(np.abs(values).max()), 4),
        }

    # Sign agreement matters more than magnitude: a flipped sign is a bug, a
    # small bias is a calibration issue.
    def sign_agreement(sensed: List[float], true: List[float]) -> Optional[float]:
        pairs = [(s, t) for s, t in zip(sensed, true)
                 if abs(s) > 1.0 and abs(t) > 1.0]
        if not pairs:
            return None
        return round(sum(1 for s, t in pairs if (s > 0) == (t > 0)) / len(pairs), 4)

    bearing_sign = sign_agreement(
        [s.sensed_bearing_deg for s in paired], [s.true_bearing_deg for s in paired]
    )
    elevation_sign = sign_agreement(
        [s.sensed_elevation_deg for s in paired], [s.true_elevation_deg for s in paired]
    )

    # Target switch latency: after each crossing, how many steps until the
    # rover's own expected beacon changes.
    latencies: List[int] = []
    previous_crossings = samples[0].gates_crossed
    pending: Optional[Tuple[int, Optional[str]]] = None
    for sample in samples:
        if sample.gates_crossed > previous_crossings:
            pending = (sample.step, sample.sensed_expected_beacon)
            previous_crossings = sample.gates_crossed
        elif pending is not None and sample.sensed_expected_beacon != pending[1]:
            latencies.append(sample.step - pending[0])
            pending = None
    switch = {
        "crossings": int(samples[-1].gates_crossed),
        "switches_observed": len(latencies),
        "median_latency_steps": int(np.median(latencies)) if latencies else None,
        "max_latency_steps": int(max(latencies)) if latencies else None,
        "late_switches": int(sum(1 for v in latencies if v > TARGET_SWITCH_DEADLINE_STEPS)),
    }

    # Timestep alignment: the observation at step t must carry the action
    # applied at t-1, which is what the encoder promises.
    misaligned = 0
    checked = 0
    for index in range(1, min(len(observations), len(actions))):
        recorded = np.asarray([
            observations[index][F["previous_surge"]],
            observations[index][F["previous_sway"]],
            observations[index][F["previous_heave"]],
            observations[index][F["previous_yaw"]],
        ], dtype=np.float32)
        checked += 1
        if not np.allclose(recorded, actions[index - 1], atol=1e-5):
            misaligned += 1

    total = len(samples)
    return {
        "samples": total,
        # Versus the referee's expected gate -- TARGETING.
        "bearing_error_deg": stats(bearing_err),
        "elevation_error_deg": stats(elevation_err),
        "range_error_m": stats(range_err),
        # Versus the rover's own expected gate -- PERCEPTION.
        "tracked_bearing_error_deg": stats(tracked_bearing_err),
        "tracked_elevation_error_deg": stats(tracked_elevation_err),
        "tracked_range_error_m": stats(tracked_range_err),
        "on_target_fraction": (
            round(on_target / len(aimed), 4) if aimed else None
        ),
        "on_target_steps": on_target,
        "aimed_steps": len(aimed),
        "bearing_sign_agreement": bearing_sign,
        "elevation_sign_agreement": elevation_sign,
        "target_switch": switch,
        "timestep_alignment": {
            "checked": checked,
            "misaligned": misaligned,
            "aligned_fraction": round(1.0 - misaligned / max(1, checked), 4),
        },
        "availability": {
            "beacon": round(sum(s.sensed_present for s in samples) / total, 4),
            "vision": round(sum(s.vision_present for s in samples) / total, 4),
            "dvl": round(sum(s.dvl_present for s in samples) / total, 4),
            "imu": round(sum(s.imu_present for s in samples) / total, 4),
            "depth": round(sum(s.depth_present for s in samples) / total, 4),
        },
        "stale": {
            "repeated_observation_fraction": round(
                sum(s.observation_repeated for s in samples) / total, 4),
            "mean_beacon_age_norm": round(
                float(np.mean([s.beacon_age_norm for s in samples])), 4),
            "max_beacon_age_norm": round(
                float(np.max([s.beacon_age_norm for s in samples])), 4),
        },
        "saturation": {
            "mean_features_at_clip_bound": round(
                float(np.mean([s.saturated_features for s in samples])), 2),
            "of_total_features": OBS_DIM_LOCAL_TRANSITION,
        },
    }


def assert_audit_is_offline_only(repo_root: Optional[str | Path] = None) -> Dict[str, Any]:
    """Confirm no reference value can reach an actor.

    Reads the inference path rather than trusting the design: the modules that
    produce actions must not import this one, must not touch the simulator's
    participant state, and must not read gate geometry.
    """
    base = Path(repo_root) if repo_root else Path.cwd()
    gen2 = base / "marine_race_arena" / "learning" / "gen2"
    findings: List[str] = []

    forbidden_in_inference = (
        "perception_audit", "true_geometry", "gate_map",
        "get_participant_state", "debug_ground_truth",
    )
    for module in ("recurrent_policy.py", "evaluation.py"):
        source = (gen2 / module).read_text(encoding="utf-8")
        for token in forbidden_in_inference:
            if token in source:
                findings.append(f"{module} references {token!r}")

    audit_source = (gen2 / "perception_audit.py").read_text(encoding="utf-8")
    # The reference must never be written into an observation array.
    for token in ("np.concatenate([obs", "observation.append(true", "obs[F[") :
        if token in audit_source and "true_" in token:
            findings.append(f"perception_audit builds an observation from truth: {token!r}")

    return {
        "offline_only": not findings,
        "findings": findings,
        "checked_modules": ["recurrent_policy.py", "evaluation.py", "perception_audit.py"],
    }


def write_report(report: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return target
