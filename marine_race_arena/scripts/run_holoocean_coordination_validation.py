"""HoloOcean validation for leader-follower coordination (onboard-only contract).

Runs a staggered team on the real HoloOcean adapter, fallback disabled,
official observations, no currents and no obstacles. Every rover navigates and
tracks its progression with its own LocalCourseTracker; coordination heartbeats
carry only locally estimated progress. The referee scores independently.

The team is deliberately heterogeneous in speed so followers catch the vehicle
ahead when uncoordinated: the leader runs the slower continuous-servo
``rule_gate_baseline`` and the followers run the faster
``rule_gate_center_then_commit``. Both are official onboard-only controllers.

Ablations supported for the coordinated condition:

* ``--min-gate-gap`` overrides the yield margin (default 1, the recommended
  setting; 2 is the conservative comparison margin);
* ``--comms-packet-loss-prob`` injects seeded acoustic packet loss.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marine_race_arena.adapters import AdapterSelectionError, RaceAdapterError, select_adapter
from marine_race_arena.arena.acoustic_comms import AcousticCommsChannel, CommsConfig
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import TrackConfigLoadError, load_track_config
from marine_race_arena.config.schema import TrackConfig
from marine_race_arena.controllers.leader_follower import LeaderFollowerController
from marine_race_arena.controllers.official_baselines import (
    RuleGateBaselineController,
    RuleGateCenterThenCommitController,
)
from marine_race_arena.participants.controller_interface import BaseController
from marine_race_arena.participants.participant import RaceParticipant
from marine_race_arena.referee.logger import RaceLogger
from marine_race_arena.referee.referee import Referee
from marine_race_arena.scripts.run_marine_race import (
    _mission_info,
    _offset_spawn_position,
    _run_race_loop,
    _staggered_lateral_offsets,
    _vector3,
)

DEFAULT_TRACK = "marine_race_arena/tracks/marine_race_horseshoe_bay.json"
DEFAULT_OUTPUT_DIR = "results/benchmarks/holoocean_coordination_validation"
DEFAULT_DURATION_S = 560.0
DEFAULT_DT = 0.033
DEFAULT_TEAM_SIZE = 3
DEFAULT_START_GAP_S = 8.0
DEFAULT_LATERAL_OFFSET_M = 1.5
DEFAULT_TEAM_ID = "holoocean_coordination"
CONDITIONS = ("no_coordination", "leader_follower")
INTER_VEHICLE_MODES = ("diagnostic", "penalize")
LEADER_ALIAS = "rule_gate_baseline"
FOLLOWER_ALIAS = "rule_gate_center_then_commit"
CONTROLLER_OBSERVATION_CONTRACT = "onboard_only_v1"


def _runtime_contract(adapter_name: str) -> Dict[str, Any]:
    """Machine-readable execution invariants for every coordination artifact."""
    return {
        "adapter": adapter_name,
        "fallback_used": adapter_name == "fallback",
        "fallback_allowed": False,
        "controller_observation_contract": CONTROLLER_OBSERVATION_CONTRACT,
    }


class _TracingController(BaseController):
    """Wrap a controller to count steps and accumulate command smoothness.

    ``mean_command_change`` is the average per-step change in the surge and yaw
    commands, a simple proxy for how smooth (low-jerk) the controller's motion
    is. The wrapper forwards the wrapped controller's honesty flags and its
    ``tracker`` so local diagnostics logging still works.
    """

    def __init__(self, inner: BaseController) -> None:
        self.inner = inner
        self.uses_ground_truth = bool(getattr(inner, "uses_ground_truth", False))
        self.debug_only = bool(getattr(inner, "debug_only", False))
        self._prev: Optional[Dict[str, float]] = None
        self.steps = 0
        self.command_change_sum = 0.0

    @property
    def tracker(self):
        tracker = getattr(self.inner, "tracker", None)
        if tracker is not None:
            return tracker
        base = getattr(self.inner, "base", None)
        return getattr(base, "tracker", None) if base is not None else None

    def reset(self, mission_info: Dict[str, Any]) -> None:
        self.inner.reset(mission_info)
        self._prev = None
        self.steps = 0
        self.command_change_sum = 0.0

    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        command = self.inner.step(observation)
        self.steps += 1
        if self._prev is not None:
            self.command_change_sum += abs(
                float(command.get("surge", 0.0)) - self._prev["surge"]
            ) + abs(float(command.get("yaw", 0.0)) - self._prev["yaw"])
        self._prev = {"surge": float(command.get("surge", 0.0)), "yaw": float(command.get("yaw", 0.0))}
        return command

    def close(self) -> None:
        self.inner.close()

    def coordination_diagnostics(self) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {
            "controller_steps": self.steps,
            "mean_command_change": self.mean_command_change,
        }
        inner_diagnostics = getattr(self.inner, "coordination_diagnostics", None)
        value = inner_diagnostics() if callable(inner_diagnostics) else inner_diagnostics
        if isinstance(value, Mapping):
            diagnostics.update(dict(value))
        return diagnostics

    @property
    def mean_command_change(self) -> float:
        return self.command_change_sum / max(1, self.steps - 1)


def _make_team(
    size: int,
    *,
    coordinated: bool,
    min_gate_gap: Optional[int] = None,
) -> List[BaseController]:
    bases: List[BaseController] = [RuleGateBaselineController()] + [
        RuleGateCenterThenCommitController() for _ in range(size - 1)
    ]
    if not coordinated:
        return [_TracingController(base) for base in bases]
    return [
        _TracingController(
            LeaderFollowerController(base_controller=base, min_gate_gap=min_gate_gap)
        )
        for base in bases
    ]


def simulate_holoocean_fleet(
    *,
    track_path: str,
    controllers: List[BaseController],
    seed: int,
    inter_vehicle_collision_mode: str,
    output_dir: Path,
    duration_s: float,
    dt: float,
    start_gap_s: float,
    lateral_offset_m: float,
    comms_enabled: bool,
    comms_packet_loss_prob: float,
    team_id: str,
    headless: bool,
    log_participant_states: bool,
) -> Dict[str, Any]:
    config = load_track_config(
        track_path,
        benchmark_task="clean_gate",
        obstacles="none",
        current_profile="none",
        seed=seed,
    )
    config = replace(
        config,
        race=replace(config.race, max_duration_s=float(duration_s), official_mode=True),
    )
    config = _with_explicit_staggered_participants(
        config,
        num_rovers=len(controllers),
        start_gap_s=start_gap_s,
        lateral_offset_m=lateral_offset_m,
    )
    arena = ArenaBuilder(config, seed=seed).build()
    logger = RaceLogger(output_dir, config.race.name, track_file=track_path)
    participants = _participants_from_controllers(config, controllers)
    referee = Referee(
        config,
        arena.gate_map,
        arena.bounds,
        logger=logger,
        inter_vehicle_collision_mode=inter_vehicle_collision_mode,
        team_id=team_id,
    )
    comms_channel = (
        AcousticCommsChannel(
            CommsConfig(enabled=True, packet_loss_prob=comms_packet_loss_prob),
            seed=seed,
        )
        if comms_enabled
        else None
    )
    adapter = None
    try:
        adapter = select_adapter(
            adapter_name="holoocean",
            config=config,
            arena=arena,
            allow_fallback=False,
            headless=headless,
            record=False,
            seed=seed,
        )
        adapter.spawn_participants(participants)
        adapter.reset()
        adapter.spawn_visual_gates(arena.visual_gates)
        adapter.spawn_obstacles(arena.obstacles)

        referee.register_participants(participants.keys())
        for participant in participants.values():
            participant.controller.reset(_mission_info(config, participant.id))

        summary = _run_race_loop(
            config=config,
            arena=arena,
            referee=referee,
            adapter=adapter,
            participants=participants,
            dt=dt,
            gate_timeout_s=None,
            log_participant_states=log_participant_states,
            comms_channel=comms_channel,
        )
        summary.update(_runtime_contract(adapter.name))
        summary["motion_compensation"] = "none"
        summary["inter_vehicle_collision_mode"] = inter_vehicle_collision_mode
        if comms_channel is not None:
            summary["comms"] = comms_channel.summary()
        summary["local_progress"] = _local_progress_snapshot(participants)
        summary["validation_setup"] = {
            "track": track_path,
            **_runtime_contract(adapter.name),
            "official": True,
            "benchmark_task": "clean_gate",
            "current_profile": "none",
            "obstacles": "none",
            "seed": seed,
            "dt": dt,
            "duration_s": duration_s,
            "team_size": len(controllers),
            "start_gap_s": start_gap_s,
            "lateral_offset_m": lateral_offset_m,
            "comms_enabled": comms_enabled,
            "comms_packet_loss_prob": comms_packet_loss_prob,
            "team_id": team_id,
        }
        logger.log_event("race_summary", adapter.get_current_time(), summary=summary)
        logger.write_summary(summary)
        return {
            "ok": True,
            "summary": summary,
            "summary_path": str(logger.summary_path),
            "event_path": str(logger.event_path),
        }
    except (AdapterSelectionError, RaceAdapterError, TrackConfigLoadError, ValueError) as exc:
        logger.log_event("validation_error", 0.0, error=f"{type(exc).__name__}: {exc}")
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "summary_path": str(logger.summary_path),
            "event_path": str(logger.event_path),
        }
    finally:
        for participant in participants.values():
            try:
                participant.controller.close()
            except Exception:
                pass
        if adapter is not None:
            adapter.close()
        logger.close()


def _local_progress_snapshot(participants: Mapping[str, RaceParticipant]) -> Dict[str, Any]:
    """Final controller-local tracker states (diagnostics; separate from referee truth)."""
    snapshot: Dict[str, Any] = {}
    for participant_id, participant in participants.items():
        tracker = getattr(participant.controller, "tracker", None)
        if tracker is None:
            continue
        diagnostics = dict(tracker.diagnostics())
        coordination_diagnostics = getattr(
            participant.controller, "coordination_diagnostics", None
        )
        if callable(coordination_diagnostics):
            coordination = coordination_diagnostics()
            if isinstance(coordination, Mapping):
                diagnostics["coordination"] = dict(coordination)
        snapshot[participant_id] = diagnostics
    return snapshot


def run_validation(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            f"Refusing to mix coordination artifacts in non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: Dict[str, Any] = {}
    for mode in args.inter_vehicle_modes:
        runs[mode] = {}
        for seed in args.seeds:
            runs[mode][str(seed)] = {}
            for condition in args.conditions:
                coordinated = condition == "leader_follower"
                controllers = _make_team(
                    args.team_size,
                    coordinated=coordinated,
                    min_gate_gap=args.min_gate_gap if coordinated else None,
                )
                run_dir = output_dir / mode / f"seed_{seed}" / condition
                run_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"Running {condition} mode={mode} seed={seed} "
                    f"team={args.team_size} start_gap={args.start_gap_s:g}s "
                    f"min_gate_gap={args.min_gate_gap} loss={args.comms_packet_loss_prob:g}",
                    flush=True,
                )
                metadata = _build_run_metadata(
                    args=args,
                    seed=int(seed),
                    condition=condition,
                    mode=mode,
                    run_dir=run_dir,
                    controllers=_controller_labels(args.team_size, coordinated=coordinated),
                )
                metadata_path = run_dir / "experiment_metadata.json"
                metadata["started_at"] = _now_iso()
                _write_json(metadata_path, metadata)
                started = time.monotonic()
                try:
                    result = simulate_holoocean_fleet(
                        track_path=args.track,
                        controllers=controllers,
                        seed=int(seed),
                        inter_vehicle_collision_mode=mode,
                        output_dir=run_dir,
                        duration_s=args.duration_s,
                        dt=args.dt,
                        start_gap_s=args.start_gap_s,
                        lateral_offset_m=args.lateral_offset_m,
                        comms_enabled=coordinated,
                        comms_packet_loss_prob=(
                            args.comms_packet_loss_prob if coordinated else 0.0
                        ),
                        team_id=f"{args.team_id}_{mode}_{condition}_seed{seed}",
                        headless=args.headless,
                        log_participant_states=args.log_participant_states,
                    )
                except Exception as exc:  # preserve failed runs in the manifest
                    result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                metadata["completed_at"] = _now_iso()
                metadata["wall_duration_s"] = round(time.monotonic() - started, 3)
                metadata["run_ok"] = bool(result.get("ok"))
                metadata["error"] = result.get("error")
                summary = result.get("summary")
                if isinstance(summary, Mapping):
                    metadata["actual_adapter"] = summary.get("adapter")
                    metadata["fallback_used"] = summary.get("fallback_used")
                    metadata["controller_observation_contract"] = summary.get(
                        "controller_observation_contract"
                    )
                    enriched_summary = dict(summary)
                    enriched_summary["experiment_metadata"] = dict(metadata)
                    result["summary"] = enriched_summary
                    summary_path = result.get("summary_path")
                    if summary_path:
                        _write_json(Path(summary_path), enriched_summary)
                _write_json(metadata_path, metadata)
                result["metadata_path"] = str(metadata_path)
                result["metadata"] = dict(metadata)
                result["condition"] = condition
                result["seed"] = int(seed)
                result["inter_vehicle_collision_mode"] = mode
                result["comms_enabled"] = coordinated
                result["controllers"] = _controller_labels(args.team_size, coordinated=coordinated)
                result["metrics"] = _metrics_from_summary(result.get("summary"))
                metrics = result["metrics"] or {}
                result["scientific_outcome"] = {
                    "all_rovers_finished": metrics.get("all_rovers_finished"),
                    "progress_consistent": metrics.get("progress_consistent"),
                    "clean_finish": metrics.get("clean_finish"),
                }
                runs[mode][str(seed)][condition] = result
    report = {
        "track": args.track,
        **_runtime_contract("holoocean"),
        "official": True,
        "benchmark_task": "clean_gate",
        "current_profile": "none",
        "obstacles": "none",
        "team_size": args.team_size,
        "start_gap_s": args.start_gap_s,
        "lateral_offset_m": args.lateral_offset_m,
        "dt": args.dt,
        "duration_s": args.duration_s,
        "leader": LEADER_ALIAS,
        "followers": FOLLOWER_ALIAS,
        "leader_follower_min_gate_gap": args.min_gate_gap,
        "comms_packet_loss_prob": args.comms_packet_loss_prob,
        "generated_at": _now_iso(),
        "invocation_argv": list(getattr(args, "invocation_argv", [])),
        "source_tree_sha256": _source_tree_sha256(),
        "runs": runs,
    }
    audit_errors = _artifact_contract_errors(runs)
    report["all_runs_executed"] = all(
        bool(result.get("ok"))
        for modes in runs.values()
        for conditions in modes.values()
        for result in conditions.values()
    )
    report["all_progress_consistent"] = all(
        bool((result.get("metrics") or {}).get("progress_consistent"))
        for modes in runs.values()
        for conditions in modes.values()
        for result in conditions.values()
        if result.get("ok")
    )
    report["artifact_contract_audit"] = {
        "ok": not audit_errors,
        "errors": audit_errors,
    }
    (output_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = _markdown(report)
    (output_dir / "validation.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nWrote {output_dir / 'validation.json'} and {output_dir / 'validation.md'}")
    return report


def _with_explicit_staggered_participants(
    config: TrackConfig,
    *,
    num_rovers: int,
    start_gap_s: float,
    lateral_offset_m: float,
) -> TrackConfig:
    if num_rovers < 1:
        raise ValueError("team size must be at least 1.")
    if not config.participants:
        raise ValueError("track has no base participant.")
    base = config.participants[0]
    base_position = _vector3(base.spawn.get("position", config.start.position))
    base_rotation = _vector3(base.spawn.get("rotation_rpy_deg", config.start.rotation_rpy_deg))
    offsets = _staggered_lateral_offsets(num_rovers, spacing_m=float(lateral_offset_m))
    participants = []
    for index in range(num_rovers):
        spawn = dict(base.spawn)
        spawn["position"] = list(_offset_spawn_position(config, base_position, base_rotation[2], offsets[index]))
        spawn["rotation_rpy_deg"] = list(base_rotation)
        spawn["start_delay_s"] = float(index) * float(start_gap_s)
        participants.append(
            replace(
                base,
                id=f"bluerov2_{index + 1:02d}",
                spawn=spawn,
                start_delay_s=float(spawn["start_delay_s"]),
            )
        )
    return replace(config, participants=participants)


def _participants_from_controllers(
    config: TrackConfig,
    controllers: List[BaseController],
) -> Dict[str, RaceParticipant]:
    participants: Dict[str, RaceParticipant] = {}
    for participant_config, controller in zip(config.participants, controllers):
        spawn = participant_config.spawn or {}
        participants[participant_config.id] = RaceParticipant(
            config=participant_config,
            controller=controller,
            position=tuple(_vector3(spawn.get("position", config.start.position))),
            rotation_rpy_deg=tuple(_vector3(spawn.get("rotation_rpy_deg", config.start.rotation_rpy_deg))),
        )
    return participants


def _controller_labels(size: int, *, coordinated: bool) -> List[str]:
    bases = [LEADER_ALIAS] + [FOLLOWER_ALIAS for _ in range(size - 1)]
    if coordinated:
        return [f"leader_follower({base})" for base in bases]
    return bases


def _metrics_from_summary(summary: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(summary, Mapping):
        return None
    participants = summary.get("participants")
    team = summary.get("team_summary") or {}
    rover_count = int(team.get("rover_count") or 0)
    expected_total_gates = team.get("expected_total_gates")
    expected_gates_per_rover = None
    if isinstance(expected_total_gates, int) and rover_count > 0:
        expected_gates_per_rover = expected_total_gates // rover_count
    local_progress = summary.get("local_progress") or {}
    rows = []
    if isinstance(participants, list):
        for participant in participants:
            if not isinstance(participant, Mapping):
                continue
            participant_id = participant.get("participant_id")
            local = local_progress.get(participant_id) if isinstance(local_progress, Mapping) else None
            rows.append(
                {
                    "participant_id": participant_id,
                    "status": participant.get("status"),
                    "completed_gates": participant.get("completed_gates"),
                    "expected_gates": participant.get("expected_gates", expected_gates_per_rover),
                    "official_time_s": _round(participant.get("official_time_s")),
                    "penalized_time_s": _round(participant.get("penalized_time_s")),
                    "gate_world_collisions": participant.get("collisions"),
                    "obstacle_collisions": participant.get("obstacle_collisions"),
                    "inter_vehicle_events": participant.get("involved_inter_vehicle_collisions"),
                    "out_of_bounds_events": participant.get("out_of_bounds_events"),
                    "stuck_events": participant.get("stuck_events"),
                    "penalties_s": _round(participant.get("penalties_s")),
                    "local_completed": (local or {}).get("local_completed"),
                    "local_status": (local or {}).get("status"),
                    "coordination": (local or {}).get("coordination"),
                }
            )
    comms = summary.get("comms") if isinstance(summary.get("comms"), Mapping) else None
    progress_consistent = bool(rows) and all(
        row.get("local_completed") == row.get("completed_gates")
        and ((row.get("local_status") == "FINISHED") == (row.get("status") == "FINISHED"))
        for row in rows
    )
    clean_finish = bool(team.get("all_rovers_finished")) and progress_consistent and all(
        int(row.get("gate_world_collisions") or 0) == 0
        and int(row.get("obstacle_collisions") or 0) == 0
        and int(row.get("inter_vehicle_events") or 0) == 0
        and int(row.get("out_of_bounds_events") or 0) == 0
        and int(row.get("stuck_events") or 0) == 0
        and float(row.get("penalties_s") or 0.0) == 0.0
        for row in rows
    )
    return {
        "all_rovers_finished": team.get("all_rovers_finished"),
        "progress_consistent": progress_consistent,
        "clean_finish": clean_finish,
        "team_completed_gates": team.get("total_completed_gates"),
        "team_expected_gates": team.get("expected_total_gates"),
        "team_elapsed_time_s": _round(team.get("team_elapsed_time_s")),
        "team_penalized_time_s": _round(team.get("team_penalized_time_s")),
        "total_gate_world_collisions": team.get("total_gate_collisions"),
        "total_obstacle_collisions": team.get("total_obstacle_collisions"),
        "total_inter_vehicle_events": team.get("total_inter_vehicle_collisions"),
        "total_collisions": team.get("total_collisions"),
        "total_penalties_s": _round(team.get("total_penalties_s")),
        "rovers": rows,
        "comms": _comms_metrics(comms),
    }


def _comms_metrics(comms: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if comms is None:
        return None
    return {
        "messages_sent": comms.get("messages_sent"),
        "messages_delivered": comms.get("messages_delivered"),
        "dropped_rate_limited": comms.get("dropped_rate_limited"),
        "dropped_oversized": comms.get("dropped_oversized"),
        "dropped_out_of_range": comms.get("dropped_out_of_range"),
        "dropped_packet_loss": comms.get("dropped_packet_loss"),
        "delivery_latency_s": comms.get("delivery_latency_s"),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# HoloOcean leader-follower coordination validation (onboard-only contract)",
        "",
        (
            f"Track: `{report['track']}` | adapter: `holoocean` | official mode | "
            "current profile: `none` | obstacles: `none`."
        ),
        (
            f"Team: {report['team_size']} rovers, start gap {report['start_gap_s']} s, "
            f"lateral offset {report['lateral_offset_m']} m. "
            f"Leader `{report['leader']}`, followers `{report['followers']}`; "
            f"`leader_follower` uses min_gate_gap={report['leader_follower_min_gate_gap']} "
            f"and packet loss {report['comms_packet_loss_prob']}. "
            "All progression is controller-local; heartbeats carry only local estimates."
        ),
        "",
    ]
    for mode, seeds in report["runs"].items():
        lines.extend(
            [
                f"## Inter-vehicle mode: `{mode}`",
                "",
                "| Seed | Condition | OK | All finished | Team gates | Inter-vehicle events | Gate/world collisions | OOB | Stuck | Team elapsed (s) | Team penalized (s) | Comms delivered/dropped |",
                "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for seed, conditions in seeds.items():
            for condition in CONDITIONS:
                result = conditions.get(condition)
                if result is None:
                    continue
                metrics = result.get("metrics") or {}
                if not result.get("ok"):
                    lines.append(
                        f"| {seed} | {condition.replace('_', ' ')} | no | - | - | - | - | - | - | - | - | "
                        f"{result.get('error', 'error')} |"
                    )
                    continue
                rovers = metrics.get("rovers") or []
                oob = sum(int(row.get("out_of_bounds_events") or 0) for row in rovers)
                stuck = sum(int(row.get("stuck_events") or 0) for row in rovers)
                comms = metrics.get("comms")
                comms_text = "disabled"
                if comms is not None:
                    dropped = sum(
                        int(comms.get(key) or 0)
                        for key in (
                            "dropped_rate_limited",
                            "dropped_oversized",
                            "dropped_out_of_range",
                            "dropped_packet_loss",
                        )
                    )
                    comms_text = f"{comms.get('messages_delivered')}/{dropped}"
                lines.append(
                    f"| {seed} | {condition.replace('_', ' ')} | yes | "
                    f"{metrics.get('all_rovers_finished')} | "
                    f"{metrics.get('team_completed_gates')}/{metrics.get('team_expected_gates')} | "
                    f"{metrics.get('total_inter_vehicle_events')} | "
                    f"{metrics.get('total_gate_world_collisions')} | {oob} | {stuck} | "
                    f"{metrics.get('team_elapsed_time_s')} | {metrics.get('team_penalized_time_s')} | "
                    f"{comms_text} |"
                )
        lines.extend(["", "### Per-rover detail", ""])
        lines.append(
            "| Seed | Condition | Rover | Status | Gates | Local gates | Official time (s) | Gate/world collisions | Inter-vehicle events | OOB | Stuck |"
        )
        lines.append("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for seed, conditions in seeds.items():
            for condition in CONDITIONS:
                metrics = (conditions.get(condition) or {}).get("metrics") or {}
                for row in metrics.get("rovers") or []:
                    lines.append(
                        f"| {seed} | {condition.replace('_', ' ')} | {row.get('participant_id')} | "
                        f"{row.get('status')} | {row.get('completed_gates')}/{row.get('expected_gates')} | "
                        f"{row.get('local_completed')} | "
                        f"{row.get('official_time_s')} | {row.get('gate_world_collisions')} | "
                        f"{row.get('inter_vehicle_events')} | {row.get('out_of_bounds_events')} | "
                        f"{row.get('stuck_events')} |"
                    )
        lines.append("")
    return "\n".join(lines)


def _round(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), 3)
    return value


def _build_run_metadata(
    *,
    args: argparse.Namespace,
    seed: int,
    condition: str,
    mode: str,
    run_dir: Path,
    controllers: List[str],
) -> Dict[str, Any]:
    environment = _holoocean_environment()
    reproduction_argv = _reproduction_argv(
        args=args,
        seed=seed,
        condition=condition,
        mode=mode,
        output_dir=run_dir / "reproduction",
    )
    return {
        "schema_version": 1,
        "experiment": "holoocean_coordination_validation",
        "created_at": _now_iso(),
        "track": args.track,
        "track_sha256": _file_sha256(Path(args.track)),
        "source_tree_sha256": _source_tree_sha256(),
        "seed": seed,
        "condition": condition,
        "inter_vehicle_collision_mode": mode,
        "controllers": list(controllers),
        "team_size": args.team_size,
        "start_gap_s": args.start_gap_s,
        "lateral_offset_m": args.lateral_offset_m,
        "min_gate_gap": args.min_gate_gap,
        "comms_packet_loss_prob": (
            args.comms_packet_loss_prob if condition == "leader_follower" else 0.0
        ),
        "duration_s": args.duration_s,
        "dt": args.dt,
        "official": True,
        "current_profile": "none",
        "obstacles": "none",
        **_runtime_contract("holoocean"),
        "git_commit_sha": _git_value("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(_git_value("status", "--porcelain")),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "holoocean_version": environment["version"],
        "holoocean_installed_packages": environment["installed_packages"],
        "invocation_argv": list(getattr(args, "invocation_argv", [])),
        "reproduction_argv": reproduction_argv,
        "reproduction_command": _shell_command(reproduction_argv),
        "run_dir": str(run_dir),
    }


def _reproduction_argv(
    *,
    args: argparse.Namespace,
    seed: int,
    condition: str,
    mode: str,
    output_dir: Path,
) -> List[str]:
    argv = [
        "-m",
        "marine_race_arena.scripts.run_holoocean_coordination_validation",
        "--track",
        str(args.track),
        "--team-size",
        str(args.team_size),
        "--start-gap-s",
        str(args.start_gap_s),
        "--lateral-offset-m",
        str(args.lateral_offset_m),
        "--seeds",
        str(seed),
        "--inter-vehicle-modes",
        mode,
        "--conditions",
        condition,
        "--min-gate-gap",
        str(args.min_gate_gap),
        "--comms-packet-loss-prob",
        str(args.comms_packet_loss_prob),
        "--duration-s",
        str(args.duration_s),
        "--dt",
        str(args.dt),
        "--team-id",
        str(args.team_id),
        "--headless" if args.headless else "--no-headless",
        "--output-dir",
        str(output_dir),
    ]
    if args.log_participant_states:
        argv.append("--log-participant-states")
    return argv


def _shell_command(python_argv: List[str]) -> str:
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    prefix = ["conda", "run", "-n", env_name, "python"] if env_name else [sys.executable]
    return subprocess.list2cmdline([*prefix, *python_argv])


def _artifact_contract_errors(runs: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    for mode, seeds in runs.items():
        for seed, conditions in seeds.items():
            for condition, result in conditions.items():
                label = f"{mode}/seed_{seed}/{condition}"
                if not isinstance(result, Mapping) or not result.get("ok"):
                    errors.append(f"{label}: run failed: {(result or {}).get('error')}")
                    continue
                summary = result.get("summary")
                if not isinstance(summary, Mapping):
                    errors.append(f"{label}: missing summary")
                    continue
                expected = _runtime_contract("holoocean")
                for key, value in expected.items():
                    if summary.get(key) != value:
                        errors.append(
                            f"{label}: {key}={summary.get(key)!r}, expected {value!r}"
                        )
                local_progress = summary.get("local_progress")
                if not isinstance(local_progress, Mapping):
                    errors.append(f"{label}: missing controller-local progress")
                participants = summary.get("participants")
                if isinstance(local_progress, Mapping) and isinstance(participants, list):
                    official_by_id = {
                        str(row.get("participant_id")): row
                        for row in participants
                        if isinstance(row, Mapping) and row.get("participant_id") is not None
                    }
                    if set(official_by_id) != set(local_progress):
                        errors.append(
                            f"{label}: participant ids differ between referee and local progress"
                        )
                    for participant_id in sorted(set(official_by_id) & set(local_progress)):
                        official = official_by_id[participant_id]
                        local = local_progress[participant_id]
                        if not isinstance(local, Mapping):
                            errors.append(f"{label}/{participant_id}: invalid local progress")
                            continue
                        official_count = official.get("completed_gates")
                        local_count = local.get("local_completed")
                        if local_count != official_count:
                            errors.append(
                                f"{label}/{participant_id}: local_completed={local_count!r}, "
                                f"referee_completed={official_count!r}"
                            )
                        if local.get("advancements") != local_count:
                            errors.append(
                                f"{label}/{participant_id}: advancements={local.get('advancements')!r}, "
                                f"local_completed={local_count!r}"
                            )
                        local_finished = local.get("status") == "FINISHED"
                        official_finished = official.get("status") == "FINISHED"
                        if local_finished != official_finished:
                            errors.append(
                                f"{label}/{participant_id}: local/referee FINISHED status differs"
                            )
                if condition == "leader_follower":
                    comms = summary.get("comms")
                    if not isinstance(comms, Mapping):
                        errors.append(f"{label}: missing comms summary")
                    else:
                        latency = comms.get("delivery_latency_s")
                        if not isinstance(latency, Mapping):
                            errors.append(f"{label}: missing delivery latency diagnostics")
                        elif latency.get("count") != comms.get("messages_delivered"):
                            errors.append(f"{label}: latency/delivery counts differ")
    return errors


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git_value(*args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _holoocean_environment() -> Dict[str, Any]:
    try:
        import holoocean  # type: ignore
    except Exception:
        return {"version": None, "installed_packages": []}
    try:
        packages = list(holoocean.installed_packages())
    except Exception:
        packages = []
    return {
        "version": getattr(holoocean, "__version__", None),
        "installed_packages": packages,
    }


def _file_sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _source_tree_sha256() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[2]
    source_root = root / "marine_race_arena"
    paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".json"}
    )
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _parse_modes(values: Iterable[str]) -> List[str]:
    modes = []
    for value in values:
        normalized = value.strip().lower()
        if normalized not in INTER_VEHICLE_MODES:
            raise argparse.ArgumentTypeError(
                f"inter-vehicle mode must be one of {', '.join(INTER_VEHICLE_MODES)}"
            )
        modes.append(normalized)
    return modes


def _parse_conditions(values: Iterable[str]) -> List[str]:
    conditions = []
    for value in values:
        normalized = value.strip().lower()
        if normalized not in CONDITIONS:
            raise argparse.ArgumentTypeError(
                f"condition must be one of {', '.join(CONDITIONS)}"
            )
        conditions.append(normalized)
    return conditions


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default=DEFAULT_TRACK)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--team-size", type=int, default=DEFAULT_TEAM_SIZE)
    parser.add_argument("--start-gap-s", type=float, default=DEFAULT_START_GAP_S)
    parser.add_argument("--lateral-offset-m", type=float, default=DEFAULT_LATERAL_OFFSET_M)
    parser.add_argument("--team-id", default=DEFAULT_TEAM_ID)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--inter-vehicle-modes",
        nargs="+",
        default=["diagnostic"],
        help="One or more modes: diagnostic penalize.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS),
        help="Subset of conditions to run: no_coordination leader_follower.",
    )
    parser.add_argument(
        "--min-gate-gap",
        type=int,
        default=1,
        help=(
            "Leader-follower yield margin in locally estimated gates. Default 1 "
            "(recommended); 2 is the conservative comparison margin."
        ),
    )
    parser.add_argument(
        "--comms-packet-loss-prob",
        type=float,
        default=0.0,
        help="Seeded per-link acoustic packet loss for the coordinated condition (ablation).",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-participant-states", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    args.inter_vehicle_modes = _parse_modes(args.inter_vehicle_modes)
    args.conditions = _parse_conditions(args.conditions)
    args.invocation_argv = list(argv if argv is not None else sys.argv[1:])
    try:
        report = run_validation(args)
    except ValueError as exc:
        print(f"Coordination validation setup failed: {exc}", file=sys.stderr)
        return 1
    return 0 if report["all_runs_executed"] and report["artifact_contract_audit"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
