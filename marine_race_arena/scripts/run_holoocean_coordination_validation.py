"""Minimal HoloOcean validation for heterogeneous leader-follower coordination.

This script intentionally leaves the paper-oriented deterministic comparison
untouched. It reuses the production race loop with the real HoloOcean adapter,
fallback disabled, official observations, no currents and no obstacles.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
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
    AcousticBaselineController,
    SmoothGateBaselineController,
)
from marine_race_arena.participants.controller_interface import BaseController
from marine_race_arena.participants.participant import RaceParticipant
from marine_race_arena.referee.logger import RaceLogger
from marine_race_arena.referee.referee import Referee
from marine_race_arena.scripts.run_algorithm_comparison import _TracingController
from marine_race_arena.scripts.run_marine_race import (
    _offset_spawn_position,
    _race_info,
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


def _make_team(size: int, *, coordinated: bool) -> List[BaseController]:
    bases: List[BaseController] = [SmoothGateBaselineController()] + [
        AcousticBaselineController() for _ in range(size - 1)
    ]
    if not coordinated:
        return [_TracingController(base) for base in bases]
    return [_TracingController(LeaderFollowerController(base_controller=base)) for base in bases]


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
        AcousticCommsChannel(CommsConfig(enabled=True), seed=seed)
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
        race_info = _race_info(config, adapter.name, "none")
        for participant in participants.values():
            participant.controller.reset(
                race_info | {"initial_target_gate_id": referee.expected_gate_id(participant.id)}
            )

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
        summary["adapter"] = adapter.name
        summary["motion_compensation"] = "none"
        summary["inter_vehicle_collision_mode"] = inter_vehicle_collision_mode
        if comms_channel is not None:
            summary["comms"] = comms_channel.summary()
        summary["validation_setup"] = {
            "track": track_path,
            "adapter": "holoocean",
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


def run_validation(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: Dict[str, Any] = {}
    for mode in args.inter_vehicle_modes:
        runs[mode] = {}
        for seed in args.seeds:
            runs[mode][str(seed)] = {}
            for condition in CONDITIONS:
                coordinated = condition == "leader_follower"
                controllers = _make_team(args.team_size, coordinated=coordinated)
                run_dir = output_dir / mode / f"seed_{seed}" / condition
                run_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"Running {condition} mode={mode} seed={seed} "
                    f"team={args.team_size} start_gap={args.start_gap_s:g}s",
                    flush=True,
                )
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
                    team_id=f"{args.team_id}_{mode}_{condition}_seed{seed}",
                    headless=args.headless,
                    log_participant_states=args.log_participant_states,
                )
                result["condition"] = condition
                result["seed"] = int(seed)
                result["inter_vehicle_collision_mode"] = mode
                result["comms_enabled"] = coordinated
                result["controllers"] = _controller_labels(args.team_size, coordinated=coordinated)
                result["metrics"] = _metrics_from_summary(result.get("summary"))
                runs[mode][str(seed)][condition] = result
    report = {
        "track": args.track,
        "adapter": "holoocean",
        "official": True,
        "benchmark_task": "clean_gate",
        "current_profile": "none",
        "obstacles": "none",
        "team_size": args.team_size,
        "start_gap_s": args.start_gap_s,
        "lateral_offset_m": args.lateral_offset_m,
        "dt": args.dt,
        "duration_s": args.duration_s,
        "leader": "smooth_gate_baseline",
        "followers": "acoustic_baseline",
        "leader_follower_min_gate_gap": 2,
        "runs": runs,
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
    bases = ["smooth_gate_baseline"] + ["acoustic_baseline" for _ in range(size - 1)]
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
    rows = []
    if isinstance(participants, list):
        for participant in participants:
            if not isinstance(participant, Mapping):
                continue
            rows.append(
                {
                    "participant_id": participant.get("participant_id"),
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
                }
            )
    comms = summary.get("comms") if isinstance(summary.get("comms"), Mapping) else None
    return {
        "all_rovers_finished": team.get("all_rovers_finished"),
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
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# HoloOcean leader-follower coordination validation",
        "",
        (
            f"Track: `{report['track']}` | adapter: `holoocean` | official mode | "
            "current profile: `none` | obstacles: `none`."
        ),
        (
            f"Team: {report['team_size']} rovers, start gap {report['start_gap_s']} s, "
            f"lateral offset {report['lateral_offset_m']} m. "
            "Leader `smooth_gate_baseline`, followers `acoustic_baseline`; "
            "`leader_follower` uses default `min_gate_gap=2` and comms enabled."
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
                result = conditions.get(condition) or {}
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
            "| Seed | Condition | Rover | Status | Gates | Official time (s) | Gate/world collisions | Inter-vehicle events | OOB | Stuck |"
        )
        lines.append("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for seed, conditions in seeds.items():
            for condition in CONDITIONS:
                metrics = (conditions.get(condition) or {}).get("metrics") or {}
                for row in metrics.get("rovers") or []:
                    lines.append(
                        f"| {seed} | {condition.replace('_', ' ')} | {row.get('participant_id')} | "
                        f"{row.get('status')} | {row.get('completed_gates')}/{row.get('expected_gates')} | "
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
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-participant-states", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    args.inter_vehicle_modes = _parse_modes(args.inter_vehicle_modes)
    run_validation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
