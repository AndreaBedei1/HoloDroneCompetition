"""Reproducible algorithm-comparison harness for Marine Race Arena.

This script backs two claims in the paper and release notes with numbers that
anyone can reproduce without the HoloOcean engine. It runs everything on the
engine-free kinematic fallback adapter, which is fully deterministic, and it
never changes track geometry, gate sizes, referee rules, scoring, current
profiles, or the official observation filter. The referee's inter-vehicle
proximity thresholds are left at their official defaults (0.8 m horizontal,
0.75 m vertical).

Two comparisons are produced:

1. Single-rover gate controllers -- the fast ``acoustic_baseline`` beacon
   controller versus the new, conservative ``smooth_gate_baseline``. This shows
   the benchmark separating two legal controllers by official time and by command
   smoothness rather than only by code.

2. Fleet coordination -- a staggered heterogeneous team (a slower leader with
   faster followers) run first with every rover racing independently and then with
   the leader-follower coordinator over the acoustic communication channel. This
   shows that coordination drives the team's inter-vehicle proximity events down to
   the single-rover level (zero) while the whole team still completes the gate
   sequence.

Usage::

    python -m marine_race_arena.scripts.run_algorithm_comparison
    python -m marine_race_arena.scripts.run_algorithm_comparison --track <track.json> \
        --output-dir results/benchmarks/algorithm_comparison
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from marine_race_arena.adapters.fallback_adapter import FallbackRaceAdapter
from marine_race_arena.arena.acoustic_comms import AcousticCommsChannel, CommsConfig
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.controllers.leader_follower import LeaderFollowerController
from marine_race_arena.controllers.official_baselines import (
    AcousticBaselineController,
    SmoothGateBaselineController,
)
from marine_race_arena.participants.controller_interface import BaseController
from marine_race_arena.participants.participant import RaceParticipant
from marine_race_arena.referee.referee import Referee
from marine_race_arena.scripts.run_marine_race import (
    _offset_spawn_position,
    _race_info,
    _run_race_loop,
    _staggered_lateral_offsets,
    _vector3,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_TRACK = "marine_race_arena/tracks/marine_race_horseshoe_bay.json"
DEFAULT_OUTPUT_DIR = "results/benchmarks/algorithm_comparison"

# Fleet scenario used for the coordination comparison. The leader is the slower
# controller and the followers are faster, so the followers would otherwise catch
# and overtake the leader near the gates. The 1.5 m lateral spacing is the CLI
# staggered-start default and stays above the referee proximity threshold, so the
# start line itself never triggers a proximity event. Several team sizes are swept
# to show the coordination benefit is not tied to one fleet size.
FLEET_SIZES = (3, 4, 5)
HEADLINE_FLEET_SIZE = 4
FLEET_START_GAP_S = 8.0
FLEET_LATERAL_OFFSET_M = 1.5
# The fleet comparison scores inter-vehicle proximity as a team penalty so the
# penalized-time column reflects the collisions each condition actually incurs.
FLEET_INTER_VEHICLE_MODE = "penalize"

# Ablations run on the four-rover team: the yield margin (a one-gate margin is
# too tight) and robustness to acoustic packet loss.
ABLATION_SIZE = 4
GAP_ABLATION = (1, 2)
LOSS_ABLATION_PROB = 0.2


class _TracingController(BaseController):
    """Wrap a controller to count steps and accumulate command smoothness.

    ``mean_command_change`` is the average per-step change in the surge and yaw
    commands, a simple proxy for how smooth (low-jerk) the controller's motion is.
    The wrapper forwards the wrapped controller's ``uses_ground_truth`` flag so the
    official-mode honesty check is unaffected.
    """

    def __init__(self, inner: BaseController) -> None:
        self.inner = inner
        self.uses_ground_truth = bool(getattr(inner, "uses_ground_truth", False))
        self.debug_only = bool(getattr(inner, "debug_only", False))
        self._prev: Optional[Dict[str, float]] = None
        self.steps = 0
        self.command_change_sum = 0.0

    def reset(self, race_info: Dict[str, Any]) -> None:
        self.inner.reset(race_info)
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

    @property
    def mean_command_change(self) -> float:
        return self.command_change_sum / max(1, self.steps - 1)


def simulate_fleet(
    track_path: str,
    controllers: List[BaseController],
    *,
    start_gap_s: float = FLEET_START_GAP_S,
    lateral_offset_m: float = FLEET_LATERAL_OFFSET_M,
    duration_s: float = 400.0,
    dt: float = 0.1,
    comms_enabled: bool = False,
    comms_packet_loss_prob: float = 0.0,
    seed: int = 0,
    inter_vehicle_collision_mode: str = "diagnostic",
    team_id: str = "comparison_team",
) -> Dict[str, Any]:
    """Run a staggered fleet of the given controllers on the fallback adapter.

    ``controllers`` supplies one controller per rover in start order; rover *i* is
    released ``i * start_gap_s`` seconds after the first and spawns at the standard
    alternating lateral offset. Returns the referee summary (with a ``comms`` block
    when the channel is enabled). This mirrors what ``run_marine_race`` does, but
    with an explicit per-rover controller list so heterogeneous teams can be built.
    """
    config = load_track_config(Path(track_path))
    config = replace(config, race=replace(config.race, max_duration_s=duration_s, official_mode=True))
    base = config.participants[0]
    num_rovers = len(controllers)
    offsets = _staggered_lateral_offsets(num_rovers, spacing_m=lateral_offset_m)
    base_position = _vector3(base.spawn["position"])
    base_rotation = _vector3(base.spawn["rotation_rpy_deg"])

    participant_configs = []
    for index in range(num_rovers):
        spawn = dict(base.spawn)
        spawn["position"] = list(_offset_spawn_position(config, base_position, base_rotation[2], offsets[index]))
        spawn["rotation_rpy_deg"] = list(base_rotation)
        spawn["start_delay_s"] = float(index) * float(start_gap_s)
        participant_configs.append(
            replace(base, id=f"bluerov2_{index + 1:02d}", spawn=spawn, start_delay_s=float(index) * float(start_gap_s))
        )
    config = replace(config, participants=participant_configs)

    # Seed the arena (and thus the beacon-noise RNG) so the whole run is
    # deterministic and the reported numbers reproduce exactly.
    arena = ArenaBuilder(config, seed=seed).build()
    participants = {
        participant_config.id: RaceParticipant(
            config=participant_config,
            controller=controllers[index],
            position=tuple(participant_config.spawn["position"]),
            rotation_rpy_deg=tuple(participant_config.spawn["rotation_rpy_deg"]),
        )
        for index, participant_config in enumerate(config.participants)
    }

    adapter = FallbackRaceAdapter(config, arena)
    adapter.initialize()
    adapter.spawn_participants(participants)
    adapter.reset()
    referee = Referee(
        config,
        arena.gate_map,
        arena.bounds,
        inter_vehicle_collision_mode=inter_vehicle_collision_mode,
        team_id=team_id,
    )
    referee.register_participants(participants.keys())
    race_info = _race_info(config, adapter.name)
    for participant in participants.values():
        participant.controller.reset(
            race_info | {"initial_target_gate_id": referee.expected_gate_id(participant.id)}
        )

    comms_channel = (
        AcousticCommsChannel(CommsConfig(enabled=True, packet_loss_prob=comms_packet_loss_prob), seed=seed)
        if comms_enabled
        else None
    )
    summary = _run_race_loop(
        config=config,
        arena=arena,
        referee=referee,
        adapter=adapter,
        participants=participants,
        dt=dt,
        gate_timeout_s=None,
        log_participant_states=False,
        comms_channel=comms_channel,
    )
    if comms_channel is not None:
        summary["comms"] = comms_channel.summary()
    return summary


def _rover_row(participant: Mapping[str, Any], controller: Optional[BaseController]) -> Dict[str, Any]:
    row = {
        "participant_id": participant.get("participant_id"),
        "status": participant.get("status"),
        "completed_gates": participant.get("completed_gates"),
        "official_time_s": _round(participant.get("official_time_s")),
        "penalized_time_s": _round(participant.get("penalized_time_s")),
        "collisions": participant.get("collisions"),
        "obstacle_collisions": participant.get("obstacle_collisions"),
        "inter_vehicle_events": participant.get("involved_inter_vehicle_collisions", 0),
        "out_of_bounds_events": participant.get("out_of_bounds_events"),
        "stuck_events": participant.get("stuck_events"),
    }
    inner = controller.inner if isinstance(controller, _TracingController) else controller
    if isinstance(controller, _TracingController):
        row["mean_command_change"] = round(controller.mean_command_change, 4)
    if isinstance(inner, LeaderFollowerController):
        row["hold_steps"] = inner.hold_steps
    return row


def _team_row(summary: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    team = summary.get("team_summary")
    if not team:
        return None
    return {
        "rover_count": team.get("rover_count"),
        "all_rovers_finished": team.get("all_rovers_finished"),
        "total_completed_gates": team.get("total_completed_gates"),
        "expected_total_gates": team.get("expected_total_gates"),
        "total_inter_vehicle_collisions": team.get("total_inter_vehicle_collisions"),
        "total_collisions": team.get("total_collisions"),
        "team_elapsed_time_s": _round(team.get("team_elapsed_time_s")),
        "team_penalized_time_s": _round(team.get("team_penalized_time_s")),
    }


def _round(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), 1)
    return value


def run_single_rover_comparison(track_path: str, duration_s: float) -> Dict[str, Any]:
    """Compare the fast acoustic baseline against the new conservative controller."""
    results: Dict[str, Any] = {}
    for name, factory in (
        ("acoustic_baseline", AcousticBaselineController),
        ("smooth_gate_baseline", SmoothGateBaselineController),
    ):
        controller = _TracingController(factory())
        summary = simulate_fleet(
            track_path,
            [controller],
            duration_s=duration_s,
            inter_vehicle_collision_mode="off",
            team_id=f"single_{name}",
        )
        results[name] = _rover_row(summary["participants"][0], controller)
    return results


def _make_team(size: int, coordinated: bool, min_gate_gap: Optional[int] = None) -> List[BaseController]:
    """A slower leader followed by faster rovers, optionally coordinated.

    The underlying navigation is identical in both conditions (same leader/follower
    base controllers); coordination only adds the leader-follower yielding layer and
    the heartbeat traffic, so the comparison isolates the coordination policy.
    ``min_gate_gap`` overrides the default yield margin when set (used by the gap
    ablation); ``None`` keeps the controller default.
    """
    bases: List[BaseController] = [SmoothGateBaselineController()] + [
        AcousticBaselineController() for _ in range(size - 1)
    ]
    if not coordinated:
        return [_TracingController(base) for base in bases]
    return [
        _TracingController(LeaderFollowerController(base_controller=base, min_gate_gap=min_gate_gap))
        for base in bases
    ]


def _run_fleet_condition(track_path: str, duration_s: float, size: int, coordinated: bool, comms: bool) -> Dict[str, Any]:
    controllers = _make_team(size, coordinated)
    summary = simulate_fleet(
        track_path,
        controllers,
        duration_s=duration_s,
        comms_enabled=comms,
        inter_vehicle_collision_mode=FLEET_INTER_VEHICLE_MODE,
        team_id=f"team_{'coord' if coordinated else 'raw'}_{size}",
    )
    rows = [
        _rover_row(participant, _controller_for(controllers, participant.get("participant_id")))
        for participant in summary["participants"]
    ]
    rows.sort(key=lambda row: row["participant_id"])
    return {"team": _team_row(summary), "rovers": rows, "comms": summary.get("comms")}


def run_fleet_comparison(track_path: str, duration_s: float) -> Dict[str, Any]:
    """Sweep team sizes, each without and with leader-follower coordination."""
    results: Dict[str, Any] = {}
    for size in FLEET_SIZES:
        results[str(size)] = {
            "no_coordination": _run_fleet_condition(track_path, duration_s, size, coordinated=False, comms=False),
            "leader_follower": _run_fleet_condition(track_path, duration_s, size, coordinated=True, comms=True),
        }
    return results


def run_ablations(track_path: str, duration_s: float) -> Dict[str, Any]:
    """Yield-margin and packet-loss ablations on the four-rover coordinated team."""
    gap: Dict[str, Any] = {}
    for delta in GAP_ABLATION:
        team = _make_team(ABLATION_SIZE, coordinated=True, min_gate_gap=delta)
        summary = simulate_fleet(
            track_path,
            team,
            duration_s=duration_s,
            comms_enabled=True,
            inter_vehicle_collision_mode=FLEET_INTER_VEHICLE_MODE,
            team_id=f"ablation_gap{delta}",
        )
        gap[str(delta)] = _team_row(summary)
    lossy = simulate_fleet(
        track_path,
        _make_team(ABLATION_SIZE, coordinated=True),
        duration_s=duration_s,
        comms_enabled=True,
        comms_packet_loss_prob=LOSS_ABLATION_PROB,
        inter_vehicle_collision_mode=FLEET_INTER_VEHICLE_MODE,
        team_id="ablation_loss",
    )
    return {
        "size": ABLATION_SIZE,
        "gap": gap,
        "packet_loss": {
            "prob": LOSS_ABLATION_PROB,
            "team": _team_row(lossy),
            "comms": lossy.get("comms"),
        },
    }


def _controller_for(controllers: List[BaseController], participant_id: Any) -> Optional[BaseController]:
    if not isinstance(participant_id, str) or not participant_id[-2:].isdigit():
        return None
    index = int(participant_id[-2:]) - 1
    if 0 <= index < len(controllers):
        return controllers[index]
    return None


def _markdown(track_path: str, single: Dict[str, Any], fleet: Dict[str, Any], ablation: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Marine Race Arena -- algorithm comparison")
    lines.append("")
    lines.append(f"Track: `{track_path}` | adapter: `fallback` (deterministic) | official mode.")
    lines.append("")
    lines.append("## 1. Single-rover gate controllers")
    lines.append("")
    lines.append("| Controller | Status | Gates | Official time (s) | Penalized (s) | OOB | Stuck | Mean cmd change |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, row in single.items():
        lines.append(
            f"| `{name}` | {row['status']} | {row['completed_gates']} | {row['official_time_s']} | "
            f"{row['penalized_time_s']} | {row['out_of_bounds_events']} | {row['stuck_events']} | "
            f"{row.get('mean_command_change')} |"
        )
    lines.append("")
    lines.append("## 2. Staggered fleet: no coordination vs leader-follower")
    lines.append("")
    lines.append(
        f"Heterogeneous team (a slower `smooth_gate_baseline` leader, faster `acoustic_baseline` "
        f"followers), {FLEET_START_GAP_S:.0f}s staggered start, {FLEET_LATERAL_OFFSET_M:.1f}m lateral "
        f"spacing, inter-vehicle mode `{FLEET_INTER_VEHICLE_MODE}`."
    )
    lines.append("")
    lines.append("| Team size | Condition | All finished | Team gates | Inter-vehicle events | Total collisions | Team elapsed (s) | Team penalized (s) |")
    lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for size in FLEET_SIZES:
        for label in ("no_coordination", "leader_follower"):
            team = fleet[str(size)][label]["team"]
            lines.append(
                f"| {size} | {label.replace('_', ' ')} | {team['all_rovers_finished']} | "
                f"{team['total_completed_gates']}/{team['expected_total_gates']} | "
                f"{team['total_inter_vehicle_collisions']} | {team['total_collisions']} | "
                f"{team['team_elapsed_time_s']} | {team['team_penalized_time_s']} |"
            )
    lines.append("")
    lines.append(f"### Per-rover detail (team size {HEADLINE_FLEET_SIZE})")
    lines.append("")
    lines.append("| Condition | Rover | Status | Gates | Official time (s) | Inter-vehicle events | Hold steps |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for label in ("no_coordination", "leader_follower"):
        for row in fleet[str(HEADLINE_FLEET_SIZE)][label]["rovers"]:
            lines.append(
                f"| {label.replace('_', ' ')} | {row['participant_id']} | {row['status']} | "
                f"{row['completed_gates']} | {row['official_time_s']} | {row['inter_vehicle_events']} | "
                f"{row.get('hold_steps', '-')} |"
            )
    lines.append("")
    lines.append(f"## 3. Ablations (team size {ablation['size']})")
    lines.append("")
    lines.append(
        "Yield margin: a two-gate margin removes all inter-vehicle events; a one-gate "
        "margin is too tight (the follower holds only a single gate of lead) and records "
        "more events than the uncoordinated team, because it does not preserve a full "
        "gate of physical spacing."
    )
    lines.append("")
    lines.append("| Yield margin | Inter-vehicle events | All finished | Team penalized (s) |")
    lines.append("| ---: | ---: | --- | ---: |")
    for delta, team in sorted(ablation["gap"].items()):
        lines.append(
            f"| $\\Delta g$ = {delta} | {team['total_inter_vehicle_collisions']} | "
            f"{team['all_rovers_finished']} | {team['team_penalized_time_s']} |"
        )
    loss = ablation["packet_loss"]
    dropped = (loss.get("comms") or {}).get("dropped_packet_loss")
    lines.append("")
    lines.append(
        f"Packet-loss robustness ($\\Delta g$=2, per-link loss {loss['prob']}): "
        f"{loss['team']['total_inter_vehicle_collisions']} inter-vehicle events, "
        f"all finished = {loss['team']['all_rovers_finished']} "
        f"({dropped} heartbeats dropped)."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    logging.disable(logging.WARNING)
    parser = argparse.ArgumentParser(description="Run the Marine Race Arena algorithm comparison.")
    parser.add_argument("--track", default=DEFAULT_TRACK, help="Track JSON to run the comparison on.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for the JSON and Markdown report.")
    parser.add_argument("--duration-s", type=float, default=400.0, help="Maximum race duration for each run.")
    args = parser.parse_args(argv)

    single = run_single_rover_comparison(args.track, args.duration_s)
    fleet = run_fleet_comparison(args.track, args.duration_s)
    ablation = run_ablations(args.track, args.duration_s)

    report = {
        "track": args.track,
        "adapter": "fallback",
        "fleet_scenario": {
            "sizes": list(FLEET_SIZES),
            "start_gap_s": FLEET_START_GAP_S,
            "lateral_offset_m": FLEET_LATERAL_OFFSET_M,
            "inter_vehicle_mode": FLEET_INTER_VEHICLE_MODE,
            "leader": "smooth_gate_baseline",
            "followers": "acoustic_baseline",
        },
        "single_rover": single,
        "fleet": fleet,
        "ablations": ablation,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = _markdown(args.track, single, fleet, ablation)
    (output_dir / "comparison.md").write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nWrote {output_dir / 'comparison.json'} and {output_dir / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
