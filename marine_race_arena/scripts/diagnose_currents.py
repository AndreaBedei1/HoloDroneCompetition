"""Diagnose configured marine currents and optional HoloOcean drift."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marine_race_arena.adapters import HoloOceanRaceAdapter, RaceAdapterError, RaceAdapterUnavailable
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.config.schema import TrackConfig, Vector3
from marine_race_arena.participants.participant import RaceParticipant


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, help="Path to a marine race track JSON file.")
    parser.add_argument(
        "--adapter",
        choices=("fallback", "holoocean"),
        default="fallback",
        help="fallback only evaluates the analytic field; holoocean also runs a stopped-rover drift check.",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="HoloOcean drift duration in seconds.")
    parser.add_argument("--dt", type=float, default=0.1, help="HoloOcean drift timestep.")
    parser.add_argument("--zero-command", action="store_true", help="Command zero thrust during drift.")
    parser.add_argument("--headless", action="store_true", help="Request headless HoloOcean mode.")
    parser.add_argument("--drift-threshold-m", type=float, default=0.05, help="Minimum drift for PASS.")
    args = parser.parse_args(argv)

    config = load_track_config(args.track)
    arena = ArenaBuilder(config).build()

    print(f"Track: {config.race.name}")
    print(f"Track file: {args.track}")
    print(f"Configured currents: {json.dumps([_current_to_dict(current) for current in config.currents], indent=2)}")

    samples = _sample_points(config)
    nonzero_samples = 0
    for label, point in samples.items():
        velocity = arena.current_manager.get_current_at(point, 0.0)
        magnitude = _norm(velocity)
        if magnitude > 1e-6:
            nonzero_samples += 1
        print(
            f"Sample {label:>18}: point={_round_vec(point)} "
            f"velocity={_round_vec(velocity)} speed={magnitude:.4f} m/s"
        )

    if not config.currents:
        if nonzero_samples == 0:
            print("PASS: no-current track evaluates to zero current at all sample points.")
        else:
            print("FAIL: no-current track produced nonzero current.")
            return 1
    elif nonzero_samples > 0:
        print("PASS: configured current field is nonzero at one or more sample points.")
    else:
        print("FAIL: configured currents were present but all sample points evaluated to zero.")
        return 1

    if args.adapter == "fallback":
        print("HoloOcean drift check: skipped because --adapter fallback was selected.")
        return 0

    return _run_holoocean_drift_check(config, arena, args)


def _run_holoocean_drift_check(config: TrackConfig, arena: Any, args: argparse.Namespace) -> int:
    participants = _build_participants(config)
    participant_id = next(iter(participants))
    adapter = HoloOceanRaceAdapter(config, arena, headless=args.headless)
    try:
        adapter.initialize()
        adapter.spawn_participants(participants)
        adapter.reset()
        print(f"HoloOcean environment: {adapter.active_environment_name}")
        print(f"Physical current coupling active: {adapter.physical_current_coupling_active}")
        print(f"Current coupling method: {adapter.current_coupling_method}")
        if not adapter.physical_current_coupling_active:
            print("WARN: HoloOcean adapter cannot physically apply currents in this installation.")
            return 0

        initial = adapter.get_participant_state(participant_id).position
        steps = max(1, int(math.ceil(args.duration / args.dt)))
        for _ in range(steps):
            command: Dict[str, Any]
            if args.zero_command:
                command = {"thrusters": [0.0] * 8}
                control_mode = "thrusters"
            else:
                command = {"surge": 0.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}
                control_mode = participants[participant_id].config.control_mode
            adapter.apply_command(participant_id, command, control_mode)
            adapter.step(args.dt)

        final = adapter.get_participant_state(participant_id).position
        displacement = _distance(initial, final)
        print(f"Initial pose position: {_round_vec(initial)}")
        print(f"Final pose position:   {_round_vec(final)}")
        print(f"Stopped-rover displacement: {displacement:.4f} m")
        if config.currents and displacement >= args.drift_threshold_m:
            print("PASS: stopped rover drifted under the configured HoloOcean current.")
            return 0
        if config.currents:
            print(
                "WARN: physical current coupling was active, but stopped-rover drift was below "
                f"{args.drift_threshold_m:.3f} m."
            )
            return 0
        print("PASS: no-current track did not require a drift check.")
        return 0
    except (RaceAdapterError, RaceAdapterUnavailable, Exception) as exc:
        print(f"FAIL: HoloOcean current diagnostic failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        adapter.close()


def _build_participants(config: TrackConfig) -> Dict[str, RaceParticipant]:
    participants: Dict[str, RaceParticipant] = {}
    for participant_config in config.participants:
        spawn = participant_config.spawn or {}
        position = _vector3(spawn.get("position", config.start.position))
        rotation = _vector3(spawn.get("rotation_rpy_deg", config.start.rotation_rpy_deg))
        participants[participant_config.id] = RaceParticipant(
            config=participant_config,
            controller=None,
            position=position,
            rotation_rpy_deg=rotation,
        )
    return participants


def _sample_points(config: TrackConfig) -> Dict[str, Vector3]:
    samples: Dict[str, Vector3] = {"start": config.start.position}
    if config.gates:
        samples["first_gate"] = config.gates[0].position
        samples["last_gate"] = config.gates[-1].position
    for index, current in enumerate(config.currents):
        center = current.params.get("center")
        if isinstance(center, list) and len(center) == 3:
            samples[f"current_{index}_center"] = _vector3(center)
    return samples


def _current_to_dict(current: Any) -> Dict[str, Any]:
    payload = dict(current.params)
    payload["type"] = current.type
    return payload


def _vector3(value: Any) -> Vector3:
    return (float(value[0]), float(value[1]), float(value[2]))


def _round_vec(value: Vector3) -> list[float]:
    return [round(float(component), 4) for component in value]


def _norm(value: Vector3) -> float:
    return math.sqrt(value[0] ** 2 + value[1] ** 2 + value[2] ** 2)


def _distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


if __name__ == "__main__":
    raise SystemExit(main())
