"""Diagnose the HoloOcean marine race adapter against an installed HoloOcean package."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marine_race_arena.adapters.holoocean_adapter import HoloOceanRaceAdapter
from marine_race_arena.adapters.visual_spawner import HoloOceanVisualSpawner
from marine_race_arena.arena.arena_builder import ArenaBuilder
from marine_race_arena.config.loader import load_track_config
from marine_race_arena.participants.participant import RaceParticipant


SENSOR_CHECKS = [
    "PoseSensor",
    "DepthSensor",
    "IMUSensor",
    "DVLSensor",
    "VelocitySensor",
    "CollisionSensor",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, help="Path to track JSON.")
    parser.add_argument("--headless", action="store_true", help="Hide viewport when HoloOcean supports it.")
    parser.add_argument("--ticks", type=int, default=40, help="Number of diagnostic ticks after the forward command.")
    parser.add_argument(
        "--skip-gate-visuals",
        action="store_true",
        help="Do not spawn runtime gate bars during the diagnostic.",
    )
    parser.add_argument(
        "--print-gate-bars",
        action="store_true",
        help="Print every generated gate bar transform and spawn method.",
    )
    args = parser.parse_args(argv)

    summary = {
        "holoocean_imported": False,
        "custom_scenario_initialized": False,
        "prebuilt_scenario_initialized": False,
        "blue_rov_spawned": False,
        "rover_moved": False,
        "gate_visual_method": "not_attempted",
        "physical_current_coupling_active": False,
        "failures": [],
        "warnings": [],
    }

    env = None
    adapter = None
    try:
        import holoocean

        summary["holoocean_imported"] = True
        print(f"HoloOcean module: {holoocean}")
        print(f"HoloOcean path: {getattr(holoocean, '__file__', None)}")
        print(f"HoloOcean version: {getattr(holoocean, '__version__', 'unknown')}")

        config = load_track_config(args.track)
        arena = ArenaBuilder(config).build()
        participants = _participants_from_config(config)
        participant_id = next(iter(participants))

        adapter = HoloOceanRaceAdapter(config, arena, headless=args.headless)
        adapter.initialize()
        adapter._participants = participants  # Diagnostic uses adapter builders without starting twice.
        adapter._states = {
            pid: adapter_state_from_participant(participant)
            for pid, participant in participants.items()
        }

        candidates = adapter._environment_candidates()
        print(f"Environment candidates: {candidates}")
        agent_config = adapter._build_agent_config(participants[participant_id], is_main_agent=True)
        print("Generated BlueROV2 agent config:")
        print(json.dumps(_json_safe(agent_config), indent=2, sort_keys=True))
        print(f"Generated gate bars: {sum(len(vg.bars) for vg in arena.visual_gates)}")

        env, method, environment_name, scenario = _initialize_environment(
            holoocean, adapter, candidates, show_viewport=not args.headless
        )
        adapter.env = env
        adapter._active_environment_name = environment_name
        adapter.visual_spawner = HoloOceanVisualSpawner(env)
        summary["custom_scenario_initialized"] = method == "custom_scenario_cfg"
        summary["prebuilt_scenario_initialized"] = method == "prebuilt_scenario"
        agent_obj = getattr(env, "agents", {}).get(participant_id) if hasattr(env, "agents") else None
        summary["blue_rov_spawned"] = bool(agent_obj is not None and "BlueROV2" in repr(agent_obj))
        if not summary["blue_rov_spawned"]:
            summary["warnings"].append(
                "Configured BlueROV2 participant was not found in env.agents; prebuilt scenarios may use another vehicle."
            )
        print(f"Initialized environment: {environment_name} via {method}")
        print(f"Configured agent object: {agent_obj}")
        print("Scenario used:")
        print(json.dumps(_json_safe(scenario), indent=2, sort_keys=True))

        raw_state = env.reset()
        adapter._raw_state = raw_state if isinstance(raw_state, dict) else {}
        adapter._refresh_states_from_raw()
        print(f"Raw state top-level keys: {list(adapter._raw_state.keys())}")
        sensors = _sensor_dict(adapter._raw_state, participant_id)
        print(f"Available sensor keys: {list(sensors.keys())}")
        for sensor_name in SENSOR_CHECKS:
            present = sensor_name in sensors
            print(f"{sensor_name}: {'present' if present else 'missing'}")
            summary[f"{sensor_name}_available"] = present
            if sensor_name == "CollisionSensor" and not present:
                summary["warnings"].append("CollisionSensor was configured but not present in state.")
        print(f"Initial collision state: {_collision_value(sensors)}")

        if args.skip_gate_visuals:
            summary["gate_visual_method"] = "skipped"
            summary["gate_bars_spawned"] = 0
            print("Gate visual spawning: skipped by --skip-gate-visuals")
            if args.print_gate_bars:
                _print_gate_bars(arena.visual_gates, "skipped")
        else:
            spawner = HoloOceanVisualSpawner(env)
            spawner.spawn_gate_bars([bar for visual_gate in arena.visual_gates for bar in visual_gate.bars])
            spawned_state = env.tick(num_ticks=1)
            if isinstance(spawned_state, dict):
                adapter._raw_state = spawned_state
            summary["gate_visual_method"] = spawner.report.method
            summary["gate_bars_spawned"] = spawner.report.spawned_bar_count
            print(f"Gate visual spawn report: {spawner.report}")
            print(f"Collision state after gate spawning: {adapter.get_collision_state(participant_id)}")
            if args.print_gate_bars:
                _print_spawned_gate_props(spawner.spawned_props)

        adapter.physical_current_coupling_active = callable(getattr(env, "set_ocean_currents", None))
        adapter.current_coupling_method = (
            "env.set_ocean_currents(agent_name, velocity)"
            if adapter.physical_current_coupling_active
            else "unavailable: environment has no set_ocean_currents method"
        )
        summary["physical_current_coupling_active"] = adapter.physical_current_coupling_active
        summary["current_coupling_method"] = adapter.current_coupling_method
        print(f"Current coupling: {adapter.current_coupling_method}")

        initial_state = adapter.get_participant_state(participant_id)
        print(f"Initial pose: position={initial_state.position}, rotation={initial_state.rotation_rpy_deg}")

        print("Testing zero action...")
        adapter.apply_command(participant_id, {"thrusters": [0.0] * 8}, "thrusters")
        adapter.step(1.0 / 30.0)
        print(f"Collision state after zero action: {adapter.get_collision_state(participant_id)}")

        print("Testing small safe forward action...")
        adapter.apply_command(participant_id, {"surge": 0.35, "sway": 0.0, "heave": 0.0, "yaw": 0.0}, "high_level")
        first_collision = None
        for _ in range(max(20, min(50, args.ticks))):
            adapter.step(1.0 / 30.0)
            if adapter.get_collision_state(participant_id) and first_collision is None:
                state = adapter.get_participant_state(participant_id)
                first_collision = {
                    "time_s": adapter.get_current_time(),
                    "position": state.position,
                }

        final_state = adapter.get_participant_state(participant_id)
        print(f"Final pose: position={final_state.position}, rotation={final_state.rotation_rpy_deg}")
        moved_distance = _distance(initial_state.position, final_state.position)
        summary["rover_moved"] = moved_distance > 0.02
        summary["moved_distance_m"] = moved_distance
        summary["collision_during_command_test"] = first_collision is not None
        summary["first_collision_during_command_test"] = first_collision
        print(f"Collision during command test: {first_collision}")
        print(f"Moved distance: {moved_distance:.4f} m")
        if not summary["rover_moved"]:
            summary["warnings"].append("The rover did not move more than 0.02 m under the diagnostic command.")

    except Exception as exc:
        summary["failures"].append(f"{type(exc).__name__}: {exc}")
        print(f"FAIL: {type(exc).__name__}: {exc}")
    finally:
        if env is not None:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    status = "PASS"
    if summary["failures"]:
        status = "FAIL"
    elif summary["warnings"] or not summary.get("rover_moved"):
        status = "WARN"
    print("Diagnostic summary:")
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    print(f"DIAGNOSTIC {status}")
    return 1 if status == "FAIL" else 0


def _participants_from_config(config: Any) -> Dict[str, RaceParticipant]:
    participants: Dict[str, RaceParticipant] = {}
    for participant_config in config.participants:
        spawn = participant_config.spawn or {}
        position = tuple(float(value) for value in spawn.get("position", config.start.position))
        rotation = tuple(float(value) for value in spawn.get("rotation_rpy_deg", config.start.rotation_rpy_deg))
        participants[participant_config.id] = RaceParticipant(
            config=participant_config,
            controller=object(),
            position=position,
            rotation_rpy_deg=rotation,
        )
    return participants


def adapter_state_from_participant(participant: RaceParticipant):
    from marine_race_arena.adapters.base import AdapterParticipantState

    return AdapterParticipantState(
        participant_id=participant.id,
        position=participant.position,
        rotation_rpy_deg=participant.rotation_rpy_deg,
        raw_sensors={},
    )


def _initialize_environment(holoocean: Any, adapter: HoloOceanRaceAdapter, candidates: list[str], show_viewport: bool):
    failures = []
    for environment_name in candidates:
        scenario = adapter._build_scenario(environment_name)
        try:
            env = holoocean.make(
                scenario_cfg=scenario,
                show_viewport=show_viewport,
                ticks_per_sec=scenario.get("ticks_per_sec", 30),
                frames_per_sec=scenario.get("frames_per_sec", True),
            )
            return env, "custom_scenario_cfg", environment_name, scenario
        except Exception as exc:
            failures.append(f"{environment_name} custom scenario failed: {type(exc).__name__}: {exc}")
        try:
            env = holoocean.make(environment_name, show_viewport=show_viewport, ticks_per_sec=30, frames_per_sec=True)
            return env, "prebuilt_scenario", environment_name, {"scenario_name": environment_name}
        except Exception as exc:
            failures.append(f"{environment_name} prebuilt scenario failed: {type(exc).__name__}: {exc}")
    raise RuntimeError("No HoloOcean environment initialized. " + " | ".join(failures))


def _sensor_dict(raw_state: Mapping[str, Any], participant_id: str) -> Dict[str, Any]:
    value = raw_state.get(participant_id)
    if isinstance(value, dict):
        return value
    if isinstance(raw_state, dict):
        return dict(raw_state)
    return {}


def _collision_value(sensors: Mapping[str, Any]) -> Any:
    value = sensors.get("CollisionSensor")
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _print_gate_bars(visual_gates: Any, method: str) -> None:
    print("Gate bar transforms:")
    print("gate_id | bar_id | position_xyz | rotation_rpy_deg | scale_xyz_m | spawn_method")
    for visual_gate in visual_gates:
        for bar in visual_gate.bars:
            print(
                f"{visual_gate.gate_id} | {bar.id} | "
                f"{_format_vector(bar.position)} | "
                f"{_format_vector(bar.rotation_rpy_deg)} | "
                f"{_format_vector(bar.dimensions_m)} | {method}"
            )


def _print_spawned_gate_props(spawned_props: Any) -> None:
    print("Spawned gate prop transforms:")
    print(
        "gate_id | prop_id | source_bar_id | part | position_xyz | "
        "logical_rotation_rpy_deg | spawn_rotation_deg | scale_xyz_m | spawn_method"
    )
    for prop in spawned_props:
        print(
            f"{prop['gate_id']} | {prop['id']} | {prop['source_bar_id']} | {prop['part']} | "
            f"{_format_vector(prop['position'])} | "
            f"{_format_vector(prop['rotation_rpy_deg'])} | "
            f"{_format_vector(prop.get('spawn_rotation_deg', prop['rotation_rpy_deg']))} | "
            f"{_format_vector(prop['dimensions_m'])} | {prop['method']}"
        )


def _format_vector(values: Any) -> str:
    return "[" + ", ".join(f"{float(value):.3f}" for value in values) + "]"


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def _json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    raise SystemExit(main())
