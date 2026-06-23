from __future__ import annotations

import copy
import json
from pathlib import Path

from marine_race_arena.config.benchmark_tasks import (
    BENCHMARK_TASK_CLEAN_GATE,
    BENCHMARK_TASK_CURRENT_GATE,
    BENCHMARK_TASK_MULTI_ROV,
    BENCHMARK_TASK_OBSTACLE_GATE,
)
from marine_race_arena.config.loader import load_track_config, parse_track_config, with_benchmark_task
from marine_race_arena.config.validation import validate_track_config
from marine_race_arena.scripts.run_marine_race import _build_arg_parser


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


def test_example_tracks_declare_benchmark_tasks() -> None:
    clean_config = load_track_config(TRACK_DIR / "marine_race_horseshoe_bay.json")
    current_config = load_track_config(TRACK_DIR / "marine_race_mixed_endurance.json")

    assert clean_config.benchmark_task.mode == BENCHMARK_TASK_CLEAN_GATE
    assert current_config.benchmark_task.mode == BENCHMARK_TASK_CURRENT_GATE
    assert validate_track_config(clean_config).errors == []
    assert validate_track_config(current_config).errors == []


def test_legacy_track_without_task_keeps_existing_optional_currents_behavior() -> None:
    raw = _raw_track("marine_race_horseshoe_bay.json")
    raw.pop("benchmark_task", None)
    raw["currents"] = [{"type": "constant", "velocity": [0.25, 0.0, 0.0]}]

    config = parse_track_config(raw)
    result = validate_track_config(config)

    assert config.benchmark_task.mode is None
    assert result.errors == []


def test_clean_gate_rejects_currents_and_obstacles() -> None:
    raw = _raw_track("marine_race_horseshoe_bay.json")
    raw["benchmark_task"] = BENCHMARK_TASK_CLEAN_GATE
    raw["currents"] = [{"type": "constant", "velocity": [0.75, 0.0, 0.0]}]
    raw["obstacles"] = [_static_obstacle()]

    result = validate_track_config(parse_track_config(raw))

    assert any("clean_gate must not configure currents" in error for error in result.errors)
    assert any("clean_gate must not activate obstacles" in error for error in result.errors)


def test_current_gate_requires_a_strong_current() -> None:
    raw = _raw_track("marine_race_horseshoe_bay.json")
    raw["benchmark_task"] = {"mode": BENCHMARK_TASK_CURRENT_GATE}
    raw["currents"] = [{"type": "constant", "velocity": [0.1, 0.0, 0.0]}]

    result = validate_track_config(parse_track_config(raw))

    assert any("current_gate requires at least one configured current" in error for error in result.errors)


def test_obstacle_gate_accepts_static_obstacle_between_adjacent_gates() -> None:
    raw = _raw_track("marine_race_horseshoe_bay.json")
    raw["benchmark_task"] = BENCHMARK_TASK_OBSTACLE_GATE
    raw["obstacles"] = [_static_obstacle()]

    result = validate_track_config(parse_track_config(raw))

    assert result.errors == []


def test_obstacle_gate_requires_obstacles_with_gate_interval() -> None:
    raw = _raw_track("marine_race_horseshoe_bay.json")
    raw["benchmark_task"] = BENCHMARK_TASK_OBSTACLE_GATE
    raw["obstacles"] = [{"id": "OBS01", "type": "pillar", "position": [-27.0, -8.0, -4.0]}]

    result = validate_track_config(parse_track_config(raw))

    assert any("requires 'size'" in error for error in result.errors)


def test_multi_rov_requires_multiple_participants() -> None:
    raw = _raw_track("marine_race_horseshoe_bay.json")
    raw["benchmark_task"] = BENCHMARK_TASK_MULTI_ROV

    result = validate_track_config(parse_track_config(raw))

    assert any("multi_rov requires at least two participants" in error for error in result.errors)


def test_multi_rov_accepts_two_participants() -> None:
    raw = _raw_track("marine_race_horseshoe_bay.json")
    raw["benchmark_task"] = BENCHMARK_TASK_MULTI_ROV
    second_participant = copy.deepcopy(raw["participants"][0])
    second_participant["id"] = "bluerov2_02"
    second_participant["spawn"]["position"] = [-33.33, -11.72, -4.0]
    raw["participants"].append(second_participant)

    result = validate_track_config(parse_track_config(raw))

    assert result.errors == []


def test_cli_benchmark_task_override_is_applied_by_loader(tmp_path: Path) -> None:
    raw = _raw_track("marine_race_horseshoe_bay.json")
    raw.pop("benchmark_task", None)
    track_path = tmp_path / "legacy_track.json"
    track_path.write_text(json.dumps(raw), encoding="utf-8")

    config = load_track_config(track_path, benchmark_task=BENCHMARK_TASK_CLEAN_GATE)

    assert config.benchmark_task.mode == BENCHMARK_TASK_CLEAN_GATE


def test_run_parser_accepts_benchmark_task_argument() -> None:
    args = _build_arg_parser().parse_args(
        [
            "--track",
            "track.json",
            "--benchmark-task",
            BENCHMARK_TASK_CURRENT_GATE,
            "--current-profile",
            "medium",
        ]
    )

    assert args.benchmark_task == BENCHMARK_TASK_CURRENT_GATE
    assert args.current_profile == "medium"


def test_benchmark_task_override_can_be_applied_after_parsing() -> None:
    raw = _raw_track("marine_race_horseshoe_bay.json")
    raw.pop("benchmark_task", None)
    config = parse_track_config(raw)

    overridden = with_benchmark_task(config, BENCHMARK_TASK_CLEAN_GATE)

    assert overridden.benchmark_task.mode == BENCHMARK_TASK_CLEAN_GATE
    assert validate_track_config(overridden).errors == []


def test_unknown_benchmark_task_mode_is_invalid() -> None:
    raw = _raw_track("marine_race_horseshoe_bay.json")
    raw["benchmark_task"] = "unknown_task"

    result = validate_track_config(parse_track_config(raw))

    assert any("benchmark_task.mode 'unknown_task' is not supported" in error for error in result.errors)


def _raw_track(track_name: str) -> dict:
    return json.loads((TRACK_DIR / track_name).read_text(encoding="utf-8"))


def _static_obstacle() -> dict:
    return {
        "id": "OBS01",
        "type": "box",
        "position": [-28.2, -6.25, -4.05],
        "size": [0.7, 0.7, 0.7],
        "rotation_rpy_deg": [0.0, 0.0, 33.7],
        "collision": True,
        "penalty_s": 5.0,
        "between_gates": ["G01", "G02"],
    }
