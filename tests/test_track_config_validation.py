from __future__ import annotations

import copy
import json
from pathlib import Path

from marine_race_arena.arena.obstacle import resolve_active_obstacles
from marine_race_arena.config.loader import parse_track_config, load_track_config
from marine_race_arena.config.validation import compute_declared_path_length_m, validate_track_config


TRACK_DIR = Path(__file__).resolve().parents[1] / "marine_race_arena" / "tracks"


def test_example_tracks_validate() -> None:
    for track_name in (
        "marine_race_horseshoe_bay.json",
        "marine_race_mixed_endurance.json",
        "marine_race_vertical_serpent.json",
    ):
        config = load_track_config(TRACK_DIR / track_name)
        result = validate_track_config(config)
        assert result.errors == []


def test_single_gate_rule_baseline_tracks_validate() -> None:
    for track_name in (
        "tests/single_gate_yaw_neg45.json",
        "tests/single_gate_yaw_neg25.json",
        "tests/single_gate_yaw_0.json",
        "tests/single_gate_yaw_25.json",
        "tests/single_gate_yaw_45.json",
    ):
        config = load_track_config(TRACK_DIR / track_name)
        result = validate_track_config(config)
        assert result.errors == []
        assert config.participants[0].controller == "rule_gate_baseline"


def test_progressive_rule_baseline_tracks_validate() -> None:
    for track_name in (
        "tests/two_gate_straight.json",
        "tests/two_gate_left_curve.json",
        "tests/two_gate_right_curve.json",
        "tests/three_gate_s_curve.json",
        "tests/four_gate_horseshoe_start.json",
    ):
        config = load_track_config(TRACK_DIR / track_name)
        result = validate_track_config(config)
        assert result.errors == []
        assert config.benchmark_task.mode == "clean_gate"
        assert config.currents == []
        assert config.obstacles == []
        assert resolve_active_obstacles(config) == []
        assert config.participants[0].controller == "rule_gate_baseline"
        assert config.participants[0].sensors.get("profile") == "official_vision_acoustic"


def test_example_tracks_use_standard_gate_opening() -> None:
    for track_name in (
        "marine_race_horseshoe_bay.json",
        "marine_race_mixed_endurance.json",
        "marine_race_vertical_serpent.json",
    ):
        config = load_track_config(TRACK_DIR / track_name)
        assert config.track.gate_inner_size_m == (1.5, 1.5)
        assert all(gate.inner_size_m == (1.5, 1.5) for gate in config.gates)


def test_duplicate_gate_id_is_invalid() -> None:
    raw = json.loads((TRACK_DIR / "marine_race_horseshoe_bay.json").read_text(encoding="utf-8"))
    raw["gates"] = copy.deepcopy(raw["gates"])
    raw["gates"][1]["id"] = raw["gates"][0]["id"]
    config = parse_track_config(raw)
    result = validate_track_config(config)
    assert any("Duplicated gate id" in error for error in result.errors)


def test_declared_length_matches_computed_length() -> None:
    config = load_track_config(TRACK_DIR / "marine_race_mixed_endurance.json")
    computed = compute_declared_path_length_m(config)
    assert abs(computed - config.track.declared_length_m) <= config.track.length_tolerance_m
