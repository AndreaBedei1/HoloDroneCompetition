"""Delta-g default: LF(1) is the recommended coordination margin.

These tests pin the new default (``min_gate_gap == 1``) in the controller and the
CLI, confirm the conservative ``min_gate_gap == 2`` margin still works when asked
for explicitly, and confirm the existing 78-run artifact matrix still contains
both the LF(2) (main) and LF(1) (min_gate_gap_1) coordination artifacts for both
start gaps and all three seeds. No HoloOcean run is launched.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from marine_race_arena.controllers.leader_follower import LeaderFollowerController
from marine_race_arena.scripts.run_holoocean_coordination_validation import _build_arg_parser

MATRIX = Path("results/onboard_only_validation/final_20260715/coordination")


class _StubBase:
    uses_ground_truth = False

    def __init__(self) -> None:
        self.tracker = SimpleNamespace(
            local_beacon_index=0, local_lap=1, local_completed=0, status="RUNNING"
        )

    def reset(self, mission_info: Dict[str, Any]) -> None:
        self.tracker = SimpleNamespace(
            local_beacon_index=0, local_lap=1, local_completed=0, status="RUNNING"
        )

    def step(self, observation: Dict[str, Any]) -> Dict[str, float]:
        return {"surge": 0.4, "sway": 0.0, "heave": 0.0, "yaw": 0.0}

    def close(self) -> None:
        pass


def _mission_info() -> Dict[str, Any]:
    return {
        "participant_id": "bluerov2_02",
        "initial_beacon_id": "B01",
        "total_beacons": 12,
        "laps": 1,
        "command_limits": {a: [-0.95, 0.95] for a in ("surge", "sway", "heave", "yaw")},
        "fleet": {
            "participant_order": ["bluerov2_01", "bluerov2_02"],
            "release_index": 1,
            "predecessor_id": "bluerov2_01",
        },
    }


def test_controller_class_default_is_one():
    assert LeaderFollowerController.MIN_GATE_GAP == 1


def test_default_constructed_coordinator_uses_one(monkeypatch):
    monkeypatch.delenv("MARINE_RACE_COORDINATION_MIN_GAP", raising=False)
    controller = LeaderFollowerController(base_controller=_StubBase())
    controller.reset(_mission_info())
    assert controller._min_gate_gap == 1
    assert controller.coordination_diagnostics["min_gate_gap"] == 1


def test_explicit_conservative_margin_two_still_works(monkeypatch):
    monkeypatch.delenv("MARINE_RACE_COORDINATION_MIN_GAP", raising=False)
    controller = LeaderFollowerController(base_controller=_StubBase(), min_gate_gap=2)
    controller.reset(_mission_info())
    assert controller._min_gate_gap == 2


def test_env_override_still_respected(monkeypatch):
    monkeypatch.setenv("MARINE_RACE_COORDINATION_MIN_GAP", "3")
    controller = LeaderFollowerController(base_controller=_StubBase())
    controller.reset(_mission_info())
    assert controller._min_gate_gap == 3


def test_cli_default_is_one():
    args = _build_arg_parser().parse_args([])
    assert args.min_gate_gap == 1


# --------------------------------------------------------------------------- #
# The existing artifact matrix must still contain both LF settings unchanged.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not MATRIX.exists(), reason="coordination artifacts not present")
@pytest.mark.parametrize(
    "variant,expected_gap",
    [("main", 2), ("min_gate_gap_1", 1)],
)
@pytest.mark.parametrize("gap_label", ["gap_0", "gap_8"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_existing_lf_artifacts_present_with_expected_margin(variant, expected_gap, gap_label, seed):
    run_dir = MATRIX / variant / gap_label / "diagnostic" / f"seed_{seed}" / "leader_follower"
    assert run_dir.is_dir(), f"missing LF artifact dir: {run_dir}"
    metadata_path = run_dir / "experiment_metadata.json"
    assert metadata_path.is_file(), f"missing metadata: {metadata_path}"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata.get("min_gate_gap") == expected_gap
    assert metadata.get("condition") == "leader_follower"
    assert int(metadata.get("seed")) == seed


@pytest.mark.skipif(not MATRIX.exists(), reason="coordination artifacts not present")
def test_main_variant_also_has_matched_no_coordination_runs():
    for gap_label in ("gap_0", "gap_8"):
        for seed in (0, 1, 2):
            run_dir = MATRIX / "main" / gap_label / "diagnostic" / f"seed_{seed}" / "no_coordination"
            assert run_dir.is_dir(), f"missing matched uncoordinated dir: {run_dir}"
