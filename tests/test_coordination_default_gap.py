"""Delta-g default: LF(1) is the recommended coordination margin.

These tests pin the new default (``min_gate_gap == 1``) in the controller and the
CLI, confirm the conservative ``min_gate_gap == 2`` margin still works when asked
for explicitly, and confirm the released
benchmark rows still contain both the LF(2) and LF(1) coordination runs for both
start gaps and all three seeds. No HoloOcean run is launched.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from marine_race_arena.controllers.leader_follower import LeaderFollowerController
from marine_race_arena.scripts.run_holoocean_coordination_validation import _build_arg_parser

RUNS = Path(__file__).resolve().parents[1] / "artifacts/paper/benchmark/runs.csv"


def _coordination_rows():
    with RUNS.open(encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["experiment"] == "coordination"]


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
# The released benchmark rows must still contain both LF settings unchanged.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("expected_gap", [1, 2])
@pytest.mark.parametrize("gap_label", ["0.0", "8.0"])
def test_released_lf_rows_present_with_expected_margin(expected_gap, gap_label):
    rows = [row for row in _coordination_rows()
            if row["condition"] == "leader_follower"
            and row["start_gap_s"] == gap_label
            and int(row["min_gate_gap_configured"]) == expected_gap]
    assert len(rows) == 3, "expected three seeds, got {}".format(len(rows))
    assert {int(row["seed"]) for row in rows} == {0, 1, 2}
    for row in rows:
        assert int(row["min_gate_gap_effective"]) == expected_gap


@pytest.mark.parametrize("gap_label", ["0.0", "8.0"])
def test_matched_uncoordinated_rows_are_present(gap_label):
    rows = [row for row in _coordination_rows()
            if row["condition"] == "no_coordination" and row["start_gap_s"] == gap_label]
    assert len(rows) == 3, "expected three seeds, got {}".format(len(rows))
    assert {int(row["seed"]) for row in rows} == {0, 1, 2}
