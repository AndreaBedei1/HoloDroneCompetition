"""Model-path plumbing through ControllerLoader and the runner CLI.

These run without torch: RLGateController stores the path at construction and only
imports torch when it actually loads a model (in reset()).
"""

import pytest

from marine_race_arena.participants.controller_loader import ControllerLoader
from marine_race_arena.scripts.run_marine_race import _build_arg_parser

MODEL_ENV = "MARINE_RACE_RL_MODEL"


def test_rule_controller_ignores_model_path():
    # A rule baseline has no model_path parameter; passing one must not error.
    controller = ControllerLoader().load(
        "rule_gate_center_then_commit", constructor_kwargs={"model_path": "/some/path.pt"}
    )
    assert type(controller).__name__ == "RuleGateCenterThenCommitController"


def test_rl_controller_receives_cli_model_path(monkeypatch):
    monkeypatch.delenv(MODEL_ENV, raising=False)
    controller = ControllerLoader().load(
        "rl_gate_controller", constructor_kwargs={"model_path": "results/rl/stage1/bc/best.pt"}
    )
    assert controller._model_path == "results/rl/stage1/bc/best.pt"


def test_none_model_path_falls_back_to_env(monkeypatch):
    monkeypatch.setenv(MODEL_ENV, "env/model.pt")
    controller = ControllerLoader().load("rl_gate_controller", constructor_kwargs={"model_path": None})
    assert controller._model_path == "env/model.pt"


def test_cli_precedence_over_env(monkeypatch):
    monkeypatch.setenv(MODEL_ENV, "env/model.pt")
    controller = ControllerLoader().load(
        "rl_gate_controller", constructor_kwargs={"model_path": "explicit/model.pt"}
    )
    assert controller._model_path == "explicit/model.pt"


def test_missing_model_raises_clear_error(monkeypatch):
    monkeypatch.delenv(MODEL_ENV, raising=False)
    controller = ControllerLoader().load("rl_gate_controller")
    with pytest.raises(ValueError, match="model path"):
        controller.reset({"participant_id": "p", "initial_beacon_id": "B01", "total_beacons": 12, "laps": 1})


def test_arg_parser_exposes_controller_model_path():
    parser = _build_arg_parser()
    args = parser.parse_args([
        "--track", "marine_race_arena/tracks/training/stage1_single_gate.json",
        "--controller", "rl_gate_controller",
        "--controller-model-path", "results/rl/stage1/bc/best_model.pt",
    ])
    assert args.controller_model_path == "results/rl/stage1/bc/best_model.pt"
    assert args.controller == "rl_gate_controller"
    # default is None so unset -> env-var fallback path in the controller
    args_default = parser.parse_args(["--track", "x", "--controller", "rule_gate_baseline"])
    assert args_default.controller_model_path is None


def test_help_lists_rl_gate_controller():
    parser = _build_arg_parser()
    help_text = parser.format_help()
    assert "rl_gate_controller" in help_text
