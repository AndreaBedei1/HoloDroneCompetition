"""Model-path plumbing through ControllerLoader and the runner CLI.

The loader must pass ``model_path`` only to controllers that declare it, and the
runner must expose ``--controller-model-path`` for externally supplied learned
controllers.
"""

from marine_race_arena.participants.controller_loader import ControllerLoader
from marine_race_arena.scripts.run_marine_race import _build_arg_parser


def test_rule_controller_ignores_model_path():
    # A rule baseline has no model_path parameter; passing one must not error.
    controller = ControllerLoader().load(
        "rule_gate_center_then_commit", constructor_kwargs={"model_path": "/some/path.pt"}
    )
    assert type(controller).__name__ == "RuleGateCenterThenCommitController"


def test_arg_parser_exposes_controller_model_path():
    parser = _build_arg_parser()
    args = parser.parse_args([
        "--track", "marine_race_arena/tracks/marine_race_horseshoe_bay.json",
        "--controller", "some.module:LearnedController",
        "--controller-model-path", "artifacts/paper/ppo/model/policy_recurrent_ppo_27d.zip",
    ])
    assert args.controller_model_path == (
        "artifacts/paper/ppo/model/policy_recurrent_ppo_27d.zip"
    )
    assert args.controller == "some.module:LearnedController"
    # default is None so an unset path leaves the controller's own fallback in charge
    args_default = parser.parse_args(["--track", "x", "--controller", "rule_gate_baseline"])
    assert args_default.controller_model_path is None


def test_help_lists_the_released_reference_controllers():
    help_text = _build_arg_parser().format_help()
    assert "rule_gate_baseline" in help_text
    assert "rule_gate_center_then_commit" in help_text
    assert "leader_follower" in help_text
    assert "rl_gate_controller" not in help_text
