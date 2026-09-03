"""Structural boundary checks for the gate-pose visual debugger."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "marine_race_arena" / "tools" / "gate_pose_debug.py"
OFFLINE_FUNCTION = "_offline_ground_truth"
FORBIDDEN_ATTRIBUTES = (
    "get_participant_state",
    "expected_gate_id",
    "gate_map",
    "referee",
)


def _tree() -> ast.Module:
    return ast.parse(TOOL.read_text(encoding="utf-8"))


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def test_simulator_truth_is_confined_to_the_named_offline_function():
    offenders = []
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.FunctionDef) or node.name == OFFLINE_FUNCTION:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Attribute) and inner.attr in FORBIDDEN_ATTRIBUTES:
                offenders.append((node.name, inner.attr))
    assert not offenders, offenders


def test_offline_ground_truth_call_is_guarded_by_the_explicit_flag():
    run = _function("run_track")
    total = guarded = 0
    for node in ast.walk(run):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == OFFLINE_FUNCTION:
            total += 1
    for node in ast.walk(run):
        if not isinstance(node, ast.If):
            continue
        names = {child.id for child in ast.walk(node.test) if isinstance(child, ast.Name)}
        if "offline_ground_truth" not in names:
            continue
        guarded += sum(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == OFFLINE_FUNCTION
            for child in ast.walk(node)
        )
    assert total == guarded == 1


def test_ground_truth_and_training_are_off_by_default():
    from marine_race_arena.tools.gate_pose_debug import build_parser

    args = build_parser().parse_args([])
    assert args.offline_ground_truth is False
    assert args.validate_all is False
    assert args.track == "horseshoe_bay"


def test_debugger_has_no_training_or_policy_checkpoint_imports():
    source = TOOL.read_text(encoding="utf-8")
    assert "stable_baselines" not in source
    assert "best_completion_policy" not in source
    assert "train_multigate_ppo" not in source
