"""The debug tool must show the rover's view, not the simulator's.

A visual debugger is worth exactly as much as the guarantee that what it draws
is what the controller had. These tests pin that guarantee structurally rather
than by reading the output: the one function that touches simulator state is
named, and nothing on the standard path is allowed to call it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "marine_race_arena" / "tools" / "obs_debug.py"

#: The single sanctioned entry point to simulator state, used only by the
#: explicitly labelled comparison mode.
GROUND_TRUTH_FUNCTION = "_ground_truth"

#: Attributes that only exist on the simulator side of the boundary.
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
    raise AssertionError(f"{name} non trovata in obs_debug.py")


def test_only_one_function_reads_simulator_state():
    """Every touch of the simulator lives in _ground_truth, so it is checkable."""
    offenders = []
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.FunctionDef) or node.name == GROUND_TRUTH_FUNCTION:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Attribute) and inner.attr in FORBIDDEN_ATTRIBUTES:
                offenders.append((node.name, inner.attr))
    assert not offenders, offenders


def test_the_frame_decoder_never_receives_the_episode():
    """read_frame builds every panel value; it must not be handed the world.

    Without the episode it cannot reach the adapter, the referee or the gate
    map even by accident, so the panels can only show onboard signals.
    """
    import inspect

    from marine_race_arena.tools import obs_debug

    parameters = list(inspect.signature(obs_debug.read_frame).parameters)
    assert parameters == ["raw", "context", "tracker", "action", "step"], parameters


def test_ground_truth_is_only_reached_behind_the_flag():
    """The call sites of _ground_truth must all sit under with_ground_truth."""
    run = _function("run")
    guarded = unguarded = 0
    for node in ast.walk(run):
        if not isinstance(node, ast.If):
            continue
        test_names = {
            n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
        }
        if "with_ground_truth" not in test_names:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) \
                    and inner.func.id == GROUND_TRUTH_FUNCTION:
                guarded += 1
    for node in ast.walk(run):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == GROUND_TRUTH_FUNCTION:
            unguarded += 1
    assert guarded >= 1, "nessuna chiamata a _ground_truth sotto il flag"
    assert guarded == unguarded, (
        f"{unguarded - guarded} chiamate a {GROUND_TRUTH_FUNCTION} fuori dal flag"
    )


def test_ground_truth_defaults_to_off():
    from marine_race_arena.tools.obs_debug import build_parser

    assert build_parser().parse_args([]).with_ground_truth is False


def test_the_window_title_states_which_mode_is_running():
    """A screenshot taken from this tool has to say what it is showing."""
    source = TOOL.read_text(encoding="utf-8")
    assert "SOLO ONBOARD" in source
    assert "NON DISPONIBILE AL ROVER" in source


def test_implied_bearing_matches_the_beacon_sign_convention():
    """Vision and beacon are compared in degrees, so the signs must agree.

    A detection right of image centre is to starboard, which the beacon reports
    as a negative bearing. Getting this backwards would make every consistent
    frame read as a 2x-bearing disagreement.
    """
    from marine_race_arena.tools.obs_debug import CAMERA_FOV_DEG, implied_bearing_deg

    assert implied_bearing_deg(0.0) == pytest.approx(0.0)
    assert implied_bearing_deg(1.0) == pytest.approx(-CAMERA_FOV_DEG / 2)
    assert implied_bearing_deg(-1.0) == pytest.approx(+CAMERA_FOV_DEG / 2)
    assert implied_bearing_deg(None) is None


def test_depth_is_read_through_the_encoder_conversion():
    """Depth and its reference must share one sign convention.

    The reference comes from the encoder's positive-down ``_depth_m``. Reading
    the raw negative-z alongside it printed a depth error of roughly twice the
    depth -- an instrument fault that looks exactly like a pipeline fault.
    """
    source = TOOL.read_text(encoding="utf-8")
    assert "_depth_m(sensors)" in source
    assert 'depth[0] if depth else None' not in source
