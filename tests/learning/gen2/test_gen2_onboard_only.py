"""The controller is onboard-only, and the audit must not weaken that.

Simulator ground truth is legitimate as an offline measuring instrument and
illegitimate anywhere near an action.  These tests pin the boundary rather than
trusting it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from marine_race_arena.learning.config_local_transition import (
    FEATURE_NAMES_LOCAL_TRANSITION,
    OBS_DIM_LOCAL_TRANSITION,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
GEN2_DIR = REPO_ROOT / "marine_race_arena" / "learning" / "gen2"

#: Anything that would mean the policy is being told about the world.
PRIVILEGED_TOKENS = (
    "gate_map",
    "get_participant_state",
    "debug_ground_truth",
    "true_geometry",
    "perception_audit",
    "OFFICIAL_TRACKS",
    "gen2_fragment",
    "circuit_id",
    "track_id",
)


def test_the_observation_contract_is_still_35_onboard_features():
    assert OBS_DIM_LOCAL_TRANSITION == 35
    assert len(FEATURE_NAMES_LOCAL_TRANSITION) == 35
    # No feature may name a world-frame or map quantity.
    for name in FEATURE_NAMES_LOCAL_TRANSITION:
        for banned in ("global", "world", "gate_x", "gate_y", "gate_z",
                       "track", "circuit", "map", "index_of_gate"):
            assert banned not in name, name


def test_the_inference_path_carries_nothing_privileged():
    """The modules that produce actions must not reach for the simulator."""
    for module in ("recurrent_policy.py", "evaluation.py"):
        source = (GEN2_DIR / module).read_text(encoding="utf-8")
        for token in PRIVILEGED_TOKENS:
            assert token not in source, f"{module} references {token!r}"


def test_the_controller_act_signature_takes_only_observation_and_state():
    """action = learned_policy(observation, recurrent_state) -- and nothing else."""
    tree = ast.parse((GEN2_DIR / "recurrent_policy.py").read_text(encoding="utf-8"))
    controller = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "Gen2RecurrentController"
    )
    act = next(n for n in controller.body
               if isinstance(n, ast.FunctionDef) and n.name == "act")
    argument_names = [a.arg for a in act.args.args]
    assert argument_names == ["self", "observation", "first_step"], argument_names
    assert act.args.kwonlyargs == []


def test_the_controller_rejects_a_widened_observation():
    """Concatenating anything to the observation must fail loudly."""
    pytest.importorskip("torch")
    pytest.importorskip("sb3_contrib")
    from marine_race_arena.learning.gen2.recurrent_policy import (
        Gen2RecurrentController,
        build_gen2_policy_for_training,
    )

    controller = Gen2RecurrentController(build_gen2_policy_for_training(seed=0))
    legal = np.zeros(OBS_DIM_LOCAL_TRANSITION, np.float32)
    controller.act(legal, first_step=True)
    # A world coordinate appended to the observation is the failure mode this
    # guards against; it must raise rather than be silently accepted.
    with pytest.raises(ValueError):
        controller.act(np.concatenate([legal, np.float32([12.5, -3.0, -4.2])]),
                       first_step=True)


def test_the_audit_keeps_sensed_and_true_in_separate_fields():
    """A single struct with both halves is fine; mixing them into one is not."""
    from marine_race_arena.learning.gen2.perception_audit import PerceptionSample

    fields = {f for f in PerceptionSample.__dataclass_fields__}
    sensed = {f for f in fields if f.startswith("sensed_")}
    true = {f for f in fields if f.startswith("true_")}
    assert sensed and true
    assert not (sensed & true)


def test_decode_sensed_reads_only_the_observation():
    """The decoder must not accept, or need, any reference input."""
    import inspect

    from marine_race_arena.learning.gen2.perception_audit import decode_sensed

    parameters = list(inspect.signature(decode_sensed).parameters)
    assert parameters == ["observation"], parameters


def test_the_offline_separation_check_passes():
    from marine_race_arena.learning.gen2.perception_audit import (
        assert_audit_is_offline_only,
    )

    report = assert_audit_is_offline_only(REPO_ROOT)
    assert report["offline_only"], report["findings"]


def test_fragments_only_build_the_scenario_never_the_observation():
    """Fragment geometry may set the world up; it may not reach the policy.

    ``evaluation.py`` does touch ``track_fragments``, for one thing only:
    installing the vehicle's initial body velocity at reset. That is scenario
    construction -- the same category as choosing which track to load -- and it
    happens before the first observation exists. What would be a violation is
    reading fragment geometry inside the stepping loop, so the test pins the
    symbol and the position rather than banning the import.
    """
    source = (GEN2_DIR / "evaluation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        and node.module.endswith("track_fragments")
        for alias in node.names
    }
    assert imported == {"apply_initial_body_velocity"}, imported

    run = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "run_policy_episode")
    loops = [n for n in ast.walk(run) if isinstance(n, ast.While)]
    assert loops, "expected a stepping loop"
    setup_line = next(
        n.lineno for n in ast.walk(run)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "apply_initial_body_velocity"
    )
    assert setup_line < loops[0].lineno, (
        "initial velocity must be installed before stepping begins, not inside the loop"
    )

    # And nothing fragment-shaped may appear inside the loop at all.
    loop_source = ast.unparse(loops[0])
    for token in ("fragment", "gate_map", "track_path", "true_"):
        assert token not in loop_source, f"stepping loop references {token!r}"

    # The encoder the policy uses takes the raw observation and a context
    # object, never a track or a fragment.
    import inspect

    from marine_race_arena.learning.observation_encoder_local_transition import (
        encode_observation_local_transition,
    )

    parameters = list(inspect.signature(encode_observation_local_transition).parameters)
    assert parameters == ["observation", "context"], parameters


def test_ground_truth_in_evaluation_is_used_only_for_metrics():
    """Path length is measured from simulator state; that must stay a metric."""
    source = (GEN2_DIR / "evaluation.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    run = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "run_policy_episode")
    loop = next(n for n in ast.walk(run) if isinstance(n, ast.While))
    body = ast.unparse(loop)
    # The true position is read, and it may only feed path_length.
    assert "current_state.position" in body
    for line in body.splitlines():
        if "current_state.position" in line:
            assert "position" in line and "encoded" not in line and "controller" not in line, line
    # The action comes from the controller, given only the encoded observation.
    assert "controller.act(encoded" in body


def _sample(step, *, sensed_bearing, true_bearing, tracked_bearing,
            true_gate="G05", tracked_gate="G05"):
    """One synthetic audit row; only the bearing triple matters here."""
    from marine_race_arena.learning.gen2.perception_audit import PerceptionSample

    return PerceptionSample(
        step=step, sensed_present=True, sensed_bearing_deg=sensed_bearing,
        sensed_elevation_deg=0.0, sensed_range_m=10.0, sensed_expected_beacon="B05",
        true_bearing_deg=true_bearing, true_elevation_deg=0.0, true_range_m=10.0,
        true_expected_gate=true_gate,
        tracked_bearing_deg=tracked_bearing, tracked_elevation_deg=0.0,
        tracked_range_m=10.0, tracked_gate=tracked_gate,
        gates_crossed=0, beacon_age_norm=0.0, vision_present=True,
        dvl_present=True, imu_present=True, depth_present=True,
        saturated_features=0, observation_repeated=False,
    )


def test_a_wrong_target_does_not_read_as_a_wrong_sensor():
    """The audit must tell "aimed elsewhere" apart from "measured badly".

    This is the regression for a real misreading: the first audit run scored a
    rover that was accurately tracking the wrong gate as a 107 deg bearing
    error, which looks like a broken sensor and would have sent the fix into
    the encoder instead of into the target tracker. Perfect perception of the
    wrong gate must show a near-zero ``tracked_`` error and a large ``true_``
    one, never the reverse.
    """
    import numpy as np

    from marine_race_arena.learning.gen2.perception_audit import summarize

    samples = [
        _sample(i, sensed_bearing=3.0, true_bearing=-120.0, tracked_bearing=3.0,
                true_gate="G05", tracked_gate="G02")
        for i in range(40)
    ]
    obs = [np.zeros(35, dtype=np.float32) for _ in samples]
    actions = [np.zeros(4, dtype=np.float32) for _ in samples]

    report = summarize(samples, obs, actions)

    assert report["tracked_bearing_error_deg"]["abs_mean"] < 1e-6
    assert report["bearing_error_deg"]["abs_mean"] > 100.0
    assert report["on_target_fraction"] == 0.0


def test_a_genuinely_wrong_sensor_still_reads_as_wrong():
    """The separation must not become a way to hide perception errors."""
    import numpy as np

    from marine_race_arena.learning.gen2.perception_audit import summarize

    samples = [
        _sample(i, sensed_bearing=40.0, true_bearing=0.0, tracked_bearing=0.0)
        for i in range(40)
    ]
    obs = [np.zeros(35, dtype=np.float32) for _ in samples]
    actions = [np.zeros(4, dtype=np.float32) for _ in samples]

    report = summarize(samples, obs, actions)

    assert report["tracked_bearing_error_deg"]["abs_mean"] == pytest.approx(40.0)
    assert report["on_target_fraction"] == 1.0
