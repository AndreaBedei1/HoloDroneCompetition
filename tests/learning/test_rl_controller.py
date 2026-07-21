"""Tests for the deployable RL controller (skipped without torch)."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytest.importorskip("torch")

from marine_race_arena.learning.bc_train import BCPolicy, save_policy
from marine_race_arena.learning.config import ACTION_AXES
from marine_race_arena.learning.episode import build_single_vehicle_race
from marine_race_arena.learning.rl_controller import RLGateController
from marine_race_arena.participants.controller_interface import validate_controller_instance
from marine_race_arena.participants.controller_loader import ControllerLoader
from marine_race_arena.scripts.run_marine_race import _run_race_loop

TRACK = "marine_race_arena/tracks/tests/single_gate_yaw_0.json"


@pytest.fixture
def bc_model(tmp_path):
    path = tmp_path / "bc.pt"
    save_policy(BCPolicy(hidden_sizes=(32, 32)), path)
    return str(path)


def _mission():
    return {"participant_id": "bluerov2_01", "initial_beacon_id": "B01", "total_beacons": 12, "laps": 1}


def test_controller_is_legal_in_official_mode(bc_model):
    ctrl = RLGateController(model_path=bc_model)
    assert ctrl.uses_ground_truth is False
    assert ctrl.debug_only is False
    validate_controller_instance(ctrl)  # has reset/step/close


def test_controller_step_returns_valid_command(bc_model):
    ctrl = RLGateController(model_path=bc_model)
    ctrl.reset(_mission())
    obs = {
        "local_time_s": 1.0,
        "sensors": {"DepthSensor": [-3.0], "DVLSensor": [0.5, 0.0, 0.0], "IMUSensor": [[0, 0, 0], [0, 0, 0.1]]},
        "beacons": [{"beacon_id": "B01", "bearing_deg": 5.0, "elevation_deg": 0.0, "range_m": 8.0, "signal_strength": 0.9, "received_at_s": 1.0}],
    }
    cmd = ctrl.step(obs)
    assert set(cmd.keys()) == set(ACTION_AXES)
    for v in cmd.values():
        assert -1.0 <= v <= 1.0 and np.isfinite(v)
    ctrl.close()


def test_controller_handles_missing_sensors(bc_model):
    ctrl = RLGateController(model_path=bc_model)
    ctrl.reset(_mission())
    cmd = ctrl.step({"local_time_s": 0.0, "sensors": {}, "beacons": []})  # everything masked
    assert set(cmd.keys()) == set(ACTION_AXES)
    assert all(np.isfinite(v) for v in cmd.values())
    ctrl.close()


def test_controller_missing_model_raises():
    ctrl = RLGateController(model_path=None)
    with pytest.raises(ValueError):
        ctrl.reset(_mission())


def test_controller_zero_after_finish(bc_model):
    ctrl = RLGateController(model_path=bc_model)
    ctrl.reset(_mission())
    ctrl.step({"local_time_s": 0.0, "sensors": {}, "beacons": []})
    ctrl._finished = True  # simulate local completion
    cmd = ctrl.step({"local_time_s": 1.0, "sensors": {}, "beacons": []})
    assert cmd == {axis: 0.0 for axis in ACTION_AXES}
    ctrl.close()


def test_controller_loads_via_alias():
    controller = ControllerLoader().load("rl_gate_controller")
    assert isinstance(controller, RLGateController)


def test_controller_closed_loop_through_unchanged_runner(bc_model):
    """Full integration: the learned controller drives the real runner + referee."""
    ctrl = RLGateController(model_path=bc_model)
    ctx = build_single_vehicle_race(TRACK, seed=1, adapter="fallback", allow_fallback=True, duration_s=3.0, controller=ctrl)
    pid = ctx.participant.id
    ctrl.reset({"participant_id": pid, "initial_beacon_id": "B01", "total_beacons": len(ctx.config.track.gate_sequence), "laps": ctx.config.race.laps})
    try:
        summary = _run_race_loop(config=ctx.config, arena=ctx.arena, referee=ctx.referee, adapter=ctx.adapter, participants={pid: ctx.participant}, dt=0.1)
        assert "participants" in summary or "ranking" in summary
        state = ctx.referee.states[pid]
        assert state.status is not None  # referee scored the run
    finally:
        ctx.adapter.close()


def test_bc_inference_needs_no_gymnasium_or_sb3(tmp_path):
    """BC inference must not import Gymnasium or Stable-Baselines3 training objects."""
    model = tmp_path / "bc.pt"
    save_policy(BCPolicy(hidden_sizes=(16, 16)), model)
    script = textwrap.dedent(
        f"""
        import sys, numpy as np
        from marine_race_arena.learning.rl_controller import RLGateController
        ctrl = RLGateController(model_path=r"{model}")
        ctrl.reset({{"participant_id":"p","initial_beacon_id":"B01","total_beacons":12,"laps":1}})
        ctrl.step({{"local_time_s":0.0,"sensors":{{}},"beacons":[]}})
        assert "gymnasium" not in sys.modules, "gymnasium imported for BC inference"
        assert "stable_baselines3" not in sys.modules, "sb3 imported for BC inference"
        print("NO_GYM_OK")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert "NO_GYM_OK" in result.stdout, result.stderr
