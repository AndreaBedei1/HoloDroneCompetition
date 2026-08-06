"""Supervisor-side governor wiring, ownership and restart safety."""

from __future__ import annotations

import json

import pytest

import marine_race_arena.learning.rl_autonomous_supervisor as supervisor
from marine_race_arena.learning.rl_resource_governor import GovernorPolicy


def _config(ceiling=10, dwell=1800.0):
    return {
        "governor": {
            "maximum_total_engines": ceiling,
            "minimum_workers_per_algorithm": 2,
            "minimum_dwell_seconds": dwell,
            "evaluator_workers": 6,
        },
        "sac_worker_options": [2, 4, 6, 8],
        "algorithms": {"ppo": {}, "sac": {}},
    }


def _rows(ppo_state, sac_state, *, ppo_tps=0.0, sac_tps=0.0, sac_collapsed=False):
    return {
        "ppo": {"state": ppo_state,
                "trainer_status": {"environment_transitions_per_second": ppo_tps}},
        "sac": {"state": sac_state, "collapsed": sac_collapsed,
                "trainer_status": {"environment_transitions_per_second": sac_tps}},
    }


def test_governor_is_built_from_config_and_supported_shardings():
    governor = supervisor._build_governor(_config(ceiling=12))
    assert governor.policy.maximum_total_engines == 12
    assert governor.policy.minimum_dwell_seconds == 1800.0
    assert 4 in governor.supported["ppo"] and 8 in governor.supported["ppo"]
    assert governor.supported["sac"] == (2, 4, 6, 8)


def test_governor_cycle_never_exceeds_the_ceiling(monkeypatch):
    monkeypatch.setattr(supervisor, "sample_resources", lambda: {"cpu_percent": 50.0})
    for ceiling in (8, 10, 12, 16, 20):
        governor = supervisor._build_governor(_config(ceiling=ceiling))
        plan, _ = supervisor._governor_cycle(
            _config(ceiling), _rows("PAUSED", "PAUSED"), governor
        )
        assert plan["total_engines"] <= ceiling


def test_collapsed_sac_lends_all_slots_to_ppo(monkeypatch):
    monkeypatch.setattr(supervisor, "sample_resources", lambda: {"cpu_percent": 40.0})
    config = _config(ceiling=10)
    governor = supervisor._build_governor(config)
    plan, _ = supervisor._governor_cycle(
        config, _rows("TRAINING", "PAUSED", sac_collapsed=True), governor
    )
    assert plan["algorithm_states"]["sac"] == "collapsed"
    assert plan["desired"]["sac"] == 0
    assert plan["desired"]["ppo"] == 10
    # A live rollout still blocks applying it until the next atomic boundary.
    assert "not_at_atomic_boundary" in plan["blocked_by"]


def test_both_idle_reserves_no_rollout_engines(monkeypatch):
    monkeypatch.setattr(supervisor, "sample_resources", lambda: {"cpu_percent": 40.0})
    config = _config(ceiling=10)
    governor = supervisor._build_governor(config)
    plan, _ = supervisor._governor_cycle(config, _rows("PAUSED", "PAUSED"), governor)
    assert plan["desired"] == {"ppo": 0, "sac": 0}


def test_live_training_blocks_a_reshard(monkeypatch):
    monkeypatch.setattr(supervisor, "sample_resources", lambda: {"cpu_percent": 40.0})
    config = _config()
    governor = supervisor._build_governor(config)
    plan, _ = supervisor._governor_cycle(config, _rows("TRAINING", "TRAINING"), governor)
    assert "not_at_atomic_boundary" in plan["blocked_by"]
    assert not plan["apply"]


def test_headroom_breach_blocks_a_reshard(monkeypatch):
    monkeypatch.setattr(supervisor, "sample_resources", lambda: {"cpu_percent": 96.0})
    config = _config()
    governor = supervisor._build_governor(config)
    plan, resources = supervisor._governor_cycle(
        config, _rows("PAUSED", "PAUSED"), governor
    )
    assert resources["cpu_percent"] == 96.0
    assert "headroom_breach" in plan["blocked_by"] and not plan["apply"]


def test_evaluations_are_listed_for_sequential_scheduling(monkeypatch):
    monkeypatch.setattr(supervisor, "sample_resources", lambda: {"cpu_percent": 40.0})
    config = _config()
    governor = supervisor._build_governor(config)
    plan, _ = supervisor._governor_cycle(
        config, _rows("EVALUATING", "EVALUATING"), governor
    )
    assert plan["evaluation_order"] == ["ppo", "sac"]
    # Both evaluating means neither collects rollouts, so nothing is reserved.
    assert plan["desired"] == {"ppo": 0, "sac": 0}


def test_a_collapse_stopped_run_is_never_restarted():
    collapsed = {
        "checkpoint_selection_state": {"rollback": {"failed_criteria": ["x"]}},
        "message": "baseline_collapse_rollback",
    }
    assert supervisor._collapsed(collapsed)
    assert not supervisor._collapsed({"message": "graceful_stop"})


def test_orphan_cleanup_requires_an_exact_pid_and_create_time(monkeypatch):
    class _Fake:
        pid = 4242

        def __init__(self, create_time):
            self._create = create_time

        def name(self):
            return "Holodeck.exe"

        def create_time(self):
            return self._create

        def terminate(self):
            raise AssertionError("must not terminate a mismatched process")

    import psutil

    monkeypatch.setattr(psutil, "Process", lambda pid: _Fake(100.0))
    # Same PID, different creation time: a recycled PID must never be killed.
    assert supervisor.cleanup_recorded_orphans(
        [{"pid": 4242, "create_time": 999.0}]
    ) == []


def test_wrapper_processes_are_not_counted_as_trainers(monkeypatch):
    class _Process:
        def __init__(self, cmdline, cwd):
            self.info = {"cmdline": cmdline, "pid": 1}
            self._cwd = cwd

        def cwd(self):
            return self._cwd

    worktree = "C:\\work"
    processes = [
        # conda wrapper repeating the module name later in its command line
        _Process(["conda.exe", "run", "-n", "env", "python", "-m",
                  "marine_race_arena.learning.train_ppo_transition"], worktree),
        # the real trainer
        _Process(["python.exe", "-m",
                  "marine_race_arena.learning.train_ppo_transition", "train"], worktree),
    ]
    monkeypatch.setattr(supervisor, "_processes", lambda: processes)
    found = supervisor._trainer_processes(
        "marine_race_arena.learning.train_ppo_transition", worktree
    )
    assert len(found) == 1


def test_governor_plan_is_recorded_for_audit(monkeypatch):
    monkeypatch.setattr(supervisor, "sample_resources", lambda: {"cpu_percent": 40.0})
    config = _config()
    governor = supervisor._build_governor(config)
    plan, _ = supervisor._governor_cycle(config, _rows("PAUSED", "PAUSED"), governor)
    assert plan["schema_version"].startswith("rl_resource_governor")
    assert set(plan) >= {
        "desired", "total_engines", "maximum_total_engines", "apply",
        "blocked_by", "dwell_remaining_seconds", "algorithm_states",
    }
    assert json.dumps(plan)  # must stay JSON-serialisable for status.json
