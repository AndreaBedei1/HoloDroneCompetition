"""Startup serialisation, engine health qualification and retry accounting.

The failure these guard against: PPO and SAC both died with a bare
``BrokenPipeError`` in the vector-env parent and no worker-side traceback, and
the supervisor charged every one of those deaths to the strict
``unexpected_trainer_crash`` budget until PPO was permanently FAILED.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path

import pytest

from marine_race_arena.learning import rl_engine_startup as startup
from marine_race_arena.learning.rl_engine_health import (
    DEFAULT_ENGINE_HEALTH,
    EngineHealthRequirements,
    EngineStartupFailed,
    qualify_engine_start,
)
from marine_race_arena.learning.rl_autonomous_supervisor import (
    NON_FAILURE_REASONS,
    RESTART_BUDGETS,
    budget_exhausted,
    classify_failure_reason,
    restart_budget,
)


# --------------------------------------------------------------- start slots


@pytest.fixture()
def isolated_coordinator(tmp_path, monkeypatch):
    monkeypatch.setattr(startup, "COORDINATOR_ROOT", tmp_path / "startup")
    monkeypatch.delenv(startup._ENV_MAX_STARTS, raising=False)
    monkeypatch.delenv(startup._ENV_STAGGER, raising=False)
    return tmp_path


def test_only_one_engine_may_start_at_a_time(isolated_coordinator):
    with startup.engine_start_slot("first", stagger_seconds=0.0) as held:
        assert held["state"] == startup.STARTING_STATE
        waits = []
        with pytest.raises(TimeoutError):
            with startup.engine_start_slot(
                "second", stagger_seconds=0.0, max_wait_seconds=0.0,
                on_wait=waits.append, sleeper=lambda _: None,
            ):
                pass
        assert waits and waits[0]["state"] == startup.WAITING_STATE
        assert "first" in (waits[0]["current_owners"] or [])


def test_a_released_slot_is_immediately_reusable(isolated_coordinator):
    with startup.engine_start_slot("first", stagger_seconds=0.0):
        pass
    with startup.engine_start_slot("second", stagger_seconds=0.0) as held:
        assert held["slot"] == 0


def test_waiting_for_a_slot_is_not_an_error(isolated_coordinator):
    """A queued starter must eventually acquire, reporting WAITING meanwhile."""

    released = {"done": False}

    def sleeper(_delay):
        released["done"] = True

    holder = startup.engine_start_slot("holder", stagger_seconds=0.0)
    holder.__enter__()
    waits = []

    def on_wait(record):
        waits.append(record)
        if len(waits) == 2:
            holder.__exit__(None, None, None)

    with startup.engine_start_slot(
        "queued", stagger_seconds=0.0, on_wait=on_wait, sleeper=sleeper
    ) as held:
        assert held["wait_cycles"] >= 2
    assert all(row["state"] == startup.WAITING_STATE for row in waits)


def test_extra_slots_allow_configured_concurrency(isolated_coordinator, monkeypatch):
    monkeypatch.setenv(startup._ENV_MAX_STARTS, "2")
    assert startup.max_concurrent_engine_starts() == 2
    with startup.engine_start_slot("a", stagger_seconds=0.0) as first:
        with startup.engine_start_slot("b", stagger_seconds=0.0) as second:
            assert {first["slot"], second["slot"]} == {0, 1}


def test_stagger_is_applied_only_after_a_successful_start(isolated_coordinator):
    slept = []
    with startup.engine_start_slot(
        "ok", stagger_seconds=4.0, sleeper=slept.append
    ):
        pass
    assert slept == [4.0]

    slept.clear()
    with pytest.raises(RuntimeError):
        with startup.engine_start_slot(
            "boom", stagger_seconds=4.0, sleeper=slept.append
        ):
            raise RuntimeError("engine died")
    assert slept == [], "a failed candidate must be retried promptly"


def test_a_dead_slot_owner_is_reported_as_stale(isolated_coordinator):
    record = {"pid": 2 ** 30, "acquired": time.time()}
    assert startup.slot_owner_is_stale(record) is True
    assert startup.slot_owner_is_stale({"pid": os.getpid(), "acquired": time.time()}) is False


def test_slot_files_are_never_read_while_locked(isolated_coordinator):
    """Regression: reading the locked byte is what broke the evaluation lock."""

    source = Path(startup.__file__).read_text(encoding="utf-8")
    body = source.split("def engine_start_slot", 1)[1]
    assert ".read(" not in body
    assert '"xb"' in source


# ------------------------------------------------------------ engine health


class _FakeEngine:
    def __init__(self, uuid="u-1", frames=None, fail_at=None, world=None):
        self._uuid = uuid
        self._frames = frames
        self._fail_at = fail_at
        self._calls = 0
        if world is not None:
            self._world_process = world

    def tick(self):
        self._calls += 1
        if self._fail_at is not None and self._calls >= self._fail_at:
            raise RuntimeError("engine went away")
        if self._frames is None:
            return {"PoseSensor": float(self._calls)}
        return self._frames


class _DeadWorld:
    pid = 4321

    def poll(self):
        return 3

    returncode = 3


def test_a_healthy_engine_qualifies():
    report = qualify_engine_start(_FakeEngine(), environment_name="OpenWater")
    assert report["healthy"] is True
    assert report["ticks_completed"] == DEFAULT_ENGINE_HEALTH.qualifying_ticks
    assert report["uuid"] == "u-1"
    assert "PoseSensor" in report["sensor_keys"]


def test_a_live_process_without_a_uuid_is_not_healthy():
    with pytest.raises(EngineStartupFailed, match="no HoloOcean UUID"):
        qualify_engine_start(_FakeEngine(uuid=""))


def test_an_engine_that_exited_is_rejected_before_ticking():
    with pytest.raises(EngineStartupFailed, match="exited during startup") as info:
        qualify_engine_start(_FakeEngine(world=_DeadWorld()))
    assert info.value.diagnostics["engine_returncode"] == 3
    assert info.value.diagnostics["stage"] == "engine_exited"


def test_an_engine_that_dies_mid_qualification_is_rejected():
    with pytest.raises(EngineStartupFailed, match="failed while ticking") as info:
        qualify_engine_start(_FakeEngine(fail_at=5))
    assert info.value.diagnostics["ticks_completed"] == 4


def test_a_frozen_initial_frame_is_rejected():
    """A stale repeated frame means sensors never really started."""

    with pytest.raises(EngineStartupFailed, match="frozen frame"):
        qualify_engine_start(_FakeEngine(frames={"PoseSensor": 1.0}))


def test_a_missing_sensor_packet_is_rejected():
    with pytest.raises(EngineStartupFailed, match="sensor packet"):
        qualify_engine_start(
            _FakeEngine(frames=None, uuid="u"),
            requirements=EngineHealthRequirements(qualifying_ticks=0),
        )


def test_qualification_diagnostics_are_serialisable():
    report = qualify_engine_start(_FakeEngine())
    assert json.dumps(report)


def test_qualification_is_bounded_by_its_timeout():
    clock = iter([0.0, 0.0] + [500.0] * 50)
    with pytest.raises(EngineStartupFailed, match="qualifying ticks"):
        qualify_engine_start(_FakeEngine(), now=lambda: next(clock))


# --------------------------------------------------- restart classification


def _run_dir(tmp_path, *, txt=None, js=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    if txt is not None:
        (tmp_path / "failure.txt").write_text(txt, encoding="utf-8")
    if js is not None:
        (tmp_path / "failure.json").write_text(json.dumps(js), encoding="utf-8")
    return tmp_path


BROKEN_PIPE = (
    "Traceback (most recent call last):\n"
    "BrokenPipeError: [WinError 109] Pipe terminata\n"
    "EOFError\n"
)


def test_a_broken_pipe_is_an_engine_problem_not_a_learner_crash(tmp_path):
    run_dir = _run_dir(tmp_path / "ppo", txt=BROKEN_PIPE)
    assert classify_failure_reason(run_dir, {}) == "engine_worker_start_retry"


def test_sac_failure_json_is_actually_read(tmp_path):
    """Regression: only failure.txt was read, so SAC always looked generic."""

    run_dir = _run_dir(tmp_path / "sac", js={"error": "EOFError()", "traceback": BROKEN_PIPE})
    assert classify_failure_reason(run_dir, {}) == "engine_worker_start_retry"


def test_engine_retries_get_a_larger_budget_than_a_real_crash():
    assert restart_budget("engine_worker_start_retry") > restart_budget(
        "unexpected_trainer_crash"
    )
    assert not budget_exhausted("engine_worker_start_retry", 3)
    assert budget_exhausted("unexpected_trainer_crash", 3)


def test_waiting_reasons_never_exhaust_a_budget():
    for reason in NON_FAILURE_REASONS:
        assert RESTART_BUDGETS[reason] is None
        assert not budget_exhausted(reason, 10_000)


def test_a_genuine_crash_still_uses_the_strict_budget(tmp_path):
    run_dir = _run_dir(tmp_path / "x", txt="ValueError: contract mismatch in learner")
    assert classify_failure_reason(run_dir, {}) == "unexpected_trainer_crash"


def test_collapse_and_invalid_checkpoint_are_never_retried():
    assert restart_budget("learning_collapse") == 0
    assert restart_budget("invalid_checkpoint") == 0


def test_an_engine_start_slot_wait_is_not_a_failure(tmp_path):
    run_dir = _run_dir(tmp_path / "w", txt="state WAITING_FOR_ENGINE_START_SLOT")
    reason = classify_failure_reason(run_dir, {})
    assert reason == "engine_start_slot_wait"
    assert reason in NON_FAILURE_REASONS
