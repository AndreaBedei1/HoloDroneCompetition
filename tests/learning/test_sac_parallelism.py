from __future__ import annotations

from contextlib import contextmanager

import pytest

# The RL stack (gymnasium/torch/SB3) lives in requirements-rl.txt and is not
# installed in the benchmark environment; skip rather than fail collection.
pytest.importorskip("gymnasium")

from marine_race_arena.learning import holoocean_capacity
from marine_race_arena.learning import benchmark_sac_parallelism
from marine_race_arena.learning import train_sac_transition


def test_unique_holoocean_uuid_ownership():
    holoocean_capacity.assert_unique_holoocean_uuids([
        {"holoocean_uuid": "a"}, {"holoocean_uuid": "b"},
    ])
    with pytest.raises(RuntimeError, match="duplicate"):
        holoocean_capacity.assert_unique_holoocean_uuids([
            {"holoocean_uuid": "a"}, {"holoocean_uuid": "a"},
        ])


def test_global_ten_engine_cap_counts_existing_ppo(monkeypatch, tmp_path):
    monkeypatch.setattr(holoocean_capacity, "_REGISTRY", tmp_path)
    monkeypatch.setattr(holoocean_capacity, "_LOCK", tmp_path / "registry.lock")
    monkeypatch.setattr(holoocean_capacity, "_RESERVATIONS", tmp_path / "reservations.json")
    monkeypatch.setattr(
        holoocean_capacity,
        "active_holodeck_processes",
        lambda: [{"pid": 1}, {"pid": 2}],
    )
    with holoocean_capacity.reserve_holoocean_engines(8, owner="SAC"):
        with pytest.raises(RuntimeError, match="cap"):
            with holoocean_capacity.reserve_holoocean_engines(1, owner="preview"):
                pass
    assert not holoocean_capacity._read_reservations()


def test_capacity_accounting_never_terminates_cross_run_processes(monkeypatch, tmp_path):
    monkeypatch.setattr(holoocean_capacity, "_REGISTRY", tmp_path)
    monkeypatch.setattr(holoocean_capacity, "_LOCK", tmp_path / "registry.lock")
    monkeypatch.setattr(holoocean_capacity, "_RESERVATIONS", tmp_path / "reservations.json")
    engines = [{"pid": 101, "uuid": "ppo-a"}, {"pid": 102, "uuid": "ppo-b"}]
    monkeypatch.setattr(holoocean_capacity, "active_holodeck_processes", lambda: list(engines))
    with holoocean_capacity.reserve_holoocean_engines(2, owner="SAC"):
        assert engines == holoocean_capacity.active_holodeck_processes()
    assert engines == holoocean_capacity.active_holodeck_processes()


def test_sac_validates_holoocean_uuid_only_after_lazy_reset():
    class LazyIdentityEnv:
        initialized = False

        def reset(self):
            self.initialized = True
            return [[0.0] * 35]

        def env_method(self, name):
            assert name == "worker_identity"
            return [{
                "holoocean_uuid": "sac-worker-0" if self.initialized else None,
            }]

    env = LazyIdentityEnv()
    with pytest.raises(RuntimeError, match="did not report"):
        train_sac_transition._validate_initialized_worker_identities(
            env, {"adapter": "holoocean"}
        )
    env.reset()
    train_sac_transition._validate_initialized_worker_identities(
        env, {"adapter": "holoocean"}
    )


def test_capacity_benchmark_does_not_call_recycled_ppo_engine_an_orphan(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        benchmark_sac_parallelism,
        "active_holodeck_processes",
        lambda: [{"pid": 202, "parent_pid": 20, "uuid": "ppo-new"}],
    )
    row = {
        "ok": True,
        "sensor_contract_valid": True,
        "total_active_engines": 3,
        "engines_before": [{"pid": 201, "parent_pid": 20, "uuid": "ppo-old"}],
        "resources": {"system_ram_percent": 20.0},
        "ppo_progress_rows_before": 0,
        "ppo_throughput_before": None,
    }
    result = benchmark_sac_parallelism.finalize_case(
        row, tmp_path / "missing-progress.jsonl"
    )
    assert result["orphan_engine_pids"] == []
    assert result["stable"] is True
