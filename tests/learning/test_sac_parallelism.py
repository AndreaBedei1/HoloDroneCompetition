from __future__ import annotations

from contextlib import contextmanager

import pytest

from marine_race_arena.learning import holoocean_capacity


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

