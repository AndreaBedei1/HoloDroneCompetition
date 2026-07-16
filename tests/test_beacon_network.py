"""Tests for independent beacon transmissions and sequential ID validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from marine_race_arena.arena.beacon import BEACON_PACKET_FIELDS, Beacon
from marine_race_arena.arena.beacon_manager import BeaconManager
from marine_race_arena.config.loader import load_track_config, parse_track_config
from marine_race_arena.config.validation import validate_track_config

TRACK = "marine_race_arena/tracks/tests/three_gate_s_curve.json"
OFFICIAL_TRACKS = (
    "marine_race_arena/tracks/marine_race_horseshoe_bay.json",
    "marine_race_arena/tracks/marine_race_vertical_serpent.json",
    "marine_race_arena/tracks/marine_race_mixed_endurance.json",
)


def make_beacon(
    beacon_id="B01",
    position=(10.0, 0.0, -4.0),
    range_m=50.0,
    noise_std=0.0,
    dropout=0.0,
    rate_hz=10.0,
):
    return Beacon(
        id=beacon_id,
        gate_id="G01",
        position=position,
        range_m=range_m,
        noise_std=noise_std,
        dropout_probability=dropout,
        update_rate_hz=rate_hz,
    )


def receive_all(manager, *, receiver="rov", position=(0.0, 0.0, -4.0), yaw=0.0, t=0.0):
    return manager.receive(
        receiver_id=receiver,
        receiver_position=position,
        receiver_yaw_deg=yaw,
        time_s=t,
    )


# ---------------------------------------------------------------- packets


def test_in_range_transmission_creates_packet_with_only_approved_fields():
    manager = BeaconManager([make_beacon()], seed=0)
    packets = receive_all(manager, t=0.0)
    assert len(packets) == 1
    packet = packets[0]
    assert sorted(packet.keys()) == sorted(BEACON_PACKET_FIELDS)
    assert packet["beacon_id"] == "B01"
    assert packet["range_m"] == pytest.approx(10.0, abs=1e-6)
    assert packet["bearing_deg"] == pytest.approx(0.0, abs=1e-6)


def test_out_of_range_transmission_creates_no_packet():
    manager = BeaconManager([make_beacon(range_m=5.0)], seed=0)
    packets = receive_all(manager, t=0.0)
    assert packets == []


def test_dropout_creates_no_packet_and_is_seed_reproducible():
    outcomes = []
    for _ in range(2):
        manager = BeaconManager([make_beacon(dropout=0.5)], seed=7)
        received = []
        for step in range(40):
            t = step * 0.1
            received.append(bool(receive_all(manager, t=t)))
        outcomes.append(received)
    assert outcomes[0] == outcomes[1], "same seed must reproduce the same dropouts"
    assert any(outcomes[0]) and not all(outcomes[0]), "0.5 dropout should mix hits and misses"


def test_different_seeds_change_dropout_pattern():
    patterns = []
    for seed in (1, 2):
        manager = BeaconManager([make_beacon(dropout=0.5)], seed=seed)
        patterns.append(
            tuple(bool(receive_all(manager, t=step * 0.1)) for step in range(60))
        )
    assert patterns[0] != patterns[1]


def test_noise_is_seed_reproducible_and_order_independent():
    def collect(seed, receivers):
        manager = BeaconManager([make_beacon(noise_std=0.5)], seed=seed)
        values = {}
        for step in range(10):
            t = step * 0.1
            for receiver in receivers:
                for packet in manager.receive(
                    receiver_id=receiver,
                    receiver_position=(0.0, 0.0, -4.0),
                    receiver_yaw_deg=0.0,
                    time_s=t,
                ):
                    values.setdefault(receiver, []).append(round(packet["range_m"], 9))
        return values

    forward = collect(3, ["a", "b"])
    reversed_order = collect(3, ["b", "a"])
    assert forward == reversed_order, "per-receiver streams must not depend on call order"


def test_update_rate_scheduling_delivers_at_most_one_packet_per_transmission():
    manager = BeaconManager([make_beacon(rate_hz=2.0)], seed=0)  # transmissions every 0.5 s
    deliveries = []
    for step in range(20):  # dt = 0.1 -> t in [0.0, 1.9]
        t = step * 0.1
        deliveries.append(len(receive_all(manager, t=t)))
    # Transmissions at t = 0.0, 0.5, 1.0, 1.5 -> exactly four deliveries.
    assert sum(deliveries) == 4
    assert max(deliveries) == 1


def test_all_beacons_transmit_independently_of_any_progress():
    beacons = [
        make_beacon("B01", position=(5.0, 0.0, -4.0)),
        make_beacon("B02", position=(0.0, 5.0, -4.0)),
        make_beacon("B03", position=(-5.0, 0.0, -4.0)),
    ]
    manager = BeaconManager(beacons, seed=0)
    packets = receive_all(manager, t=0.0)
    assert sorted(p["beacon_id"] for p in packets) == ["B01", "B02", "B03"]


def test_signal_strength_derives_from_the_noisy_range():
    manager = BeaconManager([make_beacon(noise_std=1.0, range_m=50.0)], seed=11)
    for step in range(20):
        for packet in receive_all(manager, t=step * 0.1):
            expected = max(0.0, 1.0 - packet["range_m"] / 50.0)
            assert packet["signal_strength"] == pytest.approx(expected, abs=1e-9)


def test_received_at_uses_the_provided_local_clock():
    manager = BeaconManager([make_beacon()], seed=0)
    packets = manager.receive(
        receiver_id="rov",
        receiver_position=(0.0, 0.0, -4.0),
        receiver_yaw_deg=0.0,
        time_s=12.0,
        received_at_s=2.5,
    )
    assert packets and packets[0]["received_at_s"] == pytest.approx(2.5)


# ------------------------------------------------------------- validation


def _raw_track(path=TRACK):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_official_tracks_validate_with_sequential_beacons():
    for track in OFFICIAL_TRACKS:
        config = load_track_config(track)
        sequence = config.track.gate_sequence
        for index, gate_id in enumerate(sequence):
            beacon = config.gate_by_id(gate_id).beacon
            assert beacon is not None and beacon.enabled
            assert beacon.id == f"B{index + 1:02d}"


def test_explicit_matching_beacon_ids_are_accepted():
    raw = _raw_track()
    for index, gate in enumerate(raw["gates"]):
        gate["beacon"] = {"id": f"B{index + 1:02d}"}
    config = parse_track_config(raw)
    assert validate_track_config(config).ok


def test_duplicate_beacon_ids_fail_validation():
    raw = _raw_track()
    raw["gates"][1]["beacon"] = {"id": "B01"}
    result = validate_track_config(parse_track_config(raw))
    assert any("Duplicated beacon id 'B01'" in error for error in result.errors)


def test_missing_beacon_fails_validation():
    raw = _raw_track()
    raw["gates"][1]["beacon"] = {"enabled": False}
    result = validate_track_config(parse_track_config(raw))
    assert any("no enabled" in error for error in result.errors)


def test_reordered_beacon_ids_fail_validation():
    raw = _raw_track()
    raw["gates"][0]["beacon"] = {"id": "B02"}
    raw["gates"][1]["beacon"] = {"id": "B01"}
    result = validate_track_config(parse_track_config(raw))
    assert any("does not match the required sequential id" in error for error in result.errors)


def test_non_contiguous_beacon_ids_fail_validation():
    raw = _raw_track()
    raw["gates"][2]["beacon"] = {"id": "B07"}
    result = validate_track_config(parse_track_config(raw))
    assert any("does not match the required sequential id" in error for error in result.errors)


def test_gate_outside_sequence_with_explicit_beacon_fails_validation():
    raw = _raw_track()
    extra = copy.deepcopy(raw["gates"][-1])
    extra["id"] = "G99"
    extra["position"] = list(raw["gates"][0]["position"])
    extra["position"][1] += 1.0
    extra["beacon"] = {"id": "B99"}
    raw["gates"].append(extra)
    config = parse_track_config(raw)
    g99 = config.gate_by_id("G99")
    assert g99.beacon is not None
    result = validate_track_config(config)
    assert any("not in track.gate_sequence" in error for error in result.errors)
