"""Beacon noise schema migration: units, invariants and legacy equivalence.

The beacon measurement model split the deprecated scalar ``noise_std`` into two
dimensioned parameters, ``angular_noise_std_deg`` (degrees, bearing/elevation)
and ``range_noise_std_m`` (metres, range). This module proves the split is a
pure schema/units change: for the migrated value-equal configuration it
reproduces the pre-refactor packet stream bit for bit, drawing bearing, then
elevation, then range, and applying range noise exactly once.

No HoloOcean run is involved; these are deterministic unit tests over the
beacon model and the configuration loader.
"""

from __future__ import annotations

import math
import random
import warnings

import pytest

from marine_race_arena.arena.beacon import (
    Beacon,
    _norm,
    _subtract,
    _wrap_degrees,
)
from marine_race_arena.arena.beacon_manager import BeaconManager
from marine_race_arena.config.loader import parse_track_config
from marine_race_arena.config.validation import validate_track_config

OFFICIAL_NOISE_LEVELS = (0.2, 0.45, 0.6)


# --------------------------------------------------------------------------- #
# Test-only reimplementation of the PRE-REFACTOR beacon receive algorithm.
# This is a faithful copy of the single-scalar ``noise_std`` code path as it
# existed before the migration; it is the ground truth the new model must match.
# --------------------------------------------------------------------------- #
def legacy_receive(
    *,
    position,
    range_m,
    noise_std,
    dropout_probability,
    receiver_position,
    receiver_yaw_deg,
    received_at_s,
    rng,
    beacon_id="B01",
):
    delta = _subtract(position, receiver_position)
    distance = _norm(delta)
    if distance > range_m:
        return None
    if dropout_probability > 0.0 and rng.random() < dropout_probability:
        return None
    horizontal_distance = math.hypot(delta[0], delta[1])
    global_bearing = math.degrees(math.atan2(delta[1], delta[0]))
    relative_bearing = _wrap_degrees(global_bearing - receiver_yaw_deg)
    elevation = math.degrees(math.atan2(delta[2], horizontal_distance))
    range_measurement = distance
    if noise_std > 0.0:
        relative_bearing += rng.gauss(0.0, noise_std)
        elevation += rng.gauss(0.0, noise_std)
        range_measurement = max(0.0, range_measurement + rng.gauss(0.0, noise_std))
    signal_strength = max(0.0, 1.0 - range_measurement / range_m)
    return {
        "beacon_id": beacon_id,
        "bearing_deg": _wrap_degrees(relative_bearing),
        "elevation_deg": elevation,
        "range_m": range_measurement,
        "signal_strength": signal_strength,
        "received_at_s": received_at_s,
    }


def _packet_rng(seed, beacon_id, receiver_id, index):
    # Identical keying to BeaconManager._packet_rng.
    return random.Random(f"{seed}|{beacon_id}|{receiver_id}|{index}")


GEOMETRIES = [
    ((10.0, 0.0, -4.0), (0.0, 0.0, -4.0), 0.0),
    ((3.5, -2.0, -3.0), (1.0, 1.0, -5.0), 37.0),
    ((-8.0, 6.0, -2.0), (2.0, -1.0, -4.0), -120.0),
    ((40.0, 40.0, -1.0), (0.0, 0.0, -6.0), 200.0),
]


# --------------------------------------------------------------------------- #
# Legacy equivalence (the central guarantee).
# --------------------------------------------------------------------------- #
def test_migrated_packets_match_legacy_bitwise():
    """New(angular=range=s) == legacy(noise_std=s), field for field, incl. dropout."""
    for noise in (0.0, *OFFICIAL_NOISE_LEVELS):
        for seed in (0, 1, 2, 7, 123):
            for receiver in ("rov_a", "rov_b"):
                for index in range(6):
                    for beacon_pos, rx_pos, yaw in GEOMETRIES:
                        for dropout in (0.0, 0.3):
                            new_beacon = Beacon(
                                id="B01",
                                gate_id="G01",
                                position=beacon_pos,
                                range_m=90.0,
                                angular_noise_std_deg=noise,
                                range_noise_std_m=noise,
                                dropout_probability=dropout,
                                update_rate_hz=10.0,
                            )
                            rng_new = _packet_rng(seed, "B01", receiver, index)
                            rng_legacy = _packet_rng(seed, "B01", receiver, index)
                            new_packet = new_beacon.receive(
                                receiver_position=rx_pos,
                                receiver_yaw_deg=yaw,
                                received_at_s=float(index),
                                rng=rng_new,
                            )
                            legacy_packet = legacy_receive(
                                position=beacon_pos,
                                range_m=90.0,
                                noise_std=noise,
                                dropout_probability=dropout,
                                receiver_position=rx_pos,
                                receiver_yaw_deg=yaw,
                                received_at_s=float(index),
                                rng=rng_legacy,
                            )
                            # Same received/dropout decision.
                            assert (new_packet is None) == (legacy_packet is None)
                            if new_packet is None:
                                continue
                            # Exact equality of every physical field.
                            for key in (
                                "beacon_id",
                                "bearing_deg",
                                "elevation_deg",
                                "range_m",
                                "signal_strength",
                                "received_at_s",
                            ):
                                assert new_packet[key] == legacy_packet[key], (
                                    key,
                                    noise,
                                    seed,
                                    receiver,
                                    index,
                                )


def test_manager_stream_matches_legacy_over_a_trajectory():
    """End-to-end through BeaconManager: full delivered stream equals legacy."""
    for noise in OFFICIAL_NOISE_LEVELS:
        for seed in (0, 4, 41):
            beacon = Beacon(
                id="B01",
                gate_id="G01",
                position=(12.0, -3.0, -3.5),
                range_m=90.0,
                angular_noise_std_deg=noise,
                range_noise_std_m=noise,
                dropout_probability=0.1,
                update_rate_hz=10.0,
            )
            manager = BeaconManager([beacon], seed=seed)
            receiver = "rov"
            got = []
            for step in range(50):
                t = step * 0.1
                # Moving receiver so range/bearing vary along the path.
                pos = (step * 0.2, -step * 0.1, -4.0)
                packets = manager.receive(
                    receiver_id=receiver,
                    receiver_position=pos,
                    receiver_yaw_deg=step * 2.0,
                    time_s=t,
                )
                got.append((step, packets))
            # Reconstruct the legacy stream using the same rng keying and the
            # same transmission-index gating the manager applies.
            last_index = -1
            for step, packets in got:
                t = step * 0.1
                index = int(math.floor(t * 10.0 + 1e-9))
                pos = (step * 0.2, -step * 0.1, -4.0)
                if index < 0 or index <= last_index:
                    assert packets == []
                    continue
                last_index = index
                rng_legacy = _packet_rng(seed, "B01", receiver, index)
                legacy_packet = legacy_receive(
                    position=(12.0, -3.0, -3.5),
                    range_m=90.0,
                    noise_std=noise,
                    dropout_probability=0.1,
                    receiver_position=pos,
                    receiver_yaw_deg=step * 2.0,
                    received_at_s=t,
                    rng=rng_legacy,
                )
                if legacy_packet is None:
                    assert packets == []
                else:
                    assert len(packets) == 1
                    assert packets[0] == legacy_packet


# --------------------------------------------------------------------------- #
# Units and invariants.
# --------------------------------------------------------------------------- #
def test_zero_noise_leaves_measurements_clean():
    beacon = Beacon(
        id="B01", gate_id="G01", position=(10.0, 0.0, -4.0), range_m=50.0,
        angular_noise_std_deg=0.0, range_noise_std_m=0.0,
        dropout_probability=0.0, update_rate_hz=10.0,
    )
    p = beacon.receive(
        receiver_position=(0.0, 0.0, -4.0), receiver_yaw_deg=0.0,
        received_at_s=0.0, rng=random.Random(0),
    )
    assert p["range_m"] == pytest.approx(10.0, abs=1e-12)
    assert p["bearing_deg"] == pytest.approx(0.0, abs=1e-12)
    assert p["elevation_deg"] == pytest.approx(0.0, abs=1e-12)


def test_angular_noise_only_does_not_perturb_range():
    beacon = Beacon(
        id="B01", gate_id="G01", position=(10.0, 3.0, -6.0), range_m=90.0,
        angular_noise_std_deg=5.0, range_noise_std_m=0.0,
        dropout_probability=0.0, update_rate_hz=10.0,
    )
    clean_range = _norm(_subtract((10.0, 3.0, -6.0), (0.0, 0.0, -4.0)))
    perturbed = set()
    for seed in range(25):
        p = beacon.receive(
            receiver_position=(0.0, 0.0, -4.0), receiver_yaw_deg=0.0,
            received_at_s=0.0, rng=random.Random(seed),
        )
        assert p["range_m"] == pytest.approx(clean_range, abs=1e-12)
        perturbed.add(round(p["bearing_deg"], 9))
    assert len(perturbed) > 1  # bearing actually varies with the angular sigma


def test_range_noise_only_does_not_perturb_angles():
    beacon = Beacon(
        id="B01", gate_id="G01", position=(10.0, 3.0, -6.0), range_m=90.0,
        angular_noise_std_deg=0.0, range_noise_std_m=2.0,
        dropout_probability=0.0, update_rate_hz=10.0,
    )
    delta = _subtract((10.0, 3.0, -6.0), (0.0, 0.0, -4.0))
    clean_bearing = _wrap_degrees(math.degrees(math.atan2(delta[1], delta[0])))
    clean_elev = math.degrees(math.atan2(delta[2], math.hypot(delta[0], delta[1])))
    ranges = set()
    for seed in range(25):
        p = beacon.receive(
            receiver_position=(0.0, 0.0, -4.0), receiver_yaw_deg=0.0,
            received_at_s=0.0, rng=random.Random(seed),
        )
        assert p["bearing_deg"] == pytest.approx(clean_bearing, abs=1e-12)
        assert p["elevation_deg"] == pytest.approx(clean_elev, abs=1e-12)
        ranges.add(round(p["range_m"], 9))
    assert len(ranges) > 1  # range actually varies with the range sigma


def test_range_noise_applied_exactly_once():
    """The range perturbation equals a single Gaussian draw, not two."""
    beacon = Beacon(
        id="B01", gate_id="G01", position=(10.0, 0.0, -4.0), range_m=90.0,
        angular_noise_std_deg=0.0, range_noise_std_m=3.0,
        dropout_probability=0.0, update_rate_hz=10.0,
    )
    for seed in range(20):
        rng_beacon = random.Random(seed)
        p = beacon.receive(
            receiver_position=(0.0, 0.0, -4.0), receiver_yaw_deg=0.0,
            received_at_s=0.0, rng=rng_beacon,
        )
        # With no angular noise the model still draws bearing and elevation
        # (both sigma 0 -> 0.0) before the single range draw, so replay that
        # exact sequence: gauss(0,0), gauss(0,0), gauss(0,3).
        rng_replay = random.Random(seed)
        rng_replay.gauss(0.0, 0.0)
        rng_replay.gauss(0.0, 0.0)
        expected_range = max(0.0, 10.0 + rng_replay.gauss(0.0, 3.0))
        assert p["range_m"] == pytest.approx(expected_range, abs=1e-12)


def test_signal_strength_derives_from_noisy_range():
    beacon = Beacon(
        id="B01", gate_id="G01", position=(10.0, 0.0, -4.0), range_m=50.0,
        angular_noise_std_deg=0.0, range_noise_std_m=1.0,
        dropout_probability=0.0, update_rate_hz=10.0,
    )
    p = beacon.receive(
        receiver_position=(0.0, 0.0, -4.0), receiver_yaw_deg=0.0,
        received_at_s=0.0, rng=random.Random(11),
    )
    assert p["signal_strength"] == pytest.approx(max(0.0, 1.0 - p["range_m"] / 50.0), abs=1e-12)


def test_seeded_reproducibility_of_new_model():
    def stream(seed):
        beacon = Beacon(
            id="B01", gate_id="G01", position=(10.0, 2.0, -4.0), range_m=90.0,
            angular_noise_std_deg=0.5, range_noise_std_m=0.5,
            dropout_probability=0.0, update_rate_hz=10.0,
        )
        rng = _packet_rng(seed, "B01", "rov", 3)
        return beacon.receive(
            receiver_position=(0.0, 0.0, -4.0), receiver_yaw_deg=10.0,
            received_at_s=0.3, rng=rng,
        )
    assert stream(5) == stream(5)
    assert stream(5) != stream(6)


# --------------------------------------------------------------------------- #
# Configuration loader: migration, deprecation and rejection.
# --------------------------------------------------------------------------- #
import copy
import json as _json

_BASE_TRACK = "marine_race_arena/tracks/tests/three_gate_s_curve.json"


def _minimal_track(beacon_block):
    """A valid test track with its global beacon block replaced."""
    raw = copy.deepcopy(_json.load(open(_BASE_TRACK)))
    raw["beacon"] = beacon_block
    return raw


def test_new_fields_load_independently():
    cfg = parse_track_config(
        _minimal_track({"range_m": 90.0, "angular_noise_std_deg": 0.3, "range_noise_std_m": 1.2})
    )
    b = cfg.gates[0].beacon
    assert b.angular_noise_std_deg == 0.3
    assert b.range_noise_std_m == 1.2


def test_legacy_scalar_maps_to_both_with_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = parse_track_config(_minimal_track({"range_m": 90.0, "noise_std": 0.45}))
    b = cfg.gates[0].beacon
    assert b.angular_noise_std_deg == 0.45
    assert b.range_noise_std_m == 0.45
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_mixing_legacy_and_new_is_rejected():
    from marine_race_arena.config.loader import TrackConfigLoadError

    with pytest.raises(TrackConfigLoadError):
        parse_track_config(
            _minimal_track({"range_m": 90.0, "noise_std": 0.2, "angular_noise_std_deg": 0.2})
        )
    with pytest.raises(TrackConfigLoadError):
        parse_track_config(
            _minimal_track({"range_m": 90.0, "noise_std": 0.2, "range_noise_std_m": 0.2})
        )


def test_negative_std_rejected_by_validation():
    for bad in ({"angular_noise_std_deg": -0.1}, {"range_noise_std_m": -0.1}):
        block = {"range_m": 90.0}
        block.update(bad)
        cfg = parse_track_config(_minimal_track(block))
        result = validate_track_config(cfg)
        assert not result.ok
        assert any("must be zero or positive" in e for e in result.errors)


def test_official_tracks_migrated_value_equal():
    import json

    for name, expected in (
        ("horseshoe_bay", 0.2),
        ("vertical_serpent", 0.45),
        ("mixed_endurance", 0.6),
    ):
        raw = json.load(open(f"marine_race_arena/tracks/marine_race_{name}.json"))
        beacon = raw["beacon"]
        assert "noise_std" not in beacon
        assert beacon["angular_noise_std_deg"] == expected
        assert beacon["range_noise_std_m"] == expected
