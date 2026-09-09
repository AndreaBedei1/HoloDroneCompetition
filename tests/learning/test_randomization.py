"""Tests for deterministic seeded start-pose / beacon-noise randomization."""

import math

import numpy as np
import pytest

from marine_race_arena.config.loader import load_track_config
from marine_race_arena.learning.episode import RaceEpisode, build_single_vehicle_race
from marine_race_arena.learning.randomization import StartRandomization, apply_start_randomization

TRACK = "marine_race_arena/tracks/training/stage1_single_gate.json"
SPEC = StartRandomization(
    lateral_offset_m=1.0,
    depth_offset_m=0.5,
    yaw_offset_deg=15.0,
    beacon_angular_noise_std_deg=0.2,
    beacon_range_noise_std_m=0.2,
    beacon_dropout_probability=0.02,
)


def test_noop_detection():
    assert StartRandomization().is_noop()
    assert not SPEC.is_noop()


def test_apply_is_deterministic_and_within_range():
    config = load_track_config(TRACK)
    pos = (-5.0, 0.0, -4.0)
    rot = (0.0, 0.0, 0.0)
    c1, p1, r1, a1 = apply_start_randomization(config, pos, rot, SPEC, seed=3)
    c2, p2, r2, a2 = apply_start_randomization(config, pos, rot, SPEC, seed=3)
    assert p1 == p2 and r1 == r2 and a1 == a2  # deterministic for a fixed seed
    # offsets within declared ranges
    assert abs(a1["lateral_offset_m"]) <= 1.0
    assert abs(a1["depth_offset_m"]) <= 0.5
    assert abs(a1["yaw_offset_deg"]) <= 15.0
    assert p1[1] == pytest.approx(pos[1] + a1["lateral_offset_m"])
    assert p1[2] == pytest.approx(pos[2] + a1["depth_offset_m"])
    assert r1[2] == pytest.approx(rot[2] + a1["yaw_offset_deg"])
    # beacon noise overridden in the returned config
    assert c1.beacon.angular_noise_std_deg == pytest.approx(0.2)
    assert c1.beacon.range_noise_std_m == pytest.approx(0.2)
    assert c1.beacon.dropout_probability == pytest.approx(0.02)


def test_different_seeds_give_different_start():
    config = load_track_config(TRACK)
    pos, rot = (-5.0, 0.0, -4.0), (0.0, 0.0, 0.0)
    _, p_a, _, _ = apply_start_randomization(config, pos, rot, SPEC, seed=1)
    _, p_b, _, _ = apply_start_randomization(config, pos, rot, SPEC, seed=2)
    assert p_a != p_b


def test_noop_leaves_config_and_pose_unchanged():
    config = load_track_config(TRACK)
    pos, rot = (-5.0, 0.0, -4.0), (0.0, 0.0, 0.0)
    c, p, r, applied = apply_start_randomization(config, pos, rot, StartRandomization(), seed=0)
    assert p == pos and r == rot
    assert c is config  # no beacon override -> same object


def test_episode_records_applied_randomization():
    ep = RaceEpisode(TRACK, seed=5, dt=0.1, adapter="fallback", allow_fallback=True, max_steps=5, start_randomization=SPEC)
    ep.reset()
    applied = ep.context.applied_randomization
    assert applied is not None
    assert applied["seed"] == 5
    assert abs(applied["lateral_offset_m"]) <= 1.0
    ep.close()


def test_episode_randomization_is_reproducible_per_seed():
    a = RaceEpisode(TRACK, seed=8, adapter="fallback", allow_fallback=True, max_steps=3, start_randomization=SPEC)
    b = RaceEpisode(TRACK, seed=8, adapter="fallback", allow_fallback=True, max_steps=3, start_randomization=SPEC)
    a.reset()
    b.reset()
    assert a.context.applied_randomization == b.context.applied_randomization
    a.close()
    b.close()


def test_build_without_randomization_has_none():
    ctx = build_single_vehicle_race(TRACK, seed=0, adapter="fallback", allow_fallback=True)
    assert ctx.applied_randomization is None
    ctx.adapter.close()


@pytest.mark.parametrize("yaw", [0.0, 90.0, -45.0, 30.0])
def test_offsets_applied_in_local_approach_frame(yaw):
    """World offset is the body-frame (longitudinal, lateral) rotated by the initial yaw."""
    config = load_track_config(TRACK)
    pos, rot = (-5.0, 0.0, -4.0), (0.0, 0.0, yaw)
    _, p, r, a = apply_start_randomization(config, pos, rot, SPEC, seed=7)
    psi = math.radians(yaw)
    lon, lat = a["longitudinal_offset_m"], a["lateral_offset_m"]
    exp_dx = lon * math.cos(psi) - lat * math.sin(psi)
    exp_dy = lon * math.sin(psi) + lat * math.cos(psi)
    assert a["initial_yaw_deg"] == pytest.approx(yaw)
    assert a["world_dx_m"] == pytest.approx(exp_dx)
    assert a["world_dy_m"] == pytest.approx(exp_dy)
    assert p[0] == pytest.approx(pos[0] + exp_dx)
    assert p[1] == pytest.approx(pos[1] + exp_dy)
    assert p[2] == pytest.approx(pos[2] + a["depth_offset_m"])
    assert r[2] == pytest.approx(yaw + a["yaw_offset_deg"])  # yaw relative to initial yaw


def test_pure_longitudinal_at_yaw_90_moves_along_world_y():
    spec = StartRandomization(longitudinal_offset_m=2.0)  # forward only
    config = load_track_config(TRACK)
    _, _, _, a = apply_start_randomization(config, (0.0, 0.0, -4.0), (0.0, 0.0, 90.0), spec, seed=1)
    # forward at yaw 90 deg is +y, so the world x-offset is ~0 and y-offset is the longitudinal offset
    assert abs(a["world_dx_m"]) < 1e-9
    assert a["world_dy_m"] == pytest.approx(a["longitudinal_offset_m"])
    assert a["lateral_offset_m"] == 0.0


def test_randomization_does_not_change_gate_geometry_or_referee():
    config = load_track_config(TRACK)
    new_config, _, _, _ = apply_start_randomization(config, (-5.0, 0.0, -4.0), (0.0, 0.0, 20.0), SPEC, seed=2)
    assert tuple(new_config.track.gate_inner_size_m) == tuple(config.track.gate_inner_size_m)
    assert new_config.track.gate_sequence == config.track.gate_sequence
    # only the beacon config differs (noise override); gates/referee/world untouched
    assert new_config.gates == config.gates
    assert new_config.referee == config.referee


def test_local_frame_is_deterministic_per_seed_and_varies_by_seed():
    config = load_track_config(TRACK)
    pos, rot = (-5.0, 0.0, -4.0), (0.0, 0.0, 33.0)  # non-zero yaw
    _, _, _, a1 = apply_start_randomization(config, pos, rot, SPEC, seed=5)
    _, _, _, a2 = apply_start_randomization(config, pos, rot, SPEC, seed=5)
    _, _, _, b = apply_start_randomization(config, pos, rot, SPEC, seed=6)
    assert (a1["world_dx_m"], a1["world_dy_m"]) == (a2["world_dx_m"], a2["world_dy_m"])
    assert (a1["world_dx_m"], a1["world_dy_m"]) != (b["world_dx_m"], b["world_dy_m"])
