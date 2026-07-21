"""Tests for the curriculum definitions and training tracks."""

from pathlib import Path

import pytest

from marine_race_arena.config.loader import load_track_config
from marine_race_arena.learning import curriculum as cur


def test_stage_keys_unique_and_lookup():
    keys = [s.key for s in cur.STAGES]
    assert len(keys) == len(set(keys))
    for key in keys:
        assert cur.stage(key).key == key
    with pytest.raises(KeyError):
        cur.stage("does_not_exist")


def test_next_stage_chain():
    assert cur.next_stage("stage0").key == "stage1"
    assert cur.next_stage(cur.STAGES[-1].key) is None


def test_meets_criterion():
    assert cur.meets_criterion("stage1", 0.95) is True
    assert cur.meets_criterion("stage1", 0.80) is False
    assert cur.meets_criterion("stage3", 0.80) is True


def test_completion_criteria_are_monotone_sane():
    for s in cur.STAGES:
        assert 0.0 <= s.min_completion_rate <= 1.0
        assert s.eval_episodes >= 0
    # early single-gate stages demand higher completion than multi-gate ones
    assert cur.stage("stage1").min_completion_rate >= cur.stage("stage3").min_completion_rate


def test_training_tracks_exist_and_load():
    for key in ("stage1", "stage3"):
        track = cur.stage(key).track
        assert Path(track).exists(), f"missing training track {track}"
        config = load_track_config(track)
        assert config.race.official_mode is True
        # official gate aperture preserved
        assert tuple(config.track.gate_inner_size_m) == (1.5, 1.5)


def test_official_stages_reference_unchanged_official_tracks():
    for key in ("stage5_horseshoe", "stage6_generalization", "stage7_disturbance"):
        s = cur.stage(key)
        assert s.official_track is True
        assert "tracks/marine_race_" in s.track
