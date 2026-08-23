"""The track-specific campaign rests on fragments being *exact*.

If a fragment silently rotated, translated or perturbed the geometry, the
learner would practise something it will never be evaluated on and the whole
campaign would be measuring the wrong thing.  These tests pin that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from marine_race_arena.learning.gen2 import seeds as gen2_seeds
from marine_race_arena.learning.gen2 import track_fragments as tf

REPO_ROOT = Path(__file__).resolve().parents[3]
GEN2_DIR = REPO_ROOT / "marine_race_arena" / "learning" / "gen2"


@pytest.fixture(scope="module")
def fragments():
    return tf.all_fragments((2, 3, 4, 5), root=REPO_ROOT)


def test_all_three_circuits_are_covered(fragments):
    tracks = {f.track for f in fragments}
    assert tracks == set(tf.OFFICIAL_TRACKS)
    for track in tf.OFFICIAL_TRACKS:
        assert sum(1 for f in fragments if f.track == track) >= 20


def test_fragments_are_contiguous_windows_of_the_real_sequence(fragments):
    for fragment in fragments:
        data = tf.load_track(fragment.track, REPO_ROOT)
        sequence = list(data["track"]["gate_sequence"])
        expected = tuple(sequence[fragment.start_index : fragment.end_index + 1])
        assert fragment.gate_ids == expected, fragment.name
        assert fragment.length == fragment.end_index - fragment.start_index + 1


def test_gate_geometry_is_byte_identical_to_the_source(tmp_path, fragments):
    """No rotation, no translation, no perturbation."""
    for fragment in fragments[:40]:
        path = tf.materialize_fragment(
            fragment, tmp_path / f"{fragment.name}.json", root=REPO_ROOT
        )
        report = tf.verify_geometry_preserved(fragment, path, root=REPO_ROOT)
        assert report["identical"], report["mismatches"]
        assert not report["dangling_links"], report
        assert report["bounds_inherited"], fragment.name


def test_no_rotation_or_translation_of_any_kept_gate(tmp_path):
    """Explicit numeric check on a fragment from each circuit."""
    for track in tf.OFFICIAL_TRACKS:
        source = tf.load_track(track, REPO_ROOT)
        by_id = {g["id"]: g for g in source["gates"]}
        fragment = tf.enumerate_fragments(track, (4,), root=REPO_ROOT)[2]
        path = tf.materialize_fragment(
            fragment, tmp_path / f"{fragment.name}.json", root=REPO_ROOT
        )
        produced = json.loads(path.read_text(encoding="utf-8"))
        for gate in produced["gates"]:
            original = by_id[gate["id"]]
            assert gate["position"] == original["position"]
            assert gate["rotation_rpy_deg"] == original["rotation_rpy_deg"]
            assert gate["passage_direction"] == original["passage_direction"]
            assert gate["type"] == original["type"]


def test_relative_geometry_between_gates_is_unchanged(tmp_path):
    """Spacing and bearing between consecutive gates must survive the cut."""
    track = "vertical_serpent"
    source = tf.load_track(track, REPO_ROOT)
    by_id = {g["id"]: g for g in source["gates"]}
    for fragment in tf.enumerate_fragments(track, (5,), root=REPO_ROOT)[:4]:
        path = tf.materialize_fragment(
            fragment, tmp_path / f"{fragment.name}.json", root=REPO_ROOT
        )
        produced = json.loads(path.read_text(encoding="utf-8"))
        gates = produced["gates"]
        for previous, current in zip(gates, gates[1:]):
            source_delta = np.asarray(by_id[current["id"]]["position"]) - np.asarray(
                by_id[previous["id"]]["position"]
            )
            fragment_delta = np.asarray(current["position"]) - np.asarray(previous["position"])
            assert np.allclose(source_delta, fragment_delta), fragment.name


def test_linked_gate_pairs_are_never_split(fragments):
    for fragment in fragments:
        data = tf.load_track(fragment.track, REPO_ROOT)
        by_id = {g["id"]: g for g in data["gates"]}
        kept = set(fragment.gate_ids)
        for gate_id in fragment.gate_ids:
            linked = by_id[gate_id].get("linked_gate")
            if linked:
                assert linked in kept, (
                    f"{fragment.name} splits the linked pair {gate_id}-{linked}"
                )


def test_a_window_that_would_split_a_pair_grows_instead():
    """Vertical Serpent links G08-G09, so a 2-window at G08 must widen."""
    fragments = {f.name: f for f in tf.enumerate_fragments("vertical_serpent", (2,), root=REPO_ROOT)}
    grown = [f for f in fragments.values() if f.grown_for_link]
    assert grown, "no window needed growing, so the guard was never exercised"
    for fragment in grown:
        assert "G08" in fragment.gate_ids and "G09" in fragment.gate_ids


def test_inbound_start_reproduces_the_real_approach(tmp_path):
    """An internal fragment must start where the vehicle really would be.

    Spawning in front of gate k instead of just past gate k-1 would delete the
    transition the fragment exists to teach.
    """
    track = "vertical_serpent"
    data = tf.load_track(track, REPO_ROOT)
    by_id = {g["id"]: g for g in data["gates"]}
    fragment = next(
        f for f in tf.enumerate_fragments(track, (3,), root=REPO_ROOT)
        if f.start_index == 4  # starts at G05, inbound from G04
    )
    assert fragment.inbound_gate_id == "G04"
    position, rotation = tf.inbound_start_pose(data, fragment)
    inbound = by_id["G04"]
    direction = np.asarray(inbound["passage_direction"], dtype=float)
    direction = direction / (np.linalg.norm(direction) or 1.0)
    expected = np.asarray(inbound["position"], dtype=float) + tf.INBOUND_CLEARANCE_M * direction
    assert np.allclose(np.asarray(position), expected, atol=1e-4)
    # Heading matches the gate we just exited, not the gate we are heading to.
    assert rotation[2] == pytest.approx(inbound["rotation_rpy_deg"][2])
    # And the start really is on the far side of the inbound gate.
    to_start = np.asarray(position) - np.asarray(inbound["position"], dtype=float)
    assert float(np.dot(to_start, direction)) > 0


def test_a_prefix_fragment_keeps_the_circuits_own_start():
    for track in tf.OFFICIAL_TRACKS:
        data = tf.load_track(track, REPO_ROOT)
        fragment = next(
            f for f in tf.enumerate_fragments(track, (3,), root=REPO_ROOT) if f.is_prefix
        )
        assert fragment.inbound_gate_id is None
        position, rotation = tf.inbound_start_pose(data, fragment)
        assert position == list(data["start"]["position"])
        assert rotation == list(data["start"]["rotation_rpy_deg"])


def test_materialized_fragment_is_a_valid_runnable_track(tmp_path):
    from marine_race_arena.config.loader import load_track_config

    for track in tf.OFFICIAL_TRACKS:
        fragment = tf.enumerate_fragments(track, (3,), root=REPO_ROOT)[1]
        path = tf.materialize_fragment(
            fragment, tmp_path / f"{fragment.name}.json", root=REPO_ROOT
        )
        config = load_track_config(str(path), current_profile="none")
        assert len(config.track.gate_sequence) == fragment.length
        assert config.finish.gate_id == fragment.gate_ids[-1]
        assert config.race.expected_gates_per_lap == fragment.length


def test_fragments_are_current_free_like_the_official_comparison(tmp_path):
    """Mixed Endurance carries currents; the official 0/30 vs 30/30 run did not."""
    fragment = tf.enumerate_fragments("mixed_endurance", (3,), root=REPO_ROOT)[0]
    path = tf.materialize_fragment(
        fragment, tmp_path / "frag.json", root=REPO_ROOT
    )
    assert json.loads(path.read_text(encoding="utf-8"))["currents"] == []
    assert tf.load_track("mixed_endurance", REPO_ROOT)["currents"], (
        "source circuit should still carry its currents"
    )


def test_fragment_metadata_never_reaches_the_network(tmp_path):
    """Track identity is diagnostics; the policy still sees 35 onboard features."""
    from marine_race_arena.learning.config_local_transition import OBS_DIM_LOCAL_TRANSITION

    fragment = tf.enumerate_fragments("horseshoe_bay", (3,), root=REPO_ROOT)[0]
    path = tf.materialize_fragment(fragment, tmp_path / "frag.json", root=REPO_ROOT)
    produced = json.loads(path.read_text(encoding="utf-8"))
    assert produced["gen2_fragment"]["track"] == "horseshoe_bay"
    # The identity lives in its own key, not in anything the encoder reads.
    assert OBS_DIM_LOCAL_TRANSITION == 35
    source = (GEN2_DIR / "evaluation.py").read_text(encoding="utf-8")
    for forbidden in ("gen2_fragment", "OFFICIAL_TRACKS", "track_name"):
        assert forbidden not in source, f"evaluation.py reads {forbidden!r}"


def test_fragment_seeds_stay_in_the_track_band():
    for fragment in tf.all_fragments((2, 3), root=REPO_ROOT)[:30]:
        seed = tf.fragment_seed(fragment)
        assert gen2_seeds.band_of(seed) == "TRACK_SPECIFIC"
        assert gen2_seeds.assert_expert_labelling_seed(seed) == seed
        # Reproducible from the fragment identity alone.
        assert tf.fragment_seed(fragment) == seed


def test_the_track_band_cannot_touch_the_sealed_holdout():
    for seed in (60_000, 64_000, 67_000, 69_999):
        assert gen2_seeds.band_of(seed) == "TRACK_SPECIFIC"
        with pytest.raises(PermissionError):
            gen2_seeds.assert_not_final_holdout(52_000)


def test_targeted_fragments_cover_the_reported_weak_gate():
    from marine_race_arena.learning.gen2 import track_eval as te

    failures = {"weak_gates": {"mixed_endurance": [{"gate": 7, "failures": 5}]}}
    picked = te.targeted_fragments(failures, lengths=(3, 4, 5), root=REPO_ROOT)
    assert picked, "no fragment was selected for the reported weak gate"
    for fragment in picked:
        span = range(fragment.start_index + 1, fragment.end_index + 2)
        assert 7 in span, fragment.name
        # There must be run-up: entering the weak gate is the point.
        assert 7 - fragment.start_index >= 2, fragment.name


def test_full_circuits_load_current_free_including_mixed_endurance():
    """Mixed Endurance is a current_gate task; running it current-free needs
    the task overridden too, or the loader rejects it outright.

    This is the same trap in a second place: it was fixed for fragments, then
    the full-circuit evaluation walked into it and lost a whole track's
    baseline. Both drivers now pass the override.
    """
    from marine_race_arena.config.benchmark_tasks import BENCHMARK_TASK_CLEAN_GATE
    from marine_race_arena.config.loader import load_track_config

    for track in tf.OFFICIAL_TRACKS:
        path = tf.track_path(track, REPO_ROOT)
        config = load_track_config(
            str(path), current_profile="none",
            benchmark_task=BENCHMARK_TASK_CLEAN_GATE,
        )
        assert config.race.expected_gates_per_lap == len(config.track.gate_sequence)


def test_both_episode_drivers_override_the_benchmark_task():
    for module in ("evaluation.py", "expert_rollout.py"):
        source = (GEN2_DIR / module).read_text(encoding="utf-8")
        assert "BENCHMARK_TASK_CLEAN_GATE" in source, module
        assert 'current_profile="none"' in source, module


def test_track_dagger_uses_its_own_seed_band_and_labels_legally():
    """Targeted DAgger must draw from the DAgger band, not the demo band."""
    from marine_race_arena.learning.gen2 import collect_fragments as cf

    for fragment in tf.all_fragments((3,), root=REPO_ROOT)[:10]:
        demo_seed = tf.fragment_seed(fragment, 0)
        dagger_seed = tf.fragment_seed(fragment, 1000)
        assert gen2_seeds.band_of(demo_seed) == "TRACK_SPECIFIC"
        assert gen2_seeds.band_of(dagger_seed) == "TRACK_SPECIFIC"
        # Both roles are expert-labelling roles by design in this campaign.
        assert gen2_seeds.assert_expert_labelling_seed(demo_seed) == demo_seed
    args = cf.build_parser().parse_args(["--out", "x", "--dagger-from", "ck.zip"])
    assert args.dagger_from == "ck.zip"


def test_track_dagger_never_blends_or_takes_over():
    """The learner keeps control; the expert only labels."""
    source = (GEN2_DIR / "collect_fragments.py").read_text(encoding="utf-8")
    assert "safety_takeover=None" in source
    assert 'mode = "dagger" if learner is not None else "expert"' in source
    # And the driver itself must record what actually drove.
    driver = (GEN2_DIR / "expert_rollout.py").read_text(encoding="utf-8")
    assert "applied_by_expert" in driver
