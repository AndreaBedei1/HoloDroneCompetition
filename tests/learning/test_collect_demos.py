"""Tests for resumable demonstration collection (fallback adapter, fast)."""

import json

import numpy as np
import pytest

from marine_race_arena.learning import collect_demos
from marine_race_arena.learning.dataset import BCDataset
from marine_race_arena.learning.trajectory_recorder import EpisodeRecord

TRACK = "marine_race_arena/tracks/tests/single_gate_yaw_0.json"


def _run(out, seeds, **extra):
    argv = ["--track", TRACK, "--seeds", seeds, "--out", str(out),
            "--adapter", "fallback", "--allow-fallback", "--max-steps", "6"]
    for k, v in extra.items():
        if v is True:
            argv.append(f"--{k}")
        elif v is not None:
            argv += [f"--{k}", str(v)]
    return collect_demos.main(argv)


def _manifest(out):
    return json.loads((out / "collection_manifest.json").read_text(encoding="utf-8"))


def test_episode_record_npz_roundtrip(tmp_path):
    from marine_race_arena.learning.trajectory_recorder import record_episode

    rec = record_episode(TRACK, seed=0, max_steps=6, adapter="fallback", allow_fallback=True)
    path = tmp_path / "ep.npz"
    rec.save_npz(path)
    assert not (tmp_path / "ep.npz.tmp").exists()  # atomic: no temp left behind
    loaded = EpisodeRecord.load_npz(path)
    assert loaded.seed == rec.seed and loaded.length == rec.length
    assert np.array_equal(loaded.observations, rec.observations)
    assert np.array_equal(loaded.actions, rec.actions)


def test_episode_record_metadata_captured_and_roundtrips(tmp_path):
    from marine_race_arena.learning.trajectory_recorder import record_episode

    rec = record_episode(TRACK, seed=0, max_steps=6, adapter="fallback", allow_fallback=True)
    md = rec.metadata
    # Provenance metadata is present (and separate from the policy observation).
    for key in ("track_sha256", "adapter_requested", "adapter_actual", "fallback_allowed",
                "fallback_used", "obs_encoding_version", "action_contract_version", "dt",
                "max_steps", "official", "collection_git_sha"):
        assert key in md, f"missing metadata {key}"
    assert md["adapter_requested"] == "fallback" and md["adapter_actual"] == "fallback"
    assert md["fallback_used"] is True and md["dt"] == 0.1 and md["max_steps"] == 6
    # Metadata survives the npz roundtrip.
    path = tmp_path / "ep.npz"
    rec.save_npz(path)
    assert EpisodeRecord.load_npz(path).metadata == md


def test_old_episode_file_without_metadata_still_loads(tmp_path):
    """Backward compatibility: an npz whose meta lacks 'metadata' loads with {}."""
    import json as _json
    n = 4
    path = tmp_path / "old.npz"
    meta = {"episode_id": 0, "seed": 7, "track": TRACK, "controller": "c",
            "final_status": "FINISHED", "gate_crossings": 1}  # no 'metadata' key (old schema)
    np.savez_compressed(
        path.with_suffix(""),  # np appends .npz
        observations=np.zeros((n, 36), np.float32), expert_actions_raw=np.zeros((n, 4), np.float32),
        actions=np.zeros((n, 4), np.float32), dones=np.zeros(n, bool), truncated=np.zeros(n, bool),
        step_ids=np.arange(n, dtype=np.int64), phase_ids=np.zeros(n, np.int64),
        diag_positions=np.zeros((n, 3), np.float32), diag_gate_crossings=np.zeros(n, np.int64),
        meta=np.array(_json.dumps(meta)),
    )
    rec = EpisodeRecord.load_npz(path)
    assert rec.seed == 7 and rec.metadata == {}


def test_initial_collection(tmp_path):
    out = tmp_path / "demos"
    assert _run(out, "0-2") == 0
    man = _manifest(out)
    assert man["completed_seeds"] == [0, 1, 2]
    assert man["total_episodes"] == 3
    assert (out / "stage1_demos.npz").exists()
    assert man["dataset_sha256"] and man["track_sha256"]
    assert len(list((out / "episodes").glob("ep_*.npz"))) == 3
    # Per-seed provenance in the manifest (S5).
    assert set(man["per_seed_adapter"]) == {"0", "1", "2"}
    assert all(v == "fallback" for v in man["per_seed_adapter"].values())
    assert man["adapters_actually_observed"] == ["fallback"]
    assert man["any_fallback_occurred"] is True
    assert set(man["per_episode_file_sha256"]) == {"0", "1", "2"}
    assert "0" in man["per_seed_applied_randomization"]
    assert man["action_contract_version"] and man["dt"] == 0.1 and man["max_steps"] == 6


def test_dt_or_maxsteps_change_is_incompatible(tmp_path):
    out = tmp_path / "demos"
    _run(out, "0-1")  # default dt from _run is unset -> collect_demos default 0.1, max-steps 6
    # Re-running with a different dt must be refused (identity mismatch).
    argv = ["--track", TRACK, "--seeds", "0-1", "--out", str(out),
            "--adapter", "fallback", "--allow-fallback", "--max-steps", "6", "--dt", "0.2"]
    assert collect_demos.main(argv) == 2


def test_resume_skips_completed_and_appends(tmp_path):
    out = tmp_path / "demos"
    _run(out, "0-1")
    ep0_before = (out / "episodes" / "ep_00000.npz").read_bytes()
    assert _run(out, "0-3") == 0  # 0-1 done, 2-3 new
    man = _manifest(out)
    assert man["completed_seeds"] == [0, 1, 2, 3]
    # existing episode file preserved unchanged
    assert (out / "episodes" / "ep_00000.npz").read_bytes() == ep0_before
    ds = BCDataset.load(out / "stage1_demos.npz")
    ds.check_integrity()
    assert ds.num_episodes == 4


def test_no_duplicate_groups_after_resume(tmp_path):
    out = tmp_path / "demos"
    _run(out, "0-1")
    _run(out, "1-3")  # 1 already done, add 2-3
    ds = BCDataset.load(out / "stage1_demos.npz")
    ds.check_integrity()  # asserts unique episode identity
    assert len(set(ds.group_ids.tolist())) == ds.num_episodes == 4


def test_incompatible_resume_is_refused(tmp_path):
    out = tmp_path / "demos"
    _run(out, "0-1", controller="rule_gate_center_then_commit")
    # different controller -> incompatible -> refuse (return code 2), data preserved
    rc = _run(out, "0-1", controller="rule_gate_baseline")
    assert rc == 2
    assert _manifest(out)["controller"] == "rule_gate_center_then_commit"


def test_force_new_overwrites_incompatible(tmp_path):
    out = tmp_path / "demos"
    _run(out, "0-1", controller="rule_gate_center_then_commit")
    rc = _run(out, "0-1", controller="rule_gate_baseline", **{"force-new": True})
    assert rc == 0
    assert _manifest(out)["controller"] == "rule_gate_baseline"


def test_randomization_toggle_is_incompatible(tmp_path):
    out = tmp_path / "demos"
    _run(out, "0-1")
    assert _run(out, "0-1", randomize=True) == 2  # randomization changed -> refused


def test_failure_preserves_existing_data(tmp_path, monkeypatch):
    out = tmp_path / "demos"
    _run(out, "0-1")
    dataset_before = (out / "stage1_demos.npz").read_bytes()

    real = collect_demos.record_episode

    def flaky(track, controller, *, seed, **kw):
        if seed == 2:
            raise RuntimeError("simulated engine failure")
        return real(track, controller, seed=seed, **kw)

    monkeypatch.setattr(collect_demos, "record_episode", flaky)
    assert _run(out, "0-3") == 0  # seed 2 fails, seed 3 succeeds
    man = _manifest(out)
    assert 2 in man["failed_seeds"]
    assert set(man["completed_seeds"]) == {0, 1, 3}
    # seed 0/1 data preserved (still loadable and valid)
    BCDataset.load(out / "stage1_demos.npz").check_integrity()
