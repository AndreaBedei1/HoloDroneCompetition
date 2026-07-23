"""Tests for the closed-loop evaluation manifest + safe resume compatibility.

These are numpy-only (no torch/SB3): they exercise the manifest identity check and
the resume/refusal control flow without launching any real episode.
"""

import argparse
import json
from pathlib import Path

import pytest

from marine_race_arena.learning import closed_loop_eval as cle
from marine_race_arena.learning.config import ACTION_CONTRACT_VERSION, OBS_ENCODING_VERSION

STAGE1 = "marine_race_arena/tracks/training/stage1_single_gate.json"


def _ns(**overrides):
    base = dict(
        controller="rule_gate_center_then_commit",
        model=None,
        track=STAGE1,
        adapter="fallback",
        allow_fallback=True,
        randomize=False,
        dt=0.1,
        duration=3.0,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _requested(**overrides):
    ns = _ns(**overrides)
    spec = None
    if ns.randomize:
        from dataclasses import asdict
        from marine_race_arena.learning.curriculum import STAGE2_RANDOMIZATION

        spec = asdict(STAGE2_RANDOMIZATION)
    return cle.build_requested_config(ns, model_sha256=overrides.get("_model_sha256"), randomization_spec=spec)


def test_requested_config_captures_identity_and_versions():
    cfg = _requested()
    assert cfg["observation_encoding_version"] == OBS_ENCODING_VERSION
    assert cfg["action_contract_version"] == ACTION_CONTRACT_VERSION
    assert cfg["max_steps"] is None  # time-deadline runner
    assert len(cfg["track_sha256"]) == 64


def test_manifest_compatible_has_no_incompatibilities():
    base = _requested()
    assert cle.manifest_incompatibilities(base, dict(base)) == []


@pytest.mark.parametrize(
    "field, mutate",
    [
        ("model_sha256", {"_model_sha256": "a" * 64}),
        ("track_sha256", {}),          # forced below
        ("randomization_enabled", {"randomize": True}),
        ("adapter_requested", {"adapter": "holoocean"}),
        ("fallback_allowed", {"allow_fallback": False}),
        ("dt", {"dt": 0.05}),
        ("duration_s", {"duration": 9.0}),
    ],
)
def test_manifest_detects_each_identity_change(field, mutate):
    base = _requested()
    other = _requested(**mutate)
    if field == "track_sha256":
        other = dict(base)
        other["track_sha256"] = "b" * 64
    issues = cle.manifest_incompatibilities(base, other)
    assert any(field in i for i in issues), f"expected {field} mismatch in {issues}"


def test_manifest_detects_encoding_and_action_version_change():
    base = _requested()
    other = dict(base)
    other["observation_encoding_version"] = "onboard_only_v2"
    other["action_contract_version"] = "vNEXT"
    issues = cle.manifest_incompatibilities(base, other)
    assert any("observation_encoding_version" in i for i in issues)
    assert any("action_contract_version" in i for i in issues)


def _write_experiment(out: Path, requested, seeds, *, finished_all=True):
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": cle.MANIFEST_SCHEMA_VERSION, **requested,
                "requested_seeds": seeds, "completed_seeds": seeds, "created_utc": "2026-01-01T00:00:00Z"}
    (out / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = [{"seed": s, "status": "FINISHED" if finished_all else "RUNNING",
             "referee_status": "FINISHED" if finished_all else "RUNNING",
             "evaluation_end_reason": "FINISHED" if finished_all else "TIME_LIMIT",
             "finished": finished_all, "completed_gates": 1, "expected_gates": 1,
             "official_time_s": 0.0, "penalized_time_s": 0.0, "collision_events": 0,
             "obstacle_collision_events": 0, "out_of_bounds_events": 0, "stuck_events": 0,
             "missed_gate_attempts": 0, "wrong_direction_crossings": 0, "inference_time_ms": 1.0,
             "wall_s": 1.0, "adapter_used": "fallback", "applied_randomization": None} for s in seeds]
    (out / "eval_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def test_compatible_resume_skips_completed_seeds(tmp_path, monkeypatch):
    out = tmp_path / "eval"
    _write_experiment(out, _requested(), [0, 1, 2])
    # Guard: if a completed seed were re-run, this would raise.
    monkeypatch.setattr(cle, "evaluate_controller", lambda *a, **k: pytest.fail("re-ran a completed seed"))
    rc = cle.main(["--track", STAGE1, "--seeds", "0-2", "--out", str(out),
                   "--controller", "rule_gate_center_then_commit", "--adapter", "fallback", "--allow-fallback",
                   "--duration", "3.0"])
    assert rc == 0
    summary = json.loads((out / "eval_summary.json").read_text(encoding="utf-8"))
    assert summary["n_eval"] == 3 and summary["completions"] == 3
    # No duplicate seeds in the persisted results.
    seeds = [r["seed"] for r in json.loads((out / "eval_results.json").read_text(encoding="utf-8"))]
    assert seeds == sorted(set(seeds)) == [0, 1, 2]


def test_incompatible_resume_is_refused(tmp_path, monkeypatch):
    out = tmp_path / "eval"
    # Existing experiment used randomization; the new request does not -> incompatible.
    _write_experiment(out, _requested(randomize=True), [0, 1, 2])
    monkeypatch.setattr(cle, "evaluate_controller", lambda *a, **k: pytest.fail("ran despite incompatible resume"))
    rc = cle.main(["--track", STAGE1, "--seeds", "0-2", "--out", str(out),
                   "--controller", "rule_gate_center_then_commit", "--adapter", "fallback", "--allow-fallback",
                   "--duration", "3.0"])  # no --randomize -> mismatch
    assert rc == 2
    # Original results are left intact (not merged, not deleted).
    assert (out / "eval_results.json").exists()


def test_legacy_dir_without_manifest_is_refused(tmp_path, monkeypatch):
    out = tmp_path / "eval"
    out.mkdir(parents=True)
    (out / "eval_results.json").write_text(json.dumps([{"seed": 0, "finished": True, "status": "FINISHED"}]), encoding="utf-8")
    monkeypatch.setattr(cle, "evaluate_controller", lambda *a, **k: pytest.fail("ran against unverifiable legacy dir"))
    rc = cle.main(["--track", STAGE1, "--seeds", "0", "--out", str(out),
                   "--controller", "rule_gate_center_then_commit", "--adapter", "fallback", "--allow-fallback"])
    assert rc == 2


def test_force_new_archives_instead_of_deleting(tmp_path, monkeypatch):
    out = tmp_path / "eval"
    _write_experiment(out, _requested(randomize=True), [0, 1, 2])
    # Stop before running any real episode, but after the archive/backup step.
    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("stop after archive")

    monkeypatch.setattr(cle, "evaluate_controller", _boom)
    with pytest.raises(RuntimeError):
        cle.main(["--track", STAGE1, "--seeds", "0", "--out", str(out),
                  "--controller", "rule_gate_center_then_commit", "--adapter", "fallback", "--allow-fallback",
                  "--force-new"])  # incompatible (no --randomize) + force-new
    # A timestamped backup preserved the old results; nothing was deleted.
    backups = list(tmp_path.glob("eval_backup_*"))
    assert backups and (backups[0] / "eval_results.json").exists()
    assert calls["n"] == 1  # it did proceed to run the fresh experiment
