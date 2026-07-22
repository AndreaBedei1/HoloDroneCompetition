"""Validate the committed public Stage-1 audit package (structure + onboard-only)."""

import hashlib
import json
from pathlib import Path

import pytest

PUB = Path("results/rl_public/stage1")

pytestmark = pytest.mark.skipif(not PUB.exists(), reason="public package not built")


def _load(name):
    return json.loads((PUB / name).read_text(encoding="utf-8"))


def test_required_files_present():
    for rel in (
        "README.md", "result_manifest.json",
        "dataset/dataset_summary.json", "dataset/dataset_hashes.json", "dataset/dataset_episode_manifest.csv",
        "bc/bc_report.json", "bc/model_hash.json",
        "evaluation/eval_results.json", "evaluation/eval_results.csv", "evaluation/eval_summary.json",
        "evaluation/dev_history.json", "evaluation/seed_split.json", "evaluation/randomization_manifest.json",
        "reproduction/reproduce_bc_training.txt", "reproduction/reproduce_bc_evaluation.txt", "reproduction/environment.json",
    ):
        assert (PUB / rel).exists(), f"missing public artifact {rel}"


def test_manifest_core_fields():
    m = _load("result_manifest.json")
    assert m["fallback_disabled"] is True
    assert m["adapter_actual"] == "holoocean"
    assert m["referee_clearance_margin_m"] == 0.10
    assert m["gate_aperture_m"] == [1.5, 1.5]
    assert m["observation_encoding_version"] == "onboard_only_v1"
    assert m["action_contract"]["axes"] == ["surge", "sway", "heave", "yaw"]
    assert m["model"]["sha256"] and m["track_sha256"]


def test_observation_is_onboard_only():
    """The published observation feature names must contain no privileged state."""
    m = _load("result_manifest.json")
    names = m["observation_feature_names"]
    assert len(names) == m["observation_dim"] == 36
    forbidden = ("pose", "ground_truth", "groundtruth", "referee", "gate_center", "gate_coord",
                 "passage_direction", "current", "target", "world_position", "privileged", "reward")
    for name in names:
        low = name.lower()
        assert not any(tok in low for tok in forbidden), f"privileged-looking feature {name!r}"
    # sanity: the legal families are present
    assert any(n.startswith("beacon_") for n in names)
    assert any(n.startswith("vision_") for n in names)
    assert any(n.startswith("dvl_") for n in names)
    assert any(n.startswith("prev_") for n in names)


def test_committed_model_matches_its_hash():
    mh = _load("bc/model_hash.json")
    model = PUB / "bc" / "model" / mh["filename"]
    if not model.exists():
        pytest.skip("model not committed (too large); hash published instead")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    assert digest == mh["sha256"]
    assert model.stat().st_size == mh["bytes"] <= 30 * 1024 * 1024


def test_dev_history_classifies_conditions():
    hist = _load("evaluation/dev_history.json")
    # fixed-start eval is marked not-randomized; randomized evals are marked randomized
    by_dir = {h["dir"]: h for h in hist}
    assert by_dir["eval_bc"]["eval_randomized"] is False
    assert by_dir["eval_bc_combined"]["eval_randomized"] is True
    assert by_dir["eval_bc_combined"]["completion_rate"] == 1.0


def test_no_heavy_binaries_committed():
    for p in PUB.rglob("*"):
        if p.is_file():
            assert p.suffix not in (".zip", ".npz", ".npy", ".mp4"), f"heavy binary in package: {p}"
            assert p.stat().st_size <= 30 * 1024 * 1024
