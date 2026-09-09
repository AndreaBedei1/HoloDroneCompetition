"""Audit the committed learned multi-gate v3 public package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PUB = Path("results/rl_public/multigate_rl_v3")
pytestmark = pytest.mark.skipif(not PUB.exists(), reason="public package not built")


def _load(relative: str):
    return json.loads((PUB / relative).read_text(encoding="utf-8"))


def test_required_compact_files_exist():
    for relative in (
        "README.md",
        "experiment_manifest.json",
        "observation_v3.json",
        "seed_registry_snapshot.json",
        "models/transfer_v1_to_v3.json",
        "models/bc_v3.json",
        "models/ppo_v3.json",
        "models/selected_model_sha256.json",
        "two_gate/train_summary.json",
        "two_gate/eval_results.json",
        "two_gate/failure_analysis.json",
        "three_gate/train_summary.json",
        "three_gate/eval_results.json",
        "three_gate/failure_analysis.json",
        "official_no_current/summary.json",
        "comparison/paired_per_seed.csv",
        "comparison/aggregate.json",
        "comparison/time_comparison.json",
    ):
        assert (PUB / relative).exists(), relative


def test_observation_v3_is_onboard_only_and_fixed():
    observation = _load("observation_v3.json")
    assert observation["observation_encoding_version"] == "onboard_multigate_rl_v3"
    assert observation["observation_dim"] == len(observation["feature_names"]) == 59
    assert observation["v1_prefix_columns"] == 36
    assert observation["privileged_fields_present"] is False
    forbidden = ("referee", "world_position", "gate_world", "ground_truth", "reward")
    assert not any(
        token in name.lower()
        for name in observation["feature_names"]
        for token in forbidden
    )


def test_final_manifest_proves_current_free_real_holoocean():
    manifest = _load("experiment_manifest.json")
    final = manifest["final_two_gate"]
    evaluation = final["manifest"]
    assert evaluation["adapter_actual"] == "holoocean"
    assert evaluation["fallback_used"] is False
    assert evaluation["current_profile"] == "none"
    assert evaluation["currents_actual"] == [0.0, 0.0, 0.0]
    assert final["summary"]["completions"] == 9
    assert final["summary"]["n_eval"] == 10
    assert manifest["runtime_architecture"]["rule_action_weight"] == 0
    assert manifest["runtime_architecture"]["hybrid_blending"] is False


def test_unreached_stages_are_not_misrepresented():
    manifest = _load("experiment_manifest.json")
    assert manifest["result_category"] == "PARTIAL RL SUCCESS"
    assert manifest["r2_status"] == "HOLD"
    assert manifest["three_gate_status"] == "NOT_RUN"
    assert manifest["official_status"] == "NOT_RUN"
    assert _load("three_gate/eval_results.json") == []


def test_package_contains_no_heavy_artifacts():
    for path in PUB.rglob("*"):
        if path.is_file():
            assert path.suffix.lower() not in {".zip", ".pt", ".npz", ".npy", ".mp4"}
            assert path.stat().st_size < 10 * 1024 * 1024
