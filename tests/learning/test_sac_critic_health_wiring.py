"""The critic-health detector must change training, not just exist as a utility.

Before this wiring the SAC v2 run diverged for thousands of updates and was only
diagnosed after the fact.  These tests pin the two properties that matter: the
detector is built from the run config and actually consulted, and its recovery
rebuilds critics while preserving the actor bit-for-bit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from marine_race_arena.learning import train_sac_transition as trainer
from marine_race_arena.learning.holoocean_capacity import _registry_lock
from marine_race_arena.learning.sac_health_gates import CriticHealthMonitor
from marine_race_arena.learning.sac_transition_policy import (
    SACTransitionAgent,
    critics_are_independent,
    rebuild_critics_from_actor,
)


def _observations(n=128, seed=11):
    return torch.from_numpy(
        np.random.default_rng(seed).normal(size=(n, 35)).astype(np.float32)
    )


def test_monitor_is_built_from_the_run_configuration():
    config = trainer._load_config(
        "configs/rl/sac_universal_transition_v3_critic_rebuild.json"
    )
    monitor = trainer._new_critic_health_monitor(config)
    assert isinstance(monitor, CriticHealthMonitor)
    assert monitor.thresholds.consecutive_violations == (
        config["critic_health"]["consecutive_violations"]
    )


def test_monitor_inherits_the_configured_critic_gradient_clip():
    config = trainer._load_config(
        "configs/rl/sac_universal_transition_v3_critic_rebuild.json"
    )
    clip = config["sac"].get("gradient_clip_critic")
    monitor = trainer._new_critic_health_monitor(config)
    if clip:
        assert monitor.gradient_clip == pytest.approx(float(clip))


def test_the_learner_loop_actually_consults_the_detector():
    """A detector nobody calls is worthless; pin the call site."""

    source = Path(trainer.__file__).read_text(encoding="utf-8")
    body = source.split("def run_training", 1)[1]
    assert "critic_health.observe(" in body
    assert "_recover_from_critic_divergence(" in body
    assert "actor_frozen" in body


def test_recovery_preserves_the_actor_and_replaces_the_critics(tmp_path):
    agent = SACTransitionAgent(hidden_sizes=(32, 32), anchor_coefficient=4.0)
    with torch.no_grad():
        agent.actor.mean_head.bias.fill_(0.21)
        for parameter in agent.critic.parameters():
            parameter.add_(9.0)
    path = tmp_path / "diverged.pt"
    torch.save({"agent": agent.checkpoint_state()}, path)

    observations = _observations()
    with torch.no_grad():
        before = agent.actor.deterministic(observations).clone()

    recovered, report = rebuild_critics_from_actor(path, tau=0.002)

    with torch.no_grad():
        after = recovered.actor.deterministic(observations)
    assert torch.allclose(before, after), "actor must survive a critic rebuild"
    assert critics_are_independent(recovered, agent)
    assert report["actor_transferred"] is True


def test_actor_is_frozen_then_released_only_after_a_healthy_window():
    """Freeze on divergence; unfreeze only once the rebuilt critics settle."""

    source = Path(trainer.__file__).read_text(encoding="utf-8")
    body = source.split("def run_training", 1)[1]
    assert "actor_gate.on_critic_rebuild()" in body
    assert "critic_health.begin_generation()" in body
    # The unfreeze must be conditional on a settled, healthy critic generation.
    assert "critic_healthy=critic_health.is_healthy()" in body
    # The stale-counter arithmetic must stay deleted.
    assert "actor_unfreeze_after" not in body


def test_update_actor_is_suppressed_while_frozen():
    source = Path(trainer.__file__).read_text(encoding="utf-8")
    assert "actor_gate.should_update_actor()" in source
    assert "update_actor=update_actor," in source
    assert "update_entropy=update_actor," in source


def test_recovery_record_documents_provenance():
    source = Path(trainer.__file__).read_text(encoding="utf-8")
    body = source.split("def _recover_from_critic_divergence", 1)[1]
    for field in (
        "diagnostic_checkpoint", "diagnostic_sha256", "health_reasons",
        "replay_preserved", "actor_preserved", "critics_reinitialized",
    ):
        assert field in body, field


def test_replay_is_kept_across_recovery():
    """Replay corruption was never demonstrated, so replay must be preserved."""

    source = Path(trainer.__file__).read_text(encoding="utf-8")
    body = source.split("def _recover_from_critic_divergence", 1)[1]
    assert '"replay_preserved": True' in body


# ------------------------------------------------------- capacity registry


def test_capacity_registry_lock_never_reads_the_locked_byte():
    """Regression for the PermissionError class of bug."""

    source = Path(
        __import__(
            "marine_race_arena.learning.holoocean_capacity", fromlist=["x"]
        ).__file__
    ).read_text(encoding="utf-8")
    body = source.split("def _registry_lock", 1)[1].split("def ", 1)[0]
    assert ".read(" not in body
    assert '"xb"' in body


def test_capacity_registry_lock_is_reentrant_across_sequential_use():
    with _registry_lock():
        pass
    with _registry_lock():
        pass


def test_capacity_snapshot_still_reports_a_cap():
    from marine_race_arena.learning.holoocean_capacity import capacity_snapshot

    snapshot = capacity_snapshot()
    assert snapshot["maximum"] >= 1
    assert json.dumps(snapshot["reservations"]) is not None
