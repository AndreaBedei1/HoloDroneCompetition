"""Evaluation lock, supervisor resilience, sealed circuits and curriculum."""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import marine_race_arena.learning.rl_autonomous_supervisor as supervisor
from marine_race_arena.learning.generic_sequence_curriculum import (
    BUCKETS,
    BUCKET_GATE_RANGES,
    GenericSequenceCurriculum,
    empirical_mixture,
    normalized_mixture,
)
from marine_race_arena.learning.reward_audit import build_report
from marine_race_arena.learning.rl_evaluation_lock import (
    BACKOFF_CAP_SECONDS,
    WAITING_STATE,
    backoff_delays,
    evaluation_lock,
    is_contention_error,
    owner_is_stale,
)
from marine_race_arena.learning.rl_holdout_policy import (
    FINAL_CIRCUIT_TRACKS,
    SealedCircuitViolation,
    assert_no_final_circuits,
    assert_not_final_circuit,
    assert_seed_role,
    evaluation_seeds,
    groups_are_disjoint,
    is_final_circuit,
    policy_manifest,
    record_final_circuit_use,
    role_of_seed,
    seed_group,
)


# ------------------------------------------------------- evaluation lock


def test_windows_permission_error_is_contention_not_failure():
    """The exact error that killed PPO must read as 'someone else holds it'."""

    assert is_contention_error(PermissionError(13, "Permission denied"))
    assert is_contention_error(BlockingIOError())
    for winerror in (5, 32, 33, 167):
        exc = OSError("sharing violation")
        exc.winerror = winerror
        assert is_contention_error(exc), winerror
    assert not is_contention_error(OSError(2, "No such file"))
    assert not is_contention_error(ValueError("unrelated"))


def test_lock_waits_instead_of_raising_on_contention(monkeypatch):
    """A contended lock must back off and retry, never propagate the error."""

    attempts = {"n": 0}
    real_try = None

    def flaky(handle):
        attempts["n"] += 1
        if attempts["n"] <= 3:
            raise PermissionError(13, "Permission denied")
        return real_try(handle)

    import marine_race_arena.learning.rl_evaluation_lock as lock_module

    real_try = lock_module._try_lock
    monkeypatch.setattr(lock_module, "_try_lock", flaky)
    waits = []
    slept = []
    with evaluation_lock(
        "ppo", on_wait=waits.append, sleeper=slept.append,
        rng=random.Random(0),
    ) as lease:
        assert lease["owner"] == "ppo"
    assert attempts["n"] == 4
    assert len(waits) == 3
    assert all(record["state"] == WAITING_STATE for record in waits)
    assert len(slept) == 3 and all(delay > 0 for delay in slept)


def test_backoff_is_bounded_exponential_with_jitter():
    delays = backoff_delays(8, rng=random.Random(1))
    assert len(delays) == 8
    assert all(0.0 <= d <= BACKOFF_CAP_SECONDS * 1.3 for d in delays)
    # Grows then plateaus at the cap.
    assert delays[0] < delays[3]
    assert max(delays[4:]) <= BACKOFF_CAP_SECONDS * (1 + 0.25) + 1e-9
    a = backoff_delays(4, rng=random.Random(2))
    b = backoff_delays(4, rng=random.Random(3))
    assert a != b, "jitter must decorrelate competing waiters"


def test_lock_is_mutually_exclusive_across_threads():
    order = []
    started = threading.Event()

    def hold():
        with evaluation_lock("sac"):
            order.append("sac_in")
            started.set()
            time.sleep(0.6)
            order.append("sac_out")

    thread = threading.Thread(target=hold)
    thread.start()
    started.wait(5)
    with evaluation_lock("ppo", sleeper=time.sleep, rng=random.Random(0)):
        order.append("ppo_in")
    thread.join(10)
    assert order == ["sac_in", "sac_out", "ppo_in"]


def test_lock_reports_a_waiting_state_not_a_crash(monkeypatch):
    import marine_race_arena.learning.rl_evaluation_lock as lock_module

    calls = {"n": 0}
    original = lock_module._try_lock

    def once(handle):
        calls["n"] += 1
        if calls["n"] == 1:
            return False
        return original(handle)

    monkeypatch.setattr(lock_module, "_try_lock", once)
    seen = []
    with evaluation_lock("ppo", on_wait=seen.append, sleeper=lambda d: None):
        pass
    assert seen and seen[0]["state"] == "WAITING_FOR_EVALUATION_SLOT"
    assert seen[0]["attempt"] == 1


def test_stale_owner_is_detected():
    assert owner_is_stale({}) is False
    assert owner_is_stale({"pid": 999_999_999, "acquired_monotonic": time.time()})
    assert owner_is_stale(
        {"pid": 0, "acquired_monotonic": time.time() - 100_000.0}
    )


def test_lock_can_time_out_explicitly(monkeypatch):
    import marine_race_arena.learning.rl_evaluation_lock as lock_module

    monkeypatch.setattr(lock_module, "_try_lock", lambda handle: False)
    with pytest.raises(TimeoutError):
        with evaluation_lock("ppo", max_wait_seconds=0.0, sleeper=lambda d: None):
            pass


# --------------------------------------------------- supervisor resilience


def test_lock_contention_never_consumes_restart_budget():
    assert supervisor.restart_budget("evaluation_lock_contention") is None
    assert not supervisor.budget_exhausted("evaluation_lock_contention", 500)


@pytest.mark.parametrize("reason,attempts,expected", [
    ("learning_collapse", 0, True),
    ("invalid_checkpoint", 0, True),
    ("unexpected_trainer_crash", 2, False),
    ("unexpected_trainer_crash", 3, True),
    ("transient_engine_startup", 4, False),
    ("transient_engine_startup", 5, True),
])
def test_restart_budgets_are_reason_specific(reason, attempts, expected):
    assert supervisor.budget_exhausted(reason, attempts) is expected


def test_failure_reason_classification(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    assert supervisor.classify_failure_reason(run, {}) == "unknown"
    (run / "failure.txt").write_text(
        "PermissionError: Permission denied ... evaluation.lock", encoding="utf-8")
    assert supervisor.classify_failure_reason(run, {}) == "evaluation_lock_contention"
    (run / "failure.txt").write_text(
        "TimeoutError: Timed out or error waiting for engine!", encoding="utf-8")
    assert supervisor.classify_failure_reason(run, {}) == "transient_engine_startup"
    (run / "failure.txt").write_text("boom", encoding="utf-8")
    collapsed = {"checkpoint_selection_state": {"rollback": {"x": 1}},
                 "message": "baseline_collapse_rollback"}
    assert supervisor.classify_failure_reason(run, collapsed) == "learning_collapse"


def test_supervisor_keeps_managing_a_healthy_algorithm(monkeypatch, tmp_path):
    """One failed algorithm must not end supervision of the other."""

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "config.json").write_text(json.dumps({
        "interval_seconds": 1,
        "comparison_output": str(tmp_path / "cmp"),
        "algorithms": {
            "ppo": {"algorithm": "ppo", "run_dir": str(tmp_path / "ppo"),
                    "worktree": str(tmp_path), "config": "c.json"},
            "sac": {"algorithm": "sac", "run_dir": str(tmp_path / "sac"),
                    "worktree": str(tmp_path), "config": "c.json"},
        },
    }), encoding="utf-8")

    def fake_inspect(spec, previous, retries):
        if spec["algorithm"] == "ppo":
            return {"algorithm": "ppo", "state": "FAILED",
                    "error": "restart budget exhausted", "restart_attempts": 3,
                    "trainer_pids": [], "trainer_status": {}}
        return {"algorithm": "sac", "state": "TRAINING", "restart_attempts": 0,
                "trainer_pids": [4242], "trainer_status": {}}

    monkeypatch.setattr(supervisor, "inspect_algorithm", fake_inspect)
    monkeypatch.setattr(supervisor, "sample_resources", lambda: {"cpu_percent": 30.0})
    monkeypatch.setattr(supervisor, "_refresh_comparison", lambda *a, **k: False)
    code = supervisor.worker(state_dir, once=True)
    status = json.loads((state_dir / "status.json").read_text(encoding="utf-8"))
    assert code == 0
    assert status["algorithms"]["sac"]["state"] == "TRAINING"
    assert status["algorithms"]["ppo"]["state"] == "FAILED"
    # SAC is still manageable, so the supervisor is not in a terminal state.
    assert status["manageable_algorithms"] == ["sac"]
    assert status["state"] != "FAILED"


# ------------------------------------------------------- sealed circuits


def test_all_three_official_circuits_are_recognized():
    assert len(FINAL_CIRCUIT_TRACKS) == 3
    for track in FINAL_CIRCUIT_TRACKS:
        assert is_final_circuit(track)
    assert is_final_circuit("marine_race_arena/tracks/marine_race_horseshoe_bay.json")
    assert is_final_circuit(r"C:\x\MARINE_RACE_VERTICAL_SERPENT.JSON")
    # A renamed copy that still carries the circuit identity is still sealed.
    assert is_final_circuit("my_copy_of_mixed_endurance_rotated.json")
    assert not is_final_circuit("marine_race_arena/tracks/training/stage3_three_gates.json")


@pytest.mark.parametrize("purpose", [
    "train", "training", "curriculum", "replay", "probe",
    "checkpoint_ranking", "hyperparameter", "reward_tuning", "validation",
])
def test_final_circuits_are_refused_for_every_training_purpose(purpose):
    for track in FINAL_CIRCUIT_TRACKS:
        with pytest.raises(SealedCircuitViolation):
            assert_not_final_circuit(track, purpose=purpose)


@pytest.mark.parametrize("purpose", [
    "final_holdout", "milestone_holdout", "final_comparison",
])
def test_final_circuits_are_allowed_only_for_holdout_purposes(purpose):
    for track in FINAL_CIRCUIT_TRACKS:
        assert_not_final_circuit(track, purpose=purpose)


def test_procedural_tracks_are_never_blocked():
    assert_no_final_circuits(
        ["tracks/training/stage3_three_gates.json", "generated/active_episode.json"],
        purpose="train",
    )


def test_final_circuit_use_is_recorded(tmp_path):
    ledger = tmp_path / "final_circuit_usage.jsonl"
    record_final_circuit_use(
        ledger, purpose="milestone_holdout", circuits=list(FINAL_CIRCUIT_TRACKS),
        checkpoint="ppo_534528_steps.zip", transitions=534528,
    )
    rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    assert len(rows) == 1 and rows[0]["purpose"] == "milestone_holdout"
    assert len(rows[0]["circuits"]) == 3
    with pytest.raises(SealedCircuitViolation):
        record_final_circuit_use(ledger, purpose="train", circuits=["x"])


# ------------------------------------------------ train/validation/test split


def test_seed_bands_are_disjoint():
    assert groups_are_disjoint()
    bands = [seed_group(name) for name in ("train", "validation", "test")]
    for i, a in enumerate(bands):
        for b in bands[i + 1:]:
            assert not (a.start < b.end and b.start < a.end), (a.name, b.name)


def test_a_seed_belongs_to_exactly_one_role():
    for name in ("train", "validation", "test"):
        group = seed_group(name)
        for seed in (group.start, group.start + 7, group.end - 1):
            assert role_of_seed(seed) == name
            assert_seed_role(seed, expected=name)


def test_using_a_validation_seed_for_training_is_refused():
    validation = seed_group("validation").start
    with pytest.raises(ValueError, match="validation"):
        assert_seed_role(validation, expected="train")
    test_seed = seed_group("test").start
    with pytest.raises(ValueError, match="test"):
        assert_seed_role(test_seed, expected="train")
    with pytest.raises(ValueError, match="test"):
        assert_seed_role(test_seed, expected="validation")


def test_evaluation_seeds_are_fixed_reproducible_and_in_band():
    first = evaluation_seeds("validation", 100)
    assert first == evaluation_seeds("validation", 100)
    assert len(set(first)) == 100
    assert all(seed_group("validation").contains(s) for s in first)
    assert not set(first) & set(evaluation_seeds("test", 100))
    with pytest.raises(ValueError):
        evaluation_seeds("train", 10)


def test_policy_manifest_documents_the_split():
    manifest = policy_manifest()
    assert manifest["seed_groups_disjoint"] is True
    assert len(manifest["final_circuits"]) == 3
    assert "final_circuits" in manifest["data_roles"]


# --------------------------------------------------- sequence curriculum


def test_mixture_matches_the_requested_initial_distribution():
    curriculum = GenericSequenceCurriculum()
    mixture = curriculum.mixture
    assert mixture["focus"] == pytest.approx(0.30)
    assert mixture["short"] == pytest.approx(0.30)
    assert mixture["medium"] == pytest.approx(0.25)
    assert mixture["long"] == pytest.approx(0.15)
    assert sum(mixture.values()) == pytest.approx(1.0)


def test_final_stage_shifts_toward_long_sequences():
    curriculum = GenericSequenceCurriculum()
    curriculum.state.stage_index = len(curriculum.stages) - 1
    mixture = curriculum.mixture
    assert mixture["long"] > mixture["focus"]
    assert mixture["focus"] > 0.0, "short transitions are never removed"
    assert mixture["medium"] + mixture["long"] >= 0.55


def test_short_transitions_can_never_be_eliminated():
    with pytest.raises(ValueError, match="highest-signal"):
        normalized_mixture({"focus": 0.0, "short": 0.3, "medium": 0.3, "long": 0.4})


def test_sampled_lengths_follow_the_mixture_and_stay_in_range():
    curriculum = GenericSequenceCurriculum()
    rng = np.random.default_rng(11)
    counts = [curriculum.sample_gate_count(rng) for _ in range(6000)]
    assert min(counts) == 2 and max(counts) <= 24
    observed = empirical_mixture(counts)
    for bucket in BUCKETS:
        assert observed[bucket] == pytest.approx(
            curriculum.mixture[bucket], abs=0.03
        ), bucket
    for value in counts:
        assert any(low <= value <= high
                   for low, high in BUCKET_GATE_RANGES.values())


def test_promotion_requires_repeated_measured_generalization():
    curriculum = GenericSequenceCurriculum()
    good = {"n_eval": 112, "universal_transition_success_rate": 0.80,
            "long_sequence_completion_score": 0.60, "collision_episodes": 10}
    first = curriculum.observe_validation(good, 100_000)
    assert not first["promoted"], "one good evaluation is not enough"
    second = curriculum.observe_validation(good, 200_000)
    assert second["promoted"] and curriculum.state.stage_index == 1


def test_elapsed_transitions_alone_never_promote():
    curriculum = GenericSequenceCurriculum()
    weak = {"n_eval": 112, "universal_transition_success_rate": 0.50,
            "long_sequence_completion_score": 0.10, "collision_episodes": 10}
    for step in range(1, 12):
        curriculum.observe_validation(weak, step * 100_000)
    assert curriculum.state.stage_index == 0


def test_repeated_regression_demotes_the_stage():
    curriculum = GenericSequenceCurriculum()
    curriculum.state.stage_index = 2
    bad = {"n_eval": 112, "universal_transition_success_rate": 0.10,
           "long_sequence_completion_score": 0.0, "collision_episodes": 40}
    curriculum.observe_validation(bad, 10)
    record = curriculum.observe_validation(bad, 20)
    assert record["demoted"] and curriculum.state.stage_index == 1


def test_curriculum_state_round_trip():
    curriculum = GenericSequenceCurriculum()
    good = {"n_eval": 112, "universal_transition_success_rate": 0.80,
            "long_sequence_completion_score": 0.60, "collision_episodes": 5}
    curriculum.observe_validation(good, 1)
    restored = GenericSequenceCurriculum()
    restored.load_state_dict(curriculum.state_dict())
    assert restored.state.stage_index == curriculum.state.stage_index
    assert restored.mixture == curriculum.mixture


# ------------------------------------------------------------ reward audit


def test_reward_contract_has_no_unbounded_or_duplicated_term():
    report = build_report()
    assert report["critical_failures"] == []
    assert report["verdict"] == "reward_contract_coherent_freeze_it"
    names = {row["check"] for row in report["checks"]}
    for required in (
        "every_component_is_clipped",
        "collision_is_event_based_not_per_frame",
        "progress_shaping_cannot_be_farmed_by_oscillation",
        "leaving_the_course_is_never_profitable",
        "stalling_is_not_rewarded",
        "circling_a_gate_is_penalized",
    ):
        assert required in names


def test_audit_flags_an_event_penalty_that_fires_every_frame():
    from marine_race_arena.learning.reward_audit import (
        empirical_findings, summarize_episode_components,
    )

    summary = summarize_episode_components([
        {"collision_penalty": [-50.0] * 400}
    ])
    findings = empirical_findings(summary)
    assert any("fires_like_a_dense_term" in row["check"] for row in findings)
