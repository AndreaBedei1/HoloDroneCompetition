"""Tests that the final benchmark cannot reach the sealed circuits unfrozen.

``rl_readiness_gate`` already owned the *decision*; what is under test here is
the **wiring** -- that ``final_benchmark``'s official groups actually ask it, on
every route that turns a case into a result, and that opening the holdout leaves
a ledger entry behind.  The interesting cases are therefore the refusals: no
freeze record, an unready verdict, a checkpoint swapped after freezing, and a
hand-built case that quietly drops its ``official`` flag.

Nothing here starts a simulator: the one test that exercises a whole official
episode replaces the runner, the adapter and the controller loader with fakes,
so the sealed track file is never opened.
"""

from __future__ import annotations

import copy
import json

import pytest

from marine_race_arena.learning import final_benchmark as fb
from marine_race_arena.learning.multigate_curriculum import HORSESHOE
from marine_race_arena.learning.provenance import sha256_file
from marine_race_arena.learning.rl_readiness_gate import (
    FINAL_CIRCUIT_TRIAL_SEEDS,
    FREEZE_RECORD_FILENAME,
    FREEZE_RECORD_VERSION,
    READINESS_THRESHOLDS,
    FinalCircuitsLocked,
    ProtocolViolation,
    assert_final_circuits_unlocked,
    evaluate_readiness,
    freeze_record_id,
    freeze_record_path,
    read_final_circuit_ledger,
    record_exploratory_holdout_decision,
    verify_ledger_chain,
)

OFFICIAL_GROUP = "official_horseshoe_bay"
#: Two cheap non-official groups: enough to prove a suite still builds and runs
#: on a machine that has never frozen a policy.
OPEN_GROUPS = ("two_gate_straight", "three_gate_sequence")

GIT_SHA = "0123456789abcdef0123456789abcdef01234567"


def _freeze(tmp_path, name: str = "frozen"):
    """A freeze record satisfying every check the access guard actually makes.

    Written directly rather than through ``freeze_policy`` so these tests stay
    pinned to the *guard's* contract -- self-consistent content hash, ready
    verdict, pre-registered thresholds, checkpoint bytes that still hash to the
    frozen value -- instead of having to restate whatever the readiness criteria
    currently are.  Those criteria, and the freeze that produces such a record
    from real evidence, are ``test_rl_readiness_gate.py``'s subject; what is
    under test here is only whether the benchmark asks.

    Returns ``(checkpoint, record, record_path)``.
    """

    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "ppo_8400000_steps.zip"
    checkpoint.write_bytes(b"frozen policy weights")
    record = {
        "schema_version": FREEZE_RECORD_VERSION,
        "utc": "2026-08-08T00:00:00Z",
        "git_sha": GIT_SHA,
        "policy": {
            "checkpoint": checkpoint.name,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_bytes": int(checkpoint.stat().st_size),
            "run_dir": str(run_dir),
            "algorithm": "ppo",
            "training_transitions": 8_400_000,
        },
        "readiness_verdict": {
            "ready": True,
            "failures": [],
            "thresholds_sha256": READINESS_THRESHOLDS.sha256(),
        },
        "readiness_thresholds_sha256": READINESS_THRESHOLDS.sha256(),
    }
    record["freeze_id"] = freeze_record_id(record)
    path = freeze_record_path(run_dir)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return checkpoint, record, path


def test_the_test_freeze_record_really_does_unlock_the_gate(tmp_path):
    """Guards the guard: if this stopped unlocking, every permit test would lie."""

    _, record, path = _freeze(tmp_path)
    assert assert_final_circuits_unlocked(path)["freeze_id"] == record["freeze_id"]


def _rewritten(record: dict, path, **mutations):
    """Write a freeze record with a *self-consistent* hash after mutation.

    Re-hashing matters: a mutation that broke the hash would be refused for the
    wrong reason, and the test would stop proving what it claims to.
    """

    mutated = copy.deepcopy(record)
    for key, value in mutations.items():
        mutated[key] = value
    mutated["freeze_id"] = freeze_record_id(mutated)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mutated), encoding="utf-8")
    return path


def _missing(tmp_path):
    return tmp_path / "nowhere" / FREEZE_RECORD_FILENAME


def _args(*argv):
    return fb.build_parser().parse_args(list(argv))


def _official_case(seed: int = FINAL_CIRCUIT_TRIAL_SEEDS[0], **overrides):
    """A sealed-circuit case built by hand, so no track file is ever opened."""

    fields = {
        "group": OFFICIAL_GROUP,
        "case_id": "horseshoe_bay",
        "track": HORSESHOE,
        "seeds": (int(seed),),
        "reused_seeds": (int(seed),),
        "holdout_seeds": (),
        "official": True,
        "benchmark_task": None,
        "geometry": None,
        "expected_gates": 12,
        "max_duration_s": 600.0,
    }
    fields.update(overrides)
    return fb.BenchmarkCase(**fields)


# --------------------------------------------------------------------------- #
# The pre-freeze peek guard
# --------------------------------------------------------------------------- #
def test_building_an_official_group_without_a_freeze_record_is_refused(tmp_path):
    """The single most important refusal: no freeze record, no sealed circuits."""

    with pytest.raises(FinalCircuitsLocked) as error:
        fb.build_suite(
            tmp_path / "out",
            episodes_per_case=2,
            official_episodes=2,
            groups=[OFFICIAL_GROUP],
            freeze_record=_missing(tmp_path),
        )
    assert OFFICIAL_GROUP in str(error.value)


def test_an_unready_verdict_keeps_the_official_group_sealed(tmp_path):
    _, record, _ = _freeze(tmp_path)
    verdict = copy.deepcopy(record["readiness_verdict"])
    verdict["ready"] = False
    verdict["failures"] = ["completion_12_gate"]
    path = _rewritten(
        record, tmp_path / "unready" / FREEZE_RECORD_FILENAME, readiness_verdict=verdict
    )

    with pytest.raises(FinalCircuitsLocked, match="not ready"):
        fb.build_suite(
            tmp_path / "out",
            episodes_per_case=2,
            official_episodes=2,
            groups=[OFFICIAL_GROUP],
            freeze_record=path,
        )


def test_a_valid_ready_freeze_record_permits_the_official_group(tmp_path):
    _, _, path = _freeze(tmp_path)
    cases = fb.build_suite(
        tmp_path / "out",
        episodes_per_case=2,
        official_episodes=2,
        groups=[OFFICIAL_GROUP],
        freeze_record=path,
    )
    assert [case.group for case in cases] == [OFFICIAL_GROUP]
    assert all(case.official for case in cases)


def test_committed_exploratory_failure_amendment_permits_official_group(tmp_path):
    """The explicit one-shot exception is wired through the benchmark door."""

    checkpoint = tmp_path / "ppo_final_generic_929792.zip"
    checkpoint.write_bytes(b"immutable final policy")
    metrics = {
        "seed": 5_000_000,
        "dataset_split": "validation",
        "episode_seeds": [5_000_007],
        "difficulty": "G1",
        "n_eval": 620,
        "transition_n": 500,
        "universal_transition_success_rate": 0.82,
        "first_gate_crossing_rate": 1.0,
        "target_switch_rate": 0.882,
        "new_target_alignment_rate": 0.852,
        "out_of_bounds_episodes": 0,
        "collision_episodes": 0,
        "full_sequence_success_by_length": {
            "3": {"n": 20, "completion_rate": 0.95},
            "5": {"n": 20, "completion_rate": 0.90},
            "8": {"n": 20, "completion_rate": 0.75},
            "12": {"n": 20, "completion_rate": 0.50},
            "17": {"n": 20, "completion_rate": 0.25},
            "22": {"n": 20, "completion_rate": 0.25},
        },
        "difficulty_ladder": {
            "rungs": [{
                "difficulty": "G3",
                "n": 60,
                "success": 0.75,
                "success_ci": [0.62, 0.84],
                "first_gate": 1.0,
            }],
        },
    }
    verdict = evaluate_readiness(metrics)
    assert verdict.ready is False
    decision_path = tmp_path / "exploratory_holdout_decision.json"
    record_exploratory_holdout_decision(
        checkpoint,
        run_dir=tmp_path,
        metrics=metrics,
        verdict=verdict,
        git_sha=GIT_SHA,
        contracts={"reward": "r1", "observation": "o1", "action": "a1"},
        training_transitions=929_792,
        path=decision_path,
    )

    cases = fb.build_suite(
        tmp_path / "out",
        episodes_per_case=2,
        official_episodes=2,
        groups=[OFFICIAL_GROUP],
        freeze_record=decision_path,
    )
    assert [case.group for case in cases] == [OFFICIAL_GROUP]


def test_a_suite_without_official_groups_needs_no_freeze_record(tmp_path):
    """Non-official benchmarking must work on a machine that never froze one."""

    cases = fb.build_suite(
        tmp_path / "out",
        episodes_per_case=2,
        official_episodes=2,
        groups=list(OPEN_GROUPS),
        freeze_record=_missing(tmp_path),
    )
    assert {case.group for case in cases} == set(OPEN_GROUPS)
    assert not any(case.official for case in cases)
    assert not _missing(tmp_path).exists()


def test_the_guard_does_not_fire_for_a_selection_with_no_official_group():
    assert fb.official_groups_requested(list(OPEN_GROUPS)) == ()
    assert fb.assert_official_groups_permitted(list(OPEN_GROUPS)) is None
    # An empty or absent selection is "every group", which includes all three.
    assert set(fb.official_groups_requested(None)) == set(fb.official_group_names())
    assert set(fb.official_groups_requested([])) == set(fb.official_group_names())


def test_every_official_template_is_covered_by_the_guard():
    """A fourth official group must inherit the guard without a code change."""

    templates = fb.default_groups(episodes_per_case=1, official_episodes=1)
    declared = {t.group for t in templates if t.official}
    assert declared == set(fb.official_group_names())
    assert len(declared) == 3


# --------------------------------------------------------------------------- #
# The refusal has to be actionable
# --------------------------------------------------------------------------- #
def test_the_refusal_names_the_missing_freeze_record(tmp_path):
    path = _missing(tmp_path)
    with pytest.raises(FinalCircuitsLocked) as error:
        fb.assert_final_circuit_access("a test", freeze_record=path)
    message = str(error.value)
    assert "no freeze record exists at" in message
    assert str(path) in message
    assert "--freeze-record" in message


def test_the_refusal_names_an_unready_verdict_and_its_failing_criteria(tmp_path):
    _, record, _ = _freeze(tmp_path)
    verdict = copy.deepcopy(record["readiness_verdict"])
    verdict["ready"] = False
    verdict["failures"] = ["completion_12_gate", "target_switch"]
    path = _rewritten(
        record, tmp_path / "unready" / FREEZE_RECORD_FILENAME, readiness_verdict=verdict
    )

    with pytest.raises(FinalCircuitsLocked) as error:
        fb.assert_final_circuit_access("a test", freeze_record=path)
    message = str(error.value)
    assert "not ready" in message
    assert "completion_12_gate" in message


def test_the_refusal_names_a_checkpoint_that_changed_after_freezing(tmp_path):
    checkpoint, _, path = _freeze(tmp_path)
    checkpoint.write_bytes(b"a different policy entirely")

    with pytest.raises(FinalCircuitsLocked) as error:
        fb.assert_final_circuit_access("a test", freeze_record=path)
    message = str(error.value)
    assert "no longer hashes to the frozen" in message
    assert "sha256" in message


def test_the_refusal_names_a_missing_checkpoint(tmp_path):
    checkpoint, _, path = _freeze(tmp_path)
    checkpoint.unlink()

    with pytest.raises(FinalCircuitsLocked, match="is missing"):
        fb.assert_final_circuit_access("a test", freeze_record=path)


def test_the_refusal_names_a_record_edited_after_freezing(tmp_path):
    _, record, _ = _freeze(tmp_path)
    path = tmp_path / "edited" / FREEZE_RECORD_FILENAME
    path.parent.mkdir(parents=True)
    tampered = copy.deepcopy(record)
    tampered["note"] = "looks harmless"  # freeze_id deliberately left stale
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(FinalCircuitsLocked, match="edited after freezing"):
        fb.assert_final_circuit_access("a test", freeze_record=path)


def test_the_three_refusals_are_told_apart(tmp_path):
    """Distinct causes must not collapse into one indistinguishable message."""

    checkpoint, record, swapped = _freeze(tmp_path)
    verdict = copy.deepcopy(record["readiness_verdict"])
    verdict["ready"] = False
    unready = _rewritten(
        record, tmp_path / "unready" / FREEZE_RECORD_FILENAME, readiness_verdict=verdict
    )
    checkpoint.write_bytes(b"a different policy entirely")

    causes = [_missing(tmp_path), unready, swapped]
    for path in causes:
        with pytest.raises(FinalCircuitsLocked):
            fb.assert_final_circuit_access("a test", freeze_record=path)
    assert len({fb._missing_requirement(path) for path in causes}) == 3


# --------------------------------------------------------------------------- #
# Every entry point, not just the convenient one
# --------------------------------------------------------------------------- #
def test_run_shard_is_refused_for_the_default_suite_without_a_freeze_record(tmp_path):
    """``--groups`` omitted means every group, so the circuits are in scope."""

    args = _args(
        "shard", "--out", str(tmp_path / "out"), "--shard", "0", "--shards", "1",
        "--freeze-record", str(_missing(tmp_path)),
    )
    with pytest.raises(FinalCircuitsLocked):
        fb.run_shard(args)
    assert not list((tmp_path / "out").glob("episodes.shard*.jsonl"))


def test_run_all_is_refused_for_the_default_suite_without_a_freeze_record(tmp_path):
    args = _args(
        "run", "--out", str(tmp_path / "out"), "--plan-only",
        "--freeze-record", str(_missing(tmp_path)),
    )
    with pytest.raises(FinalCircuitsLocked):
        fb.run_all(args)
    # Refused before planning, so no manifest advertises the sealed cases.
    assert not (tmp_path / "out" / "suite_manifest.json").exists()


def test_run_all_is_refused_even_when_only_planning(tmp_path):
    args = _args(
        "run", "--out", str(tmp_path / "out"), "--plan-only",
        "--groups", OFFICIAL_GROUP,
        "--freeze-record", str(_missing(tmp_path)),
    )
    with pytest.raises(FinalCircuitsLocked):
        fb.run_all(args)


def test_running_an_episode_on_a_sealed_circuit_is_refused(tmp_path):
    """The true reach point: refused before the controller is even loaded."""

    spec = fb.CONTROLLERS_BY_KEY["rule_gate_center_then_commit"]
    with pytest.raises(FinalCircuitsLocked):
        fb.run_benchmark_episode(
            spec,
            _official_case(),
            FINAL_CIRCUIT_TRIAL_SEEDS[0],
            freeze_record=_missing(tmp_path),
        )


def test_clearing_the_official_flag_does_not_unseal_the_track(tmp_path):
    """A hand-built case cannot smuggle a circuit through by lying about itself."""

    spec = fb.CONTROLLERS_BY_KEY["rule_gate_center_then_commit"]
    disguised = _official_case(official=False, group="two_gate_straight")
    with pytest.raises(FinalCircuitsLocked):
        fb.run_benchmark_episode(
            spec, disguised, 1800, freeze_record=_missing(tmp_path)
        )


def test_planning_official_episodes_still_cannot_run_them(tmp_path):
    """``plan_episodes`` is metadata, so it is guarded where it turns into a run.

    Planning is left open on purpose -- the manifest and the report describe the
    official groups -- and this pins the consequence: a planned official episode
    is still refused at the moment it would be executed.
    """

    case = _official_case()
    spec = fb.CONTROLLERS_BY_KEY["rule_gate_center_then_commit"]
    planned = fb.plan_episodes([case], [spec])
    assert [e.group for e in planned] == [OFFICIAL_GROUP]

    with pytest.raises(FinalCircuitsLocked):
        fb.run_benchmark_episode(
            spec, case, planned[0].seed, freeze_record=_missing(tmp_path)
        )


def test_a_shard_without_official_groups_runs_with_no_freeze_record(
    tmp_path, monkeypatch
):
    ran = []

    def fake_episode(spec, case, seed, **kwargs):
        ran.append((spec.key, case.uid, int(seed)))
        return {
            "controller": spec.key, "group": case.group, "case_id": case.case_id,
            "seed": int(seed), "referee_status": "FINISHED",
            "official_circuit": case.official,
        }

    monkeypatch.setattr(fb, "run_benchmark_episode", fake_episode)
    ledger = tmp_path / "ledger.jsonl"
    args = _args(
        "shard", "--out", str(tmp_path / "out"), "--shard", "0", "--shards", "1",
        "--groups", *OPEN_GROUPS, "--controllers", "rule_gate_center_then_commit",
        "--episodes-per-case", "2",
        "--freeze-record", str(_missing(tmp_path)),
        "--holdout-ledger", str(ledger),
    )
    assert fb.run_shard(args) == 0
    assert ran, "the shard must actually have run its non-official episodes"
    assert not any(uid.startswith("official_") for _, uid, _ in ran)
    assert not ledger.exists(), "no holdout was opened, so nothing may be recorded"


# --------------------------------------------------------------------------- #
# Opening the holdout leaves a trace
# --------------------------------------------------------------------------- #
def _finished_row(seed: int, **overrides) -> dict:
    row = {
        "controller": "ppo_1000814",
        "group": OFFICIAL_GROUP,
        "case_id": "horseshoe_bay",
        "seed": int(seed),
        "official_circuit": True,
        "referee_status": "FINISHED",
        "evaluation_end_reason": "FINISHED",
        "full_completion": True,
        "completed_gates": 12,
        "expected_gates": 12,
        "official_time_s": 84.5,
        "collision_events": 0,
        "collision_episode": False,
        "out_of_bounds_events": 0,
        "wrong_direction_crossings": 0,
        "missed_gate_attempts": 0,
        "previous_gate_returns": 0,
        "mean_action_jerk": 0.03,
    }
    row.update(overrides)
    return row


def test_recording_an_official_episode_appends_a_verifiable_ledger_entry(tmp_path):
    _, record, path = _freeze(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    case = _official_case()

    first = fb.record_official_episode(
        _finished_row(FINAL_CIRCUIT_TRIAL_SEEDS[0]),
        case,
        freeze_record=path,
        ledger=ledger,
    )
    second = fb.record_official_episode(
        _finished_row(FINAL_CIRCUIT_TRIAL_SEEDS[1], full_completion=False,
                      completed_gates=7, official_time_s=None,
                      evaluation_end_reason="TIME_LIMIT"),
        _official_case(FINAL_CIRCUIT_TRIAL_SEEDS[1]),
        freeze_record=path,
        ledger=ledger,
    )

    entries = verify_ledger_chain(ledger)
    assert [e["entry_index"] for e in entries] == [0, 1]
    assert entries[1]["previous_sha256"] == first["entry_sha256"]
    assert {e["circuit"] for e in entries} == {"horseshoe_bay"}
    assert all(e["freeze_id"] == record["freeze_id"] for e in entries)
    assert entries[0]["metrics"]["circuit_completed"] is True
    assert entries[0]["metrics"]["gates_crossed"] == 12
    assert entries[1]["metrics"]["circuit_completed"] is False
    assert entries[1]["metrics"]["gates_crossed_fraction"] == pytest.approx(7 / 12)
    assert second["entry_index"] == 1


def test_the_ledger_cannot_be_written_before_the_policy_is_frozen(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(FinalCircuitsLocked):
        fb.record_official_episode(
            _finished_row(FINAL_CIRCUIT_TRIAL_SEEDS[0]),
            _official_case(),
            freeze_record=_missing(tmp_path),
            ledger=ledger,
        )
    assert not ledger.exists()


def test_a_completed_official_episode_records_itself(tmp_path, monkeypatch):
    """End-to-end through ``run_benchmark_episode`` with the simulator faked out.

    This is the one test that proves the ledger call is really on the episode
    path rather than merely available next to it.
    """

    _, record, path = _freeze(tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    _fake_race(monkeypatch)

    spec = fb.CONTROLLERS_BY_KEY["rule_gate_center_then_commit"]
    case = _official_case()
    row = fb.run_benchmark_episode(
        spec,
        case,
        FINAL_CIRCUIT_TRIAL_SEEDS[0],
        adapter="fallback",
        freeze_record=path,
        ledger=ledger,
    )

    entries = read_final_circuit_ledger(ledger)
    assert len(entries) == 1
    assert entries[0]["circuit"] == "horseshoe_bay"
    assert entries[0]["trial_seeds"] == [FINAL_CIRCUIT_TRIAL_SEEDS[0]]
    assert entries[0]["checkpoint_sha256"] == record["policy"]["checkpoint_sha256"]
    assert row["holdout_ledger_entry_sha256"] == entries[0]["entry_sha256"]


def test_a_non_official_episode_leaves_the_ledger_alone(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.jsonl"
    _fake_race(monkeypatch)

    spec = fb.CONTROLLERS_BY_KEY["rule_gate_center_then_commit"]
    case = _official_case(
        official=False, group="two_gate_straight", case_id="straight",
        track="marine_race_two_gate_straight", seeds=(1,), reused_seeds=(1,),
    )
    row = fb.run_benchmark_episode(
        spec, case, 1, adapter="fallback", freeze_record=None, ledger=ledger
    )
    assert "holdout_ledger_entry_sha256" not in row
    assert not ledger.exists()


# --------------------------------------------------------------------------- #
# The protocol pins which seeds an official trial may use
# --------------------------------------------------------------------------- #
def test_official_cases_must_run_the_pre_registered_trial_seeds():
    """1800-1809 are the protocol's trial seeds; 1810+ are reserved but not it.

    The trial seeds are the first ten of seed_registry's published
    RESERVED_FINAL_MULTIGATE_SEEDS (1800-1899), which is also the pool the
    official benchmark groups draw from, so a seed can sit inside the
    reservation and still fall outside the thirty sanctioned trials.
    """

    fb.assert_official_trial_seeds_are_registered([_official_case()])
    fb.assert_official_trial_seeds_are_registered([_official_case(seed=1809)])
    with pytest.raises(ProtocolViolation, match="1850"):
        fb.assert_official_trial_seeds_are_registered(
            [_official_case(seed=1850)]
        )


def test_non_official_cases_are_not_bound_to_the_protocol_seeds():
    case = _official_case(seed=1850, official=False, group="two_gate_straight")
    fb.assert_official_trial_seeds_are_registered([case])


def test_a_shard_is_refused_before_running_unrecordable_official_trials(tmp_path):
    """A holdout that could be opened but not logged must never start."""

    _, _, path = _freeze(tmp_path)
    args = _args(
        "shard", "--out", str(tmp_path / "out"), "--shard", "0", "--shards", "1",
        "--groups", OFFICIAL_GROUP, "--controllers", "rule_gate_center_then_commit",
        "--official-episodes", "2", "--freeze-record", str(path),
    )
    with pytest.raises(ProtocolViolation, match="protocol never registered"):
        fb.run_shard(args)
    assert not list((tmp_path / "out").glob("episodes.shard*.jsonl"))


# --------------------------------------------------------------------------- #
# Path resolution and CLI surface
# --------------------------------------------------------------------------- #
def test_the_freeze_record_and_ledger_default_next_to_the_frozen_run(tmp_path):
    default = fb.resolve_freeze_record(None)
    assert default == fb.Path(fb.DEFAULT_FREEZE_RECORD)
    assert default.name == FREEZE_RECORD_FILENAME
    # One holdout, one ledger: it follows the record, not the benchmark output.
    assert fb.resolve_holdout_ledger(None, None).parent == default.parent
    explicit = tmp_path / "run" / FREEZE_RECORD_FILENAME
    assert fb.resolve_holdout_ledger(None, explicit).parent == explicit.parent
    assert fb.resolve_holdout_ledger(tmp_path / "l.jsonl", explicit) == (
        tmp_path / "l.jsonl"
    )


def test_both_run_and_shard_accept_the_freeze_record_flags():
    for command, extra in (
        ("run", []),
        ("shard", ["--shard", "0", "--shards", "1"]),
    ):
        args = _args(
            command, "--out", "out", "--freeze-record", "f.json",
            "--holdout-ledger", "l.jsonl", *extra,
        )
        assert args.freeze_record == "f.json"
        assert args.holdout_ledger == "l.jsonl"


def test_run_forwards_the_freeze_record_to_its_shards(tmp_path, monkeypatch):
    """A worker subprocess re-builds the suite, so it needs the same record."""

    _, _, path = _freeze(tmp_path)
    launched = []

    class _Process:
        def wait(self):
            return 0

    def fake_popen(command, **kwargs):
        launched.append(command)
        return _Process()

    monkeypatch.setattr(fb.subprocess, "Popen", fake_popen)
    ledger = tmp_path / "ledger.jsonl"
    args = _args(
        "run", "--out", str(tmp_path / "out"), "--workers", "2",
        "--groups", *OPEN_GROUPS, "--controllers", "rule_gate_center_then_commit",
        "--freeze-record", str(path), "--holdout-ledger", str(ledger),
    )
    assert fb.run_all(args) == 0
    # The manifest shells out for the git sha too, so keep only the shard launches.
    shards = [c for c in launched if "shard" in c]
    assert len(shards) == 2
    for command in shards:
        assert "--freeze-record" in command
        assert command[command.index("--freeze-record") + 1] == str(path)
        assert command[command.index("--holdout-ledger") + 1] == str(ledger)


def test_the_manifest_records_which_circuits_were_in_scope(tmp_path):
    args = _args(
        "run", "--out", str(tmp_path / "out"), "--plan-only",
        "--groups", *OPEN_GROUPS, "--controllers", "rule_gate_center_then_commit",
        "--freeze-record", str(_missing(tmp_path)),
    )
    assert fb.run_all(args) == 0
    manifest = json.loads(
        (tmp_path / "out" / "suite_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["official_groups"] == []
    assert manifest["freeze_record"] == str(_missing(tmp_path))


# --------------------------------------------------------------------------- #
# Fakes: a whole episode without a simulator
# --------------------------------------------------------------------------- #
class _FakeState:
    status = "FINISHED"
    valid_gate_crossings = 12
    penalties_s = 0.0
    collision_events = 0
    obstacle_collision_events = 0
    out_of_bounds_events = 0
    wrong_direction_crossings = 0
    missed_gate_attempts = 0
    stuck_events = 0
    position = (0.0, 0.0, -4.0)
    rotation_rpy_deg = (0.0, 0.0, 0.0)


class _FakeController:
    previous_gate_return_count = 0

    def step(self, observation):
        return {axis: 0.0 for axis in ("surge", "sway", "heave", "yaw")}

    def reset(self, mission):
        return None

    def close(self):
        return None


def _fake_race(monkeypatch):
    """Replace the runner, adapter and loader; the sealed track is never read."""

    from marine_race_arena.learning import episode as episode_module
    from marine_race_arena.participants import controller_loader as loader_module
    from marine_race_arena.scripts import run_marine_race as runner_module

    class _Bounds:
        x_min, x_max = -100.0, 100.0
        y_min, y_max = -100.0, 100.0
        z_min, z_max = -50.0, 0.0

    class _World:
        bounds = _Bounds()

    class _Track:
        gate_sequence = ()

    class _Config:
        world = _World()
        track = _Track()

    class _Adapter:
        name = "fake"

        def get_participant_state(self, pid):
            return _FakeState()

        def close(self):
            return None

    class _Referee:
        states = {"p0": _FakeState()}

        def summary(self):
            return {
                "participants": [
                    {"participant_id": "p0", "official_time_s": 84.5,
                     "penalized_time_s": 84.5}
                ]
            }

    class _Arena:
        gate_map = {}

    class _Participant:
        id = "p0"

    class _Ctx:
        config = _Config()
        adapter = _Adapter()
        referee = _Referee()
        arena = _Arena()
        participant = _Participant()

    monkeypatch.setattr(
        loader_module.ControllerLoader, "load",
        lambda self, alias, constructor_kwargs=None: _FakeController(),
    )
    monkeypatch.setattr(
        episode_module, "build_single_vehicle_race", lambda *a, **k: _Ctx()
    )
    monkeypatch.setattr(runner_module, "_mission_info", lambda config, pid: {})
    monkeypatch.setattr(runner_module, "_run_race_loop", lambda **kwargs: None)


# --------------------------------------------------------------------------- #
# The guard must be live code, not a documented intention
# --------------------------------------------------------------------------- #
def test_the_episode_guard_is_not_disabled():
    """This wiring once shipped with the guard behind ``if False:``.

    The behavioural tests above were correct and would have caught it, but they
    were not re-run after the edit, and the docstring above the dead branch still
    described the guarantee in full. An adversarial probe found the hole. This
    test reads the source so a guard that is switched off can never again look
    identical to a guard that is enforced.
    """

    import inspect
    import re

    source = inspect.getsource(fb.run_benchmark_episode)
    body = source.split('"""', 2)[-1]
    assert "assert_final_circuit_access(" in body, (
        "run_benchmark_episode no longer calls the access guard at all"
    )
    guarded = re.search(
        r"if\s+(?P<cond>[^\n:]+):\s*\n\s+assert_final_circuit_access\(", body
    )
    assert guarded, "the access guard is not inside a conditional we can inspect"
    condition = guarded.group("cond").strip()
    assert condition not in {"False", "0", "None"}, (
        f"the sealed-circuit guard is disabled: `if {condition}:`"
    )
    # It must key on the track itself, not only on a flag a caller can clear.
    assert "is_final_circuit" in condition, (
        f"guard condition {condition!r} does not consult the track identity, so a "
        f"case that clears its official flag would slip a sealed circuit through"
    )
