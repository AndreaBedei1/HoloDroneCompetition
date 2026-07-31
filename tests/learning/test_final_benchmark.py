"""Tests for the final common benchmark suite, instrumentation and reporting.

Numpy-only: the suite construction, episode bookkeeping, aggregation, paired
comparison and controller selection are exercised without launching HoloOcean.
"""

import json
from pathlib import Path

import pytest

from marine_race_arena.learning import final_benchmark as fb
from marine_race_arena.learning import final_benchmark_report as fbr
from marine_race_arena.learning import seed_registry


# --------------------------------------------------------------------------- #
# Suite definition
# --------------------------------------------------------------------------- #
def test_suite_is_deterministic_and_covers_every_required_group(tmp_path):
    first = fb.build_suite(tmp_path / "a", episodes_per_case=8, official_episodes=5)
    second = fb.build_suite(tmp_path / "b", episodes_per_case=8, official_episodes=5)
    assert [(c.group, c.case_id, c.seeds) for c in first] == [
        (c.group, c.case_id, c.seeds) for c in second
    ]
    groups = {case.group for case in first}
    assert groups == {
        "single_gate_retention",
        "two_gate_straight",
        "two_gate_left",
        "two_gate_right",
        "vertical_low_to_high",
        "vertical_high_to_low",
        "three_gate_sequence",
        "three_gate_s_shape",
        "official_horseshoe_bay",
        "official_vertical_serpent",
        "official_mixed_endurance",
    }


def test_every_case_mixes_reused_and_holdout_seeds(tmp_path):
    for case in fb.build_suite(tmp_path, episodes_per_case=8, official_episodes=5):
        assert case.reused_seeds, case.uid
        assert case.holdout_seeds, case.uid
        assert set(case.seeds) == set(case.reused_seeds) | set(case.holdout_seeds)
        assert set(case.holdout_seeds) <= set(
            seed_registry.MULTIGATE_FINAL_BENCHMARK_HOLDOUT_SEEDS
        )


def test_holdout_seeds_were_never_used_for_training_or_selection():
    holdout = set(seed_registry.MULTIGATE_FINAL_BENCHMARK_HOLDOUT_SEEDS)
    for role in (
        "multigate_reliability_ppo_training",
        "multigate_reliability_dev_eval",
        "multigate_reliability_checkpoint_selection",
        "multigate_reliability_validation",
        "multigate_longrun_bc_training",
        "multigate_v3_ppo_training",
    ):
        assert not holdout & seed_registry.ROLE_SEED_SETS[role], role
    seed_registry.assert_pairwise_disjoint()
    seed_registry.assert_new_allocations_are_unused()
    assert seed_registry.development_and_final_are_disjoint()


def test_official_circuits_reuse_the_published_official_seed_prefix(tmp_path):
    official = [
        case
        for case in fb.build_suite(tmp_path, episodes_per_case=8, official_episodes=5)
        if case.official
    ]
    assert len(official) == 3
    prefixes = {case.reused_seeds for case in official}
    assert len(prefixes) == 1, "every official circuit must reuse the same seeds"
    assert set(prefixes.pop()) <= set(seed_registry.RESERVED_FINAL_MULTIGATE_SEEDS)


def test_vertical_groups_move_the_second_gate_in_opposite_directions(tmp_path):
    cases = {
        case.case_id: case
        for case in fb.build_suite(tmp_path, episodes_per_case=8, official_episodes=5)
        if case.group in ("vertical_low_to_high", "vertical_high_to_low")
    }
    assert cases, "vertical transition cases must exist"
    for case_id, case in cases.items():
        step = case.geometry["vertical_displacement_m"]
        gates = json.loads(Path(case.track).read_text(encoding="utf-8"))["gates"]
        delta = gates[1]["position"][2] - gates[0]["position"][2]
        assert pytest.approx(delta, abs=1e-6) == step
        if case_id.startswith("up_"):
            assert step > 0, "low-to-high must raise the second gate"
        else:
            assert step < 0, "high-to-low must lower the second gate"


def test_expected_gate_counts_match_the_official_circuits(tmp_path):
    by_group = {
        case.group: case
        for case in fb.build_suite(tmp_path, episodes_per_case=8, official_episodes=5)
    }
    assert by_group["official_horseshoe_bay"].expected_gates == 12
    assert by_group["official_vertical_serpent"].expected_gates == 17
    assert by_group["official_mixed_endurance"].expected_gates == 22
    assert by_group["three_gate_sequence"].expected_gates == 3
    assert by_group["single_gate_retention"].expected_gates == 1


def test_plan_runs_every_controller_on_every_case_seed(tmp_path):
    cases = fb.build_suite(tmp_path, episodes_per_case=2, official_episodes=2)
    controllers = fb.default_controllers()
    episodes = fb.plan_episodes(cases, controllers)
    assert len(episodes) == sum(len(c.seeds) for c in cases) * len(controllers)
    assert len({e.key for e in episodes}) == len(episodes)
    # Non-official groups run first so the cheap evidence lands before the circuits.
    official_start = next(
        i for i, e in enumerate(episodes) if e.group.startswith("official_")
    )
    assert all(
        not e.group.startswith("official_") for e in episodes[:official_start]
    )


def test_every_case_seed_is_identical_across_controllers(tmp_path):
    cases = fb.build_suite(tmp_path, episodes_per_case=4, official_episodes=2)
    episodes = fb.plan_episodes(cases, fb.default_controllers())
    per_controller = {}
    for episode in episodes:
        per_controller.setdefault(episode.controller, set()).add(
            (episode.group, episode.case_id, episode.seed)
        )
    reference = next(iter(per_controller.values()))
    assert all(value == reference for value in per_controller.values())


# --------------------------------------------------------------------------- #
# Policy-only enforcement
# --------------------------------------------------------------------------- #
class _FakeController:
    def __init__(self, **attributes):
        for name, value in attributes.items():
            setattr(self, name, value)


def _ppo_spec():
    return fb.CONTROLLERS_BY_KEY["ppo_1000814"]


def test_policy_only_check_accepts_a_clean_learned_controller():
    evidence = fb._assert_policy_only(
        _ppo_spec(),
        _FakeController(
            rule_action_weight=0.0,
            hybrid_blending=False,
            rule_controller_instantiated=False,
            deterministic_runtime_intervention_count=0,
        ),
    )
    assert evidence["rule_action_weight"] == 0.0


@pytest.mark.parametrize(
    "taint",
    [
        {"rule_action_weight": 0.25},
        {"hybrid_blending": True},
        {"rule_controller_instantiated": True},
        {"deterministic_runtime_intervention_count": 3},
    ],
)
def test_policy_only_check_rejects_any_runtime_rule_contribution(taint):
    attributes = {
        "rule_action_weight": 0.0,
        "hybrid_blending": False,
        "rule_controller_instantiated": False,
        "deterministic_runtime_intervention_count": 0,
    }
    attributes.update(taint)
    with pytest.raises(RuntimeError, match="policy-only"):
        fb._assert_policy_only(_ppo_spec(), _FakeController(**attributes))


def test_hybrid_is_flagged_for_separate_reporting_and_never_policy_only():
    hybrid = fb.CONTROLLERS_BY_KEY["hybrid"]
    assert hybrid.separate_reporting
    assert not hybrid.policy_only
    # A non-policy-only controller is described, never rejected.
    assert fb._assert_policy_only(hybrid, _FakeController()) == {
        "rule_action_weight": None,
        "hybrid_blending": None,
        "rule_controller_instantiated": None,
        "deterministic_runtime_intervention_count": None,
    }


# --------------------------------------------------------------------------- #
# Episode instrumentation
# --------------------------------------------------------------------------- #
def test_out_of_bounds_detection_uses_all_three_axes():
    bounds = (-10.0, 10.0, -10.0, 10.0, -10.0, 0.0)
    assert not fb._outside((0.0, 0.0, -4.0), bounds)
    assert fb._outside((11.0, 0.0, -4.0), bounds)
    assert fb._outside((0.0, -11.0, -4.0), bounds)
    assert fb._outside((0.0, 0.0, 1.0), bounds)


def test_previous_gate_return_needs_a_real_clearance_then_a_real_return():
    gates = [{"gate_id": "G01", "center": [0.0, 0.0, -4.0], "normal": [1.0, 0.0, 0.0]}]
    forward_only = [
        {"gates": 1, "position": [0.5, 0.0, -4.0]},
        {"gates": 1, "position": [2.0, 0.0, -4.0]},
        {"gates": 1, "position": [3.0, 0.0, -4.0]},
    ]
    assert fb._previous_gate_returns(forward_only, gates) == 0

    went_back = forward_only + [{"gates": 1, "position": [-0.5, 0.0, -4.0]}]
    assert fb._previous_gate_returns(went_back, gates) == 1

    # Returning twice counts twice, but only after clearing the gate again.
    twice = went_back + [
        {"gates": 1, "position": [2.0, 0.0, -4.0]},
        {"gates": 1, "position": [-0.5, 0.0, -4.0]},
    ]
    assert fb._previous_gate_returns(twice, gates) == 2


def test_previous_gate_return_ignores_jitter_inside_the_clearance_band():
    gates = [{"gate_id": "G01", "center": [0.0, 0.0, -4.0], "normal": [1.0, 0.0, 0.0]}]
    jitter = [
        {"gates": 1, "position": [0.4, 0.0, -4.0]},
        {"gates": 1, "position": [0.1, 0.0, -4.0]},
        {"gates": 1, "position": [0.6, 0.0, -4.0]},
    ]
    assert fb._previous_gate_returns(jitter, gates) == 0


def test_probe_records_actions_path_and_collision_frames():
    class _Controller:
        def step(self, observation):
            return {"surge": 1.0, "sway": 0.0, "heave": 0.0, "yaw": 0.0}

    class _State:
        def __init__(self, position):
            self.position = position
            self.rotation_rpy_deg = (0.0, 0.0, 0.0)

    class _Referee:
        valid_gate_crossings = 0

    class _Ctx:
        class config:
            class world:
                class bounds:
                    x_min, x_max = -10.0, 10.0
                    y_min, y_max = -10.0, 10.0
                    z_min, z_max = -10.0, 0.0

        def __init__(self):
            self.positions = iter([(0.0, 0.0, -4.0), (1.0, 0.0, -4.0), (1.0, 4.0, -4.0)])
            self.referee = type("R", (), {"states": {"p": _Referee()}})()

        class adapter:
            pass

    ctx = _Ctx()
    ctx.adapter = type(
        "A", (), {"get_participant_state": lambda self, pid: _State(next(ctx.positions))}
    )()
    controller = _Controller()
    probe = fb._EpisodeProbe(controller)
    probe.bind(ctx, "p")
    controller.step({"local_time_s": 0.0, "sensors": {"CollisionSensor": [0]}})
    controller.step({"local_time_s": 0.1, "sensors": {"CollisionSensor": [1]}})
    controller.step({"local_time_s": 0.2, "sensors": {}})

    assert probe.steps == 3
    assert probe.collision_frames == 1
    assert probe.path_length_m == pytest.approx(5.0)
    assert len(probe.trajectory) == 3
    metrics = probe.action_metrics()
    assert metrics["mean_action_jerk"] == pytest.approx(0.0)
    assert metrics["actions_finite"] is True


def test_camera_frame_is_downscaled_to_three_channels():
    import numpy as np

    frame = np.zeros((8, 8, 4), dtype=np.uint8)
    frame[..., 2] = 255  # BGRA blue channel
    out = fb._camera_frame({"FrontCamera": frame})
    assert out.shape == (4, 4, 3)
    assert out[0, 0, 0] == 255  # converted to RGB


# --------------------------------------------------------------------------- #
# Aggregation and comparison
# --------------------------------------------------------------------------- #
def _episode(**overrides):
    row = {
        "controller": "ppo_a",
        "group": "two_gate_straight",
        "case_id": "straight",
        "case_uid": "two_gate_straight/straight",
        "seed": 1,
        "seed_role": "reused",
        "official_circuit": False,
        "referee_status": "FINISHED",
        "evaluation_end_reason": "FINISHED",
        "finished": True,
        "full_completion": True,
        "completed_gates": 2,
        "expected_gates": 2,
        "official_time_s": 6.0,
        "penalized_time_s": 6.0,
        "time_per_gate_s": 3.0,
        "collision_events": 0,
        "collision_frames": 0,
        "collision_episode": False,
        "out_of_bounds_events": 0,
        "out_of_bounds_frames": 0,
        "out_of_bounds_episode": False,
        "wrong_direction_crossings": 0,
        "wrong_direction_episode": False,
        "missed_gate_attempts": 0,
        "stuck_events": 0,
        "any_safety_event": False,
        "previous_gate_returns": 0,
        "timeout": False,
        "path_length_m": 10.0,
        "mean_action_jerk": 0.02,
        "action_saturation": 0.0,
        "mean_inference_ms": 1.0,
    }
    row.update(overrides)
    return row


def test_aggregate_counts_events_frames_and_episodes_separately():
    rows = [
        _episode(collision_events=3, collision_frames=40, collision_episode=True,
                 any_safety_event=True),
        _episode(seed=2, collision_events=0),
    ]
    aggregate = fbr.aggregate_rows(rows)
    assert aggregate["collision_events"] == 3
    assert aggregate["collision_frames"] == 40
    assert aggregate["collision_episodes"] == 1
    assert aggregate["safety_event_episodes"] == 1
    assert aggregate["episodes"] == 2


def test_aggregate_reports_timeout_rate_and_full_completion_separately():
    rows = [
        _episode(),
        _episode(seed=2, finished=False, full_completion=False, completed_gates=1,
                 referee_status="RUNNING", evaluation_end_reason="TIME_LIMIT",
                 timeout=True, official_time_s=None, penalized_time_s=None,
                 time_per_gate_s=None),
    ]
    aggregate = fbr.aggregate_rows(rows)
    assert aggregate["success_rate"] == 0.5
    assert aggregate["full_completion_rate"] == 0.5
    assert aggregate["timeout_rate"] == 0.5
    assert aggregate["completion_time_s"]["n"] == 1
    assert aggregate["gates_completed_total"] == 3
    assert aggregate["gates_expected_total"] == 4


def test_single_gate_group_is_excluded_from_timing():
    rows = [
        _episode(group="single_gate_retention", official_time_s=0.0,
                 penalized_time_s=0.0, time_per_gate_s=0.0, expected_gates=1,
                 completed_gates=1),
    ]
    aggregate = fbr.aggregate_rows(rows)
    assert aggregate["success_rate"] == 1.0
    assert aggregate["completion_time_s"]["mean"] is None
    assert aggregate["timing_meaningful"] is False


def test_wilson_interval_brackets_the_point_estimate():
    low, high = fbr.wilson_interval(8, 8)
    assert low < 1.0 <= high
    assert fbr.wilson_interval(0, 0) == (None, None)


def test_sign_test_is_symmetric_and_significant_for_a_clean_sweep():
    assert fbr.sign_test_p(3, 3) == 1.0
    assert fbr.sign_test_p(6, 0) == fbr.sign_test_p(0, 6)
    assert fbr.sign_test_p(6, 0) < 0.05
    assert fbr.sign_test_p(0, 0) is None


def test_paired_comparison_only_uses_shared_case_and_seed():
    rows = [
        _episode(controller="a", seed=1, official_time_s=6.0),
        _episode(controller="a", seed=2, official_time_s=6.0),
        _episode(controller="b", seed=1, official_time_s=8.0),
        # seed 3 exists only for b and must be ignored entirely
        _episode(controller="b", seed=3, official_time_s=1.0),
    ]
    comparison = fbr.paired_comparison(rows, "a", "b")
    assert comparison["paired_episodes"] == 1
    assert comparison["time_delta_s"]["n"] == 1
    assert comparison["time_delta_s"]["mean"] == pytest.approx(-2.0)
    assert comparison["a_faster_episodes"] == 1


def test_paired_comparison_ignores_untimed_groups_for_time_but_not_reliability():
    rows = [
        _episode(controller="a", group="single_gate_retention",
                 case_uid="single_gate_retention/single_gate", official_time_s=0.0),
        _episode(controller="b", group="single_gate_retention",
                 case_uid="single_gate_retention/single_gate", official_time_s=0.0,
                 finished=False, full_completion=False),
    ]
    comparison = fbr.paired_comparison(rows, "a", "b")
    assert comparison["paired_episodes"] == 1
    assert comparison["only_a_finished"] == 1
    assert comparison["time_delta_s"]["n"] == 0


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def _controller_rows(controller, *, official_full, safety, three_gate_ok, time_s, jerk):
    rows = []
    for seed in range(2):
        rows.append(_episode(controller=controller, seed=seed, official_time_s=time_s,
                             penalized_time_s=time_s, mean_action_jerk=jerk))
        rows.append(_episode(
            controller=controller, seed=seed, group="three_gate_sequence",
            case_uid="three_gate_sequence/three_gate_training", expected_gates=3,
            completed_gates=3 if three_gate_ok else 1, finished=three_gate_ok,
            full_completion=three_gate_ok, official_time_s=time_s * 2 if three_gate_ok else None,
            penalized_time_s=time_s * 2 if three_gate_ok else None,
            time_per_gate_s=None, mean_action_jerk=jerk,
        ))
        rows.append(_episode(
            controller=controller, seed=seed, group="vertical_low_to_high",
            case_uid="vertical_low_to_high/up", official_time_s=time_s,
            penalized_time_s=time_s, mean_action_jerk=jerk,
        ))
        rows.append(_episode(
            controller=controller, seed=seed, group="official_horseshoe_bay",
            case_uid="official_horseshoe_bay/horseshoe", official_circuit=True,
            expected_gates=12, completed_gates=12 if official_full else 3,
            finished=official_full, full_completion=official_full,
            official_time_s=60.0 if official_full else None,
            penalized_time_s=60.0 if official_full else None,
            time_per_gate_s=None, mean_action_jerk=jerk,
            any_safety_event=bool(safety), collision_events=safety,
            collision_episode=bool(safety),
        ))
    return rows


_META = {
    "reliable": {"kind": "ppo", "separate_reporting": False},
    "unsafe": {"kind": "ppo", "separate_reporting": False},
    "fast": {"kind": "ppo", "separate_reporting": False},
    "hybrid": {"kind": "hybrid", "separate_reporting": True},
}


def test_full_circuit_reliability_outranks_speed_and_smoothness():
    rows = (
        _controller_rows("reliable", official_full=True, safety=0, three_gate_ok=True,
                         time_s=9.0, jerk=0.05)
        + _controller_rows("fast", official_full=False, safety=0, three_gate_ok=True,
                           time_s=4.0, jerk=0.01)
    )
    recommendation = fbr.recommend(fbr.build_aggregates(rows), _META)
    assert recommendation["best_ppo_checkpoint"] == "reliable"


def test_a_safety_event_outranks_everything_below_full_circuit_reliability():
    rows = (
        _controller_rows("unsafe", official_full=True, safety=1, three_gate_ok=True,
                         time_s=4.0, jerk=0.01)
        + _controller_rows("reliable", official_full=True, safety=0, three_gate_ok=True,
                           time_s=9.0, jerk=0.05)
    )
    recommendation = fbr.recommend(fbr.build_aggregates(rows), _META)
    assert recommendation["best_ppo_checkpoint"] == "reliable"


def test_three_gate_reliability_outranks_completion_time():
    rows = (
        _controller_rows("reliable", official_full=True, safety=0, three_gate_ok=True,
                         time_s=9.0, jerk=0.05)
        + _controller_rows("fast", official_full=True, safety=0, three_gate_ok=False,
                           time_s=4.0, jerk=0.01)
    )
    recommendation = fbr.recommend(fbr.build_aggregates(rows), _META)
    assert recommendation["best_ppo_checkpoint"] == "reliable"


def test_time_breaks_a_tie_only_when_reliability_and_safety_match():
    rows = (
        _controller_rows("reliable", official_full=True, safety=0, three_gate_ok=True,
                         time_s=9.0, jerk=0.05)
        + _controller_rows("fast", official_full=True, safety=0, three_gate_ok=True,
                           time_s=4.0, jerk=0.01)
    )
    recommendation = fbr.recommend(fbr.build_aggregates(rows), _META)
    assert recommendation["best_ppo_checkpoint"] == "fast"


def test_hybrid_is_scored_but_excluded_from_the_recommendation():
    rows = (
        _controller_rows("reliable", official_full=True, safety=0, three_gate_ok=True,
                         time_s=9.0, jerk=0.05)
        + _controller_rows("hybrid", official_full=True, safety=0, three_gate_ok=True,
                           time_s=1.0, jerk=0.001)
    )
    recommendation = fbr.recommend(fbr.build_aggregates(rows), _META)
    assert recommendation["best_overall_policy_controller"] == "reliable"
    assert [e["controller"] for e in recommendation["excluded_from_selection"]] == ["hybrid"]
    assert "hybrid" not in [e["controller"] for e in recommendation["ranking"]]


def test_within_group_time_ranks_never_pool_across_groups():
    # "slow_short" wins the short group; "slow_long" wins the long one. Pooling raw
    # seconds would let the group with the larger scale decide; ranks must not.
    rows = []
    for controller, short, long_ in (("slow_short", 5.0, 400.0), ("slow_long", 6.0, 10.0)):
        for seed in range(2):
            rows.append(_episode(controller=controller, seed=seed, official_time_s=short,
                                 penalized_time_s=short))
            rows.append(_episode(
                controller=controller, seed=seed, group="three_gate_sequence",
                case_uid="three_gate_sequence/t", official_time_s=long_,
                penalized_time_s=long_, expected_gates=3, completed_gates=3,
            ))
    ranks = fbr.group_mean_ranks(fbr.build_aggregates(rows), "completion_time_s", timed_only=True)
    assert ranks["slow_short"] == pytest.approx(1.5)
    assert ranks["slow_long"] == pytest.approx(1.5)


def test_report_renders_and_names_the_selected_checkpoint(tmp_path):
    rows = (
        _controller_rows("reliable", official_full=True, safety=0, three_gate_ok=True,
                         time_s=9.0, jerk=0.05)
        + _controller_rows("fast", official_full=False, safety=0, three_gate_ok=True,
                           time_s=4.0, jerk=0.01)
    )
    (tmp_path / "episodes.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "suite_manifest.json").write_text(
        json.dumps(
            {
                "git_sha": "deadbeef",
                "adapter": "holoocean",
                "current_profile": "none",
                "dt": 0.1,
                "total_episodes": len(rows),
                "all_seeds": [0, 1],
                "reused_seeds": [0],
                "holdout_seeds": [1],
                "controllers": [
                    {"key": "reliable", "kind": "ppo", "policy_only": True,
                     "separate_reporting": False, "model": "a.zip", "note": ""},
                    {"key": "fast", "kind": "ppo", "policy_only": True,
                     "separate_reporting": False, "model": "b.zip", "note": ""},
                ],
            }
        ),
        encoding="utf-8",
    )
    report = fbr.build_report(tmp_path, plots=False)
    assert report["recommendation"]["best_ppo_checkpoint"] == "reliable"
    text = (tmp_path / "final_benchmark_report.md").read_text(encoding="utf-8")
    assert "Best PPO checkpoint on this benchmark: `reliable`" in text
    assert (tmp_path / "aggregate_by_group.csv").exists()
    assert (tmp_path / "paired_comparisons.csv").exists()
