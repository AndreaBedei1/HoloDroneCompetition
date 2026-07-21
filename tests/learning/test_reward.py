"""Tests for the training-only reward (pure score_step logic)."""

import numpy as np
import pytest

from marine_race_arena.learning.reward import RewardConfig, RewardState, TrainingReward, score_step

ZERO = np.zeros(4, dtype=np.float32)


def _kwargs(**over):
    base = dict(
        has_target=True,
        dist_to_gate=10.0,
        lateral_offset=1.0,
        gate_delta=0,
        d_collision=0,
        d_obstacle=0,
        d_out_of_bounds=0,
        d_stuck=0,
        d_missed=0,
        terminated=False,
        truncated=False,
        terminal_status=None,
        action=ZERO,
    )
    base.update(over)
    return base


def test_components_sum_to_reward_and_all_present():
    st, cfg = RewardState(), RewardConfig()
    reward, comps = score_step(st, cfg, **_kwargs())
    assert reward == pytest.approx(sum(comps.values()))
    # Every documented component is logged, even when zero.
    for key in (
        "progress", "alignment", "gate_bonus", "completion_bonus", "time_cost",
        "collision_penalty", "obstacle_penalty", "out_of_bounds_penalty",
        "missed_gate_penalty", "stuck_penalty", "dnf_penalty", "timeout_penalty",
        "action_change_penalty", "action_magnitude_penalty",
    ):
        assert key in comps


def test_time_cost_is_negative_each_step():
    st, cfg = RewardState(), RewardConfig()
    _, comps = score_step(st, cfg, **_kwargs())
    assert comps["time_cost"] < 0.0


def test_progress_positive_when_approaching():
    st, cfg = RewardState(), RewardConfig()
    score_step(st, cfg, **_kwargs(dist_to_gate=10.0))  # sets baseline best=10
    _, comps = score_step(st, cfg, **_kwargs(dist_to_gate=8.0))
    assert comps["progress"] == pytest.approx(cfg.progress_scale * 2.0)


def test_no_progress_farming_by_oscillation():
    st, cfg = RewardState(), RewardConfig()
    total = 0.0
    # approach to 5, retreat to 9, return to 5, then genuine new best 3
    for d in (10.0, 5.0, 9.0, 5.0, 3.0):
        _, comps = score_step(st, cfg, **_kwargs(dist_to_gate=d))
        total += comps["progress"]
    # Total progress reward == initial approach (10->5) + new best (5->3) = 7, not more.
    assert total == pytest.approx(cfg.progress_scale * 7.0)


def test_moving_away_yields_no_progress():
    st, cfg = RewardState(), RewardConfig()
    score_step(st, cfg, **_kwargs(dist_to_gate=5.0))
    _, comps = score_step(st, cfg, **_kwargs(dist_to_gate=9.0))
    assert comps["progress"] == 0.0


def test_gate_bonus_applied_once_per_delta():
    st, cfg = RewardState(), RewardConfig()
    _, c1 = score_step(st, cfg, **_kwargs(gate_delta=1))
    assert c1["gate_bonus"] == pytest.approx(cfg.gate_bonus)
    _, c2 = score_step(st, cfg, **_kwargs(gate_delta=0))
    assert c2["gate_bonus"] == 0.0


def test_gate_crossing_resets_ratchet():
    st, cfg = RewardState(), RewardConfig()
    score_step(st, cfg, **_kwargs(dist_to_gate=4.0))  # best=4 for gate N
    # cross gate: ratchet resets, so the far new gate distance becomes the new baseline
    _, c = score_step(st, cfg, **_kwargs(gate_delta=1, dist_to_gate=20.0))
    assert c["gate_bonus"] == pytest.approx(cfg.gate_bonus)
    assert c["progress"] == 0.0  # 20 is the new baseline, not "worse" than old best 4
    _, c2 = score_step(st, cfg, **_kwargs(dist_to_gate=18.0))
    assert c2["progress"] == pytest.approx(cfg.progress_scale * 2.0)


def test_collision_and_obstacle_and_oob_penalties():
    st, cfg = RewardState(), RewardConfig()
    _, comps = score_step(st, cfg, **_kwargs(d_collision=2, d_obstacle=1, d_out_of_bounds=1, d_missed=1, d_stuck=1))
    assert comps["collision_penalty"] == pytest.approx(-2 * cfg.collision_penalty)
    assert comps["obstacle_penalty"] == pytest.approx(-cfg.obstacle_penalty)
    assert comps["out_of_bounds_penalty"] == pytest.approx(-cfg.out_of_bounds_penalty)
    assert comps["missed_gate_penalty"] == pytest.approx(-cfg.missed_gate_penalty)
    assert comps["stuck_penalty"] == pytest.approx(-cfg.stuck_penalty)


def test_completion_bonus_once():
    st, cfg = RewardState(), RewardConfig()
    _, c1 = score_step(st, cfg, **_kwargs(terminated=True, terminal_status="FINISHED"))
    assert c1["completion_bonus"] == pytest.approx(cfg.completion_bonus)
    _, c2 = score_step(st, cfg, **_kwargs(terminated=True, terminal_status="FINISHED"))
    assert c2["completion_bonus"] == 0.0  # terminal latch


def test_terminal_dnf_and_timeout_and_truncation():
    for status, key in (("DNF", "dnf_penalty"), ("TIMEOUT", "timeout_penalty"), ("STUCK", "stuck_terminal_penalty")):
        st, cfg = RewardState(), RewardConfig()
        _, comps = score_step(st, cfg, **_kwargs(terminated=True, terminal_status=status))
        assert comps[key] < 0.0
    st, cfg = RewardState(), RewardConfig()
    _, comps = score_step(st, cfg, **_kwargs(truncated=True))
    assert comps["timeout_penalty"] == pytest.approx(-cfg.timeout_penalty)


def test_action_change_penalty():
    st, cfg = RewardState(), RewardConfig()
    a1 = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    score_step(st, cfg, **_kwargs(action=ZERO))
    _, comps = score_step(st, cfg, **_kwargs(action=a1))
    assert comps["action_change_penalty"] == pytest.approx(-cfg.action_change_penalty * 0.5)


def test_no_target_gives_no_progress():
    st, cfg = RewardState(), RewardConfig()
    _, comps = score_step(st, cfg, **_kwargs(has_target=False, dist_to_gate=float("inf")))
    assert comps["progress"] == 0.0 and comps["alignment"] == 0.0


def test_determinism_same_inputs_same_output():
    def run():
        st, cfg = RewardState(), RewardConfig()
        out = []
        for d in (10.0, 8.0, 6.0):
            r, _ = score_step(st, cfg, **_kwargs(dist_to_gate=d, action=np.array([0.3, 0, 0, 0], dtype=np.float32)))
            out.append(r)
        return out

    assert run() == run()


def test_alignment_rewards_reduced_lateral_offset():
    st, cfg = RewardState(), RewardConfig()
    score_step(st, cfg, **_kwargs(lateral_offset=2.0))
    _, comps = score_step(st, cfg, **_kwargs(lateral_offset=1.0))
    assert comps["alignment"] == pytest.approx(cfg.alignment_scale * 1.0)
