"""Tests for the direction-aware training-only reward (pure score_step logic)."""

import numpy as np
import pytest

from marine_race_arena.learning.reward import RewardConfig, RewardState, score_step

ZERO = np.zeros(4, dtype=np.float32)


def _kwargs(**over):
    base = dict(
        has_target=True,
        signed_distance_plane=-10.0,  # 10 m before the plane, on the legal entry side
        lateral_offset=1.0,
        heading_alignment=0.0,
        gate_delta=0,
        d_collision=0,
        d_obstacle=0,
        d_out_of_bounds=0,
        d_stuck=0,
        d_missed=0,
        d_wrong_direction=0,
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
    for key in (
        "progress", "alignment", "heading_alignment", "gate_bonus", "completion_bonus",
        "time_cost", "collision_penalty", "obstacle_penalty", "out_of_bounds_penalty",
        "missed_gate_penalty", "wrong_direction_penalty", "stuck_penalty", "dnf_penalty",
        "timeout_penalty", "action_change_penalty", "action_magnitude_penalty",
    ):
        assert key in comps


def test_time_cost_is_negative_each_step():
    st, cfg = RewardState(), RewardConfig()
    _, comps = score_step(st, cfg, **_kwargs())
    assert comps["time_cost"] < 0.0


def test_legal_side_approach_gives_positive_progress():
    st, cfg = RewardState(), RewardConfig()
    score_step(st, cfg, **_kwargs(signed_distance_plane=-10.0))  # baseline approach=10
    _, comps = score_step(st, cfg, **_kwargs(signed_distance_plane=-8.0))
    assert comps["progress"] == pytest.approx(cfg.progress_scale * 2.0)


def test_moving_away_from_entry_side_gives_no_progress():
    st, cfg = RewardState(), RewardConfig()
    score_step(st, cfg, **_kwargs(signed_distance_plane=-5.0))
    _, comps = score_step(st, cfg, **_kwargs(signed_distance_plane=-9.0))
    assert comps["progress"] == 0.0


def test_wrong_side_approach_gives_no_progress():
    """Approaching the plane from the exit side (s>0) must not be rewarded."""
    st, cfg = RewardState(), RewardConfig()
    score_step(st, cfg, **_kwargs(signed_distance_plane=5.0))  # exit side
    _, comps = score_step(st, cfg, **_kwargs(signed_distance_plane=3.0))  # closer, still wrong side
    assert comps["progress"] == 0.0
    assert comps["alignment"] == 0.0
    assert comps["heading_alignment"] == 0.0


def test_no_progress_farming_by_oscillation():
    st, cfg = RewardState(), RewardConfig()
    total = 0.0
    for s in (-10.0, -5.0, -9.0, -5.0, -3.0):
        _, comps = score_step(st, cfg, **_kwargs(signed_distance_plane=s))
        total += comps["progress"]
    assert total == pytest.approx(cfg.progress_scale * 7.0)  # 10->5 and 5->3, not more


def test_gate_bonus_applied_once_per_delta():
    st, cfg = RewardState(), RewardConfig()
    _, c1 = score_step(st, cfg, **_kwargs(gate_delta=1))
    assert c1["gate_bonus"] == pytest.approx(cfg.gate_bonus)
    _, c2 = score_step(st, cfg, **_kwargs(gate_delta=0))
    assert c2["gate_bonus"] == 0.0


def test_gate_crossing_resets_ratchet():
    st, cfg = RewardState(), RewardConfig()
    score_step(st, cfg, **_kwargs(signed_distance_plane=-4.0))  # best approach = 4 for gate N
    _, c = score_step(st, cfg, **_kwargs(gate_delta=1, signed_distance_plane=-20.0))  # new gate, far entry side
    assert c["gate_bonus"] == pytest.approx(cfg.gate_bonus)
    assert c["progress"] == 0.0  # 20 is the new baseline
    _, c2 = score_step(st, cfg, **_kwargs(signed_distance_plane=-18.0))
    assert c2["progress"] == pytest.approx(cfg.progress_scale * 2.0)


def test_valid_crossing_bonus_vs_wrong_direction():
    # Valid crossing: referee increments valid gate count -> gate_delta>0 -> bonus.
    st, cfg = RewardState(), RewardConfig()
    _, c_valid = score_step(st, cfg, **_kwargs(gate_delta=1))
    assert c_valid["gate_bonus"] > 0.0
    assert c_valid["wrong_direction_penalty"] == 0.0
    # Wrong direction: no valid-count increment (gate_delta=0), a wrong-direction event.
    st2, cfg2 = RewardState(), RewardConfig()
    _, c_wrong = score_step(st2, cfg2, **_kwargs(gate_delta=0, d_wrong_direction=1))
    assert c_wrong["gate_bonus"] == 0.0
    assert c_wrong["wrong_direction_penalty"] == pytest.approx(-cfg2.wrong_direction_penalty)


def test_collision_obstacle_oob_missed_stuck_penalties():
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
    assert c2["completion_bonus"] == 0.0


def test_terminal_dnf_timeout_stuck_and_truncation():
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


def test_heading_alignment_signed():
    st, cfg = RewardState(), RewardConfig()
    _, fwd = score_step(st, cfg, **_kwargs(heading_alignment=1.0))
    assert fwd["heading_alignment"] == pytest.approx(cfg.heading_scale)
    st2, cfg2 = RewardState(), RewardConfig()
    _, bwd = score_step(st2, cfg2, **_kwargs(heading_alignment=-1.0))
    assert bwd["heading_alignment"] == pytest.approx(-cfg2.heading_scale)
    # wrong side: no heading reward even if moving toward the plane
    st3, cfg3 = RewardState(), RewardConfig()
    _, wrong = score_step(st3, cfg3, **_kwargs(signed_distance_plane=5.0, heading_alignment=1.0))
    assert wrong["heading_alignment"] == 0.0


def test_no_target_gives_no_progress():
    st, cfg = RewardState(), RewardConfig()
    _, comps = score_step(st, cfg, **_kwargs(has_target=False, signed_distance_plane=float("inf")))
    assert comps["progress"] == 0.0 and comps["alignment"] == 0.0 and comps["heading_alignment"] == 0.0


def test_alignment_rewards_reduced_lateral_offset_on_entry_side():
    st, cfg = RewardState(), RewardConfig()
    score_step(st, cfg, **_kwargs(lateral_offset=2.0))
    _, comps = score_step(st, cfg, **_kwargs(lateral_offset=1.0))
    assert comps["alignment"] == pytest.approx(cfg.alignment_scale * 1.0)


def test_determinism_same_inputs_same_output():
    def run():
        st, cfg = RewardState(), RewardConfig()
        out = []
        for s in (-10.0, -8.0, -6.0):
            r, _ = score_step(st, cfg, **_kwargs(signed_distance_plane=s, action=np.array([0.3, 0, 0, 0], dtype=np.float32)))
            out.append(r)
        return out

    assert run() == run()
