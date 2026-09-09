"""Collision shaping: charge each impact, never accumulate contact frames."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from marine_race_arena.learning.reward_local_transition import (
    LocalTransitionRewardConfig,
    LocalTransitionTrainingReward,
)


class _Referee:
    def __init__(self):
        self.valid_gate_crossings = 0
        self.collision_events = 0
        self.obstacle_collision_events = 0
        self.out_of_bounds_events = 0
        self.wrong_direction_crossings = 0
        self.missed_gate_attempts = 0
        self.status = SimpleNamespace(value="RUNNING")


class _Tracker:
    expected_beacon_id = "B01"


class _Env:
    """Minimal environment stub exposing only what the reward reads."""

    def __init__(self, gate_count: int = 2):
        self.tracker = _Tracker()
        referee = _Referee()
        gates = [
            SimpleNamespace(position=(float(index) * 5.0, 0.0, -4.0),
                            passage_direction=(1.0, 0.0, 0.0))
            for index in range(gate_count)
        ]
        self.episode = SimpleNamespace(
            participant_id="P1",
            context=SimpleNamespace(
                referee=SimpleNamespace(states={"P1": referee}),
                config=SimpleNamespace(gates=gates),
            ),
        )

    @property
    def referee(self):
        return self.episode.context.referee.states["P1"]


def _step(*, collision=False, terminated=False, truncated=False, position=(0.0, 0.0, -4.0)):
    return SimpleNamespace(
        observation={"sensors": {}, "beacons": []},
        terminated=terminated,
        truncated=truncated,
        collision=bool(collision),
        obstacle_collisions=0,
        current_state=SimpleNamespace(position=position),
    )


def _run(reward, contacts, *, action=None):
    """Apply a contact pattern and return per-step (reward, components)."""

    env = _Env()
    action = np.zeros(4, dtype=np.float32) if action is None else action
    rows = []
    for contact in contacts:
        rows.append(reward(env, _step(collision=contact), 0, action))
    return rows


def _collision_total(rows):
    return sum(
        row[1]["collision_penalty"] + row[1]["collision_contact_penalty"]
        for row in rows
    )


def test_one_entry_followed_by_prolonged_contact_is_charged_once():
    reward = LocalTransitionTrainingReward()
    cfg = reward.config
    rows = _run(reward, [True] * 300)
    entry_charges = [row[1]["collision_penalty"] for row in rows]
    assert entry_charges[0] == pytest.approx(-cfg.collision_penalty)
    assert all(value == 0.0 for value in entry_charges[1:])
    assert reward.collision_entries == 1
    assert reward.collision_contact_frames == 300
    assert reward.collision_episode is True


def test_prolonged_contact_penalty_is_small_and_bounded():
    reward = LocalTransitionTrainingReward()
    cfg = reward.config
    rows = _run(reward, [True] * 500)
    contact_total = sum(row[1]["collision_contact_penalty"] for row in rows)
    assert contact_total == pytest.approx(-cfg.collision_contact_penalty_episode_cap)
    # Sustained contact can never dominate the impact that caused it.
    assert abs(contact_total) < cfg.collision_penalty


def test_separation_then_a_second_impact_is_charged_again():
    reward = LocalTransitionTrainingReward()
    cfg = reward.config
    pattern = [True] * 5 + [False] * 10 + [True] * 5
    rows = _run(reward, pattern)
    entries = [index for index, row in enumerate(rows) if row[1]["collision_penalty"]]
    assert len(entries) == 2
    assert reward.collision_entries == 2
    assert reward.collision_contact_frames == 10
    assert sum(row[1]["collision_penalty"] for row in rows) == pytest.approx(
        -2 * cfg.collision_penalty
    )


def test_a_single_clear_frame_does_not_reset_an_ongoing_contact():
    reward = LocalTransitionTrainingReward()
    # One dropped contact frame inside a sustained collision is sensor noise,
    # not a separation followed by a new impact.
    rows = _run(reward, [True, True, False, True, True])
    assert reward.collision_entries == 1
    assert sum(1 for row in rows if row[1]["collision_penalty"]) == 1


def test_cumulative_collision_shaping_is_bounded_per_episode():
    reward = LocalTransitionTrainingReward()
    cfg = reward.config
    pattern = ([True] * 4 + [False] * 5) * 40
    rows = _run(reward, pattern)
    total = _collision_total(rows)
    bound = (
        cfg.collision_entry_penalty_episode_cap
        + cfg.collision_contact_penalty_episode_cap
    )
    assert reward.collision_entries == 40
    assert abs(total) <= bound + 1e-9
    assert abs(total) == pytest.approx(bound)


def test_collision_penalty_dominates_efficiency_rewards():
    reward = LocalTransitionTrainingReward(
        LocalTransitionRewardConfig(efficiency_unlocked=True)
    )
    cfg = reward.config
    action = np.full(4, 0.9, dtype=np.float32)
    rows = _run(reward, [False] * 200, action=action)
    efficiency_magnitude = sum(
        abs(row[1]["action_change_penalty"])
        + abs(row[1]["jerk_penalty"])
        + abs(row[1]["energy_penalty"])
        + abs(row[1]["time_cost"])
        for row in rows
    )
    assert cfg.collision_penalty > efficiency_magnitude
    # A slow but valid correction must stay far cheaper than one collision.
    assert efficiency_magnitude < 0.1 * cfg.collision_penalty


def test_collision_accounting_does_not_depend_on_sequence_length():
    totals = []
    for gate_count in (2, 12, 22):
        reward = LocalTransitionTrainingReward()
        env = _Env(gate_count=gate_count)
        action = np.zeros(4, dtype=np.float32)
        rows = [
            reward(env, _step(collision=contact), 0, action)
            for contact in ([True] * 3 + [False] * 6) * 30
        ]
        totals.append((_collision_total(rows), reward.collision_entries))
    assert len(set(totals)) == 1


def test_counters_are_reported_separately_and_reset_per_episode():
    reward = LocalTransitionTrainingReward()
    _run(reward, [True] * 4 + [False] * 6 + [True] * 2)
    counters = reward.collision_counters()
    assert counters["collision_episode"] is True
    assert counters["collision_entry"] == 2
    assert counters["collision_contact_frames"] == 6
    reward.reset(_Env())
    assert reward.collision_counters() == {
        "collision_episode": False,
        "collision_entry": 0,
        "collision_contact_frames": 0,
        "collision_entry_penalty_paid": 0.0,
        "collision_contact_penalty_paid": 0.0,
    }


def test_collision_free_episode_is_never_charged():
    reward = LocalTransitionTrainingReward()
    rows = _run(reward, [False] * 50)
    assert _collision_total(rows) == pytest.approx(0.0)
    assert reward.collision_entries == 0
    assert reward.collision_episode is False
