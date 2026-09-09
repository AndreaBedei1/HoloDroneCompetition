"""Equivalence: the step-wise RaceEpisode matches the real runner loop.

Both paths use the identical construction (``build_single_vehicle_race``) with the
same seed/track/fallback adapter. Driving the same deterministic action sequence
through the real ``_run_race_loop`` (via a scripted controller) and through
``RaceEpisode`` must yield the same official observations and the same referee
progress. This guards the step-wise env against silently diverging from the
benchmark it is meant to reuse.
"""

import math

import numpy as np
import pytest

from marine_race_arena.learning.config import LearningContext
from marine_race_arena.learning.observation_encoder import encode_observation
from marine_race_arena.learning.episode import RaceEpisode, build_single_vehicle_race
from marine_race_arena.participants.controller_interface import BaseController
from marine_race_arena.scripts.run_marine_race import _run_race_loop

TRACK = "marine_race_arena/tracks/tests/single_gate_yaw_0.json"
DT = 0.1
SEED = 3


def _action_sequence(n):
    """A deterministic, varied, in-bounds action sequence."""
    seq = []
    for i in range(n):
        seq.append(
            {
                "surge": 0.6,
                "sway": 0.2 * math.sin(0.3 * i),
                "heave": 0.1 * math.cos(0.2 * i),
                "yaw": 0.15 * math.sin(0.1 * i),
            }
        )
    return seq


class _ScriptedController(BaseController):
    """Returns a fixed action per step and logs the observations it sees."""

    def __init__(self, actions):
        self._actions = actions
        self._i = 0
        self.obs_log = []

    def reset(self, mission_info):
        self._i = 0
        self.obs_log = []

    def step(self, observation):
        self.obs_log.append(observation)
        action = self._actions[min(self._i, len(self._actions) - 1)]
        self._i += 1
        return dict(action)

    def close(self):
        pass


def _encode(obs):
    return encode_observation(obs, LearningContext())


def test_episode_matches_runner_observations_and_progress():
    actions = _action_sequence(40)

    # --- Run A: the real runner loop with a scripted controller ---------------
    scripted = _ScriptedController(actions)
    ctx_a = build_single_vehicle_race(TRACK, seed=SEED, adapter="fallback", allow_fallback=True, duration_s=4.0, controller=scripted)
    pid = ctx_a.participant.id
    _run_race_loop(
        config=ctx_a.config,
        arena=ctx_a.arena,
        referee=ctx_a.referee,
        adapter=ctx_a.adapter,
        participants={pid: ctx_a.participant},
        dt=DT,
    )
    obs_log_a = scripted.obs_log
    gates_a = ctx_a.referee.states[pid].valid_gate_crossings
    status_a = ctx_a.referee.states[pid].status
    ctx_a.adapter.close()

    assert len(obs_log_a) >= 5, "runner should have produced several active ticks"

    # --- Run B: the step-wise episode with the same actions -------------------
    ep = RaceEpisode(TRACK, seed=SEED, dt=DT, adapter="fallback", allow_fallback=True, duration_s=4.0)
    obs0 = ep.reset()
    obs_log_b = [obs0]
    for i in range(len(obs_log_a) - 1):
        step = ep.step(actions[i])
        obs_log_b.append(step.observation)
        if step.terminated:
            break

    # Observation-sequence equivalence over the shared horizon.
    horizon = min(len(obs_log_a), len(obs_log_b))
    assert horizon >= 5
    for i in range(horizon):
        va = _encode(obs_log_a[i])
        vb = _encode(obs_log_b[i])
        assert np.allclose(va, vb, atol=1e-5), f"observation {i} diverged"
        # local_time also matches tick-for-tick
        assert obs_log_a[i]["local_time_s"] == pytest.approx(obs_log_b[i]["local_time_s"], abs=1e-6)

    gates_b = ep.referee_progress()["valid_gate_crossings"]
    assert gates_b == gates_a, "referee gate progress diverged"
    # If the runner finished, the episode reaches a terminal state too.
    if str(getattr(status_a, "value", status_a)) == "FINISHED":
        assert ep.referee_progress()["is_terminal"]
    ep.close()


def test_reset_is_deterministic_for_fixed_seed():
    a = RaceEpisode(TRACK, seed=7, dt=DT)
    b = RaceEpisode(TRACK, seed=7, dt=DT)
    obs_a = a.reset()
    obs_b = b.reset()
    assert np.allclose(_encode(obs_a), _encode(obs_b))
    # A few identical steps stay identical.
    for _ in range(10):
        sa = a.step({"surge": 0.5, "sway": 0.0, "heave": 0.0, "yaw": 0.1})
        sb = b.step({"surge": 0.5, "sway": 0.0, "heave": 0.0, "yaw": 0.1})
        assert np.allclose(_encode(sa.observation), _encode(sb.observation))
        assert sa.terminated == sb.terminated and sa.truncated == sb.truncated
    a.close()
    b.close()


def test_reset_reuses_and_reconstructs_cleanly():
    """Repeated reset()/close() on one episode object stays consistent."""
    ep = RaceEpisode(TRACK, seed=5, dt=DT, max_steps=8)
    first = _encode(ep.reset())
    for _ in range(6):
        ep.step({"surge": 0.5, "sway": 0.0, "heave": 0.0, "yaw": 0.0})
    again = _encode(ep.reset())  # fresh construction, same seed -> identical start
    assert np.allclose(first, again)
    # Truncation at max_steps is reported and the episode still closes cleanly.
    ep.reset()
    last = None
    for _ in range(8):
        last = ep.step({"surge": 0.5, "sway": 0.0, "heave": 0.0, "yaw": 0.0})
    assert last is not None and (last.truncated or last.terminated)
    ep.close()
    ep.close()  # idempotent close is safe
