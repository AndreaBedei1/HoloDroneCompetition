"""Previous-action temporal alignment across gym env, controller and recorder.

Convention: an encoded observation carries, in its ``prev_*`` features, the action
that was *actually applied on the step that produced it*. Equivalently, o_(t+1)
carries a_t; the reset observation carries a zero previous action. The Gym env
(RL training) and the RLGateController (deployment) must agree exactly.
"""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("torch")

from marine_race_arena.learning.bc_train import BCPolicy, save_policy
from marine_race_arena.learning.config import ACTION_DIM, FEATURE_NAMES
from marine_race_arena.learning.episode import RaceEpisode
from marine_race_arena.learning.gym_env import MarineRaceGymEnv
from marine_race_arena.learning.rl_controller import RLGateController

TRACK = "marine_race_arena/tracks/training/stage1_single_gate.json"
SEED = 4
PREV_IDX = [FEATURE_NAMES.index(f"prev_{a}") for a in ("surge", "sway", "heave", "yaw")]


def _prev(encoded):
    return np.asarray(encoded)[PREV_IDX]


def _env(**kw):
    params = dict(seed=SEED, dt=0.1, adapter="fallback", allow_fallback=True, max_steps=25)
    params.update(kw)
    return MarineRaceGymEnv(TRACK, **params)


def test_reset_observation_has_zero_previous_action():
    env = _env()
    try:
        obs, _ = env.reset(seed=SEED)
        assert np.allclose(_prev(obs), 0.0)
    finally:
        env.close()


def test_next_observation_contains_the_applied_action():
    env = _env()
    try:
        env.reset(seed=SEED)
        a = np.array([0.7, -0.3, 0.1, 0.2], dtype=np.float32)
        obs, *_ = env.step(a)
        assert np.allclose(_prev(obs), a, atol=1e-6), "o_(t+1) must contain a_t"
    finally:
        env.close()


def test_two_consecutive_actions_have_no_one_step_delay():
    env = _env()
    try:
        env.reset(seed=SEED)
        a0 = np.array([0.5, 0.0, 0.0, 0.1], dtype=np.float32)
        a1 = np.array([-0.4, 0.2, -0.1, 0.3], dtype=np.float32)
        o1, *_ = env.step(a0)
        o2, *_ = env.step(a1)
        assert np.allclose(_prev(o1), a0, atol=1e-6)
        assert np.allclose(_prev(o2), a1, atol=1e-6), "one-step delay detected"
    finally:
        env.close()


class _ScriptedInference:
    """Returns a fixed action sequence, ignoring the observation."""

    kind = "bc"

    def __init__(self, actions):
        self._actions = [np.asarray(a, dtype=np.float32) for a in actions]
        self._i = 0

    def act(self, observation):
        action = self._actions[min(self._i, len(self._actions) - 1)]
        self._i += 1
        return action


def test_gym_and_controller_use_the_same_temporal_convention(tmp_path):
    """Full encoded-vector parity between Gym training and controller inference."""
    model = tmp_path / "bc.pt"
    save_policy(BCPolicy(hidden_sizes=(16, 16)), model)
    actions = [
        np.array([0.6, 0.1, 0.0, 0.2], dtype=np.float32),
        np.array([-0.3, 0.2, 0.1, -0.1], dtype=np.float32),
        np.array([0.4, -0.2, -0.1, 0.3], dtype=np.float32),
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ]

    # Gym path: reset gives e0, then apply the first three actions.
    env = _env()
    gym_encs = [env.reset(seed=SEED)[0]]
    for a in actions[:3]:
        e, *_ = env.step(a)
        gym_encs.append(e)
    env.close()

    # Controller path over the identical simulation, with scripted (matching) actions.
    episode = RaceEpisode(TRACK, seed=SEED, dt=0.1, adapter="fallback", allow_fallback=True, max_steps=25)
    obs = episode.reset()
    ctrl = RLGateController(model_path=str(model))
    ctrl.reset({"participant_id": episode.participant_id, "initial_beacon_id": "B01",
                "total_beacons": len(episode.context.config.track.gate_sequence), "laps": episode.context.config.race.laps})
    ctrl._inference = _ScriptedInference(actions)
    ctrl_encs = []
    for _ in range(4):
        cmd = ctrl.step(obs)
        ctrl_encs.append(ctrl.last_encoded_observation)
        step = episode.step(cmd)
        obs = step.observation
    episode.close()

    for k in range(4):
        assert np.allclose(gym_encs[k], ctrl_encs[k], atol=1e-5), f"encoding {k} diverges (temporal convention)"
