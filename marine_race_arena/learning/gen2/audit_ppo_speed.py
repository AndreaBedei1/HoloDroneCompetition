"""Offline PPO exploration/action/reward audit; no simulator or training."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from marine_race_arena.learning.config import ACTION_AXES
from marine_race_arena.learning.gen2.ppo_reliability import (
    COMPLETION_BONUS, GATE_REWARD, TIME_PENALTY, COLLISION_PENALTY,
    OUT_OF_BOUNDS_PENALTY, MISSED_GATE_PENALTY, WRONG_DIRECTION_PENALTY,
)

def audit(paths: dict[str, str]) -> dict:
    rng = np.random.default_rng(20260908)
    obs = torch.as_tensor(rng.normal(size=(256, 1, 27)).astype("float32"))
    out = {"observation_contract": "onboard_local_transition_27d_v1", "action_axes": list(ACTION_AXES), "checkpoints": {}}
    for label, value in paths.items():
        path = Path(value)
        model = RecurrentPPO.load(str(path), device="cpu")
        policy = model.policy
        log_std = policy.log_std.detach().cpu().numpy()
        # SB3's entropy is that of the unsquashed diagonal Gaussian.  The
        # sampled action below is the post-distribution policy output after
        # SB3's action-space clipping.
        state_shape = policy.lstm_hidden_state_shape
        state = (
            torch.zeros(state_shape),
            torch.zeros(state_shape),
        )
        episode_starts = torch.ones((obs.shape[0], obs.shape[1]), dtype=torch.float32)
        distribution, _ = policy.get_distribution(obs, state, episode_starts)
        entropy = distribution.distribution.entropy().detach().cpu().numpy()
        flat_obs = obs[:, 0, :].numpy()
        deterministic, _ = model.predict(flat_obs, deterministic=True)
        sampled, _ = model.predict(flat_obs, deterministic=False)
        deterministic = np.asarray(deterministic)
        sampled = np.asarray(sampled)
        out["checkpoints"][label] = {
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "log_std": log_std.tolist(), "std": np.exp(log_std).tolist(),
            "unsquashed_gaussian_entropy_mean": float(entropy.mean()),
            "deterministic_action_mean": deterministic.mean(axis=0).tolist(),
            "deterministic_action_p95_abs": np.quantile(np.abs(deterministic), .95, axis=0).tolist(),
            "sampled_action_mean": sampled.mean(axis=0).tolist(),
            "sampled_action_std": sampled.std(axis=0).tolist(),
            "sampled_action_clip_rate": float(np.mean(np.abs(sampled) >= .995)),
        }
    out["reward_constants"] = {
        "gate_crossing_bonus": GATE_REWARD, "completion_bonus": COMPLETION_BONUS,
        "time_penalty_per_step": TIME_PENALTY, "collision_penalty": COLLISION_PENALTY,
        "out_of_bounds_penalty": OUT_OF_BOUNDS_PENALTY,
        "missed_gate_penalty": MISSED_GATE_PENALTY,
        "wrong_direction_penalty": WRONG_DIRECTION_PENALTY,
    }
    return out

if __name__ == "__main__":
    paths = {
        "parent_warmstart_27d": "artifacts_gen2/bc_27d_clean_20260907/parent_warmstart_27d.zip",
        "candidate_25k": "artifacts_gen2/ppo_27d_campaign_A_20260908/candidate_25k.zip",
        "candidate_50k": "artifacts_gen2/ppo_27d_campaign_A_20260908/candidate_50k.zip",
    }
    print(json.dumps(audit(paths), indent=2))
