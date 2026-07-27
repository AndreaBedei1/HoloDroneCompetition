"""Parity-preserving gated residual policy for multi-gate behavioral warm start."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from marine_race_arena.learning.bc_train import BCPolicy, load_policy
from marine_race_arena.learning.bc_v3_transfer import expand_bc_v1_to_v3
from marine_race_arena.learning.config import ACTION_DIM
from marine_race_arena.learning.config_v3 import (
    FEATURE_NAMES_V3,
    OBS_DIM_V3,
    OBS_ENCODING_VERSION_V3,
)
from marine_race_arena.learning.dataset import BCDataset
from marine_race_arena.learning.model_contract_v3 import (
    FROZEN_BC_V1_SHA256,
    assert_clean_worktree,
)
from marine_race_arena.learning.provenance import git_sha, now_utc, sha256_file

_GATE_FEATURE = "previous_gate_bearing_present"
_GATE_INDEX = FEATURE_NAMES_V3.index(_GATE_FEATURE)


class GatedBCV3Policy:
    """Frozen v1 behavior plus a learned residual after a real beacon switch."""

    def __new__(
        cls,
        baseline: BCPolicy,
        residual_hidden_sizes=(128, 128),
        residual_mean=None,
        residual_std=None,
    ):
        import torch
        import torch.nn as nn

        class _Impl(nn.Module):
            def __init__(self):
                super().__init__()
                self.obs_dim = OBS_DIM_V3
                self.act_dim = ACTION_DIM
                self.residual_hidden_sizes = tuple(
                    int(value) for value in residual_hidden_sizes
                )
                self.baseline = baseline
                for parameter in self.baseline.parameters():
                    parameter.requires_grad_(False)
                layers = []
                previous = OBS_DIM_V3
                for width in self.residual_hidden_sizes:
                    layers.extend((nn.Linear(previous, width), nn.Tanh()))
                    previous = width
                self.residual_extractor = nn.Sequential(*layers)
                self.residual_head = nn.Linear(previous, ACTION_DIM)
                nn.init.zeros_(self.residual_head.weight)
                nn.init.zeros_(self.residual_head.bias)
                mean = (
                    np.zeros(OBS_DIM_V3, dtype=np.float32)
                    if residual_mean is None
                    else np.asarray(residual_mean, dtype=np.float32)
                )
                std = (
                    np.ones(OBS_DIM_V3, dtype=np.float32)
                    if residual_std is None
                    else np.asarray(residual_std, dtype=np.float32)
                )
                self.register_buffer(
                    "residual_mean",
                    torch.as_tensor(mean, dtype=torch.float32),
                )
                self.register_buffer(
                    "residual_std",
                    torch.as_tensor(std, dtype=torch.float32),
                )

            def forward(self, observation):
                normalized = (
                    observation - self.residual_mean
                ) / self.residual_std
                residual = self.residual_head(
                    self.residual_extractor(normalized)
                )
                gate = observation[:, _GATE_INDEX : _GATE_INDEX + 1].clamp(
                    0.0, 1.0
                )
                return self.baseline.forward(observation) + gate * residual

            @torch.no_grad()
            def act(self, observation):
                self.eval()
                tensor = torch.as_tensor(
                    np.asarray(observation, dtype=np.float32)
                ).reshape(1, -1)
                action = self.forward(tensor).reshape(-1).cpu().numpy()
                return np.clip(action, -1.0, 1.0).astype(np.float32)

        return _Impl()


@dataclass
class GatedBCV3Config:
    learning_rate: float = 3e-4
    batch_size: int = 128
    max_epochs: int = 300
    patience: int = 40
    val_fraction: float = 0.2
    weight_decay: float = 1e-6
    seed: int = 23001


def build_gated_policy(
    source_v1: BCPolicy,
    residual_mean=None,
    residual_std=None,
    residual_hidden_sizes=(128, 128),
):
    return GatedBCV3Policy(
        expand_bc_v1_to_v3(source_v1),
        residual_hidden_sizes=residual_hidden_sizes,
        residual_mean=residual_mean,
        residual_std=residual_std,
    )


def train_gated_bc_v3(
    dataset: BCDataset,
    source_v1: BCPolicy,
    config: Optional[GatedBCV3Config] = None,
) -> Tuple[object, List[Dict[str, float]]]:
    import torch

    config = config or GatedBCV3Config()
    if dataset.observation_encoding_version != OBS_ENCODING_VERSION_V3:
        raise ValueError("gated BC-v3 requires a v3 dataset")
    dataset.check_integrity()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    train_set, val_set = dataset.train_val_split(
        config.val_fraction, seed=config.seed
    )
    mean, std = train_set.normalization_stats()
    # The gate is consumed raw in forward(); its normalization only affects
    # residual features and remains safe with the standard floor.
    policy = build_gated_policy(
        source_v1, residual_mean=mean, residual_std=std
    )
    trainable = [
        parameter
        for name, parameter in policy.named_parameters()
        if name.startswith("residual_")
    ]
    optimizer = torch.optim.Adam(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    x_train = torch.as_tensor(train_set.observations, dtype=torch.float32)
    y_train = torch.as_tensor(train_set.actions, dtype=torch.float32)
    x_val = torch.as_tensor(val_set.observations, dtype=torch.float32)
    y_val = torch.as_tensor(val_set.actions, dtype=torch.float32)
    best_loss = float("inf")
    best_state = copy.deepcopy(policy.state_dict())
    stale = 0
    history: List[Dict[str, float]] = []
    for epoch in range(config.max_epochs):
        policy.train()
        order = torch.randperm(x_train.shape[0])
        for start in range(0, x_train.shape[0], config.batch_size):
            indices = order[start : start + config.batch_size]
            optimizer.zero_grad()
            loss = (
                (policy(x_train[indices]) - y_train[indices]) ** 2
            ).mean()
            loss.backward()
            optimizer.step()
        policy.eval()
        with torch.no_grad():
            train_mse = float(
                ((policy(x_train) - y_train) ** 2).mean().item()
            )
            prediction = policy(x_val)
            val_mse = float(((prediction - y_val) ** 2).mean().item())
            per_axis = (
                ((prediction - y_val) ** 2).mean(dim=0).cpu().numpy()
            )
        row = {
            "epoch": int(epoch),
            "train_mse": train_mse,
            "val_mse": val_mse,
            "val_mse_surge": float(per_axis[0]),
            "val_mse_sway": float(per_axis[1]),
            "val_mse_heave": float(per_axis[2]),
            "val_mse_yaw": float(per_axis[3]),
        }
        history.append(row)
        if val_mse < best_loss - 1e-7:
            best_loss = val_mse
            best_state = copy.deepcopy(policy.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    policy.load_state_dict(best_state)
    policy.eval()
    return policy, history


def save_gated_policy(policy, path) -> None:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "bc_v3_gated",
            "obs_dim": OBS_DIM_V3,
            "act_dim": ACTION_DIM,
            "obs_encoding_version": OBS_ENCODING_VERSION_V3,
            "feature_names": list(FEATURE_NAMES_V3),
            "residual_hidden_sizes": list(policy.residual_hidden_sizes),
            "baseline_hidden_sizes": list(policy.baseline.hidden_sizes),
            "state_dict": policy.state_dict(),
        },
        destination,
    )


def load_gated_policy(path):
    import torch

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "bc_v3_gated":
        raise ValueError("checkpoint is not a gated BC-v3 policy")
    if checkpoint.get("obs_encoding_version") != OBS_ENCODING_VERSION_V3:
        raise ValueError("gated policy observation version mismatch")
    baseline = BCPolicy(
        obs_dim=OBS_DIM_V3,
        act_dim=ACTION_DIM,
        hidden_sizes=checkpoint["baseline_hidden_sizes"],
    )
    policy = GatedBCV3Policy(
        baseline,
        residual_hidden_sizes=checkpoint["residual_hidden_sizes"],
    )
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy


def _write_history(path: Path, history: List[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--source-v1",
        default="results/rl_public/stage1/bc/model/best_model.pt",
    )
    parser.add_argument(
        "--out",
        default="results/rl/multigate_v3/models/bc_v3_gated.pt",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args(argv)

    assert_clean_worktree()
    source_sha = sha256_file(args.source_v1)
    if source_sha != FROZEN_BC_V1_SHA256:
        raise ValueError("frozen BC-v1 hash mismatch")
    source_dataset = BCDataset.load(args.dataset)
    finished = {
        episode.group_id
        for episode in source_dataset.episodes
        if episode.final_status == "FINISHED"
    }
    if not finished:
        raise ValueError("dataset has no FINISHED trajectories")
    dataset = source_dataset._subset(finished)
    config = GatedBCV3Config(
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        patience=args.patience,
    )
    policy, history = train_gated_bc_v3(
        dataset, load_policy(args.source_v1), config
    )
    output = Path(args.out)
    save_gated_policy(policy, output)
    _write_history(output.with_suffix(".training.csv"), history)
    report = {
        "schema_version": "bc_multigate_v3_gated_v1",
        "created_utc": now_utc(),
        "code_sha": git_sha(),
        "dataset_path": args.dataset,
        "dataset_sha256": sha256_file(args.dataset),
        "source_episodes": source_dataset.num_episodes,
        "finished_episodes": dataset.num_episodes,
        "steps": len(dataset),
        "source_v1_sha256": source_sha,
        "config": asdict(config),
        "epochs_completed": len(history),
        "best_val_mse": min(row["val_mse"] for row in history),
        "model_path": str(output),
        "model_sha256": sha256_file(output),
        "observation_encoding_version": OBS_ENCODING_VERSION_V3,
        "runtime_rule_actions": False,
        "policy_gate_feature": _GATE_FEATURE,
    }
    output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
