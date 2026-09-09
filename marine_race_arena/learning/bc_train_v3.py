"""Fine-tune the transferred BC-v1 policy on v3 multi-gate demonstrations."""

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
from marine_race_arena.learning.bc_v3_transfer import (
    expand_bc_v1_to_v3,
    save_v3_policy,
)
from marine_race_arena.learning.config_v3 import (
    FEATURE_NAMES_V3,
    OBS_DIM_V3,
    OBS_ENCODING_VERSION_V3,
)
from marine_race_arena.learning.dataset import BCDataset
from marine_race_arena.learning.model_contract_v3 import (
    FROZEN_BC_V1_SHA256,
    assert_clean_worktree,
    validate_v3_model,
)
from marine_race_arena.learning.provenance import git_sha, now_utc, sha256_file

TRANSITION_FEATURE_NAMES = (
    "expected_beacon_changed",
    "steps_since_beacon_change_norm",
    "previous_gate_in_rear_sector",
    "previous_gate_bearing_present",
)


@dataclass
class BCV3Config:
    learning_rate: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 30
    val_fraction: float = 0.2
    weight_decay: float = 0.0
    anchor_weight: float = 1e-5
    transition_columns_only: bool = True
    seed: int = 23000


def rebase_observation_normalization(
    policy: BCPolicy,
    new_mean: np.ndarray,
    new_std: np.ndarray,
) -> None:
    """Change normalization buffers without changing the policy function."""
    import torch
    import torch.nn as nn

    new_mean_t = torch.as_tensor(
        np.asarray(new_mean, dtype=np.float32), dtype=torch.float32
    )
    new_std_t = torch.as_tensor(
        np.asarray(new_std, dtype=np.float32), dtype=torch.float32
    )
    if new_mean_t.shape != policy.obs_mean.shape or new_std_t.shape != policy.obs_std.shape:
        raise ValueError("normalization shape mismatch")
    new_std_t = torch.where(
        new_std_t.abs() < 1e-6, torch.ones_like(new_std_t), new_std_t
    )
    old_mean = policy.obs_mean.detach().clone()
    old_std = torch.where(
        policy.obs_std.detach().abs() < 1e-6,
        torch.ones_like(policy.obs_std.detach()),
        policy.obs_std.detach(),
    )
    first = next(
        module for module in policy.extractor if isinstance(module, nn.Linear)
    )
    with torch.no_grad():
        old_weight = first.weight.detach().clone()
        first.weight.copy_(
            old_weight * (new_std_t / old_std).reshape(1, -1)
        )
        first.bias.add_(old_weight @ ((new_mean_t - old_mean) / old_std))
        policy.obs_mean.copy_(new_mean_t)
        policy.obs_std.copy_(new_std_t)


def fine_tune_bc_v3(
    dataset: BCDataset,
    source_v1: BCPolicy,
    config: Optional[BCV3Config] = None,
) -> Tuple[BCPolicy, List[Dict[str, float]]]:
    """Fine-tune a transferred v3 policy with an anchor to the transfer."""
    import torch

    config = config or BCV3Config()
    if dataset.observation_encoding_version != OBS_ENCODING_VERSION_V3:
        raise ValueError(
            f"expected {OBS_ENCODING_VERSION_V3}, got "
            f"{dataset.observation_encoding_version}"
        )
    dataset.check_integrity()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    train_set, val_set = dataset.train_val_split(
        config.val_fraction, seed=config.seed
    )
    policy = expand_bc_v1_to_v3(source_v1)
    mean, std = train_set.normalization_stats()
    transition_indices = [
        FEATURE_NAMES_V3.index(name) for name in TRANSITION_FEATURE_NAMES
    ]
    if config.transition_columns_only:
        # Raw zeros must stay normalized zeros before the first beacon change;
        # otherwise learned transition weights would perturb the frozen v1 path.
        mean[transition_indices] = 0.0
        std[transition_indices] = 1.0
    rebase_observation_normalization(policy, mean, std)
    anchor = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
    }
    trainable_parameters = list(policy.parameters())
    restore_weight = None
    gradient_hook = None
    if config.transition_columns_only:
        import torch.nn as nn

        first = next(
            module
            for module in policy.extractor
            if isinstance(module, nn.Linear)
        )
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        first.weight.requires_grad_(True)
        mask = torch.zeros_like(first.weight)
        mask[:, transition_indices] = 1.0
        gradient_hook = first.weight.register_hook(lambda gradient: gradient * mask)
        restore_weight = (
            first,
            first.weight.detach().clone(),
            mask.detach().clone(),
        )
        trainable_parameters = [first.weight]

    x_train = torch.as_tensor(train_set.observations, dtype=torch.float32)
    y_train = torch.as_tensor(train_set.actions, dtype=torch.float32)
    x_val = torch.as_tensor(val_set.observations, dtype=torch.float32)
    y_val = torch.as_tensor(val_set.actions, dtype=torch.float32)
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

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
            prediction = policy(x_train[indices])
            imitation = ((prediction - y_train[indices]) ** 2).mean()
            anchor_loss = sum(
                ((parameter - anchor[name]) ** 2).mean()
                for name, parameter in policy.named_parameters()
            )
            loss = imitation + config.anchor_weight * anchor_loss
            loss.backward()
            optimizer.step()
            if restore_weight is not None:
                first, initial_weight, mask = restore_weight
                with torch.no_grad():
                    first.weight.copy_(
                        first.weight * mask + initial_weight * (1.0 - mask)
                    )

        policy.eval()
        with torch.no_grad():
            train_mse = float(
                ((policy(x_train) - y_train) ** 2).mean().item()
            )
            val_prediction = policy(x_val)
            val_mse = float(((val_prediction - y_val) ** 2).mean().item())
            per_axis = (
                ((val_prediction - y_val) ** 2)
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
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
    if gradient_hook is not None:
        gradient_hook.remove()
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    policy.eval()
    return policy, history


def _write_history(path: Path, history: List[Dict[str, float]]) -> None:
    if not history:
        return
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
        default="results/rl/multigate_v3/models/bc_v3_two_gate.pt",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor-weight", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=23000)
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include non-FINISHED episodes; default uses completed expert trajectories only.",
    )
    args = parser.parse_args(argv)

    assert_clean_worktree()
    source_sha = sha256_file(args.source_v1)
    if source_sha != FROZEN_BC_V1_SHA256:
        raise ValueError(
            f"frozen BC-v1 hash mismatch: {source_sha} != {FROZEN_BC_V1_SHA256}"
        )
    dataset = BCDataset.load(args.dataset)
    source_episode_count = dataset.num_episodes
    if not args.include_failed:
        finished_groups = {
            episode.group_id
            for episode in dataset.episodes
            if episode.final_status == "FINISHED"
        }
        if not finished_groups:
            raise ValueError("dataset has no FINISHED episodes")
        dataset = dataset._subset(finished_groups)
    config = BCV3Config(
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        patience=args.patience,
        anchor_weight=args.anchor_weight,
        seed=args.seed,
    )
    policy, history = fine_tune_bc_v3(
        dataset, load_policy(args.source_v1), config
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_v3_policy(policy, out)
    history_path = out.with_suffix(".training.csv")
    _write_history(history_path, history)
    model = validate_v3_model(out)
    report = {
        "schema_version": "bc_multigate_v3_v1",
        "created_utc": now_utc(),
        "code_sha": git_sha(),
        "dataset_path": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "dataset_observation_encoding_version": (
            dataset.observation_encoding_version
        ),
        "episodes": dataset.num_episodes,
        "source_episodes": source_episode_count,
        "failed_episodes_excluded": source_episode_count - dataset.num_episodes,
        "steps": len(dataset),
        "source_v1_path": args.source_v1,
        "source_v1_sha256": source_sha,
        "config": asdict(config),
        "epochs_completed": len(history),
        "best_val_mse": min(row["val_mse"] for row in history),
        "final_history_row": history[-1],
        "model": model,
        "runtime_architecture": "BC warm-start only; no runtime expert or rule blending",
    }
    report_path = out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
