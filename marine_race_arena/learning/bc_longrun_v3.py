"""Balanced BC-v3 warm-start training anchored to the selected R1 PPO policy."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.bc_train import BCPolicy
from marine_race_arena.learning.bc_v3_transfer import save_v3_policy
from marine_race_arena.learning.config import OBS_DIM, OBS_ENCODING_VERSION
from marine_race_arena.learning.config_v3 import (
    OBS_DIM_V3,
    OBS_ENCODING_VERSION_V3,
)
from marine_race_arena.learning.dataset import BCDataset
from marine_race_arena.learning.longrun_checkpoint import atomic_write_json
from marine_race_arena.learning.longrun_config import (
    DEFAULT_R1_CHECKPOINT,
    DEFAULT_R1_SHA256,
)
from marine_race_arena.learning.longrun_evaluation import (
    checkpoint_metric_key,
    evaluate_longrun_policy,
)
from marine_race_arena.learning.model_contract_v3 import assert_clean_worktree
from marine_race_arena.learning.provenance import git_sha, now_utc, sha256_file
from marine_race_arena.learning.reward_v3 import MultiGateRewardConfig
from marine_race_arena.learning.seed_registry import (
    MULTIGATE_LONGRUN_BC_EVAL_SEEDS,
    MULTIGATE_LONGRUN_BC_TRAINING_SEEDS,
)


@dataclass
class BalancedBCConfig:
    learning_rate: float = 3e-5
    batch_size: int = 256
    max_epochs: int = 300
    patience: int = 40
    weight_decay: float = 1e-6
    anchor_weight: float = 1e-4
    seed: int = 28000
    top_candidates: int = 3
    single_gate_fraction: float = 0.20
    straight_fraction: float = 0.20
    turns_fraction: float = 0.50
    corrections_fraction: float = 0.10

    def validate(self) -> None:
        fractions = (
            self.single_gate_fraction,
            self.straight_fraction,
            self.turns_fraction,
            self.corrections_fraction,
        )
        if any(v < 0 for v in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("BC mixture fractions must be nonnegative and sum to one")
        if self.batch_size <= 0 or self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("invalid BC training limits")


@dataclass
class PreparedBCData:
    observations: np.ndarray
    actions: np.ndarray
    roles: np.ndarray
    directions: np.ndarray
    splits: np.ndarray
    seeds: np.ndarray
    tracks: np.ndarray
    geometry_groups: np.ndarray


def _geometry(track: str, seed: int, role: str) -> Tuple[str, str]:
    path = Path(track)
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            geometry = raw.get("training_geometry")
            if geometry:
                angle = float(geometry.get("signed_turn_deg", 0.0))
                direction = "left" if angle > 0 else "right" if angle < 0 else "straight"
                group = (
                    f"{direction}:a{round(abs(angle) / 5) * 5}:"
                    f"d{round(float(geometry.get('gate_separation_m', 0)))}:"
                    f"z{round(float(geometry.get('vertical_displacement_m', 0)) * 2) / 2}"
                )
                return direction, group
        except Exception:
            pass
    direction = "straight" if role in {"single_gate", "straight"} else "mixed"
    # A fixed track is an indivisible split unit. Including a seed-family suffix
    # here previously allowed the same fixed track to leak across train/val/test
    # even though seeds and generated transition geometry remained disjoint.
    return direction, f"{role}:{path.as_posix()}"


def _split_for_group(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "test" if bucket == 0 else "validation" if bucket in (1, 2) else "train"


def prepare_datasets(role_paths: Dict[str, Sequence[str]]) -> PreparedBCData:
    obs_rows: List[np.ndarray] = []
    action_rows: List[np.ndarray] = []
    role_rows: List[np.ndarray] = []
    direction_rows: List[np.ndarray] = []
    split_rows: List[np.ndarray] = []
    seed_rows: List[np.ndarray] = []
    track_rows: List[np.ndarray] = []
    group_rows: List[np.ndarray] = []
    for role, paths in role_paths.items():
        for dataset_path in paths:
            dataset = BCDataset.load(dataset_path)
            dataset.check_integrity()
            for episode in dataset.episodes:
                mask = dataset.group_ids == episode.group_id
                observations = dataset.observations[mask]
                if dataset.observation_encoding_version == OBS_ENCODING_VERSION:
                    expanded = np.zeros((len(observations), OBS_DIM_V3), np.float32)
                    expanded[:, :OBS_DIM] = observations
                    observations = expanded
                elif dataset.observation_encoding_version != OBS_ENCODING_VERSION_V3:
                    raise ValueError("unsupported source dataset observation version")
                direction, group = _geometry(episode.track, episode.seed, role)
                split = _split_for_group(group)
                obs_rows.append(observations)
                action_rows.append(dataset.actions[mask])
                role_rows.append(np.full(len(observations), role, object))
                direction_rows.append(np.full(len(observations), direction, object))
                split_rows.append(np.full(len(observations), split, object))
                seed_rows.append(np.full(len(observations), episode.seed, np.int64))
                track_rows.append(np.full(len(observations), episode.track, object))
                group_rows.append(np.full(len(observations), group, object))
    if not obs_rows:
        raise ValueError("no BC datasets were provided")
    data = PreparedBCData(
        observations=np.concatenate(obs_rows).astype(np.float32),
        actions=np.concatenate(action_rows).astype(np.float32),
        roles=np.concatenate(role_rows),
        directions=np.concatenate(direction_rows),
        splits=np.concatenate(split_rows),
        seeds=np.concatenate(seed_rows),
        tracks=np.concatenate(track_rows),
        geometry_groups=np.concatenate(group_rows),
    )
    if not np.all(np.isfinite(data.observations)) or not np.all(
        np.isfinite(data.actions)
    ):
        raise ValueError("prepared BC data contain non-finite values")
    return data


def policy_from_r1_ppo(path: str) -> BCPolicy:
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO

    source = PPO.load(path, device="cpu")
    if tuple(source.observation_space.shape) != (OBS_DIM_V3,):
        raise ValueError("R1 PPO observation shape is not v3")
    policy = BCPolicy(
        obs_dim=OBS_DIM_V3,
        hidden_sizes=(256, 256),
        obs_mean=np.zeros(OBS_DIM_V3, np.float32),
        obs_std=np.ones(OBS_DIM_V3, np.float32),
    )
    source_linears = [
        layer
        for layer in source.policy.mlp_extractor.policy_net
        if isinstance(layer, nn.Linear)
    ]
    target_linears = [
        layer for layer in policy.extractor if isinstance(layer, nn.Linear)
    ]
    with torch.no_grad():
        for source_layer, target_layer in zip(source_linears, target_linears):
            target_layer.weight.copy_(source_layer.weight)
            target_layer.bias.copy_(source_layer.bias)
        policy.head.weight.copy_(source.policy.action_net.weight)
        policy.head.bias.copy_(source.policy.action_net.bias)
    policy.eval()
    return policy


def _metrics(
    policy: BCPolicy,
    observations: np.ndarray,
    actions: np.ndarray,
    directions: np.ndarray,
) -> Dict[str, float]:
    import torch

    policy.eval()
    with torch.no_grad():
        pred = policy(torch.as_tensor(observations, dtype=torch.float32))
        target = torch.as_tensor(actions, dtype=torch.float32)
        axis_mse = ((pred - target) ** 2).mean(dim=0).cpu().numpy()
        prediction = pred.cpu().numpy()
    result = {
        "mse": float(np.mean((prediction - actions) ** 2)),
        "mse_surge": float(axis_mse[0]),
        "mse_sway": float(axis_mse[1]),
        "mse_heave": float(axis_mse[2]),
        "mse_yaw": float(axis_mse[3]),
    }
    for direction in ("left", "right"):
        mask = (directions == direction) & (np.abs(actions[:, 3]) > 0.02)
        result[f"{direction}_turn_sign_accuracy"] = (
            float(np.mean(np.sign(prediction[mask, 3]) == np.sign(actions[mask, 3])))
            if np.any(mask)
            else 0.0
        )
    return result


def train_balanced_bc(
    data: PreparedBCData,
    source: BCPolicy,
    config: Optional[BalancedBCConfig] = None,
) -> Tuple[List[Tuple[float, Dict[str, Any]]], List[Dict[str, Any]]]:
    import torch

    config = config or BalancedBCConfig()
    config.validate()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    train_mask = data.splits == "train"
    val_mask = data.splits == "validation"
    test_mask = data.splits == "test"
    if not np.any(val_mask) or not np.any(test_mask):
        # Small smoke fixtures still get disjoint seed-based validation/test rows.
        order = np.argsort(data.seeds)
        test_mask = np.zeros(len(order), bool)
        val_mask = np.zeros(len(order), bool)
        test_mask[order[::10]] = True
        val_mask[order[1::5]] = True
        train_mask = ~(test_mask | val_mask)

    policy = copy.deepcopy(source)
    anchor = {
        name: value.detach().clone() for name, value in policy.named_parameters()
    }
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    role_fraction = {
        "single_gate": config.single_gate_fraction,
        "straight": config.straight_fraction,
        "turns": config.turns_fraction,
        "corrections": config.corrections_fraction,
    }
    train_indices = np.flatnonzero(train_mask)
    sample_weights = np.zeros(len(train_indices), np.float64)
    for role, fraction in role_fraction.items():
        local = data.roles[train_indices] == role
        if np.any(local) and fraction > 0:
            sample_weights[local] = fraction / int(np.sum(local))
    if sample_weights.sum() <= 0:
        sample_weights[:] = 1.0
    sample_weights /= sample_weights.sum()
    steps_per_epoch = max(1, int(np.ceil(len(train_indices) / config.batch_size)))
    history: List[Dict[str, Any]] = []
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    stale = 0
    best_val = float("inf")
    for epoch in range(config.max_epochs):
        policy.train()
        for _ in range(steps_per_epoch):
            chosen = rng.choice(
                train_indices,
                size=min(config.batch_size, len(train_indices)),
                replace=True,
                p=sample_weights,
            )
            x = torch.as_tensor(data.observations[chosen], dtype=torch.float32)
            y = torch.as_tensor(data.actions[chosen], dtype=torch.float32)
            optimizer.zero_grad()
            prediction = policy(x)
            imitation = ((prediction - y) ** 2).mean()
            anchor_loss = sum(
                ((parameter - anchor[name]) ** 2).mean()
                for name, parameter in policy.named_parameters()
            )
            loss = imitation + config.anchor_weight * anchor_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        val = _metrics(
            policy,
            data.observations[val_mask],
            data.actions[val_mask],
            data.directions[val_mask],
        )
        row = {"epoch": epoch, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        score = val["mse"]
        candidates.append((score, copy.deepcopy(policy.state_dict())))
        candidates = sorted(candidates, key=lambda item: item[0])[
            : config.top_candidates
        ]
        if score < best_val - 1e-7:
            best_val = score
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    # Add test metrics to the history without using them for early stopping.
    best_policy = copy.deepcopy(source)
    best_policy.load_state_dict(candidates[0][1])
    history[-1]["test"] = _metrics(
        best_policy,
        data.observations[test_mask],
        data.actions[test_mask],
        data.directions[test_mask],
    )
    return candidates, history


class _BCPredictor:
    def __init__(self, policy: BCPolicy):
        self.policy = policy
        self.num_timesteps = 0

    def predict(self, observation, deterministic=True):
        return self.policy.act(observation), None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-gate", action="append", default=[])
    parser.add_argument("--straight", action="append", default=[])
    parser.add_argument("--turns", action="append", default=[])
    parser.add_argument("--corrections", action="append", default=[])
    parser.add_argument("--source-ppo", default=DEFAULT_R1_CHECKPOINT)
    parser.add_argument("--source-sha256", default=DEFAULT_R1_SHA256)
    parser.add_argument(
        "--out",
        default="results/rl/multigate_longrun/bc_v3/balanced_bc_v3.pt",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=28000)
    parser.add_argument("--closed-loop", action="store_true")
    args = parser.parse_args(argv)
    assert_clean_worktree()
    if args.seed not in MULTIGATE_LONGRUN_BC_TRAINING_SEEDS:
        raise ValueError("BC seed is outside the allocated long-run training range")
    if sha256_file(args.source_ppo) != args.source_sha256:
        raise ValueError("selected R1 source checkpoint hash mismatch")
    data = prepare_datasets(
        {
            "single_gate": args.single_gate,
            "straight": args.straight,
            "turns": args.turns,
            "corrections": args.corrections,
        }
    )
    config = BalancedBCConfig(
        max_epochs=args.epochs, patience=args.patience, seed=args.seed
    )
    source = policy_from_r1_ppo(args.source_ppo)
    candidates, history = train_balanced_bc(data, source, config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    closed_loop_rows = []
    selected_index = 0
    if args.closed_loop:
        for index, (val_mse, state) in enumerate(candidates):
            candidate = copy.deepcopy(source)
            candidate.load_state_dict(state)
            report = evaluate_longrun_policy(
                _BCPredictor(candidate),
                stage="C4",
                mode="full",
                seeds=MULTIGATE_LONGRUN_BC_EVAL_SEEDS[:20],
                output_dir=out.parent / f"candidate_{index}_evaluation",
                env_kwargs={
                    "adapter": "holoocean",
                    "allow_fallback": False,
                    "current_profile": "none",
                    "max_steps": 1800,
                    "observation_encoding_version": OBS_ENCODING_VERSION_V3,
                },
                reward_config=MultiGateRewardConfig(),
            )
            closed_loop_rows.append(
                {"candidate": index, "val_mse": val_mse, **report}
            )
        selected_index = max(
            range(len(closed_loop_rows)),
            key=lambda index: checkpoint_metric_key(closed_loop_rows[index]),
        )
    selected = copy.deepcopy(source)
    selected.load_state_dict(candidates[selected_index][1])
    save_v3_policy(selected, out)
    with out.with_suffix(".training.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[key for key in history[0] if key != "test"]
        )
        writer.writeheader()
        for row in history:
            writer.writerow(
                {key: value for key, value in row.items() if key != "test"}
            )
    split_summary = {
        split: {
            "steps": int(np.sum(data.splits == split)),
            "seeds": sorted(
                set(int(v) for v in data.seeds[data.splits == split])
            ),
            "tracks": sorted(set(str(v) for v in data.tracks[data.splits == split])),
            "geometry_groups": sorted(
                set(str(v) for v in data.geometry_groups[data.splits == split])
            ),
        }
        for split in ("train", "validation", "test")
    }
    report = {
        "schema_version": "balanced_bc_v3_report_v1",
        "created_utc": now_utc(),
        "code_sha": git_sha(),
        "source_ppo": args.source_ppo,
        "source_sha256": args.source_sha256,
        "source_is_selected_r1": True,
        "config": asdict(config),
        "split_summary": split_summary,
        "split_invariants": {
            "seed_disjoint": not (
                set(split_summary["train"]["seeds"])
                & set(split_summary["validation"]["seeds"])
                or set(split_summary["train"]["seeds"])
                & set(split_summary["test"]["seeds"])
                or set(split_summary["validation"]["seeds"])
                & set(split_summary["test"]["seeds"])
            ),
            "track_disjoint": not (
                set(split_summary["train"]["tracks"])
                & set(split_summary["validation"]["tracks"])
                or set(split_summary["train"]["tracks"])
                & set(split_summary["test"]["tracks"])
            ),
            "geometry_group_disjoint": not (
                set(split_summary["train"]["geometry_groups"])
                & set(split_summary["validation"]["geometry_groups"])
                or set(split_summary["train"]["geometry_groups"])
                & set(split_summary["test"]["geometry_groups"])
            ),
        },
        "epochs_completed": len(history),
        "final_test_metrics": history[-1].get("test"),
        "candidate_validation_mse": [value for value, _ in candidates],
        "closed_loop_selection_used": bool(args.closed_loop),
        "closed_loop_candidates": closed_loop_rows,
        "selected_candidate": selected_index,
        "selection_order": [
            "completion_rate",
            "minimum_left_right_completion",
            "mean_gates",
            "safety",
            "previous_gate_returns",
            "penalized_time",
            "action_jerk",
        ],
        "model_path": str(out),
        "model_sha256": sha256_file(out),
        "runtime_expert_or_rules": False,
    }
    atomic_write_json(out.with_suffix(".report.json"), report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
