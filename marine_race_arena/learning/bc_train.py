"""Behavioral-cloning policy and trainer (PyTorch).

A small MLP maps the fixed learning observation to the four normalized command
axes. The network mirrors Stable-Baselines3's default ``MlpPolicy`` structure
(``Tanh`` hidden layers + a linear action head) so its weights can warm-start a
PPO policy exactly (see :mod:`rl_train`). Actions are bounded by clipping to
``[-1, 1]`` at inference. Observation normalization statistics are stored as
buffers, so a saved policy is self-contained and needs no dataset at deployment.

PyTorch is imported lazily by callers via this module; it is an RL-only
dependency (requirements-rl.txt) and is never needed by the benchmark.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from marine_race_arena.learning.config import ACTION_DIM, OBS_DIM


@dataclass
class BCConfig:
    hidden_sizes: Sequence[int] = (256, 256)
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    val_fraction: float = 0.2
    weight_decay: float = 0.0
    axis_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0)
    seed: int = 0


class BCPolicy(nn.Module):
    """MLP policy: Tanh hidden layers + linear action head, with obs normalization.

    Split into ``extractor`` (hidden layers) and ``head`` (linear) to mirror the
    SB3 ``mlp_extractor``/``action_net`` split for weight transfer.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACTION_DIM,
        hidden_sizes: Sequence[int] = (256, 256),
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)

        layers: List[nn.Module] = []
        prev = self.obs_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        self.extractor = nn.Sequential(*layers)
        self.head = nn.Linear(prev, self.act_dim)

        mean = np.zeros(self.obs_dim, dtype=np.float32) if obs_mean is None else np.asarray(obs_mean, dtype=np.float32)
        std = np.ones(self.obs_dim, dtype=np.float32) if obs_std is None else np.asarray(obs_std, dtype=np.float32)
        self.register_buffer("obs_mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("obs_std", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = (obs - self.obs_mean) / self.obs_std
        return self.head(self.extractor(normalized))

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Deterministic bounded action for a single observation vector."""
        self.eval()
        tensor = torch.as_tensor(np.asarray(observation, dtype=np.float32)).reshape(1, -1)
        raw = self.forward(tensor).reshape(-1).cpu().numpy()
        return np.clip(raw, -1.0, 1.0).astype(np.float32)


def train_bc(
    dataset,
    config: Optional[BCConfig] = None,
    *,
    log_csv: Optional[str] = None,
) -> Tuple[BCPolicy, List[Dict[str, float]]]:
    """Train a BC policy with an episode-level train/val split and early stopping.

    Normalization statistics come from the *training* split only. Returns the best
    (lowest validation MSE) policy and the per-epoch history.
    """
    config = config or BCConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_ds, val_ds = dataset.train_val_split(config.val_fraction, seed=config.seed)
    mean, std = train_ds.normalization_stats()
    policy = BCPolicy(hidden_sizes=config.hidden_sizes, obs_mean=mean, obs_std=std)

    x_tr = torch.as_tensor(train_ds.observations, dtype=torch.float32)
    y_tr = torch.as_tensor(train_ds.actions, dtype=torch.float32)
    x_va = torch.as_tensor(val_ds.observations, dtype=torch.float32)
    y_va = torch.as_tensor(val_ds.actions, dtype=torch.float32)
    axis_w = torch.as_tensor(np.asarray(config.axis_weights, dtype=np.float32))

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def weighted_mse(pred, target):
        return (axis_w * (pred - target) ** 2).mean()

    best_val = float("inf")
    best_state = None
    patience = 0
    history: List[Dict[str, float]] = []
    n = x_tr.shape[0]

    for epoch in range(config.max_epochs):
        policy.train()
        perm = torch.randperm(n)
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            optimizer.zero_grad()
            loss = weighted_mse(policy(x_tr[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()

        policy.eval()
        with torch.no_grad():
            tr_loss = float(((policy(x_tr) - y_tr) ** 2).mean().item())
            va_pred = policy(x_va)
            va_loss = float(((va_pred - y_va) ** 2).mean().item())
            per_axis = ((va_pred - y_va) ** 2).mean(dim=0).cpu().numpy().tolist()
        record = {"epoch": epoch, "train_mse": tr_loss, "val_mse": va_loss}
        for i, axis in enumerate(("surge", "sway", "heave", "yaw")):
            record[f"val_mse_{axis}"] = float(per_axis[i])
        history.append(record)

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = copy.deepcopy(policy.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= config.patience:
                break

    if best_state is not None:
        policy.load_state_dict(best_state)
    if log_csv is not None:
        _write_csv(log_csv, history)
    return policy, history


def _write_csv(path: str, history: List[Dict[str, float]]) -> None:
    import csv

    if not history:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_policy(policy: BCPolicy, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "bc",
            "obs_dim": policy.obs_dim,
            "act_dim": policy.act_dim,
            "hidden_sizes": list(policy.hidden_sizes),
            "state_dict": policy.state_dict(),
        },
        path,
    )


def load_policy(path) -> BCPolicy:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    policy = BCPolicy(
        obs_dim=int(checkpoint["obs_dim"]),
        act_dim=int(checkpoint["act_dim"]),
        hidden_sizes=tuple(checkpoint["hidden_sizes"]),
    )
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy
