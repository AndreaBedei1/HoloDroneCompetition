"""Recurrent behaviour cloning directly on the Gen-2 ``RecurrentPPO`` policy.

Sequence handling is **stateful truncated BPTT**, not shuffled single states:

* episodes are bucketed by length and padded within a batch, with a mask so a
  padded step contributes exactly zero loss;
* each batch is walked in consecutive chunks of ``chunk_length`` while the LSTM
  hidden/cell pair is carried between chunks and detached at the boundary;
* ``episode_starts`` is 1 at the true first step of an episode and nowhere
  else, so the hidden state is zeroed exactly where the deployed controller
  zeroes it.

That combination means the forward pass a training step sees is the same one
inference produces -- only the gradient is truncated.  Sampling random windows
and zeroing the state at each window would instead teach the policy that every
64th step is an episode start, which is precisely the transition confusion
Gen-1 suffered from.

The optimizer moves the parameters of the real SB3 policy, so what BC produces
IS a ``RecurrentPPO`` checkpoint -- there is no transfer step to get wrong.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th
from torch import nn

from marine_race_arena.learning.config import ACTION_AXES, ACTION_DIM
from marine_race_arena.learning.config_local_transition import OBS_DIM_LOCAL_TRANSITION
from marine_race_arena.learning.gen2.dataset import Gen2Episode


@dataclass
class BCConfig:
    """Recurrent BC hyper-parameters.  Defaults are deliberately conservative."""

    chunk_length: int = 64
    batch_episodes: int = 16
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    validation_fraction: float = 0.10
    axis_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    patience: int = 8
    seed: int = 0
    device: str = "cpu"
    #: Train the value head to predict nothing during BC (PPO warms it later).
    train_value_head: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BCMetrics:
    """Supervised quality of one split."""

    mse: float
    per_axis_mse: Dict[str, float]
    per_axis_mae: Dict[str, float]
    per_axis_correlation: Dict[str, float]
    samples: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BCResult:
    config: Dict[str, Any]
    epochs_run: int
    best_epoch: int
    train: Dict[str, Any]
    validation: Dict[str, Any]
    initial_train_mse: float = float("nan")
    initial_validation_mse: float = float("nan")
    #: False when the run ended no better than the policy it started from.
    improved_on_parent: Optional[bool] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    wall_time_s: float = 0.0
    corpus: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def split_episodes(
    episodes: Sequence[Gen2Episode], fraction: float, seed: int
) -> Tuple[List[Gen2Episode], List[Gen2Episode]]:
    """Split by EPISODE, never by step, so no trajectory straddles the split."""
    order = np.random.default_rng(seed).permutation(len(episodes))
    holdout = max(1, int(round(len(episodes) * float(fraction)))) if episodes else 0
    holdout = min(holdout, max(0, len(episodes) - 1))
    validation_index = set(int(i) for i in order[:holdout])
    train = [episodes[i] for i in range(len(episodes)) if i not in validation_index]
    validation = [episodes[i] for i in range(len(episodes)) if i in validation_index]
    return train, validation


def _length_bucketed_batches(
    episodes: Sequence[Gen2Episode], batch_episodes: int, rng: np.random.Generator
) -> List[List[Gen2Episode]]:
    """Group similar-length episodes so padding waste stays small."""
    order = sorted(range(len(episodes)), key=lambda i: len(episodes[i]))
    batches: List[List[Gen2Episode]] = []
    for start in range(0, len(order), batch_episodes):
        batches.append([episodes[i] for i in order[start : start + batch_episodes]])
    rng.shuffle(batches)
    return batches


def _pad_batch(batch: Sequence[Gen2Episode]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(obs, target, mask, episode_starts)`` shaped ``(B, L, ...)``."""
    size = len(batch)
    length = max(len(e) for e in batch)
    obs_dim = int(batch[0].observations.shape[1])
    obs = np.zeros((size, length, obs_dim), dtype=np.float32)
    target = np.zeros((size, length, ACTION_DIM), dtype=np.float32)
    mask = np.zeros((size, length), dtype=np.float32)
    starts = np.zeros((size, length), dtype=np.float32)
    for index, episode in enumerate(batch):
        steps = len(episode)
        obs[index, :steps] = episode.observations
        target[index, :steps] = episode.expert_actions
        mask[index, :steps] = 1.0
        starts[index, 0] = 1.0
    return obs, target, mask, starts


def _forward_actor_chunk(policy, obs_chunk: th.Tensor, starts_chunk: th.Tensor, states):
    """Run the actor path over ``(B, L, 35)`` and return ``(mean, new_states)``."""
    batch, length, _ = obs_chunk.shape
    flat = obs_chunk.reshape(batch * length, obs_chunk.shape[-1])
    features = policy.extract_features(flat)
    if isinstance(features, tuple):  # share_features_extractor=False
        features = features[0]
    latent, states = policy._process_sequence(
        features, states, starts_chunk.reshape(batch * length), policy.lstm_actor
    )
    latent = policy.mlp_extractor.forward_actor(latent)
    mean = policy.action_net(latent)
    return mean.reshape(batch, length, ACTION_DIM), states


def _zero_states(policy, batch: int, device: th.device):
    shape = (policy.lstm_actor.num_layers, batch, policy.lstm_actor.hidden_size)
    return (th.zeros(shape, device=device), th.zeros(shape, device=device))


def _evaluate(policy, episodes: Sequence[Gen2Episode], config: BCConfig, device: th.device) -> BCMetrics:
    """Teacher-forced supervised error over whole episodes."""
    if not episodes:
        return BCMetrics(float("nan"), {}, {}, {}, 0)
    policy.set_training_mode(False)
    squared = np.zeros(ACTION_DIM, dtype=np.float64)
    absolute = np.zeros(ACTION_DIM, dtype=np.float64)
    count = 0
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    rng = np.random.default_rng(0)
    with th.no_grad():
        for batch in _length_bucketed_batches(episodes, config.batch_episodes, rng):
            obs, target, mask, starts = _pad_batch(batch)
            obs_t = th.as_tensor(obs, device=device)
            target_t = th.as_tensor(target, device=device)
            mask_t = th.as_tensor(mask, device=device)
            starts_t = th.as_tensor(starts, device=device)
            states = _zero_states(policy, len(batch), device)
            length = obs.shape[1]
            for begin in range(0, length, config.chunk_length):
                end = min(begin + config.chunk_length, length)
                mean, states = _forward_actor_chunk(
                    policy, obs_t[:, begin:end], starts_t[:, begin:end], states
                )
                chunk_mask = mask_t[:, begin:end].unsqueeze(-1)
                error = (mean - target_t[:, begin:end]) * chunk_mask
                squared += error.pow(2).sum(dim=(0, 1)).cpu().numpy()
                absolute += error.abs().sum(dim=(0, 1)).cpu().numpy()
                count += int(chunk_mask.sum().item())
                selected = chunk_mask.squeeze(-1) > 0
                predictions.append(mean[selected].cpu().numpy())
                targets.append(target_t[:, begin:end][selected].cpu().numpy())
    count = max(1, count)
    prediction = np.concatenate(predictions) if predictions else np.zeros((0, ACTION_DIM))
    truth = np.concatenate(targets) if targets else np.zeros((0, ACTION_DIM))
    correlation: Dict[str, float] = {}
    for index, axis in enumerate(ACTION_AXES):
        if prediction.shape[0] > 1 and np.std(truth[:, index]) > 1e-9 and np.std(prediction[:, index]) > 1e-9:
            correlation[axis] = round(
                float(np.corrcoef(prediction[:, index], truth[:, index])[0, 1]), 5
            )
        else:
            correlation[axis] = float("nan")
    return BCMetrics(
        mse=round(float(squared.sum() / (count * ACTION_DIM)), 8),
        per_axis_mse={axis: round(float(squared[i] / count), 8) for i, axis in enumerate(ACTION_AXES)},
        per_axis_mae={axis: round(float(absolute[i] / count), 8) for i, axis in enumerate(ACTION_AXES)},
        per_axis_correlation=correlation,
        samples=int(count),
    )


def train_recurrent_bc(
    model,
    episodes: Sequence[Gen2Episode],
    config: Optional[BCConfig] = None,
    *,
    latch_normalization: bool = True,
    progress_path: Optional[str | Path] = None,
    validation_episodes: Optional[Sequence[Gen2Episode]] = None,
) -> BCResult:
    """Behaviour-clone ``model.policy`` on ``episodes``.  Mutates ``model``."""
    config = config or BCConfig()
    if not episodes:
        raise ValueError("recurrent BC needs at least one episode")
    started = time.perf_counter()
    device = th.device(config.device)
    policy = model.policy
    policy.to(device)

    if validation_episodes is None:
        train_episodes, validation_episodes = split_episodes(
            episodes, config.validation_fraction, config.seed
        )
    else:
        # Caller already split -- and, when corpus roots are replicated, it had
        # to: splitting after replication leaks duplicates across the boundary.
        train_episodes = list(episodes)
    if latch_normalization:
        values = np.concatenate([episode.observations for episode in train_episodes], axis=0).astype(np.float64)
        mean = values.mean(axis=0).astype(np.float32)
        std = values.std(axis=0).astype(np.float32)
        std[std < 1e-3] = 1.0
        extractor = policy.features_extractor
        if not hasattr(extractor, "set_normalization"):
            raise TypeError("Gen-2 BC requires the Gen2ObsEncoder features extractor")
        extractor.set_normalization(mean, std)

    trainable: List[nn.Parameter] = []
    for module_name in ("features_extractor", "lstm_actor", "action_net"):
        module = getattr(policy, module_name, None)
        if module is not None:
            trainable += list(module.parameters())
    trainable += list(policy.mlp_extractor.policy_net.parameters())
    optimizer = th.optim.Adam(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    weights = th.as_tensor(np.asarray(config.axis_weights, dtype=np.float32), device=device)

    rng = np.random.default_rng(config.seed)
    # Measure the incoming policy BEFORE any update. Without this, a fine-tune
    # that destroys its parent in the first epoch is indistinguishable from one
    # that started badly -- which is exactly how a 32x fit regression went
    # unnoticed until it cost a full circuit evaluation.
    initial_train = _evaluate(policy, train_episodes, config, device)
    initial_validation = _evaluate(policy, validation_episodes, config, device)
    best_score = (
        initial_validation.mse if validation_episodes else initial_train.mse
    )
    best_state: Optional[Dict[str, th.Tensor]] = copy.deepcopy(policy.state_dict())
    best_epoch = -1
    history: List[Dict[str, Any]] = [{
        "epoch": -1,
        "mean_chunk_loss": None,
        "train_mse": initial_train.mse,
        "validation_mse": initial_validation.mse,
        "note": "parent policy before any update",
    }]
    stale = 0
    epochs_run = 0

    for epoch in range(int(config.epochs)):
        epochs_run = epoch + 1
        policy.set_training_mode(True)
        epoch_loss = 0.0
        epoch_steps = 0
        for batch in _length_bucketed_batches(train_episodes, config.batch_episodes, rng):
            obs, target, mask, starts = _pad_batch(batch)
            obs_t = th.as_tensor(obs, device=device)
            target_t = th.as_tensor(target, device=device)
            mask_t = th.as_tensor(mask, device=device)
            starts_t = th.as_tensor(starts, device=device)
            states = _zero_states(policy, len(batch), device)
            length = obs.shape[1]
            for begin in range(0, length, config.chunk_length):
                end = min(begin + config.chunk_length, length)
                chunk_mask = mask_t[:, begin:end]
                if float(chunk_mask.sum().item()) == 0.0:
                    continue
                mean, states = _forward_actor_chunk(
                    policy, obs_t[:, begin:end], starts_t[:, begin:end], states
                )
                error = (mean - target_t[:, begin:end]).pow(2) * weights
                loss = (error.sum(dim=-1) * chunk_mask).sum() / (
                    chunk_mask.sum() * float(weights.sum().item())
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.grad_clip:
                    th.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
                optimizer.step()
                # Carry the sequence forward but cut the gradient at the chunk
                # boundary: the forward pass stays exact, the graph stays bounded.
                states = (states[0].detach(), states[1].detach())
                epoch_loss += float(loss.item())
                epoch_steps += 1

        train_metrics = _evaluate(policy, train_episodes, config, device)
        validation_metrics = _evaluate(policy, validation_episodes, config, device)
        score = validation_metrics.mse if validation_episodes else train_metrics.mse
        row = {
            "epoch": epoch,
            "mean_chunk_loss": round(epoch_loss / max(1, epoch_steps), 8),
            "train_mse": train_metrics.mse,
            "validation_mse": validation_metrics.mse,
        }
        history.append(row)
        if progress_path is not None:
            Path(progress_path).parent.mkdir(parents=True, exist_ok=True)
            Path(progress_path).write_text(
                json.dumps({"history": history}, indent=2), encoding="utf-8"
            )
        if score < best_score - 1e-9:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(policy.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(config.patience):
                break

    if best_state is not None:
        policy.load_state_dict(best_state)
    policy.set_training_mode(False)

    final_train = _evaluate(policy, train_episodes, config, device)
    final_validation = _evaluate(policy, validation_episodes, config, device)
    improved = (
        None if math.isinf(best_score)
        else bool(best_score <= (
            initial_validation.mse if validation_episodes else initial_train.mse
        ))
    )
    return BCResult(
        config=config.as_dict(),
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        initial_train_mse=initial_train.mse,
        initial_validation_mse=initial_validation.mse,
        improved_on_parent=improved,
        train=final_train.as_dict(),
        validation=final_validation.as_dict(),
        history=history,
        wall_time_s=round(time.perf_counter() - started, 2),
        corpus={
            "episodes": len(episodes),
            "train_episodes": len(train_episodes),
            "validation_episodes": len(validation_episodes),
            "transitions": int(sum(len(e) for e in episodes)),
        },
    )


def calibrate_action_std(model, episodes: Sequence[Gen2Episode], config: Optional[BCConfig] = None) -> np.ndarray:
    """Set PPO's initial exploration std from the BC residuals, per axis.

    Starting PPO at ``log_std = 0`` (std 1.0) next to actions bounded in
    ``[-1, 1]`` would drown the cloned behaviour in noise on the first update.
    Sizing the std by how well BC actually reproduces each axis keeps the first
    PPO rollouts recognisably the DAgger policy.
    """
    config = config or BCConfig()
    device = th.device(config.device)
    metrics = _evaluate(model.policy, episodes, config, device)
    residual = np.asarray(
        [max(1e-3, math.sqrt(metrics.per_axis_mse[axis])) for axis in ACTION_AXES],
        dtype=np.float32,
    )
    residual = np.clip(residual, 1e-3, 0.5)
    with th.no_grad():
        model.policy.log_std.copy_(th.as_tensor(np.log(residual), dtype=th.float32))
    return residual
