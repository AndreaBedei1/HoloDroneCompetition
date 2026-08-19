"""Does recurrence actually help? -- the matched A/B/C/D ablation.

Four arms, evaluated on **identical** VALIDATION seeds so the comparison is
paired rather than two independent samples:

======  ==========================================================
Arm     Policy
======  ==========================================================
A       Generation-1 feed-forward PPO (the frozen final checkpoint)
B       feed-forward behaviour cloning
C       recurrent behaviour cloning
D       recurrent behaviour cloning + DAgger
======  ==========================================================

Arm B exists to make the answer attributable.  A recurrent policy trained on a
new corpus differs from Gen-1 in *two* ways at once -- the architecture and the
data -- so C-vs-A cannot separate them.  B is built from the same encoder, the
same head widths, the same corpus, the same optimizer and the same schedule as
C, with the LSTM removed and nothing else changed.  C minus B is therefore the
contribution of temporal memory alone, and B minus A is the contribution of
expert bootstrapping alone.

The reported quantity is the post-crossing transition, not the headline
completion rate: the Gen-2 hypothesis is specifically that memory helps the
policy re-acquire the *next* gate after crossing one.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from marine_race_arena.learning.gen2 import seeds as gen2_seeds

#: The lengths the ablation is measured at.  Short enough that arm A has a
#: chance, long enough that compounding error is visible.
ABLATION_GATE_COUNTS: Tuple[int, ...] = (2, 3, 5, 8)


@dataclass
class ArmResult:
    arm: str
    label: str
    checkpoint: Optional[str]
    architecture: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def ablation_seeds(episodes: int) -> List[int]:
    pool = gen2_seeds.GEN2_ABLATION_SEEDS
    if episodes > len(pool):
        raise ValueError(f"ablation band holds {len(pool)} seeds, asked for {episodes}")
    chosen = pool[:episodes]
    for seed in chosen:
        if gen2_seeds.band_of(seed) != "VALIDATION":
            raise PermissionError(f"ablation drew a non-validation seed {seed}")
    return chosen


# ------------------------------------------------------- feed-forward arm B

def recurrent_actor_parameter_count() -> int:
    """Actor-path parameters of arm C: encoder + LSTM + policy head."""
    from marine_race_arena.learning.gen2.recurrent_policy import (
        build_gen2_policy_for_training,
        policy_parameter_count,
    )

    counts = policy_parameter_count(build_gen2_policy_for_training(seed=0))
    # mlp_extractor holds both policy_net and value_net; the actor uses half.
    return int(
        counts["features_extractor"]
        + counts["lstm_actor"]
        + counts["mlp_extractor"] // 2
        + counts["action_net"]
    )


def _head_parameter_count(input_dim: int, widths: Sequence[int], action_dim: int) -> int:
    total = 0
    previous = int(input_dim)
    for width in widths:
        total += previous * int(width) + int(width)
        previous = int(width)
    return total + previous * int(action_dim) + int(action_dim)


def matched_feedforward_head(tolerance: float = 0.05) -> Tuple[int, ...]:
    """Head widths that give arm B the same actor capacity as arm C.

    Without this, arm B is ~110k parameters against arm C's ~340k, and any
    C-over-B advantage could be read as capacity rather than memory.  The
    widths are searched, not hardcoded, so they follow the architecture if it
    changes.
    """
    from marine_race_arena.learning.config import ACTION_DIM
    from marine_race_arena.learning.gen2.recurrent_policy import GEN2_ARCH

    target = recurrent_actor_parameter_count()
    encoder_out = int(GEN2_ARCH.encoder_hidden[-1])
    # Encoder parameters are identical in both arms, so only the head differs.
    encoder_params = 0
    previous = 35
    for width in GEN2_ARCH.encoder_hidden:
        encoder_params += previous * int(width) + int(width)
        previous = int(width)
    head_target = target - encoder_params

    best: Optional[Tuple[int, ...]] = None
    best_error = math.inf
    for first in range(64, 1025, 8):
        for second in (GEN2_ARCH.head_hidden[-1], 256, 384, 512):
            widths = (first, int(second))
            count = _head_parameter_count(encoder_out, widths, ACTION_DIM)
            error = abs(count - head_target) / max(1, head_target)
            if error < best_error:
                best_error, best = error, widths
    if best is None or best_error > tolerance:
        raise ValueError(
            f"could not match arm B capacity within {tolerance:.0%} "
            f"(best error {best_error:.1%})"
        )
    return best


def build_feedforward_policy(
    *, obs_mean=None, obs_std=None, seed: int = 0, device: str = "cpu",
    learning_rate: float = 3e-4, head: Optional[Sequence[int]] = None,
):
    """Arm B: the Gen-2 encoder and a capacity-matched head, LSTM removed."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from torch import nn

    from marine_race_arena.learning.gen2.recurrent_policy import (
        GEN2_ARCH,
        Gen2ObsEncoder,
        _GymContractEnv,
    )

    widths = list(head) if head is not None else list(matched_feedforward_head())
    env = DummyVecEnv([lambda: _GymContractEnv()])
    model = PPO(
        "MlpPolicy", env,
        learning_rate=learning_rate,
        policy_kwargs=dict(
            activation_fn=nn.Tanh,
            net_arch=dict(pi=widths, vf=widths),
            features_extractor_class=Gen2ObsEncoder,
            features_extractor_kwargs=dict(
                hidden=tuple(GEN2_ARCH.encoder_hidden),
                obs_mean=obs_mean, obs_std=obs_std,
            ),
            share_features_extractor=True,
        ),
        seed=seed, device=device, verbose=0,
    )
    model.gen2_ablation_head = widths
    return model


def train_feedforward_bc(model, episodes, config) -> Dict[str, Any]:
    """Behaviour-clone arm B on exactly the same corpus and schedule as arm C.

    Steps are still grouped by episode and the episode split is still by
    episode, so the only difference from the recurrent trainer is that there is
    no hidden state to carry.
    """
    import torch as th

    from marine_race_arena.learning.config import ACTION_AXES, ACTION_DIM
    from marine_race_arena.learning.gen2.bc_recurrent import _pad_batch, split_episodes
    from marine_race_arena.learning.gen2.dataset import observation_statistics

    device = th.device(config.device)
    policy = model.policy
    policy.to(device)
    train_episodes, validation_episodes = split_episodes(
        episodes, config.validation_fraction, config.seed
    )
    mean, std = observation_statistics(train_episodes)
    policy.features_extractor.set_normalization(mean, std)

    trainable = list(policy.features_extractor.parameters())
    trainable += list(policy.mlp_extractor.policy_net.parameters())
    trainable += list(policy.action_net.parameters())
    optimizer = th.optim.Adam(trainable, lr=config.learning_rate)

    def forward(obs_batch):
        features = policy.extract_features(obs_batch)
        if isinstance(features, tuple):
            features = features[0]
        return policy.action_net(policy.mlp_extractor.forward_actor(features))

    def evaluate(pool):
        if not pool:
            return float("nan")
        policy.set_training_mode(False)
        total = 0.0
        count = 0
        with th.no_grad():
            for episode in pool:
                obs = th.as_tensor(episode.observations, device=device)
                target = th.as_tensor(episode.expert_actions, device=device)
                total += float((forward(obs) - target).pow(2).sum().item())
                count += int(obs.shape[0])
        return total / max(1, count * ACTION_DIM)

    rng = np.random.default_rng(config.seed)
    best = math.inf
    best_state = None
    history: List[Dict[str, Any]] = []
    stale = 0
    for epoch in range(int(config.epochs)):
        policy.set_training_mode(True)
        order = rng.permutation(len(train_episodes))
        for start in range(0, len(order), config.batch_episodes):
            batch = [train_episodes[i] for i in order[start : start + config.batch_episodes]]
            obs, target, mask, _ = _pad_batch(batch)
            obs_t = th.as_tensor(obs.reshape(-1, obs.shape[-1]), device=device)
            target_t = th.as_tensor(target.reshape(-1, ACTION_DIM), device=device)
            mask_t = th.as_tensor(mask.reshape(-1, 1), device=device)
            loss = (((forward(obs_t) - target_t) ** 2) * mask_t).sum() / (
                mask_t.sum() * ACTION_DIM
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip:
                th.nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            optimizer.step()
        score = evaluate(validation_episodes)
        history.append({"epoch": epoch, "validation_mse": round(score, 8)})
        if score < best - 1e-9:
            best, best_state, stale = score, copy.deepcopy(policy.state_dict()), 0
        else:
            stale += 1
            if stale >= int(config.patience):
                break
    if best_state is not None:
        policy.load_state_dict(best_state)
    policy.set_training_mode(False)
    return {
        "validation_mse": round(best, 8),
        "epochs_run": len(history),
        "history": history,
    }


class FeedforwardController:
    """Arm B inference wrapper, matching the recurrent controller's interface."""

    def __init__(self, model, *, deterministic: bool = True) -> None:
        self.model = model
        self.deterministic = bool(deterministic)
        self.steps = 0

    def reset(self) -> None:
        self.steps = 0

    def act(self, observation, first_step: bool = False):
        from marine_race_arena.learning.config import ACTION_DIM

        action, _ = self.model.predict(
            np.asarray(observation, dtype=np.float32).reshape(1, -1),
            deterministic=self.deterministic,
        )
        self.steps += 1
        return np.clip(np.asarray(action, np.float32).reshape(ACTION_DIM), -1.0, 1.0)

    def __call__(self, observation, first_step: bool = False):
        return self.act(observation, first_step)


# ------------------------------------------------------------- arm loading

def load_gen1_controller(checkpoint: str | Path):
    """Arm A: the frozen Generation-1 feed-forward PPO.

    Loaded read-only from wherever it was frozen; nothing here writes to it.
    """
    from stable_baselines3 import PPO

    model = PPO.load(str(checkpoint), device="cpu")
    return FeedforwardController(model, deterministic=True)


def load_recurrent_controller(checkpoint: str | Path):
    from sb3_contrib import RecurrentPPO

    from marine_race_arena.learning.gen2.recurrent_policy import Gen2RecurrentController

    return Gen2RecurrentController(
        RecurrentPPO.load(str(checkpoint), device="cpu"), deterministic=True
    )


#: arm -> (label, architecture, loader).
ARM_PLAN: Dict[str, Tuple[str, str, Any]] = {
    "A": ("gen1_feedforward_ppo", "feedforward", load_gen1_controller),
    "B": ("feedforward_bc", "feedforward", load_gen1_controller),
    "C": ("recurrent_bc", "recurrent_lstm", load_recurrent_controller),
    "D": ("recurrent_bc_dagger", "recurrent_lstm", load_recurrent_controller),
}


def run_arm(
    arm: str,
    checkpoint: str | Path,
    *,
    out_dir: str | Path,
    episodes: int = 60,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
) -> ArmResult:
    """Evaluate ONE arm and write ``arm_<X>.json``.

    Split out so the four arms can run as concurrent processes.  Running them
    sequentially costs 4x wall clock for no scientific benefit -- the arms are
    independent, and every arm sees the identical seed list either way.
    """
    from marine_race_arena.learning.gen2.evaluation import evaluate_policy

    label, architecture, loader = ARM_PLAN[arm]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = ablation_seeds(episodes)
    target = out_dir / f"arm_{arm}.json"

    if checkpoint is None or not Path(checkpoint).exists():
        result = ArmResult(arm, label, None, architecture, error="checkpoint not available")
    else:
        try:
            outcome = evaluate_policy(
                loader(checkpoint), seeds,
                gate_counts=list(ABLATION_GATE_COUNTS),
                # Per-arm track directory: two arms must never share a course
                # file, even though the content would be identical.
                track_dir=out_dir / f"tracks_arm_{arm}",
                adapter=adapter, allow_fallback=allow_fallback,
            )
            result = ArmResult(arm, label, str(checkpoint), architecture,
                               metrics=outcome["metrics"])
            (out_dir / f"arm_{arm}_episodes.json").write_text(
                json.dumps(outcome["episodes"], indent=2), encoding="utf-8"
            )
        except Exception as exc:
            result = ArmResult(arm, label, str(checkpoint), architecture,
                               error=f"{type(exc).__name__}: {exc}")
    target.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return result


def collect_arms(out_dir: str | Path, *, episodes: int) -> Dict[str, Any]:
    """Aggregate whatever ``arm_<X>.json`` files exist into one report."""
    out_dir = Path(out_dir)
    results: List[ArmResult] = []
    for arm in ("A", "B", "C", "D"):
        path = out_dir / f"arm_{arm}.json"
        label, architecture, _ = ARM_PLAN[arm]
        if not path.exists():
            results.append(ArmResult(arm, label, None, architecture, error="arm not run"))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        results.append(ArmResult(
            arm=payload["arm"], label=payload["label"],
            checkpoint=payload.get("checkpoint"), architecture=payload["architecture"],
            metrics=payload.get("metrics") or {}, error=payload.get("error"),
        ))
    seeds = ablation_seeds(episodes)
    summary = {
        "schema_version": "gen2_recurrence_ablation_v1",
        "matched_seeds": [seeds[0], seeds[-1]],
        "episodes_per_arm": len(seeds),
        "gate_counts": list(ABLATION_GATE_COUNTS),
        "arms": [r.as_dict() for r in results],
        "comparison": compare_arms(results),
    }
    (out_dir / "ablation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_ablation(
    *,
    out_dir: str | Path,
    episodes: int = 60,
    gen1_checkpoint: Optional[str | Path] = None,
    feedforward_bc_checkpoint: Optional[str | Path] = None,
    recurrent_bc_checkpoint: Optional[str | Path] = None,
    dagger_checkpoint: Optional[str | Path] = None,
    adapter: str = "holoocean",
    allow_fallback: bool = False,
) -> Dict[str, Any]:
    """Evaluate every available arm on the same seeds, sequentially."""
    started = time.perf_counter()
    checkpoints = {
        "A": gen1_checkpoint, "B": feedforward_bc_checkpoint,
        "C": recurrent_bc_checkpoint, "D": dagger_checkpoint,
    }
    for arm, checkpoint in checkpoints.items():
        run_arm(arm, checkpoint, out_dir=out_dir, episodes=episodes,
                adapter=adapter, allow_fallback=allow_fallback)
    summary = collect_arms(out_dir, episodes=episodes)
    summary["wall_time_s"] = round(time.perf_counter() - started, 1)
    (Path(out_dir) / "ablation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def compare_arms(results: Sequence[ArmResult]) -> Dict[str, Any]:
    """Attribute the difference: memory (C-B) versus bootstrapping (B-A)."""
    by_arm = {r.arm: r for r in results}

    def value(arm: str, key: str) -> Optional[float]:
        entry = by_arm.get(arm)
        if entry is None or not entry.metrics:
            return None
        return entry.metrics.get(key)

    def delta(later: str, earlier: str, key: str) -> Optional[float]:
        a, b = value(later, key), value(earlier, key)
        return None if a is None or b is None else round(a - b, 4)

    key = "gate1_to_gate2_transition_rate"
    return {
        "metric": key,
        "arms": {arm: value(arm, key) for arm in ("A", "B", "C", "D")},
        "expert_bootstrapping_effect_B_minus_A": delta("B", "A", key),
        "recurrence_effect_C_minus_B": delta("C", "B", key),
        "dagger_effect_D_minus_C": delta("D", "C", key),
        "completion": {arm: value(arm, "overall_completion_rate") for arm in ("A", "B", "C", "D")},
        "note": (
            "B removes only the LSTM from C, on the same corpus and schedule, so "
            "C-B isolates temporal memory and B-A isolates expert bootstrapping."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Gen-2 recurrence ablation")
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument(
        "--arm", default=None, choices=sorted(ARM_PLAN),
        help="run exactly one arm (the four arms are independent, so they can "
             "run as concurrent processes); omit to run all four sequentially",
    )
    parser.add_argument(
        "--collect-only", action="store_true",
        help="aggregate the per-arm files already on disk",
    )
    parser.add_argument("--train-ff-bc", default=None,
                        help="train arm B on this corpus and write the checkpoint to --ff-bc")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gen1", default=None)
    parser.add_argument("--ff-bc", default=None)
    parser.add_argument("--recurrent-bc", default=None)
    parser.add_argument("--dagger", default=None)
    parser.add_argument("--adapter", default="holoocean", choices=("holoocean", "fallback"))
    parser.add_argument("--allow-fallback", action="store_true")
    return parser


def train_arm_b(corpus: Sequence[str], checkpoint: str | Path, *, epochs: int, seed: int) -> Dict[str, Any]:
    """Train the feed-forward BC arm on the same corpus as the recurrent arm."""
    from marine_race_arena.learning.gen2.bc_recurrent import BCConfig
    from marine_race_arena.learning.gen2.dataset import (
        corpus_statistics,
        load_corpus,
        observation_statistics,
    )

    episodes = load_corpus(list(corpus))
    if not episodes:
        raise ValueError(f"no episodes under {list(corpus)}")
    mean, std = observation_statistics(episodes)
    model = build_feedforward_policy(obs_mean=mean, obs_std=std, seed=seed)
    result = train_feedforward_bc(model, episodes, BCConfig(epochs=epochs, seed=seed))
    target = Path(checkpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save(target)
    payload = {
        "schema_version": "gen2_ablation_arm_b_v1",
        "checkpoint": str(target),
        "head_widths": list(getattr(model, "gen2_ablation_head", [])),
        "actor_parameters": int(
            sum(p.numel() for p in model.policy.features_extractor.parameters())
            + sum(p.numel() for p in model.policy.mlp_extractor.policy_net.parameters())
            + sum(p.numel() for p in model.policy.action_net.parameters())
        ),
        "recurrent_actor_parameters": recurrent_actor_parameter_count(),
        "corpus": [str(c) for c in corpus],
        "corpus_statistics": corpus_statistics(episodes),
        "bc": {k: v for k, v in result.items() if k != "history"},
    }
    target.with_suffix(".manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.train_ff_bc:
        if not args.ff_bc:
            raise SystemExit("--train-ff-bc requires --ff-bc as the output checkpoint")
        payload = train_arm_b(
            [args.train_ff_bc], args.ff_bc, epochs=args.epochs, seed=args.seed
        )
        print(json.dumps({
            "checkpoint": payload["checkpoint"],
            "head_widths": payload["head_widths"],
            "actor_parameters": payload["actor_parameters"],
            "recurrent_actor_parameters": payload["recurrent_actor_parameters"],
            "validation_mse": payload["bc"]["validation_mse"],
        }, indent=2), flush=True)
        return 0
    if args.collect_only:
        print(json.dumps(collect_arms(args.out, episodes=args.episodes)["comparison"],
                         indent=2), flush=True)
        return 0
    if args.arm:
        checkpoints = {"A": args.gen1, "B": args.ff_bc, "C": args.recurrent_bc, "D": args.dagger}
        result = run_arm(
            args.arm, checkpoints[args.arm], out_dir=args.out,
            episodes=args.episodes, adapter=args.adapter,
            allow_fallback=args.allow_fallback,
        )
        print(json.dumps({
            "arm": result.arm, "label": result.label, "error": result.error,
            "completion": (result.metrics or {}).get("overall_completion_rate"),
            "gate1_to_gate2": (result.metrics or {}).get("gate1_to_gate2_transition_rate"),
        }, indent=2), flush=True)
        return 0
    summary = run_ablation(
        out_dir=args.out, episodes=args.episodes,
        gen1_checkpoint=args.gen1, feedforward_bc_checkpoint=args.ff_bc,
        recurrent_bc_checkpoint=args.recurrent_bc, dagger_checkpoint=args.dagger,
        adapter=args.adapter, allow_fallback=args.allow_fallback,
    )
    print(json.dumps(summary["comparison"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
