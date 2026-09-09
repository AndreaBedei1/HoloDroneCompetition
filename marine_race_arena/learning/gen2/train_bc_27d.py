"""Short episode-balanced recurrent BC on the approved 27-D corpus.

This is deliberately a BC-only entry point: it constructs the 27-D policy by
semantic warm-start from the existing 35-D parent and never calls PPO.learn.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

from marine_race_arena.learning.gen2.bc_recurrent import BCConfig, split_episodes, train_recurrent_bc
from marine_race_arena.learning.gen2.dataset_27d import corpus_statistics, load_corpus
from marine_race_arena.learning.gen2.transfer_27d import transfer_parent_35d_to_27d, verify_transfer_27d_tensors


def _balanced_episode_list(episodes):
    """Oversample short episodes to keep raw frame count from dominating."""
    if not episodes:
        return []
    target = max(1, int(sorted(len(e) for e in episodes)[len(episodes) // 2]))
    balanced = []
    for episode in episodes:
        repeats = max(1, int(math.ceil(target / max(1, len(episode)))))
        balanced.extend([episode] * repeats)
    return balanced


def run(*, corpus_root: str | Path, parent: str | Path, output: str | Path, seed: int = 27001) -> dict:
    root = Path(corpus_root)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    episodes = load_corpus(root)
    if len(episodes) < 10 or not all(e.completed for e in episodes):
        raise RuntimeError("BC requires the verified completed extended corpus")
    train, validation = split_episodes(episodes, 0.15, seed)
    balanced = _balanced_episode_list(train)
    model = transfer_parent_35d_to_27d(parent, seed=seed, device="cpu", learning_rate=3e-5, n_steps=256, batch_size=128, n_epochs=2)
    transfer_audit = verify_transfer_27d_tensors(parent, model)
    parent_path = out / "parent_warmstart_27d.zip"
    model.save(str(parent_path))
    config = BCConfig(
        chunk_length=64, batch_episodes=8, epochs=6, learning_rate=3e-5,
        grad_clip=1.0, validation_fraction=0.15, patience=2, seed=seed,
        device="cpu", train_value_head=True,
    )
    result = train_recurrent_bc(
        model, balanced, config, latch_normalization=True,
        progress_path=out / "progress.json", validation_episodes=validation,
    )
    bc_path = out / "bc_policy.zip"
    model.save(str(bc_path))
    report = {
        "corpus": corpus_statistics(episodes),
        "train_episodes_unique": len(train),
        "validation_episodes": len(validation),
        "balanced_train_episode_instances": len(balanced),
        "balancing_target_steps": int(sorted(len(e) for e in train)[len(train) // 2]),
        "parent_checkpoint": str(parent),
        "parent_warmstart_27d": str(parent_path),
        "bc_checkpoint": str(bc_path),
        "transfer_audit": transfer_audit,
        "result": result.as_dict(),
        "ppo_started": False,
    }
    (out / "bc_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run short recurrent BC on the verified 27-D corpus")
    parser.add_argument("--corpus-root", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=27001)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run(corpus_root=args.corpus_root, parent=args.parent, output=args.output, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
