"""Validate or create observation-v3 model artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

from marine_race_arena.learning.bc_v3_transfer import (
    load_v3_policy,
    transfer_file,
)
from marine_race_arena.learning.config import ACTION_CONTRACT_VERSION
from marine_race_arena.learning.config_v3 import OBS_DIM_V3, OBS_ENCODING_VERSION_V3
from marine_race_arena.learning.provenance import now_utc, sha256_file

FROZEN_BC_V1_SHA256 = (
    "fd6fc7e6fba9b88ccc84fa76275d72ce3c907c75d632367995cb5257ae71d5d3"
)


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def assert_clean_worktree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "RL experiments require a clean committed SHA; commit or restore "
            "the listed worktree changes first."
        )


def validate_v3_model(path: Any) -> Dict[str, Any]:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if model_path.suffix.lower() == ".zip":
        from stable_baselines3 import PPO

        model = PPO.load(str(model_path), device="cpu")
        version = getattr(model, "obs_encoding_version", None)
        shape = tuple(getattr(model.observation_space, "shape", ()) or ())
        if version != OBS_ENCODING_VERSION_V3 or shape != (OBS_DIM_V3,):
            raise ValueError(
                f"incompatible PPO: version={version!r}, shape={shape}; "
                f"expected {OBS_ENCODING_VERSION_V3!r}, {(OBS_DIM_V3,)}"
            )
        kind = "ppo"
    else:
        policy = load_v3_policy(model_path)
        if policy.obs_dim != OBS_DIM_V3:
            raise ValueError("v3 BC model dimension mismatch")
        kind = "bc"
    return {
        "kind": kind,
        "path": str(model_path),
        "bytes": model_path.stat().st_size,
        "sha256": sha256_file(model_path),
        "observation_encoding_version": OBS_ENCODING_VERSION_V3,
        "observation_dim": OBS_DIM_V3,
        "action_contract_version": ACTION_CONTRACT_VERSION,
    }


def create_transfer(
    source: Any,
    destination: Any,
    *,
    expected_source_sha256: str = FROZEN_BC_V1_SHA256,
) -> Dict[str, Any]:
    assert_clean_worktree()
    source_path = Path(source)
    destination_path = Path(destination)
    source_sha = sha256_file(source_path)
    if expected_source_sha256 and source_sha != expected_source_sha256:
        raise ValueError(
            f"frozen BC-v1 hash mismatch: {source_sha} != {expected_source_sha256}"
        )
    transfer_file(source_path, destination_path)
    model = validate_v3_model(destination_path)
    manifest = {
        "schema_version": "multigate_v3_transfer_v1",
        "created_utc": now_utc(),
        "code_sha": _git_sha(),
        "source_model": {
            "path": str(source_path),
            "sha256": source_sha,
            "observation_encoding_version": "onboard_only_v1",
            "observation_dim": 36,
            "frozen": True,
        },
        "destination_model": model,
        "transfer": {
            "copied_v1_input_columns": 36,
            "new_input_columns_initialized_to_zero": OBS_DIM_V3 - 36,
            "hidden_and_action_layers_copied": True,
            "neutral_feature_output_parity": True,
        },
    }
    manifest_path = destination_path.with_suffix(
        destination_path.suffix + ".transfer.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--model", required=True)
    transfer = sub.add_parser("transfer")
    transfer.add_argument(
        "--source",
        default="results/rl_public/stage1/bc/model/best_model.pt",
    )
    transfer.add_argument(
        "--destination",
        default="results/rl/multigate_v3/models/bc_v1_transfer_v3.pt",
    )
    transfer.add_argument(
        "--expected-source-sha256",
        default=FROZEN_BC_V1_SHA256,
    )
    args = parser.parse_args(argv)
    if args.command == "validate":
        result = validate_v3_model(args.model)
    else:
        result = create_transfer(
            args.source,
            args.destination,
            expected_source_sha256=args.expected_source_sha256,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
