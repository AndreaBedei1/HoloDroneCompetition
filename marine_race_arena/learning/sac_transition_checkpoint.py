"""Atomic, hash-verified SAC checkpoints including replay and n-step state."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from marine_race_arena.learning.config import ACTION_CONTRACT_VERSION
from marine_race_arena.learning.config_local_transition import (
    OBS_DIM_LOCAL_TRANSITION,
    OBS_ENCODING_VERSION_LOCAL_TRANSITION,
)
from marine_race_arena.learning.longrun_checkpoint import (
    atomic_write_json,
    capture_rng_state,
    restore_rng_state,
    sha256_file,
)
from marine_race_arena.learning.sac_replay_buffer import StratifiedReplayBuffer
from marine_race_arena.learning.sac_transition_policy import SACTransitionAgent


SAC_CHECKPOINT_SCHEMA = "universal_transition_sac_checkpoint_v1"


@dataclass(frozen=True)
class ValidSACCheckpoint:
    model_path: Path
    replay_path: Path
    state_path: Path
    manifest_path: Path
    manifest: Dict[str, Any]

    @property
    def timesteps(self) -> int:
        return int(self.manifest["total_environment_transitions"])


def _resolve(path: str | Path, manifest_path: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    cwd = Path.cwd() / value
    return cwd if cwd.exists() else manifest_path.parent / value.name


def atomic_save_sac_checkpoint(
    agent: SACTransitionAgent,
    replay: StratifiedReplayBuffer,
    run_dir: str | Path,
    *,
    total_environment_transitions: int,
    config_contract_sha256: str,
    curriculum_state: Mapping[str, Any],
    n_step_state: Mapping[str, Any],
    evaluation_state: Mapping[str, Any],
    aliases: Mapping[str, Any],
    selection_state: Mapping[str, Any],
    initialization: Mapping[str, Any],
    worker_identities: Iterable[Mapping[str, Any]],
    reason: str,
    status: str = "unverified",
    stop_reason: Optional[str] = None,
) -> ValidSACCheckpoint:
    import torch

    run = Path(run_dir)
    directory = run / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    steps = int(total_environment_transitions)
    stem = f"sac_{steps}_steps"
    model_path = directory / f"{stem}.pt"
    replay_path = directory / f"{stem}.replay.npz"
    state_path = directory / f"{stem}.state.json"
    manifest_path = directory / f"{stem}.manifest.json"
    model_tmp = directory / f".{stem}.partial.pt"
    replay_tmp = directory / f".{stem}.partial.replay.npz"
    state_tmp = directory / f".{stem}.partial.state.json"
    manifest_tmp = directory / f".{stem}.partial.manifest.json"
    for path in (model_tmp, replay_tmp, state_tmp, manifest_tmp):
        path.unlink(missing_ok=True)

    torch.save(
        {
            "schema_version": SAC_CHECKPOINT_SCHEMA,
            "total_environment_transitions": steps,
            "agent": agent.checkpoint_state(),
        },
        model_tmp,
    )
    replay.save(replay_tmp)
    state = {
        "schema_version": SAC_CHECKPOINT_SCHEMA,
        "total_environment_transitions": steps,
        "gradient_updates": int(agent.gradient_updates),
        "entropy_coefficient": float(agent.alpha),
        "rng": capture_rng_state(),
        "curriculum": dict(curriculum_state),
        "n_step": dict(n_step_state),
        "replay": replay.metadata_state(),
        "evaluation": dict(evaluation_state),
        "checkpoint_aliases": dict(aliases),
        "checkpoint_selection_state": dict(selection_state),
        "initialization": dict(initialization),
        "worker_identities": [dict(value) for value in worker_identities],
        "stop_reason": stop_reason,
    }
    state_tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": SAC_CHECKPOINT_SCHEMA,
        "total_environment_transitions": steps,
        "gradient_updates": int(agent.gradient_updates),
        "status": str(status),
        "reason": str(reason),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_tmp),
        "model_bytes": model_tmp.stat().st_size,
        "replay_path": str(replay_path),
        "replay_sha256": sha256_file(replay_tmp),
        "replay_bytes": replay_tmp.stat().st_size,
        "state_path": str(state_path),
        "state_sha256": sha256_file(state_tmp),
        "state_bytes": state_tmp.stat().st_size,
        "config_contract_sha256": str(config_contract_sha256),
        "observation_version": OBS_ENCODING_VERSION_LOCAL_TRANSITION,
        "observation_dim": OBS_DIM_LOCAL_TRANSITION,
        "action_version": ACTION_CONTRACT_VERSION,
        "architecture": "multistep_sac_twin_q",
    }
    manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(model_tmp, model_path)
    os.replace(replay_tmp, replay_path)
    os.replace(state_tmp, state_path)
    # Manifest is the commit record and is always made visible last.
    os.replace(manifest_tmp, manifest_path)
    return ValidSACCheckpoint(
        model_path, replay_path, state_path, manifest_path, manifest
    )


def validate_sac_checkpoint(
    manifest_path: str | Path,
    *,
    expected_contract_sha256: Optional[str] = None,
) -> Optional[ValidSACCheckpoint]:
    import torch

    manifest_file = Path(manifest_path)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SAC_CHECKPOINT_SCHEMA:
            return None
        if expected_contract_sha256 is not None and manifest.get(
            "config_contract_sha256"
        ) != expected_contract_sha256:
            return None
        if manifest.get("observation_version") != OBS_ENCODING_VERSION_LOCAL_TRANSITION:
            return None
        if int(manifest.get("observation_dim", 0)) != OBS_DIM_LOCAL_TRANSITION:
            return None
        if manifest.get("action_version") != ACTION_CONTRACT_VERSION:
            return None
        model_path = _resolve(manifest["model_path"], manifest_file)
        replay_path = _resolve(manifest["replay_path"], manifest_file)
        state_path = _resolve(manifest["state_path"], manifest_file)
        for key, path in (
            ("model", model_path), ("replay", replay_path), ("state", state_path)
        ):
            if not path.exists() or sha256_file(path) != manifest[f"{key}_sha256"]:
                return None
            if path.stat().st_size != int(manifest[f"{key}_bytes"]):
                return None
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        replay = StratifiedReplayBuffer.load(replay_path)
        steps = int(manifest["total_environment_transitions"])
        if int(payload["total_environment_transitions"]) != steps:
            return None
        if int(state["total_environment_transitions"]) != steps:
            return None
        if replay.size != int(state["replay"]["size"]):
            return None
        SACTransitionAgent.from_checkpoint_state(payload["agent"])
        return ValidSACCheckpoint(
            model_path, replay_path, state_path, manifest_file, manifest
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError):
        return None


def valid_sac_checkpoints(
    run_dir: str | Path,
    *,
    expected_contract_sha256: Optional[str] = None,
) -> list[ValidSACCheckpoint]:
    rows = []
    for path in (Path(run_dir) / "checkpoints").glob("sac_*_steps.manifest.json"):
        valid = validate_sac_checkpoint(
            path, expected_contract_sha256=expected_contract_sha256
        )
        if valid is not None:
            rows.append(valid)
    return sorted(rows, key=lambda row: row.timesteps)


def latest_valid_sac_checkpoint(
    run_dir: str | Path,
    *,
    expected_contract_sha256: Optional[str] = None,
) -> Optional[ValidSACCheckpoint]:
    rows = valid_sac_checkpoints(
        run_dir, expected_contract_sha256=expected_contract_sha256
    )
    return rows[-1] if rows else None


def load_sac_checkpoint(
    checkpoint: ValidSACCheckpoint,
) -> tuple[SACTransitionAgent, StratifiedReplayBuffer, Dict[str, Any]]:
    import torch

    payload = torch.load(
        checkpoint.model_path, map_location="cpu", weights_only=False
    )
    agent = SACTransitionAgent.from_checkpoint_state(payload["agent"])
    replay = StratifiedReplayBuffer.load(checkpoint.replay_path)
    state = json.loads(checkpoint.state_path.read_text(encoding="utf-8"))
    restore_rng_state(state["rng"])
    return agent, replay, state

