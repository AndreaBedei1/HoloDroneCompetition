"""Atomic PPO checkpoints with hash-verified fallback and full resume state."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pickle
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from marine_race_arena.learning.config import ACTION_CONTRACT_VERSION
from marine_race_arena.learning.config_v3 import OBS_ENCODING_VERSION_V3

CHECKPOINT_SCHEMA_VERSION = "multigate_longrun_checkpoint_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _model_observation_dim(model: Any) -> int:
    space = getattr(model, "observation_space", None)
    shape = getattr(space, "shape", (0,)) or (0,)
    return int(shape[0])


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, target)


def atomic_append_jsonl(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _pickle64(value: Any) -> str:
    return base64.b64encode(pickle.dumps(value, protocol=4)).decode("ascii")


def _unpickle64(value: str) -> Any:
    return pickle.loads(base64.b64decode(value.encode("ascii")))


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random": _pickle64(random.getstate()),
        "numpy_legacy": _pickle64(np.random.get_state()),
    }
    try:
        import torch

        state["torch_cpu"] = _pickle64(torch.get_rng_state())
        if torch.cuda.is_available():
            state["torch_cuda"] = _pickle64(torch.cuda.get_rng_state_all())
    except Exception:
        state["torch_cpu"] = None
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(_unpickle64(state["python_random"]))
    np.random.set_state(_unpickle64(state["numpy_legacy"]))
    try:
        import torch

        if state.get("torch_cpu"):
            torch.set_rng_state(_unpickle64(state["torch_cpu"]))
        if state.get("torch_cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(_unpickle64(state["torch_cuda"]))
    except Exception:
        pass


@dataclass(frozen=True)
class ValidCheckpoint:
    model_path: Path
    state_path: Path
    manifest_path: Path
    manifest: Dict[str, Any]

    @property
    def timesteps(self) -> int:
        return int(self.manifest["total_timesteps"])


def atomic_save_checkpoint(
    model: Any,
    run_dir: str | Path,
    *,
    total_timesteps: int,
    config_contract_hash: str,
    curriculum_state: Dict[str, Any],
    evaluation_state: Dict[str, Any],
    status: str = "safe",
    reason: str = "periodic",
    extra_state: Optional[Dict[str, Any]] = None,
) -> ValidCheckpoint:
    """Save model, optimizer, schedule, RNG, curriculum, and selection state.

    SB3 includes the policy, optimizer and algorithm counters in its ZIP.  The
    companion state persists pipeline state.  The manifest is replaced last, so
    readers never treat a partial model/state pair as valid.
    """
    run_path = Path(run_dir)
    checkpoint_dir = run_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ppo_{int(total_timesteps)}_steps"
    model_path = checkpoint_dir / f"{stem}.zip"
    state_path = checkpoint_dir / f"{stem}.state.json"
    manifest_path = checkpoint_dir / f"{stem}.manifest.json"
    model_tmp = checkpoint_dir / f".{stem}.partial.zip"

    if model_tmp.exists():
        model_tmp.unlink()
    model.save(str(model_tmp))
    # SB3 appends .zip only when absent; the explicit suffix keeps this path exact.
    if not model_tmp.exists():
        appended = Path(str(model_tmp) + ".zip")
        if appended.exists():
            model_tmp = appended
    model_sha = sha256_file(model_tmp)
    state = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "total_timesteps": int(total_timesteps),
        "rng": capture_rng_state(),
        "curriculum": curriculum_state,
        "evaluation": evaluation_state,
        "extra": dict(extra_state or {}),
    }
    state_tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    state_tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    state_sha = sha256_file(state_tmp)
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "total_timesteps": int(total_timesteps),
        "status": str(status),
        "reason": str(reason),
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "model_bytes": int(model_tmp.stat().st_size),
        "state_path": str(state_path),
        "state_sha256": state_sha,
        "config_contract_sha256": config_contract_hash,
        "observation_version": OBS_ENCODING_VERSION_V3,
        "policy_mode": getattr(model, "longrun_policy_mode", "feedforward"),
        "frame_stack": int(getattr(model, "longrun_frame_stack", 1)),
        "policy_observation_dim": _model_observation_dim(model),
        "action_version": ACTION_CONTRACT_VERSION,
    }
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(model_tmp, model_path)
    os.replace(state_tmp, state_path)
    os.replace(manifest_tmp, manifest_path)
    return ValidCheckpoint(model_path, state_path, manifest_path, manifest)


def _checkpoint_is_valid(
    manifest_path: Path,
    *,
    expected_contract_hash: Optional[str] = None,
) -> Optional[ValidCheckpoint]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            return None
        if (
            expected_contract_hash is not None
            and manifest.get("config_contract_sha256") != expected_contract_hash
        ):
            return None
        if manifest.get("observation_version") != OBS_ENCODING_VERSION_V3:
            return None
        if manifest.get("action_version") != ACTION_CONTRACT_VERSION:
            return None
        model_path = Path(manifest["model_path"])
        state_path = Path(manifest["state_path"])
        if not model_path.is_absolute():
            # Paths are normally repository-relative. Resolve from cwd first and
            # then beside the manifest for relocatable test fixtures.
            cwd_path = Path.cwd() / model_path
            model_path = cwd_path if cwd_path.exists() else manifest_path.parent / model_path.name
        if not state_path.is_absolute():
            cwd_path = Path.cwd() / state_path
            state_path = cwd_path if cwd_path.exists() else manifest_path.parent / state_path.name
        if not model_path.exists() or not state_path.exists():
            return None
        if sha256_file(model_path) != manifest["model_sha256"]:
            return None
        if sha256_file(state_path) != manifest["state_sha256"]:
            return None
        with zipfile.ZipFile(model_path, "r") as archive:
            if archive.testzip() is not None:
                return None
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if int(state["total_timesteps"]) != int(manifest["total_timesteps"]):
            return None
        return ValidCheckpoint(model_path, state_path, manifest_path, manifest)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, zipfile.BadZipFile):
        return None


def valid_checkpoints(
    run_dir: str | Path,
    *,
    expected_contract_hash: Optional[str] = None,
) -> Iterable[ValidCheckpoint]:
    checkpoint_dir = Path(run_dir) / "checkpoints"
    rows = []
    for manifest_path in checkpoint_dir.glob("ppo_*_steps.manifest.json"):
        valid = _checkpoint_is_valid(
            manifest_path, expected_contract_hash=expected_contract_hash
        )
        if valid is not None:
            rows.append(valid)
    return sorted(rows, key=lambda checkpoint: checkpoint.timesteps)


def latest_valid_checkpoint(
    run_dir: str | Path,
    *,
    expected_contract_hash: Optional[str] = None,
    safe_only: bool = True,
) -> Optional[ValidCheckpoint]:
    rows = list(
        valid_checkpoints(run_dir, expected_contract_hash=expected_contract_hash)
    )
    if safe_only:
        rows = [row for row in rows if row.manifest.get("status") == "safe"]
    return rows[-1] if rows else None


def load_checkpoint_state(checkpoint: ValidCheckpoint) -> Dict[str, Any]:
    return json.loads(checkpoint.state_path.read_text(encoding="utf-8"))


def atomic_copy_checkpoint(
    checkpoint: ValidCheckpoint,
    destination: str | Path,
) -> Path:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix != ".zip":
        target = target.with_suffix(".zip")
    tmp = target.with_suffix(".partial.zip")
    shutil.copyfile(checkpoint.model_path, tmp)
    if sha256_file(tmp) != checkpoint.manifest["model_sha256"]:
        tmp.unlink(missing_ok=True)
        raise OSError("checkpoint copy hash mismatch")
    os.replace(tmp, target)
    atomic_write_json(
        target.with_suffix(".json"),
        {
            "source_manifest": str(checkpoint.manifest_path),
            "total_timesteps": checkpoint.timesteps,
            "sha256": checkpoint.manifest["model_sha256"],
        },
    )
    return target
