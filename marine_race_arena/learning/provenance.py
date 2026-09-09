"""Shared provenance helpers for reproducible RL runs (no heavy dependencies)."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def git_sha() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover - git absent
        return None


def package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("numpy", "torch", "gymnasium", "stable_baselines3", "holoocean"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = None
    return versions


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
