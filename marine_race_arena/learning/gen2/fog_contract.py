"""Fail-fast underwater-fog contract for every Gen-2 PPO source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


APPROVED_WATER_FOG: Dict[str, Any] = {
    "enabled": True,
    "density": 5.0,
    "start_distance_m": 1.0,
    "color_rgb": [0.4, 0.6, 1.0],
}


def read_water_fog(track_path: str | Path) -> Mapping[str, Any]:
    raw = json.loads(Path(track_path).read_text(encoding="utf-8"))
    value = raw.get("water_fog")
    return value if isinstance(value, Mapping) else {}


def assert_approved_water_fog(track_path: str | Path) -> Dict[str, Any]:
    """Return the actual values or fail before a simulator/trainer is built."""

    path = Path(track_path)
    actual = dict(read_water_fog(path))
    if actual != APPROVED_WATER_FOG:
        raise ValueError(
            f"fog mismatch in {path}: actual={actual!r}, "
            f"required={APPROVED_WATER_FOG!r}"
        )
    return {"track_path": str(path), "water_fog": actual, "valid": True}


def verify_fog_sources(track_paths: Iterable[str | Path]) -> Dict[str, Any]:
    rows = [assert_approved_water_fog(path) for path in track_paths]
    if not rows:
        raise ValueError("at least one PPO source track is required")
    return {
        "required": dict(APPROVED_WATER_FOG),
        "sources": rows,
        "all_valid": True,
    }

