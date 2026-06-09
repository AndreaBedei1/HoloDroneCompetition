"""Obstacle placeholder definitions.

Obstacle physical spawning is intentionally adapter-driven because HoloOcean and
Unreal object APIs vary by repository. Configured obstacles are preserved so a
future adapter can spawn them without changing the track schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class Obstacle:
    id: str
    type: str
    params: Dict[str, Any]

