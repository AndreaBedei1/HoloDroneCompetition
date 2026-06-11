"""Race simulator adapters."""

from __future__ import annotations

import logging
from typing import Optional

from marine_race_arena.adapters.base import BaseRaceAdapter, RaceAdapterError, RaceAdapterUnavailable
from marine_race_arena.adapters.fallback_adapter import FallbackRaceAdapter
from marine_race_arena.adapters.holoocean_adapter import HoloOceanRaceAdapter
from marine_race_arena.arena.arena_builder import Arena
from marine_race_arena.config.schema import TrackConfig

LOGGER = logging.getLogger(__name__)


class AdapterSelectionError(RaceAdapterError):
    """Raised when the requested adapter cannot be selected."""


def select_adapter(
    adapter_name: str,
    config: TrackConfig,
    arena: Arena,
    allow_fallback: bool,
    headless: bool = False,
    record: bool = False,
    seed: Optional[int] = None,
) -> BaseRaceAdapter:
    """Create and minimally initialize an adapter according to CLI policy."""

    normalized = adapter_name.lower()
    if normalized == "fallback":
        adapter = FallbackRaceAdapter(config, arena, seed=seed, headless=headless, record=record)
        adapter.initialize()
        return adapter
    if normalized not in {"auto", "holoocean"}:
        raise AdapterSelectionError(f"Unknown race adapter '{adapter_name}'.")

    try:
        adapter = HoloOceanRaceAdapter(config, arena, seed=seed, headless=headless, record=record)
        adapter.initialize()
        return adapter
    except RaceAdapterUnavailable as exc:
        if allow_fallback:
            LOGGER.warning("HoloOcean adapter is unavailable; falling back because --allow-fallback is set: %s", exc)
            adapter = FallbackRaceAdapter(config, arena, seed=seed, headless=headless, record=record)
            adapter.initialize()
            return adapter
        raise AdapterSelectionError(
            "HoloOcean adapter is unavailable and fallback is not allowed. "
            "Use --adapter fallback for the kinematic runner or pass --allow-fallback explicitly."
        ) from exc


__all__ = [
    "AdapterSelectionError",
    "BaseRaceAdapter",
    "FallbackRaceAdapter",
    "HoloOceanRaceAdapter",
    "RaceAdapterError",
    "RaceAdapterUnavailable",
    "select_adapter",
]

