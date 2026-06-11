"""Visual gate spawning adapters for simulator-specific environments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

from marine_race_arena.arena.gate_factory import GateBar

LOGGER = logging.getLogger(__name__)


class HoloOceanVisualSpawner:
    """Attempt runtime gate-bar spawning and keep exportable metadata as fallback."""

    def __init__(self, env: Any = None, export_path: Optional[str | Path] = None):
        self.env = env
        self.export_path = Path(export_path) if export_path else None
        self.spawned_bar_count = 0
        self.metadata_only = True

    def spawn_gate_bars(self, bars: Iterable[GateBar]) -> None:
        bar_list = list(bars)
        if not bar_list:
            return
        if self._try_supported_runtime_spawn(bar_list):
            self.metadata_only = False
            self.spawned_bar_count = len(bar_list)
            return
        self._export_if_requested(bar_list)
        LOGGER.warning(
            "HoloOcean runtime gate spawning is not available; %d gate bars remain debug metadata.",
            len(bar_list),
        )
        for bar in bar_list:
            LOGGER.debug(
                "Gate bar %s position=%s rotation_rpy_deg=%s dimensions_m=%s color=%s",
                bar.id,
                bar.position,
                bar.rotation_rpy_deg,
                bar.dimensions_m,
                bar.color,
            )

    def spawn_box(
        self,
        id: str,
        position: tuple[float, float, float],
        rotation_rpy_deg: tuple[float, float, float],
        dimensions_m: tuple[float, float, float],
        color: Any,
    ) -> None:
        self.spawn_gate_bars(
            [
                GateBar(
                    id=id,
                    gate_id=id.split("_", 1)[0],
                    part=id.rsplit("_", 1)[-1],
                    position=position,
                    rotation_rpy_deg=rotation_rpy_deg,
                    dimensions_m=dimensions_m,
                    color=color,
                )
            ]
        )

    def _try_supported_runtime_spawn(self, bars: list[GateBar]) -> bool:
        if self.env is None:
            return False
        if hasattr(self.env, "spawn_gate_bars"):
            self.env.spawn_gate_bars(bars)
            return True
        for method_name in ("spawn_box", "spawn_object", "add_object"):
            method = getattr(self.env, method_name, None)
            if not callable(method):
                continue
            for bar in bars:
                try:
                    method(
                        id=bar.id,
                        position=bar.position,
                        rotation_rpy_deg=bar.rotation_rpy_deg,
                        dimensions_m=bar.dimensions_m,
                        color=bar.color,
                    )
                except TypeError:
                    try:
                        method(bar.id, bar.position, bar.rotation_rpy_deg, bar.dimensions_m)
                    except Exception as exc:
                        LOGGER.warning("Gate bar spawn via %s failed: %s", method_name, exc)
                        return False
                except Exception as exc:
                    LOGGER.warning("Gate bar spawn via %s failed: %s", method_name, exc)
                    return False
            return True
        return False

    def _export_if_requested(self, bars: list[GateBar]) -> None:
        if self.export_path is None:
            return
        self.export_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "id": bar.id,
                "gate_id": bar.gate_id,
                "part": bar.part,
                "position": list(bar.position),
                "rotation_rpy_deg": list(bar.rotation_rpy_deg),
                "dimensions_m": list(bar.dimensions_m),
                "color": bar.color,
            }
            for bar in bars
        ]
        with self.export_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        LOGGER.info("Exported gate visual metadata to %s.", self.export_path)

