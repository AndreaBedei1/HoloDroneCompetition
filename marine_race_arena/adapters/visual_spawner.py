"""Visual gate spawning adapters for simulator-specific environments."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from marine_race_arena.arena.gate_factory import GateBar

LOGGER = logging.getLogger(__name__)


@dataclass
class VisualSpawnReport:
    physically_spawned: bool = False
    spawned_bar_count: int = 0
    method: str = "metadata_only"
    message: str = "No visual gate spawning attempted."
    export_path: Optional[str] = None


class HoloOceanVisualSpawner:
    """Attempt runtime gate-bar spawning and keep exportable metadata as fallback."""

    def __init__(self, env: Any = None, export_path: Optional[str | Path] = None):
        self.env = env
        self.export_path = Path(export_path) if export_path else None
        self.spawned_bar_count = 0
        self.metadata_only = True
        self.report = VisualSpawnReport()

    def spawn_gate_bars(self, bars: Iterable[GateBar]) -> None:
        bar_list = list(bars)
        if not bar_list:
            self.report = VisualSpawnReport(message="No gate bars were provided.")
            return
        if self._try_holoocean_spawn_prop(bar_list):
            self.metadata_only = False
            self.spawned_bar_count = len(bar_list)
            self.report = VisualSpawnReport(
                physically_spawned=True,
                spawned_bar_count=len(bar_list),
                method="runtime_spawn_prop",
                message="Gate bars spawned with HoloOcean env.spawn_prop('box', ...).",
            )
            return
        exported = self._export_if_requested(bar_list)
        method = "export_only" if exported else "metadata_only"
        message = (
            f"Gate bars exported to {self.export_path} for external placement."
            if exported
            else "HoloOcean runtime gate spawning is not available; gate bars remain metadata only."
        )
        self.report = VisualSpawnReport(
            physically_spawned=False,
            spawned_bar_count=0,
            method=method,
            message=message,
            export_path=str(self.export_path) if exported and self.export_path else None,
        )
        LOGGER.warning(
            "%s %d requested gate bars were not physically spawned.",
            message,
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

    def _try_holoocean_spawn_prop(self, bars: list[GateBar]) -> bool:
        if self.env is None:
            return False
        spawn_prop = getattr(self.env, "spawn_prop", None)
        if not callable(spawn_prop):
            return False
        for bar in bars:
            try:
                spawn_prop(
                    "box",
                    location=list(bar.position),
                    rotation=list(bar.rotation_rpy_deg),
                    scale=list(bar.dimensions_m),
                    sim_physics=False,
                    material=_material_from_color(bar.color),
                    tag=bar.id,
                )
            except Exception as exc:
                LOGGER.warning("Gate bar spawn_prop failed for %s: %s", bar.id, exc)
                return False
        return True

    def _export_if_requested(self, bars: list[GateBar]) -> bool:
        if self.export_path is None:
            return False
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
        return True


def _material_from_color(color: Any) -> str:
    if isinstance(color, str):
        lowered = color.lower()
        if lowered in {"white", "gold", "cobblestone", "brick", "wood", "grass", "steel", "black"}:
            return lowered
    return "white"
