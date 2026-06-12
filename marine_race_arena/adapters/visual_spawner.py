"""Visual gate spawning adapters for simulator-specific environments."""

from __future__ import annotations

import json
import logging
import math
import os
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

    def __init__(self, env: Any = None, export_path: Optional[str | Path] = None, mode: Optional[str] = None):
        self.env = env
        self.export_path = Path(export_path) if export_path else None
        self.mode = (mode or os.getenv("MARINE_RACE_GATE_VISUAL_MODE", "uniform")).strip().lower()
        self.spawned_bar_count = 0
        self.metadata_only = True
        self.report = VisualSpawnReport()
        self.spawned_props: list[dict[str, Any]] = []

    def spawn_gate_bars(self, bars: Iterable[GateBar]) -> None:
        bar_list = list(bars)
        if not bar_list:
            self.report = VisualSpawnReport(message="No gate bars were provided.")
            return
        if self._try_holoocean_spawn_prop(bar_list):
            self.metadata_only = False
            self.spawned_bar_count = len(self.spawned_props)
            self.report = VisualSpawnReport(
                physically_spawned=True,
                spawned_bar_count=len(self.spawned_props),
                method=(
                    "runtime_spawn_prop_segmented_cubes"
                    if self.mode == "segmented"
                    else "runtime_spawn_prop_uniform_box"
                ),
                message=(
                    "Gate visuals spawned with HoloOcean env.spawn_prop('box', ...) "
                    + (
                        "using segmented cubes along each logical gate bar."
                        if self.mode == "segmented"
                        else "using one uniform box per logical gate bar."
                    )
                ),
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
        self.spawned_props = []
        for bar in bars:
            if self.mode == "segmented":
                if not self._spawn_segmented_bar(spawn_prop, bar):
                    return False
                continue
            if not self._spawn_uniform_bar(spawn_prop, bar):
                return False
        return True

    def _spawn_uniform_bar(self, spawn_prop: Any, bar: GateBar) -> bool:
        try:
            spawn_rotation = _holoocean_spawn_prop_rotation(bar.rotation_rpy_deg)
            spawn_prop(
                "box",
                location=list(bar.position),
                rotation=list(spawn_rotation),
                scale=list(bar.dimensions_m),
                sim_physics=False,
                material=_material_from_color(bar.color),
                tag=bar.id,
            )
            self.spawned_props.append(
                {
                    "id": bar.id,
                    "source_bar_id": bar.id,
                    "gate_id": bar.gate_id,
                    "part": bar.part,
                    "position": bar.position,
                    "rotation_rpy_deg": bar.rotation_rpy_deg,
                    "spawn_rotation_deg": spawn_rotation,
                    "spawn_rotation_order": "holoocean_spawn_prop_yaw_pitch_roll",
                    "dimensions_m": bar.dimensions_m,
                    "method": "uniform_four_bar_box",
                }
            )
            return True
        except Exception as exc:
            LOGGER.warning("Gate bar spawn_prop failed for %s: %s", bar.id, exc)
            return False

    def _spawn_segmented_bar(self, spawn_prop: Any, bar: GateBar) -> bool:
        for segment in _segment_gate_bar(bar):
            try:
                # HoloOcean SpawnProp can interpret elongated rotated boxes
                # differently across builds. Small axis-aligned cubes follow
                # the mathematical bar centerline without relying on Euler
                # order, so yaw and pitch cannot lay pillars on their side.
                spawn_rotation = (0.0, 0.0, 0.0)
                spawn_tag = f"{bar.id}_s{segment['index']:02d}"
                spawn_position = segment["position"]
                spawn_dimensions = segment["dimensions_m"]
                spawn_prop(
                    "box",
                    location=list(spawn_position),
                    rotation=list(spawn_rotation),
                    scale=list(spawn_dimensions),
                    sim_physics=False,
                    material=_material_from_color(bar.color),
                    tag=spawn_tag,
                )
                self.spawned_props.append(
                    {
                        "id": spawn_tag,
                        "source_bar_id": bar.id,
                        "gate_id": bar.gate_id,
                        "part": bar.part,
                        "segment_index": segment["index"],
                        "segment_count": segment["count"],
                        "position": spawn_position,
                        "rotation_rpy_deg": bar.rotation_rpy_deg,
                        "spawn_rotation_deg": spawn_rotation,
                        "spawn_rotation_order": "none_axis_aligned_segment",
                        "dimensions_m": spawn_dimensions,
                        "logical_bar_dimensions_m": bar.dimensions_m,
                        "method": "segmented_axis_aligned_cube",
                    }
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


def _holoocean_spawn_prop_rotation(rotation_rpy_deg: tuple[float, float, float]) -> tuple[float, float, float]:
    roll, pitch, yaw = (float(value) for value in rotation_rpy_deg)
    # HoloOcean 2.3.0 documents SpawnProp rotation as [roll, pitch, yaw], but
    # the tested Unreal backend renders box yaw correctly when the yaw angle is
    # sent in the first component. Keep the internal gate frame as RPY and apply
    # this runtime-specific conversion only at the HoloOcean boundary.
    return (yaw, pitch, roll)


def _segment_gate_bar(bar: GateBar) -> list[dict[str, Any]]:
    axes = _rotation_axes(bar.rotation_rpy_deg)
    if bar.part in {"top", "bottom"}:
        long_axis = axes[1]
        length_m = float(bar.dimensions_m[1])
    elif bar.part in {"left", "right"}:
        long_axis = axes[2]
        length_m = float(bar.dimensions_m[2])
    else:
        dimensions = [float(value) for value in bar.dimensions_m]
        axis_index = max(range(3), key=lambda index: dimensions[index])
        long_axis = axes[axis_index]
        length_m = dimensions[axis_index]

    cube_size_m = max(
        float(bar.dimensions_m[0]),
        min(float(bar.dimensions_m[1]), float(bar.dimensions_m[2])),
        0.12,
    )
    step_m = cube_size_m * 0.75
    segment_count = max(2, int(math.ceil(length_m / step_m)) + 1)
    offsets = [
        -length_m / 2.0 + index * (length_m / float(segment_count - 1))
        for index in range(segment_count)
    ]
    return [
        {
            "index": index,
            "count": segment_count,
            "position": _add(bar.position, _scale(long_axis, offset)),
            "dimensions_m": (cube_size_m, cube_size_m, cube_size_m),
        }
        for index, offset in enumerate(offsets)
    ]


def _rotation_axes(
    rotation_rpy_deg: tuple[float, float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    roll, pitch, yaw = [math.radians(float(value)) for value in rotation_rpy_deg]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    matrix = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(vector: tuple[float, float, float], scalar: float) -> tuple[float, float, float]:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)
