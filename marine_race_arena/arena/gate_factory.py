"""Gate visual representation factory.

The referee uses abstract gate geometry. Visual bars are optional and are only
spawned when a repository-specific simulator adapter exposes a compatible method.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from marine_race_arena.arena.gate import Gate
from marine_race_arena.config.schema import TrackConfig, Vector3

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateBar:
    id: str
    gate_id: str
    part: str
    position: Vector3
    rotation_rpy_deg: Vector3
    dimensions_m: Vector3
    color: Any


@dataclass(frozen=True)
class VisualGate:
    gate_id: str
    bars: List[GateBar]


class GateFactory:
    """Build abstract gates and optional four-bar visual representations."""

    def __init__(self, config: TrackConfig):
        self.config = config

    def build_gates(self) -> List[Gate]:
        return [Gate.from_config(gate_config) for gate_config in self.config.gates]

    def build_visual_gate(self, gate: Gate) -> VisualGate:
        depth = float(self.config.track.gate_depth_m)
        thickness = gate.bar_thickness_m
        width = gate.inner_width_m
        height = gate.inner_height_m
        right = gate.right_axis
        up = gate.up_axis

        bars = [
            GateBar(
                id=f"{gate.id}_top",
                gate_id=gate.id,
                part="top",
                position=_add(gate.center, _scale(up, height / 2.0 + thickness / 2.0)),
                rotation_rpy_deg=gate.rotation_rpy_deg,
                dimensions_m=(depth, width + 2.0 * thickness, thickness),
                color=gate.color,
            ),
            GateBar(
                id=f"{gate.id}_bottom",
                gate_id=gate.id,
                part="bottom",
                position=_add(gate.center, _scale(up, -height / 2.0 - thickness / 2.0)),
                rotation_rpy_deg=gate.rotation_rpy_deg,
                dimensions_m=(depth, width + 2.0 * thickness, thickness),
                color=gate.color,
            ),
            GateBar(
                id=f"{gate.id}_left",
                gate_id=gate.id,
                part="left",
                position=_add(gate.center, _scale(right, -width / 2.0 - thickness / 2.0)),
                rotation_rpy_deg=gate.rotation_rpy_deg,
                dimensions_m=(depth, thickness, height),
                color=gate.color,
            ),
            GateBar(
                id=f"{gate.id}_right",
                gate_id=gate.id,
                part="right",
                position=_add(gate.center, _scale(right, width / 2.0 + thickness / 2.0)),
                rotation_rpy_deg=gate.rotation_rpy_deg,
                dimensions_m=(depth, thickness, height),
                color=gate.color,
            ),
        ]
        return VisualGate(gate_id=gate.id, bars=bars)

    def build_visual_gates(self, gates: Iterable[Gate]) -> List[VisualGate]:
        return [self.build_visual_gate(gate) for gate in gates]

    def spawn_visuals(self, visual_gates: Iterable[VisualGate], spawner: Optional[Any] = None) -> None:
        """Spawn visual gate bars when a simulator adapter is available.

        Supported optional adapter methods:
        - spawn_gate_bars(list[GateBar])
        - spawn_box(id=..., position=..., rotation_rpy_deg=..., dimensions_m=..., color=...)
        """

        bars = [bar for visual_gate in visual_gates for bar in visual_gate.bars]
        if spawner is None:
            LOGGER.warning(
                "Physical visual gate spawning is not implemented for this repository context; "
                "using abstract gate geometry and debug bar metadata only."
            )
            return
        if hasattr(spawner, "spawn_gate_bars"):
            spawner.spawn_gate_bars(bars)
            return
        if hasattr(spawner, "spawn_box"):
            for bar in bars:
                spawner.spawn_box(
                    id=bar.id,
                    position=bar.position,
                    rotation_rpy_deg=bar.rotation_rpy_deg,
                    dimensions_m=bar.dimensions_m,
                    color=bar.color,
                )
            return
        LOGGER.warning(
            "The provided spawner has no supported gate spawning method; "
            "using debug bar metadata only."
        )


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(vector: Vector3, scalar: float) -> Vector3:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)

