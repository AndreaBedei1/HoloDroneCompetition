"""Arena assembly from a validated track config."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from marine_race_arena.arena.beacon_manager import BeaconManager
from marine_race_arena.arena.bounds import ArenaBounds
from marine_race_arena.arena.currents import CurrentFieldManager
from marine_race_arena.arena.gate import Gate
from marine_race_arena.arena.gate_factory import GateFactory, VisualGate
from marine_race_arena.arena.obstacle import Obstacle, resolve_active_obstacles
from marine_race_arena.config.schema import TrackConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class Arena:
    config: TrackConfig
    bounds: ArenaBounds
    gates: List[Gate]
    gate_map: Dict[str, Gate]
    visual_gates: List[VisualGate]
    obstacles: List[Obstacle]
    beacon_manager: BeaconManager
    current_manager: CurrentFieldManager
    environment_name: str


class ArenaBuilder:
    def __init__(self, config: TrackConfig, seed: Optional[int] = None):
        self.config = config
        self.seed = seed

    def build(self, visual_spawner: object | None = None) -> Arena:
        factory = GateFactory(self.config)
        gates = factory.build_gates()
        visual_gates = factory.build_visual_gates(gates)
        if visual_spawner is not None:
            factory.spawn_visuals(visual_gates, spawner=visual_spawner)

        obstacles = resolve_active_obstacles(self.config)

        gate_map = {gate.id: gate for gate in gates}
        environment_name = self._select_environment_name()
        return Arena(
            config=self.config,
            bounds=ArenaBounds.from_config(self.config.world.bounds),
            gates=gates,
            gate_map=gate_map,
            visual_gates=visual_gates,
            obstacles=obstacles,
            beacon_manager=BeaconManager.from_gates(gates, self.config.gates, seed=self.seed),
            current_manager=CurrentFieldManager(self.config.currents),
            environment_name=environment_name,
        )

    def _select_environment_name(self) -> str:
        preferred = self.config.world.preferred_environment or "OpenWater-Hovering"
        fallback = self.config.world.fallback_environment or "PierHarbor-Hovering"
        if self.config.world.map:
            return self.config.world.map
        LOGGER.warning("No explicit world.map configured; using preferred environment '%s'.", preferred)
        return preferred or fallback
