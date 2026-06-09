"""Beacon manager that exposes competition-safe acoustic observations."""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, Optional

from marine_race_arena.arena.beacon import Beacon
from marine_race_arena.arena.gate import Gate
from marine_race_arena.config.schema import GateConfig, Vector3


class BeaconManager:
    def __init__(self, beacons_by_gate_id: Dict[str, Beacon], seed: Optional[int] = None):
        self.beacons_by_gate_id = dict(beacons_by_gate_id)
        self.rng = random.Random(seed)

    @classmethod
    def from_gates(
        cls,
        gates: Iterable[Gate],
        gate_configs: Iterable[GateConfig],
        seed: Optional[int] = None,
    ) -> "BeaconManager":
        configs_by_id = {gate_config.id: gate_config for gate_config in gate_configs}
        beacons: Dict[str, Beacon] = {}
        for gate in gates:
            gate_config = configs_by_id[gate.id]
            beacon = Beacon.from_gate(gate, gate_config.beacon)
            if beacon is not None:
                beacons[gate.id] = beacon
        return cls(beacons, seed=seed)

    def observe(
        self,
        receiver_position: Vector3,
        receiver_yaw_deg: float,
        target_gate_id: str,
        target_sequence_index: int,
        observation_mode: str,
        official_mode: bool,
    ) -> Dict[str, Any]:
        safe_mode = observation_mode
        if official_mode and safe_mode == "oracle":
            safe_mode = "acoustic_noisy"
        beacon = self.beacons_by_gate_id.get(target_gate_id)
        if beacon is None:
            return {
                "valid": False,
                "reason": "missing_beacon",
                "active_beacon_id": None,
                "target_gate_id": target_gate_id,
                "sequence_index": target_sequence_index,
                "bearing_deg": None,
                "elevation_deg": None,
                "range_m": None,
                "signal_strength": 0.0,
                "noise_level": None,
                "mode": safe_mode,
                "message": {},
            }
        observation = beacon.observe(
            receiver_position=receiver_position,
            receiver_yaw_deg=receiver_yaw_deg,
            target_sequence_index=target_sequence_index,
            observation_mode=safe_mode,
            official_mode=official_mode,
            rng=self.rng,
        )
        if beacon.mode == "always_on":
            observation["visible_beacon_ids"] = list(self.beacons_by_gate_id.keys())
        elif beacon.mode == "sequential_channel":
            observation["channel"] = target_sequence_index % max(1, len(self.beacons_by_gate_id))
        return observation

