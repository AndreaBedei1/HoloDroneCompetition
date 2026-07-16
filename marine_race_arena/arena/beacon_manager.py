"""Beacon field manager: independent periodic transmissions to each receiver.

The manager owns every gate beacon and delivers physically received packets to
a receiver. It never asks the referee (or anyone else) which beacon a receiver
"should" hear: all beacons transmit on their own schedule and a receiver hears
whatever is in range and not dropped.

Reproducibility: dropout and measurement noise are drawn from a dedicated RNG
stream keyed by ``(seed, beacon_id, receiver_id, transmission_index)``. The
stream is therefore independent of call order, participant count and wall
clock, so a fixed seed reproduces the exact same reception sequence for the
same trajectories.

Diagnostics (why a packet was not delivered) are recorded on the manager for
debug logging only; they are never part of a delivered packet.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

from marine_race_arena.arena.beacon import Beacon
from marine_race_arena.arena.gate import Gate
from marine_race_arena.config.schema import GateConfig, Vector3


class BeaconManager:
    def __init__(self, beacons: Iterable[Beacon], seed: Optional[int] = None):
        self.beacons: List[Beacon] = list(beacons)
        self.beacons_by_id: Dict[str, Beacon] = {beacon.id: beacon for beacon in self.beacons}
        self.seed = 0 if seed is None else int(seed)
        # Last transmission index delivered (or consumed) per (beacon, receiver).
        self._last_indices: Dict[Tuple[str, str], int] = {}
        # Debug-only reception counters; never exposed to controllers.
        self.diagnostics: Dict[str, Dict[str, int]] = {}

    @classmethod
    def from_gates(
        cls,
        gates: Iterable[Gate],
        gate_configs: Iterable[GateConfig],
        seed: Optional[int] = None,
        ordered_gate_ids: Optional[Iterable[str]] = None,
    ) -> "BeaconManager":
        configs_by_id = {gate_config.id: gate_config for gate_config in gate_configs}
        beacons_by_gate: Dict[str, Beacon] = {}
        for gate in gates:
            gate_config = configs_by_id[gate.id]
            beacon = Beacon.from_gate(gate, gate_config.beacon)
            if beacon is not None:
                beacons_by_gate[gate.id] = beacon
        if ordered_gate_ids is not None:
            ordered_ids = [gate_id for gate_id in ordered_gate_ids if gate_id in beacons_by_gate]
            leftovers = [
                beacon
                for gate_id, beacon in beacons_by_gate.items()
                if gate_id not in set(ordered_ids)
            ]
            return cls([beacons_by_gate[gate_id] for gate_id in ordered_ids] + leftovers, seed=seed)
        return cls(beacons_by_gate.values(), seed=seed)

    def receive(
        self,
        *,
        receiver_id: str,
        receiver_position: Vector3,
        receiver_yaw_deg: float,
        time_s: float,
        received_at_s: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return the packets physically received by ``receiver_id`` at ``time_s``.

        ``time_s`` is simulator time used for the transmission schedule.
        ``received_at_s`` is the receiver-local timestamp stamped on packets
        (defaults to ``time_s``). Each transmission is delivered at most once
        per receiver; if a beacon has not transmitted since the last call, no
        new packet appears.
        """
        local_time = time_s if received_at_s is None else received_at_s
        packets: List[Dict[str, Any]] = []
        for beacon in self.beacons:
            key = (beacon.id, receiver_id)
            index = beacon.transmission_index(time_s)
            if index < 0:
                continue
            last = self._last_indices.get(key, -1)
            if index <= last:
                self._count(receiver_id, "no_new_transmission")
                continue
            # Consume every missed transmission; only the newest can still be
            # heard (acoustic pings are ephemeral).
            self._last_indices[key] = index
            rng = self._packet_rng(beacon.id, receiver_id, index)
            packet = beacon.receive(
                receiver_position=receiver_position,
                receiver_yaw_deg=receiver_yaw_deg,
                received_at_s=local_time,
                rng=rng,
            )
            if packet is None:
                self._count(receiver_id, "not_received")
                continue
            self._count(receiver_id, "received")
            packets.append(packet)
        return packets

    def reset_receiver(self, receiver_id: str) -> None:
        """Forget delivery state for a receiver (e.g. between runs)."""
        self._last_indices = {
            key: value for key, value in self._last_indices.items() if key[1] != receiver_id
        }
        self.diagnostics.pop(receiver_id, None)

    def _packet_rng(self, beacon_id: str, receiver_id: str, index: int) -> random.Random:
        # random.Random(str) seeds deterministically from the string bytes
        # (unlike hash(), which is salted per process).
        return random.Random(f"{self.seed}|{beacon_id}|{receiver_id}|{index}")

    def _count(self, receiver_id: str, outcome: str) -> None:
        counters = self.diagnostics.setdefault(receiver_id, {})
        counters[outcome] = counters.get(outcome, 0) + 1
