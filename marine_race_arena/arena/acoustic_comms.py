"""Optional inter-rover acoustic communication channel.

The channel models the underwater acoustic medium rather than a perfect radio
link: messages travel at the speed of sound, so latency grows with range; the
range is bounded; packets can be dropped; payloads are small; and a sender
cannot transmit faster than a minimum interval. This mirrors the low-bandwidth,
high-latency, lossy acoustic channel discussed in the related work.

No-cheat boundary: a message payload is authored by the sending controller,
which sees only its official observation, so it can only ever carry information
the controller legally knows. The channel uses the true rover positions solely
to compute the channel physics (range, latency, loss); it never injects pose,
geometry or any other privileged state into a delivered message.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

Vector3 = Tuple[float, float, float]


@dataclass
class CommsConfig:
    enabled: bool = False
    sound_speed_m_s: float = 1500.0
    max_range_m: float = 100.0
    processing_delay_s: float = 0.05
    packet_loss_prob: float = 0.0
    max_payload_bytes: int = 128
    min_send_interval_s: float = 0.5


class AcousticCommsChannel:
    """Buffered, range/latency/loss-limited broadcast channel between rovers."""

    def __init__(self, config: CommsConfig, seed: Optional[int] = None) -> None:
        self.config = config
        self._rng = random.Random(0 if seed is None else int(seed))
        # Scheduled deliveries: (deliver_time_s, receiver_id, message).
        self._pending: List[Tuple[float, str, Dict[str, Any]]] = []
        self._last_send_time: Dict[str, float] = {}
        self.sent = 0
        self.delivered = 0
        self.dropped_rate_limited = 0
        self.dropped_oversized = 0
        self.dropped_out_of_range = 0
        self.dropped_packet_loss = 0

    def send(
        self,
        *,
        sender_id: str,
        payload: Any,
        send_time_s: float,
        sender_position: Vector3,
        receiver_positions: Mapping[str, Vector3],
    ) -> None:
        """Broadcast a controller-authored payload to the rovers in range."""
        if not self.config.enabled or payload is None:
            return

        last = self._last_send_time.get(sender_id)
        if last is not None and (send_time_s - last) < self.config.min_send_interval_s:
            self.dropped_rate_limited += 1
            return

        try:
            serialized = json.dumps(payload, default=str)
        except (TypeError, ValueError):
            self.dropped_oversized += 1
            return
        if len(serialized.encode("utf-8")) > self.config.max_payload_bytes:
            self.dropped_oversized += 1
            return

        self._last_send_time[sender_id] = send_time_s
        self.sent += 1

        for receiver_id in sorted(receiver_positions):
            distance_m = _distance(sender_position, receiver_positions[receiver_id])
            if distance_m > self.config.max_range_m:
                self.dropped_out_of_range += 1
                continue
            if self.config.packet_loss_prob > 0.0 and self._rng.random() < self.config.packet_loss_prob:
                self.dropped_packet_loss += 1
                continue
            latency_s = distance_m / max(1e-6, self.config.sound_speed_m_s) + self.config.processing_delay_s
            message = {
                "from": sender_id,
                "payload": copy.deepcopy(payload),
                "sent_at_s": float(send_time_s),
            }
            self._pending.append((float(send_time_s) + latency_s, receiver_id, message))

    def deliver(self, receiver_id: str, current_time_s: float) -> List[Dict[str, Any]]:
        """Pop and return the messages that have reached ``receiver_id`` by now."""
        if not self.config.enabled:
            return []
        ready: List[Tuple[float, Dict[str, Any]]] = []
        remaining: List[Tuple[float, str, Dict[str, Any]]] = []
        for deliver_time_s, rid, message in self._pending:
            if rid == receiver_id and deliver_time_s <= current_time_s + 1e-9:
                arrived = dict(message)
                arrived["received_at_s"] = float(current_time_s)
                ready.append((deliver_time_s, arrived))
            else:
                remaining.append((deliver_time_s, rid, message))
        self._pending = remaining
        ready.sort(key=lambda item: item[0])
        self.delivered += len(ready)
        return [message for _, message in ready]

    def summary(self) -> Dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "sound_speed_m_s": self.config.sound_speed_m_s,
            "max_range_m": self.config.max_range_m,
            "processing_delay_s": self.config.processing_delay_s,
            "packet_loss_prob": self.config.packet_loss_prob,
            "max_payload_bytes": self.config.max_payload_bytes,
            "min_send_interval_s": self.config.min_send_interval_s,
            "messages_sent": self.sent,
            "messages_delivered": self.delivered,
            "dropped_rate_limited": self.dropped_rate_limited,
            "dropped_oversized": self.dropped_oversized,
            "dropped_out_of_range": self.dropped_out_of_range,
            "dropped_packet_loss": self.dropped_packet_loss,
        }


def _distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
