from __future__ import annotations

from marine_race_arena.arena.acoustic_comms import AcousticCommsChannel, CommsConfig


def _send(channel: AcousticCommsChannel, sender, payload, t, sender_pos, receivers):
    channel.send(
        sender_id=sender,
        payload=payload,
        send_time_s=t,
        sender_position=sender_pos,
        receiver_positions=receivers,
    )


def test_disabled_channel_is_a_noop() -> None:
    channel = AcousticCommsChannel(CommsConfig(enabled=False))
    _send(channel, "a", {"x": 1}, 0.0, (0.0, 0.0, 0.0), {"b": (1.0, 0.0, 0.0)})

    assert channel.deliver("b", 100.0) == []
    assert channel.sent == 0


def test_message_delivered_only_after_acoustic_latency() -> None:
    channel = AcousticCommsChannel(
        CommsConfig(enabled=True, sound_speed_m_s=10.0, processing_delay_s=0.0, max_range_m=1000.0, min_send_interval_s=0.0)
    )
    _send(channel, "a", {"target": "g03"}, 0.0, (0.0, 0.0, 0.0), {"b": (30.0, 0.0, 0.0)})

    # 30 m at 10 m/s => 3.0 s latency.
    assert channel.deliver("b", 2.0) == []
    delivered = channel.deliver("b", 3.0)
    assert len(delivered) == 1
    assert delivered[0]["payload"] == {"target": "g03"}


def test_out_of_range_message_is_dropped() -> None:
    channel = AcousticCommsChannel(CommsConfig(enabled=True, max_range_m=10.0, min_send_interval_s=0.0))
    _send(channel, "a", {"x": 1}, 0.0, (0.0, 0.0, 0.0), {"b": (50.0, 0.0, 0.0)})

    assert channel.deliver("b", 100.0) == []
    assert channel.dropped_out_of_range == 1


def test_oversized_payload_is_dropped() -> None:
    channel = AcousticCommsChannel(CommsConfig(enabled=True, max_payload_bytes=16, min_send_interval_s=0.0))
    _send(channel, "a", {"blob": "x" * 100}, 0.0, (0.0, 0.0, 0.0), {"b": (1.0, 0.0, 0.0)})

    assert channel.sent == 0
    assert channel.dropped_oversized == 1
    assert channel.deliver("b", 100.0) == []


def test_rate_limit_drops_rapid_transmissions() -> None:
    channel = AcousticCommsChannel(
        CommsConfig(enabled=True, min_send_interval_s=1.0, max_range_m=1000.0)
    )
    pos_a, recv = (0.0, 0.0, 0.0), {"b": (1.0, 0.0, 0.0)}
    _send(channel, "a", {"n": 1}, 0.0, pos_a, recv)
    _send(channel, "a", {"n": 2}, 0.5, pos_a, recv)  # too soon -> dropped
    _send(channel, "a", {"n": 3}, 1.0, pos_a, recv)

    assert channel.sent == 2
    assert channel.dropped_rate_limited == 1


def test_packet_loss_is_deterministic_for_a_fixed_seed() -> None:
    def run() -> tuple[int, int]:
        channel = AcousticCommsChannel(
            CommsConfig(enabled=True, packet_loss_prob=0.5, max_range_m=1000.0, min_send_interval_s=0.0),
            seed=42,
        )
        for i in range(50):
            _send(channel, "a", {"n": i}, float(i), (0.0, 0.0, 0.0), {"b": (1.0, 0.0, 0.0)})
        received = channel.deliver("b", 10000.0)
        return len(received), channel.dropped_packet_loss

    first = run()
    second = run()
    assert first == second
    # Some delivered and some dropped, i.e. loss actually fired.
    assert first[0] > 0 and first[1] > 0


def test_delivered_message_carries_only_controller_payload() -> None:
    channel = AcousticCommsChannel(
        CommsConfig(enabled=True, sound_speed_m_s=1500.0, max_range_m=1000.0, min_send_interval_s=0.0)
    )
    _send(channel, "rover_a", {"intent": "yield"}, 0.0, (0.0, 0.0, -4.0), {"rover_b": (5.0, 0.0, -4.0)})

    delivered = channel.deliver("rover_b", 100.0)
    assert len(delivered) == 1
    message = delivered[0]
    # No privileged geometry/state may leak: only the controller-authored payload
    # plus channel metadata (sender id and timestamps).
    assert set(message.keys()) == {"from", "payload", "sent_at_s", "received_at_s"}
    assert message["from"] == "rover_a"
    assert message["payload"] == {"intent": "yield"}


def test_lone_rover_with_no_receivers_hears_nothing() -> None:
    channel = AcousticCommsChannel(CommsConfig(enabled=True, min_send_interval_s=0.0))
    _send(channel, "a", {"x": 1}, 0.0, (0.0, 0.0, 0.0), {})

    assert channel.delivered == 0
    assert channel.deliver("a", 100.0) == []
