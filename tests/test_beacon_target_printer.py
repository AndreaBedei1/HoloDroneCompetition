from __future__ import annotations

from io import StringIO

from marine_race_arena.scripts.run_marine_race import BeaconTargetPrinter


def test_beacon_target_printer_formats_available_target() -> None:
    stream = StringIO()
    printer = BeaconTargetPrinter(enabled=True, stream=stream)

    printed = printer.update(_observation(time_s=12.4), "bluerov2_01")

    assert printed is True
    assert stream.getvalue().strip() == (
        "[BEACON] t=12.4s participant=bluerov2_01 status=RUNNING target=G03 "
        "index=2 range=8.20 bearing=-0.14 elevation=0.03 completed=2"
    )


def test_beacon_target_printer_handles_missing_fields() -> None:
    stream = StringIO()
    printer = BeaconTargetPrinter(enabled=True, stream=stream)

    printed = printer.update({"time_s": 12.4, "sensors": {}}, "bluerov2_01")

    assert printed is True
    assert stream.getvalue().strip() == "[BEACON] t=12.4s participant=bluerov2_01 target unavailable"


def test_beacon_target_printer_suppresses_repeated_frames() -> None:
    stream = StringIO()
    printer = BeaconTargetPrinter(enabled=True, periodic_interval_s=2.0, stream=stream)

    assert printer.update(_observation(time_s=1.0), "bluerov2_01") is True
    assert printer.update(_observation(time_s=1.1), "bluerov2_01") is False

    assert stream.getvalue().count("[BEACON]") == 1


def test_beacon_target_printer_prints_target_changes() -> None:
    stream = StringIO()
    printer = BeaconTargetPrinter(enabled=True, periodic_interval_s=2.0, stream=stream)

    printer.update(_observation(time_s=1.0, target="G03", index=2), "bluerov2_01")
    printed = printer.update(_observation(time_s=1.1, target="G04", index=3), "bluerov2_01")

    assert printed is True
    assert "target=G04" in stream.getvalue()


def _observation(time_s: float, target: str = "G03", index: int = 2) -> dict:
    return {
        "time_s": time_s,
        "beacon": {
            "valid": True,
            "target_gate_id": target,
            "sequence_index": index,
            "range_m": 8.2,
            "bearing_deg": -0.14,
            "elevation_deg": 0.03,
        },
        "race": {
            "status": "RUNNING",
            "target_gate_id": target,
            "target_sequence_index": index,
            "completed_gates": 2,
        },
    }
