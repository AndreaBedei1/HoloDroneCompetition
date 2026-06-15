from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


TRACK_DIR = Path("marine_race_arena") / "tracks"

DEFAULT_COLORS = [
    "#00ff88", "#38bdf8", "#facc15", "#fb7185", "#f97316",
    "#a78bfa", "#22d3ee", "#84cc16", "#e879f9", "#22c55e",
    "#f43f5e", "#60a5fa", "#fde047", "#34d399", "#c084fc",
    "#fb923c", "#14b8a6", "#e11d48", "#7dd3fc",
]


def round_value(value: float, ndigits: int = 3) -> float:
    rounded = round(float(value), ndigits)
    return 0.0 if abs(rounded) < 0.0005 else rounded


def normalize_xy(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.hypot(vector[0], vector[1])
    if length <= 1e-9:
        return (1.0, 0.0, 0.0)
    return (vector[0] / length, vector[1] / length, 0.0)


def yaw_from_direction(direction: tuple[float, float, float]) -> float:
    return math.degrees(math.atan2(direction[1], direction[0]))


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def path_length(start: tuple[float, float, float], points: list[tuple[float, float, float]]) -> float:
    return sum(distance(a, b) for a, b in zip([start] + points, points))


def build_gates(
    points: list[tuple[float, float, float]],
    start: tuple[float, float, float],
    gate_types: dict[int, str] | None = None,
    linked_gates: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    gates = []
    gate_types = gate_types or {}
    linked_gates = linked_gates or {}

    for index, point in enumerate(points, start=1):
        previous_point = start if index == 1 else points[index - 2]
        next_point = points[index % len(points)]

        tangent = (
            next_point[0] - previous_point[0],
            next_point[1] - previous_point[1],
            0.0,
        )
        direction = normalize_xy(tangent)
        yaw_deg = round_value(yaw_from_direction(direction), 1)

        gate = {
            "id": f"G{index:02d}",
            "type": gate_types.get(index, "single"),
            "position": [round_value(point[0], 2), round_value(point[1], 2), round_value(point[2], 2)],
            "rotation_rpy_deg": [0.0, 0.0, yaw_deg],
            "color": DEFAULT_COLORS[(index - 1) % len(DEFAULT_COLORS)],
            "passage_direction": [
                round_value(direction[0], 3),
                round_value(direction[1], 3),
                0.0,
            ],
        }

        if index in linked_gates:
            gate["linked_gate"] = f"G{linked_gates[index]:02d}"

        gates.append(gate)

    return gates


def make_track(
    name: str,
    track_label: str,
    laps: int,
    start: tuple[float, float, float],
    points: list[tuple[float, float, float]],
    bounds: dict[str, float],
    max_duration_s: float,
    beacon_noise: float,
    beacon_dropout: float,
    currents: list[dict[str, Any]],
    clearance_margin_m: float,
    gate_types: dict[int, str] | None = None,
    linked_gates: dict[int, int] | None = None,
) -> dict[str, Any]:
    gates = build_gates(points, start, gate_types, linked_gates)

    return {
        "race": {
            "name": name,
            "format": "ai_grand_challenge",
            "laps": laps,
            "expected_gates_per_lap": len(points),
            "timing_mode": "first_gate_to_last_gate",
            "max_duration_s": max_duration_s,
            "official_mode": False,
        },
        "world": {
            "package": "Ocean",
            "map": "OpenWater-Hovering",
            "arena_origin": [0.0, 0.0, 0.0],
            "preferred_environment": "OpenWater-Hovering",
            "fallback_environment": "PierHarbor-Hovering",
            "bounds": bounds,
        },
        "track": {
            "declared_length_m": round(path_length(start, points), 1),
            "length_tolerance_m": 4.0 if len(points) < 16 else 6.0,
            "gate_inner_size_m": [1.5, 1.5],
            "gate_bar_thickness_m": 0.18,
            "gate_depth_m": 0.22,
            "gate_sequence": [f"G{index:02d}" for index in range(1, len(points) + 1)],
        },
        "start": {
            "position": [round_value(start[0], 2), round_value(start[1], 2), round_value(start[2], 2)],
            "rotation_rpy_deg": [0.0, 0.0, 0.0],
        },
        "finish": {
            "gate_id": f"G{len(points):02d}",
        },
        "beacon": {
            "enabled": True,
            "mode": "active_when_target",
            "position_offset": [0.0, 0.0, 0.35],
            "range_m": 80.0 if len(points) < 16 else 110.0,
            "noise_std": beacon_noise,
            "dropout_probability": beacon_dropout,
            "update_rate_hz": 10.0,
            "message": {
                "track": track_label,
                "channel_plan": "target_only",
            },
        },
        "gates": gates,
        "currents": currents,
        "obstacles": [],
        "participants": [
            {
                "id": "bluerov2_01",
                "vehicle": "BlueROV2",
                "controller": "pygame",
                "controller_class": None,
                "spawn": {
                    "position": [round_value(start[0], 2), round_value(start[1], 2), round_value(start[2], 2)],
                    "rotation_rpy_deg": [0.0, 0.0, 0.0],
                },
                "sensors": {
                    "profile": "official_acoustic",
                },
                "control_mode": "high_level",
                "official_sensor_profile": True,
            }
        ],
        "referee": {
            "gate_validation": {
                "vehicle_model": "center_point",
                "vehicle_clearance_margin_m": clearance_margin_m,
                "stuck_timeout_s": 40.0 if len(points) < 16 else 55.0,
                "stuck_speed_threshold_m_s": 0.02,
            },
            "penalties": {
                "minor_collision_s": 5.0,
                "gate_collision_s": 10.0,
                "wrong_direction_s": 20.0,
                "missed_gate_dnf": True,
                "severe_collision_dnf": True,
                "wrong_direction_dsq": False,
            },
            "scoring": {
                "rank_finished_by": "penalized_time",
                "rank_unfinished_by": "completed_gates",
            },
        },
    }


def main() -> None:
    TRACK_DIR.mkdir(parents=True, exist_ok=True)

    sprint_start = (-14.0, -2.0, -4.0)
    sprint_points = [
        (-14.0, -4.0, -4.0),
        (-8.0, -4.0, -4.0),
        (-2.0, -3.5, -4.0),
        (4.0, -2.0, -4.1),
        (10.0, 1.5, -4.1),
        (12.0, 7.0, -4.2),
        (7.0, 11.0, -4.2),
        (0.0, 12.0, -4.1),
        (-8.0, 10.0, -4.0),
        (-14.0, 5.0, -4.0),
        (-16.0, -1.0, -4.0),
    ]

    technical_start = (-22.0, -10.0, -4.0)
    technical_points = [
        (-23.0, -12.0, -4.0),
        (-15.0, -12.0, -4.1),
        (-7.0, -10.0, -4.2),
        (1.0, -7.0, -4.4),
        (10.0, -4.0, -4.4),
        (18.0, 0.0, -4.5),
        (23.0, 7.0, -4.3),
        (20.0, 14.0, -4.2),
        (12.0, 18.0, -4.4),
        (2.0, 18.0, -4.7),
        (-8.0, 15.0, -4.8),
        (-17.0, 10.0, -4.5),
        (-24.0, 4.0, -4.2),
        (-27.0, -4.0, -4.1),
        (-25.0, -10.0, -4.0),
    ]

    abyss_start = (-30.0, 0.0, -4.0)
    abyss_points = [
        (-30.0, -15.0, -4.0),
        (-22.0, -16.0, -4.3),
        (-14.0, -18.0, -4.7),
        (-5.0, -20.0, -5.0),
        (5.0, -19.0, -5.3),
        (15.0, -15.0, -4.8),
        (24.0, -9.0, -4.4),
        (31.0, -1.0, -4.2),
        (31.0, 8.0, -4.3),
        (25.0, 16.0, -5.7),
        (15.0, 20.0, -5.0),
        (4.0, 22.0, -4.4),
        (-8.0, 21.0, -4.1),
        (-19.0, 17.0, -4.8),
        (-29.0, 11.0, -5.2),
        (-34.0, 3.0, -4.6),
        (-34.0, -6.0, -4.3),
        (-31.0, -12.0, -4.1),
        (-28.0, -15.0, -4.0),
    ]

    tracks = {
        "marine_race_sprint_loop.json": make_track(
            name="Marine Race Sprint Loop",
            track_label="sprint_loop",
            laps=2,
            start=sprint_start,
            points=sprint_points,
            bounds={
                "x_min": -18.0,
                "x_max": 18.0,
                "y_min": -12.0,
                "y_max": 14.0,
                "z_min": -8.0,
                "z_max": -1.0,
            },
            max_duration_s=360,
            beacon_noise=0.25,
            beacon_dropout=0.0,
            currents=[
                {
                    "type": "constant",
                    "velocity": [0.02, 0.05, 0.0],
                }
            ],
            clearance_margin_m=0.10,
        ),
        "marine_race_technical_canyon.json": make_track(
            name="Marine Race Technical Canyon",
            track_label="technical_canyon",
            laps=2,
            start=technical_start,
            points=technical_points,
            bounds={
                "x_min": -28.0,
                "x_max": 26.0,
                "y_min": -18.0,
                "y_max": 18.0,
                "z_min": -8.0,
                "z_max": -1.0,
            },
            max_duration_s=720,
            beacon_noise=0.45,
            beacon_dropout=0.02,
            currents=[
                {
                    "type": "constant",
                    "velocity": [0.03, 0.10, 0.0],
                },
                {
                    "type": "localized_jet",
                    "center": [2.0, 13.0, -4.4],
                    "radius": 5.0,
                    "velocity": [0.22, -0.05, 0.02],
                    "falloff": "gaussian",
                },
                {
                    "type": "localized_jet",
                    "center": [18.0, -2.0, -4.3],
                    "radius": 4.5,
                    "velocity": [-0.05, 0.24, 0.0],
                    "falloff": "gaussian",
                },
            ],
            clearance_margin_m=0.18,
            gate_types={5: "double", 6: "double"},
            linked_gates={5: 6, 6: 5},
        ),
        "marine_race_abyss_grand_prix.json": make_track(
            name="Marine Race Abyss Grand Prix",
            track_label="abyss_grand_prix",
            laps=2,
            start=abyss_start,
            points=abyss_points,
            bounds={
                "x_min": -35.0,
                "x_max": 32.0,
                "y_min": -24.0,
                "y_max": 24.0,
                "z_min": -8.0,
                "z_max": -1.0,
            },
            max_duration_s=1100,
            beacon_noise=0.65,
            beacon_dropout=0.04,
            currents=[
                {
                    "type": "constant",
                    "velocity": [0.04, 0.09, 0.0],
                },
                {
                    "type": "localized_jet",
                    "center": [22.0, 0.0, -4.2],
                    "radius": 5.5,
                    "velocity": [0.0, -0.28, 0.02],
                    "falloff": "gaussian",
                },
                {
                    "type": "localized_jet",
                    "center": [-16.0, -15.0, -5.4],
                    "radius": 6.0,
                    "velocity": [0.26, 0.08, -0.02],
                    "falloff": "gaussian",
                },
                {
                    "type": "sinusoidal",
                    "axis": "z",
                    "amplitude": 0.06,
                    "frequency_hz": 0.08,
                    "phase": 0.0,
                },
            ],
            clearance_margin_m=0.22,
            gate_types={
                9: "split_s_upper",
                10: "split_s_lower",
                14: "double",
                15: "double",
            },
            linked_gates={
                9: 10,
                10: 9,
                14: 15,
                15: 14,
            },
        ),
    }

    for filename, payload in tracks.items():
        path = TRACK_DIR / filename
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path} with {len(payload['gates'])} gates and declared length {payload['track']['declared_length_m']} m")


if __name__ == "__main__":
    main()