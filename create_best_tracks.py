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


def first_gate_direction_from_points(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    if len(points) < 2:
        return (1.0, 0.0, 0.0)

    return normalize_xy(
        (
            points[1][0] - points[0][0],
            points[1][1] - points[0][1],
            0.0,
        )
    )


def compute_start_from_first_gate(
    first_gate_position: tuple[float, float, float],
    first_gate_direction: tuple[float, float, float],
    start_distance_m: float = 4.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    direction = normalize_xy(first_gate_direction)

    start_position = (
        first_gate_position[0] - direction[0] * start_distance_m,
        first_gate_position[1] - direction[1] * start_distance_m,
        first_gate_position[2],
    )

    start_rotation = (
        0.0,
        0.0,
        round_value(yaw_from_direction(direction), 1),
    )

    return start_position, start_rotation


def gate_direction_for_index(
    points: list[tuple[float, float, float]],
    start: tuple[float, float, float],
    index_zero_based: int,
) -> tuple[float, float, float]:
    if len(points) == 1:
        return normalize_xy(
            (
                points[0][0] - start[0],
                points[0][1] - start[1],
                0.0,
            )
        )

    if index_zero_based == 0:
        current = points[0]
        next_point = points[1]
        return normalize_xy(
            (
                next_point[0] - current[0],
                next_point[1] - current[1],
                0.0,
            )
        )

    if index_zero_based == len(points) - 1:
        previous = points[index_zero_based - 1]
        current = points[index_zero_based]
        return normalize_xy(
            (
                current[0] - previous[0],
                current[1] - previous[1],
                0.0,
            )
        )

    previous = points[index_zero_based - 1]
    next_point = points[index_zero_based + 1]
    return normalize_xy(
        (
            next_point[0] - previous[0],
            next_point[1] - previous[1],
            0.0,
        )
    )


def build_gates(
    points: list[tuple[float, float, float]],
    start: tuple[float, float, float],
    gate_types: dict[int, str] | None = None,
    linked_gates: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    gates = []
    gate_types = gate_types or {}
    linked_gates = linked_gates or {}

    for index_zero_based, point in enumerate(points):
        gate_index = index_zero_based + 1
        direction = gate_direction_for_index(points, start, index_zero_based)
        yaw_deg = round_value(yaw_from_direction(direction), 1)

        gate = {
            "id": f"G{gate_index:02d}",
            "type": gate_types.get(gate_index, "single"),
            "position": [
                round_value(point[0], 2),
                round_value(point[1], 2),
                round_value(point[2], 2),
            ],
            "rotation_rpy_deg": [0.0, 0.0, yaw_deg],
            "color": DEFAULT_COLORS[(gate_index - 1) % len(DEFAULT_COLORS)],
            "passage_direction": [
                round_value(direction[0], 3),
                round_value(direction[1], 3),
                0.0,
            ],
        }

        if gate_index in linked_gates:
            gate["linked_gate"] = f"G{linked_gates[gate_index]:02d}"

        gates.append(gate)

    return gates


def make_track(
    name: str,
    track_label: str,
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
    first_direction = first_gate_direction_from_points(points)
    start, start_rotation = compute_start_from_first_gate(
        first_gate_position=points[0],
        first_gate_direction=first_direction,
        start_distance_m=4.0,
    )

    gates = build_gates(points, start, gate_types, linked_gates)
    declared_length = round(path_length(start, points), 1)
    finish_distance_from_start = distance(start, points[-1])

    return {
        "race": {
            "name": name,
            "format": "ai_grand_challenge",
            "laps": 1,
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
            "declared_length_m": declared_length,
            "length_tolerance_m": 4.0 if len(points) < 16 else 7.0,
            "gate_inner_size_m": [1.5, 1.5],
            "gate_bar_thickness_m": 0.18,
            "gate_depth_m": 0.22,
            "gate_sequence": [f"G{index:02d}" for index in range(1, len(points) + 1)],
            "metadata": {
                "style": "point_to_point",
                "finish_distance_from_start_m": round_value(finish_distance_from_start, 1),
                "design_note": "The finish is intentionally far from the start to make the route visually clear.",
            },
        },
        "start": {
            "position": [
                round_value(start[0], 2),
                round_value(start[1], 2),
                round_value(start[2], 2),
            ],
            "rotation_rpy_deg": [
                round_value(start_rotation[0], 1),
                round_value(start_rotation[1], 1),
                round_value(start_rotation[2], 1),
            ],
        },
        "finish": {
            "gate_id": f"G{len(points):02d}",
        },
        "beacon": {
            "enabled": True,
            "mode": "active_when_target",
            "position_offset": [0.0, 0.0, 0.35],
            "range_m": 90.0 if len(points) < 16 else 130.0,
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
                    "position": [
                        round_value(start[0], 2),
                        round_value(start[1], 2),
                        round_value(start[2], 2),
                    ],
                    "rotation_rpy_deg": [
                        round_value(start_rotation[0], 1),
                        round_value(start_rotation[1], 1),
                        round_value(start_rotation[2], 1),
                    ],
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
                "stuck_timeout_s": 45.0 if len(points) < 16 else 65.0,
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

    # Track 1: open horseshoe.
    # Simple and readable: the rover starts at one side, follows a U-shaped route,
    # and finishes far from the starting area.
    horseshoe_points = [
        (-30.0, -10.0, -4.0),
        (-24.0, -6.0, -4.1),
        (-20.0, 0.0, -4.2),
        (-20.0, 7.0, -4.0),
        (-14.0, 13.0, -4.1),
        (-5.0, 16.0, -4.3),
        (6.0, 16.0, -4.4),
        (15.0, 13.0, -4.2),
        (21.0, 7.0, -4.1),
        (21.0, 0.0, -4.3),
        (25.0, -6.0, -4.2),
        (31.0, -10.0, -4.0),
    ]

    # Track 2: vertical serpent.
    # Clearly a snake-like route, but with strong depth variation.
    # This is meant to test altitude control and smooth direction changes.
    vertical_serpent_points = [
        (-38.0, 0.0, -4.0),
        (-32.0, 7.0, -4.8),
        (-26.0, -7.0, -5.5),
        (-20.0, 7.0, -4.2),
        (-14.0, -7.0, -5.8),
        (-8.0, 7.0, -4.1),
        (-2.0, -7.0, -5.9),
        (4.0, 7.0, -4.5),
        (10.0, -7.0, -5.4),
        (16.0, 7.0, -3.9),
        (22.0, -7.0, -5.2),
        (28.0, 7.0, -4.2),
        (34.0, -7.0, -5.7),
        (40.0, 7.0, -4.4),
        (46.0, -7.0, -5.0),
        (53.0, -2.0, -4.4),
        (60.0, 0.0, -4.0),
    ]

    # Track 3: long mixed endurance route.
    # Different from the first two: it mixes diagonals, mild chicanes,
    # altitude changes, double gates, and a split-S-like section.
    mixed_endurance_points = [
        (-50.0, -14.0, -4.0),
        (-42.0, -12.0, -4.1),
        (-34.0, -9.0, -4.3),
        (-26.0, -3.0, -4.8),
        (-18.0, 4.0, -5.2),
        (-9.0, 7.0, -4.7),
        (-1.0, 4.0, -4.2),
        (7.0, -2.0, -5.4),
        (16.0, -5.0, -5.9),
        (26.0, -3.0, -5.1),
        (35.0, 2.0, -4.4),
        (42.0, 9.0, -3.8),
        (50.0, 14.0, -4.5),
        (58.0, 12.0, -5.3),
        (65.0, 6.0, -5.8),
        (70.0, -2.0, -5.0),
        (76.0, -10.0, -4.3),
        (85.0, -12.0, -4.8),
        (94.0, -8.0, -5.5),
        (102.0, -1.0, -4.9),
        (109.0, 7.0, -4.2),
        (116.0, 14.0, -4.6),
    ]

    tracks = {
        "marine_race_horseshoe_bay.json": make_track(
            name="Marine Race Horseshoe Bay",
            track_label="horseshoe_bay",
            points=horseshoe_points,
            bounds={
                "x_min": -36.0,
                "x_max": 36.0,
                "y_min": -16.0,
                "y_max": 20.0,
                "z_min": -8.0,
                "z_max": -1.0,
            },
            max_duration_s=500,
            beacon_noise=0.20,
            beacon_dropout=0.0,
            currents=[
                {
                    "type": "constant",
                    "velocity": [0.02, 0.04, 0.0],
                }
            ],
            clearance_margin_m=0.10,
        ),
        "marine_race_vertical_serpent.json": make_track(
            name="Marine Race Vertical Serpent",
            track_label="vertical_serpent",
            points=vertical_serpent_points,
            bounds={
                "x_min": -44.0,
                "x_max": 64.0,
                "y_min": -12.0,
                "y_max": 12.0,
                "z_min": -8.0,
                "z_max": -1.0,
            },
            max_duration_s=850,
            beacon_noise=0.45,
            beacon_dropout=0.02,
            currents=[
                {
                    "type": "constant",
                    "velocity": [0.03, 0.08, 0.0],
                },
                {
                    "type": "localized_jet",
                    "center": [-8.0, 7.0, -4.1],
                    "radius": 5.5,
                    "velocity": [0.18, -0.10, 0.03],
                    "falloff": "gaussian",
                },
                {
                    "type": "localized_jet",
                    "center": [22.0, -7.0, -5.2],
                    "radius": 5.5,
                    "velocity": [-0.08, 0.22, -0.02],
                    "falloff": "gaussian",
                },
            ],
            clearance_margin_m=0.16,
            gate_types={
                8: "vertical_double",
                9: "vertical_double",
            },
            linked_gates={
                8: 9,
                9: 8,
            },
        ),
        "marine_race_mixed_endurance.json": make_track(
            name="Marine Race Mixed Endurance",
            track_label="mixed_endurance",
            points=mixed_endurance_points,
            bounds={
                "x_min": -56.0,
                "x_max": 120.0,
                "y_min": -18.0,
                "y_max": 18.0,
                "z_min": -8.0,
                "z_max": -1.0,
            },
            max_duration_s=1300,
            beacon_noise=0.60,
            beacon_dropout=0.04,
            currents=[
                {
                    "type": "constant",
                    "velocity": [0.04, 0.07, 0.0],
                },
                {
                    "type": "localized_jet",
                    "center": [7.0, -2.0, -5.4],
                    "radius": 6.0,
                    "velocity": [0.20, -0.12, 0.02],
                    "falloff": "gaussian",
                },
                {
                    "type": "localized_jet",
                    "center": [58.0, 12.0, -5.3],
                    "radius": 6.5,
                    "velocity": [-0.06, 0.25, -0.02],
                    "falloff": "gaussian",
                },
                {
                    "type": "localized_jet",
                    "center": [94.0, -8.0, -5.5],
                    "radius": 6.0,
                    "velocity": [0.18, 0.12, 0.03],
                    "falloff": "gaussian",
                },
                {
                    "type": "sinusoidal",
                    "axis": "z",
                    "amplitude": 0.05,
                    "frequency_hz": 0.08,
                    "phase": 0.0,
                },
            ],
            clearance_margin_m=0.20,
            gate_types={
                8: "vertical_double",
                9: "vertical_double",
                14: "double",
                15: "double",
                18: "split_s_upper",
                19: "split_s_lower",
            },
            linked_gates={
                8: 9,
                9: 8,
                14: 15,
                15: 14,
                18: 19,
                19: 18,
            },
        ),
    }

    for filename, payload in tracks.items():
        path = TRACK_DIR / filename
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"Wrote {path} with {len(payload['gates'])} gates, "
            f"declared length {payload['track']['declared_length_m']} m, "
            f"finish distance from start {payload['track']['metadata']['finish_distance_from_start_m']} m, "
            f"start {payload['start']['position']}, "
            f"yaw {payload['start']['rotation_rpy_deg'][2]} deg"
        )

if __name__ == "__main__":
    main()