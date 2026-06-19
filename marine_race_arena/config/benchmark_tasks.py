"""Benchmark task mode definitions for Marine Race Arena tracks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


BENCHMARK_TASK_CLEAN_GATE = "clean_gate"
BENCHMARK_TASK_OBSTACLE_GATE = "obstacle_gate"
BENCHMARK_TASK_CURRENT_GATE = "current_gate"
BENCHMARK_TASK_MULTI_ROV = "multi_rov"

BENCHMARK_TASK_MODES: Tuple[str, ...] = (
    BENCHMARK_TASK_CLEAN_GATE,
    BENCHMARK_TASK_OBSTACLE_GATE,
    BENCHMARK_TASK_CURRENT_GATE,
    BENCHMARK_TASK_MULTI_ROV,
)

STRONG_CURRENT_MIN_SPEED_M_S = 0.5


@dataclass(frozen=True)
class BenchmarkTaskConfig:
    """Optional benchmark task mode attached to a track config."""

    mode: Optional[str] = None

    @property
    def is_explicit(self) -> bool:
        return self.mode is not None


def normalize_benchmark_task_mode(mode: str | None) -> str | None:
    """Normalize a benchmark task override supplied by code or CLI."""

    if mode is None:
        return None
    normalized = str(mode).strip()
    return normalized or None
