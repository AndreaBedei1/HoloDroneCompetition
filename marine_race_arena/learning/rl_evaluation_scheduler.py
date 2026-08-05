"""Cross-algorithm scheduling and dynamic HoloOcean evaluator allocation."""

from __future__ import annotations

import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

from marine_race_arena.learning.holoocean_capacity import (
    capacity_snapshot,
    reserve_holoocean_engines,
)
from marine_race_arena.learning.longrun_checkpoint import atomic_append_jsonl
from marine_race_arena.learning.provenance import now_utc


_SCHEDULER_ROOT = Path(tempfile.gettempdir()) / "holodrone_evaluation_scheduler"
_SCHEDULER_LOCK = _SCHEDULER_ROOT / "evaluation.lock"


@contextmanager
def _exclusive_evaluation_lock() -> Iterator[None]:
    """Serialize PPO/SAC evaluations without coupling either algorithm."""

    _SCHEDULER_ROOT.mkdir(parents=True, exist_ok=True)
    with _SCHEDULER_LOCK.open("a+b") as handle:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(1.0)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - production is Windows
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def select_evaluation_workers(
    evaluation_config: Mapping[str, Any],
    *,
    fixed_workers: int,
    snapshot: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Choose a measured evaluator count that fits the current global cap."""

    dynamic = dict(evaluation_config.get("dynamic_parallelism") or {})
    capacity = dict(snapshot or capacity_snapshot())
    available = int(capacity.get("available", 0))
    if available < 1:
        raise RuntimeError("no HoloOcean slot is available for evaluation")
    if not bool(dynamic.get("enabled", False)):
        requested = int(fixed_workers)
        source = "fixed_config"
    else:
        candidates = sorted({
            int(value) for value in dynamic.get("candidates", [4, 6, 8])
            if int(value) > 0
        })
        if not candidates:
            raise ValueError("dynamic evaluator candidates cannot be empty")
        requested = int(dynamic.get("selected_workers", candidates[0]))
        if requested not in candidates:
            raise ValueError("selected evaluator count is not a benchmark candidate")
        source = "measured_dynamic_config"
    selected = min(requested, available)
    if selected < 1:
        raise RuntimeError("dynamic evaluator allocation selected zero workers")
    return {
        "schema_version": "rl_evaluation_allocation_v1",
        "selected_workers": selected,
        "requested_workers": requested,
        "selection_source": source,
        "capacity": capacity,
        "degraded_for_current_capacity": selected < requested,
    }


@contextmanager
def scheduled_evaluation(
    evaluation_config: Mapping[str, Any],
    *,
    fixed_workers: int,
    owner: str,
    audit_path: str | Path | None = None,
) -> Iterator[Dict[str, Any]]:
    """Wait for the shared evaluation turn and hold it until engines close."""

    with _exclusive_evaluation_lock():
        allocation = select_evaluation_workers(
            evaluation_config,
            fixed_workers=fixed_workers,
        )
        allocation.update({"owner": str(owner), "allocated_utc": now_utc()})
        if audit_path is not None:
            atomic_append_jsonl(audit_path, {"event": "allocated", **allocation})
        with reserve_holoocean_engines(
            int(allocation["selected_workers"]), owner=str(owner)
        ):
            try:
                yield allocation
            finally:
                if audit_path is not None:
                    atomic_append_jsonl(audit_path, {
                        "event": "released",
                        "owner": str(owner),
                        "workers": allocation["selected_workers"],
                        "utc": now_utc(),
                    })
