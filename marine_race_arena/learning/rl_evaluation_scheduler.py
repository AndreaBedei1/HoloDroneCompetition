"""Cross-algorithm scheduling and dynamic HoloOcean evaluator allocation."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

from marine_race_arena.learning.holoocean_capacity import (
    capacity_snapshot,
    reserve_holoocean_engines,
)
from marine_race_arena.learning.longrun_checkpoint import atomic_append_jsonl
from marine_race_arena.learning.rl_evaluation_lock import (
    WAITING_STATE, evaluation_lock,
)
from marine_race_arena.learning.provenance import now_utc


# The lock itself lives in rl_evaluation_lock: contention is normal operation
# and must never surface as a trainer failure.
_exclusive_evaluation_lock = evaluation_lock


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

    def _record_wait(record):
        if audit_path is not None:
            atomic_append_jsonl(audit_path, {"event": "waiting", **record})

    with evaluation_lock(str(owner), on_wait=_record_wait) as lease:
        allocation = select_evaluation_workers(
            evaluation_config,
            fixed_workers=fixed_workers,
        )
        allocation.update({
            "owner": str(owner), "allocated_utc": now_utc(),
            "lock_waited_seconds": lease["waited_seconds"],
            "lock_wait_cycles": lease["wait_cycles"],
        })
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
