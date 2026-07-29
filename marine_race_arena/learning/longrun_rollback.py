"""Rollback journal normalization and backward-compatible history recovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


ROLLBACK_EVENT_SCHEMA_VERSION = "multigate_longrun_rollback_v2"


def jsonl_rows_through(
    path: str | Path, timesteps: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Return valid JSONL objects, optionally bounded by their timestep."""
    rows: List[Dict[str, Any]] = []
    source = Path(path)
    if not source.exists():
        return rows
    for line in source.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            row_timesteps = int(
                row.get("timesteps", row.get("num_timesteps", 0))
            )
            if timesteps is None or row_timesteps <= int(timesteps):
                rows.append(dict(row))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return rows


def _rollback_key(event: Mapping[str, Any]) -> Tuple[int, int]:
    return int(event.get("timesteps", 0)), int(event.get("attempt", 0))


def normalize_rollback_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    """Add v2 aliases when they can be inferred without inventing state."""
    normalized = dict(event)
    reasons = normalized.get("reasons")
    if isinstance(reasons, str):
        reasons = [reasons]
    elif reasons is None:
        reason = normalized.get("reason")
        reasons = [reason] if reason else []
    else:
        reasons = list(reasons)
    if reasons:
        normalized["reasons"] = reasons
        normalized.setdefault(
            "reason", reasons[0] if len(reasons) == 1 else "+".join(reasons)
        )
    source = normalized.get("source_checkpoint", normalized.get("source"))
    if source is not None:
        normalized.setdefault("source", source)
        normalized.setdefault("source_checkpoint", source)
    transition = normalized.get("curriculum_transition")
    if isinstance(transition, Mapping):
        normalized.setdefault("stage_before", transition.get("from"))
        normalized.setdefault("stage_after", transition.get("to"))
    if normalized.get("stage_after") is None and normalized.get("stage") is not None:
        normalized["stage_after"] = normalized["stage"]
    if normalized.get("stage_after") is not None:
        normalized.setdefault("stage", normalized["stage_after"])
    return normalized


def merge_rollback_histories(
    *histories: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge duplicate rollback views, preferring every non-null detail."""
    merged: Dict[Tuple[int, int], Dict[str, Any]] = {}
    order: List[Tuple[int, int]] = []
    for history in histories:
        for raw in history or []:
            event = normalize_rollback_event(raw)
            key = _rollback_key(event)
            if key not in merged:
                merged[key] = {}
                order.append(key)
            merged[key].update(
                {name: value for name, value in event.items() if value is not None}
            )
    return [
        normalize_rollback_event(merged[key])
        for key in sorted(order, key=lambda item: (item[0], item[1]))
    ]


def _curriculum_key(change: Mapping[str, Any]) -> Tuple[int, Any, Any, Any]:
    reason = change.get("reason")
    if reason is None and change.get("metrics"):
        reason = "evaluation_promotion"
    return (
        int(change.get("timesteps", 0)),
        change.get("from"),
        change.get("to"),
        reason,
    )


def merge_curriculum_histories(
    *histories: Iterable[Mapping[str, Any]],
    rollback_history: Iterable[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Merge stage histories and transitions embedded in v2 rollback events."""
    merged: Dict[Tuple[int, Any, Any, Any], Dict[str, Any]] = {}
    order: List[Tuple[int, Any, Any, Any]] = []
    for history in histories:
        for raw in history or []:
            change = dict(raw)
            key = _curriculum_key(change)
            if key not in merged:
                merged[key] = {}
                order.append(key)
            merged[key].update(
                {name: value for name, value in change.items() if value is not None}
            )
    for event in rollback_history or []:
        transition = event.get("curriculum_transition")
        if not isinstance(transition, Mapping):
            continue
        change = dict(transition)
        key = _curriculum_key(change)
        if key not in merged:
            merged[key] = {}
            order.append(key)
        merged[key].update(
            {name: value for name, value in change.items() if value is not None}
        )
    return [
        merged[key]
        for key in sorted(order, key=lambda item: (item[0], str(item[1]), str(item[2])))
    ]


def enrich_rollbacks_with_curriculum(
    rollback_history: Iterable[Mapping[str, Any]],
    curriculum_history: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach an unambiguous rollback transition to legacy journal rows."""
    transitions: Dict[int, List[Dict[str, Any]]] = {}
    for raw in curriculum_history or []:
        change = dict(raw)
        if change.get("reason") != "automatic_full_evaluation_rollback":
            continue
        transitions.setdefault(int(change.get("timesteps", 0)), []).append(change)
    enriched = []
    for raw in rollback_history or []:
        event = normalize_rollback_event(raw)
        if not isinstance(event.get("curriculum_transition"), Mapping):
            matches = transitions.get(int(event.get("timesteps", 0)), [])
            stage_after = event.get("stage_after")
            if stage_after is not None:
                matches = [
                    change for change in matches if change.get("to") == stage_after
                ]
            if len(matches) == 1:
                event["curriculum_transition"] = dict(matches[0])
                event["stage_before"] = matches[0].get("from")
                event["stage_after"] = matches[0].get("to")
                event["stage"] = matches[0].get("to")
        enriched.append(event)
    return enriched


def rollback_status_fields(
    rollback_history: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Return status fields that must agree with the latest rollback event."""
    history = merge_rollback_histories(rollback_history)
    if not history:
        return {
            "rollback_count": 0,
            "last_rollback_reason": None,
            "last_rollback_source": None,
            "last_rollback_timestep": None,
            "last_rollback_attempt": None,
            "last_rollback_outcome": None,
            "last_rollback_event": None,
            "rollback_history": [],
        }
    latest = history[-1]
    count = max(int(event.get("attempt", 0)) for event in history)
    return {
        "rollback_count": count,
        "last_rollback_reason": latest.get("reasons"),
        "last_rollback_source": latest.get(
            "source_checkpoint", latest.get("source")
        ),
        "last_rollback_timestep": latest.get("timesteps"),
        "last_rollback_attempt": latest.get("attempt"),
        "last_rollback_outcome": latest.get("outcome"),
        "last_rollback_event": dict(latest),
        "rollback_history": history,
    }
