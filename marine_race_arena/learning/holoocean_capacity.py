"""Cross-run HoloOcean engine accounting with a strict global cap."""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping


MAX_ACTIVE_HOLOOCEAN_ENGINES = 10
_REGISTRY = Path(tempfile.gettempdir()) / "holodrone_holoocean_capacity"
_LOCK = _REGISTRY / "registry.lock"
_RESERVATIONS = _REGISTRY / "reservations.json"


def active_holodeck_processes() -> list[Dict[str, Any]]:
    """Return live Holodeck ownership data without mutating any process."""

    try:
        import psutil
    except ImportError:  # pragma: no cover - diagnostics dependency
        return []
    rows = []
    for process in psutil.process_iter(
        attrs=("pid", "ppid", "name", "cmdline", "create_time")
    ):
        try:
            if str(process.info.get("name") or "").lower() != "holodeck.exe":
                continue
            command = " ".join(process.info.get("cmdline") or [])
            uuid = None
            marker = "--HolodeckUUID="
            for token in command.split():
                if token.startswith(marker):
                    uuid = token[len(marker):]
                    break
            rows.append({
                "pid": int(process.info["pid"]),
                "parent_pid": int(process.info.get("ppid") or 0),
                "uuid": uuid,
                "created": float(process.info.get("create_time") or 0.0),
                "command_line": command,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(rows, key=lambda row: row["pid"])


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(int(pid))
    except ImportError:  # pragma: no cover
        return int(pid) == os.getpid()


@contextmanager
def _registry_lock() -> Iterator[None]:
    _REGISTRY.mkdir(parents=True, exist_ok=True)
    with _LOCK.open("a+b") as handle:
        handle.seek(0)
        if handle.tell() == 0 and handle.read(1) == b"":
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - Windows is the production platform
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_reservations() -> list[Dict[str, Any]]:
    try:
        rows = json.loads(_RESERVATIONS.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return []
    return [
        dict(row) for row in rows
        if _pid_alive(int(row.get("pid", 0)))
        and time.time() - float(row.get("created", 0.0)) < 300.0
    ]


def _write_reservations(rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = _RESERVATIONS.with_suffix(".tmp")
    temporary.write_text(json.dumps(list(rows), indent=2), encoding="utf-8")
    os.replace(temporary, _RESERVATIONS)


def capacity_snapshot() -> Dict[str, Any]:
    with _registry_lock():
        reservations = _read_reservations()
        _write_reservations(reservations)
    engines = active_holodeck_processes()
    return {
        "maximum": MAX_ACTIVE_HOLOOCEAN_ENGINES,
        "active": len(engines),
        "reserved": sum(int(row["count"]) for row in reservations),
        "available": max(
            0,
            MAX_ACTIVE_HOLOOCEAN_ENGINES
            - len(engines)
            - sum(int(row["count"]) for row in reservations),
        ),
        "engines": engines,
        "reservations": reservations,
    }


@contextmanager
def reserve_holoocean_engines(
    count: int,
    *,
    owner: str,
    maximum: int = MAX_ACTIVE_HOLOOCEAN_ENGINES,
) -> Iterator[Dict[str, Any]]:
    """Atomically reserve launch slots, then release after engines exist."""

    requested = int(count)
    if requested < 0:
        raise ValueError("engine reservation cannot be negative")
    reservation = {
        "pid": os.getpid(),
        "count": requested,
        "owner": str(owner),
        "created": time.time(),
        "token": f"{os.getpid()}-{time.time_ns()}",
    }
    with _registry_lock():
        rows = _read_reservations()
        active = len(active_holodeck_processes())
        reserved = sum(int(row["count"]) for row in rows)
        if active + reserved + requested > int(maximum):
            raise RuntimeError(
                f"HoloOcean engine cap would be exceeded: active={active}, "
                f"reserved={reserved}, requested={requested}, maximum={maximum}. "
                "Use the geometry preview backend or reduce workers."
            )
        rows.append(reservation)
        _write_reservations(rows)
    try:
        yield {
            "active_before": active,
            "reserved_before": reserved,
            "requested": requested,
            "maximum": int(maximum),
            "owner": str(owner),
        }
    finally:
        with _registry_lock():
            rows = [
                row for row in _read_reservations()
                if row.get("token") != reservation["token"]
            ]
            _write_reservations(rows)


def assert_unique_holoocean_uuids(
    identities: Iterable[Mapping[str, Any]],
) -> None:
    values = [str(row.get("holoocean_uuid") or "") for row in identities]
    if any(not value for value in values):
        raise RuntimeError("a HoloOcean worker did not report its UUID")
    if len(set(values)) != len(values):
        raise RuntimeError(f"duplicate HoloOcean UUID ownership: {values}")

