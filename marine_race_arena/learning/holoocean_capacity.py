"""Cross-run HoloOcean engine accounting with a strict global cap."""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping

from marine_race_arena.learning.rl_evaluation_lock import (
    backoff_delays,
    is_contention_error,
)


MAX_ACTIVE_HOLOOCEAN_ENGINES = 10
_REGISTRY = Path(tempfile.gettempdir()) / "holodrone_holoocean_capacity"
_LOCK = _REGISTRY / "registry.lock"
_RESERVATIONS = _REGISTRY / "reservations.json"
_REGISTRY_LOCK_TIMEOUT_SECONDS = 120.0


def _try_exclusive_lock(handle) -> bool:
    """Non-blocking exclusive lock; ``False`` means another process holds it."""

    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if is_contention_error(exc):
            return False
        raise


def _release_lock(handle) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


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
    """Serialise registry access without ever reading the locked byte.

    The previous implementation read byte 0 to decide whether the file needed
    initialising.  ``msvcrt.locking`` is a *mandatory* range lock on Windows, so
    that read raised ``PermissionError`` in whichever process arrived second --
    the same defect that killed PPO through the evaluation lock.  The byte is now
    created with an exclusive ``open(..., "xb")`` and never read again.
    """

    _REGISTRY.mkdir(parents=True, exist_ok=True)
    if not _LOCK.exists():
        try:
            with open(_LOCK, "xb") as creator:
                creator.write(b"0")
        except FileExistsError:
            pass
        except OSError as exc:
            if not is_contention_error(exc):
                raise
    with open(_LOCK, "r+b") as handle:
        started = time.time()
        attempt = 0
        while not _try_exclusive_lock(handle):
            attempt += 1
            if time.time() - started > _REGISTRY_LOCK_TIMEOUT_SECONDS:
                raise TimeoutError(
                    "HoloOcean capacity registry lock not acquired after "
                    f"{time.time() - started:.1f}s"
                )
            time.sleep(backoff_delays(attempt)[-1])
        try:
            yield
        finally:
            _release_lock(handle)


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

