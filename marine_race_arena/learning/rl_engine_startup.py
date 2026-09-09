"""Global HoloOcean engine-startup coordination shared by every launcher.

Two independent problems are solved here.

**Startup slots.**  ``reserve_holoocean_engines`` used to wrap ``SubprocVecEnv``
construction, but ``UniversalTransitionEnv`` creates its adapter lazily on the
first ``reset()``.  The reservation was therefore always released *before* a
single Unreal process existed, so it constrained nothing.  The coordinator here
is acquired around the actual ``holoocean.make()`` call inside the adapter, which
is the only place an engine is really born, and it is shared by PPO, SAC,
evaluators and benchmarks alike because it lives in the system temp directory.

**Startup is serialised; running engines are not.**  A slot is held only while an
engine is starting and being qualified.  Once it answers ticks the slot is
released and the engine keeps running outside any accounting, so throughput of
already-warm workers is untouched.

Waiting for a slot is normal operation and is reported as
``WAITING_FOR_ENGINE_START_SLOT`` -- never as a failure.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

from marine_race_arena.learning.rl_evaluation_lock import (
    backoff_delays,
    is_contention_error,
)

COORDINATOR_VERSION = "rl_engine_startup_v1"
WAITING_STATE = "WAITING_FOR_ENGINE_START_SLOT"
STARTING_STATE = "STARTING_WORKERS"

COORDINATOR_ROOT = Path(tempfile.gettempdir()) / "holodrone_engine_startup"

#: Only this many Unreal engines may be in the STARTING state at once.
DEFAULT_MAX_CONCURRENT_ENGINE_STARTS = 1
#: Quiet period after a successful start before the next one may begin.
DEFAULT_ENGINE_START_STAGGER_SECONDS = 3.0
#: A slot whose owner died or which is impossibly old must not wedge the system.
SLOT_STALE_SECONDS = 600.0

_ENV_MAX_STARTS = "HOLODRONE_MAX_ENGINE_STARTS"
_ENV_STAGGER = "HOLODRONE_ENGINE_START_STAGGER_SECONDS"


def max_concurrent_engine_starts() -> int:
    try:
        value = int(os.environ.get(_ENV_MAX_STARTS, ""))
    except ValueError:
        return DEFAULT_MAX_CONCURRENT_ENGINE_STARTS
    return max(1, value)


def engine_start_stagger_seconds() -> float:
    try:
        value = float(os.environ.get(_ENV_STAGGER, ""))
    except ValueError:
        return DEFAULT_ENGINE_START_STAGGER_SECONDS
    return max(0.0, value)


def _pid_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        import psutil

        return psutil.Process(int(pid)).is_running()
    except Exception:
        return False


def _slot_path(index: int) -> Path:
    return COORDINATOR_ROOT / f"start_slot_{index:02d}.lock"


def _owner_path(index: int) -> Path:
    return COORDINATOR_ROOT / f"start_slot_{index:02d}.owner.json"


def _ensure_slot_file(path: Path) -> None:
    """Create the one-byte slot file without ever reading the locked byte."""

    COORDINATOR_ROOT.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    try:
        with open(path, "xb") as handle:
            handle.write(b"0")
    except FileExistsError:
        return
    except OSError as exc:
        if not is_contention_error(exc):
            raise


def _try_lock(handle) -> bool:
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


def _unlock(handle) -> None:
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


def _write_slot_owner(index: int, owner: str) -> None:
    record = {
        "schema_version": COORDINATOR_VERSION,
        "owner": str(owner),
        "pid": int(os.getpid()),
        "acquired": time.time(),
        "state": STARTING_STATE,
    }
    try:
        _owner_path(index).write_text(json.dumps(record), encoding="utf-8")
    except OSError:
        pass


def _clear_slot_owner(index: int) -> None:
    try:
        _owner_path(index).unlink(missing_ok=True)
    except OSError:
        pass


def read_slot_owners() -> list:
    owners = []
    for index in range(max_concurrent_engine_starts()):
        try:
            owners.append(json.loads(_owner_path(index).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
    return owners


def slot_owner_is_stale(record: Dict[str, Any], *, now: Optional[float] = None) -> bool:
    """A slot whose holder is gone, or which has been held absurdly long."""

    if not record:
        return False
    pid = int(record.get("pid", 0) or 0)
    if pid and not _pid_alive(pid):
        return True
    acquired = float(record.get("acquired", 0.0) or 0.0)
    current = time.time() if now is None else float(now)
    return bool(acquired and current - acquired > SLOT_STALE_SECONDS)


@contextmanager
def engine_start_slot(
    owner: str,
    *,
    slots: Optional[int] = None,
    stagger_seconds: Optional[float] = None,
    on_wait: Optional[Callable[[Dict[str, Any]], None]] = None,
    max_wait_seconds: Optional[float] = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> Iterator[Dict[str, Any]]:
    """Hold one of the global engine-startup slots.

    Contention is never an error: the caller waits with bounded exponential
    backoff and publishes ``WAITING_FOR_ENGINE_START_SLOT`` through ``on_wait``.
    """

    total = int(slots if slots is not None else max_concurrent_engine_starts())
    total = max(1, total)
    quiet = (
        float(stagger_seconds)
        if stagger_seconds is not None
        else engine_start_stagger_seconds()
    )
    for index in range(total):
        _ensure_slot_file(_slot_path(index))

    started = time.time()
    attempt = 0
    handle = None
    acquired_index = -1
    while acquired_index < 0:
        for index in range(total):
            candidate = None
            try:
                candidate = open(_slot_path(index), "r+b")
                if _try_lock(candidate):
                    handle = candidate
                    acquired_index = index
                    break
                candidate.close()
            except OSError as exc:
                if candidate is not None:
                    try:
                        candidate.close()
                    except OSError:
                        pass
                if not is_contention_error(exc):
                    raise
        if acquired_index >= 0:
            break
        attempt += 1
        waited = time.time() - started
        record = {
            "schema_version": COORDINATOR_VERSION,
            "state": WAITING_STATE,
            "owner": str(owner),
            "attempt": attempt,
            "waited_seconds": round(waited, 3),
            "slots": total,
            "current_owners": [row.get("owner") for row in read_slot_owners()],
        }
        if on_wait is not None:
            on_wait(record)
        if max_wait_seconds is not None and waited >= float(max_wait_seconds):
            raise TimeoutError(
                f"engine start slot not acquired after {waited:.1f}s "
                f"(owners {record['current_owners']})"
            )
        sleeper(backoff_delays(attempt)[-1])

    _write_slot_owner(acquired_index, owner)
    outcome: Dict[str, Any] = {
        "schema_version": COORDINATOR_VERSION,
        "state": STARTING_STATE,
        "owner": str(owner),
        "slot": acquired_index,
        "slots": total,
        "waited_seconds": round(time.time() - started, 3),
        "wait_cycles": attempt,
        "stagger_seconds": quiet,
    }
    succeeded = False
    try:
        yield outcome
        succeeded = True
    finally:
        # Stagger only after a *successful* start: a failed candidate should be
        # retried promptly rather than punished with an extra quiet period.
        if succeeded and quiet > 0:
            sleeper(quiet)
        _clear_slot_owner(acquired_index)
        _unlock(handle)
        try:
            handle.close()
        except (OSError, AttributeError):
            pass
