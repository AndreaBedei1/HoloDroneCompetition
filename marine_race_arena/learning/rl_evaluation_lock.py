"""Robust cross-process evaluation lock.

PPO and SAC must never evaluate at the same time, but waiting for the other
algorithm's turn is *normal operation*, not a failure.

The previous implementation initialised the lock file by reading byte 0 before
entering its retry loop.  On Windows ``msvcrt.locking`` places a mandatory range
lock, so that read raised ``PermissionError`` in the waiting process and the
exception propagated out of the trainer.  PPO died on it three times in a row and
the supervisor then gave up, which is how a healthy 534,528-transition run was
lost to pure contention.

Every acquisition path here therefore treats access-denied, sharing-violation and
lock-violation as contention: the caller waits with bounded exponential backoff
plus jitter, reports ``WAITING_FOR_EVALUATION_SLOT`` while it waits, and checks
whether the recorded owner is still alive so a dead owner cannot wedge the system.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

LOCK_VERSION = "rl_evaluation_lock_v1"
WAITING_STATE = "WAITING_FOR_EVALUATION_SLOT"

SCHEDULER_ROOT = Path(tempfile.gettempdir()) / "holodrone_evaluation_scheduler"
LOCK_PATH = SCHEDULER_ROOT / "evaluation.lock"
OWNER_PATH = SCHEDULER_ROOT / "evaluation.owner.json"

BACKOFF_SEQUENCE = (1.0, 2.0, 4.0, 8.0)
BACKOFF_CAP_SECONDS = 20.0
JITTER_FRACTION = 0.25
STALE_OWNER_SECONDS = 3600.0

# Windows surfaces a contended mandatory range lock in several ways depending on
# whether the conflict is detected by the file system or by the C runtime.
_WINDOWS_CONTENTION_WINERRORS = frozenset({5, 32, 33, 167})
_CONTENTION_ERRNOS = frozenset({11, 13, 35, 36})


def is_contention_error(exc: BaseException) -> bool:
    """Whether an OS error means "someone else owns the lock right now"."""

    if isinstance(exc, (PermissionError, BlockingIOError)):
        return True
    if not isinstance(exc, OSError):
        return False
    if getattr(exc, "winerror", None) in _WINDOWS_CONTENTION_WINERRORS:
        return True
    return getattr(exc, "errno", None) in _CONTENTION_ERRNOS


def backoff_delays(
    attempts: int,
    *,
    sequence: tuple = BACKOFF_SEQUENCE,
    cap: float = BACKOFF_CAP_SECONDS,
    jitter: float = JITTER_FRACTION,
    rng: Optional[random.Random] = None,
) -> list:
    """Bounded exponential backoff with jitter, never exceeding ``cap``."""

    generator = rng or random.Random()
    delays = []
    for index in range(int(attempts)):
        base = sequence[index] if index < len(sequence) else cap
        base = min(float(base), float(cap))
        spread = base * float(jitter)
        delays.append(max(0.0, base + generator.uniform(-spread, spread)))
    return delays


def _pid_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        import psutil

        return psutil.Process(int(pid)).is_running()
    except Exception:
        return False


def read_owner() -> Dict[str, Any]:
    try:
        return json.loads(OWNER_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def _write_owner(owner: str) -> None:
    record = {
        "schema_version": LOCK_VERSION,
        "owner": str(owner),
        "pid": int(os.getpid()),
        "acquired_monotonic": time.time(),
    }
    try:
        OWNER_PATH.write_text(json.dumps(record), encoding="utf-8")
    except OSError:
        # The sidecar is diagnostic only; failing to write it must never break
        # an otherwise valid acquisition.
        pass


def _clear_owner() -> None:
    try:
        OWNER_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def owner_is_stale(record: Optional[Dict[str, Any]] = None, *, now: Optional[float] = None) -> bool:
    """A recorded owner whose process is gone, or that is impossibly old."""

    value = read_owner() if record is None else dict(record)
    if not value:
        return False
    pid = int(value.get("pid", 0) or 0)
    if pid and not _pid_alive(pid):
        return True
    acquired = float(value.get("acquired_monotonic", 0.0) or 0.0)
    current = time.time() if now is None else float(now)
    return bool(acquired and current - acquired > STALE_OWNER_SECONDS)


def _try_lock(handle) -> bool:
    """Attempt a non-blocking exclusive lock; False means contention."""

    if os.name == "nt":
        import msvcrt

        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError as exc:
            if is_contention_error(exc):
                return False
            raise
    import fcntl

    try:
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


def _ensure_lock_file() -> None:
    """Create the one-byte lock file without ever reading the locked byte."""

    SCHEDULER_ROOT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        return
    try:
        # Exclusive create: whoever wins writes the byte, everyone else moves on.
        with open(LOCK_PATH, "xb") as handle:
            handle.write(b"0")
    except FileExistsError:
        return
    except OSError as exc:
        if not is_contention_error(exc):
            raise


@contextmanager
def evaluation_lock(
    owner: str,
    *,
    on_wait: Optional[Callable[[Dict[str, Any]], None]] = None,
    max_wait_seconds: Optional[float] = None,
    sleeper: Callable[[float], None] = time.sleep,
    rng: Optional[random.Random] = None,
) -> Iterator[Dict[str, Any]]:
    """Hold the exclusive evaluation turn, waiting politely for contention.

    Never raises on contention.  ``on_wait`` is called on each wait cycle with a
    record whose ``state`` is ``WAITING_FOR_EVALUATION_SLOT``, so the trainer can
    publish that it is queued rather than appearing stalled or crashed.
    """

    _ensure_lock_file()
    started = time.time()
    attempt = 0
    generator = rng or random.Random()
    handle = None
    while True:
        try:
            if handle is None:
                handle = open(LOCK_PATH, "r+b")
            if _try_lock(handle):
                break
            contended = True
            error = None
        except OSError as exc:
            if not is_contention_error(exc):
                raise
            contended = True
            error = repr(exc)
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
                handle = None
        if contended:
            attempt += 1
            waited = time.time() - started
            record = {
                "schema_version": LOCK_VERSION,
                "state": WAITING_STATE,
                "owner": str(owner),
                "attempt": attempt,
                "waited_seconds": round(waited, 3),
                "current_owner": read_owner().get("owner"),
                "current_owner_pid": read_owner().get("pid"),
                "owner_stale": owner_is_stale(),
                "error": error,
            }
            if on_wait is not None:
                on_wait(record)
            if max_wait_seconds is not None and waited >= float(max_wait_seconds):
                if handle is not None:
                    handle.close()
                raise TimeoutError(
                    f"evaluation lock not acquired after {waited:.1f}s "
                    f"(current owner {record['current_owner']!r})"
                )
            delay = backoff_delays(attempt, rng=generator)[-1]
            sleeper(delay)
    _write_owner(owner)
    try:
        yield {
            "schema_version": LOCK_VERSION,
            "owner": str(owner),
            "waited_seconds": round(time.time() - started, 3),
            "wait_cycles": attempt,
        }
    finally:
        _clear_owner()
        _unlock(handle)
        try:
            handle.close()
        except OSError:
            pass
