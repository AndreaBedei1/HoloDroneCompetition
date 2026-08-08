"""Qualify a freshly started HoloOcean engine before it is trusted.

A live ``Holodeck.exe`` proves almost nothing.  The engine may have spawned, won
its loading semaphore and still be unable to deliver sensor packets -- and a
half-initialised engine handed to a learner shows up much later as an
unexplained ``BrokenPipeError`` in the vector-env parent, with no worker-side
traceback, because the worker dies inside the shared-memory client rather than
raising Python-side.

Qualification therefore requires positive evidence: a UUID, a real sensor
packet, a stable (non-repeating) frame sequence and a run of successful ticks.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Optional

#: Ticks the engine must answer before it counts as started.
DEFAULT_QUALIFYING_TICKS = 30
#: How long the whole qualification may take before it is declared failed.
DEFAULT_QUALIFY_TIMEOUT_SECONDS = 120.0


class EngineStartupFailed(RuntimeError):
    """A newly created engine did not qualify as healthy."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class EngineHealthRequirements:
    qualifying_ticks: int = DEFAULT_QUALIFYING_TICKS
    timeout_seconds: float = DEFAULT_QUALIFY_TIMEOUT_SECONDS
    require_uuid: bool = True
    require_sensor_packet: bool = True
    reject_repeated_initial_frame: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_ENGINE_HEALTH = EngineHealthRequirements()


def _state_signature(state: Any) -> Optional[tuple]:
    """A cheap fingerprint used to detect a frozen initial frame."""

    if not isinstance(state, Mapping):
        return None
    signature = []
    for key in sorted(state):
        value = state[key]
        # Arrays fingerprint by their sum; scalars by their own value.  Getting
        # this order wrong makes every frame look identical and would reject a
        # perfectly healthy engine as frozen.
        try:
            signature.append((str(key), float(value)))
            continue
        except (TypeError, ValueError):
            pass
        try:
            signature.append((str(key), float(value.sum())))
        except Exception:
            signature.append((str(key), None))
    return tuple(signature)


def qualify_engine_start(
    env: Any,
    *,
    environment_name: str = "",
    requirements: EngineHealthRequirements = DEFAULT_ENGINE_HEALTH,
    now: Any = time.monotonic,
) -> Dict[str, Any]:
    """Return health diagnostics, or raise :class:`EngineStartupFailed`.

    The engine is ticked ``qualifying_ticks`` times.  Every tick must return a
    state; the sequence must not be a single frozen frame repeated forever.
    """

    started = now()
    diagnostics: Dict[str, Any] = {
        "environment_name": str(environment_name),
        "pid": os.getpid(),
        "stage": "qualify",
        "requirements": requirements.as_dict(),
        "uuid": None,
        "ticks_completed": 0,
        "distinct_states": 0,
        "healthy": False,
    }

    uuid = getattr(env, "_uuid", None)
    diagnostics["uuid"] = None if uuid is None else str(uuid)
    if requirements.require_uuid and not diagnostics["uuid"]:
        raise EngineStartupFailed("engine reported no HoloOcean UUID", diagnostics)

    world_process = getattr(env, "_world_process", None)
    if world_process is not None:
        diagnostics["engine_pid"] = int(getattr(world_process, "pid", 0) or 0)
        if world_process.poll() is not None:
            diagnostics["stage"] = "engine_exited"
            diagnostics["engine_returncode"] = world_process.returncode
            raise EngineStartupFailed(
                "engine process exited during startup", diagnostics
            )

    signatures = set()
    last_state = None
    for index in range(int(requirements.qualifying_ticks)):
        if now() - started > float(requirements.timeout_seconds):
            diagnostics["stage"] = "timeout"
            raise EngineStartupFailed(
                f"engine answered only {index} of "
                f"{requirements.qualifying_ticks} qualifying ticks",
                diagnostics,
            )
        try:
            state = env.tick()
        except Exception as exc:
            diagnostics["stage"] = "tick"
            diagnostics["error"] = f"{type(exc).__name__}: {exc}"
            diagnostics["ticks_completed"] = index
            raise EngineStartupFailed("engine failed while ticking", diagnostics)
        if state is None and requirements.require_sensor_packet:
            diagnostics["stage"] = "sensor_packet"
            diagnostics["ticks_completed"] = index
            raise EngineStartupFailed("engine returned no sensor packet", diagnostics)
        last_state = state
        signature = _state_signature(state)
        if signature is not None:
            signatures.add(signature)
        diagnostics["ticks_completed"] = index + 1

    diagnostics["distinct_states"] = len(signatures)
    diagnostics["sensor_keys"] = (
        sorted(str(key) for key in last_state) if isinstance(last_state, Mapping) else []
    )
    if requirements.require_sensor_packet and not diagnostics["sensor_keys"]:
        diagnostics["stage"] = "sensor_packet"
        raise EngineStartupFailed("engine never delivered a sensor packet", diagnostics)
    if (
        requirements.reject_repeated_initial_frame
        and signatures
        and len(signatures) == 1
        and requirements.qualifying_ticks > 1
    ):
        diagnostics["stage"] = "stale_frame"
        raise EngineStartupFailed(
            "engine repeated a single frozen frame for every qualifying tick",
            diagnostics,
        )

    diagnostics["stage"] = "qualified"
    diagnostics["healthy"] = True
    diagnostics["elapsed_seconds"] = round(float(now() - started), 3)
    return diagnostics
