"""Wall-clock deadline so a run can halt cleanly before a parent kill."""
from __future__ import annotations

import os
import time
from typing import Any

HALT_REASON = "wall_clock_deadline_exceeded"

_started_mono: float | None = None


def _parse_sec(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def deadline_sec() -> float | None:
    """Seconds allowed for this process, or None when unset/disabled."""
    return _parse_sec("MAADS_RUN_DEADLINE_SEC")


def start_deadline_clock() -> None:
    """Mark the start of the run. No-op when no deadline is configured."""
    global _started_mono
    if deadline_sec() is None:
        _started_mono = None
        return
    _started_mono = time.monotonic()


def reset_deadline_clock() -> None:
    """Clear the clock (tests)."""
    global _started_mono
    _started_mono = None


def deadline_exceeded() -> bool:
    cap = deadline_sec()
    if cap is None:
        return False
    started = _started_mono
    if started is None:
        return False
    return (time.monotonic() - started) >= cap


def deadline_status() -> dict[str, Any]:
    cap = deadline_sec()
    started = _started_mono
    elapsed = None if started is None else time.monotonic() - started
    remaining = None
    if cap is not None and elapsed is not None:
        remaining = max(cap - elapsed, 0.0)
    return {
        "cap_sec": cap,
        "elapsed_sec": elapsed,
        "remaining_sec": remaining,
        "exceeded": deadline_exceeded(),
    }
