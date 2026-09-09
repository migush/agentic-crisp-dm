"""Scoped sys.settrace for maads package functions."""
from __future__ import annotations

import inspect
import os
import sys
import time
from types import FrameType
from typing import TYPE_CHECKING, Any

from maads.observability import context as ctx

if TYPE_CHECKING:
    from maads.observability.collector import TraceCollector

_DENYLIST = frozenset({
    "json.dumps", "json.loads", "model_dump", "model_dump_json",
    "append_log", "emit", "emit_start", "emit_end",
})
_MAX_DEPTH = int(os.getenv("MAADS_TRACE_PYTHON_DEPTH", "8"))

# Handled KeyErrors from dependency env lookups (not MAADS bugs).
_IGNORED_EXCEPTION_NAMES = frozenset({"KeyError"})


def _skip_traced_exception(exc_type: type[BaseException], exc_val: BaseException) -> bool:
    if getattr(exc_type, "__name__", "") not in _IGNORED_EXCEPTION_NAMES:
        return False
    msg = str(exc_val).strip("'\"")
    if "token cap" in msg.lower() or "token budget" in msg.lower():
        return True
    return msg in {
        "MAX_TOKENS_PER_RUN",
        "MAADS_TRACE_OTEL",
        "MAADS_TRACE_INCREMENTAL",
    } or "MAX_TOKENS_PER_RUN" in msg


class PythonTracer:
    def __init__(self, collector: TraceCollector) -> None:
        self._collector = collector
        self._prev_trace: Any = None
        self._call_stack: list[tuple[str, float, str]] = []

    def install(self) -> None:
        self._prev_trace = sys.gettrace()
        sys.settrace(self._trace)

    def uninstall(self) -> None:
        sys.settrace(self._prev_trace)
        self._call_stack.clear()

    def _should_trace(self, frame: FrameType) -> bool:
        if not ctx.python_trace_active.get():
            return False
        filename = frame.f_code.co_filename.replace("\\", "/")
        if "/maads/" not in filename:
            return False
        if "/maads/observability/" in filename:
            return False
        qual = f"{frame.f_globals.get('__name__', '')}.{frame.f_code.co_name}"
        for denied in _DENYLIST:
            if denied in qual or denied in frame.f_code.co_name:
                return False
        return True

    def _trace(self, frame: FrameType, event: str, arg: Any) -> Any:
        if event == "call" and self._should_trace(frame):
            depth = len(self._call_stack)
            if depth < _MAX_DEPTH:
                mod = inspect.getmodule(frame)
                qualname = f"{mod.__name__ if mod else '?'}:{frame.f_code.co_name}"
                self._call_stack.append((f"py:{qualname}:{frame.f_lineno}", time.monotonic(), qualname))
                self._collector.emit(
                    "python.call",
                    name=qualname,
                    source="maads.python_tracer",
                    attributes={"line": frame.f_lineno, "depth": depth},
                )
        elif event == "return" and self._call_stack:
            _, t0, qualname = self._call_stack.pop()
            duration_ms = round((time.monotonic() - t0) * 1000, 2)
            self._collector.emit(
                "python.return",
                name=qualname,
                source="maads.python_tracer",
                duration_ms=duration_ms,
                attributes={"duration_ms": duration_ms},
            )
        elif event == "exception" and self._call_stack:
            exc_type, exc_val, _ = arg
            if _skip_traced_exception(exc_type, exc_val):
                return self._trace
            self._collector.emit(
                "exception",
                name=getattr(exc_type, "__name__", "Exception"),
                source="maads.python_tracer",
                attributes={"message": str(exc_val), "qualname": self._call_stack[-1][2]},
            )
        return self._trace
