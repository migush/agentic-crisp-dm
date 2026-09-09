"""Import-time monkey-patches for flow, crew, and tools."""
from __future__ import annotations

import functools
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Callable

from maads.observability import context as ctx
from maads.observability.collector import get_collector
from maads.observability.python_tracer import PythonTracer
from maads.state import SUBSTEP_NAMES, SUBSTEP_OWNER

# Caps for code/stdout/stderr captured into the trace (override via env).
_CODE_TRACE_LIMIT = int(os.getenv("MAADS_TRACE_CODE_LIMIT", "20000"))
_IO_TRACE_LIMIT = int(os.getenv("MAADS_TRACE_IO_LIMIT", "4000"))


def _clip(text: str | None, limit: int) -> str:
    """Keep traces bounded: return text, or a head+tail slice with an elision marker."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    head = text[: limit - 200]
    tail = text[-200:]
    return f"{head}\n... [clipped {len(text) - limit} chars] ...\n{tail}"


def _subprocess_key(code: str) -> str:
    return hashlib.md5(code.encode(), usedforsecurity=False).hexdigest()[:8]


def _flush_trace_snapshot() -> None:
    """Write the current trace to disk without closing the run."""
    out = ctx.export_dir.get()
    if out is None:
        from maads.run_status import flush_status

        flush_status()
        return
    coll = get_collector()
    if coll.run is None:
        from maads.run_status import flush_status

        flush_status()
        return
    from maads.observability.exporter import write_trace_artifacts
    from maads.run_status import flush_status

    write_trace_artifacts(coll, out, finalize=False)
    flush_status()


def apply_patches() -> None:
    _patch_flow()
    _patch_crew()
    _patch_tools()


def _patch_flow() -> None:
    import maads.flow.crisp_dm_flow as flow_mod

    CrispDMFlow = flow_mod.CrispDMFlow
    if getattr(CrispDMFlow.kickoff, "_maads_traced", False):
        return

    orig_kickoff = CrispDMFlow.kickoff

    @functools.wraps(orig_kickoff)
    def traced_kickoff(self, *args: Any, **kwargs: Any):
        # CrispDMFlow.kickoff already emits run.start/end; ensure status flush.
        try:
            return orig_kickoff(self, *args, **kwargs)
        finally:
            from maads.run_status import flush_status

            flush_status()

    traced_kickoff._maads_traced = True  # type: ignore[attr-defined]
    CrispDMFlow.kickoff = traced_kickoff  # type: ignore[method-assign]


def _patch_crew() -> None:
    import maads.crew as crew_mod

    if getattr(crew_mod.run_json_task, "_maads_traced", False):
        return

    from maads.crew import (
        build_task_description,
        pop_last_json_task_meta,
        pop_last_kickoff_output,
    )
    from maads.observability.llm_communications import get_communication_registry
    from maads.output_contracts import validate_agent_output
    from maads.prompts import AGENT_PROMPTS

    def _trace_crew_kickoff(
        agent_name: str,
        instruction: str,
        state: Any,
        schema_hint: str,
        kickoff: Callable[[], Any],
        *,
        span_suffix: str,
        json_enforced: bool = True,
        parsed_ok: Callable[[Any], bool] | None = None,
    ) -> Any:
        coll = get_collector()
        registry = get_communication_registry()
        substep = state.substep
        ctx.current_maads_agent.set(agent_name)
        ctx.current_substep.set(substep)

        description, state_view, agent = build_task_description(
            agent_name, instruction, state, schema_hint, json_enforced=json_enforced,
        )
        model_name = getattr(getattr(agent, "llm", None), "model", None)
        role = getattr(agent, "role", None) or AGENT_PROMPTS.get(agent_name, {}).get("role")

        run = coll.run
        run_id = run.run_id if run else ""
        case_id = run.case_id if run else None

        crew_evt_id = coll.emit_start(
            "crew.start",
            span_key=f"crew.{substep}.{agent_name}.{span_suffix}",
            name=f"Crew kickoff ({agent_name})",
            source="maads.crew",
            attributes={
                "agent_name": agent_name,
                "substep": substep,
                "instruction_preview": instruction[:200],
            },
        )

        comm_id = registry.open_record(
            run_id=run_id,
            case_id=case_id,
            substep=substep,
            agent_name=agent_name,
            role=role,
            model=str(model_name) if model_name else None,
            maads={
                "task_description": description,
                "instruction": instruction,
                "schema_hint": schema_hint,
                "state_view": state_view,
                "state_view_bytes": len(state_view.encode("utf-8")),
            },
            trace_event_id=crew_evt_id,
            parent_comm_id=(
                registry.open_record_ids()[0]
                if registry.open_record_ids()
                else None
            ),
        )

        crew_attrs = {
            "agent_name": agent_name,
            "substep": substep,
            "communication_id": comm_id,
            "prompt_preview": description[:200],
        }
        with coll._lock:
            if coll.run is not None:
                for e in reversed(coll.run.events):
                    if e.id == crew_evt_id:
                        e.attributes.update(crew_attrs)
                        break

        from maads.progress import on_crew_end, on_crew_start

        on_crew_start(agent_name, substep)
        _flush_trace_snapshot()
        t0 = time.monotonic()
        try:
            result = kickoff()
            raw_output, total_tokens = pop_last_kickoff_output()
            duration_ms = round((time.monotonic() - t0) * 1000, 2)
            sizes = registry.preview_sizes(comm_id)

            task_meta = pop_last_json_task_meta() if span_suffix == "json" else None
            response_shape: str | None = None
            if task_meta is not None:
                json_valid = bool(task_meta.get("json_valid"))
                schema_ok = bool(task_meta.get("schema_ok"))
                schema_errors = list(task_meta.get("schema_errors") or [])
                repair = task_meta.get("repair")
                parse_ok = json_valid and schema_ok
                parsed_for_record = result if isinstance(result, dict) else None
            elif span_suffix == "text":
                from maads.codegen import classify_codegen_response

                raw_text = raw_output if raw_output is not None else str(result or "")
                classified = classify_codegen_response(raw_text)
                response_shape = classified["response_shape"]
                json_valid = bool(classified["json_valid"])
                parse_ok = bool(classified["parse_ok"])
                schema_ok = parse_ok
                schema_errors = []
                repair = None
                parsed_for_record = classified.get("parsed_json")
            elif parsed_ok:
                parse_ok = parsed_ok(result)
                json_valid = parse_ok
                schema_ok = parse_ok
                schema_errors = []
                repair = None
                parsed_for_record = result if isinstance(result, dict) else None
                if span_suffix == "json" and isinstance(result, dict):
                    schema_errors = validate_agent_output(
                        agent_name, result, substep=substep,
                    )
                    schema_ok = not schema_errors
                    json_valid = True
                    parse_ok = schema_ok
            else:
                parse_ok = bool(result)
                json_valid = parse_ok
                schema_ok = parse_ok
                schema_errors = []
                repair = None
                parsed_for_record = result if isinstance(result, dict) else None

            registry.close_record(
                comm_id,
                raw_response=raw_output,
                parsed_json=parsed_for_record,
                parse_ok=parse_ok,
                json_valid=json_valid,
                schema_ok=schema_ok,
                schema_errors=schema_errors or None,
                repair=repair,
                response_shape=response_shape,
                tokens={"total": total_tokens},
                duration_ms=duration_ms,
            )

            end_attrs = {
                "agent_name": agent_name,
                "substep": substep,
                "parsed": parse_ok,
                "communication_id": comm_id,
                **sizes,
            }
            coll.emit_end(
                "crew.end",
                span_key=f"crew.{substep}.{agent_name}.{span_suffix}",
                attributes=end_attrs,
            )
            on_crew_end(agent_name, parsed=parse_ok)
            _flush_trace_snapshot()
            return result
        except Exception as exc:
            duration_ms = round((time.monotonic() - t0) * 1000, 2)
            raw_output, total_tokens = pop_last_kickoff_output()
            registry.close_record(
                comm_id,
                raw_response=raw_output,
                parsed_json=None,
                parse_ok=False,
                tokens={"total": total_tokens},
                error=str(exc),
                duration_ms=duration_ms,
            )
            coll.emit(
                "exception",
                name=type(exc).__name__,
                source="maads.crew",
                attributes={"agent_name": agent_name, "message": str(exc), "communication_id": comm_id},
            )
            coll.emit_end(
                "crew.end",
                span_key=f"crew.{substep}.{agent_name}.{span_suffix}",
                attributes={
                    "agent_name": agent_name,
                    "error": str(exc),
                    "communication_id": comm_id,
                },
            )
            on_crew_end(agent_name, parsed=False)
            _flush_trace_snapshot()
            raise

    orig_json = crew_mod.run_json_task

    @functools.wraps(orig_json)
    def traced_run_json_task(
        agent_name: str,
        instruction: str,
        state: Any,
        schema_hint: str = "",
        *,
        artifact_dir: Path | None = None,
    ):
        return _trace_crew_kickoff(
            agent_name,
            instruction,
            state,
            schema_hint,
            lambda: orig_json(
                agent_name, instruction, state, schema_hint, artifact_dir=artifact_dir,
            ),
            span_suffix="json",
            json_enforced=True,
            parsed_ok=lambda result: result is not None,
        )

    orig_text = crew_mod.run_text_task

    @functools.wraps(orig_text)
    def traced_run_text_task(
        agent_name: str,
        instruction: str,
        state: Any,
        expected_output: str = "The requested output.",
    ):
        return _trace_crew_kickoff(
            agent_name,
            instruction,
            state,
            "",
            lambda: orig_text(
                agent_name, instruction, state, expected_output=expected_output,
            ),
            span_suffix="text",
            json_enforced=False,
        )

    traced_run_json_task._maads_traced = True  # type: ignore[attr-defined]
    traced_run_text_task._maads_traced = True  # type: ignore[attr-defined]
    crew_mod.run_json_task = traced_run_json_task
    crew_mod.run_text_task = traced_run_text_task

    # agents.py does `from maads.crew import run_json_task` at import time, so
    # replacing crew_mod.run_json_task alone does not affect agent LLM calls.
    import maads.agents as agents_mod

    if agents_mod.run_json_task is orig_json:
        agents_mod.run_json_task = traced_run_json_task

    # codegen.run_authored_code calls crew.run_text_task via the crew module,
    # so patching crew_mod.run_text_task above covers codegen paths.


def _patch_tools() -> None:
    import maads.tools as tools_mod

    PythonExec = tools_mod.PythonExec
    FileIO = tools_mod.FileIO

    if not getattr(PythonExec.run, "_maads_traced", False):
        orig_pyexec = PythonExec.run

        @functools.wraps(orig_pyexec)
        def traced_pyexec_run(
            self,
            code: str,
            extra_env: dict | None = None,
            *,
            label: str = "",
        ):
            coll = get_collector()
            key = f"pyexec.{_subprocess_key(code)}"
            coll.emit_start(
                "python.subprocess",
                span_key=key,
                name="PythonExec.run",
                source="maads.tools",
                attributes={
                    # Keep the full executed code in the trace: agents now author it,
                    # so it must be inspectable in trace.json / narrative.md.
                    "code": _clip(code, _CODE_TRACE_LIMIT),
                    "code_bytes": len(code),
                    "substep": ctx.current_substep.get(),
                    "agent": ctx.current_maads_agent.get(),
                    "label": label or None,
                },
            )
            from maads.progress import on_code_end, on_code_start

            on_code_start()
            try:
                result = orig_pyexec(self, code, extra_env, label=label)
                coll.emit_end(
                    "python.subprocess",
                    span_key=key,
                    attributes={
                        "ok": result.ok,
                        "return_code": result.return_code,
                        "timed_out": result.timed_out,
                        "stdout": _clip(result.stdout, _IO_TRACE_LIMIT),
                        "stdout_len": len(result.stdout),
                        # stderr only matters when something went wrong; keep it on failure.
                        "stderr": "" if result.ok else _clip(result.stderr, _IO_TRACE_LIMIT),
                        "stderr_len": len(result.stderr),
                    },
                )
                on_code_end(ok=result.ok)
                return result
            except Exception as exc:
                coll.emit(
                    "exception",
                    name=type(exc).__name__,
                    source="maads.tools",
                    attributes={"message": str(exc)},
                )
                coll.emit_end("python.subprocess", span_key=key, attributes={"error": str(exc)})
                on_code_end(ok=False)
                raise

        traced_pyexec_run._maads_traced = True  # type: ignore[attr-defined]
        PythonExec.run = traced_pyexec_run  # type: ignore[method-assign]

    if not getattr(FileIO.write_text, "_maads_traced", False):
        orig_write_text = FileIO.write_text
        orig_write_json = FileIO.write_json

        @functools.wraps(orig_write_text)
        def traced_write_text(self, name: str, content: str):
            get_collector().emit(
                "tool.end",
                name="FileIO.write_text",
                source="maads.tools",
                attributes={"path": name, "bytes": len(content)},
            )
            return orig_write_text(self, name, content)

        @functools.wraps(orig_write_json)
        def traced_write_json(self, name: str, data: Any):
            get_collector().emit(
                "tool.end",
                name="FileIO.write_json",
                source="maads.tools",
                attributes={"path": name},
            )
            return orig_write_json(self, name, data)

        traced_write_text._maads_traced = True  # type: ignore[attr-defined]
        traced_write_json._maads_traced = True  # type: ignore[attr-defined]
        FileIO.write_text = traced_write_text  # type: ignore[method-assign]
        FileIO.write_json = traced_write_json  # type: ignore[method-assign]
