"""Run agent-authored Python with validation, self-debug retry, and optional fallback.

The owning agent's LLM proposes Python code; `PythonExec` runs it; the code must
print a single JSON line matching a *contract*. On failure (crash, timeout, or a
contract violation) we feed the captured stderr back to the agent for a revision
(self-debug) up to a retry budget. DE/DS/Developer execution substeps have no
baseline fallback — a failed generation surfaces as a halt the PM can respond to.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import maads.crew as crew
from maads.state import CrispDMState
from maads.token_budget import (
    TokenBudgetExceeded,
    code_retries,
    repairs_allowed,
    soft_limit_reached,
)
from maads.tools import ExecResult, PythonExec

MAX_CODE_RETRIES = 3

# A contract validates the JSON the authored code printed: returns a list of
# human-readable errors (empty list == the payload satisfies the contract).
Contract = Callable[[dict], list[str]]


@dataclass
class CodeResult:
    payload: dict[str, Any]          # contract-valid JSON the code printed
    code: str                        # the code that produced it (authored or fallback)
    attempts: int                    # how many authoring attempts were made
    degraded: bool = False           # True when we fell back to the fixed snippet
    error: str | None = None         # last failure reason (when degraded)

    @property
    def ok(self) -> bool:
        return not self.degraded


_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_+\-]+)?[ \t]*\r?\n?(.*?)```", re.DOTALL)


def _strip_fence(text: str) -> str:
    """Return the contents of the first ```python fence, or the text unchanged."""
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _compiles(src: str) -> bool:
    try:
        compile(src, "<authored>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


_CODE_KEYS = ("code", "python", "source", "script", "content")
_BLOCKED_STATUSES = frozenset({"BLOCKED", "ERROR", "REFUSED"})


def _codegen_response_blocked(raw: str) -> str | None:
    """Return a refusal message when the model returned JSON instead of a code fence."""
    stripped = raw.strip()
    if not stripped.startswith("{"):
        return None
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    status = str(obj.get("status", "")).upper()
    if status not in _BLOCKED_STATUSES:
        return None
    return str(obj.get("message") or f"LLM returned status={status}")


def classify_codegen_response(raw: str) -> dict[str, Any]:
    """Classify a codegen ``run_text_task`` response for observability."""
    from maads.crew import _extract_json

    text = str(raw or "")
    stripped = text.strip()
    parsed = _extract_json(stripped) if stripped else None
    code_extractable = _extract_code(text) is not None

    if stripped.startswith("```"):
        shape = "markdown_fence"
    elif parsed is not None and any(
        isinstance(parsed.get(key), str) and parsed.get(key, "").strip()
        for key in _CODE_KEYS
    ):
        shape = "json_code_wrapper"
    elif parsed is not None:
        shape = "other_json"
    else:
        shape = "plain_text"

    parsed_json: dict[str, Any] | None = None
    if parsed is not None:
        parsed_json = parsed
    elif code_extractable and stripped.startswith("```"):
        parsed_json = {"code": _strip_fence(stripped)[:500]}

    return {
        "response_shape": shape,
        "parsed_json": parsed_json,
        "json_valid": parsed is not None,
        "parse_ok": code_extractable,
    }


def _unwrap_json_code(text: str) -> str | None:
    """If `text` is JSON the model wrapped the code in, return the Python.

    gpt-class models occasionally return ``{"code": "import ...\\n..."}`` or a
    bare JSON-escaped string instead of a code block. Returns None when `text`
    is not such a wrapper, so callers can fall through to the raw text.
    """
    stripped = text.strip()
    if not stripped or stripped[0] not in '{["':
        return None
    try:
        obj = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(obj, str):
        code = obj
    elif isinstance(obj, dict):
        code = next(
            (
                obj[key]
                for key in ("code", "python", "source", "script", "content")
                if isinstance(obj.get(key), str) and obj[key].strip()
            ),
            None,
        )
        if code is None:
            return None
    else:
        return None
    return _strip_fence(code) or None


def _extract_code(text: str) -> str | None:
    """Pull runnable Python out of LLM output, tolerating several shapes.

    Order: strip a ```python fence, then prefer the candidate as-is if it already
    compiles; otherwise try to unwrap JSON-wrapped code (``{"code": "..."}`` or a
    JSON string), and finally decode literal ``\\n`` escapes — each only when it
    yields compilable Python. Falls back to the raw text so a genuinely broken
    response still surfaces as a normal exec failure for Developer DEBUG.
    """
    if not text or not text.strip():
        return None
    candidate = _strip_fence(text)
    if not candidate:
        return None

    # 1. Normal: already valid Python (but not a JSON object masquerading as a
    #    dict literal — those compile yet aren't the intended code).
    if candidate[0] not in '{["' and _compiles(candidate):
        return candidate

    # 2. Model wrapped the code in JSON ({"code": ...} or a JSON string).
    unwrapped = _unwrap_json_code(candidate)
    if unwrapped is not None:
        return unwrapped

    # 3. Code arrived with literal \n escapes (e.g. escaped inside a fence) and
    #    won't compile — decode the escapes if that produces valid Python.
    if "\\n" in candidate and not _compiles(candidate):
        try:
            decoded = candidate.encode("utf-8").decode("unicode_escape")
        except (UnicodeDecodeError, ValueError):
            decoded = candidate
        if _compiles(decoded):
            return decoded

    return candidate


def _header(header_vars: dict[str, Any], *, helpers: str = "") -> str:
    """Render predefined variables the authored code may use, safely quoted."""
    lines = [
        "# --- injected by maads; do not redefine ---",
        "import json",
        "from pathlib import Path",
        "import pandas as pd",
    ]
    for name, value in header_vars.items():
        lines.append(f"{name} = {value!r}")
    if helpers.strip():
        lines.append(helpers.strip())
    return "\n".join(lines) + "\n"


def _last_json_line(stdout: str) -> dict | None:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def _build_instruction(
    base: str,
    header_vars: dict[str, Any],
    contract_hint: str,
    prior_code: str | None,
    prior_error: str | None,
) -> str:
    var_list = ", ".join(header_vars.keys())
    parts = [
        base,
        "",
        "Write Python 3 code to accomplish the task above.",
        f"These variables are ALREADY DEFINED for you (do not redefine): {var_list}.",
        "DATASET_INSPECT_JSON is a JSON string from inspect_dataset — parse with json.loads if needed.",
        "Use ONLY the injected TRAIN_CSV / TEST_CSV / SOURCE_PATHS (and other header vars). "
        "Do NOT call os.walk, pathlib.rglob, or recursively search under data/ to find files.",
        "Your code MUST finish by printing exactly one line to stdout: a single "
        f"JSON object. {contract_hint}",
        "Return ONLY a ```python ...``` code block — no prose before or after.",
    ]
    if prior_code and prior_error:
        parts += [
            "",
            "Your previous attempt FAILED. Fix it. Previous code:",
            "```python",
            prior_code,
            "```",
            f"Error / reason it failed:\n{prior_error}",
        ]
    return "\n".join(parts)


def run_authored_code(
    *,
    pyexec: PythonExec,
    agent_name: str,
    instruction: str,
    state: CrispDMState,
    header_vars: dict[str, Any],
    contract: Contract,
    contract_hint: str = "",
    header_helpers: str = "",
    fallback: Callable[[], dict] | None = None,
    fallback_code: str = "<fixed baseline snippet>",
    max_retries: int = MAX_CODE_RETRIES,
    artifact_dir: Path | None = None,
) -> CodeResult:
    """Have `agent_name` author Python, run it, validate, self-debug, then fall back.

    `header_vars` are injected as predefined variables before the agent's code.
    `contract` validates the JSON the code prints. `fallback` (if given) runs the
    fixed baseline when all authoring attempts fail.
    """
    header = _header(header_vars, helpers=header_helpers)
    if header_helpers.strip():
        instruction = (
            f"{instruction} Use injected helpers drop_feature_columns(df) and "
            "text_vectorizer_pipeline(**kwargs) when building text pipelines."
        )
    prior_code: str | None = None
    prior_error: str | None = None
    last_exec: ExecResult | None = None
    attempt_budget = min(
        max_retries,
        code_retries(soft=soft_limit_reached(state)),
    )
    if attempt_budget <= 0:
        attempt_budget = 1

    for attempt in range(1, attempt_budget + 1):
        prompt = _build_instruction(
            instruction, header_vars, contract_hint, prior_code, prior_error
        )
        try:
            raw = crew.run_text_task(agent_name, prompt, state, expected_output="A Python code block.")
        except TokenBudgetExceeded:
            raise
        except (crew.CrewKickoffError, RuntimeError) as exc:
            prior_error = f"LLM call failed: {exc}"
            state.append_log(agent_name, f"authored code attempt {attempt} -> LLM error", level="warn")
            continue

        blocked = _codegen_response_blocked(raw)
        if blocked:
            prior_error = blocked
            state.append_log(
                agent_name,
                f"authored code attempt {attempt} -> blocked by JSON response conflict",
                level="warn",
            )
            continue

        code = _extract_code(raw)
        if not code:
            prior_error = "no Python code block found in the response"
            state.append_log(agent_name, f"authored code attempt {attempt} -> no code", level="warn")
            continue

        full_code = header + code
        res = pyexec.run(full_code, label=f"{agent_name}_attempt{attempt}")
        last_exec = res
        if not res.ok:
            prior_code = code
            prior_error = (res.stderr or "").strip()[-1500:] or "non-zero exit, no stderr"
            state.append_log(agent_name, f"authored code attempt {attempt} -> failed", level="warn")
            continue

        payload = _last_json_line(res.stdout)
        if payload is None:
            prior_code = code
            prior_error = "code ran but printed no JSON object on its last line"
            state.append_log(agent_name, f"authored code attempt {attempt} -> no JSON output", level="warn")
            continue

        errors = contract(payload)
        if errors:
            prior_code = code
            prior_error = "output failed validation: " + "; ".join(errors)
            state.append_log(agent_name, f"authored code attempt {attempt} -> invalid output", level="warn")
            continue

        state.append_log(agent_name, f"authored code attempt {attempt} -> ok")
        return CodeResult(payload=payload, code=full_code, attempts=attempt)

    # Specialist retries exhausted — route to Developer DEBUG before baseline fallback.
    if prior_code and artifact_dir is not None and repairs_allowed(state):
        from maads.debug import debug_python_exec

        debug_out = debug_python_exec(
            pyexec=pyexec,
            state=state,
            artifact_dir=artifact_dir,
            requesting_agent=agent_name,
            failing_code=prior_code,
            header_vars=header_vars,
            contract=contract,
            contract_hint=contract_hint,
            last_error=prior_error or "",
            last_exec=last_exec,
        )
        if debug_out.status == "FIXED" and debug_out.payload is not None:
            return CodeResult(
                payload=debug_out.payload,
                code=debug_out.code or header + prior_code,
                attempts=max_retries + len(debug_out.fix_attempts),
                degraded=False,
            )

    # DEBUG stuck or unavailable -> fall back (degraded).
    if fallback is not None:
        state.append_log(
            agent_name,
            f"authored code exhausted {max_retries} attempts; using fallback. "
            f"last error: {(prior_error or '')[:300]}",
            level="warn",
        )
        from maads.capabilities.shared import record_degraded

        record_degraded(
            state,
            state.substep or "?",
            agent_name,
            prior_error or f"fell back to {fallback_code}",
        )
        payload = fallback()
        return CodeResult(
            payload=payload,
            code=fallback_code,
            attempts=max_retries,
            degraded=True,
            error=prior_error,
        )

    raise crew.CrewKickoffError(
        f"{agent_name} authored code failed after {max_retries} attempts "
        f"and no fallback was provided: {prior_error}"
    )