## Executive summary

From three Titanic runs in `artifacts/titanic/`:

| Run | Duration | Tokens (`status.json`) | Outcome |
|-----|----------|------------------------|---------|
| `2913279c…` (completed) | **~116 min** (08:00→09:56 UTC) | **2.75M** | Workflow finished; ML marked failed |
| `d858314f…` | interrupted at 2.3 | 258k | Ctrl+C at substep 6/24 |
| `c521cf3f…` / `current` | in progress at 2.3 | **1.39M** at 12/24 substeps | Same failure pattern as completed run |

The completed run spent **~88 min** in LLM calls alone (median **101 s/call**, max **633 s**). Tokens and wall time are dominated by **Data Engineer code-generation retries**, **Developer DEBUG repair**, and **duplicate LLM calls per substep** — not by CRISP-DM breadth itself.

---

## What the artifacts show

### Token breakdown (completed run)

```json
"token_spend": {
  "data_engineer": 1647459,  // 60%
  "developer":     488648,   // 18%
  "pm":            364441,   // 13%
  "data_scientist": 201506,  // 7%
  "domain":         51282    // 2%
}
```

`communications.jsonl` only accounts for **1.41M** of **2.75M** tokens. Almost all of the gap is **data_engineer (+1.22M)**:

| Agent | `status.json` | In `communications.jsonl` | Hidden |
|-------|---------------|----------------------------|--------|
| data_engineer | 1,647,459 | 426,424 | **1,221,035** |
| developer | 488,648 | 488,648 | 0 |
| data_scientist | 201,506 | 123,861 | ~78k |

Hidden calls are **`run_text_task` codegen attempts** (author Python → execute → retry). They charge tokens via `_kickoff` but are largely missing from the comm export (see §1 below).

### Retry / DEBUG churn

From the completed run log and `sandbox/exec/manifest.jsonl`:

- **17** `authored code attempt` log lines  
- **7** `data_engineer_attempt1`, **4** `attempt2`, **3** `attempt3`  
- **3×** `developer_debug_data_engineer` (3 attempts each)  
- **8** Developer JSON-repair interventions (`DEBUG json_parse` / `repaired JSON`)  
- **1** baseline fallback at **3.5** (pandas `read_csv` encoding error)

Hot substeps by tokens (from communications):

- **3.5** – 5 calls, 228k tokens  
- **3.4** – 4 calls, 213k tokens  
- **3.2** – 1 comm record but 79k tokens (+ many untraced codegen calls)

The **in-progress run** already shows the same pattern at phase 2:

- **2.2**: 9 LLM calls, **305k tokens**  
- **2.3**: 7 calls, **290k tokens**  
- **developer**: 12 calls, **477k tokens** (mostly DEBUG)

### A false Loop C wasted more time

Timeline from `state.json` log:

1. **09:28** – DS finishes 4.4; `cv_score = 0.795` (above 0.77 threshold)  
2. **09:31** – At substep **5.1**, PM decides **before** DS runs 5.1  
3. PM fires **Loop C → phase 1.3** because `business_goal_met: false`  
4. DS runs **“no-op for 1.3”** — `assessment_of_dm_results` is never set  
5. Run continues to deployment anyway; final `ml_success: false` despite a good model

Root cause: orchestration runs PM at decision substeps **before** the owning agent:

```347:370:src/maads/flow/phase_runner.py
    for substep in subs[start_idx:]:
        ctx.state.substep = substep
        ...
        if substep in PM_DECISION_SUBSTEPS:
            route = handle_plan(ctx, resolve_plan(ctx))  # PM FIRST
            ...
        try:
            if not execute_substep(ctx, substep):       # Agent SECOND
```

At 5.1, PM sees `business_goal_met = bool(assessment.get("meets"))` → **False when assessment is still `None`**:

```382:383:src/maads/state.py
            assessment = self.ev.assessment_of_dm_results or {}
            base["business_goal_met"] = bool(assessment.get("meets"))
```

That triggers a wasteful loop + extra PM/Developer tokens, and leaves `assessment_of_dm_results` permanently unset.

---

## Root causes (code + run data)

### 1. Double LLM call per execution substep

For DE-owned execution substeps (`2.1`, `2.2`, `2.4`, `3.2`–`3.5`), the flow is:

1. **`run_authored_code`** → `run_text_task` (generate Python, up to 3 retries)  
2. **`_crew.kickoff_substep`** → `run_json_task` (interpret results into state)

```159:176:src/maads/agents.py
        execution = _de_execution_evidence(...)   # codegen LLM calls
        ...
        data = self._crew.kickoff_substep(...)    # second LLM call
        return _apply_data_engineer_response(data, state, s, execution)
```

Same pattern for DS on `2.3`, `4.3`, `4.4`. That roughly **doubles** LLM cost for the heaviest agents.

### 2. Retry stack multiplies failures

```27:27:src/maads/codegen.py
MAX_CODE_RETRIES = 3
```

```17:17:src/maads/debug.py
MAX_DEBUG_RETRIES = 3
```

On failure: **3 specialist retries** (each resends prior code + error) → **3 Developer DEBUG attempts** → baseline fallback → often **Developer JSON repair** on the follow-up `kickoff_substep`. A single bad substep can cost **7–10+ LLM calls**.

Retry prompts grow because `_build_instruction` embeds the full failing code and stderr:

```100:107:src/maads/codegen.py
        parts += [
            "",
            "Your previous attempt FAILED. Fix it. Previous code:",
            "```python",
            prior_code,
            "```",
            f"Error / reason it failed:\n{prior_error}",
        ]
```

### 3. Codegen tracing import-order bug (blind spot)

`__main__.py` imports `CrispDMFlow` (which pulls in `codegen`) **before** `auto_enable()` patches `run_text_task`:

```29:42:src/maads/__main__.py
from maads.flow.crisp_dm_flow import CrispDMFlow   # imports codegen → binds old run_text_task
...
auto_enable()                                      # patches crew.run_text_task too late
```

`patches.py` only re-exports `run_json_task` to `agents.py`, not `run_text_task`:

```318:323:src/maads/observability/patches.py
    if agents_mod.run_json_task is orig_json:
        agents_mod.run_json_task = traced_run_json_task
```

So **~1.2M DE tokens** burn without appearing in `communications.jsonl`, making optimization harder to see.

### 4. Large fixed system prompts on every call

Backstory sizes:

| Agent | Backstory |
|-------|-----------|
| data_engineer | **16.7 KB** |
| developer | 12.7 KB |
| pm | 10.8 KB |
| data_scientist | 10.1 KB |

These are sent on **every** kickoff (~3–5k tokens before task content). PM’s comm record shows **10,701 chars** of system prompt alone.

### 5. Local Ollama latency

All tokens are `ollama`. Communications show **median 101 s/call**; one 5.2 PM call took **633 s** for only 5,615 tokens. Wall time is often model speed, not orchestration logic.

### 6. JSON fragility → Developer tax

Completed run: Developer = **488k tokens (18%)** — almost entirely DEBUG JSON repair for PM, DE, DS. Current run: **477k Developer tokens before substep 13/24**.

Small/local models frequently return non-JSON or schema-invalid JSON, triggering `debug_json_parse` → another full Developer LLM call.

### 7. No default token cap

`MAX_TOKENS_PER_RUN` exists but is commented out in `.env.example`. Nothing stops a 2.75M-token Titanic run.

---

## Proposed optimisations (prioritised)

### P0 — High impact, structural

**1. Collapse execution substeps to one LLM call (or zero)**  
For deterministic tasks (describe CSV, clean parquet, train logistic regression on Titanic), skip the second `kickoff_substep` when `execution` payload already satisfies the contract. Apply results mechanically from `execution_evidence` (you already do this partially in `_apply_*_response`).

**2. Fix PM-before-agent ordering at 5.1**  
Either:
- Run DS 5.1 **before** PM decides at 5.1, or  
- Remove 5.1 from `PM_DECISION_SUBSTEPS`, or  
- Treat `assessment is None` as “not yet evaluated”, not `business_goal_met: false`.

Also compute `business_goal_met` deterministically from `chosen_model.cv_score` vs threshold when assessment is missing.

**3. Fix import order for traced `run_text_task`**  
Call `auto_enable()` before any import that transitively loads `codegen`, and patch `maads.codegen.run_text_task` (or use a lazy import in `run_authored_code`). Restores visibility into ~45% of token spend.

**4. Tiered retry budget**  
e.g. `MAX_CODE_RETRIES=1` for describe/collect/quality; keep 3 only for 3.2–3.5. Cap DEBUG at 1 for “simple” error classes (`syntax_error`, `json_parse`). Fall back to baselines faster on Titanic-scale data.

### P1 — Large token savings

**5. Shorter backstories for code agents**  
Split `data_engineer.md` (460 lines) into a **short runtime persona** (~2 KB) + reference doc not injected every call. Same for developer/PM.

**6. Slim codegen retry prompts**  
On retry, send **diff context** (error class + last 30 lines of stderr + column list), not full prior code. Or use a smaller `MODEL_CODE` only for retries.

**7. Deterministic JSON repair before Developer**  
You already have `_extract_json` / `_repair_json` in `crew.py`. Extend schema repair (fill defaults from execution evidence) before invoking Developer DEBUG.

**8. Skip LLM for PM-owned mechanical substeps**  
`1.4`, `5.2`, `5.3` already have fallbacks in `agents.py`; consider making the LLM optional when state is sufficient.

### P2 — Runtime / ops

**9. Set `MAX_TOKENS_PER_RUN`**  
e.g. 500k–1M for dev; forces early halt instead of 2.75M runs.

**10. Faster models for JSON roles**  
Use a cloud `MODEL_JSON` for PM/domain/structured steps; keep `MODEL_CODE` local. Or use `gemma2:2b` for PM decisions only.

**11. Baseline-first on known-simple cases**  
For bundled cases like Titanic, run baselines in `_de_execution_evidence` first; call LLM only if baseline contract fails or user requests exploration.

**12. Reduce observability payload**  
`MAADS_TRACE_LLM_IO=preview` (or `off`) — `communications.jsonl` is **1.9 MB for 38 records** because full prompts are stored. Doesn’t reduce LLM tokens but speeds I/O and disk.

### P3 — Correctness fixes that also save tokens

**13. Don’t fire Loop B on `degraded_flags` at 5.1 if CV meets threshold**  
A 3.5 baseline fallback still produced `cv_score 0.795`. Tune `_suggested_pm_action` to weigh model outcome, not only degradation flags.

**14. Pre-validate codegen contracts**  
Many DE failures are schema shape issues (`columns` as parallel lists vs dict). Inject a **tiny contract example** in the first attempt only, not on every retry.

**15. Cache idempotent substeps across loops**  
Loop C re-entered 1.3 as no-op but still ran PM LLM. Skip LLM when substep output unchanged and inputs identical.

---

## Expected impact (rough)

| Change | Est. token reduction | Est. time reduction |
|--------|---------------------|---------------------|
| Single LLM call per execution substep | **30–40%** | 30–40% |
| Retry budget 3→1 + faster fallback | **20–35%** on failing substeps | Large |
| Fix false Loop C + assessment ordering | **5–15%** + correct `ml_success` | 10–20 min |
| Shorter backstories | **10–15%** | 10–15% |
| Faster/cloud JSON model | modest tokens | **50–70%** wall time |

Combined, a Titanic run could plausibly drop from **~2.75M / ~2 h** to **~400–800k / ~20–40 min** without changing CRISP-DM coverage.

---

## Reasoning-tier cost on Data Engineer codegen (task20 / gpt-5.5-pro)

Disaster-tweets / task20 communications show Data Engineer **codegen** calls where
`tokens.total` is far larger than prompt + completion reported in the transcript.
That gap is typical of **hidden reasoning tokens** on reasoning-tier models such as
`gpt-5.5-pro`: the API bills internal chain-of-thought that never appears as visible
completion text.

Prompt slimming (this change set) cuts repeated backstory/state/schema waste, but it
does **not** remove reasoning-tier overhead on `run_text_task` authoring. When tuning
cost for code-authoring only, evaluate:

- lower `reasoning_effort` for DE/developer codegen paths, and/or
- a non-reasoning / cheaper code model for `MODEL_CODE` / DE text tasks

Keep JSON planning/decision roles on the stronger model if quality requires it.
**No default model or env change** is recommended here — this is a cost/quality dial
for operators after measuring a live run.

---

## What to inspect next

If you want to dig deeper in artifacts:

- `artifacts/titanic/archive/2913279c…/state.json` — full retry/loop chronology  
- `artifacts/titanic/archive/2913279c…/sandbox/exec/` — failing generated code  
- `artifacts/titanic/current/collected/communications.jsonl` — live 2.2/2.3 DEBUG spiral  

I’m in **Ask mode**, so I can’t apply fixes. Switch to **Agent mode** if you want me to implement the P0 items (import-order fix, 5.1 ordering, single-call execution path).