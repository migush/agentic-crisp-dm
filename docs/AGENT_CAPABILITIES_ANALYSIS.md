# Agent capability gap analysis

Scope: what **tools, skills, and capabilities** could be added to `maads` to raise
CRISP-DM output quality (better models, sounder evaluation, fewer false loops) and
to make the six-agent orchestration itself more reliable and efficient. This is a
capability-gap analysis, not a token/latency audit — see `docs/Optimisation_Research.md`
for the token/time breakdown and `docs/task20-unified-refactor.md` for the reliability
fixes already in flight. Findings here are additive to both.

Method: read `state.py`, `agents.py`, `flow/*`, `capabilities/*`, `crew_tools.py`,
`codegen.py`, `tools.py`, `rag.py`, `validators.py`, `success_criterion.py`,
`experience_ledger.py`, `model_capabilities.py`, and all seven `skills/*/SKILL.md` files.

---

## 1. What the system already has (inventory)

| Capability | Where | Notes |
|---|---|---|
| Sandboxed Python execution | `tools.py::PythonExec` | Subprocess, wall-clock timeout, scripts kept under `sandbox/exec/` for audit |
| Freeform code authoring + retry | `codegen.py::run_authored_code` | LLM writes a full script per substep; `MAX_CODE_RETRIES = 3`, resends full prior code + stderr on retry |
| Structured CrewAI tools | `crew_tools.py` | Only **two**: `read_case_config_summary`, `validate_submission_file` |
| RAG retrieval | `rag.py` | Markdown corpus in `knowledge/<case>/`, embeddings via Ollama/OpenAI; wired to **Domain agent prompts only** |
| Post-hoc artifact validators | `validators.py` | `validate_phase_3_artifacts`, `validate_phase_4_models` — re-read the parquet/model state from disk and turn deficits into Loop B signals instead of trusting the LLM's self-report |
| Canonical success-criterion math | `success_criterion.py` | Deterministic minimize/maximize + threshold logic, used to normalize the LLM's `assessment_of_dm_results` |
| Structured-output capability probing | `model_capabilities.py` | Detects per-model JSON mode vs strict Structured Outputs support |
| LLM-param experience ledger | `experience_ledger.py` | **Write-only** — records model/effort/token/outcome per run to `llm_param_ledger.jsonl`; explicitly "no auto-tuning yet" |
| Declarative skills | `skills/*/SKILL.md` | 7 files, 7–12 lines each: loop contours, Kaggle contract, leakage/CV discipline, tabular/NLP prep, JSON contract, debug rubric |

The design intent is sound — deterministic verification (`validators.py`,
`success_criterion.py`) already exists precisely so the system doesn't have to trust
an LLM's word for it. The gap is that this pattern is applied to only two of six
phases, and the "give the agent a tool" pattern (`crew_tools.py`) is applied to only
two trivial, non-modeling operations.

---

## 2. Agent-facing ML/data-science tool gaps

### 2.1 Deterministic ML operations are still freeform LLM code (highest impact)

Every Data Engineer and Data Scientist execution substep (`2.1`–`2.4`, `3.2`–`3.5`,
`4.3`–`4.4`) has the LLM **author a full Python script from scratch** via
`run_authored_code`, rather than calling a tested tool function:

```27:src/maads/codegen.py
MAX_CODE_RETRIES = 3
```

Even known, recurring gotchas are patched by growing the prompt instead of fixing
the tool:

```45:60:src/maads/capabilities/data_scientist.py
def _text_modeling_hint(state: CrispDMState) -> str:
    ...
    " Text modeling: if using TfidfVectorizer inside ColumnTransformer, "
    "ColumnTransformer passes 2-D arrays — add "
    "FunctionTransformer(lambda x: x.ravel().astype(str), validate=False) "
    "before TfidfVectorizer, or vectorize train[text_col].astype(str) directly. ..."
```

That is a workaround for a bug class, expressed as prose the model may or may not
follow correctly — instead of a `build_text_pipeline()` tool that gets it right
every time. This is the single largest lever for both **quality** (fewer failed
sklearn pipelines, no leakage from ad hoc transforms) and **efficiency** (fewer
retries — this is the same root cause `Optimisation_Research.md` §1–2 identifies
from the token side).

**Recommendation:** add a small library of `@tool`-wrapped, parametrized operations
in `crew_tools.py` (or a new `capabilities/ml_tools.py`) for the CRISP-DM-standard
cases actually seen across titanic / house_prices / disaster_tweets:
- `profile_dataframe(path)` — dtypes, missingness, cardinality, target balance (replaces ad hoc `print()`-based evidence gathering)
- `build_preprocess_pipeline(numeric_cols, categorical_cols, text_col=None)` — returns a fitted `ColumnTransformer` (handles the TF-IDF `.ravel()` case natively, always)
- `train_baseline_models(X, y, problem_type, cv_folds)` — runs 2–3 standard baselines (logistic/linear, tree ensemble) with stratified CV and returns scores — the same operation `Optimisation_Research.md` §P2-11 already recommends running "baseline-first"
- `write_kaggle_submission(...)` — codifies `skills/kaggle-submission-contract/SKILL.md` as code instead of relying on the Developer to reimplement it correctly from prose every run

Reserve LLM-authored code for genuinely novel feature engineering the agent
proposes; route everything else through tools. This also removes the JSON-repair
tax on the *interpretation* call, because a tool's return value is already
structured (`ModelRun`-shaped dict), not a script's stdout the Developer has to
parse.

### 2.2 No bounded AutoML / hyperparameter search

Phase 4 (Modeling) currently depends on whatever model + hyperparameters the LLM
decides to write in its authored script. There's no `FLAML`/`Optuna`-style bounded
search tool, so model quality is capped by what the LLM happens to propose in one
shot, and reruns don't converge toward better configurations. A
`tune_model(estimator_family, X, y, budget_seconds)` tool with a small time/trial
budget would raise the ceiling on `cv_score` without adding LLM calls — the search
runs deterministically inside the tool, and the agent just reads back the winning
config. This directly serves the project's stated success criterion #2 (beat the
trivial Kaggle baseline) and gives the Data Scientist agent a stronger lever than
prompting for "try a better model."

### 2.3 RAG is a one-agent capability

`rag.py` retrieves case knowledge for the Domain agent only
(`retrieve_for_state` / `retrieve_passages_for_state` are called from Domain task
prompts). Two capabilities are sitting in this same infrastructure and unused:

- **Skills-as-retrieval instead of skills-as-always-injected-prose.** The 7
  `SKILL.md` files are compact today, but as they grow (see §3.3) always-injecting
  all of them into every relevant agent's backstory will re-create the "large
  fixed system prompt" cost `Optimisation_Research.md` §4 already flags. Indexing
  `skills/*.md` into the same RAG store and retrieving only the 1–2 relevant skills
  per substep (e.g. `leakage-cv-discipline` + `tabular-prep` for a DE prep
  substep) keeps skill content growable without a fixed per-call tax.
- **Past-run experience as few-shot context.** `experience_ledger.py` already
  captures per-run outcomes but is write-only. Feeding the *successful* authored
  code snippets and their outcomes (not just params) back through the same
  RAG/retrieval path for the Data Engineer/Scientist would let the system learn
  "what worked last time for this case" instead of re-deriving it from a cold
  prompt every run — this is exactly the "Loop D … experience feeds next run
  knowledge" contour already declared in `skills/crisp-dm-loops/SKILL.md` but not
  yet backed by a retrieval mechanism for the code-authoring agents.

### 2.4 No model interpretability / evaluation depth

`capabilities/data_scientist.py` and the Evaluation phase (5.x) rely on CV score
and validator checks (`validators.py::validate_phase_4_models`) but there's no
feature-importance / permutation-importance / SHAP-style tool. CRISP-DM's
Evaluation phase explicitly calls for reviewing *why* a model works, not just
whether its score clears a threshold — and the Storyteller's final report
(`capabilities/storyteller.py`, 91 lines) has nothing structured to draw on for
"what drove the prediction," so it can only narrate the score. A
`explain_model(model, X, feature_names)` tool returning top-N importances would
give both the Data Scientist (sanity-check: is the model leaking via ID-like
columns even though `validate_phase_3_artifacts` didn't catch it structurally?)
and the Storyteller a grounded, non-hallucinated basis for narrative claims.

### 2.5 No experiment/model registry across runs

`state.md.models` holds `ModelRun`s for the current run only; nothing persists a
lightweight per-case model history the way `llm_param_ledger.jsonl` persists LLM
params. A `model_ledger.jsonl` (technique, features, cv_score, run_id) alongside
the existing experience ledger would let a future Data Scientist substep answer
"has a better model already been found for this case" before spending tokens
re-deriving it, and would give the human operator a real leaderboard-style view
across repeated runs — useful for the paper deliverable (success criterion #3:
report scores/cost across cases).

---

## 3. Orchestration / architecture quality gaps

### 3.1 Structured Outputs infra exists but isn't the default path

`model_capabilities.py` already probes whether a configured model supports
`STRUCTURED_OUTPUTS` vs plain `JSON_MODE`. Given that the Developer's JSON-repair
tax is ~18% of total tokens on a completed Titanic run (`Optimisation_Research.md`
§6), the highest-leverage use of this existing probe is: whenever a model reports
`STRUCTURED_OUTPUTS`, always call it with the substep's Pydantic/JSON schema bound
via the API's native structured-output parameter, and reserve
`debug_json_parse`/Developer repair for models that only support (or don't
support) plain JSON mode. Confirm this is actually wired end-to-end in
`crew.py`'s `run_json_task` — from the module inventory it looks like the
capability is detected but the retry/repair path doesn't appear to branch on it.

### 3.2 The verify-before-trust pattern stops at Phase 4

`validators.py` re-derives ground truth from disk for Phase 3 (parquet columns)
and Phase 4 (model/CV sanity, including a nice touch — a sanity ceiling that flags
`cv_score >= 0.999` as suspicious). Nothing analogous exists for:
- **Phase 2 (Data Understanding):** no check that claimed EDA findings
  (`domain_data_quality_flags`, missingness claims) match what's actually in the
  raw file — the same class of "LLM claims X, disk says Y" risk the existing
  validators were built to close.
- **Phase 5 (Evaluation):** `assessment_of_dm_results` is taken from the LLM
  with only `success_criterion.py`'s normalization, not a disk-backed check. This
  is the exact mechanism behind the false Loop C bug `Optimisation_Research.md`
  documents (PM reads `assessment` before Data Scientist substep 5.1 runs, sees
  `None → business_goal_met=False`, fires an unnecessary loop). A
  `validate_phase_5_assessment(state)` that computes `business_goal_met`
  deterministically from `chosen_model.cv_score` vs. the configured threshold
  whenever `assessment_of_dm_results` is `None` — using the same
  `criterion_direction`/`score_meets_threshold` helpers already in
  `success_criterion.py` — would close this without waiting on a prompt fix.
- **Phase 6 (Deployment):** no check that the submission file validator
  (`validate_submission_file`, already a tool!) is actually invoked before the
  run is marked complete, vs. just trusting the Developer said it wrote a file.

Extending the existing validator pattern to all six phases — rather than adding
more LLM self-assessment — is the most direct way to raise trustworthiness of
`ml_success`/`workflow_complete` without new prompts.

### 3.3 Experience ledger is write-only; no closed loop

`experience_ledger.py`'s own docstring says "no auto-tuning yet." Every run
appends model/effort/outcome/token data to `llm_param_ledger.jsonl`, but nothing
reads it back to adjust `resolve_all_agent_llm_params()` defaults for the next
run. Given `docs/task20-unified-refactor.md` already identifies that
`reasoning_effort: high` for slow reasoning models burns the wall clock, the
ledger is exactly the data needed to justify (and eventually automate) per-model
default tuning — e.g., "for `gpt-5.5-pro`, `medium` effort produced equal
`ml_success` at 40% of the tokens across N runs, so make that the default for
that model." Closing this loop turns a manual, one-off investigation (like the
task20 postmortem) into a standing feedback mechanism.

### 3.4 Skills lack worked examples and are all-or-nothing per agent

The skills are commendably terse, but several encode judgment calls that are
easy for a small/local model to get wrong without an example:
- `crisp-dm-loops/SKILL.md`'s Loop A trigger ("actionable quality issues... not
  when blockers are only structural absence already documented") is a nuanced
  distinction with zero examples — this is precisely the kind of ambiguity that
  produced the false Loop C incident from a different angle (PM making a loop
  decision from incomplete state).
- `developer-debug-rubric/SKILL.md` step 1 ("classify error: syntax, schema,
  shape, type, timeout, leakage") has no mapping from stderr patterns to these
  classes, so classification consistency depends entirely on model quality.

Recommendation: add a short **decision table** (2–4 rows, not prose) per
ambiguous rule — e.g. for Loop A: `column entirely absent from raw file` →
no-loop (structural), `column present but >60% missing after imputation attempt`
→ loop. Keep it inside the same skill file (still well under backstory-bloat
territory) rather than growing agent backstories, and combine with §2.3's RAG-based
retrieval so the added detail doesn't cost tokens on substeps that don't need it.

### 3.5 No automated regression check for orchestration quality

The false-Loop-C bug and the task20 wall-clock kill were both found by manually
archaeologizing `state.json`/`communications.jsonl` after the fact. There's no
equivalent of `tests/test_path_coverage.py` (which validates the mocked *happy*
path) that asserts orchestration invariants on a fixed fake-LLM script, e.g.:
"if `assessment_of_dm_results` is `None` at substep 5.1, the router must not
fire Loop C" or "`PM_DECISION_SUBSTEPS` must never be scheduled before the
substep that produces the state it decides on." Given the existing
`flow_harness.py`/`fake_llm.py` scaffolding already used for flow tests, these
would be cheap, fast, deterministic tests to add and would catch this whole bug
class in CI rather than in a 2-hour hosted run.

---

## 4. Prioritized recommendations

| # | Recommendation | Type | Effort | Payoff |
|---|---|---|---|---|
| 1 | Convert deterministic prep/train/submit operations into `@tool`-wrapped functions (profile, preprocess pipeline, baseline train, write submission) | Tool | Medium | High — quality + token/retry reduction |
| 2 | Add `validate_phase_5_assessment` computing `business_goal_met` deterministically when LLM assessment is missing | Validator | Small | High — fixes false Loop C at the root |
| 3 | Wire `model_capabilities.py`'s Structured Outputs detection into the actual JSON-call path in `crew.py` | Orchestration | Small–Medium | High — cuts Developer JSON-repair tax |
| 4 | Add a bounded AutoML/HPO tool for Phase 4 | Tool | Medium | Medium–High — raises quality ceiling |
| 5 | Extend RAG retrieval to skills content and to Data Engineer/Scientist (not just Domain) | Capability | Medium | Medium — scalable skill growth, less prompt bloat |
| 6 | Add `validate_phase_2_findings` (EDA claims vs. raw file) and confirm submission validator tool is actually invoked before phase 6 completes | Validator | Small | Medium — closes remaining trust gaps |
| 7 | Add a model-explainability tool for Evaluation/Storyteller | Tool | Small–Medium | Medium — grounds the final report |
| 8 | Add a `model_ledger.jsonl` alongside the existing LLM-param ledger | Capability | Small | Medium — supports the paper deliverable |
| 9 | Feed `llm_param_ledger.jsonl` back into per-model default tuning | Orchestration | Medium | Medium — closes the write-only loop |
| 10 | Add worked-example decision tables to `crisp-dm-loops` and `developer-debug-rubric` skills | Skill content | Small | Medium — reduces ambiguous-routing errors |
| 11 | Add fake-LLM orchestration-invariant regression tests (ordering, loop triggers) | Testing | Small–Medium | High — catches this bug class in CI |

Items 1–3 are the highest-leverage starting point: they attack the same root
cause (freeform code + unverified LLM self-report) that both the token-cost
analysis and the correctness bug in `Optimisation_Research.md` independently
converge on.
