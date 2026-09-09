# MAADS Tools, Skills, and Capabilities — Deep Codebase Synthesis

## Executive summary

MAADS is a substantial six-agent CRISP-DM implementation rather than a prompt-only prototype. It has an explicit CrewAI Flow state machine, typed shared state, bounded CRISP-DM loops, execution-backed data-science steps, Pydantic output contracts, per-run artifacts, token/deadline guards, trace collection, dashboards, and a hosted account product.

The strongest architectural choice is the separation between probabilistic agent reasoning and deterministic execution/validation. That principle should be extended. Too many important controls currently exist only in prompts or nominal skills, while a few critical decisions are mechanically wrong or not verified end to end.

The highest-priority findings are:

1. **Runtime skills are configured but are not loaded into constructed CrewAI agents.** Runtime inspection showed populated `skills_for(...)` paths but `Agent.skills == None` for PM, Data Engineer, Data Scientist, and Developer.
2. **Model selection always maximizes `cv_score`.** Minimization metrics such as RMSE therefore select the worst candidate.
3. **The evaluated pipeline is not persisted and reused.** Phase 4.4 rebuilds a model from a technique name, and Phase 6.1 rebuilds it again. The deployed predictions are not guaranteed to come from the evaluated pipeline.
4. **LLM-authored Python is explicitly not sandboxed.** In hosted mode it can access the process environment, filesystem, network, user data, and secrets available to the service account.
5. **Hosted authentication is username-only.** Anyone who knows a username can obtain that account's session.
6. **Leakage, split design, data contracts, and submission integrity are mostly advisory.** They need deterministic, typed gates.
7. **Agent output contracts are role-wide rather than substep-specific.** Prompt/contract mismatches create avoidable repair calls and permit partial PM/Domain outputs to bypass full validation.
8. **The deployment phase is primarily Kaggle submission generation.** Monitoring, maintenance, rollback, human approval, drift, fairness, and explainability are not operational capabilities.
9. **Observability is rich locally but lacks immutable provenance, semantic agent evaluation, durable telemetry export, alerting, and application-data recovery.**
10. **Engineering feedback is not an enforced gate.** The only GitHub workflow deploys `master`; the repository lacks authoritative CI, linting, typing, security scanning, frontend tests, and dependency-policy automation.

The best improvement strategy is not to add more autonomous agents first. It is to make existing agents' inputs, tools, outputs, and decisions reliable. Add deterministic tools for dataset contracts, split/leakage auditing, persisted experiment execution, submission conformance, policy-based recovery, and model governance. Then add focused reviewer gates and semantic evaluation.

---

## 1. Scope and method

### Repository inspected

- Repository: `/root/work/agentic-crisp-dm`
- Branch: `master`
- Revision: `02961f41c5a8a90a70b156f92b1b64363d73949c`
- Remote: `git@github.com:migush/agentic-crisp-dm.git`
- Existing unrelated untracked paths left untouched: `backups/`, `secrets/`

### Scale

A tracked-file inventory found:

- 331 tracked files
- 207 Python files / 31,057 lines
- 39 TSX files / 5,429 lines
- 11 TypeScript files / 1,407 lines
- 30 Markdown files / 2,034 lines
- 54,462 total text lines, including generated lock data

### Evidence sources

The analysis inspected:

- CRISP-DM state, phase graph, routers, loop guards, and phase dispatch
- Agent adapters, CrewAI construction, model selection, prompts, and output contracts
- Runtime skills and knowledge/RAG setup
- Data Engineer, Data Scientist, Developer, and reporting capabilities
- Code-generation and Python execution paths
- Validators, success criteria, artifacts, reports, trace collection, and dashboards
- Hosted authentication, run launch, storage, task handling, and deployment workflow
- Relevant Python and frontend tests

Statements below are classified as:

- **Present:** confirmed in executable code.
- **Partial:** present, but incomplete or not mechanically enforced.
- **Missing:** no operational implementation found in the inspected code.
- **Recommended:** proposed addition; not a claim about current behavior.

### Verification snapshot

A focused test run covering capabilities, success criteria, evaluation, runtime knowledge, CrewAI model behavior, output contracts, execution tools, hosted auth/tasks, and observability returned:

- **109 passed**
- **4 failed**
- Failures were in embedder configuration and CrewAI structured-output/LLM behavior:
  - `tests/test_knowledge_setup.py::test_resolve_embedder_defaults_to_openai_when_key_set`
  - `tests/test_crew_llm.py::test_openai_json_agents_keep_structured_output`
  - `tests/test_crew_llm.py::test_openai_force_does_not_override_code_agents`
  - `tests/test_crew_llm.py::test_make_agent_binds_resolved_llm`

These failures are important evidence of policy/configuration drift, not just cosmetic test debt.

---

## 2. Current capability map

| Area | Current status | Evidence-grounded assessment |
|---|---|---|
| CRISP-DM phase coverage | Present | Six phases and 24 substeps are represented in typed state and flow code. |
| Iterative loops | Partial/strong | Loops A–C are bounded and recorded; Loop D is described but lacks a valid post-6.4 route. |
| Shared state | Present | `CrispDMState` is the central typed state; role-scoped views reduce context size. |
| Agent specialization | Present | PM, Domain, Data Engineer, Data Scientist, Developer, and Storyteller have distinct adapters/personas. |
| Multi-agent collaboration | Limited by design | Crews are mainly single-agent façades under deterministic hub-and-spoke orchestration. |
| Runtime skills | Broken | Skill paths are configured, but runtime construction produced `Agent.skills == None`. |
| Deterministic tools | Partial | Execution, RAG, config summary, and submission checking exist; agent-callable tool coverage is sparse. |
| Structured outputs | Partial | Pydantic schemas and repair exist, but contracts are role-wide and tests expose policy drift. |
| Execution-backed evidence | Present/strong | Agent-authored Python executes and measured outputs can override narrative claims. |
| Data contracts | Partial | Preflight and phase validators exist, but full schema/semantic checks are absent. |
| Leakage/CV controls | Mostly advisory | Prompt and skill guidance exists; split/preprocessing discipline is not mechanically proved. |
| Experiment tracking | Partial | Run/model/artifact history exists, but fitted model lineage and reproducibility metadata are incomplete. |
| Evaluation | Partial | CV and OOF classification evidence exist; selection, uncertainty, regression, robustness, and calibration need strengthening. |
| Exact train/evaluate/deploy parity | Missing | Pipelines are reconstructed between phases rather than persisted and loaded. |
| Human model approval | Missing | PM is an LLM; there is no accountable pause/resume approval gate. |
| Explainability | Missing operationally | No SHAP/permutation/local explanation pipeline was found. |
| Fairness | Missing | No protected-group configuration, disparity metrics, or governance gate was found. |
| Drift monitoring | Missing | No reference distributions, drift tests, alerts, or retraining policy were found. |
| Traceability | Present/strong locally | Hierarchical events, communications, token data, reports, and run directories are available. |
| Production observability | Partial | No durable OTLP backend, alerting, SLOs, retention policy, or cross-run quality analytics. |
| Hosted isolation | Critical gap | Generated code runs with host-level process access rather than a hardened sandbox. |
| Hosted authentication | Critical gap | Login is username-only. |
| CI/CD quality gate | Missing | The sole workflow deploys pushes to `master` without tests/security/build gates. |
| Developer quality tooling | Missing/weak | No Ruff, formatter, mypy/pyright, pre-commit, ESLint/Prettier, or unified check command. |
| Recovery/resume | Partial | Artifacts survive, but a durable queue and stateful run resume are absent. |
| Backup/restore | Missing for application data | Deploy backups do not provide tested database, user-data, or artifact recovery. |

---

## 3. What should be retained and expanded

### 3.1 Deterministic orchestration over probabilistic agents

`CrispDMFlow` explicitly routes phases and checkpoints (`src/maads/flow/crisp_dm_flow.py:25-269`). `phase_runner.py` controls prerequisites, bounded loops, token/deadline halts, and advancement. This is safer and easier to audit than letting agents freely delegate.

**Recommendation:** retain this architecture. Add focused deterministic reviewer gates rather than unconstrained agent-to-agent delegation.

### 3.2 Execution authority

Data-science code is authored, executed, contract-checked, and persisted. In authoritative paths, measured execution can override fictional LLM claims (`src/maads/capabilities/data_scientist.py:508-537`). This is an excellent foundation.

**Recommendation:** apply the same pattern to data contracts, split audits, experiment selection, submission checks, governance decisions, and deployment verification.

### 3.3 Role-scoped state views

Agents do not need the entire state on every call. Role/substep views reduce prompt size and leakage of irrelevant context.

**Recommendation:** take this further with capability-scoped views and content-addressed context caching. Provide references/hashes for large unchanged artifacts rather than repeatedly embedding them.

### 3.4 Bounded loops and budgets

Phase visits, inner Loop B iterations, token limits, and deadlines constrain agentic failure modes (`src/maads/flow/phase_runner.py:240-275`).

**Recommendation:** preserve hard bounds while introducing a typed recovery policy that distinguishes transient provider errors, schema repair, code defects, data defects, methodological defects, and budget exhaustion.

### 3.5 Run artifacts and local observability

The purpose-tagged layout separates collected, derived, deliverable, and report artifacts (`src/maads/artifact_paths.py:81-145`). Communications, sandbox scripts, traces, postmortems, case reports, workbooks, handoff archives, and improvement bundles provide a strong audit trail.

**Recommendation:** add immutable provenance, digests, external telemetry export, retention, redaction, and restore procedures instead of replacing this subsystem.

---

## 4. Priority findings

## P0.1 — Make runtime skills real

### Evidence

`skills_for()` returns individual skill directories (`src/maads/knowledge_setup.py:67-78`), and `build_agent()` passes them to CrewAI (`src/maads/crew_base.py:202-222`). Runtime inspection produced configured paths for PM, Data Engineer, Data Scientist, and Developer, but every constructed agent reported `skills=None`.

The current skills therefore exist in the repository but are not an effective runtime capability. The leakage skill is also only seven guidance lines (`skills/leakage-cv-discipline/SKILL.md:1-7`).

### Addition

Implement a validated `SkillRegistry`:

1. Discover skills from the common `skills/` root.
2. Parse required frontmatter and validate each `SKILL.md` at startup.
3. Resolve named skills into CrewAI-supported skill objects or the exact directory structure expected by the installed CrewAI version.
4. Attach skills by **substep and workload**, not just broad role.
5. Record disclosed skill IDs, versions, and hashes in each communication record.
6. Fail startup or disable the affected capability explicitly when configured skills cannot load.
7. Add tests asserting discovery, attachment, disclosure, and use.

### Skills to expand or add

- `leakage-split-auditing`
- `dataset-contracts-and-lineage`
- `statistical-evaluation`
- `experiment-reproducibility`
- `model-governance-and-approval`
- `explainability-and-model-cards`
- `fairness-and-slice-evaluation`
- `drift-monitoring-and-retraining`
- `agent-recovery-policy`
- `prompt-contract-consistency`

### Impact

Very high. This converts advertised guidance into actual runtime context and reduces prompt duplication by disclosing only relevant procedures.

---

## P0.2 — Correct direction-aware model selection

### Evidence

Both `_model_technique_from_state()` and 4.4 selection use:

```python
max(state.md.models, key=lambda m: m.cv_score or 0.0)
```

See `src/maads/capabilities/data_scientist.py:308-312` and `:584-613`.

The project already has `criterion_direction()` and direction-aware threshold logic (`src/maads/success_criterion.py:11-30`), but selection does not use it. For RMSE, MAE, MSE, loss, or error, this selects the worst candidate. `or 0.0` also conflates a real zero with absence and can mishandle negative scorer conventions.

### Addition

Create one shared `select_best_model()` capability that:

- requires one comparable primary metric;
- reads explicit optimize direction;
- distinguishes raw metric values from negated sklearn scorers;
- rejects missing/non-finite scores;
- prevents comparison across different split designs or datasets;
- uses uncertainty/tie-breaking rules;
- records a machine-readable selection rationale.

### Acceptance tests

- Lower RMSE wins.
- Higher AUC wins.
- Zero remains a valid score.
- Negative sklearn scorers are normalized explicitly.
- Incompatible metrics cannot be compared.
- Ties consider uncertainty, complexity, latency, and policy.

---

## P0.3 — Persist and deploy the exact evaluated pipeline

### Evidence

Phase 4.3 stores a lightweight `ModelRun` containing technique, CV score/std, description, and parameter settings (`src/maads/capabilities/data_scientist.py:430-439`). Phase 4.4 explicitly instructs the agent to rebuild the pipeline because pipelines are not persisted (`:472-498`). Phase 6.1 repeats the same warning and rebuilds from `CHOSEN_MODEL` (`src/maads/capabilities/developer.py:17-33`, `:173-224`).

This allows train/evaluate/deploy skew: evaluation and submission may use different preprocessing, feature order, encoders, parameters, random seeds, or model implementations.

### Addition: persisted experiment runner tool

Build a deterministic experiment API that returns:

- stable experiment and trial IDs;
- fitted sklearn-compatible pipeline artifact;
- artifact SHA-256 digest;
- feature schema and order;
- target transformation and label encoder;
- dataset and split fingerprints;
- fold-level metrics and OOF prediction artifact;
- primary metric and optimize direction;
- random seeds;
- package/lock digest and Git revision;
- code/prompt/skill/config versions;
- fit and inference resource metrics.

Phase 4.4 must load and assess this exact artifact. Phase 6.1 must load the approved digest, validate the input schema, and produce predictions without regenerating the modeling pipeline.

### Suggested technology

Start with a repository-native artifact contract using `joblib`, JSON metadata, and SHA-256. Add MLflow only if its experiment UI/registry/remote storage benefits justify the operational footprint. The invariant matters more than the vendor.

---

## P0.4 — Isolate LLM-authored code

### Evidence

`PythonExec` explicitly states it is not a security sandbox (`src/maads/tools.py:49-57`). It invokes the current interpreter with the run directory as CWD, captures output, and inherits `os.environ.copy()` (`:100-108`, `:178-183`).

A timeout does not prevent generated code from reading accessible files, opening network connections, reading environment secrets, or spawning descendants. In a hosted multi-user product, this is a critical trust-boundary failure.

### Addition: hardened worker sandbox

Run every generated program in an isolated per-task worker with:

- rootless container or microVM boundary;
- non-host UID and no supplementary groups;
- read-only root filesystem;
- only the task's inputs mounted read-only;
- task output directory mounted writable;
- no host repository, database, socket, or unrelated user-data mounts;
- an allowlisted environment, never a copied host environment;
- no-new-privileges, dropped capabilities, seccomp, and AppArmor;
- CPU, RAM, PID, wall-time, disk, output-size, and file-count limits;
- process-group/cgroup termination;
- default-deny network, with narrowly scoped egress only when needed;
- immutable image digest and dependency manifest;
- adversarial tests for filesystem, environment, network, fork-bomb, symlink, and oversized-output attacks.

Provider API calls should be made by a broker outside the generated-code sandbox. Generated data-processing code should not receive provider, JWT, database, or deployment secrets.

---

## P0.5 — Replace username-only authentication

### Evidence

The authentication module documents that anyone knowing a username can open the session (`webapp/backend/auth.py:1-5`). Login selects a user by username and issues a token without a password or identity proof (`webapp/backend/routes_auth.py:71-79`). JWTs remain valid after logout until expiration (`:82-86`).

Secure/HttpOnly/SameSite cookies and fail-closed production secret configuration are good controls, but they do not compensate for absent authentication.

### Addition

Prefer OIDC or passkeys. If password authentication is used:

- Argon2id password hashing;
- verified recovery flow;
- generic login errors;
- registration/login/task/model rate limits;
- short-lived access token plus rotating server-side refresh session;
- revocation and session management;
- JWT `iss`, `aud`, and `jti` checks;
- session in HttpOnly cookies rather than local storage;
- CSRF protection where applicable;
- CSP, HSTS, frame restrictions, and standard security headers;
- audit events for login, key operations, task launch, artifact access, and approval.

Do not expose the hosted product to untrusted users until authentication and generated-code isolation are fixed.

---

## P0.6 — Turn leakage and split design into executable gates

### Evidence

The current leakage skill says to fit preprocessing on training data only, avoid target/post-outcome features, use stratification, report CV mean/std, and flag unavailable features (`skills/leakage-cv-discipline/SKILL.md:1-7`). Phase 4.2 stores a test-design dictionary, but 4.3 accepts only basic model score output and does not mechanically execute that design (`src/maads/capabilities/data_scientist.py:384-429`, `:561-566`).

Learned transformations may be performed in Phase 3 before CV, and no validator proves that imputation, encoding, scaling, feature selection, resampling, or target encoding were fit inside each fold.

### Addition: typed `TestDesign` and split auditor

The typed contract should include:

- splitter type;
- fold/repeat count;
- shuffle and seed;
- stratification target;
- group/time/spatial column;
- temporal gap/embargo;
- nested tuning design;
- sealed holdout policy;
- primary scorer and direction;
- required slice evaluations.

The execution tool should emit fold membership hashes and verify:

- duplicate entities do not cross protected folds;
- time ordering is valid;
- target-derived and post-outcome features are absent;
- train/test overlap is measured;
- learned preprocessing is inside the fitted pipeline;
- prediction-time availability is documented;
- OOF predictions cover each eligible training row exactly once per repeat.

---

## P0.7 — Align prompts, contracts, and call modes

### Evidence

Output schemas are selected by agent role (`src/maads/crew_base.py:114-137`, `src/maads/output_contracts.py:503-505`) even though required outputs differ by substep. Validation also returns early for PM responses without `action` and Domain responses without `business_objectives` (`src/maads/output_contracts.py:468-480`).

Focused tests show contradictory structured-output expectations. Role-level classification also sends Data Engineer and Data Scientist code and JSON workloads through policies that cannot distinguish authored Python from strict JSON.

### Addition

Resolve contracts by:

```text
(agent, substep, workload_mode, retry_mode)
```

Suggested workload modes:

- `directive_json`
- `specialist_json`
- `authored_code`
- `repair_json`
- `repair_code`
- `narrative_report`

For each substep:

1. Define one exact Pydantic input/output model.
2. Generate the schema hint from that model.
3. Select model and response format by call mode.
4. Validate without PM/Domain bypasses.
5. Ensure `apply_response()` consumes every control field that affects routing or downstream safety.
6. Add a prompt-contract consistency test that validates minimal and representative outputs from every prompt against its exact model.

This should reduce JSON-repair spend while preventing strict JSON mode from interfering with code generation.

---

## P1.1 — Add full dataset contracts and lineage

### Current partial capability

Preflight checks file existence/readability and target presence. `inspect_dataset()` samples only the first 5,000 rows (`src/maads/tools.py:218-257`). Phase validators compare selected state claims to artifacts.

### Missing controls

- full-row schema and dtype compatibility;
- required/nullability/uniqueness rules;
- finite numeric values and ranges;
- categorical domains and cardinality limits;
- duplicated records and entity overlap;
- target viability/class support;
- ID uniqueness and stability;
- row-count preservation through transformations;
- train/test distribution differences;
- exact sample-submission ID value and order checks;
- content fingerprints and transformation lineage.

### Addition

Implement a typed `DatasetContract` and deterministic validator. Great Expectations, Pandera, or Pydantic-based native code can implement the checks; Pandera is a lightweight fit for DataFrame contracts. Store validation results as versioned JSON artifacts and make failed critical checks block advancement.

Each prepared artifact should record:

- input digest(s);
- transformation code digest;
- output digest;
- row/column counts;
- schema;
- target/ID invariants;
- deterministic vs learned transformation classification;
- parent artifact IDs.

---

## P1.2 — Strengthen evaluation beyond one aggregate score

### Current partial capability

Classification fallback evaluation uses stratified five-fold CV, OOF predictions, balanced accuracy, per-class metrics, and a confusion matrix. `EvaluationBundle` validates general shape.

### Addition

Create problem-type-specific evaluation contracts:

**Classification**

- fold metrics and confidence intervals;
- class support;
- calibration/Brier/log loss where probabilities exist;
- threshold selection on OOF data only;
- ROC/PR metrics appropriate to imbalance;
- subgroup/slice metrics;
- robustness checks;
- baseline deltas and statistical comparison.

**Regression**

- metric matching the configured objective;
- MAE/RMSE/RMSLE where valid;
- residual distribution and heteroscedasticity diagnostics;
- prediction interval or uncertainty strategy;
- target and prediction range checks;
- temporal/group/slice residual analysis.

**All problems**

- exact dataset/split/model digests;
- OOF artifact;
- sealed holdout when available;
- reproducible metric recomputation;
- explicit limitations.

Model approval must consume these gates rather than only copy the chosen model into an approved list.

---

## P1.3 — Add accountable human governance

### Evidence

The PM is an LLM. Phase 5/6 flow has no durable human pause/resume authorization. Deployment proceeds through automated routing, and Phase 6 focuses on submission/report creation.

### Addition

Introduce a typed approval record:

- approver identity and role;
- timestamp;
- model artifact digest;
- evaluation evidence digest;
- intended use and prohibited use;
- decision (`approve`, `reject`, `revise_model`, `revise_objectives`, `waiver`);
- limitation/waiver justification;
- expiry/review date;
- fairness, robustness, privacy, and security gate results.

Add a pause/resume checkpoint after 5.3. Keep the PM recommendation separate from authorization. Require an explicit override for failed critical gates and audit every override.

---

## P1.4 — Restore full CRISP-DM Deployment semantics and Loop D

### Evidence

Phase 6 currently maps largely to submission, report evidence, final report, and project review. `plan_monitoring()` exists but is a one-line static plan and is not a meaningful deployed capability (`src/maads/capabilities/developer.py:234-236`). After Phase 6, the flow routes directly to completion (`src/maads/flow/crisp_dm_flow.py:239-264`).

### Addition

Use canonical deployment outcomes:

1. **Plan deployment:** batch/online interface, inference schema, environment, ownership, SLO, rollout, rollback.
2. **Plan monitoring and maintenance:** data quality, drift, prediction, delayed-label performance, resource, cost, and service health.
3. **Produce final report:** evidence-backed model card and operational report.
4. **Review project:** lessons, unresolved risk, reuse constraints.
5. **Post-review Loop D checkpoint:** feed versioned lessons into a new run only after 6.4 evidence exists.

Kaggle submission remains one deployment artifact, not the entire semantics of deployment.

---

## P1.5 — Add explainability, fairness, robustness, and drift capabilities

### Explainability tool

- global permutation importance as a model-agnostic baseline;
- SHAP where model/data size permits;
- representative local explanations;
- explanation stability across folds;
- feature dependence and leakage review;
- model card sections tied to artifact digests.

### Fairness and slice evaluation tool

- configurable sensitive/protected columns;
- minimum support rules;
- per-group performance;
- demographic-parity/equalized-odds gaps only where appropriate;
- intersectional slices;
- uncertainty for group metrics;
- explicit `not_assessed` status and rationale when attributes are unavailable.

### Robustness tool

- missingness perturbation;
- category novelty;
- numeric shift/noise;
- text corruption/length/language slices;
- adversarial or stress scenarios defined from business context.

### Drift and monitoring tool

- reference schema/distributions;
- missingness and categorical novelty drift;
- PSI, KS, Jensen-Shannon, or domain-appropriate tests;
- prediction drift;
- delayed-label performance drift;
- alert thresholds and minimum sample sizes;
- retraining, rollback, and investigation actions.

These artifacts must feed Phase 5 approval and Phase 6 monitoring, not exist only as optional visualizations.

---

## P1.6 — Add workload-aware model routing and recovery

### Evidence

Model routing is primarily role-based (`src/maads/crew_base.py:55-95`). Every Data Engineer/Data Scientist call can inherit a code model even for JSON planning, while Storyteller does not participate in the same JSON specialization. Ordinary LLM calls lack a typed provider fallback/recovery path, and specialist failures normally halt the phase (`src/maads/flow/phase_runner.py:395-404`).

### Addition

Implement:

```text
resolve_model_for_call(agent, substep, workload, context_size,
                       remaining_budget, retry_index, prior_failure_class)
```

Support:

- independent code, JSON, reasoning, and narrative models;
- bounded fallback chains;
- transient retry with jitter/backoff;
- schema repair without expensive general-purpose retries;
- context-size-aware routing;
- per-call cost/latency/quality policy;
- offline policy evaluation using completed run evidence.

Represent failures as typed state:

- provider/transient;
- rate limit/quota;
- schema/parse;
- code syntax/runtime;
- data contract;
- methodological/semantic;
- budget/deadline;
- security/policy.

Only retry retryable classes. Let deterministic policy or PM checkpoints decide loop/halt for semantic failures.

---

## P1.7 — Improve observability, provenance, and agent evaluation

### Current strengths

- hierarchical trace events and durations;
- CrewAI/OpenTelemetry instrumentation bridge;
- communication capture with parse/schema/repair outcomes;
- token accounting and cost lookup;
- per-run dashboards and run comparison;
- postmortem, case report, execution analysis, workbook, and improvement bundle.

### Gaps

`setup_otel()` mirrors spans into the local collector and optionally uses a console exporter; no durable OTLP exporter is configured (`src/maads/observability/otel.py:103-136`). The initial manifest records run/case/model/timestamps but not immutable code, config, prompt, skill, dataset, or environment digests (`src/maads/artifact_paths.py:97-145`). The experience ledger describes itself as write-only and does not tune or evaluate policy (`src/maads/experience_ledger.py:1`, `:82-103`).

### Additions

**Immutable provenance**

- Git SHA and dirty-tree status;
- config content digest;
- prompt/persona/task digests;
- runtime skill IDs and digests;
- dataset and prepared-artifact digests;
- dependency lock and container image digests;
- model/provider/API mode and model capability probe;
- random seeds and test design;
- model artifact digest.

**Durable telemetry**

- OTLP exporter to a trace backend;
- Prometheus metrics for request/run/task/LLM/worker/database health;
- structured logs with correlation IDs;
- Sentry or equivalent error aggregation;
- redaction and PII/secrets policy;
- retention and deletion policy;
- p50/p95/p99 latency, queue delay, time-to-first-token, retry overhead, tokens/second, and cost by agent/substep/model/error class;
- alerting and SLOs.

**Semantic agent evaluation**

- golden cases and frozen fixtures;
- trajectory checks for required tool use and forbidden shortcuts;
- prompt/contract compatibility tests;
- evidence-grounding score;
- tool-call success and recovery quality;
- decision consistency under repeated seeds/models;
- cost/latency/quality Pareto comparison;
- regression thresholds before prompt/model/skill changes are promoted.

Avoid an LLM judge as the sole evaluator. Combine deterministic checks, outcome metrics, rubric-based review, and sampled human adjudication.

---

## P1.8 — Establish CI/CD and supply-chain gates

### Evidence

The only GitHub workflow deploys pushes to `master` and performs no checkout, test, typecheck, build, audit, or provenance step (`.github/workflows/easy-deploy.yml:1-51`). It obtains the SSH host key dynamically with `ssh-keyscan` and passes a dispatch-provided ref into a remote shell command.

### Addition

Create required PR CI:

1. Frozen dependency sync/lock verification.
2. Ruff format and lint.
3. Mypy or pyright with staged strictness.
4. Pytest unit/integration/security suites with branch coverage.
5. Dashboard and hosted frontend typecheck/build.
6. Vitest/React Testing Library tests.
7. Playwright critical-path E2E tests.
8. `pip-audit` and `npm audit` policy.
9. Bandit/Semgrep/CodeQL.
10. Secret scanning.
11. SBOM generation.
12. Prompt/contract/skill registry tests.
13. Agent evaluation regression suite on deterministic fixtures.

Deployment should consume an immutable SHA/artifact that passed CI, validate a 40-character commit reachable from `master`, use a pinned SSH host key, use environment approval for production, run health/smoke checks, and automatically expose rollback status.

Make `uv.lock` canonical and use frozen installs. Add Dependabot or Renovate and a scheduled vulnerability workflow.

---

## P1.9 — Make hosted execution durable and concurrency-safe

### Risks to address

- mutable case-wide `current` pointers can misattribute overlapping same-case runs;
- read-modify-write indexes can lose entries under concurrency despite atomic replacement;
- direct JSON rewrites can expose partial state during polling or crashes;
- append-only JSONL files lack inter-process coordination;
- FastAPI background tasks are not a durable queue;
- SQLite needs explicit concurrency and migration policy;
- uploads/output capture need streaming and quotas.

### Addition

- allocate immutable run ID before task launch and store it on the task row;
- pass that run ID explicitly to the CLI/worker;
- use atomic temp-write/fsync/replace for snapshots;
- lock append/index operations or derive indexes from immutable manifests;
- use a durable worker queue with leases, heartbeats, retries, cancellation, and startup reconciliation;
- enable SQLite WAL and busy timeout, or move to PostgreSQL when concurrency warrants it;
- add schema-versioned migrations;
- stream uploads and enforce request/user/storage quotas;
- cap stdout/stderr and artifact size;
- expose cancel/retry/resume operations with idempotency keys.

---

## P1.10 — Add application backup and restore

Deploy-time Git backups do not protect the hosted database, user cases, encrypted key records, artifacts, or run metadata.

### Addition

Define and test:

- database-consistent backups;
- user data and artifact snapshots;
- encryption at rest and in backup;
- retention schedule;
- checksum verification;
- off-host copy;
- restore drill to an isolated location;
- RPO/RTO targets;
- per-user export/deletion procedure;
- disaster-recovery runbook.

A backup capability is incomplete until a restore is exercised and verified.

---

## 5. Recommended agent and tool topology

Do not begin by increasing unrestricted agent autonomy. Use the current deterministic flow and introduce capability services plus narrow reviewer roles.

### Core role topology

| Role | Keep/change | Recommended authority |
|---|---|---|
| Project Manager | Keep | Recommend route; cannot authorize production deployment. |
| Domain Expert | Keep | Own business objective, constraints, decision cost, and acceptance context. |
| Data Engineer | Keep | Propose transformations; deterministic contract/lineage tool validates them. |
| Data Scientist | Keep | Propose experiments; experiment runner executes and records them. |
| Developer | Narrow | Repair code and package approved artifacts; never self-certify security or deployment. |
| Storyteller | Keep | Render evidence; cannot create or alter scientific results. |
| Human approver | Add | Accountable approval/waiver at Phase 5/6 boundary. |

### Focused reviewer gates

1. **Data Contract Reviewer** after 2.4 and 3.5
   - Consumes deterministic validation and lineage artifacts.
   - Can block modeling on critical defects.

2. **Experiment Reviewer** after 4.4
   - Checks split design, leakage audit, metric direction, reproducibility, robustness, and exact artifact identity.

3. **Deployment Verifier** after 6.1
   - Confirms approved artifact digest, inference schema, submission/API conformance, smoke tests, monitoring, and rollback readiness.

These can initially be deterministic policies with human escalation. An LLM reviewer may summarize evidence but must not replace hard gates.

### Agent-callable tools to add

| Tool | Purpose | Deterministic output |
|---|---|---|
| `validate_dataset_contract` | Full schema/semantic validation | Contract report + severity + dataset digest |
| `build_data_lineage` | Trace transformations | Parent/output digests + transformation metadata |
| `audit_split_and_leakage` | Verify fold and preprocessing discipline | Split artifact + leakage findings |
| `run_experiment` | Fit/CV/evaluate/persist one pipeline | Model bundle + fold metrics + OOF + digest |
| `compare_experiments` | Direction-aware comparable selection | Ranked candidates + rationale |
| `evaluate_model` | Recompute typed evaluation evidence | Problem-specific evaluation bundle |
| `explain_model` | Global/local explanations | Explanation artifacts tied to model digest |
| `evaluate_fairness_and_slices` | Group/slice assessment | Metrics, uncertainty, gate results |
| `stress_test_model` | Robustness tests | Scenario results + failure thresholds |
| `validate_submission` | Exact Kaggle/output conformance | ID/order/dtype/domain/null checks |
| `verify_deployment` | End-to-end artifact/inference validation | Smoke result + approved digest match |
| `classify_failure` | Typed bounded recovery | Retry/repair/loop/halt recommendation |
| `record_approval` | Accountable governance | Signed/audited approval record |
| `checkpoint_run` / `resume_run` | Fault recovery | State/context snapshot + integrity hashes |

Every tool must have typed inputs/outputs, explicit side effects, timeout/budget policy, audit events, and tests.

---

## 6. Skills catalogue to implement

Skills should be short, procedural, versioned, testable, and disclosed only when relevant.

### 6.1 `dataset-contracts-and-lineage`

Trigger: data ingestion, understanding, integration, cleaning, formatting.

Rules:

- distinguish observed statistics from enforced constraints;
- validate full datasets where feasible;
- preserve ID/target invariants;
- fingerprint every material dataset;
- classify transformations as deterministic or learned;
- require lineage from raw to prepared artifacts.

### 6.2 `leakage-split-auditing`

Trigger: feature engineering, test design, modeling, tuning, assessment.

Rules:

- select split strategy from entity/time/group/business structure;
- fit all learned transforms inside folds;
- prohibit post-outcome/prediction-time-unavailable features;
- materialize fold membership and OOF coverage evidence;
- use nested or sealed evaluation when tuning affects selection.

### 6.3 `statistical-evaluation`

Trigger: model comparison and Phase 5 assessment.

Rules:

- use configured metric and direction;
- report fold distribution and uncertainty;
- compare to meaningful baselines;
- evaluate calibration/thresholds/residuals as appropriate;
- separate model selection evidence from final holdout evidence.

### 6.4 `experiment-reproducibility`

Trigger: every model-building call.

Rules:

- persist exact pipeline;
- capture seeds, environment, code, config, dataset, split, and prompt/skill digests;
- refuse deployment when artifact identity cannot be proved;
- never rebuild from a technique name for deployment.

### 6.5 `model-governance-and-approval`

Trigger: Phase 5.3 and Phase 6 entry.

Rules:

- separate recommendation from authorization;
- require intended use, limitations, evidence digest, approver, and expiry;
- block or require explicit waiver when critical gates fail.

### 6.6 `explainability-and-model-cards`

Trigger: Phase 4.4–6.3.

Rules:

- choose methods appropriate to model/data;
- tie explanations to exact model digest;
- quantify stability and limitations;
- prohibit causal claims from predictive importance alone.

### 6.7 `fairness-and-slice-evaluation`

Trigger: business-risk review and evaluation.

Rules:

- define groups and fairness relevance from domain context;
- enforce minimum support and uncertainty reporting;
- report `not_assessed` honestly when attributes or mandate are absent;
- prohibit universal fairness claims from incomplete attributes.

### 6.8 `drift-monitoring-and-retraining`

Trigger: deployment planning and monitoring.

Rules:

- define reference windows, tests, thresholds, sample sizes, owners, and actions;
- distinguish schema, covariate, prediction, concept, and performance drift;
- pair alerts with investigation, rollback, or retraining decisions.

### 6.9 `agent-recovery-policy`

Trigger: provider, parse, schema, execution, data, or budget failure.

Rules:

- classify before retrying;
- retry only transient failures;
- cap repair attempts;
- preserve failed evidence;
- route semantic defects to CRISP-DM loops rather than blind repetition.

### 6.10 `prompt-contract-consistency`

Trigger: prompt/schema/model-policy changes.

Rules:

- one exact contract per substep/mode;
- derive schema hints from code;
- validate representative outputs;
- ensure state application consumes governance fields;
- regression-test supported provider response modes.

---

## 7. Quality and optimization opportunities

### 7.1 Reduce duplicate LLM work

Existing optimization research observed expensive duplicate execution-plus-interpretation calls. When deterministic execution already returns a validated payload, apply it directly and use an LLM only for genuinely semantic interpretation.

Use content-addressed caching keyed by:

- substep;
- agent/workload mode;
- state-view digest;
- input artifact digests;
- prompt/skill/model/config versions.

Never reuse a cached result when mutable business assumptions, datasets, split design, code, or model policy differ.

### 7.2 Keep retry prompts bounded

Do not resend complete failed code, full stderr, and full persona on every repair. Send:

- error class;
- minimal failing excerpt;
- bounded stderr tail;
- contract violation;
- dataset/schema reference;
- prior artifact ID.

Escalate models only when the failure class and expected benefit justify cost.

### 7.3 Baseline-first execution

Run deterministic baselines before expensive agent-generated alternatives. Baselines provide:

- data/split validation;
- minimum expected performance;
- runtime/cost reference;
- fallback artifact;
- evidence for whether agentic complexity adds value.

### 7.4 Pareto-based optimization

Track quality, success, latency, token use, cost, retries, and failure rates by workload. Select policies on a Pareto frontier rather than optimizing token count alone.

### 7.5 Frontend quality

Add Vitest/React Testing Library for components/hooks and Playwright for:

- registration/login/session expiry;
- case creation/upload validation;
- task launch/status/cancel/retry;
- run selection/comparison;
- artifact/report access controls;
- error and empty states;
- approval workflow.

The current build/typecheck is useful but does not verify behavior.

---

## 8. Phased implementation roadmap

## Phase A — Immediate correctness and exposure reduction

1. Fix direction-aware model selection and add regression tests.
2. Repair runtime skill loading and add discovery/attachment tests.
3. Resolve the four focused test failures and make model/structured-output policy call-mode-specific.
4. Disable untrusted hosted access or restrict it to trusted users until authentication and sandboxing are complete.
5. Implement real identity proof for hosted accounts.
6. Add strict upload, concurrency, output, and storage quotas.

**Exit criteria**

- Lower-is-better metrics select correctly.
- Every configured runtime skill is demonstrably loaded.
- Focused architecture/capability suite is green.
- No username-only account access.
- Generated code cannot access host secrets or other users' data.

## Phase B — Scientific validity

1. Implement typed `DatasetContract` and lineage artifacts.
2. Implement typed `TestDesign` and leakage/split auditor.
3. Implement persisted experiment runner and exact model digest.
4. Require 4.4 and 6.1 to load the exact pipeline.
5. Implement exact submission ID/order/dtype/domain/null checks.
6. Add problem-specific evaluation bundles and direction-aware comparison.

**Exit criteria**

- Dataset, split, model, and evaluation lineage is reproducible from digests.
- Train/evaluate/deploy artifact identity is proven.
- CV/preprocessing discipline is mechanically checked.
- Submission conformance checks semantic identity, not only shape.

## Phase C — Governance and lifecycle completeness

1. Add human approval pause/resume.
2. Restore full deployment/monitoring semantics.
3. Add post-6.4 Loop D.
4. Add explainability/model cards.
5. Add fairness/slice and robustness gates.
6. Add drift monitoring and retraining/rollback actions.

**Exit criteria**

- Deployment requires accountable approval.
- Every deployed artifact has intended use, limitations, monitoring, owner, and rollback.
- Loop D consumes evidence produced by the completed project review.

## Phase D — Engineering and operational maturity

1. Add required CI, quality, security, and agent-evaluation gates.
2. Make dependency lock/builds immutable and auditable.
3. Add durable task queue and resumable checkpoints.
4. Add concurrency-safe artifact/index/state writes.
5. Add OTLP/metrics/logging/error aggregation and SLO alerts.
6. Implement and test backup/restore.
7. Add frontend unit/E2E coverage.

**Exit criteria**

- Production deploys only immutable CI-approved artifacts.
- Interrupted runs recover or reach a reconciled terminal state.
- Telemetry supports cross-run reliability/cost/quality analysis.
- Restore drills meet stated RPO/RTO.

---

## 9. Suggested success metrics

### Scientific quality

- percentage of runs with valid dataset/split/model/config digests;
- percentage with exact evaluate/deploy artifact match;
- leakage audit pass rate;
- baseline-relative improvement with uncertainty;
- calibration/robustness/slice gate pass rates;
- submission exact-conformance rate.

### Agent quality

- schema-valid first-response rate by substep/model;
- deterministic tool success rate;
- unsupported-claim/evidence mismatch rate;
- repair and retry rate by failure class;
- loop precision: loops that resolve a measured deficit;
- human override/waiver rate;
- semantic regression-suite pass rate.

### Efficiency

- tokens, cost, and wall time per successful run;
- cost per completed/validated substep;
- p50/p95 LLM latency and worker duration;
- repeated-context ratio;
- cache hit rate with correctness safeguards;
- model quality/cost/latency Pareto position.

### Reliability and security

- cross-tenant access failures in adversarial tests;
- sandbox escape test pass rate;
- orphaned task and recovery rate;
- artifact corruption/index-loss rate under concurrency tests;
- authentication abuse/rate-limit events;
- backup success and restore-drill success;
- mean time to detect and recover.

---

## 10. Decision summary

### Add first

1. Runtime skill registry/validation.
2. Direction-aware model selector.
3. Persisted experiment runner and exact pipeline reuse.
4. Hardened generated-code sandbox.
5. Real hosted authentication.
6. Dataset contract/lineage and split/leakage tools.
7. Substep/call-mode-specific prompt contracts.
8. Exact submission conformance.

### Apply next

1. Human governance gate.
2. Problem-specific statistical evaluation.
3. Explainability, fairness, robustness, and drift capabilities.
4. Workload-aware model routing and typed recovery.
5. Immutable provenance and semantic agent evaluation.
6. Required CI/security/supply-chain gates.
7. Durable task execution, concurrency-safe storage, and run resume.
8. Production telemetry and tested backup/restore.

### Do not prioritize yet

- adding many more autonomous agents;
- enabling broad peer delegation;
- introducing an LLM judge as the sole quality gate;
- deploying a heavy experiment platform before defining artifact contracts;
- optimizing token cost before correcting scientific and security invariants.

The current architecture is capable of becoming a defensible agentic data-science platform. The fastest route is to operationalize the skills already implied by the design, move scientific controls from prose into deterministic tools, persist exact artifacts across CRISP-DM boundaries, and treat hosted execution as a real multi-tenant security boundary.
