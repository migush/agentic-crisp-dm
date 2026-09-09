# Task 20 unified refactor

Compiled from `docs/task20_refactor_plan_5b5440bb.plan.md` and
`docs/docs-task20-analyse-the-maads-indexed-wigderson.md`.

## Verdict

Task 20 (`disaster_tweets` / `gpt-5.5-pro`, run `773f177a`) was progressing
correctly and was killed by the hosted 1-hour `subprocess.run` timeout during
substep 4.2. That substep has no bug. Default `reasoning_effort: high` for
data engineer / data scientist / developer on a slow reasoning model burned
the wall clock in phases 1–3.

The parent kill never set `halt_reason`, never wrote reports, and left
`run_artifact_path` null.

## What we keep from each plan

| Topic | Plan A (5b5440bb) | Plan B (wigderson) | Unified |
|---|---|---|---|
| Graceful in-flow deadline | Yes | Yes (reuse halt path) | Yes — `MAADS_RUN_DEADLINE_SEC` + existing `force_halt` |
| Outer launcher timeout | Env + SIGTERM + persist path | Last-resort backstop | Both |
| DE/DS/developer effort | `medium` | `medium` | `medium` (env overrides stay) |
| Token accounting | prompt+completion, not cumulative `total_tokens` | Investigate gap | Prefer prompt+completion |
| `config/tasks.yaml` | Keep (scaffolds) | Delete as dead | **Keep** — `compile_task_payload` loads it |
| Phase `@CrewBase` facades | Collapse | Share YAML template | Collapse — kickoff never used `.crew()` |
| `MaadsCrew` `@task` methods | Keep with canonical YAML | Delete | Keep — CrewBase still loads `config/tasks.yaml` |
| `disaster_tweets` NA hint | `na_means_absent` | Out of scope | Add — cheap, already in the DE/PM contract |
| Hosted data paths | Extra resolution | One-off / already fixed | Keep `resolve_path` vs repo root (already in `load_case_config`) |
| Investigation dumps | Delete | Confirm first | Gitignore; do not commit or delete local copies |
| Dashboard orphans | Delete unused pages | Not in plan | Delete pages with no importers |
| TRACE KeyErrors | `.get` defaults | Not in plan | Ignore known env KeyErrors in the Python tracer |
| LLM env-matrix shrink | P1 | Not in plan | Deferred — `resolve_agent_llm_params` already exists |
| Flow topology / webapp+dashboard merge | Out of scope | Out of scope | Out of scope |

## Workstreams in this PR

1. **Reliability:** in-flow wall-clock halt, launcher SIGTERM + artifact path on
   timeout, token deltas, medium effort defaults, disaster-tweets NA hint.
2. **Simplify:** thin phase crews (no unused `@CrewBase` / per-crew YAML).
3. **Hygiene:** gitignore dumps, drop dead dashboard pages, quiet TRACE KeyErrors,
   align README / CLAUDE.md with AGENTS.md (six agents; `current` is a text pointer).
