---
name: inspect-run-artifacts
description: Locate, read, and regenerate maads run artifacts, traces, communications, and reports. Use when debugging a run, reading status.json, communications.md, handoff zips, or writing reports.
---

# Inspect run artifacts

## Find the run

```
artifacts/<case_id>/current          # text file: active run id
artifacts/<case_id>/runs/<run_id>/   # the run directory
artifacts/<case_id>/archive/         # superseded runs
```

Resolve in code with `maads.artifact_runs.resolve_active_run_dir`. Do not treat `current` as a directory symlink.

## What to read

| File | Meaning |
|---|---|
| `status.json` | Live phase/substep/halt |
| `final_state.json` | `CrispDMState.model_dump_json` at end |
| `derived/live_summary.json` | Dashboard polling |
| `collected/communications.jsonl` | LLM I/O |
| `trace/communications.md` | Human transcript (full prompts) |
| `reports/` | postmortem, case_report, execution_analysis, workbook, handoff |

Privacy: transcripts are local-only.

## Regenerate views

```bash
python -m maads artifacts render --run artifacts/<case>/runs/<run_id>
python -m maads artifacts backfill-timing --run artifacts/<case>/runs/<run_id>
```

Reports are written at end of `maads run` via `write_run_reports` unless disabled (`MAADS_REPORTS=0`) or a `.generated` stamp exists.

## Dashboard

`python -m maads dashboard --artifact-dir artifacts --no-open` then Inspect / Process / Results tabs. Hosted users see per-user roots plus read-only demo artifacts.
