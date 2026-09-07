#!/bin/bash
# Inject maads session context. Fail open.
input=$(cat)
printf '%s\n' '{
  "additional_context": "maads: read AGENTS.md and workflow_state.md first. Six agents including Storyteller. FastAPI dashboard is src/maads/dashboard/ (Vite UI is repo-root dashboard/). artifacts/<case>/current is a text run-id file, not a symlink. No ruff/eslint/prettier in this repo — pytest and tsc. Do not commit unless asked. Do not expose the trace dashboard publicly."
}'
exit 0
