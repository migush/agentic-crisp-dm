#!/bin/bash
# Gate destructive git / secret-touching shell. Fail open on parse errors.
set -euo pipefail
input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.command // empty' 2>/dev/null || true)

if [ -z "$cmd" ]; then
  echo '{ "permission": "allow" }'
  exit 0
fi

# Force-push to default branches
if echo "$cmd" | grep -Eq 'git[[:space:]]+push' && echo "$cmd" | grep -Eq -- '(^|[[:space:]])(-f|--force|--force-with-lease)([[:space:]]|$)'; then
  if echo "$cmd" | grep -Eq '(^|[[:space:]])(origin[[:space:]]+)?(main|master)([[:space:]]|$)'; then
    jq -n '{
      permission: "deny",
      user_message: "Blocked force-push to main/master.",
      agent_message: "Do not force-push main or master. Use a feature branch."
    }'
    exit 0
  fi
  jq -n '{
    permission: "ask",
    user_message: "This force-pushes a git remote. Review before continuing.",
    agent_message: "Force-push requires explicit user approval."
  }'
  exit 0
fi

# Staging or committing secrets
if echo "$cmd" | grep -Eq 'git[[:space:]]+(add|commit|rm)' && echo "$cmd" | grep -Eq '(^|[[:space:]])(\.env|.*credentials\.json|.*id_rsa)([[:space:]]|$)'; then
  jq -n '{
    permission: "deny",
    user_message: "Blocked git operation on secret files (.env / credentials).",
    agent_message: "Do not add or commit .env or credential files."
  }'
  exit 0
fi

echo '{ "permission": "allow" }'
exit 0
