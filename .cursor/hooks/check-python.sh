#!/bin/bash
# Syntax-check edited Python files. No project formatter exists (no ruff/black).
# Fail open: never block the edit.
set -euo pipefail
input=$(cat)
path=$(printf '%s' "$input" | jq -r '.file_path // .path // .uri // empty' 2>/dev/null || true)
path=${path#file://}

case "$path" in
  *.py) ;;
  *) echo '{}'; exit 0 ;;
esac

if [ ! -f "$path" ]; then
  echo '{}'
  exit 0
fi

if command -v python3 >/dev/null 2>&1; then
  if ! python3 -m py_compile "$path" 2>/tmp/maads-py-compile.err; then
    err=$(tr '\n' ' ' </tmp/maads-py-compile.err | head -c 500)
    jq -n --arg msg "Python syntax check failed for $path: $err" \
      '{agent_message: $msg}'
    exit 0
  fi
fi

echo '{}'
exit 0
