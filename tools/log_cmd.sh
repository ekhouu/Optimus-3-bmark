#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: TRAIN_CMD_LOG=/path/to/train_commands.md tools/log_cmd.sh '<command>'" >&2
  exit 2
fi

LOG_FILE="${TRAIN_CMD_LOG:-outputs/train_commands.md}"
CMD="$*"

mkdir -p "$(dirname "$LOG_FILE")"

{
  echo "## $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo
  echo '```bash'
  echo "$CMD"
  echo '```'
  echo
} >> "$LOG_FILE"

echo "[log_cmd] logged to $LOG_FILE"

exec bash -lc "$CMD"
