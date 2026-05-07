#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_PYTHON_BIN="$ROOT_DIR/venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
SOURCE_NAME="${SOURCE_NAME:-whoscored_championship}"
RUN_MODE="${RUN_MODE:-incremental}"
LOG_FILE="${LOG_FILE:-data/pipeline/server_pipeline.log}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
CLEANUP_TEMP="${CLEANUP_TEMP:-1}"
STATUS_FILE="${STATUS_FILE:-$ROOT_DIR/data/pipeline/server_pipeline_status.txt}"
ALERT_WEBHOOK_URL="${ALERT_WEBHOOK_URL:-}"

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$STATUS_FILE")"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

write_status() {
  local status="$1"
  local message="$2"
  {
    echo "timestamp=$(timestamp)"
    echo "status=$status"
    echo "source=$SOURCE_NAME"
    echo "run_mode=$RUN_MODE"
    echo "cleanup_temp=$CLEANUP_TEMP"
    echo "message=$message"
  } > "$STATUS_FILE"
}

send_alert() {
  local status="$1"
  local message="$2"

  echo "[$(timestamp)] ALERT [$status] $message" >> "$LOG_FILE"

  if [[ -n "$ALERT_WEBHOOK_URL" ]] && command -v curl >/dev/null 2>&1; then
    curl -sS -X POST "$ALERT_WEBHOOK_URL" \
      -H "Content-Type: application/json" \
      -d "{\"text\":\"Watford pipeline [$status]: $message\"}" \
      >/dev/null || true
  fi
}

on_error() {
  local line_number="${1:-unknown}"
  local message="Pipeline failed near line $line_number. Review log: $LOG_FILE"
  write_status "failed" "$message"
  send_alert "failed" "$message"
}

trap 'on_error $LINENO' ERR

if [[ "$PYTHON_BIN" == "$DEFAULT_PYTHON_BIN" && ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

for required_var in DB_USER DB_PASSWORD DB_HOST DB_PORT DB_NAME; do
  if [[ -z "${!required_var:-}" ]]; then
    echo "Missing required environment variable: $required_var" >&2
    exit 1
  fi
done

case "$RUN_MODE" in
  bootstrap)
    STRATEGY="bootstrap"
    ;;
  incremental)
    STRATEGY="incremental"
    ;;
  auto)
    STRATEGY="auto"
    ;;
  *)
    echo "Invalid RUN_MODE: $RUN_MODE. Use bootstrap, incremental, or auto." >&2
    exit 1
    ;;
esac

if [[ "$CLEANUP_TEMP" == "1" ]]; then
  "$PYTHON_BIN" scripts/run_pipeline.py \
    --source "$SOURCE_NAME" \
    --mode full-db \
    --strategy "$STRATEGY" \
    --incremental-state db \
    --db-mode append_new \
    --cleanup-temp \
    --log-file "$LOG_FILE"
else
  "$PYTHON_BIN" scripts/run_pipeline.py \
    --source "$SOURCE_NAME" \
    --mode full-db \
    --strategy "$STRATEGY" \
    --incremental-state db \
    --db-mode append_new \
    --log-file "$LOG_FILE"
fi

write_status "ok" "Pipeline completed successfully. Review log: $LOG_FILE"
