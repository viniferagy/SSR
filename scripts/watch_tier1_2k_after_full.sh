#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/pengguangyue/workspace/proj/SSR"
cd "$ROOT"

WATCH_DIR="runs/tier1_baselines/2k/watch"
STATUS_FILE="$WATCH_DIR/status.env"
WATCH_LOG="$WATCH_DIR/watch.log"
PIPELINE_LOG="$WATCH_DIR/pipeline.log"
SESSION_NAME="${SESSION_NAME:-tier1_2k_after_full}"
PY_GEN="${PY_GEN:-/home/pengguangyue/workspace/.venv/bin/python}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"

mkdir -p "$WATCH_DIR"

log() {
  local now
  now="$(date '+%F %T')"
  echo "[$now] $*" | tee -a "$WATCH_LOG"
}

write_status() {
  local state="$1"
  local detail="${2:-}"
  {
    printf 'STATE=%q\n' "$state"
    printf 'DETAIL=%q\n' "$detail"
    printf 'UPDATED_AT=%q\n' "$(date '+%F %T')"
    printf 'SESSION_NAME=%q\n' "$SESSION_NAME"
    printf 'PIPELINE_LOG=%q\n' "$PIPELINE_LOG"
  } > "$STATUS_FILE"
}

full_ready() {
  local files=(
    "runs/final_reviewer_protocol_2k/full/inputs/methods_2k.metric.jsonl"
    "runs/final_reviewer_protocol_2k/full/inputs/blind_2k.metric.jsonl"
    "runs/final_reviewer_protocol_2k/full/inputs/paraphrases_2k.jsonl"
    "runs/final_reviewer_protocol_2k/full/inputs/controlled_2k.metric.jsonl"
    "runs/final_reviewer_protocol_2k/full/results/report.md"
  )
  local dirs=(
    "runs/final_reviewer_protocol_2k/full/metrics/blind"
    "runs/final_reviewer_protocol_2k/full/metrics/controlled"
  )
  local missing=()
  local path
  for path in "${files[@]}"; do
    [[ -s "$path" ]] || missing+=("$path")
  done
  for path in "${dirs[@]}"; do
    [[ -d "$path" ]] || missing+=("$path")
  done
  if ((${#missing[@]} > 0)); then
    printf '%s\n' "${missing[@]}"
    return 1
  fi
  return 0
}

tier1_done() {
  [[ -s "runs/tier1_baselines/2k/results/report.md" ]] \
    && [[ -s "runs/tier1_baselines/2k/endpoint/endpoint.summary.json" ]]
}

tier1_running() {
  pgrep -af 'runs/tier1_baselines/run_tier1_2k.sh|scripts/baseline_tier1.py' \
    | grep -v grep >/dev/null 2>&1
}

start_tier1() {
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    log "tmux session already exists: $SESSION_NAME"
    write_status "running" "tmux session already exists"
    return 0
  fi
  log "starting Tier1 2k pipeline in tmux session $SESSION_NAME"
  write_status "running" "starting tier1 2k"
  tmux new-session -d -s "$SESSION_NAME" \
    "cd '$ROOT' && env PY_GEN='$PY_GEN' bash runs/tier1_baselines/run_tier1_2k.sh all 2>&1 | tee '$PIPELINE_LOG'"
}

log "watcher started; interval=${INTERVAL_SECONDS}s"
write_status "watching" "waiting for full 2k reviewer protocol"

while true; do
  if tier1_done; then
    log "Tier1 2k outputs already complete"
    write_status "complete" "Tier1 2k report and endpoint summary exist"
    exit 0
  fi

  if tier1_running; then
    log "Tier1 2k appears to be running"
    write_status "running" "Tier1 2k process/session detected"
    sleep "$INTERVAL_SECONDS"
    continue
  fi

  missing="$(full_ready 2>/dev/null || true)"
  if [[ -n "$missing" ]]; then
    local_detail="$(echo "$missing" | paste -sd ';' -)"
    log "waiting for full 2k prerequisites: $local_detail"
    write_status "waiting" "$local_detail"
    sleep "$INTERVAL_SECONDS"
    continue
  fi

  start_tier1
  sleep "$INTERVAL_SECONDS"
done
