#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/ssr_plus_struct_compare_qwen3_8b_1k}"
COMMANDS="${COMMANDS:-${RUN_DIR}/commands.sh}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/watch}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
POLL_SECONDS="${POLL_SECONDS:-1800}"
TMUX_SESSION="${TMUX_SESSION:-ssr_plus_struct_compare}"
mkdir -p "${LOG_DIR}"

WATCH_LOG="${LOG_DIR}/watch.log"
STATUS_FILE="${LOG_DIR}/status.env"

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "${WATCH_LOG}"
}

write_status() {
  local state="$1"
  local reason="${2:-}"
  {
    printf 'STATE=%q\n' "${state}"
    printf 'REASON=%q\n' "${reason}"
    printf 'UPDATED_AT=%q\n' "$(date '+%F %T %Z')"
    printf 'RUN_DIR=%q\n' "${RUN_DIR}"
    printf 'COMMANDS=%q\n' "${COMMANDS}"
  } > "${STATUS_FILE}"
}

gpu_busy_reason() {
  python3 - "${GPU_IDS}" <<'PY'
import os
import subprocess
import sys

gpu_ids = {x.strip() for x in sys.argv[1].split(",") if x.strip()}

try:
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    gpu_map = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
except Exception as exc:
    print(f"nvidia-smi failed: {exc}")
    sys.exit(1)

uuid_to_index = {}
for line in gpu_map.splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) >= 2:
        uuid_to_index[parts[1]] = parts[0]

busy = []
holder = []
for line in apps.splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4 or not parts[1].isdigit():
        continue
    gpu_index = uuid_to_index.get(parts[0], "")
    if gpu_ids and gpu_index and gpu_index not in gpu_ids:
        continue
    pid = int(parts[1])
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "ignore")
    except Exception:
        cmd = parts[2]
    item = f"gpu{gpu_index}:pid{pid}:{parts[3]}MiB:{cmd[:180]}"
    if "hold_gpu.py" in cmd:
        holder.append(item)
    else:
        busy.append(item)

if busy:
    print("busy: " + " | ".join(busy))
    sys.exit(1)

if holder:
    print("holder_only: " + " | ".join(holder))
else:
    print("idle")
PY
}

if [ ! -x "${COMMANDS}" ]; then
  log "error: commands not executable: ${COMMANDS}"
  write_status "error" "missing commands"
  exit 1
fi

while true; do
  if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    log "already_running: tmux session ${TMUX_SESSION}"
    write_status "running" "tmux session exists"
    exit 0
  fi

  set +e
  reason="$(gpu_busy_reason 2>&1)"
  status=$?
  set -e
  if [ "${status}" -eq 0 ]; then
    log "gpu_available: ${reason}"
    write_status "launching" "${reason}"
    tmux new-session -d -s "${TMUX_SESSION}" "cd '${ROOT}' && bash '${COMMANDS}' > '${LOG_DIR}/pipeline.log' 2>&1; status=\$?; printf '[%s] pipeline exited with status %s\n' \"\$(date '+%F %T %Z')\" \"\$status\" >> '${WATCH_LOG}'; exit \$status"
    log "launched: tmux=${TMUX_SESSION} log=${LOG_DIR}/pipeline.log"
    write_status "running" "tmux=${TMUX_SESSION}"
    exit 0
  fi

  log "waiting: ${reason}"
  write_status "waiting" "${reason}"
  sleep "${POLL_SECONDS}"
done
