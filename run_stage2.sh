#!/usr/bin/env bash
# DEPRECATED EXPERIMENT: retained for reference only.
# Use run_nohup.sh with the single-stage training config instead.
# Usage:
# ./run_stage2.sh start configs/config_stage2_nan_token.py
# ./run_stage2.sh stop <pid_or_pid_file>
# ./run_stage2.sh restart <pid_or_pid_file> <config>
# ./run_stage2.sh status

set -euo pipefail

BASE_DIR="/home/jovyan/work/sylvia/myWhisperTrainer"
LOG_DIR="${BASE_DIR}/logs"

ensure_env() {
  source /home/jovyan/work/sylvia/miniconda3/etc/profile.d/conda.sh
  conda activate asr_trainer
}

start_train() {
  local cfg="${1:-configs/config_stage2_nan_token.py}"
  shift || true
  local extra_args="$*"
  local label
  label="stage2_$(basename "${cfg}")"
  label="${label%.*}"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local pid_file="${LOG_DIR}/train_${label}_${ts}.pid"

  mkdir -p "${LOG_DIR}"
  cd "${BASE_DIR}"
  ensure_env

  nohup python train_stage2.py --config "${cfg}" ${extra_args} > "${LOG_DIR}/nohup_${label}_${ts}.log" 2>&1 &
  echo $! > "${pid_file}"
  echo "started: pid=$(cat "${pid_file}") config=${cfg} extra_args='${extra_args}' pid_file=${pid_file} log=${LOG_DIR}/nohup_${label}_${ts}.log"
}

stop_train() {
  local pid_or_file="${1:-}"
  if [[ -z "${pid_or_file}" ]]; then
    echo "usage: $0 stop <pid_or_pid_file>"
    exit 1
  fi

  local pid_file=""
  local pid=""
  if [[ -f "${pid_or_file}" ]]; then
    pid_file="${pid_or_file}"
    pid="$(cat "${pid_file}")"
  else
    pid="${pid_or_file}"
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}"
    echo "stopped: pid=${pid}"
  else
    echo "not running: pid=${pid} not found"
  fi

  if [[ -n "${pid_file}" ]]; then
    rm -f "${pid_file}"
  fi
}

status_train() {
  local pid_files
  pid_files=("${LOG_DIR}"/train_stage2_*.pid)
  if [[ ! -e "${pid_files[0]}" ]]; then
    echo "not running"
    exit 0
  fi

  local any_running=0
  for pid_file in "${pid_files[@]}"; do
    local pid
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      any_running=1
      echo "running: pid=${pid} pid_file=${pid_file}"
    fi
  done

  if [[ "${any_running}" -eq 0 ]]; then
    echo "not running"
  fi
}

case "${1:-start}" in
  start) shift; start_train "$@" ;;
  stop) stop_train "${2:-}" ;;
  restart) stop_train "${2:-}"; shift 2; start_train "$@" ;;
  status) status_train ;;
  *) echo "usage: $0 {start [config] | stop <pid_or_pid_file> | restart <pid_or_pid_file> [config] | status}" ;;
esac
