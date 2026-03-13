#!/bin/bash
# Whisper LoRA ASR Service deployment script
# Author: Sylvia
# Date: 2026-03-11
#
# Usage:
#   conda activate asr_trainer
#
#   bash whisper_service/deploy.sh start     # Start the service (background)
#   bash whisper_service/deploy.sh stop      # Stop the service (graceful -> SIGKILL)
#   bash whisper_service/deploy.sh restart   # Restart
#   bash whisper_service/deploy.sh status    # Show PID, ports, recent logs
#
# Before starting, edit whisper_service/config/whisper_config.yaml:
#   - Set model.adapter_path to your LoRA adapter directory
#   - Adjust deployment.num_replicas, batch.max_batch_size, etc.
#
# Logs:  whisper_service/logs/service.log
# PID:   whisper_service/whisper_lora_server.pid

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="whisper_lora_server"
PYTHON_SCRIPT="${SCRIPT_DIR}/${SERVICE_NAME}.py"
PID_FILE="${SCRIPT_DIR}/${SERVICE_NAME}.pid"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/service.log"

mkdir -p "${LOG_DIR}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }

is_running() {
    if [ -f "${PID_FILE}" ]; then
        PID=$(cat "${PID_FILE}")
        if ps -p "${PID}" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

start() {
    info "Starting ${SERVICE_NAME}..."

    if is_running; then
        warn "Service is already running (PID: $(cat ${PID_FILE}))"
        return 1
    fi

    nohup python "${PYTHON_SCRIPT}" >> "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "${PID}" > "${PID_FILE}"

    sleep 2

    if is_running; then
        success "Service started successfully (PID: ${PID})"
        info "Log file: ${LOG_FILE}"
        info "View logs: tail -f ${LOG_FILE}"
        return 0
    else
        error "Service failed to start"
        error "Check log file: ${LOG_FILE}"
        return 1
    fi
}

stop() {
    info "Stopping ${SERVICE_NAME}..."

    if ! is_running; then
        warn "Service is not running"
        [ -f "${PID_FILE}" ] && rm -f "${PID_FILE}"
        return 1
    fi

    PID=$(cat "${PID_FILE}")
    info "Sending SIGTERM to process ${PID}..."
    kill "${PID}" 2>/dev/null

    for i in {1..10}; do
        if ! ps -p "${PID}" > /dev/null 2>&1; then
            success "Service stopped successfully"
            rm -f "${PID_FILE}"
            return 0
        fi
        sleep 1
    done

    warn "Process did not stop gracefully, sending SIGKILL..."
    kill -9 "${PID}" 2>/dev/null
    sleep 1

    if ! ps -p "${PID}" > /dev/null 2>&1; then
        success "Service stopped (forced)"
        rm -f "${PID_FILE}"
        return 0
    else
        error "Failed to stop service"
        return 1
    fi
}

restart() {
    info "Restarting ${SERVICE_NAME}..."
    stop
    sleep 2
    start
}

status() {
    echo "================================"
    echo "  ${SERVICE_NAME} Status"
    echo "================================"

    if is_running; then
        PID=$(cat "${PID_FILE}")
        echo -e "Status: ${GREEN}RUNNING${NC}"
        echo "PID: ${PID}"
        echo ""
        echo "Process Info:"
        ps -p "${PID}" -o pid,ppid,%cpu,%mem,etime,cmd 2>/dev/null | tail -n +2
        echo ""
        echo "Listening Ports:"
        netstat -tulpn 2>/dev/null | grep "${PID}" | grep LISTEN || echo "  (no ports found)"
        echo ""
        echo "Recent Logs (last 10 lines):"
        echo "----------------------------"
        tail -10 "${LOG_FILE}" 2>/dev/null || echo "  (no logs available)"
    else
        echo -e "Status: ${RED}STOPPED${NC}"
        [ -f "${PID_FILE}" ] && echo "(PID file exists but process not running)" && rm -f "${PID_FILE}"
    fi

    echo ""
    echo "Log file: ${LOG_FILE}"
    echo "================================"
}

main() {
    case "$1" in
        start)   start   ;;
        stop)    stop    ;;
        restart) restart ;;
        status)  status  ;;
        *)
            echo "Usage: $0 {start|stop|restart|status}"
            echo ""
            echo "Commands:"
            echo "  start   - Start the service"
            echo "  stop    - Stop the service"
            echo "  restart - Restart the service"
            echo "  status  - Show service status"
            echo ""
            echo "Log file: ${LOG_FILE}"
            exit 1
            ;;
    esac
}

main "$@"
