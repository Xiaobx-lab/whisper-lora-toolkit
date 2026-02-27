#!/usr/bin/env bash
# ============================================================================
# Cleanup old checkpoints, keep best + latest N step-* folders
# 
# START:
#   cd /home/jovyan/work/sylvia/myWhisperTrainer
#   nohup ./cleanup_checkpoints.sh whisper-turbo-minnan-zh-lora/minnan_zh_20260204_074430 5 > logs/cleanup.log 2>&1 &
#
# STOP:
#   pkill -f cleanup_checkpoints.sh
#
# Usage: ./cleanup_checkpoints.sh <run_dir> [keep_count] [interval_seconds]
#   - run_dir:          Path to the training run directory
#   - keep_count:       Number of latest step-* checkpoints to keep (default: 5)
#   - interval_seconds: Check interval in seconds (default: 1800 = 30min)
#
# Note: 'best/' directory is ALWAYS preserved, never deleted.
# ============================================================================

set -euo pipefail

RUN_DIR="${1:?Usage: $0 <run_dir> [keep_count] [interval_seconds]}"
KEEP_COUNT="${2:-5}"
INTERVAL="${3:-1800}"

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "Error: Directory not found: ${RUN_DIR}"
    exit 1
fi

echo "Cleanup daemon started"
echo "  Run dir: ${RUN_DIR}"
echo "  Keep latest: ${KEEP_COUNT} checkpoints (+ best)"
echo "  Check interval: ${INTERVAL} seconds"
echo ""

cleanup_once() {
    # Find all step-* directories, sorted by step number (descending)
    local step_dirs
    step_dirs=$(find "${RUN_DIR}" -maxdepth 1 -type d -name "step-*" | \
        sed 's/.*step-//' | sort -rn | sed "s|^|${RUN_DIR}/step-|")
    
    if [[ -z "${step_dirs}" ]]; then
        return
    fi
    
    # Count total
    local total
    total=$(echo "${step_dirs}" | wc -l)
    
    if [[ ${total} -le ${KEEP_COUNT} ]]; then
        return
    fi
    
    # Delete old ones (keep first KEEP_COUNT)
    local to_delete
    to_delete=$(echo "${step_dirs}" | tail -n +$((KEEP_COUNT + 1)))
    
    for dir in ${to_delete}; do
        if [[ -d "${dir}" ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Removing: ${dir}"
            rm -rf "${dir}"
        fi
    done
}

# Main loop
while true; do
    cleanup_once
    sleep "${INTERVAL}"
done
