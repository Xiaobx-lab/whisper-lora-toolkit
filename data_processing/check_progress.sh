#!/bin/bash
# Check progress of WenetSpeech4TTS Premium processing
# Usage: bash check_progress.sh

WAV_DIR="/home/jovyan/work/sylvia/asr-data/zh/wav"
JSONL_PATH="/home/jovyan/work/sylvia/myWhisperTrainer/data_processing/wenetspeech4tts_premium_all.jsonl"
LOG_PATH="/home/jovyan/work/sylvia/myWhisperTrainer/data_processing/wenetspeech4tts_premium.log"

# Estimated total (10 tars x ~40k each)
ESTIMATED_TOTAL=408250

echo "============================================"
echo "  WenetSpeech4TTS Premium Processing Status"
echo "============================================"

# Check if process is still running
PID=$(ps aux | grep 'process_wenetspeech4tts_premium' | grep -v grep | grep 'python process' | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "  Status:   RUNNING (PID: $PID)"
else
    echo "  Status:   FINISHED or NOT RUNNING"
fi

# Count WAV files
WAV_COUNT=$(ls "$WAV_DIR" 2>/dev/null | wc -l)
echo "  WAV files: $WAV_COUNT / ~$ESTIMATED_TOTAL (estimated)"

# Count JSONL lines
if [ -f "$JSONL_PATH" ]; then
    JSONL_COUNT=$(wc -l < "$JSONL_PATH")
    echo "  JSONL lines: $JSONL_COUNT"
else
    echo "  JSONL lines: 0 (file not yet created)"
fi

# Progress percentage based on WAV count
if [ "$WAV_COUNT" -gt 0 ]; then
    PCT=$(echo "scale=1; $WAV_COUNT * 100 / $ESTIMATED_TOTAL" | bc)
    echo "  Progress: ~${PCT}%"
else
    echo "  Progress: 0%"
fi

# WAV dir size
if [ -d "$WAV_DIR" ]; then
    DIR_SIZE=$(du -sh "$WAV_DIR" 2>/dev/null | cut -f1)
    echo "  WAV dir size: $DIR_SIZE"
fi

# Log tail (if anything in it)
LOG_LINES=$(wc -l < "$LOG_PATH" 2>/dev/null || echo 0)
if [ "$LOG_LINES" -gt 0 ]; then
    echo ""
    echo "--- Last 5 lines of log ---"
    tail -5 "$LOG_PATH"
fi

echo "============================================"
