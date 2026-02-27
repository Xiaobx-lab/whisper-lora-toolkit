#!/bin/bash
# Check LibriSpeech parquet -> WAV conversion progress.
# Usage: bash check_convert_progress.sh

WAV_ROOT="/home/jovyan/work/sylvia/asr-data/en/librispeech_asr/all/wav"
JSONL="/home/jovyan/work/sylvia/myWhisperTrainer/data_processing/librispeech_all.jsonl"
LOG="/home/jovyan/work/sylvia/myWhisperTrainer/logs/convert_librispeech.log"

declare -A EXPECTED=(
    ["train.clean.100"]=28539
    ["train.clean.360"]=104014
    ["train.other.500"]=148688
    ["validation.clean"]=2703
    ["validation.other"]=2864
    ["test.clean"]=2620
    ["test.other"]=2939
)
TOTAL_EXPECTED=292367

# Check if process is alive
PID=$(ps aux | grep 'convert_librispeech_to_wav' | grep -v grep | awk 'NR==1{print $2}')

echo "============================================================"
echo "  LibriSpeech Conversion Progress"
echo "============================================================"

if [ -n "$PID" ]; then
    ELAPSED=$(ps -o etime= -p "$PID" 2>/dev/null | xargs)
    CPU=$(ps -o %cpu= -p "$PID" 2>/dev/null | xargs)
    echo "  Status:  RUNNING (PID $PID, elapsed $ELAPSED, CPU ${CPU}%)"
else
    echo "  Status:  NOT RUNNING"
fi
echo ""

# Count per split
TOTAL_DONE=0
printf "  %-20s %8s / %-8s %s\n" "Split" "Done" "Total" "Progress"
echo "  ----------------------------------------------------------"

ORDER=("train.clean.100" "train.clean.360" "train.other.500" "validation.clean" "validation.other" "test.clean" "test.other")

for SPLIT in "${ORDER[@]}"; do
    EXP=${EXPECTED[$SPLIT]}
    DIR="$WAV_ROOT/$SPLIT"
    if [ -d "$DIR" ]; then
        COUNT=$(find "$DIR" -name '*.wav' 2>/dev/null | wc -l)
    else
        COUNT=0
    fi
    TOTAL_DONE=$((TOTAL_DONE + COUNT))

    if [ "$COUNT" -eq 0 ]; then
        BAR="          waiting..."
    elif [ "$COUNT" -ge "$EXP" ]; then
        BAR="[##########] done!"
    else
        PCT=$((COUNT * 100 / EXP))
        FILLED=$((PCT / 10))
        EMPTY=$((10 - FILLED))
        BAR="[$(printf '#%.0s' $(seq 1 $FILLED 2>/dev/null))$(printf '.%.0s' $(seq 1 $EMPTY 2>/dev/null))] ${PCT}%"
    fi
    printf "  %-20s %8d / %-8d %s\n" "$SPLIT" "$COUNT" "$EXP" "$BAR"
done

echo "  ----------------------------------------------------------"
TOTAL_PCT=$((TOTAL_DONE * 100 / TOTAL_EXPECTED))
printf "  %-20s %8d / %-8d %d%%\n" "TOTAL" "$TOTAL_DONE" "$TOTAL_EXPECTED" "$TOTAL_PCT"
echo ""

# JSONL status
if [ -f "$JSONL" ]; then
    LINES=$(wc -l < "$JSONL")
    echo "  JSONL:   $JSONL ($LINES lines)"
else
    echo "  JSONL:   not yet created"
fi

# Final verdict
echo ""
if [ "$TOTAL_DONE" -ge "$TOTAL_EXPECTED" ] && [ -z "$PID" ]; then
    echo "  >>> ALL DONE! Go grab your data and train that model! <<<"
elif [ -z "$PID" ] && [ "$TOTAL_DONE" -lt "$TOTAL_EXPECTED" ]; then
    echo "  >>> Process stopped but conversion incomplete."
    echo "      Re-run the script to resume (it skips existing WAVs). <<<"
else
    REMAINING=$((TOTAL_EXPECTED - TOTAL_DONE))
    echo "  >>> $REMAINING samples remaining, hang tight... <<<"
fi
echo "============================================================"
