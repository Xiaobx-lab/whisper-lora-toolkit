# audio_duration_stats.py
# 统计音频文件的时长
# Author: Sylvia 
# Date: 2026-01-22
# Usage: python audio_duration_stats.py <jsonl_path> (default: config.TRAIN_JSONL)

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchaudio

from configs import config


def _setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def _get_duration_seconds(path: str) -> float:
    wav, sr = torchaudio.load(path)
    if sr == 0:
        return 0.0
    return float(wav.shape[1]) / float(sr)


def main():
    logger = _setup_logger()
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else config.TRAIN_JSONL

    total_seconds = 0.0
    count = 0
    missing = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            audio_path = item.get("audio")
            if not audio_path:
                continue
            if not os.path.exists(audio_path):
                missing += 1
                continue
            total_seconds += _get_duration_seconds(audio_path)
            count += 1

    logger.info("jsonl=%s", jsonl_path)
    logger.info("audio_files=%d missing=%d", count, missing)
    logger.info("total_seconds=%.2f", total_seconds)
    total_int = int(total_seconds)
    hours = total_int // 3600
    minutes = (total_int % 3600) // 60
    seconds = total_int % 60
    logger.info("total_hms=%02d:%02d:%02d", hours, minutes, seconds)


if __name__ == "__main__":
    main()

