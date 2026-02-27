#!/usr/bin/env python3
"""
Regenerate ori_train.jsonl, ori_dev.jsonl, ori_test.jsonl by scanning
the actual WAV+TXT files on disk under the MinSpeech data directory.

This does NOT rely on the CSV files (which may be stale after data updates).
Instead, it walks the filesystem and uses directory names (train/dev/test)
to determine the split.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(
    "/home/jovyan/work/zhengbolin/Youtube_cantonese_datasets_pipeline/"
    "MinSpeech/minspeech_full_adjust_audios"
)
OUTPUT_DIR = Path("/home/jovyan/work/sylvia/myWhisperTrainer/our_data")
SPLITS = ("train", "dev", "test")


def main():
    if not BASE_DIR.exists():
        print(f"Error: base directory not found: {BASE_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Find all Sxx show directories
    show_dirs = sorted(
        d for d in BASE_DIR.iterdir()
        if d.is_dir() and d.name.startswith("S") and d.name[1:].isdigit()
    )
    print(f"Found {len(show_dirs)} show directories: {[d.name for d in show_dirs]}")

    data_by_split = defaultdict(list)
    stats = {
        "total_wav": 0,
        "missing_txt": 0,
        "empty_txt": 0,
        "episodes_scanned": 0,
    }

    for show_dir in show_dirs:
        # Find episode sub-folders (e.g. S01001, S01002, ...)
        episode_dirs = sorted(
            d for d in show_dir.iterdir()
            if d.is_dir() and d.name.startswith(show_dir.name)
        )

        show_count = 0
        for ep_dir in episode_dirs:
            segments_dir = ep_dir / "segments"
            if not segments_dir.exists():
                continue

            stats["episodes_scanned"] += 1

            for split_name in SPLITS:
                split_dir = segments_dir / split_name
                if not split_dir.exists():
                    continue

                wav_files = sorted(split_dir.glob("*.wav"))
                for wav_path in wav_files:
                    stats["total_wav"] += 1
                    show_count += 1

                    # Find matching txt
                    txt_path = wav_path.with_suffix(".txt")
                    if not txt_path.exists():
                        stats["missing_txt"] += 1
                        continue

                    text = txt_path.read_text(encoding="utf-8").strip()
                    if not text:
                        stats["empty_txt"] += 1
                        continue

                    segment_id = wav_path.stem  # filename without extension

                    entry = {
                        "audio": str(wav_path),
                        "text": text,
                        "segment_id": segment_id,
                    }
                    data_by_split[split_name].append(entry)

        print(f"  {show_dir.name}: {len(episode_dirs)} episodes, {show_count} wav files")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Scan complete!")
    print(f"  Episodes scanned : {stats['episodes_scanned']}")
    print(f"  Total WAV files  : {stats['total_wav']}")
    print(f"  Missing TXT      : {stats['missing_txt']}")
    print(f"  Empty TXT        : {stats['empty_txt']}")
    print()

    # Write JSONL files
    for split_name in SPLITS:
        data = data_by_split.get(split_name, [])
        if not data:
            print(f"  ori_{split_name}.jsonl: 0 samples (skipped)")
            continue

        output_file = OUTPUT_DIR / f"ori_{split_name}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"  ori_{split_name}.jsonl: {len(data):>10,} samples -> {output_file}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
