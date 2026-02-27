#!/usr/bin/env python3
"""
Convert LibriSpeech parquet data to WAV files + JSONL manifest.

Reads parquet files from each split under the 'all' config,
decodes the embedded FLAC audio bytes to WAV (16kHz, PCM-16),
and writes a single JSONL manifest with audio path, text, and split.

Usage:
    python convert_librispeech_to_wav.py            # full run
    python convert_librispeech_to_wav.py --dry-run   # preview only, no file I/O
"""

import os
import io
import sys
import json
import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm


# --------------- paths ---------------
PARQUET_ROOT = "/home/jovyan/work/sylvia/asr-data/en/librispeech_asr/all"
WAV_ROOT = os.path.join(PARQUET_ROOT, "wav")
OUTPUT_JSONL = (
    "/home/jovyan/work/sylvia/myWhisperTrainer/data_processing/librispeech_all.jsonl"
)

SPLITS = [
    "train.clean.100",
    "train.clean.360",
    "train.other.500",
    "validation.clean",
    "validation.other",
    "test.clean",
    "test.other",
]


def parse_args():
    parser = argparse.ArgumentParser(description="LibriSpeech parquet -> WAV + JSONL")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count samples, do not write any files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Only process these splits (e.g. --splits test.clean validation.clean).",
    )
    return parser.parse_args()


def convert_split(split_name, dry_run=False):
    """Process one split: read parquets, extract WAV, return metadata list."""
    split_dir = os.path.join(PARQUET_ROOT, split_name)
    wav_dir = os.path.join(WAV_ROOT, split_name)

    parquet_files = sorted(Path(split_dir).glob("*.parquet"))
    if not parquet_files:
        print(f"  WARNING: no parquet files found in {split_dir}", file=sys.stderr)
        return []

    if not dry_run:
        os.makedirs(wav_dir, exist_ok=True)

    entries = []
    total_rows = 0

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        total_rows += len(df)

    print(f"[{split_name}] {len(parquet_files)} parquet(s), {total_rows} samples")

    if dry_run:
        return [None] * total_rows  # just for counting

    for pf in parquet_files:
        df = pd.read_parquet(pf)
        pbar = tqdm(df.iterrows(), total=len(df), desc=f"  {pf.name}", leave=False)

        for _, row in pbar:
            uid = row["id"]
            text = row["text"]
            audio_dict = row["audio"]

            wav_path = os.path.join(wav_dir, f"{uid}.wav")

            # Skip if already converted (resume-friendly)
            if not os.path.exists(wav_path):
                flac_bytes = audio_dict["bytes"]
                audio_data, sr = sf.read(io.BytesIO(flac_bytes))
                sf.write(wav_path, audio_data, sr, subtype="PCM_16")

            entries.append({
                "audio": wav_path,
                "text": text,
                "split": split_name,
            })

    return entries


def main():
    args = parse_args()
    splits = args.splits if args.splits else SPLITS

    print("=" * 60)
    print("LibriSpeech Parquet -> WAV + JSONL")
    print(f"  Source:  {PARQUET_ROOT}")
    print(f"  WAV:    {WAV_ROOT}")
    print(f"  JSONL:  {OUTPUT_JSONL}")
    print(f"  Mode:   {'DRY RUN' if args.dry_run else 'CONVERT'}")
    print(f"  Splits: {splits}")
    print("=" * 60)

    all_entries = []
    for split in splits:
        if split not in SPLITS:
            print(f"  WARNING: unknown split '{split}', skipping.", file=sys.stderr)
            continue
        entries = convert_split(split, dry_run=args.dry_run)
        all_entries.extend(entries)
        print(f"  -> cumulative: {len(all_entries)} entries")

    if args.dry_run:
        print(f"\n[DRY RUN] Would produce {len(all_entries)} entries. No files written.")
        return

    # Write JSONL
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nAll done! {len(all_entries)} entries -> {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
