#!/usr/bin/env python3
# Merge multiple JSONL files and split into train/valid sets.
# Author: Sylvia
# Date: 2026-01-23
# Usage: python merge_and_split_jsonl.py --output-dir data file1.jsonl file2.jsonl

import argparse
import json
import random
import sys
from pathlib import Path


def read_jsonl_files(input_files):
    records = []
    for input_path in input_files:
        with input_path.open("r", encoding="utf-8") as in_f:
            for line_no, line in enumerate(in_f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"Invalid JSON in {input_path} on line {line_no}"
                    ) from exc
                if not isinstance(record, dict):
                    raise SystemExit(
                        f"Invalid record type in {input_path} on line {line_no}"
                    )
                if "audio" not in record or "text" not in record:
                    raise SystemExit(
                        f"Missing required keys in {input_path} on line {line_no}"
                    )
                records.append(record)
    return records


def write_jsonl(output_path: Path, records):
    with output_path.open("w", encoding="utf-8") as out_f:
        for record in records:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge JSONL files and split into train/valid sets."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        type=Path,
        help="Output directory for train.jsonl and valid.jsonl.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle (default: 42).",
    )
    args = parser.parse_args()

    for input_path in args.inputs:
        if not input_path.exists():
            raise SystemExit(f"Input file not found: {input_path}")

    if not (0.0 < args.train_ratio < 1.0):
        raise SystemExit("train-ratio must be between 0 and 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_file = args.output_dir / "train.jsonl"
    valid_file = args.output_dir / "valid.jsonl"

    if train_file.exists() or valid_file.exists():
        raise SystemExit(
            f"Output files already exist in {args.output_dir}. "
            "Remove them or choose a different output dir."
        )

    records = read_jsonl_files(args.inputs)
    if not records:
        raise SystemExit("No valid records found in input files.")

    random.seed(args.seed)
    random.shuffle(records)

    split_idx = int(len(records) * args.train_ratio)
    if split_idx < 1:
        split_idx = 1
    if split_idx >= len(records):
        split_idx = len(records) - 1

    train_records = records[:split_idx]
    valid_records = records[split_idx:]

    write_jsonl(train_file, train_records)
    write_jsonl(valid_file, valid_records)

    print(f"Total records: {len(records)}")
    print(f"Train records: {len(train_records)} -> {train_file}")
    print(f"Valid records: {len(valid_records)} -> {valid_file}")


if __name__ == "__main__":
    main()

