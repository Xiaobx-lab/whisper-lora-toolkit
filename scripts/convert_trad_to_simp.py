#!/usr/bin/env python3
# convert traditional chinese to simplified chinese for our jsonl file
# Author: Sylvia 
# Date: 2026-01-23
# Usage: python convert_trad_to_simp.py <input_jsonl> <output_jsonl>

import argparse
import json
import sys
from pathlib import Path


def load_converter():
    try:
        from opencc import OpenCC
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: opencc. Install with:\n"
            "  pip install opencc-python-reimplemented"
        ) from exc
    return OpenCC("t2s")


def convert_jsonl(input_path: Path, output_path: Path, text_key: str) -> None:
    converter = load_converter()
    with input_path.open("r", encoding="utf-8") as in_f, output_path.open(
        "w", encoding="utf-8"
    ) as out_f:
        for line_no, line in enumerate(in_f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Invalid JSON on line {line_no} in {input_path}"
                ) from exc
            if text_key in record and isinstance(record[text_key], str):
                record[text_key] = converter.convert(record[text_key])
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Traditional Chinese text field in JSONL to Simplified."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSONL file path.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--text-key",
        default="text",
        help="JSON key to convert (default: text).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")
    if args.output.exists():
        raise SystemExit(f"Output file already exists: {args.output}")

    convert_jsonl(args.input, args.output, args.text_key)


if __name__ == "__main__":
    main()

