"""
Create a unified benchmark test set for multi-language Whisper evaluation.

Samples from each language source, adds `language` and `source` fields,
and writes everything to a single JSONL file for easy evaluation.

Usage:
    python benchmark/create_benchmark.py

Output:
    benchmark/benchmark_test.jsonl
"""

import json
import os
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent

# (source_file, language_tag, source_tag, sample_size)
SOURCES = [
    (
        BASE_DIR / "data_processing" / "wenetspeech4tts_zh_all.jsonl",
        "zh",
        "wenetspeech4tts",
        2000,
    ),
    (
        BASE_DIR / "data_processing" / "librispeech_all_en.jsonl",
        "en",
        "librispeech",
        2000,
    ),
    (
        BASE_DIR / "data_processing" / "japanese_tidy_v4_all.jsonl",
        "ja",
        "japanese_tidy_v4",
        2000,
    ),
    (
        BASE_DIR / "data_processing" / "guangzhou_yue_all.jsonl",
        "yue",
        "guangzhou_yue",
        1000,
    ),
    (
        BASE_DIR / "data_processing" / "mixed_cantonese_english_all.jsonl",
        "yue",
        "mixed_cantonese_english",
        1000,
    ),
    (
        BASE_DIR / "our_data" / "valid.jsonl",
        "nan",
        "minspeech_valid",
        2000,
    ),
]

OUTPUT_FILE = BASE_DIR / "benchmark" / "benchmark_test.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    """Load all lines from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sample_records(
    records: list[dict], n: int, rng: random.Random
) -> list[dict]:
    """Sample n records; if n >= len(records), return all (shuffled)."""
    if n >= len(records):
        sampled = records[:]
        rng.shuffle(sampled)
        print(f"  [WARN] Requested {n} but only {len(records)} available, "
              f"taking all.")
        return sampled
    return rng.sample(records, n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = random.Random(SEED)
    all_samples = []
    stats = {}

    for source_path, lang, source_tag, n in SOURCES:
        print(f"Loading {source_path.name} ...")
        records = load_jsonl(source_path)
        print(f"  Total records: {len(records)}")

        sampled = sample_records(records, n, rng)

        # Augment each record with traceability fields
        for rec in sampled:
            rec["language"] = lang
            rec["source"] = source_tag

        all_samples.extend(sampled)
        stats[f"{lang}/{source_tag}"] = {
            "total_in_source": len(records),
            "sampled": len(sampled),
        }
        print(f"  Sampled: {len(sampled)}")

    # Shuffle the final dataset so languages are interleaved
    rng.shuffle(all_samples)

    # Write output
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in all_samples:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    print("\n" + "=" * 60)
    print(f"Benchmark written to: {OUTPUT_FILE}")
    print(f"Total samples: {len(all_samples)}")
    print("-" * 60)
    for key, val in stats.items():
        print(f"  {key:40s}  {val['sampled']:>5d} / {val['total_in_source']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
