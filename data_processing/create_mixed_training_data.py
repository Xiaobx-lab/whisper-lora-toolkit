"""
Create mixed multi-language training data for Whisper fine-tuning.

- Minnan: full train.jsonl (NO sampling from valid/test)
- Other languages: sampled from their *_all.jsonl files,
  with benchmark records EXCLUDED to prevent data leakage.

Usage:
    python data_processing/create_mixed_training_data.py

Output:
    data_processing/mixed_train.jsonl
"""

import json
import random
from pathlib import Path

SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent

BENCHMARK_FILE = BASE_DIR / "benchmark" / "benchmark_v0_0_1.jsonl"
OUTPUT_FILE = BASE_DIR / "data_processing" / "mixed_train.jsonl"

# (source_file, language_tag, sample_size)
#   sample_size=0 means use ALL records
SOURCES = [
    (BASE_DIR / "our_data" / "train.jsonl", "nan", 0),
    (BASE_DIR / "data_processing" / "wenetspeech4tts_zh_all.jsonl", "zh", 200_000),
    (BASE_DIR / "data_processing" / "librispeech_all_en.jsonl", "en", 150_000),
    (BASE_DIR / "data_processing" / "japanese_tidy_v4_all.jsonl", "ja", 0),
    (BASE_DIR / "data_processing" / "guangzhou_yue_all.jsonl", "yue", 0),
    (BASE_DIR / "data_processing" / "mixed_cantonese_english_all.jsonl", "yue", 0),
]


def load_benchmark_exclusion_set(path: Path) -> set[str]:
    """Build a set of audio paths from the benchmark for exclusion.

    Uses audio path only (not text) because benchmark text may have been
    normalized (e.g. English lowercased) and won't match the raw source.
    """
    excluded = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            excluded.add(rec["audio"])
    print(f"Loaded {len(excluded)} benchmark audio paths for exclusion.")
    return excluded


def load_and_filter(
    path: Path, exclusion_set: set[str]
) -> tuple[list[dict], int]:
    """Load JSONL, filter out benchmark records. Returns (records, num_excluded)."""
    records = []
    excluded_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["audio"] in exclusion_set:
                excluded_count += 1
            else:
                records.append(rec)
    return records, excluded_count


def main():
    rng = random.Random(SEED)

    exclusion_set = load_benchmark_exclusion_set(BENCHMARK_FILE)

    all_records = []
    stats = []

    for source_path, lang, sample_size in SOURCES:
        print(f"\nLoading {source_path.name} ({lang}) ...")
        records, num_excluded = load_and_filter(source_path, exclusion_set)
        total_after_filter = len(records)
        print(f"  Total: {total_after_filter + num_excluded}  "
              f"Excluded (benchmark): {num_excluded}  "
              f"Available: {total_after_filter}")

        if sample_size > 0 and sample_size < total_after_filter:
            sampled = rng.sample(records, sample_size)
            print(f"  Sampled: {len(sampled)}")
        else:
            sampled = records
            if sample_size > 0:
                print(f"  [WARN] Requested {sample_size} but only "
                      f"{total_after_filter} available, taking all.")
            else:
                print(f"  Using all: {len(sampled)}")

        all_records.extend(sampled)
        stats.append((lang, source_path.name, len(sampled),
                       total_after_filter, num_excluded))

    rng.shuffle(all_records)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total training records: {len(all_records)}")
    print("-" * 70)
    print(f"  {'Lang':<6s} {'Source':<40s} {'Used':>8s} {'Avail':>8s} {'Excl':>6s}")
    print("-" * 70)
    for lang, name, used, avail, excl in stats:
        print(f"  {lang:<6s} {name:<40s} {used:>8d} {avail:>8d} {excl:>6d}")
    print("-" * 70)

    lang_totals = {}
    for lang, _, used, _, _ in stats:
        lang_totals[lang] = lang_totals.get(lang, 0) + used
    total = sum(lang_totals.values())
    print("\n  Language mix:")
    for lang, count in sorted(lang_totals.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"    {lang}: {count:>10d}  ({pct:.1f}%)")
    print(f"    {'TOTAL':}: {total:>10d}")
    print("=" * 70)


if __name__ == "__main__":
    main()
