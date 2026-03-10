"""
Update training data using colleague's quality-profiled MinSpeech CSVs.

What this script does:
  1. Loads benchmark + validation audio paths as an exclusion set (anti-leakage).
  2. Reads the colleague's profiling CSVs as a "whitelist" of quality-verified
     MinSpeech audio files, with optional quality-metric filters on top.
  3. Builds an audio_path -> text mapping from the current mixed_train.jsonl.
     For new paths not yet in mixed_train, reads the companion .txt file that
     sits next to each .wav in the MinSpeech directory tree.
  4. Keeps all non-MinSpeech records from the existing mixed_train.jsonl
     (also filtered against the exclusion set).
  5. Shuffles and writes a new mixed_train_v2.jsonl.

Usage:
    conda run -n asr_trainer python data_processing/update_training_data.py [--dry-run]

The --dry-run flag prints statistics without writing the output file.
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent

BENCHMARK_FILE = BASE_DIR / "benchmark" / "benchmark_v0_0_1.jsonl"
VALIDATION_FILE = BASE_DIR / "our_data" / "valid.jsonl"
CURRENT_TRAIN_FILE = BASE_DIR / "data_processing" / "mixed_train.jsonl"
OUTPUT_FILE = BASE_DIR / "data_processing" / "mixed_train_v2.jsonl"

CSV_DIR = Path(
    "/home/jovyan/work/zhengbolin/Youtube_cantonese_datasets_pipeline"
    "/MinSpeech/minspeech_full_adjust_audios/Analysis_audio/outputs"
)

QUALITY_FILTERS = {
    "min_duration": 0.5,
    "max_duration": 30.0,
    "min_snr": 10.0,
}


def _extract_segment_id(audio_path: str) -> str:
    """Extract segment ID from audio path (e.g. 'S010590771' from '...S010590771.wav')."""
    return Path(audio_path).stem


def load_exclusion_set() -> tuple[set[str], set[str]]:
    """Collect audio paths AND segment IDs from benchmark + validation.

    Returns (excluded_paths, excluded_segment_ids).
    Path-level matching catches exact duplicates; segment-ID matching catches
    the same audio placed in different split directories (e.g. train/ vs dev/).
    """
    excluded_paths = set()
    excluded_sids = set()
    for fpath in [BENCHMARK_FILE, VALIDATION_FILE]:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                audio = rec["audio"]
                excluded_paths.add(audio)
                if "MinSpeech" in audio or "minspeech" in audio:
                    excluded_sids.add(_extract_segment_id(audio))
    return excluded_paths, excluded_sids


def load_csv_whitelist(
    csv_dir: Path, quality_filters: dict
) -> tuple[set[str], Counter]:
    """Read all profiling CSVs and return the set of quality-verified train-split paths.

    Also returns a Counter of rejection reasons for reporting.
    """
    whitelist = set()
    rejection_reasons = Counter()
    total = 0

    csv_files = sorted(glob.glob(str(csv_dir / "profiling_S*.csv")))
    for i, fpath in enumerate(csv_files, 1):
        fname = os.path.basename(fpath)
        t0 = time.time()
        file_count = 0
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                file_count += 1

                if row["split"] != "train":
                    rejection_reasons["non-train split"] += 1
                    continue

                if row.get("error"):
                    rejection_reasons["has error"] += 1
                    continue

                duration = float(row["duration"])
                if duration < quality_filters["min_duration"]:
                    rejection_reasons[f"duration < {quality_filters['min_duration']}s"] += 1
                    continue
                if duration > quality_filters["max_duration"]:
                    rejection_reasons[f"duration > {quality_filters['max_duration']}s"] += 1
                    continue

                snr = float(row["segmental_snr"])
                if snr < quality_filters["min_snr"]:
                    rejection_reasons[f"SNR < {quality_filters['min_snr']}dB"] += 1
                    continue

                whitelist.add(row["path"])
        elapsed = time.time() - t0
        print(f"    [{i}/{len(csv_files)}] {fname}: {file_count:,d} rows ({elapsed:.1f}s)", flush=True)

    print(f"\n  [CSV Whitelist] Scanned {total:,d} rows across {len(csv_files)} CSVs")
    print(f"  Accepted: {len(whitelist):,d}")
    for reason, count in rejection_reasons.most_common():
        print(f"  Rejected ({reason}): {count:,d}")

    return whitelist, rejection_reasons


def load_existing_minspeech_text(train_file: Path) -> dict[str, dict]:
    """Build audio_path -> record mapping for MinSpeech entries in current training data."""
    mapping = {}
    count = 0
    t0 = time.time()
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            count += 1
            if count % 500_000 == 0:
                print(f"    ... scanned {count:,d} lines ({time.time()-t0:.1f}s)", flush=True)
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            audio = rec.get("audio", "")
            if "MinSpeech" in audio or "minspeech" in audio:
                mapping[audio] = rec
    print(f"    ... done: {count:,d} lines, {len(mapping):,d} MinSpeech ({time.time()-t0:.1f}s)", flush=True)
    return mapping


def read_txt_for_wav(wav_path: str) -> str | None:
    """Read the companion .txt file for a .wav path."""
    txt_path = wav_path.rsplit(".", 1)[0] + ".txt"
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def load_non_minspeech_records(train_file: Path) -> list[dict]:
    """Load all non-MinSpeech records from the current training file."""
    records = []
    t0 = time.time()
    count = 0
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            count += 1
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            audio = rec.get("audio", "")
            if "MinSpeech" not in audio and "minspeech" not in audio:
                records.append(rec)
    print(f"    ... done: {len(records):,d} non-MinSpeech records ({time.time()-t0:.1f}s)", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description="Update training data with cleaned MinSpeech")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, don't write output")
    parser.add_argument("--min-duration", type=float, default=QUALITY_FILTERS["min_duration"])
    parser.add_argument("--max-duration", type=float, default=QUALITY_FILTERS["max_duration"])
    parser.add_argument("--min-snr", type=float, default=QUALITY_FILTERS["min_snr"])
    args = parser.parse_args()

    quality_filters = {
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "min_snr": args.min_snr,
    }

    print("=" * 70)
    print("Update Training Data Pipeline")
    print("=" * 70)
    print(f"Quality filters: {quality_filters}")

    # --- Step 1: exclusion set ---
    print("\n[Step 1] Loading exclusion set (benchmark + validation) ...")
    exclusion_paths, exclusion_sids = load_exclusion_set()
    print(f"  Exclusion paths: {len(exclusion_paths)}")
    print(f"  Exclusion segment IDs (MinSpeech): {len(exclusion_sids)}")

    # --- Step 2: CSV whitelist ---
    print("\n[Step 2] Loading colleague's CSV whitelist ...")
    csv_whitelist, _ = load_csv_whitelist(CSV_DIR, quality_filters)

    # --- Step 3: text mapping from current training data ---
    print("\n[Step 3] Building text mapping from current mixed_train.jsonl ...")
    existing_text = load_existing_minspeech_text(CURRENT_TRAIN_FILE)
    print(f"  Existing MinSpeech text records: {len(existing_text)}")

    # --- Step 4: assemble new MinSpeech records ---
    print("\n[Step 4] Assembling new MinSpeech training records ...")
    new_minspeech = []
    stats = Counter()
    t0 = time.time()

    paths_need_existing = []
    paths_need_txt = []
    for audio_path in csv_whitelist:
        sid = _extract_segment_id(audio_path)
        if audio_path in exclusion_paths or sid in exclusion_sids:
            stats["excluded (anti-leakage)"] += 1
        elif audio_path in existing_text:
            paths_need_existing.append(audio_path)
        else:
            paths_need_txt.append(audio_path)

    for audio_path in paths_need_existing:
        rec = existing_text[audio_path]
        if "language" not in rec:
            rec["language"] = "nan"
        new_minspeech.append(rec)
        stats["from existing data"] += 1
    print(f"  Matched from existing data: {stats['from existing data']:,d} ({time.time()-t0:.1f}s)", flush=True)

    print(f"  Reading {len(paths_need_txt):,d} companion .txt files (parallel I/O) ...", flush=True)

    def _read_one(wav_path: str) -> tuple[str, str | None]:
        return wav_path, read_txt_for_wav(wav_path)

    done = 0
    with ThreadPoolExecutor(max_workers=32) as pool:
        for wav_path, text in pool.map(_read_one, paths_need_txt, chunksize=512):
            done += 1
            if done % 50_000 == 0:
                print(f"    ... read {done:,d}/{len(paths_need_txt):,d} txt files ({time.time()-t0:.1f}s)", flush=True)
            if text is None:
                stats["txt file missing"] += 1
                continue
            if not text:
                stats["txt file empty"] += 1
                continue
            segment_id = Path(wav_path).stem
            rec = {
                "audio": wav_path,
                "text": text,
                "segment_id": segment_id,
                "language": "nan",
            }
            new_minspeech.append(rec)
            stats["from txt file (new)"] += 1

    print(f"  New MinSpeech total: {len(new_minspeech)}")
    for reason, count in stats.most_common():
        print(f"    {reason}: {count}")

    # --- Step 5: non-MinSpeech records ---
    print("\n[Step 5] Loading non-MinSpeech records ...")
    non_minspeech = load_non_minspeech_records(CURRENT_TRAIN_FILE)
    non_ms_before = len(non_minspeech)
    non_minspeech = [r for r in non_minspeech if r["audio"] not in exclusion_paths]
    non_ms_excluded = non_ms_before - len(non_minspeech)
    print(f"  Non-MinSpeech records: {len(non_minspeech)} (excluded {non_ms_excluded} for anti-leakage)")

    # --- Step 6: combine, shuffle, write ---
    all_records = new_minspeech + non_minspeech
    rng = random.Random(SEED)
    rng.shuffle(all_records)

    lang_counts = Counter(r.get("language", "unknown") for r in all_records)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Old training size:   {len(existing_text) + non_ms_before:>10,d}")
    print(f"  New training size:   {len(all_records):>10,d}")
    print(f"  MinSpeech (old):     {len(existing_text):>10,d}")
    print(f"  MinSpeech (new):     {len(new_minspeech):>10,d}")
    print(f"    kept from old:     {stats['from existing data']:>10,d}")
    print(f"    newly added:       {stats['from txt file (new)']:>10,d}")
    print(f"    removed (dirty):   {len(existing_text) - stats['from existing data']:>10,d}")
    print(f"    anti-leakage excl: {stats['excluded (anti-leakage)']:>10,d}")
    print(f"  Non-MinSpeech:       {len(non_minspeech):>10,d}")
    print()
    print("  Language distribution:")
    for lang, count in lang_counts.most_common():
        pct = count / len(all_records) * 100
        print(f"    {lang:<6s}: {count:>10,d}  ({pct:.1f}%)")
    print(f"    {'TOTAL':<6s}: {len(all_records):>10,d}")
    print("=" * 70)

    # --- Step 7: verify zero leakage ---
    print("\n[Step 7] Verifying zero data leakage ...")
    leak_count = 0
    for rec in all_records:
        audio = rec["audio"]
        if audio in exclusion_paths:
            leak_count += 1
        if ("MinSpeech" in audio or "minspeech" in audio):
            sid = _extract_segment_id(audio)
            if sid in exclusion_sids:
                leak_count += 1
    if leak_count == 0:
        print("  PASS: Zero leakage confirmed (checked both path and segment_id).")
    else:
        print(f"  FAIL: {leak_count} leaking records found! Aborting.")
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY RUN] No output file written.")
    else:
        print(f"\n[Step 8] Writing to {OUTPUT_FILE} ...")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  Done! {len(all_records):,d} records written.")


if __name__ == "__main__":
    main()
