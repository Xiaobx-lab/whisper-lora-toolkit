#!/usr/bin/env python3
"""
Pool all ori_train/ori_dev/ori_test data together and resplit into:
  - train.jsonl  : training set (bulk)
  - valid.jsonl  : validation set (for model selection / hyperparameter tuning)
  - test.jsonl   : test set (~5000, benchmark only, NOT in model selection loop)

Overwrites existing files in our_data/.
"""

import json
import random
from pathlib import Path

# --- Config ---
DATA_DIR = Path("/home/jovyan/work/sylvia/myWhisperTrainer/our_data")
SEED = 42
TEST_SIZE = 5000
VALID_SIZE = 20000  # ~1.4% of total, reasonable for validation

# --- Load all data ---
all_data = []
for fname in ["ori_train.jsonl", "ori_dev.jsonl", "ori_test.jsonl"]:
    fpath = DATA_DIR / fname
    if not fpath.exists():
        print(f"WARNING: {fpath} not found, skipping")
        continue
    count = 0
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_data.append(line)
                count += 1
    print(f"Loaded {count:>10,} samples from {fname}")

print(f"\nTotal pooled samples: {len(all_data):,}")

# --- Shuffle ---
random.seed(SEED)
random.shuffle(all_data)

# --- Split ---
test_data = all_data[:TEST_SIZE]
valid_data = all_data[TEST_SIZE:TEST_SIZE + VALID_SIZE]
train_data = all_data[TEST_SIZE + VALID_SIZE:]

print(f"\nSplit result:")
print(f"  train : {len(train_data):>10,} samples")
print(f"  valid : {len(valid_data):>10,} samples")
print(f"  test  : {len(test_data):>10,} samples (benchmark)")

# --- Write ---
for split_name, data in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
    out_path = DATA_DIR / f"{split_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line + "\n")
    print(f"  Written: {out_path}")

print("\nDone!")
