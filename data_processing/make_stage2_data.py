"""
Build a balanced stage-2 training set for language-token calibration.

Sources
-------
- nan / en / ja : sampled from mixed_train.jsonl
- zh            : sampled from wenetspeech4tts_zh_all.jsonl
- yue           : sampled from guangzhou_cabin_all.jsonl + wordshk_yue_all.jsonl

Output
------
mix_stage2_train.jsonl  (N_PER_LANG * 5 samples, shuffled)
"""

import json
import random
from pathlib import Path

SEED = 42
N_PER_LANG = 10_000

DATA_DIR = Path(__file__).resolve().parent
MIXED_TRAIN = DATA_DIR / "mixed_train.jsonl"
ZH_ALL = DATA_DIR / "wenetspeech4tts_zh_all.jsonl"
YUE_CABIN = DATA_DIR / "guangzhou_cabin_all.jsonl"
YUE_WORDSHK = DATA_DIR / "wordshk_yue_all.jsonl"
OUTPUT = DATA_DIR / "mix_stage2_train.jsonl"

SAMPLE_FROM_MIXED = {"nan", "en", "ja", "yue"}


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def sample_n(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    if len(pool) < n:
        raise ValueError(
            f"Not enough data: need {n}, only have {len(pool)}"
        )
    return rng.sample(pool, n)


def main():
    rng = random.Random(SEED)
    result: list[dict] = []

    # --- nan / en / ja / yue from mixed_train ---
    buckets: dict[str, list[dict]] = {lang: [] for lang in SAMPLE_FROM_MIXED}
    print(f"Loading {MIXED_TRAIN.name} ...")
    with open(MIXED_TRAIN, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            lang = row.get("language")
            if lang in SAMPLE_FROM_MIXED:
                buckets[lang].append(row)

    # --- yue: merge mixed_train + new sources ---
    print(f"Loading {YUE_CABIN.name} + {YUE_WORDSHK.name} ...")
    yue_extra = load_jsonl(YUE_CABIN) + load_jsonl(YUE_WORDSHK)
    for row in yue_extra:
        row["language"] = "yue"
    buckets["yue"].extend(yue_extra)

    for lang in sorted(SAMPLE_FROM_MIXED):
        pool = buckets[lang]
        print(f"  {lang}: {len(pool)} available, sampling {N_PER_LANG}")
        sampled = sample_n(pool, N_PER_LANG, rng)
        result.extend(sampled)

    # --- zh from wenetspeech4tts ---
    print(f"Loading {ZH_ALL.name} ...")
    zh_pool = load_jsonl(ZH_ALL)
    print(f"  zh: {len(zh_pool)} available, sampling {N_PER_LANG}")
    zh_sampled = sample_n(zh_pool, N_PER_LANG, rng)
    for row in zh_sampled:
        row["language"] = "zh"
    result.extend(zh_sampled)

    # --- shuffle and write ---
    rng.shuffle(result)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for row in result:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDone! Wrote {len(result)} samples to {OUTPUT.name}")
    from collections import Counter
    dist = Counter(r["language"] for r in result)
    for lang, cnt in sorted(dist.items()):
        print(f"  {lang}: {cnt}")


if __name__ == "__main__":
    main()
