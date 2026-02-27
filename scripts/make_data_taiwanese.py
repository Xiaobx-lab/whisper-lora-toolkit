# taiwanese data from https://huggingface.co/datasets/sarahwei/Taiwanese-Minnan-Example-Sentences
# local copy: /home/jovyan/work/sylvia/minnan_audio_data/Taiwanese-Minnan-Example-Sentences/data


import json
from pathlib import Path

import pyarrow.parquet as pq


DATA_DIR = Path("/home/jovyan/work/sylvia/minnan_audio_data/Taiwanese-Minnan-Example-Sentences/data")  # 这里放着 train-*.parquet
OUT_WAV_DIR = Path("/home/jovyan/work/sylvia/minnan_audio_data/wavs")
OUT_JSONL = Path("/home/jovyan/work/sylvia/myWhisperTrainer/taiwanese_roaman.jsonl")

# 用哪一列当标签
TEXT_COL = "minnan roman"     # 或 "hanzi" / "minnan roman" / "chinese"
AUDIO_COL = "audio"      # 音频列名
WAV_PREFIX = ""          # 文件名前缀，可留空


BATCH_SIZE = 256

OUT_WAV_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

parquet_files = sorted(DATA_DIR.glob("train-*-of-*.parquet"))
assert parquet_files, f"No parquet files found in {DATA_DIR.resolve()}"

idx = 0
with OUT_JSONL.open("w", encoding="utf-8") as jf:
    for p in parquet_files:
        pf = pq.ParquetFile(str(p))
        added = 0

        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=[TEXT_COL, AUDIO_COL]):
            texts = batch.column(0).to_pylist()
            audio_struct = batch.column(1)

            # audio 是 struct，取其中的 bytes（WAV 文件内容）
            bytes_list = audio_struct.field("bytes").to_pylist()

            for txt, b in zip(texts, bytes_list):
                if txt is None or b is None:
                    continue

                wav_path = OUT_WAV_DIR / f"{WAV_PREFIX}{idx:08d}.wav"
                wav_path.write_bytes(b)

                jf.write(json.dumps({"audio": str(wav_path), "text": txt}, ensure_ascii=False) + "\n")

                idx += 1
                added += 1

        print(f"{p.name}: +{added} (total={idx})")

print(f"\nDONE\nwav_dir: {OUT_WAV_DIR}\njsonl:   {OUT_JSONL}\nrows:    {idx}")
