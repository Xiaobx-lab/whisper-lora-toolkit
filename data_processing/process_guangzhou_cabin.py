"""
Download AlienKevin/guangzhou-cabin-speech from Hugging Face
and convert it to JSONL format with WAV files saved locally.

Dataset: Scripted Cantonese cabin speech (in-vehicle), ~5 hours, 6219 utterances.

Output:
  - WAV files: /home/jovyan/work/sylvia/asr-data/yue/wav_cabin/*.wav
  - JSONL file: /home/jovyan/work/sylvia/myWhisperTrainer/data_processing/guangzhou_cabin_all.jsonl
"""

import os
import json
import soundfile as sf
from datasets import load_dataset

WAV_DIR = "/home/jovyan/work/sylvia/asr-data/yue/wav_cabin"
JSONL_PATH = "/home/jovyan/work/sylvia/myWhisperTrainer/data_processing/guangzhou_cabin_all.jsonl"

os.makedirs(WAV_DIR, exist_ok=True)

print("Loading AlienKevin/guangzhou-cabin-speech from Hugging Face (~1.34 GB)...")
ds = load_dataset("AlienKevin/guangzhou-cabin-speech", split="train")

print(f"Dataset loaded: {len(ds)} samples")
print(f"Features: {ds.features}")
print(f"Saving WAV files to: {WAV_DIR}")
print(f"Saving JSONL to: {JSONL_PATH}")

count = 0
with open(JSONL_PATH, "w", encoding="utf-8") as f_out:
    for idx, sample in enumerate(ds):
        wav_filename = f"guangzhou_cabin_{idx:05d}.wav"
        wav_path = os.path.join(WAV_DIR, wav_filename)

        audio_data = sample["audio"]
        audio_array = audio_data["array"]
        sampling_rate = audio_data["sampling_rate"]

        sf.write(wav_path, audio_array, sampling_rate)

        text = sample["sentence"].replace(" ", "")

        record = {
            "audio": wav_path,
            "text": text,
            "split": "train",
        }
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(ds)} samples...")

print(f"\nDone! Processed {count} samples total.")
print(f"  WAV files: {WAV_DIR}/ ({count} files)")
print(f"  JSONL file: {JSONL_PATH}")
