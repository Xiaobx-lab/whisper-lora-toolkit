"""
Download AlienKevin/wordshk_cantonese_speech from Hugging Face
and convert it to JSONL format with WAV files saved locally.

Dataset: Words.hk Cantonese speech, ~114,988 utterances, ~2.29 GB.

Output:
  - WAV files: /home/jovyan/work/sylvia/asr-data/yue/wav_wordshk/*.wav
  - JSONL file: /home/jovyan/work/sylvia/myWhisperTrainer/data_processing/wordshk_all.jsonl
"""

import os
import json
import soundfile as sf
from datasets import load_dataset

WAV_DIR = "/home/jovyan/work/sylvia/asr-data/yue/wav_wordshk"
JSONL_PATH = "/home/jovyan/work/sylvia/myWhisperTrainer/data_processing/wordshk_all.jsonl"

os.makedirs(WAV_DIR, exist_ok=True)

print("Loading AlienKevin/wordshk_cantonese_speech from Hugging Face (~2.29 GB)...")
ds = load_dataset("AlienKevin/wordshk_cantonese_speech", split="train")

print(f"Dataset loaded: {len(ds)} samples")
print(f"Features: {ds.features}")
print(f"Saving WAV files to: {WAV_DIR}")
print(f"Saving JSONL to: {JSONL_PATH}")

count = 0
with open(JSONL_PATH, "w", encoding="utf-8") as f_out:
    for idx, sample in enumerate(ds):
        wav_filename = f"wordshk_{idx:06d}.wav"
        wav_path = os.path.join(WAV_DIR, wav_filename)

        audio_data = sample["audio"]
        audio_array = audio_data["array"]
        sampling_rate = audio_data["sampling_rate"]

        sf.write(wav_path, audio_array, sampling_rate)

        text = sample["text"].strip()

        record = {
            "audio": wav_path,
            "text": text,
            "split": "train",
        }
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1}/{len(ds)} samples...")

print(f"\nDone! Processed {count} samples total.")
print(f"  WAV files: {WAV_DIR}/ ({count} files)")
print(f"  JSONL file: {JSONL_PATH}")
