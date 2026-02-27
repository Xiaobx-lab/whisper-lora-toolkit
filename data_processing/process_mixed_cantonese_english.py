"""
Download AlienKevin/mixed_cantonese_and_english_speech from Hugging Face
and convert it to JSONL format with WAV files saved locally.

Output:
  - WAV files: /home/jovyan/work/sylvia/asr-data/yue/wav_mce/*.wav
  - JSONL file: /home/jovyan/work/sylvia/myWhisperTrainer/data_processing/mixed_cantonese_english_all.jsonl
"""

import os
import json
import soundfile as sf
from datasets import load_dataset

# -- Paths --
WAV_DIR = "/home/jovyan/work/sylvia/asr-data/yue/wav_mce"
JSONL_PATH = "/home/jovyan/work/sylvia/myWhisperTrainer/data_processing/mixed_cantonese_english_all.jsonl"

os.makedirs(WAV_DIR, exist_ok=True)

print("Loading dataset from Hugging Face (~3.74GB, grab a coffee)...")
ds = load_dataset("AlienKevin/mixed_cantonese_and_english_speech", split="train")

print(f"Dataset loaded: {len(ds)} samples")
print(f"Features: {ds.features}")
print(f"Saving WAV files to: {WAV_DIR}")
print(f"Saving JSONL to: {JSONL_PATH}")

count = 0
with open(JSONL_PATH, "w", encoding="utf-8") as f_out:
    for idx, sample in enumerate(ds):
        # Build filename like mce_00000.wav
        wav_filename = f"mce_{idx:05d}.wav"
        wav_path = os.path.join(WAV_DIR, wav_filename)

        # Extract audio data
        audio_data = sample["audio"]
        audio_array = audio_data["array"]
        sampling_rate = audio_data["sampling_rate"]

        # Save as WAV
        sf.write(wav_path, audio_array, sampling_rate)

        # Get the text (sentence field)
        text = sample["sentence"].strip()

        # Build JSONL record matching librispeech format
        record = {
            "audio": wav_path,
            "text": text,
            "split": "train",
        }
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(ds)} samples...")

print(f"\nDone! Processed {count} samples total.")
print(f"  WAV files: {WAV_DIR}/ ({count} files)")
print(f"  JSONL file: {JSONL_PATH}")
