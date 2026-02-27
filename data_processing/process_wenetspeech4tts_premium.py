"""
Extract WenetSpeech4TTS Premium tar.gz files and convert to JSONL format.
Single-pass extraction for speed.

Output:
  - WAV files: /home/jovyan/work/sylvia/asr-data/zh/wav/*.wav
  - JSONL file: /home/jovyan/work/sylvia/myWhisperTrainer/data_processing/wenetspeech4tts_premium_all.jsonl
"""

import os
import json
import tarfile
import time

SRC_DIR = "/home/jovyan/work/zhengbolin/Youtube_cantonese_datasets_pipeline/Wenet_tts/WenetSpeech4TTS/Premium"
WAV_DIR = "/home/jovyan/work/sylvia/asr-data/zh/wav"
JSONL_PATH = "/home/jovyan/work/sylvia/myWhisperTrainer/data_processing/wenetspeech4tts_premium_all.jsonl"

os.makedirs(WAV_DIR, exist_ok=True)

tar_files = sorted(
    [f for f in os.listdir(SRC_DIR) if f.endswith(".tar.gz")],
    key=lambda x: int(x.split("_")[-1].replace(".tar.gz", ""))
)

print(f"Found {len(tar_files)} tar.gz files")
print(f"Saving WAV files to: {WAV_DIR}")
print(f"Saving JSONL to: {JSONL_PATH}")

total_count = 0
global_start = time.time()

with open(JSONL_PATH, "w", encoding="utf-8") as f_out:
    for tar_idx, tar_name in enumerate(tar_files):
        tar_path = os.path.join(SRC_DIR, tar_name)
        tar_start = time.time()
        print(f"\n[{tar_idx + 1}/{len(tar_files)}] Processing {tar_name} ...")

        # Single pass: collect txts and extract wavs simultaneously
        txt_map = {}       # basename -> text
        wav_paths = {}     # basename -> wav_path on disk

        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isdir():
                    continue

                basename = os.path.splitext(os.path.basename(member.name))[0]

                if member.name.endswith(".txt"):
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    content = f.read().decode("utf-8").strip()
                    first_line = content.split("\n")[0]
                    parts = first_line.split("\t")
                    if len(parts) >= 2:
                        txt_map[parts[0].strip()] = parts[1].strip()

                elif member.name.endswith(".wav"):
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    wav_filename = os.path.basename(member.name)
                    wav_path = os.path.join(WAV_DIR, wav_filename)
                    with open(wav_path, "wb") as wf:
                        wf.write(f.read())
                    wav_paths[basename] = wav_path

        # Match and write JSONL
        matched = 0
        for basename, wav_path in wav_paths.items():
            text = txt_map.get(basename)
            if text is None:
                continue
            record = {
                "audio": wav_path,
                "text": text,
                "split": "train",
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            matched += 1

        elapsed = time.time() - tar_start
        total_count += matched
        print(f"  {matched} matched samples extracted in {elapsed:.0f}s (running total: {total_count})")

total_elapsed = time.time() - global_start
print(f"\nAll done! {total_count} samples in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
print(f"  WAV files: {WAV_DIR}/")
print(f"  JSONL file: {JSONL_PATH}")
