"""
Verify audio-text alignment for wenetspeech4tts JSONL data.

Steps:
  1. Basic sanity checks (file existence, audio readability)
  2. Random sampling + Whisper transcription vs. label comparison (CER)
"""

import json
import os
import random
import argparse
import soundfile as sf
from collections import Counter

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def basic_checks(data, check_limit=None):
    """Check file existence and audio readability for all (or a subset of) entries."""
    print("=" * 60)
    print("PHASE 1: Basic Sanity Checks")
    print("=" * 60)

    total = len(data)
    subset = data if check_limit is None else random.sample(data, min(check_limit, total))
    print(f"Checking {len(subset)} / {total} entries for file existence & readability...\n")

    missing = []
    unreadable = []
    durations = []
    zero_text = []

    for i, entry in enumerate(subset):
        audio_path = entry["audio"]
        text = entry.get("text", "")

        # text check
        if not text or not text.strip():
            zero_text.append(audio_path)

        # file existence
        if not os.path.exists(audio_path):
            missing.append(audio_path)
            continue

        # readability + duration
        try:
            info = sf.info(audio_path)
            durations.append(info.duration)
        except Exception as e:
            unreadable.append((audio_path, str(e)))

        if (i + 1) % 5000 == 0:
            print(f"  ... checked {i + 1}/{len(subset)}")

    print(f"\nResults:")
    print(f"  Total checked:     {len(subset)}")
    print(f"  Missing files:     {len(missing)}")
    print(f"  Unreadable audio:  {len(unreadable)}")
    print(f"  Empty text labels: {len(zero_text)}")

    if durations:
        avg_dur = sum(durations) / len(durations)
        print(f"  Avg duration:      {avg_dur:.2f}s")
        print(f"  Min duration:      {min(durations):.2f}s")
        print(f"  Max duration:      {max(durations):.2f}s")

    if missing:
        print(f"\n  Sample missing files (up to 5):")
        for p in missing[:5]:
            print(f"    {p}")

    if unreadable:
        print(f"\n  Sample unreadable files (up to 5):")
        for p, e in unreadable[:5]:
            print(f"    {p} -> {e}")

    print()
    return missing, unreadable, zero_text


def compute_cer(ref, hyp):
    """Compute Character Error Rate between ref and hyp strings."""
    # Remove spaces for Chinese CER
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))

    # Simple edit distance
    n = len(ref_chars)
    m = len(hyp_chars)
    if n == 0:
        return float(m) if m > 0 else 0.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[n][m] / n


def whisper_spot_check(data, n_samples=20, model_name="openai/whisper-small"):
    """Randomly sample entries, transcribe with Whisper, compare to labels."""
    print("=" * 60)
    print("PHASE 2: Whisper Transcription Spot Check")
    print("=" * 60)

    import torch
    import numpy as np
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model:  {model_name}")
    print(f"Sampling {n_samples} entries for transcription check...\n")

    # Filter to entries where audio exists
    valid = [e for e in data if os.path.exists(e["audio"])]
    samples = random.sample(valid, min(n_samples, len(valid)))

    print("Loading model...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    # Force Chinese language
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")

    results = []
    total_cer = 0.0

    for i, entry in enumerate(samples):
        audio_path = entry["audio"]
        ref_text = entry["text"]

        # Load audio and resample to 16kHz if needed
        audio, sr = sf.read(audio_path, dtype="float32")
        if sr != 16000:
            # Simple resampling via numpy interpolation
            duration = len(audio) / sr
            target_len = int(duration * 16000)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, target_len),
                np.arange(len(audio)),
                audio
            )
        # If stereo, take first channel
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Transcribe
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        hyp_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        cer = compute_cer(ref_text, hyp_text)
        total_cer += cer

        status = "OK" if cer < 0.3 else "WARN" if cer < 0.6 else "BAD"

        print(f"[{i+1:3d}/{n_samples}] CER={cer:.2%} [{status}]")
        print(f"  REF: {ref_text[:80]}{'...' if len(ref_text) > 80 else ''}")
        print(f"  HYP: {hyp_text[:80]}{'...' if len(hyp_text) > 80 else ''}")
        print()

        results.append({
            "audio": audio_path,
            "ref": ref_text,
            "hyp": hyp_text,
            "cer": cer,
            "status": status,
        })

    avg_cer = total_cer / len(results) if results else 0
    status_counts = Counter(r["status"] for r in results)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Samples checked: {len(results)}")
    print(f"  Average CER:     {avg_cer:.2%}")
    print(f"  OK  (CER<30%):   {status_counts.get('OK', 0)}")
    print(f"  WARN (30-60%):   {status_counts.get('WARN', 0)}")
    print(f"  BAD  (CER>60%):  {status_counts.get('BAD', 0)}")
    print()

    if avg_cer < 0.25:
        print("  --> Audio-text alignment looks GOOD overall.")
    elif avg_cer < 0.45:
        print("  --> Audio-text alignment is MIXED -- some mismatches detected.")
    else:
        print("  --> Audio-text alignment looks PROBLEMATIC -- many mismatches!")

    # Save detailed results
    out_path = os.path.join(os.path.dirname(__file__), "verify_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  Detailed results saved to: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify audio-text alignment in JSONL data")
    parser.add_argument("--jsonl", type=str,
                        default="/home/jovyan/work/sylvia/myWhisperTrainer/data_processing/wenetspeech4tts_zh_all.jsonl",
                        help="Path to the JSONL file")
    parser.add_argument("--basic-check-limit", type=int, default=5000,
                        help="Number of entries for basic checks (None=all)")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of samples for Whisper spot check")
    parser.add_argument("--model", type=str, default="openai/whisper-small",
                        help="Whisper model to use for transcription")
    parser.add_argument("--skip-transcribe", action="store_true",
                        help="Skip the Whisper transcription phase")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading data from: {args.jsonl}")
    data = load_jsonl(args.jsonl)
    print(f"Total entries: {len(data)}\n")

    # Phase 1: basic checks
    basic_checks(data, check_limit=args.basic_check_limit)

    # Phase 2: Whisper spot check
    if not args.skip_transcribe:
        whisper_spot_check(data, n_samples=args.n_samples, model_name=args.model)
    else:
        print("Skipping Whisper transcription phase (--skip-transcribe)")


if __name__ == "__main__":
    main()
