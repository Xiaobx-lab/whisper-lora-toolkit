#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concurrent test client for the Whisper LoRA ASR service.

Sends N requests in parallel via ThreadPoolExecutor to trigger
Ray Serve's @serve.batch and verify true batch inference.

Usage:
    conda activate asr_trainer
    python whisper_service/test_client.py examples/en.mp3
    python whisper_service/test_client.py examples/en.mp3 --n 20 --url http://127.0.0.1:8009
"""

import argparse
import base64
import time
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_audio_pcm16_b64(file_path: str, target_sr: int = 16000) -> str:
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_channels(1).set_frame_rate(target_sr).set_sample_width(2)
        return base64.b64encode(audio.raw_data).decode("utf-8")
    except ImportError:
        import torchaudio
        wav, sr = torchaudio.load(file_path)
        wav = wav.float()
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        pcm16 = (wav.squeeze(0).numpy() * 32768.0).clip(-32768, 32767).astype(np.int16)
        return base64.b64encode(pcm16.tobytes()).decode("utf-8")


def send_request(audio_b64: str, sample_rate: int, url: str, idx: int):
    t0 = time.time()
    resp = requests.post(
        f"{url}/asr",
        json={"audio_data": audio_b64, "sample_rate": sample_rate},
    )
    elapsed = time.time() - t0
    result = resp.json()
    text = result.get("text", result.get("error", "???"))
    lang = result.get("language", "?")
    print(f"  [{idx:>3d}] {elapsed:.3f}s | lang={lang} | {text[:80]}")
    return result, elapsed


def main():
    parser = argparse.ArgumentParser(description="Test client for Whisper LoRA ASR")
    parser.add_argument("audio_file", type=str, help="Path to audio file")
    parser.add_argument("--n", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8009", help="Service URL")
    parser.add_argument("--workers", type=int, default=None, help="Thread pool size (default: n)")
    args = parser.parse_args()

    workers = args.workers or args.n

    print(f"Loading audio: {args.audio_file}")
    audio_b64 = load_audio_pcm16_b64(args.audio_file)
    audio_bytes = len(base64.b64decode(audio_b64))
    duration_s = audio_bytes / 2 / 16000
    print(f"Audio: {duration_s:.1f}s, {audio_bytes} bytes PCM16\n")

    print(f"Sending {args.n} concurrent requests (workers={workers})...")
    t_all = time.time()

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(send_request, audio_b64, 16000, args.url, i)
            for i in range(args.n)
        ]
        for f in as_completed(futures):
            results.append(f.result())

    total = time.time() - t_all
    avg = sum(e for _, e in results) / len(results)
    ok = sum(1 for r, _ in results if "text" in r)
    err = sum(1 for r, _ in results if "error" in r)

    print(f"\n{'=' * 50}")
    print(f"Total wall time:  {total:.3f}s")
    print(f"Avg per request:  {avg:.3f}s")
    print(f"Throughput:       {args.n / total:.1f} req/s")
    print(f"Success: {ok}  Errors: {err}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
