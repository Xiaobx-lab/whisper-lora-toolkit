#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client library & example for the Whisper LoRA ASR service.

Provides helper functions to encode audio files and call the ASR API,
compatible with both the new (/asr) and legacy (/infer) endpoints.

Usage:
    conda activate asr_trainer

    # Quick transcribe from command line:
    python whisper_service/asr_client.py path/to/audio.wav
    python whisper_service/asr_client.py path/to/audio.mp3 --url http://127.0.0.1:8009

    # Use as a library in your own code:
    from whisper_service.asr_client import transcribe_file, transcribe_bytes
    result = transcribe_file("audio.wav")
    print(result["text"], result["language"])
"""

import argparse
import base64
import time
from typing import Optional

import requests


DEFAULT_URL = "http://127.0.0.1:8009"


def read_audio_and_encode_base64(file_path: str, target_sample_rate: int = 16000) -> str:
    """Read any audio file (wav/mp3/flac/ogg/...) and return PCM16 base64 string."""
    from pydub import AudioSegment
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(target_sample_rate).set_sample_width(2)
    return base64.b64encode(audio.raw_data).decode("utf-8")


def read_pcm16_and_encode_base64(file_path: str) -> str:
    """Read a raw PCM16 file and return base64 string (no format conversion)."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def transcribe(
    audio_b64: str,
    sample_rate: int = 16000,
    url: str = DEFAULT_URL,
    timeout: float = 30.0,
) -> dict:
    """
    Call the ASR service with base64-encoded PCM16 audio.

    Args:
        audio_b64:   PCM16 audio data encoded as base64 string.
        sample_rate: Sample rate of the audio (default 16000).
        url:         Service base URL.
        timeout:     HTTP request timeout in seconds.

    Returns:
        dict with "text" and "language" on success,
        or "error" on failure.
    """
    payload = {"audio_data": audio_b64, "sample_rate": sample_rate}
    resp = requests.post(f"{url}/asr", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def transcribe_file(
    file_path: str,
    sample_rate: int = 16000,
    url: str = DEFAULT_URL,
    timeout: float = 30.0,
) -> dict:
    """
    Convenience: read an audio file, encode, and transcribe in one call.

    Supports wav, mp3, flac, ogg, m4a, etc. (anything pydub/ffmpeg handles).
    """
    audio_b64 = read_audio_and_encode_base64(file_path, target_sample_rate=sample_rate)
    return transcribe(audio_b64, sample_rate=sample_rate, url=url, timeout=timeout)


def transcribe_legacy(
    audio_b64: str,
    url: str = DEFAULT_URL,
    timeout: float = 30.0,
) -> dict:
    """Call the legacy /infer endpoint (compatible with whipserZoo)."""
    payload = {"action": "infer", "audio": audio_b64}
    resp = requests.post(f"{url}/infer", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def health_check(url: str = DEFAULT_URL, timeout: float = 5.0) -> bool:
    """Check if the service is alive."""
    try:
        resp = requests.get(f"{url}/healthz", timeout=timeout)
        return resp.status_code == 200 and resp.text.strip() == "ok"
    except requests.RequestException:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file via the Whisper LoRA ASR service",
    )
    parser.add_argument("audio_file", type=str, help="Path to audio file (wav/mp3/flac/...)")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Service URL")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate")
    args = parser.parse_args()

    if not health_check(args.url):
        print(f"Service at {args.url} is not reachable. Is it running?")
        return

    print(f"Transcribing: {args.audio_file}")
    t0 = time.time()
    result = transcribe_file(args.audio_file, sample_rate=args.sample_rate, url=args.url)
    elapsed = time.time() - t0

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Language: {result['language']}")
        print(f"Text:     {result['text']}")
        print(f"Time:     {elapsed:.3f}s")


if __name__ == "__main__":
    main()
