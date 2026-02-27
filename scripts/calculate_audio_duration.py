#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate total audio duration from a JSONL file.
Each line in the JSONL should have an "audio" field with the audio file path.
"""

import json
import wave
import argparse
from pathlib import Path
from tqdm import tqdm


def get_wav_duration(file_path: str) -> float:
    """Get duration of a WAV file in seconds."""
    with wave.open(file_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        return frames / float(rate)


def get_mp3_duration(file_path: str) -> float:
    """Get duration of an MP3 file in seconds."""
    from mutagen.mp3 import MP3
    audio = MP3(file_path)
    return audio.info.length


def get_audio_duration(file_path: str) -> float:
    """Get duration of an audio file in seconds."""
    file_path_lower = file_path.lower()
    if file_path_lower.endswith('.wav'):
        return get_wav_duration(file_path)
    elif file_path_lower.endswith('.mp3'):
        return get_mp3_duration(file_path)
    else:
        # Try mutagen for other formats
        from mutagen import File
        audio = File(file_path)
        if audio is not None:
            return audio.info.length
        raise ValueError(f"Unsupported audio format: {file_path}")


def format_duration(seconds: float) -> str:
    """Format duration as hours, minutes, seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.2f}s"


def calculate_total_duration(jsonl_path: str) -> tuple:
    """
    Calculate total duration of all audio files in a JSONL file.
    
    Returns:
        Tuple of (total_seconds, file_count, missing_count, error_count)
    """
    total_duration = 0.0
    file_count = 0
    missing_files = []
    error_files = []
    
    # Count lines first for progress bar
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Found {len(lines)} entries in JSONL file")
    
    for line in tqdm(lines, desc="Calculating duration"):
        line = line.strip()
        if not line:
            continue
            
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            error_files.append(("JSON parse error", str(e)))
            continue
            
        audio_path = data.get('audio', '')
        if not audio_path:
            continue
            
        if not Path(audio_path).exists():
            missing_files.append(audio_path)
            continue
            
        try:
            duration = get_audio_duration(audio_path)
            total_duration += duration
            file_count += 1
        except Exception as e:
            error_files.append((audio_path, str(e)))
    
    # Report issues
    if missing_files:
        print(f"\nWarning: {len(missing_files)} files not found")
        for f in missing_files[:5]:
            print(f"  - {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
    
    if error_files:
        print(f"\nWarning: {len(error_files)} files had errors")
        for f, e in error_files[:5]:
            print(f"  - {f}: {e}")
        if len(error_files) > 5:
            print(f"  ... and {len(error_files) - 5} more")
    
    return total_duration, file_count, len(missing_files), len(error_files)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate total audio duration from a JSONL file"
    )
    parser.add_argument(
        "jsonl_file",
        type=str,
        help="Path to the JSONL file containing audio paths"
    )
    args = parser.parse_args()
    
    jsonl_path = Path(args.jsonl_file)
    
    if not jsonl_path.exists():
        print(f"Error: File not found: {jsonl_path}")
        return
    
    print(f"JSONL file: {jsonl_path}")
    print()
    
    total_seconds, file_count, missing, errors = calculate_total_duration(str(jsonl_path))
    
    print()
    print("=" * 50)
    print(f"Processed files: {file_count}")
    print(f"Missing files: {missing}")
    print(f"Error files: {errors}")
    print()
    print(f"Total duration: {format_duration(total_seconds)}")
    print(f"Total seconds: {total_seconds:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
