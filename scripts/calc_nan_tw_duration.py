#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate total audio duration from Common Voice nan-tw validated.tsv
"""

import os
import csv
from pathlib import Path
from mutagen.mp3 import MP3
from tqdm import tqdm


def calculate_total_duration(tsv_path: str, clips_dir: str) -> float:
    """
    Calculate total duration of audio files listed in the TSV file.
    
    Args:
        tsv_path: Path to the validated.tsv file
        clips_dir: Path to the directory containing the audio clips
        
    Returns:
        Total duration in seconds
    """
    total_duration = 0.0
    missing_files = []
    error_files = []
    
    # Read the TSV file
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
    
    print(f"Found {len(rows)} audio entries in validated.tsv")
    
    for row in tqdm(rows, desc="Calculating duration"):
        audio_filename = row['path']
        audio_path = os.path.join(clips_dir, audio_filename)
        
        if not os.path.exists(audio_path):
            missing_files.append(audio_filename)
            continue
            
        try:
            audio = MP3(audio_path)
            total_duration += audio.info.length
        except Exception as e:
            error_files.append((audio_filename, str(e)))
    
    # Report issues
    if missing_files:
        print(f"\nWarning: {len(missing_files)} files not found")
        if len(missing_files) <= 5:
            for f in missing_files:
                print(f"  - {f}")
    
    if error_files:
        print(f"\nWarning: {len(error_files)} files had errors")
        if len(error_files) <= 5:
            for f, e in error_files:
                print(f"  - {f}: {e}")
    
    return total_duration


def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.2f}s"


def main():
    # Define paths
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "common_voice_taiwan_nan" / "cv-corpus-24.0-2025-12-05" / "nan-tw"
    tsv_path = data_dir / "validated.tsv"
    clips_dir = data_dir / "clips"
    
    print(f"TSV file: {tsv_path}")
    print(f"Clips directory: {clips_dir}")
    print()
    
    # Verify paths exist
    if not tsv_path.exists():
        print(f"Error: TSV file not found at {tsv_path}")
        return
    
    if not clips_dir.exists():
        print(f"Error: Clips directory not found at {clips_dir}")
        return
    
    # Calculate duration
    total_seconds = calculate_total_duration(str(tsv_path), str(clips_dir))
    
    print()
    print("=" * 50)
    print(f"Total audio duration: {format_duration(total_seconds)}")
    print(f"Total seconds: {total_seconds:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
