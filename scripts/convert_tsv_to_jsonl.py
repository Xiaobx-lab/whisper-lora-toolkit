#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert Common Voice validated.tsv to JSONL format.
Supports both Chinese character labels and Roman pinyin labels.
"""

import csv
import json
import re
from pathlib import Path


def contains_japanese_kana(text: str) -> bool:
    """
    Check if text contains Japanese Hiragana or Katakana characters.
    
    Hiragana: U+3040 - U+309F
    Katakana: U+30A0 - U+30FF
    """
    # Match any hiragana or katakana character
    kana_pattern = re.compile(r'[\u3040-\u309f\u30a0-\u30ff]')
    return bool(kana_pattern.search(text))


def extract_chinese_only(text: str) -> str:
    """
    Extract only Chinese characters from text.
    Removes parentheses with romanization and any non-Chinese characters.
    """
    # First remove content in parentheses (both Chinese and English parentheses)
    text = re.sub(r'[（(][^）)]*[）)]', '', text)
    # Keep only Chinese characters (CJK Unified Ideographs)
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return ''.join(chinese_chars)


def extract_roman_pinyin(text: str) -> str:
    """
    Extract Roman pinyin from parentheses in text.
    If multiple variants separated by '|', take the left one.
    
    Example: "愛共兩台機器（Ài kā nn̄g-tâi ki-khì）" -> "Ài kā nn̄g-tâi ki-khì"
    Example: "濁水溪公社（Lô-tsuí-khe Kong-siā | Lô-tsuí-khue Kong-siā）" -> "Lô-tsuí-khe Kong-siā"
    """
    # Match content inside parentheses (both Chinese and English style)
    match = re.search(r'[（(]([^）)]+)[）)]', text)
    if match:
        roman = match.group(1).strip()
        # If contains '|', take the left part only
        if '|' in roman:
            roman = roman.split('|')[0].strip()
        return roman
    return ""


def convert_tsv_to_jsonl(tsv_path: str, clips_dir: str, output_path: str, skipped_path: str):
    """
    Convert validated.tsv to JSONL format with Chinese character labels.
    
    Args:
        tsv_path: Path to validated.tsv
        clips_dir: Absolute path to clips directory
        output_path: Output JSONL file path
        skipped_path: Output JSONL file path for skipped (roman-only) entries
    """
    clips_dir = Path(clips_dir).resolve()
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
    
    print(f"Found {len(rows)} entries in TSV file")
    
    count = 0
    skipped_count = 0
    missing_count = 0
    
    with open(output_path, 'w', encoding='utf-8') as out_f, \
         open(skipped_path, 'w', encoding='utf-8') as skip_f:
        for row in rows:
            audio_filename = row['path']
            sentence = row['sentence']
            
            # Build absolute path
            audio_path = clips_dir / audio_filename
            
            # Skip if file doesn't exist
            if not audio_path.exists():
                missing_count += 1
                continue
            
            # Extract Chinese characters only
            chinese_text = extract_chinese_only(sentence)
            
            # If no Chinese characters, write to skipped file with original text
            if not chinese_text:
                entry = {
                    "audio": str(audio_path),
                    "text": sentence
                }
                skip_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                skipped_count += 1
                continue
            
            # Create JSONL entry
            entry = {
                "audio": str(audio_path),
                "text": chinese_text
            }
            
            out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            count += 1
    
    print(f"Written {count} entries to {output_path}")
    print(f"Written {skipped_count} roman-only entries to {skipped_path}")
    if missing_count > 0:
        print(f"Skipped {missing_count} entries (files not found)")


def convert_tsv_to_roman_jsonl(tsv_path: str, clips_dir: str, output_path: str, skipped_path: str):
    """
    Convert validated.tsv to JSONL format with Roman pinyin labels.
    Skips entries containing Japanese kana (hiragana/katakana).
    
    Args:
        tsv_path: Path to validated.tsv
        clips_dir: Absolute path to clips directory
        output_path: Output JSONL file path
        skipped_path: Output JSONL file path for skipped entries (no roman / has kana)
    """
    clips_dir = Path(clips_dir).resolve()
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
    
    print(f"Found {len(rows)} entries in TSV file")
    
    count = 0
    skipped_no_roman = 0
    skipped_kana = 0
    missing_count = 0
    
    with open(output_path, 'w', encoding='utf-8') as out_f, \
         open(skipped_path, 'w', encoding='utf-8') as skip_f:
        for row in rows:
            audio_filename = row['path']
            sentence = row['sentence']
            
            # Build absolute path
            audio_path = clips_dir / audio_filename
            
            # Skip if file doesn't exist
            if not audio_path.exists():
                missing_count += 1
                continue
            
            # Skip entries with Japanese kana
            if contains_japanese_kana(sentence):
                entry = {
                    "audio": str(audio_path),
                    "text": sentence,
                    "skip_reason": "contains_japanese_kana"
                }
                skip_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                skipped_kana += 1
                continue
            
            # Extract Roman pinyin from parentheses
            roman_text = extract_roman_pinyin(sentence)
            
            # If no Roman pinyin found, skip
            if not roman_text:
                entry = {
                    "audio": str(audio_path),
                    "text": sentence,
                    "skip_reason": "no_roman_pinyin"
                }
                skip_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                skipped_no_roman += 1
                continue
            
            # Create JSONL entry
            entry = {
                "audio": str(audio_path),
                "text": roman_text
            }
            
            out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            count += 1
    
    print(f"Written {count} entries to {output_path}")
    print(f"Skipped {skipped_kana} entries (contains Japanese kana)")
    print(f"Skipped {skipped_no_roman} entries (no Roman pinyin found)")
    if missing_count > 0:
        print(f"Skipped {missing_count} entries (files not found)")


def main_chinese():
    """Generate JSONL with Chinese character labels."""
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "common_voice_taiwan_nan" / "cv-corpus-24.0-2025-12-05" / "nan-tw"
    
    tsv_path = data_dir / "validated.tsv"
    clips_dir = data_dir / "clips"
    output_path = base_dir / "common_voice_taiwanese.jsonl"
    skipped_path = base_dir / "common_voice_skipped_roman.jsonl"
    
    print("=== Generating Chinese Character Labels ===")
    print(f"TSV file: {tsv_path}")
    print(f"Clips directory: {clips_dir}")
    print(f"Output file: {output_path}")
    print(f"Skipped file: {skipped_path}")
    print()
    
    if not tsv_path.exists():
        print(f"Error: TSV file not found: {tsv_path}")
        return
    
    if not clips_dir.exists():
        print(f"Error: Clips directory not found: {clips_dir}")
        return
    
    convert_tsv_to_jsonl(str(tsv_path), str(clips_dir), str(output_path), str(skipped_path))


def main_roman():
    """Generate JSONL with Roman pinyin labels."""
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "common_voice_taiwan_nan" / "cv-corpus-24.0-2025-12-05" / "nan-tw"
    
    tsv_path = data_dir / "validated.tsv"
    clips_dir = data_dir / "clips"
    output_path = base_dir / "common_voice_taiwanese_roman.jsonl"
    skipped_path = base_dir / "common_voice_skipped_roman_mode.jsonl"
    
    print("=== Generating Roman Pinyin Labels ===")
    print(f"TSV file: {tsv_path}")
    print(f"Clips directory: {clips_dir}")
    print(f"Output file: {output_path}")
    print(f"Skipped file: {skipped_path}")
    print()
    
    if not tsv_path.exists():
        print(f"Error: TSV file not found: {tsv_path}")
        return
    
    if not clips_dir.exists():
        print(f"Error: Clips directory not found: {clips_dir}")
        return
    
    convert_tsv_to_roman_jsonl(str(tsv_path), str(clips_dir), str(output_path), str(skipped_path))


def main():
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "chinese":
            main_chinese()
        elif mode == "roman":
            main_roman()
        elif mode == "both":
            main_chinese()
            print()
            main_roman()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python convert_tsv_to_jsonl.py [chinese|roman|both]")
            print("  chinese - Generate Chinese character labels (default)")
            print("  roman   - Generate Roman pinyin labels")
            print("  both    - Generate both")
    else:
        # Default: Chinese mode for backward compatibility
        main_chinese()


if __name__ == "__main__":
    main()
