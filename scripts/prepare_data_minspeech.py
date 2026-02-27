#!/usr/bin/env python
"""
从 MinSpeech 结构中生成训练和验证的 JSONL 文件
结构说明：
  BASE_DIR/电视剧/集/segments/ 下包含 segment_*.wav 和 segment_*.txt
"""

import json
import random
from pathlib import Path

# 配置
BASE_DIR = Path("/home/jovyan/work/zhengbolin/Youtube_cantonese_datasets_pipeline/MinSpeech/minspeech_test_dev_bound0")
OUTPUT_DIR = Path("data1")
TRAIN_RATIO = 0.8  # 训练集比例
SEED = 42  # 随机种子，保证可重复性


def collect_pairs(base_dir: Path):
    segment_dirs = sorted(base_dir.glob("**/segments"))
    if not segment_dirs:
        print(f"错误: 在 {base_dir} 中未找到 segments 文件夹")
        return []

    data_pairs = []
    missing_txt = 0
    empty_text = 0

    for segment_dir in segment_dirs:
        wav_files = sorted(segment_dir.glob("*.wav"))
        for wav_path in wav_files:
            txt_path = wav_path.with_suffix(".txt")
            if not txt_path.exists():
                missing_txt += 1
                continue
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                empty_text += 1
                continue
            data_pairs.append({
                "audio": str(wav_path.absolute()),
                "text": text
            })

    if missing_txt > 0:
        print(f"跳过 {missing_txt} 个缺失文本的音频")
    if empty_text > 0:
        print(f"跳过 {empty_text} 个文本为空的样本")

    return data_pairs


def main():
    if not BASE_DIR.exists():
        print(f"错误: 数据路径不存在 {BASE_DIR}")
        return

    output_dir = OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    data_pairs = collect_pairs(BASE_DIR)
    if not data_pairs:
        print("错误: 没有有效的音频-文本对")
        return

    print(f"找到 {len(data_pairs)} 个有效的音频-文本对")

    # 打乱并划分数据
    random.seed(SEED)
    random.shuffle(data_pairs)

    split_idx = int(len(data_pairs) * TRAIN_RATIO)
    if split_idx >= len(data_pairs):
        split_idx = len(data_pairs) - 1
    if split_idx < 1:
        split_idx = 1

    train_data = data_pairs[:split_idx]
    valid_data = data_pairs[split_idx:]

    print(f"训练集: {len(train_data)} 样本")
    print(f"验证集: {len(valid_data)} 样本")

    train_file = output_dir / "train.jsonl"
    valid_file = output_dir / "valid.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(valid_file, "w", encoding="utf-8") as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n生成完成:")
    print(f"  训练集: {train_file}")
    print(f"  验证集: {valid_file}")

    print("\n训练集样本预览:")
    for i, item in enumerate(train_data[:3], 1):
        audio_name = Path(item["audio"]).name
        print(f"  {i}. {audio_name}: {item['text']}")

    print("\n验证集样本预览:")
    for i, item in enumerate(valid_data[:3], 1):
        audio_name = Path(item["audio"]).name
        print(f"  {i}. {audio_name}: {item['text']}")


if __name__ == "__main__":
    main()

