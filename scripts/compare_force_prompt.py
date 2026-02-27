"""
Quick A/B comparison: FORCE_DECODER_PROMPT = True vs False
Uses the encoder-unfreeze adapter to test both settings.
"""
import importlib
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transcribe_audio import transcribe_audio, load_config

ADAPTER_PATH = (
    "whisper-turbo-minnan-zh-lora-encoder-unfreeze/"
    "minnan_zh_encoder_unfreeze_20260204_094713/best"
)

TEST_SAMPLES = [
    {
        "audio": "/home/jovyan/work/zhengbolin/Youtube_cantonese_datasets_pipeline/MinSpeech/minspeech_full_adjust_audios/S01/S01097/segments/train/S010970380.wav",
        "ref": "所以我我喝水就好了",
        "dur": "2.6s",
    },
    {
        "audio": "/home/jovyan/work/zhengbolin/Youtube_cantonese_datasets_pipeline/MinSpeech/minspeech_full_adjust_audios/S01/S01165/segments/train/S011651231.wav",
        "ref": "找不到添富收贿证据 随便把他变成被告 根本是乱来嘛",
        "dur": "6.1s",
    },
    {
        "audio": "/home/jovyan/work/zhengbolin/Youtube_cantonese_datasets_pipeline/MinSpeech/minspeech_full_adjust_audios/S03/S03009/segments/train/S030090552.wav",
        "ref": "阿叶 你真的很厉害 一个人抵我们三个",
        "dur": "3.7s",
    },
]


def make_config_variant(base_cfg, force_decoder_prompt: bool):
    """Clone a config module with overridden FORCE_DECODER_PROMPT."""
    clone = types.ModuleType("cfg_clone")
    for attr in dir(base_cfg):
        if not attr.startswith("_"):
            setattr(clone, attr, getattr(base_cfg, attr))
    clone.FORCE_DECODER_PROMPT = force_decoder_prompt
    return clone


def main():
    base_cfg = load_config("configs.config_our_data_encoder_unfreeze")

    cfg_forced = make_config_variant(base_cfg, force_decoder_prompt=True)
    cfg_auto = make_config_variant(base_cfg, force_decoder_prompt=False)

    separator = "=" * 80

    print(separator)
    print("A/B Test: FORCE_DECODER_PROMPT = True  vs  False")
    print(f"Adapter: {ADAPTER_PATH}")
    print(separator)

    for i, sample in enumerate(TEST_SAMPLES, 1):
        print(f"\n--- Sample {i} ({sample['dur']}) ---")
        print(f"  Reference : {sample['ref']}")

        # --- FORCE_DECODER_PROMPT = True ---
        result_forced = transcribe_audio(
            sample["audio"], adapter_path=ADAPTER_PATH, cfg=cfg_forced,
        )
        lang_f = result_forced.language or "N/A"
        print(f"  [Forced]   lang={lang_f:<8s}  text: {result_forced.text}")

        # --- FORCE_DECODER_PROMPT = False (auto-detect) ---
        result_auto = transcribe_audio(
            sample["audio"], adapter_path=ADAPTER_PATH, cfg=cfg_auto,
        )
        lang_a = result_auto.language or "N/A"
        print(f"  [Auto]     lang={lang_a:<8s}  text: {result_auto.text}")

    print(f"\n{separator}")
    print("Done!")


if __name__ == "__main__":
    main()
