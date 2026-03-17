#!/usr/bin/env python3
"""Quick per-sample timing comparison: benchmark samples vs the yue service test clip."""

import json
import os
import random
import sys
import time

import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file as safe_load_file

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from whisper_peft import PeftModelForWhisper
from custom_lang import load_custom_lang_for_inference

BASE = "/home/jovyan/work/sylvia/models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
ADAPTER = "whisper-turbo-multilingual-nan-token/multilingual_nan_token_v2_20260310_063505/best"
YUE_CLIP = "/home/jovyan/work/sylvia/asr-data/yue/wav_mce/mce_13237.wav"


def load_audio(path):
    wav, sr = torchaudio.load(path)
    wav = wav.float()
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0)


def main():
    processor = WhisperProcessor.from_pretrained(BASE)
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    load_custom_lang_for_inference(base_model, processor, ADAPTER)
    lora_cfg = LoraConfig.from_pretrained(ADAPTER)
    model = PeftModelForWhisper(base_model, lora_cfg)
    weights = safe_load_file(os.path.join(ADAPTER, "adapter_model.safetensors"))
    set_peft_model_state_dict(model, weights)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []
    model.to("cuda")
    model.eval()

    benchmark_path = os.path.join(os.path.dirname(__file__), "..", "benchmark", "benchmark_v0_0_1.jsonl")
    with open(benchmark_path) as f:
        all_samples = [json.loads(l.strip()) for l in f if l.strip()]
    by_lang = {}
    for s in all_samples:
        by_lang.setdefault(s["language"], []).append(s)
    rng = random.Random(42)
    picked = []
    for lang in sorted(by_lang):
        picked.extend(rng.sample(by_lang[lang], min(2, len(by_lang[lang]))))
    rng.shuffle(picked)

    items = []
    for s in picked:
        wav = load_audio(s["audio"])
        feat = processor.feature_extractor(wav.numpy(), sampling_rate=16000, return_tensors="pt").input_features[0]
        items.append((feat, wav.shape[0] / 16000, s["language"], s.get("source", "?")))

    wav_yue = load_audio(YUE_CLIP)
    feat_yue = processor.feature_extractor(wav_yue.numpy(), sampling_rate=16000, return_tensors="pt").input_features[0]
    items.append((feat_yue, wav_yue.shape[0] / 16000, "yue", "mce_13237 << service test clip"))

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_features=items[0][0].unsqueeze(0).to("cuda", dtype=torch.bfloat16), max_new_tokens=10, num_beams=1)
    torch.cuda.synchronize()

    print(f"{'#':>3s}  {'Lang':<5s} {'Dur':>5s}  {'Source':<40s}  {'Time':>7s} {'Tok':>5s}  Text")
    print("-" * 130)
    total_time = 0
    for i, (feat, dur, lang, src) in enumerate(items):
        inp = feat.unsqueeze(0).to("cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            gen_ids = model.generate(input_features=inp, max_new_tokens=440, num_beams=1)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        total_time += elapsed
        text = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        n_tok = len(gen_ids[0])
        marker = " <<<" if "service test" in src else ""
        print(f"{i+1:>3d}  {lang:<5s} {dur:5.1f}s  {src:<40s}  {elapsed:6.3f}s {n_tok:>5d}  {text[:50]}{marker}")

    n = len(items)
    print("-" * 130)
    print(f"Total: {total_time:.3f}s  Avg: {total_time/n:.3f}s/sample  ({n} samples)")


if __name__ == "__main__":
    main()
