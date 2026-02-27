#!/usr/bin/env python
"""Quick sanity check: detect language + transcribe 3 samples."""

import json
import re
import sys
import os

import numpy as np
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from whisper_peft import PeftModelForWhisper

BASE_MODEL = (
    "/home/jovyan/work/sylvia/models--openai--whisper-large-v3-turbo"
    "/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
)
ADAPTER_PATH = (
    "/home/jovyan/work/sylvia/myWhisperTrainer"
    "/whisper-turbo-multilingual-lora/multilingual_mixed_20260224_030226/best"
)
BENCHMARK = (
    "/home/jovyan/work/sylvia/myWhisperTrainer/benchmark/benchmark_v0_0_1.jsonl"
)


def detect_language(model, input_features, processor):
    gen_cfg = model.generation_config
    lang_to_id = getattr(gen_cfg, "lang_to_id", None)
    if not lang_to_id:
        return "unknown", 0.0

    id_to_lang = {v: k for k, v in lang_to_id.items()}
    device = input_features.device

    decoder_input_ids = torch.ones(
        (1, 1), device=device, dtype=torch.long,
    ) * gen_cfg.decoder_start_token_id

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    encoder_outputs = base.get_encoder()(input_features[:, :, :3000])
    logits = base(
        encoder_outputs=encoder_outputs,
        decoder_input_ids=decoder_input_ids,
        use_cache=False,
    ).logits[:, -1]

    non_lang_mask = torch.ones_like(logits[0], dtype=torch.bool)
    non_lang_mask[list(lang_to_id.values())] = False
    logits[:, non_lang_mask] = -np.inf

    probs = torch.softmax(logits.float(), dim=-1)

    top_probs, top_ids = probs[0].topk(3)
    top3 = {id_to_lang.get(tid.item(), "?").strip("<|>"): round(p.item(), 4)
            for p, tid in zip(top_probs, top_ids)}

    best_id = probs[0].argmax().item()
    best_tag = id_to_lang.get(best_id, "")
    best_lang = best_tag.strip("<|>") or "unknown"
    best_prob = round(probs[0, best_id].item(), 4)

    return best_lang, best_prob, top3


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"Loading base model on {device} ...")
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=dtype)

    print(f"Loading LoRA adapter from {ADAPTER_PATH} ...")
    model = PeftModelForWhisper.from_pretrained(base_model, ADAPTER_PATH)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []
    model.to(device).eval()

    with open(BENCHMARK, "r") as f:
        samples = [json.loads(line) for i, line in enumerate(f) if i < 3]

    print(f"\nTesting {len(samples)} samples:\n")
    for i, sample in enumerate(samples):
        wav, sr = torchaudio.load(sample["audio"])
        wav = wav.float().mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.squeeze(0)

        features = processor.feature_extractor(
            wav.numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device, dtype=dtype)

        with torch.no_grad():
            detected, prob, top3 = detect_language(model, features, processor)
            gen_ids = model.generate(input_features=features, max_length=448, num_beams=1)

        hyp = processor.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        print(f"--- Sample {i+1} (true lang: {sample.get('language', '?')}) ---")
        print(f"  Detected language  : {detected} (prob={prob})")
        print(f"  Top-3 langs        : {top3}")
        print(f"  Hypothesis (trunc) : {hyp[:80]}...")
        print()


if __name__ == "__main__":
    main()
