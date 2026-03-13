#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone batch inference sanity check -- no Ray, no HTTP.

Randomly samples a few items from the benchmark JSONL (covering each
language), loads the LoRA adapter, and runs a single model.generate()
call on the full batch to verify true batch inference works correctly.

Usage:
    conda activate asr_trainer
    python whisper_service/test_batch_standalone.py \
        --adapter_path whisper-turbo-multilingual-nan-token/multilingual_nan_token_v2_20260310_063505/best
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file as safe_load_file

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from whisper_peft import PeftModelForWhisper
from custom_lang import load_custom_lang_for_inference

DEFAULT_BASE_MODEL = (
    "/home/jovyan/work/sylvia/models--openai--whisper-large-v3-turbo"
    "/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
)
DEFAULT_BENCHMARK = os.path.join(PROJECT_ROOT, "benchmark", "benchmark_v0_0_1.jsonl")


def load_audio_16k_mono(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    wav = wav.float()
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0)


def sample_from_benchmark(benchmark_path: str, n_per_lang: int = 2, seed: int = 42):
    with open(benchmark_path, "r", encoding="utf-8") as f:
        all_samples = [json.loads(line.strip()) for line in f if line.strip()]

    by_lang = {}
    for s in all_samples:
        lang = s.get("language", "unknown")
        by_lang.setdefault(lang, []).append(s)

    rng = random.Random(seed)
    picked = []
    for lang in sorted(by_lang.keys()):
        pool = by_lang[lang]
        k = min(n_per_lang, len(pool))
        picked.extend(rng.sample(pool, k))

    rng.shuffle(picked)
    return picked


def detect_languages_batch(model, input_features, processor, encoder_outputs=None):
    """
    Detect languages from input_features. If encoder_outputs is provided, reuse it
    (skip redundant encoder forward). Returns (results, encoder_outputs).
    """
    gen_cfg = model.generation_config
    lang_to_id = getattr(gen_cfg, "lang_to_id", None)

    base = model.get_base_model() if hasattr(model, "get_base_model") else model

    if encoder_outputs is None:
        encoder_outputs = base.get_encoder()(input_features[:, :, :3000])

    if not lang_to_id:
        return [("unknown", 0.0)] * input_features.shape[0], encoder_outputs

    id_to_lang = {v: k for k, v in lang_to_id.items()}
    device = input_features.device
    bs = input_features.shape[0]

    decoder_input_ids = torch.full(
        (bs, 1), gen_cfg.decoder_start_token_id,
        device=device, dtype=torch.long,
    )

    logits = base(
        encoder_outputs=encoder_outputs,
        decoder_input_ids=decoder_input_ids,
        use_cache=False,
    ).logits[:, -1]

    non_lang_mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=device)
    non_lang_mask[list(lang_to_id.values())] = False
    logits[:, non_lang_mask] = -np.inf

    probs = torch.softmax(logits.float(), dim=-1)
    best_ids = probs.argmax(dim=-1)

    results = []
    for i in range(bs):
        bid = best_ids[i].item()
        tag = id_to_lang.get(bid, "")
        lang = tag.strip("<|>") or "unknown"
        prob = round(probs[i, bid].item(), 4)
        results.append((lang, prob))
    return results, encoder_outputs


def main():
    parser = argparse.ArgumentParser(description="Standalone batch inference test")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--benchmark_path", type=str, default=DEFAULT_BENCHMARK)
    parser.add_argument("--n_per_lang", type=int, default=2,
                        help="Number of samples per language to test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    adapter_path = os.path.abspath(args.adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # ---- Sample from benchmark ----
    print(f"[1/4] Sampling {args.n_per_lang} per language from benchmark...")
    samples = sample_from_benchmark(args.benchmark_path, args.n_per_lang, args.seed)
    langs_in_sample = sorted(set(s["language"] for s in samples))
    print(f"  Got {len(samples)} samples covering languages: {langs_in_sample}")

    # ---- Load model + adapter ----
    print(f"[2/4] Loading base model + LoRA adapter...")
    processor = WhisperProcessor.from_pretrained(args.base_model)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=dtype,
    )

    custom_langs = None
    if os.path.isfile(os.path.join(adapter_path, "custom_embeddings.pt")):
        custom_langs = load_custom_lang_for_inference(
            base_model, processor, adapter_path,
        )
        if custom_langs:
            print(f"  Custom language tokens restored: {custom_langs}")

    lora_cfg = LoraConfig.from_pretrained(adapter_path)
    model = PeftModelForWhisper(base_model, lora_cfg)
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.isfile(adapter_file):
        adapter_weights = safe_load_file(adapter_file)
    else:
        adapter_file = os.path.join(adapter_path, "adapter_model.bin")
        adapter_weights = torch.load(adapter_file, map_location="cpu", weights_only=True)
    set_peft_model_state_dict(model, adapter_weights)

    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []
    model.to(device)
    model.eval()
    print(f"  Model on {device}, dtype={dtype}")

    # ---- Extract features ----
    print(f"[3/4] Loading audio & extracting features...")
    feats_list = []
    valid_samples = []
    for s in samples:
        try:
            wav = load_audio_16k_mono(s["audio"])
            feat = processor.feature_extractor(
                wav.numpy(), sampling_rate=16000, return_tensors="pt",
            ).input_features[0]
            feats_list.append(feat)
            valid_samples.append(s)
        except Exception as e:
            print(f"  SKIP {s['audio']}: {e}")

    if not feats_list:
        print("No valid audio samples found!")
        return

    stacked = torch.stack(feats_list).to(device, dtype=dtype)
    print(f"  Batch tensor shape: {stacked.shape}  (batch={stacked.shape[0]})")

    # ---- Warmup (important! first call has CUDA kernel compilation overhead) ----
    print(f"\n[4/7] Warming up CUDA kernels...")
    with torch.no_grad():
        _ = model.generate(
            input_features=stacked[:1], max_new_tokens=10, num_beams=1,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"  Warmup done.")

    # ---- A) SEQUENTIAL inference ----
    print(f"\n[5/7] A) SEQUENTIAL inference (one model.generate per sample)...")
    seq_texts = []
    seq_det = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_seq = time.time()
    with torch.no_grad():
        for feat in feats_list:
            single = feat.unsqueeze(0).to(device, dtype=dtype)
            det_one, _ = detect_languages_batch(model, single, processor)
            seq_det.append(det_one[0])
            ids = model.generate(
                input_features=single, max_new_tokens=440, num_beams=1,
            )
            seq_texts.append(
                processor.tokenizer.decode(ids[0], skip_special_tokens=True)
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seq_elapsed = time.time() - t_seq
    seq_tokens = sum(len(processor.tokenizer.encode(t)) for t in seq_texts)

    # ---- B) BATCH inference (double encoder -- old way) ----
    print(f"[6/7] B) BATCH inference (double encoder, no cache)...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_batch_old = time.time()
    with torch.no_grad():
        det_old, _ = detect_languages_batch(model, stacked, processor)
        gen_ids_old = model.generate(
            input_features=stacked, max_new_tokens=440, num_beams=1,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    batch_old_elapsed = time.time() - t_batch_old
    texts_old = processor.tokenizer.batch_decode(gen_ids_old, skip_special_tokens=True)

    # ---- C) BATCH inference (encoder cached -- optimized) ----
    print(f"[7/7] C) BATCH inference (encoder cached, single encoder pass)...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_batch_new = time.time()
    with torch.no_grad():
        det, enc_out = detect_languages_batch(model, stacked, processor)
        lang_list = [lang for lang, _ in det]
        lang_list = [l if l != "unknown" else "en" for l in lang_list]
        gen_ids = model.generate(
            input_features=stacked,
            language=lang_list,
            task="transcribe",
            encoder_outputs=enc_out,
            max_new_tokens=440,
            num_beams=1,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    batch_new_elapsed = time.time() - t_batch_new
    texts = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    batch_tokens = sum(len(processor.tokenizer.encode(t)) for t in texts)

    # ---- Verify: cached version produces same results ----
    print(f"\n{'=' * 80}")
    print(f"RESULT COMPARISON: cached vs uncached")
    print(f"{'=' * 80}")
    all_match = True
    for i, (t_old, t_new) in enumerate(zip(texts_old, texts)):
        match = t_old.strip() == t_new.strip()
        if not match:
            all_match = False
        status = "OK" if match else "MISMATCH"
        print(f"  [{i+1}] {status}")
        if not match:
            print(f"       uncached: {t_old[:80]}")
            print(f"       cached:   {t_new[:80]}")
    print(f"  All match: {all_match}")

    # ---- Transcription results ----
    print(f"\n{'=' * 80}")
    print(f"TRANSCRIPTION RESULTS  ({len(valid_samples)} samples)")
    print(f"{'=' * 80}")
    for i, (s, text, (det_lang, det_prob)) in enumerate(zip(valid_samples, texts, det)):
        ref = s.get("text", "")
        true_lang = s.get("language", "?")
        src = s.get("source", "?")
        print(f"\n--- Sample {i+1} ---")
        print(f"  Source:    {src}")
        print(f"  Language:  true={true_lang}, detected={det_lang} (p={det_prob})")
        print(f"  Reference: {ref[:100]}{'...' if len(ref) > 100 else ''}")
        print(f"  Hypothesis:{text[:100]}{'...' if len(text) > 100 else ''}")

    # ---- Speed comparison ----
    n = len(valid_samples)
    speedup_old = seq_elapsed / batch_old_elapsed if batch_old_elapsed > 0 else 0
    speedup_new = seq_elapsed / batch_new_elapsed if batch_new_elapsed > 0 else 0
    cache_gain = batch_old_elapsed / batch_new_elapsed if batch_new_elapsed > 0 else 0

    print(f"\n{'=' * 80}")
    print(f"SPEED COMPARISON  (all include lang detection, CUDA warmed up)")
    print(f"{'=' * 80}")
    print(f"  Samples:               {n}")
    print(f"  Tokens decoded:        {batch_tokens}")
    print(f"")
    print(f"  A) Sequential:         {seq_elapsed:.3f}s  ({seq_elapsed/n:.3f}s/sample)  {n/seq_elapsed:.1f} samples/s")
    print(f"  B) Batch (2x encoder): {batch_old_elapsed:.3f}s  ({batch_old_elapsed/n:.3f}s/sample)  {n/batch_old_elapsed:.1f} samples/s")
    print(f"  C) Batch (cached enc): {batch_new_elapsed:.3f}s  ({batch_new_elapsed/n:.3f}s/sample)  {n/batch_new_elapsed:.1f} samples/s")
    print(f"")
    print(f"  B vs A (batch gain):   {speedup_old:.2f}x")
    print(f"  C vs A (total gain):   {speedup_new:.2f}x")
    print(f"  C vs B (cache gain):   {cache_gain:.2f}x")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
