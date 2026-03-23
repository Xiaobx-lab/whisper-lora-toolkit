#!/usr/bin/env python
"""
Analyze language detection probability distribution for misdetected samples.

Reads transcription_results.jsonl from a benchmark output directory, finds
samples where detected_language != true language, runs language detection
through the model, and writes a JSON report with full probability distributions.

Usage:
    conda activate asr_trainer

    # Basic usage (auto-detect adapter path from results JSONL):
    python benchmark/analyze_misdetected_langs.py \
        --results_dir benchmark/results_multilingual_nan_token_v2_20260310_063505_best

    # Specify adapter path explicitly:
    python benchmark/analyze_misdetected_langs.py \
        --results_dir benchmark/results_multilingual_nan_token_v2_20260310_063505_best \
        --adapter_path whisper-turbo-multilingual-nan-token/multilingual_nan_token_v2_20260310_063505/best

    # Custom base model:
    python benchmark/analyze_misdetected_langs.py \
        --results_dir benchmark/results_multilingual_nan_token_v2_20260310_063505_best \
        --base_model /path/to/whisper-large-v3-turbo
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import LoraConfig
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file as safe_load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from whisper_peft import PeftModelForWhisper

DEFAULT_BASE_MODEL = (
    "/home/jovyan/work/sylvia/models--openai--whisper-large-v3-turbo"
    "/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
)


def load_audio_16k_mono(path):
    wav, sr = torchaudio.load(path)
    wav = wav.float()
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0)


def find_misdetected_samples(results_dir):
    jsonl_path = os.path.join(results_dir, "transcription_results.jsonl")
    misdetected = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line.strip())
            if d["language"] != d["detected_language"]:
                misdetected.append(d)
    return misdetected


def extract_adapter_path(results_dir):
    jsonl_path = os.path.join(results_dir, "transcription_results.jsonl")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        first = json.loads(f.readline().strip())
    return first.get("adapter", None)


def load_model(base_model_path, adapter_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    processor = WhisperProcessor.from_pretrained(base_model_path)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_path, torch_dtype=torch_dtype,
    )

    if os.path.isfile(os.path.join(adapter_path, "custom_embeddings.pt")):
        from custom_lang import load_custom_lang_for_inference
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
        adapter_weights = torch.load(
            adapter_file, map_location="cpu", weights_only=True,
        )
    set_peft_model_state_dict(model, adapter_weights)

    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []
    model.to(device)
    model.eval()

    return model, processor, device, torch_dtype


def get_lang_probs(model, processor, audio_path, device, torch_dtype):
    gen_cfg = model.generation_config
    lang_to_id = gen_cfg.lang_to_id
    id_to_lang = {v: k for k, v in lang_to_id.items()}

    waveform = load_audio_16k_mono(audio_path)
    duration = round(len(waveform) / 16000, 2)

    feat = processor.feature_extractor(
        waveform.numpy(), sampling_rate=16000, return_tensors="pt",
    ).input_features.to(device, dtype=torch_dtype)

    decoder_input_ids = torch.full(
        (1, 1), gen_cfg.decoder_start_token_id,
        device=device, dtype=torch.long,
    )

    base = model.get_base_model()
    with torch.no_grad():
        encoder_outputs = base.get_encoder()(feat[:, :, :3000])
        logits = base(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        ).logits[:, -1]

    non_lang_mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=device)
    non_lang_mask[list(lang_to_id.values())] = False
    logits[:, non_lang_mask] = -np.inf

    probs = torch.softmax(logits.float(), dim=-1)[0]

    lang_probs = {}
    for token_id, tag in id_to_lang.items():
        lang_code = tag.strip("<|>")
        p = probs[token_id].item()
        if p > 0.0001:
            lang_probs[lang_code] = round(p, 6)

    return dict(sorted(lang_probs.items(), key=lambda x: -x[1])), duration


def main():
    parser = argparse.ArgumentParser(
        description="Analyze language detection for misdetected samples",
    )
    parser.add_argument(
        "--results_dir", type=str, required=True,
        help="Benchmark results directory containing transcription_results.jsonl",
    )
    parser.add_argument(
        "--adapter_path", type=str, default=None,
        help="LoRA adapter path. If not set, read from results JSONL.",
    )
    parser.add_argument(
        "--base_model", type=str, default=DEFAULT_BASE_MODEL,
        help="Base Whisper model path.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path. Defaults to <results_dir>/misdetected_lang_analysis.json",
    )
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    print(f"[1/4] Scanning {results_dir} for misdetected samples...")
    misdetected = find_misdetected_samples(results_dir)
    if not misdetected:
        print("No misdetected samples found. All good!")
        return
    print(f"  Found {len(misdetected)} misdetected samples")

    adapter_path = args.adapter_path or extract_adapter_path(results_dir)
    if not adapter_path:
        print("ERROR: Could not determine adapter path.")
        sys.exit(1)
    print(f"\n[2/4] Loading model from {adapter_path} ...")
    model, processor, device, torch_dtype = load_model(args.base_model, adapter_path)
    print(f"  Model loaded on {device}")

    print(f"\n[3/4] Analyzing language probabilities...")
    output_records = []
    for sample in misdetected:
        lang_probs, duration = get_lang_probs(
            model, processor, sample["audio"], device, torch_dtype,
        )

        record = {
            "index": sample["index"],
            "audio": sample["audio"],
            "true_language": sample["language"],
            "detected_language": sample["detected_language"],
            "duration_sec": duration,
            "reference": sample["reference"],
            "hypothesis": sample["hypothesis"],
            "nan_probability": lang_probs.get("nan", 0.0),
            "language_probabilities": lang_probs,
        }
        output_records.append(record)

        top = list(lang_probs.items())[:3]
        top_str = ", ".join(f"{k}={v:.4f}" for k, v in top)
        print(
            f"  #{sample['index']:>5d} "
            f"{sample['language']}->{sample['detected_language']}  "
            f"dur={duration}s  [{top_str}]"
        )

    out_path = args.output or os.path.join(
        results_dir, "misdetected_lang_analysis.json",
    )
    print(f"\n[4/4] Writing results to {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
