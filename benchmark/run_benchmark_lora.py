#!/usr/bin/env python
"""
Benchmark a LoRA-finetuned Whisper model on the multilingual benchmark.

Supports both auto language detection and forced language/task modes.

Usage:
    conda activate asr_trainer

    # Auto language detection (like the vanilla benchmark):
    python benchmark/run_benchmark_lora.py \
        --adapter_path whisper-turbo-multilingual-lora/multilingual_mixed_20260224_030226/best

    # Force language & task (e.g. for a Minnan-Chinese model):
    python benchmark/run_benchmark_lora.py \
        --adapter_path whisper-turbo-minnan-zh-lora/minnan_zh_20260223_073421/best \
        --force_language zh --force_task transcribe

    # Custom benchmark file & output:
    python benchmark/run_benchmark_lora.py \
        --adapter_path path/to/adapter \
        --benchmark_path benchmark/benchmark_v0_0_1.jsonl \
        --output_dir benchmark/results_my_model
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftConfig, LoraConfig
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file as safe_load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from whisper_peft import PeftModelForWhisper


DEFAULT_BASE_MODEL = (
    "/home/jovyan/work/sylvia/models--openai--whisper-large-v3-turbo"
    "/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
)
DEFAULT_BENCHMARK = (
    "/home/jovyan/work/sylvia/myWhisperTrainer/benchmark/benchmark_v0_0_1.jsonl"
)
MAX_GEN_LEN = 448


# --------------- Text Utilities ---------------

def normalize_text(s: str) -> str:
    s = s.strip()
    s = " ".join(s.split())
    return s


def strip_for_cer(text: str) -> str:
    text = "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("P")
    )
    text = re.sub(r"\s+", "", text)
    text = text.lower()
    return text


def edit_distance(ref: str, hyp: str) -> int:
    m, n = len(ref), len(hyp)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n]


def compute_cer(ref: str, hyp: str) -> dict:
    ref_clean = strip_for_cer(ref)
    hyp_clean = strip_for_cer(hyp)
    ref_len = len(ref_clean)
    if ref_len == 0:
        ed = len(hyp_clean)
        return {
            "cer": 0.0 if ed == 0 else float("inf"),
            "edit_distance": ed,
            "ref_char_count": 0,
        }
    ed = edit_distance(ref_clean, hyp_clean)
    return {
        "cer": ed / ref_len,
        "edit_distance": ed,
        "ref_char_count": ref_len,
    }


# --------------- Audio ---------------

def load_audio_16k_mono(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    wav = wav.float()
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0)


# --------------- I/O ---------------

def load_benchmark(path: str) -> list:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def batch_detect_language(model, input_features, processor, num_sot=1):
    """
    Batch language detection.
    num_sot=1 for strip_bos-trained models, num_sot=2 for legacy models.
    Returns list of (lang_code, probability) tuples.
    """
    gen_cfg = model.generation_config
    lang_to_id = getattr(gen_cfg, "lang_to_id", None)
    if not lang_to_id:
        return [("unknown", 0.0)] * input_features.shape[0]

    id_to_lang = {v: k for k, v in lang_to_id.items()}
    device = input_features.device
    bs = input_features.shape[0]

    decoder_input_ids = torch.full(
        (bs, num_sot), gen_cfg.decoder_start_token_id,
        device=device, dtype=torch.long,
    )

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    encoder_outputs = base.get_encoder()(input_features[:, :, :3000])
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
    return results


# --------------- Report ---------------

def generate_report(results: list, output_dir: str, adapter_path: str,
                    force_language: str, force_task: str):
    report_path = os.path.join(output_dir, "transcription_report.txt")
    total = len(results)
    success = sum(1 for r in results if r["status"] == "success")
    failed = total - success

    successful = [r for r in results if r["status"] == "success"]
    total_ed = sum(r["edit_distance"] for r in successful)
    total_ref = sum(
        r["ref_char_count"] for r in successful if r["ref_char_count"] > 0
    )
    micro_cer = (total_ed / total_ref * 100) if total_ref > 0 else 0.0

    cer_values = [
        r["cer"] * 100 for r in successful if r["ref_char_count"] > 0
    ]
    macro_cer = (sum(cer_values) / len(cer_values)) if cer_values else 0.0

    lang_stats = defaultdict(
        lambda: {
            "samples": 0, "success": 0,
            "total_ed": 0, "total_ref": 0,
            "cer_values": [],
        }
    )
    for r in results:
        lang = r["language"]
        lang_stats[lang]["samples"] += 1
        if r["status"] == "success":
            lang_stats[lang]["success"] += 1
            lang_stats[lang]["total_ed"] += r["edit_distance"]
            if r["ref_char_count"] > 0:
                lang_stats[lang]["total_ref"] += r["ref_char_count"]
                lang_stats[lang]["cer_values"].append(r["cer"] * 100)

    source_stats = defaultdict(
        lambda: {
            "lang": "", "samples": 0, "success": 0,
            "total_ed": 0, "total_ref": 0,
        }
    )
    for r in results:
        src = r["source"]
        source_stats[src]["lang"] = r["language"]
        source_stats[src]["samples"] += 1
        if r["status"] == "success":
            source_stats[src]["success"] += 1
            source_stats[src]["total_ed"] += r["edit_distance"]
            if r["ref_char_count"] > 0:
                source_stats[src]["total_ref"] += r["ref_char_count"]

    detected_stats = defaultdict(lambda: defaultdict(int))
    for r in results:
        if r["status"] == "success":
            detected_stats[r["language"]][r.get("detected_language", "unknown")] += 1

    lang_mode = f"forced ({force_language}/{force_task})" if force_language else "auto"
    adapter_label = os.path.basename(os.path.dirname(adapter_path)) + "/" + os.path.basename(adapter_path)

    lines = []
    lines.append("=" * 72)
    lines.append(
        "Whisper Large V3 Turbo (LoRA Finetuned) - Multilingual Benchmark Report"
    )
    lines.append("=" * 72)
    lines.append(
        f"Timestamp       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append(f"Adapter         : {adapter_path}")
    lines.append(f"Language mode   : {lang_mode}")
    lines.append(f"Output dir      : {output_dir}")
    lines.append("")

    lines.append("=" * 72)
    lines.append("Overall CER")
    lines.append("=" * 72)
    lines.append(f"  Micro CER (char-weighted) : {micro_cer:.4f}%")
    lines.append(f"  Macro CER (sample-avg)    : {macro_cer:.4f}%")
    lines.append("")
    lines.append(f"  Total samples    : {total}")
    lines.append(
        f"  Success          : {success} ({success / total * 100:.1f}%)"
    )
    lines.append(f"  Failed           : {failed}")
    lines.append("")

    lines.append("=" * 72)
    lines.append("Per-Language CER")
    lines.append("=" * 72)
    lines.append(
        f"  {'Lang':<10} {'Samples':>8} {'Success':>8}"
        f" {'Micro CER':>12} {'Macro CER':>12}"
    )
    lines.append(
        f"  {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 12} {'-' * 12}"
    )
    for lang in sorted(lang_stats.keys()):
        s = lang_stats[lang]
        lang_micro = (
            (s["total_ed"] / s["total_ref"] * 100) if s["total_ref"] > 0 else 0.0
        )
        lang_macro = (
            (sum(s["cer_values"]) / len(s["cer_values"]))
            if s["cer_values"] else 0.0
        )
        lines.append(
            f"  {lang:<10} {s['samples']:>8} {s['success']:>8}"
            f" {lang_micro:>11.4f}% {lang_macro:>11.4f}%"
        )
    lines.append("")

    lines.append("=" * 72)
    lines.append("Per-Source CER")
    lines.append("=" * 72)
    lines.append(
        f"  {'Source':<30} {'Lang':<8} {'Samples':>8}"
        f" {'Success':>8} {'Micro CER':>12}"
    )
    lines.append(
        f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 12}"
    )
    for src in sorted(source_stats.keys()):
        s = source_stats[src]
        src_micro = (
            (s["total_ed"] / s["total_ref"] * 100) if s["total_ref"] > 0 else 0.0
        )
        lines.append(
            f"  {src:<30} {s['lang']:<8} {s['samples']:>8}"
            f" {s['success']:>8} {src_micro:>11.4f}%"
        )
    lines.append("")

    lines.append("=" * 72)
    lines.append("Language Detection Summary (true -> detected)")
    lines.append("=" * 72)
    for true_lang in sorted(detected_stats.keys()):
        det_counts = detected_stats[true_lang]
        det_str = ", ".join(
            f"{dl}={cnt}" for dl, cnt in sorted(
                det_counts.items(), key=lambda x: -x[1]
            )
        )
        lines.append(f"  {true_lang:<10} : {det_str}")
    lines.append("")
    lines.append("=" * 72)

    report_text = "\n".join(lines) + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)


# --------------- Main ---------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a LoRA-finetuned Whisper model"
    )
    parser.add_argument(
        "--adapter_path", type=str, required=True,
        help="Path to the LoRA adapter directory (contains adapter_config.json)",
    )
    parser.add_argument(
        "--base_model", type=str, default=None,
        help="Path to base Whisper model. If not set, read from adapter_config.json.",
    )
    parser.add_argument(
        "--benchmark_path", type=str, default=DEFAULT_BENCHMARK,
        help="Path to the benchmark JSONL file",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results. Defaults to benchmark/results_<adapter_name>",
    )
    parser.add_argument(
        "--force_language", type=str, default=None,
        help="Force decoder language (e.g. 'zh'). If not set, auto-detect.",
    )
    parser.add_argument(
        "--force_task", type=str, default="transcribe",
        help="Forced task (default: transcribe)",
    )
    parser.add_argument(
        "--num_sot", type=int, default=1, choices=[1, 2],
        help="Number of SOT tokens for language detection. "
             "1 = strip_bos-trained models (default), 2 = legacy models.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for language detection and generation (default: 16).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    adapter_path = os.path.abspath(args.adapter_path)
    if not os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
        print(f"ERROR: No adapter_config.json found in {adapter_path}")
        sys.exit(1)

    peft_config = PeftConfig.from_pretrained(adapter_path)

    base_model_path = args.base_model or peft_config.base_model_name_or_path
    if not base_model_path:
        base_model_path = DEFAULT_BASE_MODEL

    if args.output_dir:
        output_dir = args.output_dir
    else:
        adapter_name = os.path.basename(os.path.dirname(adapter_path))
        ckpt_name = os.path.basename(adapter_path)
        output_dir = os.path.join(
            os.path.dirname(__file__),
            f"results_{adapter_name}_{ckpt_name}",
        )
    os.makedirs(output_dir, exist_ok=True)

    timestamp_start = time.time()

    # ---- Load model + adapter ----
    print(f"[1/3] Loading base model from {base_model_path} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    processor = WhisperProcessor.from_pretrained(base_model_path)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
    )

    custom_langs = None
    if os.path.isfile(os.path.join(adapter_path, "custom_embeddings.pt")):
        from custom_lang import load_custom_lang_for_inference
        custom_langs = load_custom_lang_for_inference(
            base_model, processor, adapter_path,
        )
        if custom_langs:
            print(f"  Custom language tokens restored: {custom_langs}")

    print(f"  Loading LoRA adapter from {adapter_path} ...")
    lora_cfg = LoraConfig.from_pretrained(adapter_path)
    model = PeftModelForWhisper(base_model, lora_cfg)
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.isfile(adapter_file):
        adapter_weights = safe_load_file(adapter_file)
    else:
        adapter_file = os.path.join(adapter_path, "adapter_model.bin")
        adapter_weights = torch.load(adapter_file, map_location="cpu", weights_only=True)
    set_peft_model_state_dict(model, adapter_weights)

    if args.force_language:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=args.force_language,
            task=args.force_task,
        )
        model.config.forced_decoder_ids = forced_decoder_ids
        model.generation_config.forced_decoder_ids = forced_decoder_ids
        print(f"  Forced decoder: language={args.force_language}, task={args.force_task}")
    else:
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None
        print("  Language detection: auto (no forced language)")

    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []

    model.to(device)
    model.eval()
    print(f"  Model loaded on {device} (dtype={torch_dtype})")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} total, {trainable:,} trainable (LoRA)")

    # ---- Load benchmark ----
    print(f"[2/3] Loading benchmark from {args.benchmark_path} ...")
    samples = load_benchmark(args.benchmark_path)
    print(f"  Total samples: {len(samples)}")

    # ---- Transcribe (batched) ----
    batch_size = args.batch_size
    print(
        f"[3/3] Running transcription "
        f"(batch_size={batch_size}, num_sot={args.num_sot}) ..."
    )

    # Phase 1: Load audio and extract features
    features = []
    load_errors = {}
    for idx, sample in enumerate(
        tqdm(samples, desc="Extracting features", unit="sample")
    ):
        try:
            waveform = load_audio_16k_mono(sample["audio"])
            feat = processor.feature_extractor(
                waveform.numpy(), sampling_rate=16000, return_tensors="pt",
            ).input_features[0]
            features.append((idx, feat))
        except Exception as e:
            load_errors[idx] = str(e)

    # Phase 2: Batch language detection
    detected_langs = {}
    if args.force_language:
        for idx, _ in features:
            detected_langs[idx] = args.force_language
    else:
        for b_start in tqdm(
            range(0, len(features), batch_size),
            desc="Detecting languages",
        ):
            batch = features[b_start : b_start + batch_size]
            stacked = torch.stack([f for _, f in batch]).to(
                device, dtype=torch_dtype,
            )
            with torch.no_grad():
                det = batch_detect_language(
                    model, stacked, processor, args.num_sot,
                )
            for (idx, _), (lang, _prob) in zip(batch, det):
                detected_langs[idx] = lang

    # Phase 3: Group by detected language, batch generate
    lang_groups = defaultdict(list)
    for idx, feat in features:
        lang_groups[detected_langs[idx]].append((idx, feat))

    hypotheses = {}
    total_gen = sum(len(g) for g in lang_groups.values())
    with tqdm(total=total_gen, desc="Generating", unit="sample") as pbar:
        for lang, group in lang_groups.items():
            for b_start in range(0, len(group), batch_size):
                batch = group[b_start : b_start + batch_size]
                indices = [b[0] for b in batch]
                stacked = torch.stack([b[1] for b in batch]).to(
                    device, dtype=torch_dtype,
                )
                with torch.no_grad():
                    gen_ids = model.generate(
                        input_features=stacked,
                        language=lang if lang != "unknown" else None,
                        max_length=MAX_GEN_LEN,
                        num_beams=1,
                    )
                decoded = processor.tokenizer.batch_decode(
                    gen_ids, skip_special_tokens=True,
                )
                for i, hyp in zip(indices, decoded):
                    hypotheses[i] = normalize_text(hyp)
                pbar.update(len(batch))

    # Phase 4: Assemble results
    results = []
    results_path = os.path.join(output_dir, "transcription_results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f_out:
        for idx, sample in enumerate(samples):
            if idx in load_errors:
                cer_info = compute_cer(sample.get("text", ""), "")
                result = {
                    "index": idx + 1,
                    "audio": sample["audio"],
                    "reference": sample.get("text", ""),
                    "hypothesis": "",
                    "language": sample.get("language", "unknown"),
                    "source": sample.get("source", "unknown"),
                    "detected_language": "unknown",
                    "model": "whisper-large-v3-turbo+lora",
                    "adapter": adapter_path,
                    "status": "failed",
                    "cer": round(cer_info["cer"], 6),
                    "edit_distance": cer_info["edit_distance"],
                    "ref_char_count": cer_info["ref_char_count"],
                    "elapsed_time": 0.0,
                    "error": load_errors[idx],
                }
            else:
                hyp = hypotheses[idx]
                reference = sample.get("text", "")
                cer_info = compute_cer(reference, hyp)
                result = {
                    "index": idx + 1,
                    "audio": sample["audio"],
                    "reference": reference,
                    "hypothesis": hyp,
                    "language": sample.get("language", "unknown"),
                    "source": sample.get("source", "unknown"),
                    "detected_language": detected_langs.get(idx, "unknown"),
                    "model": "whisper-large-v3-turbo+lora",
                    "adapter": adapter_path,
                    "status": "success",
                    "cer": round(cer_info["cer"], 6),
                    "edit_distance": cer_info["edit_distance"],
                    "ref_char_count": cer_info["ref_char_count"],
                    "elapsed_time": 0.0,
                    "error": None,
                }
            results.append(result)
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    total_time = time.time() - timestamp_start
    print(f"\nTranscription complete in {total_time / 60:.1f} minutes")

    # ---- Generate report ----
    generate_report(results, output_dir, adapter_path,
                    args.force_language, args.force_task)
    print(f"\nAll done! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
