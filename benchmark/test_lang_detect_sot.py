"""
Quick A/B test: 1-SOT vs 2-SOT language detection on a checkpoint.
Only runs the decoder's first step — no full transcription, so it's fast.

Usage:
    python benchmark/test_lang_detect_sot.py \
        --adapter_path whisper-turbo-multilingual-nan-token/multilingual_nan_token_.../step-120000 \
        [--benchmark_jsonl benchmark/benchmark_v0_0_1.jsonl] \
        [--max_samples 500]
"""

import argparse
import json
import os
import sys
import random
from collections import defaultdict

import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftConfig, LoraConfig
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file as safe_load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from whisper_peft import PeftModelForWhisper
from custom_lang import load_custom_lang_for_inference


def load_audio_16k_mono(path):
    wav, sr = torchaudio.load(path)
    wav = wav.float()
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.squeeze(0)


def detect_language_ab(model, input_features, gen_cfg, num_sot=1):
    """Run language detection with `num_sot` SOT tokens fed to the decoder."""
    lang_to_id = getattr(gen_cfg, "lang_to_id", {})
    id_to_lang = {}
    for k, v in lang_to_id.items():
        name = k[2:-2] if k.startswith("<|") and k.endswith("|>") else k
        id_to_lang[v] = name

    device = input_features.device
    decoder_input_ids = torch.full(
        (1, num_sot), gen_cfg.decoder_start_token_id,
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
    best_id = probs[0].argmax().item()
    best_lang = id_to_lang.get(best_id, f"id_{best_id}")
    return best_lang


def print_confusion(confusion, title):
    all_langs = sorted(
        set(confusion.keys()) | {pl for d in confusion.values() for pl in d}
    )
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    header = f"  {'true/pred':<10}" + "".join(f"{l:>8}" for l in all_langs)
    print(header)
    overall_correct = 0
    overall_total = 0
    for tl in sorted(confusion.keys()):
        row = f"  {tl:<10}" + "".join(
            f"{confusion[tl].get(pl, 0):>8}" for pl in all_langs
        )
        correct = confusion[tl].get(tl, 0)
        lang_total = sum(confusion[tl].values())
        pct = correct / lang_total * 100 if lang_total else 0
        print(f"{row}  acc={pct:.1f}%")
        overall_correct += correct
        overall_total += lang_total
    pct = overall_correct / overall_total * 100 if overall_total else 0
    print(f"\n  Overall: {overall_correct}/{overall_total} = {pct:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument(
        "--benchmark_jsonl",
        default="benchmark/benchmark_v0_0_1.jsonl",
    )
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    adapter_path = os.path.abspath(args.adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    peft_cfg = PeftConfig.from_pretrained(adapter_path)
    base_model_path = peft_cfg.base_model_name_or_path
    print(f"Base model: {base_model_path}")
    print(f"Adapter:    {adapter_path}")

    processor = WhisperProcessor.from_pretrained(base_model_path)
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_path)
    base_model.config.use_cache = False

    emb_path = os.path.join(adapter_path, "custom_embeddings.pt")
    custom_langs = None
    if os.path.isfile(emb_path):
        custom_langs = load_custom_lang_for_inference(base_model, processor, adapter_path)
        print(f"Custom languages: {custom_langs}")

    lora_cfg = LoraConfig.from_pretrained(adapter_path)
    model = PeftModelForWhisper(base_model, lora_cfg)
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.isfile(adapter_file):
        adapter_weights = safe_load_file(adapter_file)
    else:
        adapter_file = os.path.join(adapter_path, "adapter_model.bin")
        adapter_weights = torch.load(adapter_file, map_location="cpu", weights_only=True)
    set_peft_model_state_dict(model, adapter_weights)
    model.to(device).eval()
    print("Model loaded.\n")

    gen_cfg = model.generation_config

    samples = []
    with open(args.benchmark_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if obj.get("language"):
                    samples.append(obj)

    if args.max_samples and len(samples) > args.max_samples:
        rng = random.Random(args.seed)
        samples = rng.sample(samples, args.max_samples)

    print(f"Testing {len(samples)} samples ...\n")

    confusion_1sot = defaultdict(lambda: defaultdict(int))
    confusion_2sot = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for s in tqdm(samples, desc="Detecting"):
            wav = load_audio_16k_mono(s["audio"])
            feats = processor.feature_extractor(
                wav.numpy(), sampling_rate=16000, return_tensors="pt",
            ).input_features.to(device)

            true_lang = s["language"]

            pred_1 = detect_language_ab(model, feats, gen_cfg, num_sot=1)
            pred_2 = detect_language_ab(model, feats, gen_cfg, num_sot=2)

            confusion_1sot[true_lang][pred_1] += 1
            confusion_2sot[true_lang][pred_2] += 1

    print_confusion(dict(confusion_1sot), "1-SOT (original inference)")
    print_confusion(dict(confusion_2sot), "2-SOT (aligned with training)")


if __name__ == "__main__":
    main()
