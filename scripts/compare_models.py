# compare_models.py
# Compare two LoRA-adapted Whisper models on the same validation samples.
# Randomly selects N samples from valid.jsonl, runs inference with both models,
# and writes a jsonl with ground_truth + both hypotheses side by side.
#
# Usage:
#   python compare_models.py
#   python compare_models.py --num_samples 200
#   python compare_models.py --output logs/model_comparison.jsonl
#   python compare_models.py --adapter_a /path/to/adapter_a --adapter_b /path/to/adapter_b

import argparse
import gc
import json
import logging
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from asrDataLoader import JsonlASRDataset, CollatorWhisper
from util import normalize_text, set_seed
from whisper_peft import PeftModelForWhisper

# ---------- defaults ----------
BASE_MODEL = (
    "/home/jovyan/work/sylvia/"
    "models--openai--whisper-large-v3-turbo/"
    "snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
)
VALID_JSONL = "/home/jovyan/work/sylvia/myWhisperTrainer/our_data/valid.jsonl"

ADAPTER_A = (
    "whisper-turbo-minnan-zh-lora-encoder-unfreeze/"
    "minnan_zh_encoder_unfreeze_20260204_094713/best"
)
ADAPTER_B = (
    "whisper-turbo-minnan-zh-lora/"
    "minnan_zh_20260204_074430/best"
)

LABEL_A = "encoder_unfreeze"
LABEL_B = "encoder_freeze"

NUM_SAMPLES = 200
SEED = 42
EVAL_BS = 8
MAX_LABEL_LEN = 448
NUM_WORKERS = 4
USE_BF16 = True
FORCE_LANGUAGE = "zh"
FORCE_TASK = "transcribe"

OUTPUT_PATH = "logs/model_comparison.jsonl"


def _setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def _load_model(adapter_path: str, device: str, torch_dtype):
    """Load base Whisper + LoRA adapter, configure decoder prompt, return (model, processor)."""
    processor = WhisperProcessor.from_pretrained(BASE_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, dtype=torch_dtype,
    )
    model = PeftModelForWhisper.from_pretrained(model, adapter_path)

    # forced decoder prompt
    processor.tokenizer.set_prefix_tokens(
        language=FORCE_LANGUAGE, task=FORCE_TASK,
    )
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=FORCE_LANGUAGE, task=FORCE_TASK,
    )
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids

    # suppress unwanted tokens
    model.config.suppress_tokens = []
    if model.generation_config.suppress_tokens:
        model.generation_config.suppress_tokens = []

    model.to(device)
    return model, processor


def _unload_model(model):
    """Free GPU memory so the next model can load comfortably."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def _run_inference(model, processor, data_loader, device):
    """Run inference on the data_loader and return lists of (ref, hyp)."""
    refs, hyps = [], []
    for batch in data_loader:
        input_features = batch["input_features"].to(
            device, non_blocking=True, dtype=model.dtype
        )
        labels = batch["labels"].to(device, non_blocking=True)

        gen_ids = model.generate(
            input_features=input_features,
            forced_decoder_ids=model.generation_config.forced_decoder_ids,
            max_length=MAX_LABEL_LEN,
            num_beams=1,
        )

        labels_dec = labels.clone()
        labels_dec[labels_dec == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        ref_str = processor.tokenizer.batch_decode(labels_dec, skip_special_tokens=True)

        refs.extend(ref_str)
        hyps.extend(pred_str)

    return refs, hyps


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two Whisper LoRA models side-by-side"
    )
    parser.add_argument(
        "--valid_jsonl", default=VALID_JSONL,
        help="Path to validation jsonl.",
    )
    parser.add_argument(
        "--adapter_a", default=ADAPTER_A,
        help="Path to adapter A (encoder-unfreeze model).",
    )
    parser.add_argument(
        "--adapter_b", default=ADAPTER_B,
        help="Path to adapter B (encoder-freeze model).",
    )
    parser.add_argument(
        "--label_a", default=LABEL_A,
        help="Human-readable label for model A.",
    )
    parser.add_argument(
        "--label_b", default=LABEL_B,
        help="Human-readable label for model B.",
    )
    parser.add_argument(
        "--num_samples", "-n", type=int, default=NUM_SAMPLES,
        help="Number of samples to randomly select (0 = all).",
    )
    parser.add_argument(
        "--output", "-o", default=OUTPUT_PATH,
        help="Output jsonl path.",
    )
    return parser.parse_args()


def main():
    logger = _setup_logger()
    args = _parse_args()
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if (USE_BF16 and torch.cuda.is_available()) else None

    logger.info("device: %s | dtype: %s", device, torch_dtype)
    logger.info("adapter A (%s): %s", args.label_a, args.adapter_a)
    logger.info("adapter B (%s): %s", args.label_b, args.adapter_b)

    # ---- dataset ----
    full_ds = JsonlASRDataset(args.valid_jsonl)
    total = len(full_ds)
    n = args.num_samples if args.num_samples > 0 else total

    if n < total:
        random.seed(SEED)
        indices = random.sample(range(total), n)
        ds = Subset(full_ds, indices)
        logger.info("sampled %d / %d validation examples", n, total)
    else:
        indices = list(range(total))
        ds = full_ds
        logger.info("using all %d validation examples", total)

    # collect audio paths in dataloader order (shuffle=False, so order is stable)
    audio_paths = [full_ds[i]["audio"] for i in (indices if n < total else range(total))]

    # build a processor just for the collator (lightweight, no model)
    processor_for_collator = WhisperProcessor.from_pretrained(BASE_MODEL)
    processor_for_collator.tokenizer.set_prefix_tokens(
        language=FORCE_LANGUAGE, task=FORCE_TASK,
    )
    collator = CollatorWhisper(processor=processor_for_collator, max_label_len=MAX_LABEL_LEN)
    loader = DataLoader(
        ds,
        batch_size=EVAL_BS,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )

    # ---- model A ----
    logger.info("loading model A (%s) ...", args.label_a)
    t0 = time.time()
    model_a, proc_a = _load_model(args.adapter_a, device, torch_dtype)
    logger.info("model A loaded in %.1fs", time.time() - t0)

    logger.info("running inference with model A ...")
    t0 = time.time()
    refs_a, hyps_a = _run_inference(model_a, proc_a, loader, device)
    logger.info("model A inference done in %.1fs (%d samples)", time.time() - t0, len(hyps_a))
    _unload_model(model_a)

    # ---- model B ----
    logger.info("loading model B (%s) ...", args.label_b)
    t0 = time.time()
    model_b, proc_b = _load_model(args.adapter_b, device, torch_dtype)
    logger.info("model B loaded in %.1fs", time.time() - t0)

    logger.info("running inference with model B ...")
    t0 = time.time()
    refs_b, hyps_b = _run_inference(model_b, proc_b, loader, device)
    logger.info("model B inference done in %.1fs (%d samples)", time.time() - t0, len(hyps_b))
    _unload_model(model_b)

    # ---- sanity check ----
    assert len(refs_a) == len(refs_b), "ref length mismatch between model runs"
    assert len(hyps_a) == len(hyps_b), "hyp length mismatch between model runs"

    # ---- write output ----
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    diff_count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for audio, ref, hyp_a, hyp_b in zip(audio_paths, refs_a, hyps_a, hyps_b):
            hyp_a_clean = normalize_text(hyp_a)
            hyp_b_clean = normalize_text(hyp_b)
            is_different = (hyp_a_clean != hyp_b_clean)
            if is_different:
                diff_count += 1

            item = {
                "audio": audio,
                "ground_truth": normalize_text(ref),
                f"hyp_{args.label_a}": hyp_a_clean,
                f"hyp_{args.label_b}": hyp_b_clean,
                "models_differ": is_different,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("wrote %d samples to %s", len(refs_a), args.output)
    logger.info(
        "models produced different outputs on %d / %d samples (%.1f%%)",
        diff_count, len(refs_a), 100.0 * diff_count / max(len(refs_a), 1),
    )
    logger.info("done -- go grab a coffee and review the results!")


if __name__ == "__main__":
    main()
