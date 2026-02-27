# dump_valid_ref_hyp.py
# 导出验证集的参考和预测结果
# Usage:
# python dump_valid_ref_hyp.py --config path/to/your_config.py --adapter path/to/adapter
# python dump_valid_ref_hyp.py -c path/to/your_config.py -a path/to/adapter
# python dump_valid_ref_hyp.py --config whisper-turbo-minnan-zh-lora/config_minnan.py --adapter whisper-turbo-minnan-zh-lora/minnan_zh_20260123_105009/best --output logs/minnan_valid_ref_hyp.jsonl
# python dump_valid_ref_hyp.py --config whisper-turbo-minnan-zh-lora/config_minnan.py --adapter whisper-turbo-minnan-zh-lora/minnan_zh_20260123_105009/best --output logs/minnan_valid_ref_hyp.jsonl --num_samples 100

import argparse
import importlib.util
import json
import logging
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from configs import config
from asrDataLoader import JsonlASRDataset, CollatorWhisper
from util import normalize_text, set_seed
from whisper_peft import PeftModelForWhisper


ADAPTER_PATH = "whisper-turbo-minnan-zh-lora/minnan_zh_20260123_105009/best"


def _setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def _load_config(config_path: str | None):
    if not config_path:
        return config

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config file not found: {config_path}")

    module_name = os.path.splitext(os.path.basename(config_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load config: {config_path}")

    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


@torch.no_grad()
def dump_ref_hyp(output_path: str, cfg, adapter_path: str, num_samples: int = 100) -> None:
    set_seed(cfg.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = None
    if cfg.USE_BF16 and torch.cuda.is_available():
        torch_dtype = torch.bfloat16

    processor = WhisperProcessor.from_pretrained(cfg.MODEL_NAME)
    # load base model
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.MODEL_NAME,
        dtype=torch_dtype,
    )
    # load lora adapter
    model = PeftModelForWhisper.from_pretrained(model, adapter_path)

    # set forced decoder prompt
    if cfg.FORCE_DECODER_PROMPT:
        processor.tokenizer.set_prefix_tokens(
            language=cfg.FORCE_LANGUAGE,
            task=cfg.FORCE_TASK,
        )
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=cfg.FORCE_LANGUAGE,
            task=cfg.FORCE_TASK,
        )
        model.config.forced_decoder_ids = forced_decoder_ids
        model.generation_config.forced_decoder_ids = forced_decoder_ids
    else:
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None

    # suppress unwanted tokens during generation
    model.config.suppress_tokens = []
    if model.generation_config.suppress_tokens:
        model.generation_config.suppress_tokens = []

    model.to(device)

    valid_ds = JsonlASRDataset(cfg.VALID_JSONL)

    # randomly sample from validation set
    total_samples = len(valid_ds)
    if num_samples > 0 and num_samples < total_samples:
        random.seed(cfg.SEED)
        indices = random.sample(range(total_samples), num_samples)
        valid_ds = Subset(valid_ds, indices)
        logging.getLogger(__name__).info(
            "randomly sampled %d / %d samples", num_samples, total_samples
        )

    collator = CollatorWhisper(processor=processor, max_label_len=cfg.MAX_LABEL_LEN)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.EVAL_BS,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for batch in valid_loader:
            input_features = batch["input_features"].to(
                device, non_blocking=True, dtype=model.dtype
            )
            labels = batch["labels"].to(device, non_blocking=True)

            gen_ids = model.generate(
                input_features=input_features,
                forced_decoder_ids=model.generation_config.forced_decoder_ids,
                max_length=cfg.MAX_LABEL_LEN,
                num_beams=1,
            )

            labels_dec = labels.clone()
            labels_dec[labels_dec == -100] = processor.tokenizer.pad_token_id

            pred_str = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            ref_str = processor.tokenizer.batch_decode(
                labels_dec, skip_special_tokens=True
            )

            for raw_ref, raw_hyp in zip(ref_str, pred_str):
                item = {
                    "ref": raw_ref,
                    "hyp": raw_hyp,
                    "ref_norm": normalize_text(raw_ref),
                    "hyp_norm": normalize_text(raw_hyp),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump validation ref/hyp jsonl")
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to config file (python module).",
    )
    parser.add_argument(
        "--adapter",
        "-a",
        default=ADAPTER_PATH,
        help="Path to LoRA adapter.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="/home/jovyan/work/sylvia/myWhisperTrainer/logs/valid_ref_hyp_our_data_encoder_unfreeze.jsonl",
        help="Output jsonl path.",
    )
    parser.add_argument(
        "--num_samples",
        "-n",
        type=int,
        default=100,
        help="Number of samples to randomly select (default: 100, use 0 or -1 for all).",
    )
    return parser.parse_args()


def main():
    logger = _setup_logger()
    args = _parse_args()
    cfg = _load_config(args.config)
    logger.info("using config: %s", args.config or "configs/config.py")
    logger.info("using adapter: %s", args.adapter)
    logger.info("num_samples: %s", args.num_samples if args.num_samples > 0 else "all")
    logger.info("writing to: %s", args.output)
    dump_ref_hyp(args.output, cfg, args.adapter, args.num_samples)
    logger.info("done")


if __name__ == "__main__":
    main()

