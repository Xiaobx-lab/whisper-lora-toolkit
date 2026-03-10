# train.py
import os
import math
import json
import shutil
import logging
import argparse
import importlib.util
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import evaluate

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

from util import set_seed, normalize_text
from asrDataLoader import JsonlASRDataset, CollatorWhisper
from whisper_peft import PeftModelForWhisper


# Eval
@torch.no_grad()
def evaluate_cer(model, processor, dataloader, device, max_gen_len: int, cer_metric) -> float:
    model.eval()

    force_task = getattr(config, "FORCE_TASK", "transcribe")

    preds, refs = [], []
    for batch in dataloader:
        input_features = batch["input_features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        gen_kwargs = {"max_length": max_gen_len, "num_beams": 1, "task": force_task}
        if config.FORCE_DECODER_PROMPT:
            gen_kwargs["language"] = config.FORCE_LANGUAGE

        gen_ids = model.generate(
            input_features=input_features,
            **gen_kwargs,
        )

        labels_dec = labels.clone()
        labels_dec[labels_dec == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        ref_str = processor.tokenizer.batch_decode(labels_dec, skip_special_tokens=True)

        pred_str = [normalize_text(s) for s in pred_str]
        ref_str = [normalize_text(s) for s in ref_str]

        preds.extend(pred_str)
        refs.extend(ref_str)

    model.train()
    return float(cer_metric.compute(predictions=preds, references=refs))


def save_ckpt(model, processor, out_dir: str, custom_lang_info=None):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    if custom_lang_info:
        from custom_lang import save_custom_embeddings
        save_custom_embeddings(
            model, out_dir, custom_lang_info["orig_vocab_size"],
            lang_token_ids=custom_lang_info.get("lang_token_ids"),
        )


def _sanitize_run_name(name: str) -> str:
    if not name:
        return "run"
    clean = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    return clean.strip("_") or "run"


def _build_run_dir(base_dir: str) -> str:
    run_name = _sanitize_run_name(getattr(config, "RUN_NAME", "run"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"{run_name}_{ts}")


def update_latest_pointer(base_dir: str, run_dir: str):
    latest_link = os.path.join(base_dir, "latest")
    if os.path.lexists(latest_link):
        if os.path.islink(latest_link) or os.path.isfile(latest_link):
            os.unlink(latest_link)
        else:
            shutil.rmtree(latest_link)
    os.symlink(os.path.abspath(run_dir), latest_link)

    last_run_path = os.path.join(base_dir, "last_run.txt")
    with open(last_run_path, "w", encoding="utf-8") as f:
        f.write(run_dir + "\n")


def save_config_snapshot(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.abspath(config.__file__)
    shutil.copy2(config_path, os.path.join(out_dir, "config.py"))

    cfg = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    with open(os.path.join(out_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def _load_config(config_path: str):
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise SystemExit(f"Config file not found: {config_path}")
    spec = importlib.util.spec_from_file_location("config_runtime", config_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Failed to load config from: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description="Train Whisper with a config file.")
    parser.add_argument(
        "--config",
        default="configs/config.py",
        help="Path to config file (default: configs/config.py).",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="Override EVAL_SAMPLES from config. 0 = use all data.",
    )
    args = parser.parse_args()

    global config
    config = _load_config(args.config)

    # Override config values from command line
    if args.eval_samples is not None:
        config.EVAL_SAMPLES = args.eval_samples

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    set_seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    run_dir = _build_run_dir(config.OUTPUT_DIR)
    os.makedirs(run_dir, exist_ok=True)
    save_config_snapshot(run_dir)
    update_latest_pointer(config.OUTPUT_DIR, run_dir)

    tb_dir = os.path.join(run_dir, "tb")
    writer = SummaryWriter(log_dir=tb_dir)
    logger.info("device=%s, run_dir=%s, tb_dir=%s, log=%s", device, run_dir, tb_dir, log_path)

    # 1) Load
    processor = WhisperProcessor.from_pretrained(config.MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(config.MODEL_NAME)

    model.config.use_cache = False
    if config.FORCE_DECODER_PROMPT:
        processor.tokenizer.set_prefix_tokens(
            language=config.FORCE_LANGUAGE,
            task=config.FORCE_TASK,
        )

    model.generation_config.forced_decoder_ids = None
    
    # Suppress unwanted tokens during generation
    model.config.suppress_tokens = []
    if model.generation_config.suppress_tokens:
        model.generation_config.suppress_tokens = []

    # 2a) Custom language tokens (must happen before PEFT wrapping)
    custom_lang_info = None
    custom_tokens = getattr(config, "ADD_CUSTOM_LANG_TOKENS", None)
    if custom_tokens:
        from custom_lang import add_language_tokens
        init_map = getattr(config, "CUSTOM_LANG_TOKEN_INIT", {})
        custom_lang_info = add_language_tokens(
            model, processor, custom_tokens, init_map,
        )

    # 2b) LoRA - Apply FIRST before freezing
    target_modules = [s.strip() for s in config.TARGET_MODULES.split(",") if s.strip()]
    lora_cfg = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules=target_modules,
    )
    # Use custom PeftModelForWhisper instead of default get_peft_model
    model = PeftModelForWhisper(model, lora_cfg)
    
    # 3) Freeze encoder (but keep LoRA layers trainable)
    if config.FREEZE_ENCODER:
        for name, param in model.named_parameters():
            # Only freeze non-LoRA encoder parameters
            if 'model.encoder' in name and 'lora' not in name:
                param.requires_grad = False
    
    model.print_trainable_parameters()
    
    model.to(device)

    # 3b) Gradient hooks for custom-token embeddings
    embed_hooks = []
    if custom_lang_info:
        from custom_lang import setup_embedding_grad_hooks
        lang_token_ids = None
        if getattr(config, "UNFREEZE_LANG_EMBEDDINGS", False):
            gen_cfg = getattr(
                model.get_base_model() if hasattr(model, "get_base_model") else model,
                "generation_config", None,
            )
            if gen_cfg and hasattr(gen_cfg, "lang_to_id"):
                lang_token_ids = set(gen_cfg.lang_to_id.values())
                logger.info(
                    "UNFREEZE_LANG_EMBEDDINGS=True: %d language token rows will be trainable",
                    len(lang_token_ids),
                )
        embed_hooks = setup_embedding_grad_hooks(
            model, custom_lang_info["orig_vocab_size"],
            trainable_token_ids=lang_token_ids,
        )
        if lang_token_ids:
            custom_lang_info["lang_token_ids"] = lang_token_ids
        model.print_trainable_parameters()

    # 4) Enable gradient checkpointing AFTER applying PEFT and moving to device
    if config.USE_GRAD_CHECKPOINT:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # 4) Data
    train_ds = JsonlASRDataset(config.TRAIN_JSONL, processor.feature_extractor)
    valid_ds_full = JsonlASRDataset(config.VALID_JSONL, processor.feature_extractor)

    # Subsample validation set if EVAL_SAMPLES > 0
    eval_samples = getattr(config, "EVAL_SAMPLES", 0)
    if eval_samples > 0 and eval_samples < len(valid_ds_full):
        import random
        random.seed(config.SEED)
        indices = random.sample(range(len(valid_ds_full)), eval_samples)
        valid_ds = Subset(valid_ds_full, indices)
        logger.info("Subsampled validation set: %d -> %d samples", len(valid_ds_full), eval_samples)
    else:
        valid_ds = valid_ds_full

    force_task = getattr(config, "FORCE_TASK", "transcribe")
    lang_mapping = getattr(config, "LANGUAGE_MAPPING", None)
    if config.FORCE_DECODER_PROMPT:
        collator = CollatorWhisper(
            processor=processor, max_label_len=config.MAX_LABEL_LEN,
            strip_bos=True,
        )
    else:
        collator = CollatorWhisper(
            processor=processor, max_label_len=config.MAX_LABEL_LEN,
            per_sample_language=True,
            default_language=getattr(config, "DEFAULT_LANGUAGE", "zh"),
            default_task=force_task,
            language_mapping=lang_mapping,
            strip_bos=True,
        )
        logger.info(
            "Multilingual mode: per_sample_language=True, "
            "default_language=%s, language_mapping=%s",
            getattr(config, "DEFAULT_LANGUAGE", "zh"), lang_mapping,
        )

    persistent = config.NUM_WORKERS > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=config.TRAIN_BS,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=persistent,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.EVAL_BS,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=persistent,
    )

    # 5) Optim/Sched
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)

    steps_per_epoch = math.ceil(len(train_loader) / config.GRAD_ACCUM)
    total_steps = steps_per_epoch * config.EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    use_bf16 = (config.USE_BF16 and device == "cuda")
    autocast_dtype = torch.bfloat16 if use_bf16 else None

    cer_metric = evaluate.load("cer")

    # 6) Train loop
    global_step = 0
    best_cer = None
    model.train()

    for epoch in range(1, config.EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        running_raw = 0.0
        
        logger.info("epoch=%s/%s", epoch, config.EPOCHS)

        for i, batch in enumerate(train_loader, start=1):
            input_features = batch["input_features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if epoch == 1 and i == 1:
                lab = labels[0].detach().cpu().clone()
                lab[lab == -100] = processor.tokenizer.pad_token_id
                logger.info(
                    "LABEL TOKENS HEAD: %s",
                    processor.tokenizer.convert_ids_to_tokens(lab[:30].tolist()),
                )
                logger.info(
                    "LABEL TEXT: %s",
                    processor.tokenizer.decode(lab.tolist(), skip_special_tokens=False)[:200],
                )


            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    # Use custom PeftModelForWhisper that properly handles input_features
                    outputs = model(input_features=input_features, labels=labels)
                    loss = outputs.loss
            else:
                outputs = model(input_features=input_features, labels=labels)
                loss = outputs.loss

            raw_loss = loss.detach()
            loss = loss / config.GRAD_ACCUM
            loss.backward()
            running += float(loss.item())
            running_raw += float(raw_loss.item())

            if global_step % 10 == 0 and i == 1:
                logger.info(
                    "debug feat=%s labels=%s grad_accum=%s bf16=%s",
                    tuple(input_features.shape),
                    tuple(labels.shape),
                    config.GRAD_ACCUM,
                    use_bf16,
                )

            if i % config.GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    avg_loss = running / 10.0
                    avg_raw_loss = running_raw / 10.0
                    lr = scheduler.get_last_lr()[0] if scheduler is not None else config.LR
                    logger.info(
                        "epoch=%s step=%s avg_loss=%.4f avg_raw_loss=%.4f lr=%.2e",
                        epoch,
                        global_step,
                        avg_loss,
                        avg_raw_loss,
                        lr,
                    )
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/raw_loss", avg_raw_loss, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    running = 0.0
                    running_raw = 0.0

                if config.EVAL_STEPS > 0 and global_step % config.EVAL_STEPS == 0:
                    cer_val = evaluate_cer(model, processor, valid_loader, device, config.MAX_LABEL_LEN, cer_metric)
                    logger.info("eval step=%s CER=%.4f", global_step, cer_val)
                    writer.add_scalar("eval/cer", cer_val, global_step)

                    if best_cer is None or cer_val < best_cer:
                        best_cer = cer_val
                        save_ckpt(model, processor, os.path.join(run_dir, "best"), custom_lang_info)
                        logger.info("save best -> %s", os.path.join(run_dir, "best"))

                if config.SAVE_STEPS > 0 and global_step % config.SAVE_STEPS == 0:
                    out = os.path.join(run_dir, f"step-{global_step}")
                    save_ckpt(model, processor, out, custom_lang_info)
                    logger.info("save -> %s", out)

        # epoch end eval
        cer_epoch = evaluate_cer(model, processor, valid_loader, device, config.MAX_LABEL_LEN, cer_metric)
        logger.info("eval epoch=%s CER=%.4f", epoch, cer_epoch)
        writer.add_scalar("eval/cer_epoch", cer_epoch, epoch)

    # final save
    save_ckpt(model, processor, os.path.join(run_dir, "final"), custom_lang_info)
    logger.info("done saved -> %s", os.path.join(run_dir, "final"))
    writer.close()


if __name__ == "__main__":
    main()
