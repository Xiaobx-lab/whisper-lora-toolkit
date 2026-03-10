# train_stage2.py
"""
Stage 2: Language token embedding calibration.

Loads a trained stage-1 LoRA adapter (with custom language tokens),
freezes everything (encoder, decoder, all LoRA weights), and only
trains the ~100 language-token embedding rows on a balanced dataset.

Fixes language detection bias caused by imbalanced stage-1 data
without affecting transcription quality.
"""

import os
import math
import json
import random
import shutil
import logging
import argparse
import importlib.util
from collections import defaultdict
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
from peft import PeftConfig, LoraConfig
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file as safe_load_file

from util import set_seed, normalize_text
from asrDataLoader import JsonlASRDataset, CollatorWhisper
from whisper_peft import PeftModelForWhisper
from custom_lang import (
    load_custom_lang_for_inference,
    save_custom_embeddings,
)

_EMBEDDING_PARAM_KEYWORDS = ("embed_tokens.weight", "proj_out.weight", "lm_head.weight")


@torch.no_grad()
def evaluate_cer(model, processor, dataloader, device, max_gen_len, cer_metric):
    model.eval()
    force_task = getattr(config, "FORCE_TASK", "transcribe")

    preds, refs = [], []
    for batch in dataloader:
        input_features = batch["input_features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        gen_kwargs = {"max_length": max_gen_len, "num_beams": 1, "task": force_task}
        if getattr(config, "FORCE_DECODER_PROMPT", False):
            gen_kwargs["language"] = config.FORCE_LANGUAGE

        gen_ids = model.generate(input_features=input_features, **gen_kwargs)

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


@torch.no_grad()
def evaluate_lang_detection(
    model, dataset, device, max_samples=500, batch_size=16, use_bf16=False,
):
    """
    Evaluate language detection accuracy with confusion matrix.

    Feeds [SOT] to the decoder and reads the language logit at position 0,
    matching both training (with strip_bos) and generate() behavior.
    """
    model.eval()

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    gen_cfg = base.generation_config
    lang_to_id = getattr(gen_cfg, "lang_to_id", {})
    id_to_lang = {}
    for k, v in lang_to_id.items():
        name = k[2:-2] if k.startswith("<|") and k.endswith("|>") else k
        id_to_lang[v] = name
    lang_id_list = sorted(set(lang_to_id.values()))

    if isinstance(dataset, Subset):
        base_ds = dataset.dataset
        all_indices = list(dataset.indices)
    else:
        base_ds = dataset
        all_indices = list(range(len(dataset)))

    labeled = [
        (i, base_ds.languages[i])
        for i in all_indices
        if base_ds.languages[i] is not None
    ]
    if len(labeled) > max_samples:
        rng = random.Random(42)
        labeled = rng.sample(labeled, max_samples)

    confusion: dict = defaultdict(lambda: defaultdict(int))
    correct = 0
    total = 0
    autocast_dtype = torch.bfloat16 if use_bf16 else None

    for batch_start in range(0, len(labeled), batch_size):
        batch_items = labeled[batch_start : batch_start + batch_size]
        features, true_langs = [], []
        for idx, lang in batch_items:
            sample = base_ds[idx]
            features.append(sample["input_features"])
            true_langs.append(lang)

        input_features = torch.stack(features).to(device)
        bs = input_features.shape[0]

        decoder_input_ids = torch.full(
            (bs, 1), gen_cfg.decoder_start_token_id,
            device=device, dtype=torch.long,
        )

        if autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                encoder_outputs = base.get_encoder()(input_features)
                logits = base(
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=decoder_input_ids,
                    use_cache=False,
                ).logits[:, -1]
        else:
            encoder_outputs = base.get_encoder()(input_features)
            logits = base(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
            ).logits[:, -1]

        mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=device)
        mask[lang_id_list] = False
        logits[:, mask] = float("-inf")

        pred_ids = logits.argmax(dim=-1)
        for pred_id, true_lang in zip(pred_ids.tolist(), true_langs):
            pred_lang = id_to_lang.get(pred_id, f"id_{pred_id}")
            confusion[true_lang][pred_lang] += 1
            if pred_lang == true_lang:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0

    per_lang_acc = {}
    for tl in sorted(confusion):
        lang_total = sum(confusion[tl].values())
        lang_correct = confusion[tl].get(tl, 0)
        per_lang_acc[tl] = lang_correct / lang_total if lang_total else 0.0

    all_langs = sorted(
        set(list(confusion.keys()))
        | {pl for d in confusion.values() for pl in d}
    )
    header_label = "true\\pred"
    lines = [
        "  " + f"{header_label:<10}" + "".join(f"{l:>8}" for l in all_langs),
    ]
    for tl in sorted(confusion.keys()):
        row = "  " + f"{tl:<10}" + "".join(
            f"{confusion[tl].get(pl, 0):>8}" for pl in all_langs
        )
        lines.append(row)
    lines.append("")
    for tl in sorted(per_lang_acc):
        lines.append(f"  {tl}: {per_lang_acc[tl]:.1%}")
    lines.append(f"  Overall: {accuracy:.1%} ({correct}/{total})")
    report = "\n".join(lines)

    model.train()
    return dict(confusion), accuracy, per_lang_acc, report


def save_ckpt(model, processor, out_dir, custom_lang_info):
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
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


def _setup_lang_embedding_hooks(model, orig_vocab_size, lang_token_ids):
    """
    Unfreeze embedding parameters and register gradient hooks so that
    only language-token rows receive updates.  All other rows (text tokens,
    special tokens) are zeroed in the backward pass.
    """
    keep_ids = {tid for tid in lang_token_ids if 0 <= tid < orig_vocab_size}
    hooks = []
    seen_ptrs: set = set()

    logger = logging.getLogger(__name__)

    for name, param in model.named_parameters():
        if not any(kw in name for kw in _EMBEDDING_PARAM_KEYWORDS):
            continue
        ptr = param.data_ptr()
        if ptr in seen_ptrs:
            logger.info("Skipping %s (tied weight, already hooked)", name)
            continue
        seen_ptrs.add(ptr)

        param.requires_grad = True

        def _make_hook(pname, ovs, keep):
            freeze_mask = torch.ones(ovs, dtype=torch.bool)
            for tid in keep:
                freeze_mask[tid] = False
            mask_cache: dict = {}

            def hook(grad):
                if grad.stride(0) == 0:
                    grad = grad.clone()
                dev = grad.device
                if dev not in mask_cache:
                    mask_cache[dev] = freeze_mask.to(dev)
                grad[:ovs].masked_fill_(mask_cache[dev].unsqueeze(1), 0)
                return grad
            return hook

        h = param.register_hook(_make_hook(name, orig_vocab_size, keep_ids))
        hooks.append(h)

        n_trainable = len(keep_ids)
        n_custom = max(0, param.shape[0] - orig_vocab_size)
        logger.info(
            "Grad hook: %s | freeze %d/%d orig rows, "
            "keep %d lang rows + %d custom rows trainable",
            name, orig_vocab_size - n_trainable, orig_vocab_size,
            n_trainable, n_custom,
        )

    if not hooks:
        logger.warning("No embedding params found for gradient hooks")
    return hooks


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: calibrate language-token embeddings on balanced data.",
    )
    parser.add_argument(
        "--config",
        default="configs/config_stage2.py",
        help="Path to stage-2 config file.",
    )
    args = parser.parse_args()

    global config
    config = _load_config(args.config)

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join(
        "logs", f"train_stage2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
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
    logger.info(
        "Stage 2 | device=%s  run_dir=%s  log=%s",
        device, run_dir, log_path,
    )

    adapter_path = os.path.abspath(config.STAGE1_ADAPTER_PATH)
    logger.info("Stage-1 adapter: %s", adapter_path)

    if not os.path.isfile(os.path.join(adapter_path, "adapter_config.json")):
        raise SystemExit(
            f"No adapter_config.json in {adapter_path} -- "
            "STAGE1_ADAPTER_PATH must point to a saved LoRA checkpoint."
        )

    # ---- 1) Load base model ----
    peft_cfg = PeftConfig.from_pretrained(adapter_path)
    base_model_path = (
        getattr(config, "MODEL_NAME", None) or peft_cfg.base_model_name_or_path
    )
    logger.info("Base model: %s", base_model_path)

    processor = WhisperProcessor.from_pretrained(base_model_path)
    model = WhisperForConditionalGeneration.from_pretrained(base_model_path)
    model.config.use_cache = False
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    if model.generation_config.suppress_tokens:
        model.generation_config.suppress_tokens = []

    # ---- 2) Restore custom language tokens (must be before PEFT wrapping) ----
    emb_path = os.path.join(adapter_path, "custom_embeddings.pt")
    if not os.path.isfile(emb_path):
        raise SystemExit(
            f"No custom_embeddings.pt in {adapter_path} -- "
            "stage 2 requires a stage-1 checkpoint with custom language tokens."
        )

    custom_langs = load_custom_lang_for_inference(model, processor, adapter_path)
    logger.info("Restored custom language tokens: %s", custom_langs)

    emb_state = torch.load(emb_path, map_location="cpu", weights_only=True)
    orig_vocab_size = emb_state["orig_vocab_size"]

    # ---- 3) Create LoRA wrapper (same approach as stage 1) ----
    lora_cfg = LoraConfig.from_pretrained(adapter_path)
    model = PeftModelForWhisper(model, lora_cfg)

    # ---- 3b) Load trained adapter weights from stage 1 ----
    adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
    if os.path.isfile(adapter_file):
        adapter_weights = safe_load_file(adapter_file)
    else:
        adapter_file = os.path.join(adapter_path, "adapter_model.bin")
        adapter_weights = torch.load(adapter_file, map_location="cpu", weights_only=True)
    set_peft_model_state_dict(model, adapter_weights)
    logger.info("Loaded LoRA adapter weights from %s", adapter_path)

    # ---- 4) Freeze ALL parameters ----
    for param in model.parameters():
        param.requires_grad = False
    logger.info(
        "Froze all %d parameter tensors",
        sum(1 for _ in model.parameters()),
    )

    # ---- 5) Collect language token IDs ----
    base_model = (
        model.get_base_model() if hasattr(model, "get_base_model") else model
    )
    gen_cfg = getattr(base_model, "generation_config", None)
    lang_token_ids: set = set()
    if gen_cfg and hasattr(gen_cfg, "lang_to_id"):
        lang_token_ids = set(gen_cfg.lang_to_id.values())
    logger.info(
        "Language tokens to train: %d IDs (built-in + custom)",
        len(lang_token_ids),
    )

    model.to(device)

    # ---- 6) Unfreeze language-token embedding rows via gradient hooks ----
    embed_hooks = _setup_lang_embedding_hooks(model, orig_vocab_size, lang_token_ids)

    custom_lang_info = {
        "orig_vocab_size": orig_vocab_size,
        "lang_token_ids": lang_token_ids,
    }

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable: %d / %d params (%.4f%%)",
        trainable_count, total_count, trainable_count / total_count * 100,
    )

    # ---- 7) Gradient checkpointing (optional, saves VRAM) ----
    if getattr(config, "USE_GRAD_CHECKPOINT", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # ---- 8) Data ----
    train_ds = JsonlASRDataset(config.TRAIN_JSONL, processor.feature_extractor)
    valid_ds_full = JsonlASRDataset(config.VALID_JSONL, processor.feature_extractor)

    eval_samples = getattr(config, "EVAL_SAMPLES", 0)
    if eval_samples > 0 and eval_samples < len(valid_ds_full):
        random.seed(config.SEED)
        indices = random.sample(range(len(valid_ds_full)), eval_samples)
        valid_ds = Subset(valid_ds_full, indices)
        logger.info("Subsampled validation: %d -> %d", len(valid_ds_full), eval_samples)
    else:
        valid_ds = valid_ds_full

    force_task = getattr(config, "FORCE_TASK", "transcribe")
    lang_mapping = getattr(config, "LANGUAGE_MAPPING", None)
    if getattr(config, "FORCE_DECODER_PROMPT", False):
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
        train_ds, batch_size=config.TRAIN_BS, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=collator,
        pin_memory=True, persistent_workers=persistent,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=config.EVAL_BS, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collator,
        pin_memory=True, persistent_workers=persistent,
    )

    # ---- 9) Optimizer (only embedding params with requires_grad) ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.LR)

    steps_per_epoch = math.ceil(len(train_loader) / config.GRAD_ACCUM)
    total_steps = steps_per_epoch * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    use_bf16 = getattr(config, "USE_BF16", False) and device == "cuda"
    autocast_dtype = torch.bfloat16 if use_bf16 else None

    cer_metric = evaluate.load("cer")

    lang_eval_samples = getattr(config, "LANG_EVAL_SAMPLES", 500)
    logger.info(
        "Training plan: %d samples, bs=%d, accum=%d, "
        "%d steps/epoch, %d epochs, %d total steps, "
        "lang_eval_samples=%d",
        len(train_ds), config.TRAIN_BS, config.GRAD_ACCUM,
        steps_per_epoch, config.EPOCHS, total_steps,
        lang_eval_samples,
    )

    # ---- 10) Training loop ----
    global_step = 0
    best_accuracy = None
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

            if autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
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

            if i % config.GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, config.CLIP_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0:
                    avg_loss = running / 10.0
                    avg_raw_loss = running_raw / 10.0
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        "epoch=%s step=%s avg_loss=%.4f avg_raw_loss=%.4f lr=%.2e",
                        epoch, global_step, avg_loss, avg_raw_loss, lr,
                    )
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/raw_loss", avg_raw_loss, global_step)
                    writer.add_scalar("train/lr", lr, global_step)
                    running = 0.0
                    running_raw = 0.0

                if config.EVAL_STEPS > 0 and global_step % config.EVAL_STEPS == 0:
                    _, acc, per_lang_acc, report = evaluate_lang_detection(
                        model, train_ds, device,
                        max_samples=lang_eval_samples,
                        batch_size=config.EVAL_BS,
                        use_bf16=use_bf16,
                    )
                    logger.info(
                        "eval step=%s lang_acc=%.4f\n%s",
                        global_step, acc, report,
                    )
                    writer.add_scalar("eval/lang_accuracy", acc, global_step)
                    for lang_key, lang_acc in per_lang_acc.items():
                        writer.add_scalar(
                            f"eval/lang_acc_{lang_key}", lang_acc, global_step,
                        )

                    cer_val = evaluate_cer(
                        model, processor, valid_loader, device,
                        config.MAX_LABEL_LEN, cer_metric,
                    )
                    logger.info("eval step=%s CER=%.4f", global_step, cer_val)
                    writer.add_scalar("eval/cer", cer_val, global_step)

                    if best_accuracy is None or acc > best_accuracy:
                        best_accuracy = acc
                        save_ckpt(
                            model, processor,
                            os.path.join(run_dir, "best"),
                            custom_lang_info,
                        )
                        logger.info(
                            "save best (acc=%.4f, CER=%.4f) -> %s",
                            acc, cer_val, os.path.join(run_dir, "best"),
                        )

                if config.SAVE_STEPS > 0 and global_step % config.SAVE_STEPS == 0:
                    out = os.path.join(run_dir, f"step-{global_step}")
                    save_ckpt(model, processor, out, custom_lang_info)
                    logger.info("save -> %s", out)

        _, acc_epoch, _, report_epoch = evaluate_lang_detection(
            model, train_ds, device,
            max_samples=lang_eval_samples,
            batch_size=config.EVAL_BS,
            use_bf16=use_bf16,
        )
        cer_epoch = evaluate_cer(
            model, processor, valid_loader, device,
            config.MAX_LABEL_LEN, cer_metric,
        )
        logger.info(
            "eval epoch=%s lang_acc=%.4f CER=%.4f\n%s",
            epoch, acc_epoch, cer_epoch, report_epoch,
        )
        writer.add_scalar("eval/lang_accuracy_epoch", acc_epoch, epoch)
        writer.add_scalar("eval/cer_epoch", cer_epoch, epoch)

    save_ckpt(
        model, processor, os.path.join(run_dir, "final"), custom_lang_info,
    )
    logger.info("done -> %s", os.path.join(run_dir, "final"))
    writer.close()


if __name__ == "__main__":
    main()
