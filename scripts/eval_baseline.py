import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import evaluate
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from configs import config
from util import set_seed, normalize_text
from asrDataLoader import JsonlASRDataset, CollatorWhisper


@torch.no_grad()
def evaluate_cer(model, processor, dataloader, device, max_gen_len: int) -> float:
    cer = evaluate.load("cer")
    model.eval()

    forced_decoder_ids = None
    if config.FORCE_DECODER_PROMPT:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=config.FORCE_LANGUAGE,
            task=config.FORCE_TASK,
        )

    preds, refs = [], []
    for batch in dataloader:
        input_features = batch["input_features"].to(
            device, non_blocking=True, dtype=model.dtype
        )
        labels = batch["labels"].to(device, non_blocking=True)

        gen_ids = model.generate(
            input_features=input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_length=max_gen_len,
            num_beams=1,
        )

        labels_dec = labels.clone()
        labels_dec[labels_dec == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        ref_str = processor.tokenizer.batch_decode(labels_dec, skip_special_tokens=True)

        pred_str = [normalize_text(s) for s in pred_str]
        ref_str = [normalize_text(s) for s in ref_str]

        preds.extend(pred_str)
        refs.extend(ref_str)

    return float(cer.compute(predictions=preds, references=refs))


def _setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def main():
    logger = _setup_logger()
    set_seed(config.SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s", device)

    torch_dtype = None
    if config.USE_BF16 and torch.cuda.is_available():
        torch_dtype = torch.bfloat16

    processor = WhisperProcessor.from_pretrained(config.MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(
        config.MODEL_NAME,
        dtype=torch_dtype,
    )

    if config.FORCE_DECODER_PROMPT:
        processor.tokenizer.set_prefix_tokens(
            language=config.FORCE_LANGUAGE,
            task=config.FORCE_TASK,
        )
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=config.FORCE_LANGUAGE,
            task=config.FORCE_TASK,
        )
        model.config.forced_decoder_ids = forced_decoder_ids
        model.generation_config.forced_decoder_ids = forced_decoder_ids
    else:
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None

    model.config.suppress_tokens = []
    if model.generation_config.suppress_tokens:
        model.generation_config.suppress_tokens = []

    model.to(device)

    valid_ds = JsonlASRDataset(config.VALID_JSONL)
    collator = CollatorWhisper(processor=processor, max_label_len=config.MAX_LABEL_LEN)
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.EVAL_BS,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )

    cer_val = evaluate_cer(model, processor, valid_loader, device, config.MAX_LABEL_LEN)
    logger.info("baseline whisper-large-v3-turbo CER=%.4f", cer_val)


if __name__ == "__main__":
    main()

