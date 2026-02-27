"""
transcribe_audio.py
用 Whisper (+ 可选 LoRA adapter) 对单条音频做推理转录，同时输出检测到的语言。

用法:
    python transcribe_audio.py --audio <音频路径> [--config <配置模块名>] [--adapter <adapter路径>] [--force-prompt]

示例:
    # 自动检测语言（默认不强制 decoder prompt）
    python transcribe_audio.py --audio test.wav

    # 强制 decoder prompt（与训练时一致）
    python transcribe_audio.py --audio test.wav --force-prompt --adapter path/to/best
"""

import argparse
import importlib
import logging
import re
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from util import load_audio_16k_mono, set_seed, normalize_text
from whisper_peft import PeftModelForWhisper


def _setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def load_config(config_name: str = "configs.config"):
    """根据模块名动态加载配置文件（支持 'configs/config_xxx' 或 'configs.config_xxx' 格式）。"""
    if config_name.endswith(".py"):
        config_name = config_name[:-3]
    config_name = config_name.replace("/", ".").replace("\\", ".")
    return importlib.import_module(config_name)


@dataclass
class TranscribeResult:
    """转录结果：文本 + 检测到的语言 + 语言概率分布。"""
    text: str
    language: str | None = None          # 最可能的语言代码，如 "zh", "en", "ja"
    language_prob: float | None = None   # 该语言的概率
    language_top5: dict = field(default_factory=dict)  # top-5 语言概率 {lang: prob}


def _detect_language(model, input_features, processor) -> tuple[str | None, float | None, dict]:
    """在 encoder 侧做语言检测，返回 (语言代码, 概率, top-5 概率字典)。

    原理：将 <|startoftranscript|> 喂给 decoder，取第一步 logits 在语言 token 上的分布。
    这与 model.detect_language() 的内部逻辑一致，但额外保留了 softmax 概率。
    """
    gen_cfg = model.generation_config
    lang_to_id = getattr(gen_cfg, "lang_to_id", None)
    if not lang_to_id:
        return None, None, {}

    id_to_lang = {v: k for k, v in lang_to_id.items()}
    device = input_features.device

    # decoder 输入：仅 <|startoftranscript|>
    decoder_input_ids = torch.ones(
        (1, 1), device=device, dtype=torch.long,
    ) * gen_cfg.decoder_start_token_id

    # 前向拿 logits；通过 get_base_model() 兼容 PeftModel
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    encoder_outputs = base.get_encoder()(input_features[:, :, :3000])
    logits = base(
        encoder_outputs=encoder_outputs,
        decoder_input_ids=decoder_input_ids,
        use_cache=False,
    ).logits[:, -1]

    # 屏蔽非语言 token，只保留语言 token 的 logits
    non_lang_mask = torch.ones_like(logits[0], dtype=torch.bool)  # 先假设所有token都是非语言token mask=True
    non_lang_mask[list(lang_to_id.values())] = False  # 然后只保留语言token mask=False
    logits[:, non_lang_mask] = -np.inf

    # softmax 得到概率分布
    probs = torch.softmax(logits.float(), dim=-1)
    top_probs, top_ids = probs[0].topk(5)

    top5 = {}
    for p, tid in zip(top_probs, top_ids):
        tag = id_to_lang.get(tid.item(), "")
        lang_code = tag.strip("<|>")
        if lang_code:
            top5[lang_code] = round(p.item(), 4)

    # 最高概率的语言
    best_id = probs[0].argmax().item()
    best_tag = id_to_lang.get(best_id, "")
    best_lang = best_tag.strip("<|>") or None
    best_prob = round(probs[0, best_id].item(), 4) if best_lang else None

    return best_lang, best_prob, top5


@torch.no_grad()
def transcribe_audio(audio_path, adapter_path=None, cfg=None,
                     force_decoder_prompt=False) -> TranscribeResult:
    """对单条音频做推理，返回转录文本和检测到的语言。

    Args:
        audio_path:           音频文件路径
        adapter_path:         LoRA adapter 路径，为 None 则使用原始模型
        cfg:                  配置模块（通过 load_config() 加载），为 None 则默认用 configs/config.py
        force_decoder_prompt: 是否强制 decoder prompt（指定语言+任务），默认 False（自动检测）
    """
    if cfg is None:
        cfg = load_config("configs.config")

    set_seed(cfg.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型 & processor
    torch_dtype = torch.bfloat16 if cfg.USE_BF16 and torch.cuda.is_available() else None

    processor = WhisperProcessor.from_pretrained(cfg.MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(cfg.MODEL_NAME, dtype=torch_dtype)

    if adapter_path:
        model = PeftModelForWhisper.from_pretrained(model, adapter_path)

    # 设置 decoder 前缀（强制语言 & 任务）
    if force_decoder_prompt:
        # 强制指定语言和任务，保证推理与训练时行为一致
        processor.tokenizer.set_prefix_tokens(language=cfg.FORCE_LANGUAGE, task=cfg.FORCE_TASK)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=cfg.FORCE_LANGUAGE, task=cfg.FORCE_TASK)
        model.config.forced_decoder_ids = forced_decoder_ids
        model.generation_config.forced_decoder_ids = forced_decoder_ids
    else:
        # 不强制——让模型自行检测语言
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None

    # 清空 suppress_tokens，避免某些 token 被意外屏蔽
    model.config.suppress_tokens = []
    if model.generation_config.suppress_tokens:
        model.generation_config.suppress_tokens = []

    model.to(device)

    # 音频预处理
    wav = load_audio_16k_mono(audio_path)
    inputs = processor.feature_extractor(wav.numpy(), sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device, dtype=model.dtype)

    # 语言检测（encoder 侧，带概率）
    detected_lang, lang_prob, lang_top5 = _detect_language(model, input_features, processor)

    # 生成转录
    forced_decoder_ids = None
    if force_decoder_prompt:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=cfg.FORCE_LANGUAGE, task=cfg.FORCE_TASK)

    gen_ids = model.generate(
        input_features=input_features,
        forced_decoder_ids=forced_decoder_ids,
        max_length=cfg.MAX_LABEL_LEN,
        num_beams=1,
    )

    text = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return TranscribeResult(
        text=normalize_text(text),
        language=detected_lang,
        language_prob=lang_prob,
        language_top5=lang_top5,
    )


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Whisper 单条音频推理（支持 LoRA adapter + 语言检测）")
    parser.add_argument("--config", type=str, default="configs.config_our_data_encoder_unfreeze",
                        help="配置模块名，如 'configs.config_our_data_encoder_unfreeze' 或 'configs/config_our_data_encoder_unfreeze'")
    parser.add_argument("--audio", type=str, required=True, help="待转录的音频文件路径")
    parser.add_argument("--adapter", type=str, default="/home/jovyan/work/sylvia/myWhisperTrainer/whisper-turbo-minnan-zh-lora-encoder-unfreeze/minnan_zh_encoder_unfreeze_20260204_094713/best", help="LoRA adapter 路径（可选）")
    parser.add_argument("--force-prompt", action="store_true", default=False,
                        help="强制 decoder prompt（指定语言+任务），默认关闭（自动检测）")
    return parser.parse_args()


def main():
    logger = _setup_logger()
    args = parse_args()

    cfg = load_config(args.config)
    logger.info("Loaded config: %s", args.config)
    logger.info("Model: %s", cfg.MODEL_NAME)
    logger.info("Force decoder prompt: %s (lang=%s, task=%s)",
                args.force_prompt,
                getattr(cfg, "FORCE_LANGUAGE", "N/A"),
                getattr(cfg, "FORCE_TASK", "N/A"))

    result = transcribe_audio(args.audio, adapter_path=args.adapter, cfg=cfg,
                              force_decoder_prompt=args.force_prompt)

    lang_str = result.language or "unknown"
    prob_str = f"{result.language_prob:.2%}" if result.language_prob is not None else "N/A"
    logger.info("Detected language: %s (%s)", lang_str, prob_str)
    if result.language_top5:
        top5_str = ", ".join(f"{k}={v:.2%}" for k, v in result.language_top5.items())
        logger.info("Language top-5: %s", top5_str)
    logger.info("Transcription: %s", result.text)
    print(result.text)


if __name__ == "__main__":
    main()
