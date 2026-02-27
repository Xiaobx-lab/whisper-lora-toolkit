"""
Evaluate ASR transcription quality by scoring against reference subtitles.

Since the ASR model now outputs Mandarin directly, we skip the translator agent
and let the LLM scorer compare transcription vs. reference on a 1-10 scale.

Pipeline:
1. Randomly sample 200 entries (fixed seed) from our_data/valid.jsonl
2. Transcribe audio with the trained Whisper model
3. Use Llama-70B scorer to rate semantic similarity (1-10)
4. Report average score and pass rate (score >= 6)
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from configs import config
from util import load_audio_16k_mono, set_seed, normalize_text
from whisper_peft import PeftModelForWhisper


# LLM Configuration
LLM_MODEL_PATH = "meta-llama/Llama-3.3-70B-Instruct"
LLM_URL = "http://hk-daniel2:8007/v1/chat/completions"

# Defaults
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_SEED = 42
PASS_THRESHOLD = 6


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def load_whisper_model(adapter_path: str | None = None):
    """Load Whisper model with optional LoRA adapter."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if config.USE_BF16 and torch.cuda.is_available() else None

    processor = WhisperProcessor.from_pretrained(config.MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch_dtype,
    )

    if adapter_path:
        model = PeftModelForWhisper.from_pretrained(model, adapter_path)

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
    return model, processor, device


@torch.no_grad()
def transcribe_audio(
    audio_path: str,
    model,
    processor,
    device: str,
) -> str:
    """Transcribe a single audio file."""
    wav = load_audio_16k_mono(audio_path)
    inputs = processor.feature_extractor(
        wav.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device, dtype=model.dtype)

    forced_decoder_ids = None
    if config.FORCE_DECODER_PROMPT:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=config.FORCE_LANGUAGE,
            task=config.FORCE_TASK,
        )

    gen_ids = model.generate(
        input_features=input_features,
        forced_decoder_ids=forced_decoder_ids,
        max_length=config.MAX_LABEL_LEN,
        num_beams=1,
    )
    text = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
    return normalize_text(text)


def call_llm(
    messages: list[dict],
    llm_url: str = LLM_URL,
    model_path: str = LLM_MODEL_PATH,
    temperature: float = 0.1,
    max_tokens: int = 800,
    timeout: int = 60,
) -> tuple[str, bool]:
    """Call LLM API and return (content, success)."""
    payload = {
        "model": model_path,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        response = requests.post(llm_url, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return content, True
    except requests.exceptions.RequestException as e:
        return f"LLM request failed: {str(e)}", False


def parse_json_response(content: str) -> dict | None:
    """Parse JSON from LLM response, handling markdown code blocks."""
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
        return json.loads(content.strip())
    except json.JSONDecodeError:
        return None


def score_transcription(
    transcription: str,
    reference: str,
    llm_url: str = LLM_URL,
    model_path: str = LLM_MODEL_PATH,
) -> dict:
    """
    Directly score how well the ASR transcription matches the reference subtitle.

    Both texts are expected to be in Mandarin Chinese.
    Returns dict with keys: score (1-10), reasoning, is_pass (bool).
    """
    system_prompt = (
        "You are an expert evaluator for Mandarin Chinese speech recognition systems.\n"
        "\n"
        "Your task: compare two Mandarin Chinese texts and rate their semantic similarity.\n"
        "  - transcription: the ASR model's output\n"
        "  - reference: the ground-truth subtitle\n"
        "\n"
        "Scoring rubric (1-10):\n"
        "  10 - Perfect match: identical meaning, trivial wording differences at most\n"
        "   9 - Excellent: same meaning, slight expression variations\n"
        "   8 - Very good: core meaning matches, minor detail differences\n"
        "   7 - Good: main idea preserved, some information lost or added\n"
        "   6 - Above average: mostly consistent, noticeable differences\n"
        "   5 - Average: roughly half the meaning is preserved\n"
        "   4 - Below average: some overlap but significant divergence\n"
        "   3 - Poor: loosely related, major meaning gaps\n"
        "   2 - Very poor: barely any semantic overlap\n"
        "   1 - No match: completely unrelated content\n"
        "\n"
        "Important notes:\n"
        "- Focus on SEMANTIC equivalence, not surface-form match.\n"
        "- Synonyms, colloquial vs. formal, word-order changes are acceptable.\n"
        "- Homophone substitutions that preserve meaning should not be penalized heavily.\n"
        "- Missing or hallucinated segments should reduce the score proportionally.\n"
        "- PAY SPECIAL ATTENTION to pronoun / referent mismatches (e.g. "
        "'wo/我' vs 'ni/你', '我们' vs '你们'). "
        "- 他' vs '她' could be considered as not mismatch because their pronunciation is the same"
        "A pronoun swap changes who is performing or receiving the action, "
        "which alters the core meaning of the sentence. "
        "Apply a light but clear penalty (typically -1 points) for each "
        "pronoun or referent inconsistency, proportional to how much it distorts "
        "the overall semantics. Mention every such mismatch explicitly in your reasoning.\n"
        "\n"
        "Respond with a single JSON object ONLY (no markdown fences):\n"
        '{"score": <int 1-10>, "reasoning": "<your analysis in Chinese>"}'
    )

    user_prompt = (
        f"transcription: {transcription}\n"
        f"reference: {reference}\n"
        "\n"
        "Please evaluate and return JSON."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content, success = call_llm(messages, llm_url, model_path)

    if not success:
        return {
            "score": -1,
            "reasoning": f"LLM call failed: {content}",
            "is_pass": None,
            "error": True,
        }

    result = parse_json_response(content)
    if result is None:
        return {
            "score": -1,
            "reasoning": f"Failed to parse JSON: {content}",
            "is_pass": None,
            "error": True,
        }

    score = result.get("score", -1)
    if isinstance(score, (int, float)) and 1 <= score <= 10:
        result["score"] = int(score)
        result["is_pass"] = score >= PASS_THRESHOLD
    else:
        result["score"] = -1
        result["is_pass"] = None

    return result


def load_data(jsonl_path: str) -> list[dict]:
    """Load data from JSONL file."""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def random_sample_data(
    data: list[dict],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Randomly sample data with fixed seed for reproducibility."""
    random.seed(seed)
    if len(data) <= sample_size:
        return data
    indices = list(range(len(data)))
    random.shuffle(indices)
    sampled_indices = sorted(indices[:sample_size])
    return [data[i] for i in sampled_indices]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ASR transcription quality via LLM scoring (1-10)"
    )
    parser.add_argument(
        "--input", type=str, default="our_data/valid.jsonl",
        help="Path to input JSONL (fields: audio, text)",
    )
    parser.add_argument(
        "--output", type=str, default="benchmark/mandarin_score_evaluation1.jsonl",
        help="Path to output JSONL with per-sample results",
    )
    parser.add_argument(
        "--adapter", type=str,
        default="whisper-turbo-minnan-zh-lora/latest/best",
        help="Path to LoRA adapter (omit for base model)",
    )
    parser.add_argument(
        "--llm-url", type=str, default=LLM_URL,
        help="URL of the vLLM endpoint",
    )
    parser.add_argument(
        "--llm-model", type=str, default=LLM_MODEL_PATH,
        help="Model path for the LLM scorer",
    )
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help="Number of samples to randomly select",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--skip-transcribe", action="store_true",
        help="Skip transcription; reuse 'transcription' field from input",
    )
    args = parser.parse_args()

    logger = setup_logger()
    set_seed(args.seed)

    # ---- Load & sample data ----
    logger.info(f"Loading data from {args.input}")
    data = load_data(args.input)
    logger.info(f"Total samples in file: {len(data)}")

    logger.info(
        f"Randomly sampling {args.sample_size} samples (seed={args.seed})"
    )
    data = random_sample_data(data, args.sample_size, args.seed)
    logger.info(f"Sampled {len(data)} samples for evaluation")

    # ---- Load Whisper model ----
    if not args.skip_transcribe:
        logger.info(f"Loading Whisper model with adapter: {args.adapter}")
        model, processor, device = load_whisper_model(args.adapter)
        logger.info("Whisper model loaded")

    # ---- Evaluate ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    score_sum = 0
    scored_count = 0
    pass_count = 0
    error_count = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for item in tqdm(data, desc="Scoring"):
            audio_path = item["audio"]
            reference = item["text"]

            # Step 1: Transcribe
            if args.skip_transcribe and "transcription" in item:
                transcription = item["transcription"]
            else:
                try:
                    transcription = transcribe_audio(
                        audio_path, model, processor, device
                    )
                except Exception as e:
                    logger.warning(f"Transcription failed for {audio_path}: {e}")
                    transcription = ""

            # Step 2: Score directly (no translator needed)
            if not transcription.strip():
                scorer_result = {
                    "score": 0,
                    "reasoning": "ASR returned empty transcription, scored 0.",
                    "is_pass": False,
                }
            else:
                scorer_result = score_transcription(
                    transcription=transcription,
                    reference=reference,
                    llm_url=args.llm_url,
                    model_path=args.llm_model,
                )

            score = scorer_result.get("score", -1)

            result = {
                "audio": audio_path,
                "transcription": transcription,
                "reference": reference,
                "score": score,
                "reasoning": scorer_result.get("reasoning", ""),
                "is_pass": scorer_result.get("is_pass"),
            }

            if scorer_result.get("error"):
                result["error"] = True
                error_count += 1
            elif score >= 0:
                scored_count += 1
                score_sum += score
                if scorer_result.get("is_pass"):
                    pass_count += 1

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("Evaluation Complete!")
    logger.info(f"Total samples with valid scores: {scored_count}")
    logger.info(f"LLM errors (score = -1):          {error_count}")

    if scored_count > 0:
        avg_score = score_sum / scored_count
        pass_rate = pass_count / scored_count * 100
        logger.info(f"Average score (1-10):             {avg_score:.2f}")
        logger.info(
            f"Pass rate (score >= {PASS_THRESHOLD}):           "
            f"{pass_count}/{scored_count} = {pass_rate:.1f}%"
        )
    else:
        logger.warning("No valid scores collected -- nothing to summarize.")

    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
