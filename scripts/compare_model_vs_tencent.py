"""
Evaluate custom Whisper model transcription quality using LLaMA scoring,
with Tencent ASR results placed alongside for manual comparison.

Workflow:
1. Load model and Tencent result JSONL files
2. Match results using reference text as key
3. Randomly sample matched pairs
4. Use LLaMA-70B to score the model's transcription against reference
5. Output JSON with model scores, bad cases, and Tencent results for reference
"""

import argparse
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

LLM_MODEL_PATH = "meta-llama/Llama-3.3-70B-Instruct"
LLM_URL = "http://hk-daniel2:8007/v1/chat/completions"

DEFAULT_SAMPLE_SIZE = 200
DEFAULT_SEED = 42
DEFAULT_BAD_THRESHOLD = 6
DEFAULT_MAX_WORKERS = 16


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def normalize_ref(text: str) -> str:
    import unicodedata
    text = text.strip()
    text = unicodedata.normalize("NFKC", text)
    return text


def build_reference_lookup(data: list[dict]) -> dict[str, dict]:
    lookup = {}
    for entry in data:
        ref = normalize_ref(entry.get("reference", ""))
        if ref and ref not in lookup:
            lookup[ref] = entry
    return lookup


def call_llm(
    messages: list[dict],
    llm_url: str = LLM_URL,
    model_path: str = LLM_MODEL_PATH,
    temperature: float = 0.1,
    max_tokens: int = 512,
    timeout: int = 90,
) -> tuple[str, bool]:
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
        return result["choices"][0]["message"]["content"], True
    except requests.exceptions.RequestException as e:
        return f"LLM request failed: {e}", False


def parse_json_response(content: str) -> dict | None:
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


LANG_HINTS = {
    "nan": "闽南语/台语 (Taiwanese Hokkien/Minnan) - ASR output may use dialect characters or Mandarin approximations",
    "yue": "粤语/广东话 (Cantonese) - ASR output may use Cantonese-specific characters or Mandarin approximations",
    "zh": "普通话/中文 (Mandarin Chinese)",
    "en": "English",
    "ja": "日本語 (Japanese)",
}


def score_transcription(
    hypothesis: str,
    reference: str,
    language: str,
    llm_url: str = LLM_URL,
    model_path: str = LLM_MODEL_PATH,
) -> dict:
    """Score a single ASR hypothesis against its reference using LLM."""
    lang_context = LANG_HINTS.get(language, f"Language code: {language}")

    system_prompt = f"""You are an expert ASR (Automatic Speech Recognition) evaluation specialist.

Task: Score an ASR system's transcription output against the ground-truth reference.

Language context: {lang_context}

Scoring criteria (1-10):
- 10: Perfect - identical or virtually identical to reference (punctuation/formatting differences are fine)
- 9: Excellent - same meaning, tiny wording variations (e.g. synonym substitution)
- 8: Very good - core content correct, minor word-level errors not affecting meaning
- 7: Good - main idea preserved, some information loss or minor additions
- 6: Acceptable - mostly correct with noticeable errors
- 5: Mediocre - roughly half the content accurately captured
- 4: Below average - significant errors but some correct fragments
- 3: Poor - major errors, only loosely related to reference
- 2: Very poor - barely recognizable, most content wrong
- 1: Failed - completely wrong or unrelated

Key guidelines:
- For dialect languages (nan/yue), capturing the MEANING with different characters or Mandarin equivalents is still a good result.
- Focus on SEMANTIC accuracy over exact character matching.
- Punctuation-only differences should not reduce the score.

You MUST respond with ONLY a valid JSON object, no markdown fences:
{{"score": <int 1-10>, "analysis": "<Chinese analysis explaining score>"}}"""

    user_prompt = (
        f"Reference (ground truth): {reference}\n\n"
        f"ASR output: {hypothesis}\n\n"
        f"Please score the ASR output against the reference."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    content, success = call_llm(messages, llm_url, model_path, max_tokens=512)

    if not success:
        return {"score": -1, "analysis": f"LLM call failed: {content}", "error": True}

    result = parse_json_response(content)
    if result is None:
        return {"score": -1, "analysis": f"Failed to parse: {content}", "error": True}

    score = result.get("score", -1)
    if not (isinstance(score, (int, float)) and 1 <= score <= 10):
        return {"score": -1, "analysis": f"Invalid score: {score}", "error": True}

    return {"score": int(score), "analysis": result.get("analysis", "")}


def main():
    parser = argparse.ArgumentParser(
        description="Score model transcriptions with LLM, paired with Tencent results for comparison"
    )
    parser.add_argument(
        "--model-results", type=str, required=True,
        help="Path to model transcription_results.jsonl",
    )
    parser.add_argument(
        "--tencent-results", type=str, required=True,
        help="Path to Tencent ASR transcription_results.jsonl",
    )
    parser.add_argument(
        "--output", type=str, default="scripts/comparison_model_vs_tencent.json",
        help="Output path for comparison JSON",
    )
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of samples to randomly evaluate (default: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--llm-url", type=str, default=LLM_URL,
        help="URL of the vLLM endpoint",
    )
    parser.add_argument(
        "--llm-model", type=str, default=LLM_MODEL_PATH,
        help="Model path for the LLM",
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Filter by language code (nan, yue, zh, en, ja). Default: all",
    )
    parser.add_argument(
        "--bad-threshold", type=int, default=DEFAULT_BAD_THRESHOLD,
        help=f"Model score below this is a bad case (default: {DEFAULT_BAD_THRESHOLD})",
    )
    parser.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        help=f"Max concurrent LLM requests (default: {DEFAULT_MAX_WORKERS})",
    )
    args = parser.parse_args()

    logger = setup_logger()
    random.seed(args.seed)

    logger.info(f"Loading model results: {args.model_results}")
    model_data = load_jsonl(args.model_results)
    logger.info(f"Loaded {len(model_data)} model results")

    logger.info(f"Loading Tencent results: {args.tencent_results}")
    tencent_data = load_jsonl(args.tencent_results)
    logger.info(f"Loaded {len(tencent_data)} Tencent results")

    tencent_lookup = build_reference_lookup(tencent_data)
    logger.info(f"Tencent lookup: {len(tencent_lookup)} unique references")

    candidates = [e for e in model_data if e.get("status") == "success"]

    if args.language:
        candidates = [
            e for e in candidates if e.get("language") == args.language
        ]
        logger.info(f"After language filter '{args.language}': {len(candidates)}")

    matched_pairs = []
    unmatched = 0
    for entry in candidates:
        ref = normalize_ref(entry.get("reference", ""))
        tencent_entry = tencent_lookup.get(ref)
        if tencent_entry:
            matched_pairs.append((entry, tencent_entry))
        else:
            unmatched += 1

    logger.info(f"Matched pairs: {len(matched_pairs)}, unmatched: {unmatched}")

    if len(matched_pairs) > args.sample_size:
        matched_pairs = random.sample(matched_pairs, args.sample_size)
        logger.info(f"Randomly sampled {args.sample_size} pairs (seed={args.seed})")

    def _evaluate(idx_pair):
        idx, (model_entry, tencent_entry) = idx_pair
        result = score_transcription(
            hypothesis=model_entry["hypothesis"],
            reference=model_entry["reference"],
            language=model_entry.get("language", "unknown"),
            llm_url=args.llm_url,
            model_path=args.llm_model,
        )
        record = {
            "reference": model_entry["reference"],
            "language": model_entry.get("language", ""),
            "source": model_entry.get("source", ""),
            "audio": model_entry.get("audio", ""),
            "model_hypothesis": model_entry["hypothesis"],
            "model_cer": model_entry.get("cer", -1),
            "model_llm_score": result["score"],
            "model_analysis": result["analysis"],
            "tencent_hypothesis": tencent_entry["hypothesis"],
        }
        if result.get("error"):
            record["error"] = True
        return idx, record

    results_by_idx: dict[int, dict] = {}
    stats = {"total": 0, "errors": 0, "score_sum": 0}
    lang_stats: dict[str, dict] = {}

    logger.info(f"Scoring with {args.max_workers} concurrent workers...")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(_evaluate, (i, pair)): i
            for i, pair in enumerate(matched_pairs)
        }
        with tqdm(total=len(futures), desc="LLM scoring") as pbar:
            for future in as_completed(futures):
                idx, record = future.result()
                results_by_idx[idx] = record

                if record.get("error"):
                    stats["errors"] += 1
                else:
                    score = record["model_llm_score"]
                    stats["total"] += 1
                    stats["score_sum"] += score

                    lang = record["language"]
                    if lang not in lang_stats:
                        lang_stats[lang] = {"count": 0, "score_sum": 0}
                    lang_stats[lang]["count"] += 1
                    lang_stats[lang]["score_sum"] += score

                pbar.update(1)

    all_results = [results_by_idx[i] for i in sorted(results_by_idx)]

    per_lang = {}
    for lang, ls in sorted(lang_stats.items()):
        per_lang[lang] = {
            "count": ls["count"],
            "avg_score": round(ls["score_sum"] / ls["count"], 2),
        }

    bad_cases = [
        r for r in all_results
        if isinstance(r.get("model_llm_score"), (int, float))
        and r["model_llm_score"] < args.bad_threshold
        and not r.get("error")
    ]
    bad_cases.sort(key=lambda x: x["model_llm_score"])

    output = {
        "metadata": {
            "model_results_path": args.model_results,
            "tencent_results_path": args.tencent_results,
            "sample_size": len(all_results),
            "total_matched": len(matched_pairs),
            "seed": args.seed,
            "llm_model": args.llm_model,
            "language_filter": args.language,
            "bad_threshold": args.bad_threshold,
        },
        "statistics": {
            "total_evaluated": stats["total"],
            "errors": stats["errors"],
            "avg_llm_score": (
                round(stats["score_sum"] / stats["total"], 2)
                if stats["total"] > 0 else None
            ),
        },
        "per_language": per_lang,
        "bad_case_summary": {
            "threshold": args.bad_threshold,
            "count": len(bad_cases),
            "ratio": (
                f"{len(bad_cases)}/{stats['total']}"
                if stats["total"] > 0 else "N/A"
            ),
        },
        "bad_cases": bad_cases,
        "all_results": all_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("Evaluation Complete!")
    logger.info(f"Total evaluated: {stats['total']}, errors: {stats['errors']}")
    if stats["total"] > 0:
        logger.info(f"Model avg LLM score: {stats['score_sum'] / stats['total']:.2f}")
    logger.info("-" * 40)
    logger.info("Per-language:")
    for lang, ls in sorted(per_lang.items()):
        logger.info(f"  [{lang}] n={ls['count']} avg={ls['avg_score']:.1f}")
    logger.info("-" * 40)
    logger.info(
        f"Bad cases (score < {args.bad_threshold}): "
        f"{len(bad_cases)}/{stats['total']}"
    )
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
