import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from util import load_audio_16k_mono, normalize_text

logger = logging.getLogger(__name__)


class JsonlASRDataset(Dataset):
    """
    JSONL ASR dataset.

    Each line: {"audio": "path.wav", "text": "transcript"[, "language": "zh"]}

    Audio I/O and feature extraction run inside __getitem__ so that
    DataLoader workers can parallelize the disk reads.
    Metadata is stored as parallel lists instead of a list-of-dicts
    to cut per-item memory overhead (~50% less for million-scale datasets).
    """
    def __init__(self, jsonl_path: str, feature_extractor):
        self.feature_extractor = feature_extractor
        self.audio_paths: List[str] = []
        self.texts: List[str] = []
        self.languages: List[Optional[str]] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "audio" not in obj or "text" not in obj:
                    raise ValueError(
                        "Each JSONL line must contain 'audio' and 'text'"
                    )
                self.audio_paths.append(obj["audio"])
                self.texts.append(obj["text"])
                self.languages.append(obj.get("language"))

        logger.info("Loaded %d samples from %s", len(self.audio_paths), jsonl_path)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        wav = load_audio_16k_mono(self.audio_paths[idx])
        feats = self.feature_extractor(
            wav.numpy(), sampling_rate=16000, return_tensors="pt"
        )
        result: Dict[str, Any] = {
            "input_features": feats.input_features[0],
            "text": normalize_text(self.texts[idx]),
        }
        lang = self.languages[idx]
        if lang is not None:
            result["language"] = lang
        return result


@dataclass
class CollatorWhisper:
    """
    Collator: stacks pre-extracted mel features and tokenizes text labels.

    When per_sample_language=True, each sample's "language" field sets the
    decoder prefix tokens for multilingual training.
    """
    processor: Any
    max_label_len: int = 448
    padding: Union[bool, str] = True
    per_sample_language: bool = False
    default_language: str = "zh"
    default_task: str = "transcribe"
    language_mapping: Optional[Dict[str, str]] = None
    strip_bos: bool = False

    def _resolve_language(self, lang: str) -> str:
        if self.language_mapping and lang in self.language_mapping:
            return self.language_mapping[lang]
        return lang

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = torch.stack([item["input_features"] for item in batch])
        texts = [item["text"] for item in batch]

        if self.per_sample_language:
            all_label_ids = []
            for item, text in zip(batch, texts):
                lang = item.get("language", self.default_language)
                lang = self._resolve_language(lang)
                self.processor.tokenizer.set_prefix_tokens(
                    language=lang, task=self.default_task
                )
                token_result = self.processor.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_label_len,
                )
                ids = token_result.input_ids
                if self.strip_bos and ids and ids[0] == self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"):
                    ids = ids[1:]
                all_label_ids.append(ids)

            max_len = max(len(ids) for ids in all_label_ids)
            padded = torch.full(
                (len(all_label_ids), max_len),
                self.processor.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            mask = torch.zeros(len(all_label_ids), max_len, dtype=torch.long)
            for i, ids in enumerate(all_label_ids):
                padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                mask[i, :len(ids)] = 1
            labels = padded.masked_fill(mask.ne(1), -100)
        else:
            label_features = self.processor.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_label_len,
                return_tensors="pt"
            )
            input_ids = label_features.input_ids
            attn_mask = label_features.attention_mask
            if self.strip_bos:
                input_ids = input_ids[:, 1:]
                attn_mask = attn_mask[:, 1:]
            labels = input_ids.masked_fill(attn_mask.ne(1), -100)

        return {
            "input_features": input_features,
            "labels": labels
        }
