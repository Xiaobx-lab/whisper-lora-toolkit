"""
Custom language token support for Whisper multilingual fine-tuning.

Adds new language tokens (e.g. <|nan|> for Minnan/Hokkien) to Whisper's
vocabulary with smart embedding initialization and selective gradient
freezing — only new-token embeddings receive updates while all original
embeddings stay frozen.

Toggle between mapping-based (nan -> zh) and new-token approaches purely
via config parameters, with zero impact on existing training pipelines.

Config parameters (all optional, omit to keep original behaviour):
    ADD_CUSTOM_LANG_TOKENS   list[str] | None   e.g. ["nan"]
    CUSTOM_LANG_TOKEN_INIT   dict[str, list[str]]   e.g. {"nan": ["zh", "yue"]}
    UNFREEZE_LANG_EMBEDDINGS bool (default False)
        When True, Whisper's ~50 built-in language-token embedding rows
        are kept trainable alongside the new custom tokens, so the
        language classification head can adapt to LoRA-shifted
        representations.  Text-token embeddings stay frozen.
"""

import logging
import os
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

_EMBEDDING_PARAM_KEYWORDS = ("embed_tokens.weight", "proj_out.weight", "lm_head.weight")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_language_tokens(
    model,
    processor,
    tokens_to_add: List[str],
    init_from: Dict[str, List[str]],
) -> Optional[dict]:
    """
    Add custom language tokens and initialize their embeddings.

    Must be called BEFORE LoRA/PEFT wrapping.

    Returns dict with 'orig_vocab_size' and 'new_token_ids',
    or None if nothing was actually added.
    """
    if hasattr(model, "peft_config"):
        raise RuntimeError(
            "add_language_tokens() must be called BEFORE PEFT wrapping. "
            "The model already has peft_config — call this on the raw "
            "WhisperForConditionalGeneration before applying LoRA."
        )

    tokenizer = processor.tokenizer
    orig_vocab_size = model.get_input_embeddings().weight.shape[0]

    new_special = [
        f"<|{lang}|>" for lang in tokens_to_add
        if f"<|{lang}|>" not in tokenizer.get_vocab()
    ]
    if not new_special:
        logger.info("Custom language tokens already exist in vocab")
        return None

    actually_added_langs = [
        tok[2:-2] for tok in new_special
    ]

    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": new_special}
    )
    model.resize_token_embeddings(len(tokenizer))
    logger.info(
        "Added %d token(s) %s | vocab %d -> %d",
        num_added, new_special, orig_vocab_size, len(tokenizer),
    )

    new_token_ids = _init_embeddings(
        model, tokenizer, actually_added_langs, init_from,
    )
    _patch_tokenizer_prefix(tokenizer, actually_added_langs)
    _patch_generation_config(model, tokenizer, actually_added_langs)

    return {"orig_vocab_size": orig_vocab_size, "new_token_ids": new_token_ids}


def setup_embedding_grad_hooks(
    model,
    orig_vocab_size: int,
    trainable_token_ids: Optional[set] = None,
):
    """
    Freeze original vocabulary embeddings via gradient hooks.

    Rows >= orig_vocab_size (custom tokens) always receive updates.
    If *trainable_token_ids* is provided, those rows within [0, orig_vocab_size)
    are also kept trainable (e.g. language-token rows so the language
    classification head can adapt to LoRA-shifted representations).

    Must be called AFTER PEFT wrapping AND model.to(device), because
    .to() replaces Parameter objects and would invalidate earlier hooks.

    Returns a list of hook handles (keep a reference to prevent GC).
    """
    keep_ids = set()
    if trainable_token_ids:
        keep_ids = {tid for tid in trainable_token_ids if 0 <= tid < orig_vocab_size}

    hooks = []
    seen_ptrs: set = set()

    for name, param in model.named_parameters():
        if not any(kw in name for kw in _EMBEDDING_PARAM_KEYWORDS):
            continue

        ptr = param.data_ptr()
        if ptr in seen_ptrs:
            logger.info(
                "Skipping %s (tied weight, same tensor already hooked)", name,
            )
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
        n_trainable_orig = len(keep_ids)
        logger.info(
            "Grad hook: %s | freeze %d/%d orig rows, "
            "keep %d lang-token rows trainable, "
            "train [%d, %d) (custom tokens)",
            name,
            orig_vocab_size - n_trainable_orig,
            orig_vocab_size,
            n_trainable_orig,
            orig_vocab_size,
            param.shape[0],
        )

    if not hooks:
        logger.warning(
            "No embedding params found — gradient hooks not set"
        )
    return hooks


def load_custom_lang_for_inference(model, processor, adapter_path):
    """
    Restore custom language tokens saved by save_custom_embeddings().

    Detects custom_embeddings.pt in adapter_path; if absent, returns None
    (backward compatible with old adapters that don't have it).

    Adds the custom tokens back to the tokenizer, resizes the model's
    embedding layer, loads the trained weights, and patches both the
    tokenizer prefix_tokens property and generation_config.lang_to_id.

    Must be called BEFORE PEFT wrapping.

    Returns list of custom lang codes (e.g. ["nan"]), or None.
    """
    emb_path = os.path.join(adapter_path, "custom_embeddings.pt")
    if not os.path.isfile(emb_path):
        return None

    if hasattr(model, "peft_config"):
        raise RuntimeError(
            "load_custom_lang_for_inference() must be called BEFORE "
            "PEFT wrapping."
        )

    state = torch.load(emb_path, map_location="cpu", weights_only=True)
    orig_vocab_size = state["orig_vocab_size"]

    from transformers import WhisperTokenizerFast
    saved_tok = WhisperTokenizerFast.from_pretrained(adapter_path)
    vocab = saved_tok.get_vocab()
    custom_tokens = sorted(
        [t for t, tid in vocab.items() if tid >= orig_vocab_size],
        key=lambda t: vocab[t],
    )

    if not custom_tokens:
        logger.warning(
            "custom_embeddings.pt exists but saved tokenizer has no new tokens"
        )
        return None

    num_added = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": custom_tokens}
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    logger.info(
        "Restored %d custom token(s) %s | vocab %d -> %d",
        num_added, custom_tokens, orig_vocab_size, len(processor.tokenizer),
    )

    embed_w = model.get_input_embeddings().weight
    with torch.no_grad():
        for key, val in state.items():
            if key == "lang_token_embeddings":
                continue
            if not isinstance(val, torch.Tensor):
                continue
            n = val.shape[0]
            embed_w[orig_vocab_size:orig_vocab_size + n].copy_(
                val.to(embed_w.dtype)
            )
            logger.info(
                "Loaded %d embedding row(s) [%r] -> positions [%d, %d)",
                n, key, orig_vocab_size, orig_vocab_size + n,
            )

        lang_ids = state.get("lang_token_ids")
        lang_embs = state.get("lang_token_embeddings")
        if lang_ids is not None and lang_embs is not None:
            for i, tid in enumerate(lang_ids):
                embed_w[tid].copy_(lang_embs[i].to(embed_w.dtype))
            logger.info(
                "Restored %d trained language-token embedding rows",
                len(lang_ids),
            )

    custom_langs = [
        t[2:-2] for t in custom_tokens
        if t.startswith("<|") and t.endswith("|>")
    ]
    _patch_tokenizer_prefix(processor.tokenizer, custom_langs)
    _patch_generation_config(model, processor.tokenizer, custom_langs)

    return custom_langs


def save_custom_embeddings(
    model,
    out_dir: str,
    orig_vocab_size: int,
    lang_token_ids: Optional[set] = None,
):
    """Save custom and (optionally) language-token embedding rows."""
    os.makedirs(out_dir, exist_ok=True)
    state: dict = {"orig_vocab_size": orig_vocab_size}
    seen_ptrs: set = set()

    for name, param in model.named_parameters():
        if not any(kw in name for kw in _EMBEDDING_PARAM_KEYWORDS):
            continue
        ptr = param.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)

        key = name.rsplit(".", 1)[-1].replace(".weight", "")
        state[f"{key}_new"] = param.data[orig_vocab_size:].cpu().clone()

    if lang_token_ids:
        ids_sorted = sorted(tid for tid in lang_token_ids if 0 <= tid < orig_vocab_size)
        if ids_sorted:
            emb_param = next(
                p for n, p in model.named_parameters()
                if any(kw in n for kw in _EMBEDDING_PARAM_KEYWORDS)
            )
            state["lang_token_ids"] = ids_sorted
            state["lang_token_embeddings"] = (
                emb_param.data[ids_sorted].cpu().clone()
            )
            logger.info(
                "Including %d language-token embedding rows in checkpoint",
                len(ids_sorted),
            )

    path = os.path.join(out_dir, "custom_embeddings.pt")
    torch.save(state, path)
    n = next(
        (v.shape[0] for v in state.values() if isinstance(v, torch.Tensor)),
        0,
    )
    logger.info("Saved custom embeddings -> %s", path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _init_embeddings(model, tokenizer, langs_to_init, init_from):
    """
    Initialize new-token embeddings from the mean of source language tokens.
    Only operates on langs that were actually added (not pre-existing ones).
    """
    new_token_ids = []

    embed_w = model.get_input_embeddings().weight
    out_proj = model.get_output_embeddings()
    proj_w = out_proj.weight if out_proj is not None else None
    tied = proj_w is not None and proj_w.data_ptr() == embed_w.data_ptr()

    with torch.no_grad():
        for lang in langs_to_init:
            tok = f"<|{lang}|>"
            new_id = tokenizer.convert_tokens_to_ids(tok)
            if new_id == tokenizer.unk_token_id:
                logger.warning("%s resolved to UNK — skipping init", tok)
                continue
            new_token_ids.append(new_id)

            src_ids = _resolve_source_ids(tokenizer, init_from.get(lang, []))
            if not src_ids:
                logger.info(
                    "%s (id=%d): no valid sources, keeping default init",
                    tok, new_id,
                )
                continue

            avg = torch.stack([embed_w[s] for s in src_ids]).mean(0)
            embed_w[new_id].copy_(avg)

            if proj_w is not None and not tied:
                avg_proj = torch.stack([proj_w[s] for s in src_ids]).mean(0)
                proj_w[new_id].copy_(avg_proj)

            src_names = [f"<|{s}|>" for s in init_from.get(lang, [])]
            logger.info(
                "Init %s (id=%d) <- mean(%s)%s",
                tok, new_id, src_names,
                " (tied weights)" if tied else "",
            )

    return new_token_ids


def _resolve_source_ids(tokenizer, source_langs):
    ids = []
    for sl in source_langs:
        sid = tokenizer.convert_tokens_to_ids(f"<|{sl}|>")
        if sid != tokenizer.unk_token_id:
            ids.append(sid)
        else:
            logger.warning("Source <|%s|> not in vocab, skipping", sl)
    return ids


def _patch_tokenizer_prefix(tokenizer, custom_langs):
    """
    Patch THIS tokenizer instance's prefix_tokens property so that
    set_prefix_tokens(language="nan") works for custom languages
    not listed in Whisper's built-in LANGUAGES dict.

    Creates a one-off subclass bound only to this instance, so other
    tokenizer instances of the same class are not affected.
    """
    custom_id_map = {}
    for lang in custom_langs:
        tid = tokenizer.convert_tokens_to_ids(f"<|{lang}|>")
        if tid != tokenizer.unk_token_id:
            custom_id_map[lang] = tid

    if not custom_id_map:
        return

    original_cls = type(tokenizer)
    original_getter = original_cls.prefix_tokens.fget

    def _patched_prefix_tokens(self):
        if self.language is not None and self.language.lower() in custom_id_map:
            bos = self.convert_tokens_to_ids("<|startoftranscript|>")
            lang_id = custom_id_map[self.language.lower()]
            transcribe = self.convert_tokens_to_ids("<|transcribe|>")
            translate = self.convert_tokens_to_ids("<|translate|>")
            notimestamps = self.convert_tokens_to_ids("<|notimestamps|>")

            seq = [bos, lang_id]
            if self.task is not None:
                seq.append(
                    transcribe if self.task == "transcribe" else translate
                )
            if not self.predict_timestamps:
                seq.append(notimestamps)
            return seq

        return original_getter(self)

    patched_cls = type(
        f"{original_cls.__name__}_CustomLang",
        (original_cls,),
        {"prefix_tokens": property(_patched_prefix_tokens)},
    )
    tokenizer.__class__ = patched_cls

    logger.info(
        "Patched tokenizer instance (%s -> %s) for custom languages: %s",
        original_cls.__name__,
        patched_cls.__name__,
        list(custom_id_map.keys()),
    )


def _patch_generation_config(model, tokenizer, custom_langs):
    """
    Register custom language tokens in generation_config.lang_to_id so
    that model.generate(language="nan") works without crashing in
    _retrieve_init_tokens -> language_to_id().
    """
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        return

    lang_to_id = getattr(gen_cfg, "lang_to_id", None)
    if lang_to_id is None:
        return

    for lang in custom_langs:
        token_str = f"<|{lang}|>"
        tid = tokenizer.convert_tokens_to_ids(token_str)
        if tid != tokenizer.unk_token_id:
            lang_to_id[token_str] = tid
            lang_to_id[lang] = tid
            logger.info(
                "Registered %r and %r (id=%d) in generation_config.lang_to_id",
                token_str, lang, tid,
            )
