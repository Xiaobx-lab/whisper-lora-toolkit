#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper ASR Service with LoRA adapter support.
Ray Serve + FastAPI with @serve.batch for true batch inference.

Based on whipserZoo architecture, extended for LoRA fine-tuned models
with custom language token support (e.g. <|nan|>).

Usage:
    conda activate asr_trainer
    python whisper_service/whisper_lora_server.py
    python whisper_service/whisper_lora_server.py --config whisper_service/config/whisper_config.yaml
"""

import os

os.environ["RAY_DEDUP_LOGS"] = "0"

import argparse
import base64
import signal
import sys
import time
import numpy as np
from typing import List, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from ray import serve

from utils.config_loader import load_config

_config_path = None
for i, arg in enumerate(sys.argv):
    if arg == "--config" and i + 1 < len(sys.argv):
        _config_path = sys.argv[i + 1]
        break

CONFIG = load_config(_config_path)

app = FastAPI(title="Whisper LoRA ASR (Ray Serve batched)")


class AsrRequest(BaseModel):
    audio_data: str
    sample_rate: int = 16000


class InferRequest(BaseModel):
    action: str = "infer"
    audio: str


def _pcm16_b64_to_f32(b64: str) -> np.ndarray:
    buf = base64.b64decode(b64)
    x = np.frombuffer(buf, dtype=np.int16)
    return (x.astype(np.float32) / 32768.0).copy()


@serve.deployment(
    num_replicas=CONFIG.num_replicas,
    ray_actor_options={"num_gpus": CONFIG.gpu_per_replica},
    max_ongoing_requests=CONFIG.max_ongoing_requests,
)
@serve.ingress(app)
class WhisperLoRAASR:

    def __init__(self):
        self._replica_id = os.getpid()
        self._ready = False
        self._model = None
        self._processor = None
        self._torch = None
        self._taF = None
        self._device = "cpu"
        self._config = CONFIG
        self._load()

    def _load(self):
        import torch
        import torchaudio
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        self._torch = torch
        self._taF = torchaudio.functional
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        torch.set_grad_enabled(False)
        torch.set_float32_matmul_precision("high")

        t0 = time.time()
        print(
            f"[replica-{self._replica_id}][init] loading model on {self._device} ...",
            file=sys.stderr, flush=True,
        )
        print(
            f"[replica-{self._replica_id}][config] base_model={self._config.base_model_id}",
            file=sys.stderr, flush=True,
        )
        print(
            f"[replica-{self._replica_id}][config] adapter={self._config.adapter_path}",
            file=sys.stderr, flush=True,
        )
        print(
            f"[replica-{self._replica_id}][config] batch_size={self._config.max_batch_size}, "
            f"replicas={self._config.num_replicas}",
            file=sys.stderr, flush=True,
        )

        self._processor = WhisperProcessor.from_pretrained(
            self._config.base_model_id,
            local_files_only=self._config.local_files_only,
        )
        base_model = WhisperForConditionalGeneration.from_pretrained(
            self._config.base_model_id,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=self._config.local_files_only,
        )

        if self._config.has_adapter:
            self._model = self._apply_lora(base_model, model_dtype)
        else:
            self._model = base_model

        if self._config.language is not None:
            forced_ids = self._processor.get_decoder_prompt_ids(
                language=self._config.language,
                task=self._config.task,
            )
            self._model.config.forced_decoder_ids = forced_ids
            self._model.generation_config.forced_decoder_ids = forced_ids
            print(
                f"[replica-{self._replica_id}][config] forced language={self._config.language}, "
                f"task={self._config.task}",
                file=sys.stderr, flush=True,
            )
        else:
            self._model.config.forced_decoder_ids = None
            self._model.generation_config.forced_decoder_ids = None

        self._model.config.suppress_tokens = []
        self._model.generation_config.suppress_tokens = []
        self._model.to(self._device)
        self._model.eval()

        self._warmup(model_dtype)

        elapsed = time.time() - t0
        total_params = sum(p.numel() for p in self._model.parameters())
        print(
            f"[replica-{self._replica_id}][ready] model loaded in {elapsed:.2f}s "
            f"({total_params:,} params, dtype={model_dtype})",
            file=sys.stderr, flush=True,
        )
        self._ready = True

    def _apply_lora(self, base_model, model_dtype):
        from peft import LoraConfig
        from peft.utils.save_and_load import set_peft_model_state_dict
        from safetensors.torch import load_file as safe_load_file

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, project_root)
        from whisper_peft import PeftModelForWhisper
        from custom_lang import load_custom_lang_for_inference

        adapter_path = self._config.adapter_path
        print(
            f"[replica-{self._replica_id}][lora] loading adapter from {adapter_path}",
            file=sys.stderr, flush=True,
        )

        custom_emb_file = os.path.join(adapter_path, "custom_embeddings.pt")
        if os.path.isfile(custom_emb_file):
            custom_langs = load_custom_lang_for_inference(
                base_model, self._processor, adapter_path,
            )
            if custom_langs:
                print(
                    f"[replica-{self._replica_id}][lora] custom language tokens: {custom_langs}",
                    file=sys.stderr, flush=True,
                )

        lora_cfg = LoraConfig.from_pretrained(adapter_path)
        model = PeftModelForWhisper(base_model, lora_cfg)

        adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
        if os.path.isfile(adapter_file):
            adapter_weights = safe_load_file(adapter_file)
        else:
            adapter_file = os.path.join(adapter_path, "adapter_model.bin")
            adapter_weights = self._torch.load(
                adapter_file, map_location="cpu", weights_only=True,
            )
        set_peft_model_state_dict(model, adapter_weights)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[replica-{self._replica_id}][lora] adapter loaded, "
            f"{trainable:,} trainable params",
            file=sys.stderr, flush=True,
        )
        return model

    def _warmup(self, model_dtype):
        try:
            silent = np.zeros(
                int(self._config.target_sample_rate * 0.2), dtype=np.float32,
            )
            feats = self._processor(
                [silent],
                sampling_rate=self._config.target_sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_features = feats.input_features.to(self._device, dtype=model_dtype)
            if input_features.shape[-1] < 3000:
                pad_len = 3000 - input_features.shape[-1]
                input_features = self._torch.nn.functional.pad(
                    input_features, (0, pad_len), value=0.0,
                )
            with self._torch.no_grad():
                _ = self._model.generate(input_features=input_features, max_new_tokens=1)
            print(
                f"[replica-{self._replica_id}][init] warmup done",
                file=sys.stderr, flush=True,
            )
        except Exception as e:
            print(
                f"[replica-{self._replica_id}][init] warmup skipped: {e}",
                file=sys.stderr, flush=True,
            )

    # ------------------------------------------------------------------
    # Audio pre-processing
    # ------------------------------------------------------------------

    def _prepare(self, items: List[Dict]):
        waves, errors = [], {}
        for i, it in enumerate(items):
            try:
                b64 = it.get("audio_data", "")
                sr = int(it.get("sample_rate", self._config.target_sample_rate))
                if not b64:
                    errors[i] = "audio_data_missing"
                    continue

                wav = _pcm16_b64_to_f32(b64)
                if sr != self._config.target_sample_rate and wav.size > 0:
                    t = self._torch.from_numpy(wav).float().unsqueeze(0)
                    with self._torch.no_grad():
                        y = self._taF.resample(
                            t,
                            orig_freq=sr,
                            new_freq=self._config.target_sample_rate,
                        ).squeeze(0)
                    wav = y.numpy().astype(np.float32)

                waves.append(wav)
            except Exception as e:
                errors[i] = f"decode_or_resample_error: {e}"
        return waves, errors

    # ------------------------------------------------------------------
    # True batch inference via @serve.batch
    # ------------------------------------------------------------------

    @serve.batch(
        max_batch_size=CONFIG.max_batch_size,
        batch_wait_timeout_s=CONFIG.batch_wait_timeout_s,
    )
    async def _batched_infer(self, items: List[Dict]):
        if not self._ready:
            return [{"error": "service_not_ready"} for _ in items]

        t0 = time.time()
        waves, errors = self._prepare(items)
        if not waves:
            return [
                {"error": errors.get(i, "invalid_request")}
                for i in range(len(items))
            ]

        feats = self._processor(
            waves,
            sampling_rate=self._config.target_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_features = feats.input_features.to(
            self._device, dtype=self._model.dtype,
        )

        if input_features.shape[-1] < 3000:
            pad_len = 3000 - input_features.shape[-1]
            input_features = self._torch.nn.functional.pad(
                input_features, (0, pad_len), value=0.0,
            )

        # --- encoder forward (once!) ---
        base = (
            self._model.get_base_model()
            if hasattr(self._model, "get_base_model")
            else self._model
        )
        encoder_outputs = base.get_encoder()(input_features[:, :, :3000])

        # --- language detection reusing encoder_outputs ---
        languages = []
        if self._config.language is not None:
            languages = [self._config.language] * len(waves)
        else:
            languages = self._detect_languages(input_features, encoder_outputs)

        # --- generation reusing encoder_outputs ---
        gen_kwargs = dict(
            encoder_outputs=encoder_outputs,
            num_beams=self._config.num_beams,
            max_new_tokens=self._config.max_new_tokens,
        )

        if self._config.language is not None:
            gen_kwargs["language"] = self._config.language
            gen_kwargs["task"] = self._config.task or "transcribe"
        else:
            gen_kwargs["language"] = [
                l if l != "unknown" else "en" for l in languages
            ]
            gen_kwargs["task"] = self._config.task or "transcribe"

        with self._torch.no_grad():
            token_ids = self._model.generate(
                input_features=input_features, **gen_kwargs,
            )

        texts = self._processor.batch_decode(token_ids, skip_special_tokens=True)

        res, j = [], 0
        for i in range(len(items)):
            if i in errors:
                res.append({"error": errors[i]})
            else:
                res.append({"text": texts[j], "language": languages[j]})
                j += 1

        batch_time = time.time() - t0
        print(
            f"[replica-{self._replica_id}][batch] n={len(items)} "
            f"time={batch_time:.3f}s",
            file=sys.stderr, flush=True,
        )
        return res

    def _detect_languages(self, input_features, encoder_outputs):
        """Detect languages using pre-computed encoder_outputs (no redundant encoder run)."""
        gen_cfg = self._model.generation_config
        lang_to_id = getattr(gen_cfg, "lang_to_id", None)

        if not lang_to_id:
            return ["unknown"] * input_features.shape[0]

        id_to_lang = {v: k for k, v in lang_to_id.items()}
        bs = input_features.shape[0]

        start_token_id = self._model.config.decoder_start_token_id
        decoder_input_ids = self._torch.full(
            (bs, 1), start_token_id,
            device=input_features.device, dtype=self._torch.long,
        )

        base = (
            self._model.get_base_model()
            if hasattr(self._model, "get_base_model")
            else self._model
        )
        logits = base(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        ).logits[:, -1]

        non_lang_mask = self._torch.ones(
            logits.shape[-1], dtype=self._torch.bool, device=logits.device,
        )
        non_lang_mask[list(lang_to_id.values())] = False
        logits[:, non_lang_mask] = -np.inf

        probs = self._torch.softmax(logits.float(), dim=-1)
        best_ids = probs.argmax(dim=-1)

        results = []
        for i in range(bs):
            bid = best_ids[i].item()
            tag = id_to_lang.get(bid, "")
            lang = tag.strip("<|>") or "unknown"
            results.append(lang)
        return results

    # ------------------------------------------------------------------
    # HTTP endpoints
    # ------------------------------------------------------------------

    @app.get("/healthz")
    async def health(self):
        return PlainTextResponse("ok")

    @app.post("/asr")
    async def asr(self, req: AsrRequest):
        item = {"audio_data": req.audio_data, "sample_rate": req.sample_rate}
        out = await self._batched_infer(item)
        return JSONResponse(out)

    @app.post("/infer")
    async def infer_legacy(self, req: InferRequest):
        item = {"audio_data": req.audio, "sample_rate": 16000}
        out = await self._batched_infer(item)
        return JSONResponse(out)


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------

def start():
    import ray

    if not ray.is_initialized():
        ray.init()

    serve.start(http_options={"host": "127.0.0.1", "port": CONFIG.port})
    serve.run(WhisperLoRAASR.bind())

    print("\n" + "=" * 60, flush=True)
    print("Whisper LoRA ASR Service Started", flush=True)
    print("=" * 60, flush=True)
    print(f"Endpoint (new):    http://{CONFIG.host}:{CONFIG.port}/asr", flush=True)
    print(f"Endpoint (legacy): http://{CONFIG.host}:{CONFIG.port}/infer", flush=True)
    print(f"Health check:      http://{CONFIG.host}:{CONFIG.port}/healthz", flush=True)
    print(f"Base model:        {CONFIG.base_model_id}", flush=True)
    print(f"LoRA adapter:      {CONFIG.adapter_path or 'none'}", flush=True)
    print(f"Replicas:          {CONFIG.num_replicas}", flush=True)
    print(f"GPU per replica:   {CONFIG.gpu_per_replica}", flush=True)
    print(f"Batch size:        {CONFIG.max_batch_size}", flush=True)
    print(f"Batch timeout:     {CONFIG.batch_wait_timeout_s}s", flush=True)
    print(f"Language:          {CONFIG.language or 'auto-detect'}", flush=True)

    try:
        status = serve.status()
        print(f"\nDeployment status: {status}", flush=True)
    except Exception as e:
        print(f"\nNote: Could not fetch deployment status: {e}", flush=True)

    print("=" * 60, flush=True)
    print("Press Ctrl+C to stop the service\n", flush=True)

    def _stop(*_):
        print("\nShutting down ...", flush=True)
        try:
            serve.shutdown()
        finally:
            ray.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    while True:
        time.sleep(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper LoRA ASR Service")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to whisper_config.yaml (default: config/whisper_config.yaml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    start()
