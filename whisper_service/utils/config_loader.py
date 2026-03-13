#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config loader for the LoRA-finetuned Whisper ASR service.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class WhisperLoRAConfig:

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

        model_cfg = config_dict.get("model", {})
        self.base_model_id = model_cfg.get("base_model_id")
        self.adapter_path = model_cfg.get("adapter_path")
        self.local_files_only = model_cfg.get("local_files_only", True)
        self.target_sample_rate = model_cfg.get("target_sample_rate", 16000)
        self.language = model_cfg.get("language")
        self.task = model_cfg.get("task", "transcribe")

        deploy_cfg = config_dict.get("deployment", {})
        self.num_replicas = deploy_cfg.get("num_replicas", 1)
        self.max_ongoing_requests = deploy_cfg.get("max_ongoing_requests", 100)

        gpu_per_replica_config = deploy_cfg.get("gpu_per_replica")
        if gpu_per_replica_config is None:
            self.gpu_per_replica = 1.0 / self.num_replicas if self.num_replicas > 0 else 1.0
        else:
            self.gpu_per_replica = float(gpu_per_replica_config)

        batch_cfg = config_dict.get("batch", {})
        self.max_batch_size = batch_cfg.get("max_batch_size", 8)
        self.batch_wait_timeout_s = batch_cfg.get("batch_wait_timeout_s", 0.05)

        gen_cfg = config_dict.get("generation", {})
        self.max_new_tokens = gen_cfg.get("max_new_tokens", 448)
        self.num_beams = gen_cfg.get("num_beams", 1)

        server_cfg = config_dict.get("server", {})
        self.host = "127.0.0.1"
        self.port = server_cfg.get("port", 8009)

    @property
    def has_adapter(self) -> bool:
        return self.adapter_path is not None

    def __repr__(self):
        adapter_label = self.adapter_path or "none"
        return (
            f"WhisperLoRAConfig(replicas={self.num_replicas}, "
            f"batch_size={self.max_batch_size}, adapter={adapter_label})"
        )


def load_config(config_path: Optional[str] = None) -> WhisperLoRAConfig:
    if config_path is None:
        current_dir = Path(__file__).parent.parent
        config_path = current_dir / "config" / "whisper_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return WhisperLoRAConfig(config_dict)


def save_config(config: WhisperLoRAConfig, output_path: str):
    config_dict = {
        "model": {
            "base_model_id": config.base_model_id,
            "adapter_path": config.adapter_path,
            "local_files_only": config.local_files_only,
            "target_sample_rate": config.target_sample_rate,
            "language": config.language,
            "task": config.task,
        },
        "deployment": {
            "num_replicas": config.num_replicas,
            "gpu_per_replica": config.gpu_per_replica,
            "max_ongoing_requests": config.max_ongoing_requests,
        },
        "batch": {
            "max_batch_size": config.max_batch_size,
            "batch_wait_timeout_s": config.batch_wait_timeout_s,
        },
        "generation": {
            "max_new_tokens": config.max_new_tokens,
            "num_beams": config.num_beams,
        },
        "server": {
            "port": config.port,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
