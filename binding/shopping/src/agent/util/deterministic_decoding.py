"""
Greedy / deterministic decoding defaults for paper main-table runs (vLLM OpenAI API).

Maps expert protocol to OpenAI-compatible + vLLM extra_body fields.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Frozen paper model roles (see docs/paper_proposal_v3.md §8).
PRIMARY_AGENT_MODEL_ID = "Qwen/Qwen3-32B"
PRIMARY_JUDGE_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
ROBUSTNESS_AGENT_MODEL_ID = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
DETERMINISTIC_SEED = 42

# Version tags for reproducibility (bump when prompts/protocol change).
PROMPT_TEMPLATE_VERSION = "shopping_rollout_v1_official"
SYSTEM_PROMPT_VERSION = "rollout.md@official_upstream_2026-05-18"
TOOL_PROTOCOL_VERSION = "shoppingbench_product_tools_v1"
BENCHMARK_ADAPTER_VERSION = "phantom_merge_shopping_v2"


def openai_chat_decoding_kwargs(
    *,
    max_tokens: int,
    seed: int = DETERMINISTIC_SEED,
    stream: bool | None = None,
) -> dict[str, Any]:
    """Greedy decoding: temperature=0, top_p=1, top_k disabled, fixed seed."""
    out: dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "n": 1,
        "seed": seed,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "extra_body": {
            "top_k": -1,
            "repetition_penalty": 1.0,
            "do_sample": False,
        },
    }
    if stream is not None:
        out["stream"] = stream
    return out


def merge_model_config(base: dict[str, Any], *, max_tokens: int, seed: int = DETERMINISTIC_SEED) -> dict[str, Any]:
    """Apply deterministic decoding onto a bench model_config dict."""
    cfg = dict(base)
    dec = openai_chat_decoding_kwargs(max_tokens=max_tokens, seed=seed, stream=cfg.get("stream"))
    for key, value in dec.items():
        if key == "extra_body":
            extra = dict(cfg.get("extra_body") or {})
            extra.update(value)
            cfg["extra_body"] = extra
        else:
            cfg[key] = value
    return cfg


def _run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=10).strip()
    except Exception:
        return ""


def collect_environment_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "timestamp_unix": int(time.time()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "backend": "vllm_openai_compatible",
    }
    try:
        import torch

        meta["torch_version"] = torch.__version__
        meta["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            meta["gpu_type"] = torch.cuda.get_device_name(0)
            meta["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass
    meta["driver_version"] = _run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]) or None
    try:
        import vllm

        meta["vllm_version"] = getattr(vllm, "__version__", None)
    except Exception:
        meta["vllm_version"] = os.environ.get("VLLM_VERSION")
    return meta


def build_run_metadata(
    *,
    role: str,
    model_id: str,
    decoding: dict[str, Any],
    dataset_split: str,
    run_id: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full metadata blob for a paper run directory."""
    meta = {
        "role": role,
        "model_id": model_id,
        "model_revision_or_commit": os.environ.get("HF_MODEL_REVISION") or os.environ.get("VLLM_MODEL_REVISION"),
        "backend": "vllm",
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "tool_protocol_version": TOOL_PROTOCOL_VERSION,
        "benchmark_adapter_version": BENCHMARK_ADAPTER_VERSION,
        "dataset_split": dataset_split,
        "run_id": run_id,
        "decoding": decoding,
        "environment": collect_environment_metadata(),
    }
    if extra:
        meta.update(extra)
    return meta


def write_run_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
