"""Qwen3-32B forward + span mean-pooling."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch


def char_span_to_token_indices(
    offset_mapping: list[tuple[int, int]],
    start: int,
    end: int,
) -> list[int]:
    idx: list[int] = []
    for i, (a, b) in enumerate(offset_mapping):
        if b <= start or a >= end:
            continue
        if a == 0 and b == 0:
            continue
        idx.append(i)
    return idx


def find_char_span(text: str, needle: str, *, after: str | None = None) -> tuple[int, int]:
    hay = str(text)
    pos = 0
    if after:
        anchor = hay.find(after)
        if anchor >= 0:
            pos = anchor + len(after)
    start = hay.find(needle, pos)
    if start < 0:
        return -1, -1
    return start, start + len(needle)


def section_span(text: str, header: str, next_header: str | None = None) -> tuple[int, int]:
    start = text.find(header)
    if start < 0:
        return -1, -1
    start += len(header)
    end = len(text)
    if next_header:
        nxt = text.find(next_header, start)
        if nxt >= 0:
            end = nxt
    return start, end


def last_token_hidden(
    hidden: torch.Tensor,
    indices: list[int],
    *,
    fallback_index: int = -1,
) -> np.ndarray:
    if hidden.dim() == 3:
        hidden = hidden[0]
    idx = indices[-1] if indices else fallback_index
    out = hidden[idx].detach().float().cpu().numpy()
    n = np.linalg.norm(out)
    if n > 1e-8:
        out = out / n
    return out


def mean_pool_hidden(
    hidden: torch.Tensor,
    indices: list[int],
    *,
    fallback_last: int = 32,
) -> np.ndarray:
    """hidden: (seq, dim); return (dim,) float32."""
    if hidden.dim() == 3:
        hidden = hidden[0]
    if indices:
        vec = hidden[indices].mean(dim=0)
    else:
        vec = hidden[-fallback_last:].mean(dim=0)
    out = vec.detach().float().cpu().numpy()
    n = np.linalg.norm(out)
    if n > 1e-8:
        out = out / n
    return out


def bcp_claim_indices(prompt: str, claim_text: str, offset_mapping: list[tuple[int, int]]) -> list[int]:
    needle = claim_text.strip()
    if not needle:
        return []
    marker = "[Claim]"
    mpos = prompt.find(marker)
    search_from = mpos + len(marker) if mpos >= 0 else 0
    start = prompt.find(needle, search_from)
    if start < 0:
        start = prompt.find(needle)
    if start < 0:
        return []
    return char_span_to_token_indices(offset_mapping, start, start + len(needle))


def oc_section_indices(prompt: str, offset_mapping: list[tuple[int, int]]) -> dict[str, list[int]]:
    spans = {
        "anchor": section_span(prompt, "Object A:\n", "Object B:"),
        "source": section_span(prompt, "Object B:\n", "Claim:"),
        "claim": section_span(prompt, "Claim:\n", None),
    }
    out: dict[str, list[int]] = {}
    for name, (s, e) in spans.items():
        out[name] = char_span_to_token_indices(offset_mapping, s, e) if s >= 0 else []
    return out


def load_model_and_tokenizer(
    model_path: str,
    *,
    device_map: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs: dict = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    use_device_map = device_map and device_map != "none"
    if use_device_map:
        try:
            import accelerate  # noqa: F401
            load_kwargs["device_map"] = device_map
        except ImportError:
            use_device_map = False
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        attn_impl = "sdpa"
    load_kwargs["attn_implementation"] = attn_impl

    if not use_device_map:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if torch.cuda.is_available():
            model = model.to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    n_layers = getattr(model.config, "num_hidden_layers", None) or getattr(
        model.config, "n_layer", None
    )
    return model, tokenizer, n_layers


def forward_hidden_states(
    model,
    tokenizer,
    prompt: str,
    *,
    max_length: int,
    layer_indices: list[int],
) -> tuple[dict[int, torch.Tensor], list[tuple[int, int]]]:
    """One forward; return {layer_index: (seq, dim)} and offset_mapping."""
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")[0]
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hs_by_layer: dict[int, torch.Tensor] = {}
    for li in layer_indices:
        hs_by_layer[li] = out.hidden_states[li + 1][0]
    return hs_by_layer, offsets


def claim_features_multi_layer(
    hs_by_layer: dict[int, torch.Tensor],
    layer_order: list[int],
    prompt: str,
    claim_text: str,
    offsets: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Returns mean_arr, last_arr each shape (n_layers, dim), L2-normalized."""
    idx = bcp_claim_indices(prompt, claim_text, offsets)
    mean_rows = []
    last_rows = []
    for li in layer_order:
        hs = hs_by_layer[li]
        mean_rows.append(mean_pool_hidden(hs, idx))
        last_rows.append(last_token_hidden(hs, idx))
    return np.stack(mean_rows, axis=0), np.stack(last_rows, axis=0)
