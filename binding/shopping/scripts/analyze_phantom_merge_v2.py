#!/usr/bin/env python3
"""
ShoppingBench anchor–claim PM judge (VNEXT).

Runs an LLM judge over agent rollouts and writes per-trajectory PM labels.
Protocol definition: README.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import random
import re
import sys
import time
import socket
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
AGENT = ROOT / "src" / "agent"
for p in (SRC, AGENT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from util.deterministic_decoding import (  # noqa: E402
    DETERMINISTIC_SEED,
    PRIMARY_JUDGE_MODEL_ID,
    build_run_metadata,
    openai_chat_decoding_kwargs,
    write_run_metadata,
)

from analysis.binding_pipeline import (  # noqa: E402
    build_kb_from_steps,
    extract_user_query,
    last_nonempty_response_before_terminate,
    load_rollout_lines,
    recover_anchor_pids,
    trajectory_integrity,
)
from analysis.pm_eval_vnext import (  # noqa: E402
    PM_PIPELINE_VERSION,
    claim_label_counts_from_claims,
    compute_claim_structure_counts,
    derive_support_labels,
    detect_protocol_narrative_pid_mismatch,
    is_pm_relevant_claim,
    load_slot_config,
    normalize_anchor_resolution,
    primary_support_label,
    process_claim_record,
    qa_claim_label_consistency,
    refresh_row_labels,
    trajectory_labels_from_claims,
)


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def read_jsonl(path: Path) -> list[Any]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_rewards(path: Path) -> dict[str, dict[str, Any]]:
    rewards: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rewards
    for row in read_jsonl(path):
        query = str(row.get("query", "")).strip()
        reward = row.get("reward") or {}
        if query:
            rewards[query] = reward
    return rewards


def selected_product_id(anchor_pids: list[str]) -> str:
    return str(anchor_pids[0]) if anchor_pids else ""


def exact_task_correct(anchor_pids: list[str], reward: dict[str, Any] | None) -> bool | None:
    if not reward:
        return None
    gold = str(reward.get("product_id", "")).strip()
    if not gold:
        return None
    return selected_product_id(anchor_pids) == gold


def evidence_excerpt(slot: dict[str, Any], limit: int) -> str:
    bits = [
        f"title: {slot.get('title', '')}",
        f"find: {slot.get('find_snippet', '')}",
        f"detail: {slot.get('detail_text', '')}",
    ]
    return norm(" ".join(bits))[:limit]


def compact_kb(kb: dict[str, dict[str, Any]], anchor_pids: list[str], limit: int) -> list[dict[str, Any]]:
    ordered = []
    seen = set()
    for pid in anchor_pids + sorted(kb.keys()):
        if pid in seen or pid not in kb:
            continue
        seen.add(pid)
        slot = kb[pid]
        ordered.append(
            {
                "product_id": pid,
                "is_selected_anchor": pid in set(anchor_pids),
                "title": slot.get("title", ""),
                "evidence_excerpt": evidence_excerpt(slot, limit),
            }
        )
    return ordered


def lexical_support_pids(kb: dict[str, dict[str, Any]], value: str, quote: str) -> list[str]:
    value_n = norm(value)
    quote_n = norm(quote)
    needles = []
    for candidate in {value_n, quote_n}:
        if len(candidate) >= 3 and candidate not in {"true", "false", "yes", "no"}:
            needles.append(candidate)
    out = []
    for pid, slot in kb.items():
        blob = slot.get("evidence_blob", "")
        blob_n = norm(blob)
        if any(n in blob_n for n in needles):
            out.append(pid)
    return out


def lexical_query_support(query: str, value: str, quote: str) -> bool:
    q = norm(query)
    return any(n in q for n in {norm(value), norm(quote)} if len(n) >= 3)


def fallback_claims_from_response(response: str, anchor_pids: list[str]) -> list[dict[str, Any]]:
    """High-precision, low-recall fallback when no judge is used."""
    patterns = [
        ("shipping", r"\bfree\s+shipping\b|\bfreeshipping\b"),
        ("service", r"\bCOD\b|\bcash\s+on\s+delivery\b"),
        ("service", r"\bofficial\b|\blazmall\b"),
        ("service", r"\bflash\s+sale\b|\blazflash\b"),
        ("material", r"\bbronze\b|\bbrass\b|\balloy\b|\bABS\b|\brubber\b"),
        ("size", r"\b\d+(?:\.\d+)?\s?(?:inch|inches|cm|mm|kg|x)\b"),
    ]
    claims: list[dict[str, Any]] = []
    anchor = selected_product_id(anchor_pids)
    for slot, rx in patterns:
        for m in re.finditer(rx, response or "", flags=re.I):
            span = m.group(0)
            claims.append(
                {
                    "anchor_pid": anchor,
                    "claim": span,
                    "slot": slot,
                    "value": span,
                    "response_quote": span,
                    "confidence": 0.72,
                    "include": True,
                    "judge_rationale": "rule_fallback_high_precision_pattern",
                }
            )
    return claims


@dataclass
class JudgeConfig:
    enabled: bool
    model: str
    base_url: str
    api_key_env: str
    temperature: float
    top_p: float
    max_tokens: int
    seed: int
    timeout: float
    max_retries: int
    evidence_chars: int
    min_confidence: float

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    def decoding_kwargs(self) -> dict[str, Any]:
        return openai_chat_decoding_kwargs(
            max_tokens=self.max_tokens,
            seed=self.seed,
            stream=False,
        )


def _is_transient_judge_failure(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, socket.timeout):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, TimeoutError | socket.timeout):
            return True
        msg = str(exc).lower()
        if "timed out" in msg or "timeout" in msg:
            return True
    msg = str(exc).lower()
    if "timed out" in msg or "temporarily unavailable" in msg:
        return True
    if isinstance(exc, json.JSONDecodeError):
        return True
    if isinstance(exc, ValueError) and (
        "judge did not return json" in msg or "judge json root is not an object" in msg
    ):
        return True
    return False


def _retry_sleep_sec(attempt: int) -> float:
    # Cap backoff so a single stuck request does not dominate wall clock.
    return min(45.0, 2.5 * (2**attempt))


def call_openai_compatible(prompt: dict[str, Any], cfg: JudgeConfig) -> dict[str, Any]:
    if not cfg.api_key:
        raise RuntimeError(f"Missing API key env var: {cfg.api_key_env}")

    url = cfg.base_url.rstrip("/") + "/chat/completions"
    dec = cfg.decoding_kwargs()
    dec["temperature"] = cfg.temperature
    dec["top_p"] = cfg.top_p
    body = {
        "model": cfg.model,
        **dec,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict evidence-grounded evaluator for agent failures. "
                    "Return valid JSON only. Do not use markdown. "
                    "Return structured JSON per the user schema. "
                    "For each claim set anchor_evidence_state (supports|contradicts|absent|unknown), "
                    "cross_supported, query_supported; Python derives final support_labels. "
                    "Never use legacy labels Anchored Composite or Out-of-Set Composite."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last_error: BaseException | None = None
    for attempt in range(cfg.max_retries + 1):
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
                raw_bytes = resp.read()
            obj = json.loads(raw_bytes.decode("utf-8"))
            text = obj["choices"][0]["message"].get("content") or ""
            return parse_json_object(text)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= cfg.max_retries or not _is_transient_judge_failure(exc):
                break
            time.sleep(_retry_sleep_sec(attempt))
    raise RuntimeError(str(last_error))


def parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"Judge did not return JSON object: {text[:200]}")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Judge JSON root is not an object")
    return obj


def build_judge_prompt(
    query: str,
    response: str,
    anchor_pids: list[str],
    kb: dict[str, dict[str, Any]],
    reward: dict[str, Any] | None,
    cfg: JudgeConfig,
) -> dict[str, Any]:
    protocol_pid = selected_product_id(anchor_pids)
    return {
        "task": "phantom_merge_vNEXT_claim_extract_and_classify",
        "pipeline_version": PM_PIPELINE_VERSION,
        "definition": {
            "unit": "anchor-claim pair bound to protocol_anchor_pid",
            "protocol_anchor_rule": (
                "protocol_anchor_pid is the last recommend_product id. "
                "All claims must be factual assertions about that product in the final answer. "
                "Do not use benchmark gold as anchor."
            ),
            "anchor_evidence_state": {
                "supports": "anchor KB has this slot and agrees with the claim",
                "contradicts": "anchor KB has this slot and disagrees with the claim",
                "absent": "anchor KB lacks this slot; claim still asserted in final answer",
                "unknown": (
                    "RARE: evaluator cannot apply supports/contradicts/absent after reading anchor KB "
                    "(NOT for subjective text, NOT for 'unsure' when cross/query flags are known). "
                    "Maps to evidence_gap only in this case."
                ),
            },
            "evidence_gap_definition": (
                "evidence_gap = abstain: KB excerpt insufficient, claim not attributable to a checkable "
                "slot on the anchor, or genuinely conflicting snippets. Do NOT use for marketing fluff, "
                "generic suitability, or claims you could classify as supports/contradicts/absent with more care."
            ),
            "label_mutex": (
                "Final support_labels are derived in code. Only cross_object_merge and "
                "constraint_projection may appear together. correct_binding, anchored_hallucination, "
                "pure_hallucination, evidence_gap are mutually exclusive with each other and with PM pair."
            ),
            "parallel_flags": (
                "When anchor_evidence_state is contradicts or absent, set cross_supported and "
                "query_supported independently (both may be true). Do not pick a single mutually "
                "exclusive failure label in the JSON; downstream code derives support_labels."
            ),
            "derived_labels": {
                "correct_binding": "supports",
                "cross_object_merge": "(contradicts or absent) and cross_supported",
                "constraint_projection": "(contradicts or absent) and query_supported",
                "anchored_hallucination": "contradicts and not cross_supported and not query_supported",
                "pure_hallucination": "absent and not cross_supported and not query_supported (NOT phantom merge)",
                "evidence_gap": "unknown anchor state",
            },
            "has_phantom_merge": "trajectory true if any claim has cross, projection, or anchored_hallucination",
        },
        "claim_policy": [
            "INCLUDE (as claims): concrete checkable product facts — price, size, shape, color, material, "
            "brand, model, feature, compatibility, shipping, seller, quantity, power, etc.",
            "EXCLUDE (put in excluded_spans, NOT claims): subjective praise, 'great choice', 'meets your needs', "
            "ranking, duplicate restatement of title only, vague benefits without a verifiable value.",
            "Each claim needs: slot + normalized value + exact response_quote + anchor_pid = protocol_anchor_pid.",
            "Typical volume: about 4-12 claims per trajectory when the final answer is rich; do not invent micro-claims.",
            "Do NOT use anchor_evidence_state=unknown to avoid deciding — pick supports/contradicts/absent when KB allows.",
        ],
        "query": query,
        "benchmark_gold_product_id_for_task_correctness_only": str((reward or {}).get("product_id", "")),
        "protocol_anchor_pid": protocol_pid,
        "selected_anchor_pids": anchor_pids,
        "final_answer": response,
        "observed_products": compact_kb(kb, anchor_pids, cfg.evidence_chars),
        "output_schema": {
            "anchor_recoverable": "bool",
            "anchor_resolution": {
                "narrative_referent_pid": "product id the final answer describes as recommended",
                "narrative_text_pid": "explicit pid string in final answer if any else null",
                "protocol_matches_narrative": "bool",
                "suspected_wrong_anchor": "bool if narrative seems to be another observed product",
                "confidence": "float",
                "rationale": "string",
            },
            "claims": [
                {
                    "anchor_pid": "protocol anchor pid",
                    "claim": "normalized claim",
                    "slot": "domain slot string",
                    "value": "normalized value",
                    "response_quote": "exact quote from final answer",
                    "anchor_evidence_state": "supports|contradicts|absent|unknown",
                    "cross_supported": "bool",
                    "query_supported": "bool",
                    "supporting_anchor_evidence": "short quote or empty",
                    "supporting_other_product_ids": ["product id"],
                    "supporting_query_text": "short quote or empty",
                    "confidence": "float 0-1",
                    "review_needed": "bool",
                    "rationale": "one sentence",
                }
            ],
            "excluded_spans": [{"response_quote": "str", "reason": "str"}],
        },
    }


def process_judge_response(
    obj: dict[str, Any],
    anchor_pids: list[str],
    query: str,
    kb: dict[str, dict[str, Any]],
    min_conf: float,
) -> tuple[bool, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    canonical, aliases = load_slot_config()
    anchor_recoverable = bool(obj.get("anchor_recoverable", bool(anchor_pids)))
    protocol_pid = selected_product_id(anchor_pids)
    anchor_resolution = normalize_anchor_resolution(obj.get("anchor_resolution"), protocol_pid, anchor_pids)

    raw_claims = obj.get("claims") or []
    claims_extracted: list[dict[str, Any]] = []
    claims_dropped_not_pm_relevant: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    for c in raw_claims:
        if not isinstance(c, dict):
            continue
        extracted = {k: v for k, v in c.items()}
        claims_extracted.append(extracted)
        keep, drop_reason = is_pm_relevant_claim(c)
        if not keep:
            claims_dropped_not_pm_relevant.append(
                {"response_quote": extracted.get("response_quote") or extracted.get("claim"), "reason": drop_reason}
            )
            continue
        row = process_claim_record(c, anchor_pids, query, kb, aliases, canonical)
        if safe_float(row.get("confidence"), 0.0) < min_conf:
            row["review_needed"] = True
        claims.append(row)

    excluded = list(obj.get("excluded_spans") or []) if isinstance(obj.get("excluded_spans"), list) else []
    excluded.extend(claims_dropped_not_pm_relevant)
    return anchor_recoverable, anchor_resolution, claims_extracted, claims, excluded


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def classify_fallback_claim(
    claim: dict[str, Any],
    query: str,
    kb: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    anchor = str(claim.get("anchor_pid") or "").strip()
    value = str(claim.get("value") or "")
    quote = str(claim.get("response_quote") or claim.get("claim") or "")
    support = lexical_support_pids(kb, value, quote)
    if anchor in support:
        state, cross_b, query_b = "supports", False, False
        other: list[str] = []
    else:
        other = [p for p in support if p != anchor]
        cross_b = bool(other)
        query_b = lexical_query_support(query, value, quote)
        state = "absent"
    labels = derive_support_labels(state, cross_b, query_b)
    primary = primary_support_label(labels)
    return {
        **claim,
        "anchor_evidence_state": state,
        "cross_supported": cross_b,
        "query_supported": query_b,
        "support_labels": labels,
        "primary_support_label": primary,
        "support_label": primary,
        "supporting_other_product_ids": other,
        "confidence": safe_float(claim.get("confidence"), 0.7),
        "review_needed": bool(set(labels) & {"cross_object_merge", "constraint_projection"}),
        "rationale": claim.get("judge_rationale", "deterministic fallback lexical support"),
    }


def load_orig_index_map(rollout_path: Path) -> list[int] | None:
    """Map clean-line index -> orig rollout line index (from filter_clean manifest)."""
    stem = rollout_path.name.replace(".clean.jsonl", "").replace(".jsonl", "")
    manifest = ROOT / "filtered" / f"{stem}.manifest.json"
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        mapping = data.get("new_index_to_orig_index")
        if isinstance(mapping, list) and mapping:
            return [int(x) for x in mapping]
    except Exception:
        return None
    return None


def analyze_one(
    index: int,
    steps: list[dict[str, Any]],
    rewards: dict[str, dict[str, Any]],
    judge_cfg: JudgeConfig,
    *,
    orig_index: int | None = None,
) -> dict[str, Any]:
    query = extract_user_query(steps)
    integrity = trajectory_integrity(steps)
    kb = build_kb_from_steps(steps)
    anchor_pids, anchor_step, anchor_src = recover_anchor_pids(
        steps, kb=kb, allow_nl_fallback=True
    )
    response, response_step = last_nonempty_response_before_terminate(steps)
    reward = rewards.get(query)
    task_correct = exact_task_correct(anchor_pids, reward)

    judge_error = None
    judge_raw: dict[str, Any] | None = None
    anchor_resolution: dict[str, Any] = {}
    claims_extracted: list[dict[str, Any]] = []
    excluded_spans: list[dict[str, Any]] = []
    protocol_pid = selected_product_id(anchor_pids)

    if judge_cfg.enabled and response and kb and anchor_pids:
        try:
            prompt = build_judge_prompt(query, response, anchor_pids, kb, reward, judge_cfg)
            judge_raw = call_openai_compatible(prompt, judge_cfg)
            anchor_recoverable, anchor_resolution, claims_extracted, claims, excluded_spans = (
                process_judge_response(judge_raw, anchor_pids, query, kb, judge_cfg.min_confidence)
            )
        except Exception as exc:  # noqa: BLE001
            judge_error = str(exc)
            anchor_recoverable = bool(anchor_pids)
            anchor_resolution = normalize_anchor_resolution(None, protocol_pid, anchor_pids)
            claims_extracted = list(fallback_claims_from_response(response, anchor_pids))
            claims = [classify_fallback_claim(c, query, kb) for c in claims_extracted]
    else:
        anchor_recoverable = bool(anchor_pids)
        anchor_resolution = normalize_anchor_resolution(None, protocol_pid, anchor_pids)
        claims_extracted = list(fallback_claims_from_response(response, anchor_pids))
        claims = [classify_fallback_claim(c, query, kb) for c in claims_extracted]

    pid_mismatch = detect_protocol_narrative_pid_mismatch(response, protocol_pid)
    traj_labels = trajectory_labels_from_claims(claims)
    primary_counts, multiset_counts = claim_label_counts_from_claims(claims)
    structure_counts = compute_claim_structure_counts(claims)
    qa_issues = qa_claim_label_consistency(claims)

    review_priority = "none"
    review_reasons: list[str] = []
    if not anchor_recoverable:
        review_priority = "high"
        review_reasons.append("anchor_unrecoverable")
    if pid_mismatch.get("protocol_text_pid_mismatch"):
        review_reasons.append("protocol_text_pid_mismatch")
    if anchor_resolution.get("suspected_wrong_anchor"):
        review_reasons.append("suspected_wrong_anchor")
    if structure_counts.get("n_claims_cross_and_projection", 0) > 0:
        review_reasons.append("anchor_may_be_wrong")
    for c in claims:
        if c.get("review_needed"):
            review_reasons.append("judge_requested_review")
        pm_labels = set(c.get("support_labels") or []) & {
            "cross_object_merge",
            "constraint_projection",
            "anchored_hallucination",
        }
        if pm_labels and safe_float(c.get("confidence"), 0.0) < 0.8:
            review_reasons.append("low_confidence_phantom_claim")
        if "evidence_gap" in (c.get("support_labels") or []):
            review_reasons.append("evidence_gap_claim")
    if qa_issues:
        review_reasons.append("qa_label_inconsistency")
    if traj_labels["has_phantom_merge"] and task_correct is True:
        review_reasons.append("task_correct_binding_failure")
    if review_reasons:
        review_priority = "high" if traj_labels["has_phantom_merge"] or not anchor_recoverable else "calibration"

    row = {
        "pipeline_version": PM_PIPELINE_VERSION,
        "line_index": index,
        "orig_index": orig_index if orig_index is not None else index,
        "query": query,
        "selected_anchor_pids": anchor_pids,
        "selected_product_id": protocol_pid,
        "gold_product_id": str((reward or {}).get("product_id", "")),
        "task_correct_exact": task_correct,
        "anchor_recoverable": anchor_recoverable,
        "anchor_source": anchor_src,
        "anchor_step": anchor_step,
        "response_step": response_step,
        "final_answer": response,
        "integrity": integrity,
        "exposure": {
            "observed_products": len(kb),
            "selected_anchor_in_kb": protocol_pid in kb,
        },
        "anchor_resolution": anchor_resolution,
        "protocol_pid_check": pid_mismatch,
        "claims_extracted": claims_extracted,
        "claims_dropped_not_pm_relevant_count": len(
            [s for s in excluded_spans if isinstance(s, dict) and s.get("reason") in {
                "empty_claim", "subjective_or_vague", "no_checkable_slot_or_value", "too_short_to_verify",
            }]
        ),
        "claims": claims,
        "excluded_spans": excluded_spans,
        "trajectory_labels": traj_labels,
        "claim_label_counts": primary_counts,
        "claim_label_multiset_counts": multiset_counts,
        "claim_structure_counts": structure_counts,
        "qa_claim_issues": qa_issues,
        "review": {
            "priority": review_priority,
            "reasons": sorted(set(review_reasons)),
        },
        "judge_artifacts": {
            "raw_json": judge_raw,
        },
        "judge": {
            "enabled": judge_cfg.enabled,
            "model": judge_cfg.model if judge_cfg.enabled else None,
            "error": judge_error,
        },
    }
    return row


def aggregate(rows: list[dict[str, Any]], calibration_sample: int, seed: int) -> dict[str, Any]:
    rows = [refresh_row_labels(dict(r)) for r in rows]
    n = len(rows)
    complete = [r for r in rows if r["integrity"].get("trajectory_complete")]
    recoverable = [r for r in rows if r.get("anchor_recoverable")]
    claim_counts = Counter()
    claim_multiset = Counter()
    structure_totals: Counter[str] = Counter()
    for r in rows:
        claim_counts.update(r.get("claim_label_counts") or {})
        claim_multiset.update(r.get("claim_label_multiset_counts") or {})
        structure_totals.update(r.get("claim_structure_counts") or {})

    def rate(num: int, den: int) -> float:
        return (num / den) if den else 0.0

    traj_counts = {
        "has_cross_object_merge": sum(r["trajectory_labels"]["has_cross_object_merge"] for r in rows),
        "has_constraint_projection": sum(r["trajectory_labels"]["has_constraint_projection"] for r in rows),
        "has_anchored_hallucination": sum(r["trajectory_labels"]["has_anchored_hallucination"] for r in rows),
        "has_pure_hallucination": sum(r["trajectory_labels"].get("has_pure_hallucination", False) for r in rows),
        "has_evidence_gap": sum(r["trajectory_labels"].get("has_evidence_gap", False) for r in rows),
        "has_phantom_merge": sum(r["trajectory_labels"]["has_phantom_merge"] for r in rows),
    }

    by_task_binding = defaultdict(int)
    for r in rows:
        tc = r.get("task_correct_exact")
        if tc is True:
            task_bucket = "task_correct"
        elif tc is False:
            task_bucket = "task_incorrect"
        else:
            task_bucket = "task_unknown"
        if r["trajectory_labels"]["has_phantom_merge"]:
            bind_bucket = "phantom_merge"
        elif r["trajectory_labels"]["has_anchored_hallucination"]:
            bind_bucket = "anchored_hallucination_only"
        elif r.get("claims"):
            bind_bucket = "binding_clean_or_correct"
        else:
            bind_bucket = "no_checkable_claims"
        by_task_binding[f"{task_bucket}__{bind_bucket}"] += 1

    high_review = [r for r in rows if r["review"]["priority"] == "high"]
    calibration_pool = [r for r in rows if r["review"]["priority"] == "calibration"]
    rng = random.Random(seed)
    calibration_ids = [r["line_index"] for r in rng.sample(calibration_pool, min(calibration_sample, len(calibration_pool)))]
    review_ids = sorted({r["line_index"] for r in high_review} | set(calibration_ids))

    return {
        "num_trajectories": n,
        "trajectory_complete": len(complete),
        "trajectory_incomplete": n - len(complete),
        "anchor_recoverable": len(recoverable),
        "anchor_recovery_rate": rate(len(recoverable), n),
        "task_correct_exact": sum(r.get("task_correct_exact") is True for r in rows),
        "task_incorrect_exact": sum(r.get("task_correct_exact") is False for r in rows),
        "task_unknown_exact": sum(r.get("task_correct_exact") is None for r in rows),
        "pipeline_version": PM_PIPELINE_VERSION,
        "claim_label_counts": dict(claim_counts),
        "claim_label_rates": {
            k: rate(v, sum(claim_counts.values())) for k, v in sorted(claim_counts.items())
        },
        "claim_label_multiset_counts": dict(claim_multiset),
        "claim_structure_counts": dict(structure_totals),
        "trajectory_label_counts": traj_counts,
        "trajectory_label_rates": {k: rate(v, n) for k, v in traj_counts.items()},
        "task_correctness_x_binding": dict(sorted(by_task_binding.items())),
        "review_policy": {
            "high_priority_cases": len(high_review),
            "calibration_pool_cases": len(calibration_pool),
            "sampled_calibration_cases": len(calibration_ids),
            "total_review_cases": len(review_ids),
            "total_review_rate": rate(len(review_ids), n),
            "review_line_indices": review_ids,
        },
    }


def output_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "per": out_dir / "phantom_merge_v2_per_trajectory.jsonl",
        "summary": out_dir / "phantom_merge_v2_summary.json",
        "review": out_dir / "phantom_merge_v2_review_queue.jsonl",
        "manifest": out_dir / "canonical_v1_complete250_manifest.json",
        "partial_summary": out_dir / "phantom_merge_v2_summary.partial.json",
        "partial_review": out_dir / "phantom_merge_v2_review_queue.partial.jsonl",
    }


def load_existing_rows(per_path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not per_path.exists():
        return rows
    with per_path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
                idx = int(row["line_index"])
                rows[idx] = row
            except Exception:
                continue
    return rows


def load_rows_from_per_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    rows.sort(key=lambda r: int(r["line_index"]))
    return rows


def finalize_from_per_jsonl(
    per_path: Path,
    out_dir: Path,
    rollout: Path,
    synthesize: Path,
    judge_cfg: JudgeConfig,
    calibration_sample: int,
    seed: int,
) -> None:
    rows = load_rows_from_per_jsonl(per_path)
    if not rows:
        raise SystemExit(f"Empty or missing per-trajectory file: {per_path}")
    summary = aggregate(rows, calibration_sample=calibration_sample, seed=seed)
    rewards_n = len(load_rewards(synthesize))
    summary["input"] = {
        "rollout": str(rollout),
        "rollout_lines": len(rows),
        "synthesize": str(synthesize),
        "rewards_loaded": rewards_n,
    }
    summary["judge"] = {
        "enabled": judge_cfg.enabled,
        "model": judge_cfg.model if judge_cfg.enabled else None,
        "base_url": judge_cfg.base_url if judge_cfg.enabled else None,
        "decoding": judge_cfg.decoding_kwargs() if judge_cfg.enabled else None,
        "workers": None,
        "min_confidence": judge_cfg.min_confidence,
        "finalize_from_per_jsonl": True,
    }
    write_outputs(rows, summary, out_dir, rollout, synthesize, judge_cfg, run_id="shopping_pm_v2_judge")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def write_checkpoint(
    rows: list[dict[str, Any]],
    out_dir: Path,
    calibration_sample: int,
    seed: int,
    total: int,
    completed: int,
) -> None:
    paths = output_paths(out_dir)
    partial = aggregate(rows, calibration_sample=calibration_sample, seed=seed)
    partial["checkpoint"] = {
        "status": "partial",
        "completed": completed,
        "total": total,
        "updated_at_unix": int(time.time()),
    }
    with paths["partial_summary"].open("w", encoding="utf-8") as f:
        json.dump(partial, f, ensure_ascii=False, indent=2)
    review_ids = set(partial["review_policy"]["review_line_indices"])
    with paths["partial_review"].open("w", encoding="utf-8") as f:
        for row in sorted(rows, key=lambda r: r["line_index"]):
            if row["line_index"] in review_ids:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def progress_bar(completed: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = int(width * completed / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def write_outputs(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    out_dir: Path,
    rollout: Path,
    synthesize: Path,
    judge_cfg: JudgeConfig,
    run_id: str = "shopping_pm_v2_judge",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(out_dir)
    per_path = paths["per"]
    summary_path = paths["summary"]
    review_path = paths["review"]
    manifest_path = paths["manifest"]

    with per_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    review_ids = set(summary["review_policy"]["review_line_indices"])
    with review_path.open("w", encoding="utf-8") as f:
        for row in rows:
            if row["line_index"] in review_ids:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "purpose": "Phantom Merge VNEXT outputs (shopping_pm_judge_vNEXT).",
        "pipeline_version": PM_PIPELINE_VERSION,
        "notes": [
            "Judge outputs use pipeline_version shopping_pm_judge_vNEXT.",
            "Sealed paper cohorts are under results/table1_characterization/.",
        ],
        "canonical_inputs": {
            "rollout": str(rollout),
            "rollout_sha256_16": sha256_short(rollout),
            "synthesize": str(synthesize),
            "synthesize_sha256_16": sha256_short(synthesize) if synthesize.exists() else None,
        },
        "canonical_outputs": {
            "per_trajectory": str(per_path),
            "summary": str(summary_path),
            "review_queue": str(review_path),
        },
        "judge": {
            "enabled": judge_cfg.enabled,
            "model": judge_cfg.model if judge_cfg.enabled else None,
            "base_url": judge_cfg.base_url if judge_cfg.enabled else None,
            "decoding": judge_cfg.decoding_kwargs() if judge_cfg.enabled else None,
        },
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if judge_cfg.enabled:
        write_run_metadata(
            out_dir / "run_metadata.json",
            build_run_metadata(
                role="judge",
                model_id=judge_cfg.model,
                decoding=judge_cfg.decoding_kwargs(),
                dataset_split="synthesize_product_test",
                run_id=run_id,
                extra={
                    "rollout": str(rollout),
                    "synthesize": str(synthesize),
                    "out_dir": str(out_dir),
                },
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout",
        type=Path,
        default=None,
        help="ShoppingBench rollout JSONL (not shipped; obtain from upstream benchmark).",
    )
    parser.add_argument(
        "--synthesize",
        type=Path,
        default=None,
        help="Optional synthesize metadata JSONL from ShoppingBench.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "shopping_pm_eval_out",
        help="Judge output directory when re-running (sealed results are under results/table1_characterization/).",
    )
    parser.add_argument("--judge-enabled", action="store_true")
    parser.add_argument("--judge-model", default=PRIMARY_JUDGE_MODEL_ID)
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--judge-api-key-env", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--judge-seed", type=int, default=DETERMINISTIC_SEED)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--run-id", default="shopping_pm_judge_vNEXT")
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=180.0,
        help="Per-request read timeout for judge HTTP calls (seconds).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Transient network/HTTP/JSON parse failures: extra attempts after the first call.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel judge threads (lower reduces timeouts when the API is slow).",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--evidence-chars", type=int, default=1200)
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--calibration-sample", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--flush-every", type=int, default=5)
    parser.add_argument("--fresh", action="store_true", help="Delete prior outputs in out-dir before running.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore prior per-trajectory rows.")
    parser.add_argument(
        "--retry-judge-errors",
        action="store_true",
        help="Drop cached trajectories whose judge.error is set so they are re-evaluated.",
    )
    parser.add_argument(
        "--repair-patch-jsonl",
        type=Path,
        default=None,
        help="Write only re-computed trajectories (pending indices) to this JSONL path.",
    )
    parser.add_argument(
        "--no-write-canonical",
        action="store_true",
        help="Do not append to or rewrite out-dir phantom_merge_v2_* canonical files (use with patch + merge).",
    )
    parser.add_argument(
        "--from-per-jsonl",
        type=Path,
        default=None,
        help="Skip judge run: load an existing per-trajectory JSONL and regenerate summary/review/manifest only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge_cfg = JudgeConfig(
        enabled=bool(args.judge_enabled),
        model=args.judge_model,
        base_url=args.judge_base_url,
        api_key_env=args.judge_api_key_env,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.judge_max_tokens,
        seed=args.judge_seed,
        timeout=args.timeout_sec,
        max_retries=args.max_retries,
        evidence_chars=args.evidence_chars,
        min_confidence=args.min_confidence,
    )
    if args.no_write_canonical and not args.repair_patch_jsonl:
        raise SystemExit("--no-write-canonical requires --repair-patch-jsonl (otherwise no outputs are written).")

    if args.rollout is None and not args.from_per_jsonl:
        raise SystemExit(
            "Provide --rollout for a new judge run, or --from-per-jsonl to rebuild "
            "summaries from an existing per-trajectory JSONL (e.g. results/table1_characterization/...)."
        )

    if args.from_per_jsonl:
        finalize_from_per_jsonl(
            args.from_per_jsonl,
            args.out_dir,
            args.rollout,
            args.synthesize,
            judge_cfg,
            calibration_sample=args.calibration_sample,
            seed=args.seed,
        )
        return

    if judge_cfg.enabled and not judge_cfg.api_key:
        raise SystemExit(f"Missing {judge_cfg.api_key_env}; export it before running judge-enabled mode.")

    lines = load_rollout_lines(args.rollout)
    if args.limit and args.limit > 0:
        lines = lines[: args.limit]
    orig_index_map = load_orig_index_map(args.rollout)
    rewards = load_rewards(args.synthesize)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(args.out_dir)
    if args.fresh:
        for path in paths.values():
            if path.exists():
                path.unlink()

    existing_by_idx = {} if args.no_resume else load_existing_rows(paths["per"])
    if args.retry_judge_errors:
        before = len(existing_by_idx)
        existing_by_idx = {
            i: r for i, r in existing_by_idx.items() if not (r.get("judge") or {}).get("error")
        }
        dropped = before - len(existing_by_idx)
        print(f"RETRY_JUDGE_ERRORS dropped {dropped} cached rows with judge.error", flush=True)

    target_indices = list(range(len(lines)))
    pending_indices = [i for i in target_indices if i not in existing_by_idx]
    retried_indices = list(pending_indices)
    rows: list[dict[str, Any]] = [existing_by_idx[i] for i in sorted(existing_by_idx) if i < len(lines)]
    start_time = time.time()
    total = len(lines)
    completed = len(rows)
    print(
        (
            f"START PhantomMergeV2 total={total} resume_completed={completed} "
            f"pending={len(pending_indices)} judge={judge_cfg.enabled} "
            f"model={judge_cfg.model if judge_cfg.enabled else 'none'} workers={args.workers} "
            f"no_write_canonical={args.no_write_canonical}"
        ),
        flush=True,
    )
    if completed:
        print(
            f"PROGRESS {progress_bar(completed, total)} {completed}/{total} "
            f"({completed / total:.1%}) resumed",
            flush=True,
        )

    append_mode = "a" if existing_by_idx and not args.fresh and not args.no_resume else "w"
    n_existing_start = len(existing_by_idx)
    with futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [
            pool.submit(
                analyze_one,
                i,
                lines[i],
                rewards,
                judge_cfg,
                orig_index=orig_index_map[i] if orig_index_map and i < len(orig_index_map) else i,
            )
            for i in pending_indices
        ]
        judge_errors = 0
        per_f = None
        try:
            if not args.no_write_canonical:
                per_f = paths["per"].open(append_mode, encoding="utf-8")
            for fut in futures.as_completed(futs):
                row = fut.result()
                rows.append(row)
                if per_f is not None:
                    per_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    per_f.flush()
                completed += 1
                if (row.get("judge") or {}).get("error"):
                    judge_errors += 1
                should_report = completed == total or completed % max(1, args.progress_every) == 0
                should_flush = completed == total or completed % max(1, args.flush_every) == 0
                if should_flush and not args.no_write_canonical:
                    write_checkpoint(
                        rows,
                        args.out_dir,
                        calibration_sample=args.calibration_sample,
                        seed=args.seed,
                        total=total,
                        completed=completed,
                    )
                if should_report:
                    elapsed = time.time() - start_time
                    done_pending = completed - n_existing_start
                    rate = done_pending / elapsed if elapsed > 0 else 0.0
                    eta = (len(pending_indices) - done_pending) / rate if rate > 0 else 0.0
                    print(
                        (
                            f"PROGRESS {progress_bar(completed, total)} {completed}/{total} "
                            f"({completed / total:.1%}) judge_errors={judge_errors} "
                            f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
                        ),
                        flush=True,
                    )
        finally:
            if per_f is not None:
                per_f.close()
    rows.sort(key=lambda r: r["line_index"])

    summary = aggregate(rows, calibration_sample=args.calibration_sample, seed=args.seed)
    summary["input"] = {
        "rollout": str(args.rollout),
        "rollout_lines": len(lines),
        "synthesize": str(args.synthesize),
        "rewards_loaded": len(rewards),
        "pipeline_version": PM_PIPELINE_VERSION,
    }
    summary["judge"] = {
        "enabled": judge_cfg.enabled,
        "model": judge_cfg.model if judge_cfg.enabled else None,
        "base_url": judge_cfg.base_url if judge_cfg.enabled else None,
        "decoding": judge_cfg.decoding_kwargs() if judge_cfg.enabled else None,
        "workers": args.workers,
        "min_confidence": judge_cfg.min_confidence,
    }

    if args.repair_patch_jsonl:
        args.repair_patch_jsonl.parent.mkdir(parents=True, exist_ok=True)
        retried_set = set(retried_indices)
        n_patch = 0
        with args.repair_patch_jsonl.open("w", encoding="utf-8") as pf:
            for row in rows:
                if row["line_index"] in retried_set:
                    pf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_patch += 1
        print(f"WROTE_REPAIR_PATCH lines={n_patch} -> {args.repair_patch_jsonl}", flush=True)

    if not args.no_write_canonical:
        write_outputs(rows, summary, args.out_dir, args.rollout, args.synthesize, judge_cfg, run_id=args.run_id)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({**summary, "note": "canonical files not updated (--no-write-canonical)"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
