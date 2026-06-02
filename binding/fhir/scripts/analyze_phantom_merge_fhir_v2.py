#!/usr/bin/env python3
"""
FHIR-AgentBench anchor–claim PM judge (VNEXT).

Shares claim taxonomy with Shopping via pm_eval_vnext_import.py.
Protocol definition: README.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pm_eval_vnext_import import (  # noqa: E402
    FHIR_PM_PIPELINE_VERSION,
    claim_label_counts_from_claims,
    compute_claim_structure_counts,
    derive_support_labels,
    is_pm_relevant_claim_fhir,
    load_fhir_slot_config,
    normalize_fhir_anchor_resolution,
    primary_support_label,
    process_fhir_claim_record,
    qa_claim_label_consistency,
    refresh_row_labels,
    trajectory_labels_from_claims,
)

PIPELINE_VERSION = FHIR_PM_PIPELINE_VERSION

BENCH = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(SCRIPTS))

from fhir_binding_pipeline import (  # noqa: E402
    compact_kb_fhir,
    extract_final_answer,
    extract_query,
    build_kb_from_row,
    infer_anchor_ids,
    parse_true_fhir_ids,
    trajectory_integrity_fhir,
    fallback_claims_from_response,
    lexical_support_ids,
    lexical_query_support,
    norm,
)

# Import judge utilities from Shopping V2 (same OpenAI-compatible contract).
SHOPPING_V2 = BENCH.parent / "shopping" / "scripts" / "analyze_phantom_merge_v2.py"
_spec = importlib.util.spec_from_file_location("shopping_pm_v2_judge", SHOPPING_V2)
_shopping = importlib.util.module_from_spec(_spec)
assert _spec.name is not None and _spec.loader is not None
# Required before exec_module: @dataclass resolves cls.__module__ via sys.modules.
sys.modules[_spec.name] = _shopping
_spec.loader.exec_module(_shopping)

JudgeConfig = _shopping.JudgeConfig
call_openai_compatible = _shopping.call_openai_compatible
safe_float = _shopping.safe_float


def _fhir_claim_policy(mitigation_closure: bool) -> list[str]:
    base = [
        "INCLUDE: checkable clinical facts (yes/no, counts, dates, values, meds, procedures, diagnoses).",
        "EXCLUDE: vague narrative, 'based on information', duplicate query paraphrase → excluded_spans.",
        "Each claim: slot + value + response_quote + anchor_resource_id = protocol anchor.",
        "Typical 2-8 claims when answer is rich; split yes/no vs numeric/temporal when both present.",
        "cross_object_merge only when another retrieved resource supports the same fact (not mere missing data).",
    ]
    if mitigation_closure:
        base.extend(
            [
                "MITIGATION CLOSURE: observed_resources lists ONLY the protocol anchor.",
                "Set cross_supported=false and query_supported=false on EVERY claim.",
                "Do not treat the user query as evidence for facts absent from the anchor resource.",
            ]
        )
    return base


def _apply_mitigation_judge_closure(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-derive labels without COM/CP channels (mitigation eval only)."""
    out: list[dict[str, Any]] = []
    for c in claims:
        row = dict(c)
        state = str(row.get("anchor_evidence_state") or "unknown")
        labels = derive_support_labels(state, False, False)
        prim = primary_support_label(labels)
        row["cross_supported"] = False
        row["query_supported"] = False
        row["support_labels"] = labels
        row["primary_support_label"] = prim
        row["support_label"] = prim
        out.append(row)
    return out


def build_fhir_judge_prompt(
    query: str,
    response: str,
    anchor_ids: list[str],
    kb: dict[str, dict[str, Any]],
    true_fhir_ids: dict[str, list[str]],
    cfg: JudgeConfig,
    *,
    mitigation_closure: bool = False,
) -> dict[str, Any]:
    protocol_anchor = anchor_ids[0] if anchor_ids else ""
    return {
        "task": "phantom_merge_fhir_vNEXT_claim_extract_and_classify",
        "pipeline_version": PIPELINE_VERSION,
        "definition": {
            "unit": "anchor-claim pair bound to protocol anchor resource",
            "protocol_anchor_rule": (
                "selected_anchor_resource_ids are the agent's committed clinical object. "
                "Judge ONLY against observed_resources excerpts. No benchmark gold leakage."
            ),
            "anchor_evidence_state": {
                "supports": "anchor resource KB agrees",
                "contradicts": "anchor KB has this clinical fact and disagrees",
                "absent": "anchor KB lacks this slot; answer still asserts it",
                "unknown": "RARE: cannot apply supports/contradicts/absent → evidence_gap",
            },
            "parallel_flags": "Set cross_supported and query_supported independently when state is contradicts or absent.",
            "derived_labels": {
                "correct_binding": "supports",
                "cross_object_merge": "(contradicts|absent) and cross_supported",
                "constraint_projection": "(contradicts|absent) and query_supported",
                "anchored_hallucination": "contradicts and not cross and not query",
                "pure_hallucination": "absent and not cross and not query (NOT phantom merge)",
                "evidence_gap": "unknown",
            },
            "has_phantom_merge": "cross OR projection OR anchored_hallucination",
        },
        "claim_policy": _fhir_claim_policy(mitigation_closure),
        "query": query,
        "protocol_anchor_resource_id": protocol_anchor,
        "selected_anchor_resource_ids": anchor_ids,
        "final_answer": response,
        "observed_resources": compact_kb_fhir(
            kb, anchor_ids, true_fhir_ids, cfg.evidence_chars, max_resources=30
        ),
        "output_schema": {
            "anchor_recoverable": "bool",
            "anchor_resolution": {
                "narrative_referent_resource_id": "string",
                "protocol_matches_narrative": "bool",
                "suspected_wrong_anchor": "bool",
                "confidence": "float",
                "rationale": "string",
            },
            "claims": [
                {
                    "anchor_resource_id": "string",
                    "claim": "string",
                    "slot": "string",
                    "value": "string",
                    "response_quote": "string",
                    "anchor_evidence_state": "supports|contradicts|absent|unknown",
                    "cross_supported": "bool",
                    "query_supported": "bool",
                    "supporting_anchor_evidence": "string",
                    "supporting_other_resource_ids": ["string"],
                    "supporting_query_text": "string",
                    "confidence": "float",
                    "review_needed": "bool",
                    "rationale": "string",
                }
            ],
            "excluded_spans": [{"response_quote": "str", "reason": "str"}],
        },
    }


def process_fhir_judge_response(
    obj: dict[str, Any],
    anchor_ids: list[str],
    query: str,
    kb: dict[str, dict[str, Any]],
    min_conf: float,
) -> tuple[bool, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    canonical, aliases = load_fhir_slot_config()
    anchor_recoverable = bool(obj.get("anchor_recoverable", bool(anchor_ids)))
    anchor_resolution = normalize_fhir_anchor_resolution(obj.get("anchor_resolution"), anchor_ids)

    claims_extracted: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for c in obj.get("claims") or []:
        if not isinstance(c, dict):
            continue
        claims_extracted.append(dict(c))
        keep, reason = is_pm_relevant_claim_fhir(c)
        if not keep:
            dropped.append(
                {"response_quote": c.get("response_quote") or c.get("claim"), "reason": reason}
            )
            continue
        row = process_fhir_claim_record(c, anchor_ids, query, kb, aliases, canonical)
        if safe_float(row.get("confidence"), 0.0) < min_conf:
            row["review_needed"] = True
        claims.append(row)

    excluded = list(obj.get("excluded_spans") or []) if isinstance(obj.get("excluded_spans"), list) else []
    excluded.extend(dropped)
    return anchor_recoverable, anchor_resolution, claims_extracted, claims, excluded


def classify_fallback_claim_fhir(
    claim: dict[str, Any], query: str, kb: dict[str, dict[str, Any]], anchor_ids: list[str]
) -> dict[str, Any]:
    anchor = str(claim.get("anchor_resource_id") or (anchor_ids[0] if anchor_ids else ""))
    value = str(claim.get("value") or "")
    quote = str(claim.get("response_quote") or claim.get("claim") or "")
    support = lexical_support_ids(kb, value, quote)
    if anchor in support:
        state, cross_b, query_b = "supports", False, False
        other: list[str] = []
    else:
        other = [rid for rid in support if rid != anchor]
        cross_b = bool(other)
        query_b = lexical_query_support(query, value, quote)
        state = "absent"
    labels = derive_support_labels(state, cross_b, query_b)
    primary = primary_support_label(labels)
    slot_raw = norm(claim.get("slot") or "clinical_attribute")
    return {
        "anchor_resource_id": anchor,
        "claim": str(claim.get("claim") or "").strip(),
        "slot_raw": slot_raw,
        "slot": slot_raw,
        "value": value,
        "response_quote": quote,
        "anchor_evidence_state": state,
        "cross_supported": cross_b,
        "query_supported": query_b,
        "support_labels": labels,
        "primary_support_label": primary,
        "support_label": primary,
        "supporting_other_resource_ids": other,
        "supporting_query_text": "",
        "confidence": safe_float(claim.get("confidence"), 0.72),
        "review_needed": bool(set(labels) & {"cross_object_merge", "constraint_projection"}),
        "rationale": str(claim.get("judge_rationale") or "lexical_fallback_fhir"),
    }


def analyze_one_fhir(
    index: int,
    row: dict[str, Any],
    task_correct: int | None,
    judge_cfg: JudgeConfig,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta = meta or {}
    qid = str(row.get("question_id", ""))
    query = extract_query(row)
    integrity = trajectory_integrity_fhir(row)
    kb = build_kb_from_row(row)
    response = extract_final_answer(str(row.get("agent_answer") or ""))
    true_fhir = parse_true_fhir_ids(meta.get("true_fhir_ids") or row.get("true_fhir_ids"))
    preset_anchor = meta.get("selected_anchor_resource_ids")
    if preset_anchor:
        anchor_ids = [str(x).strip() for x in preset_anchor if str(x).strip()]
    else:
        anchor_ids = infer_anchor_ids(
            response,
            kb,
            agent_fhir_resources=row.get("agent_fhir_resources"),
            true_fhir_ids=true_fhir,
            query=query,
            max_anchors=1,
        )
    if meta.get("kb_scope") == "anchor_only" and anchor_ids:
        allow = set(anchor_ids)
        kb = {rid: slot for rid, slot in kb.items() if rid in allow}
    fallback = [
        classify_fallback_claim_fhir(c, query, kb, anchor_ids)
        for c in fallback_claims_from_response(response, anchor_ids, query)
    ]

    judge_error = None
    judge_raw: dict[str, Any] | None = None
    anchor_resolution: dict[str, Any] = {}
    claims_extracted: list[dict[str, Any]] = []
    excluded_spans: list[dict[str, Any]] = []

    if judge_cfg.enabled and response and kb and anchor_ids:
        try:
            prompt = build_fhir_judge_prompt(
                query,
                response,
                anchor_ids,
                kb,
                true_fhir,
                judge_cfg,
                mitigation_closure=bool(meta.get("mitigation_judge_closure")),
            )
            judge_raw = call_openai_compatible(prompt, judge_cfg)
            anchor_recoverable, anchor_resolution, claims_extracted, claims, excluded_spans = (
                process_fhir_judge_response(
                    judge_raw, anchor_ids, query, kb, judge_cfg.min_confidence
                )
            )
            if len(claims) < 2 and fallback:
                seen = {norm(c.get("claim")) for c in claims}
                for fb in fallback:
                    if norm(fb.get("claim")) not in seen:
                        claims.append(fb)
                        seen.add(norm(fb.get("claim")))
        except Exception as exc:  # noqa: BLE001
            judge_error = str(exc)[:500]
            anchor_recoverable = bool(anchor_ids)
            anchor_resolution = normalize_fhir_anchor_resolution(None, anchor_ids)
            claims_extracted = list(fallback)
            claims = list(fallback)
    else:
        anchor_recoverable = bool(anchor_ids)
        anchor_resolution = normalize_fhir_anchor_resolution(None, anchor_ids)
        claims_extracted = list(fallback)
        claims = list(fallback)

    if meta.get("mitigation_judge_closure"):
        claims = _apply_mitigation_judge_closure(claims)

    traj_labels = trajectory_labels_from_claims(claims)
    primary_counts, multiset_counts = claim_label_counts_from_claims(claims)
    structure_counts = compute_claim_structure_counts(claims)
    qa_issues = qa_claim_label_consistency(claims)

    review_reasons: list[str] = []
    if not anchor_recoverable:
        review_reasons.append("anchor_unrecoverable")
    if anchor_resolution.get("suspected_wrong_anchor"):
        review_reasons.append("suspected_wrong_anchor")
    if structure_counts.get("n_claims_cross_and_projection", 0) > 0:
        review_reasons.append("anchor_may_be_wrong")
    if traj_labels["has_phantom_merge"]:
        review_reasons.append("phantom_merge")
    if traj_labels["has_phantom_merge"] and task_correct == 1:
        review_reasons.append("task_correct_binding_failure")
    if judge_error:
        review_reasons.append("judge_error")
    if not claims and response:
        review_reasons.append("no_claims_extracted")
    if qa_issues:
        review_reasons.append("qa_label_inconsistency")

    priority = "none"
    if review_reasons:
        priority = "high" if traj_labels["has_phantom_merge"] or not anchor_recoverable else "calibration"

    return {
        "pipeline_version": PIPELINE_VERSION,
        "line_index": index,
        "question_id": qid,
        "primary_stratum": meta.get("primary_stratum"),
        "query": query[:300],
        "selected_anchor_resource_ids": anchor_ids,
        "task_correct_llm": task_correct,
        "anchor_recoverable": anchor_recoverable,
        "anchor_resolution": anchor_resolution,
        "final_answer": response[:800],
        "integrity": integrity,
        "exposure": {"observed_resources": len(kb), "anchor_in_kb": all(a in kb for a in anchor_ids)},
        "claims_extracted": claims_extracted,
        "claims": claims,
        "excluded_spans": excluded_spans,
        "trajectory_labels": traj_labels,
        "claim_label_counts": primary_counts,
        "claim_label_multiset_counts": multiset_counts,
        "claim_structure_counts": structure_counts,
        "qa_claim_issues": qa_issues,
        "review": {
            "priority": priority,
            "reasons": sorted(set(review_reasons)),
        },
        "judge_artifacts": {"raw_json": judge_raw},
        "judge": {
            "enabled": judge_cfg.enabled,
            "model": judge_cfg.model if judge_cfg.enabled else None,
            "error": judge_error,
        },
    }


def _binding_bucket(r: dict[str, Any]) -> str:
    tl = r.get("trajectory_labels") or {}
    if tl.get("has_phantom_merge"):
        return "phantom_merge"
    if tl.get("has_pure_hallucination"):
        return "pure_hallucination_only"
    if tl.get("has_anchored_hallucination"):
        return "anchored_hallucination_only"
    if r.get("claims"):
        return "binding_clean_or_correct"
    return "no_checkable_claims"


def aggregate_fhir(
    rows: list[dict[str, Any]],
    *,
    calibration_sample: int = 25,
    seed: int = 42,
) -> dict[str, Any]:
    """VNEXT summary stats (counts + rates + cross-tabs + claim structure)."""
    rows = [refresh_row_labels(dict(r)) for r in rows]
    n = len(rows)

    def rate(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    complete = [r for r in rows if r.get("integrity", {}).get("trajectory_complete")]
    recoverable = [r for r in rows if r.get("anchor_recoverable")]

    claim_counts: Counter[str] = Counter()
    claim_multiset: Counter[str] = Counter()
    structure_totals: Counter[str] = Counter()
    claims_per_traj: list[int] = []
    exposure_sizes: list[int] = []
    for r in rows:
        claim_counts.update(r.get("claim_label_counts") or {})
        claim_multiset.update(r.get("claim_label_multiset_counts") or {})
        structure_totals.update(r.get("claim_structure_counts") or {})
        claims_per_traj.append(len(r.get("claims") or []))
        exposure_sizes.append(int((r.get("exposure") or {}).get("observed_resources") or 0))

    traj_counts = {
        "has_cross_object_merge": sum(r["trajectory_labels"]["has_cross_object_merge"] for r in rows),
        "has_constraint_projection": sum(r["trajectory_labels"]["has_constraint_projection"] for r in rows),
        "has_anchored_hallucination": sum(r["trajectory_labels"]["has_anchored_hallucination"] for r in rows),
        "has_pure_hallucination": sum(r["trajectory_labels"].get("has_pure_hallucination", False) for r in rows),
        "has_evidence_gap": sum(r["trajectory_labels"].get("has_evidence_gap", False) for r in rows),
        "has_phantom_merge": sum(r["trajectory_labels"]["has_phantom_merge"] for r in rows),
    }

    by_task_binding: dict[str, int] = defaultdict(int)
    for r in rows:
        tc = r.get("task_correct_llm")
        task_bucket = "task_correct" if tc == 1 else "task_incorrect" if tc == 0 else "task_unknown"
        by_task_binding[f"{task_bucket}__{_binding_bucket(r)}"] += 1

    by_stratum: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "pm": 0, "task_ok": 0, "cross": 0, "proj": 0, "hall": 0}
    )
    for r in rows:
        st = str(r.get("primary_stratum") or "unknown")
        b = by_stratum[st]
        b["n"] += 1
        if r["trajectory_labels"]["has_phantom_merge"]:
            b["pm"] += 1
        if r.get("task_correct_llm") == 1:
            b["task_ok"] += 1
        if r["trajectory_labels"]["has_cross_object_merge"]:
            b["cross"] += 1
        if r["trajectory_labels"]["has_constraint_projection"]:
            b["proj"] += 1
        if r["trajectory_labels"]["has_anchored_hallucination"]:
            b["hall"] += 1

    for st, b in by_stratum.items():
        b["pm_rate"] = rate(b["pm"], b["n"])
        b["task_acc"] = rate(b["task_ok"], b["n"])

    high_review = [r for r in rows if r.get("review", {}).get("priority") == "high"]
    calibration_pool = [r for r in rows if r.get("review", {}).get("priority") == "calibration"]
    rng = random.Random(seed)
    calibration_ids = [
        r["line_index"] for r in rng.sample(calibration_pool, min(calibration_sample, len(calibration_pool)))
    ]
    review_ids = sorted({r["line_index"] for r in high_review} | set(calibration_ids))

    total_claims = sum(claim_counts.values())
    return {
        "pipeline_version": PIPELINE_VERSION,
        "num_trajectories": n,
        "trajectory_complete": len(complete),
        "trajectory_incomplete": n - len(complete),
        "anchor_recoverable": len(recoverable),
        "anchor_recovery_rate": rate(len(recoverable), n),
        "task_correct_llm": sum(r.get("task_correct_llm") == 1 for r in rows),
        "task_incorrect_llm": sum(r.get("task_correct_llm") == 0 for r in rows),
        "task_unknown_llm": sum(r.get("task_correct_llm") not in (0, 1) for r in rows),
        "task_accuracy_llm": rate(sum(r.get("task_correct_llm") == 1 for r in rows), n),
        "claim_label_counts": dict(claim_counts),
        "claim_label_rates": {k: rate(v, total_claims) for k, v in sorted(claim_counts.items())},
        "claim_label_multiset_counts": dict(claim_multiset),
        "claim_structure_counts": dict(structure_totals),
        "claims_per_trajectory": {
            "mean": round(sum(claims_per_traj) / n, 2) if n else 0,
            "min": min(claims_per_traj) if claims_per_traj else 0,
            "max": max(claims_per_traj) if claims_per_traj else 0,
            "distribution": dict(Counter(claims_per_traj)),
        },
        "exposure_observed_resources": {
            "mean": round(sum(exposure_sizes) / n, 1) if n else 0,
            "median": sorted(exposure_sizes)[n // 2] if exposure_sizes else 0,
        },
        "trajectory_label_counts": traj_counts,
        "trajectory_label_rates": {k: rate(v, n) for k, v in traj_counts.items()},
        "task_correctness_x_binding": dict(sorted(by_task_binding.items())),
        "by_primary_stratum": dict(sorted(by_stratum.items(), key=lambda x: -x[1]["n"])),
        "review_policy": {
            "high_priority_cases": len(high_review),
            "calibration_pool_cases": len(calibration_pool),
            "sampled_calibration_cases": len(calibration_ids),
            "total_review_cases": len(review_ids),
            "total_review_rate": rate(len(review_ids), n),
            "review_line_indices": review_ids[:50],
        },
        "judge_errors": sum(1 for r in rows if (r.get("judge") or {}).get("error")),
    }


def write_review_queue(rows: list[dict], review_path: Path, limit: int = 40) -> None:
    review_rows = [
        r
        for r in rows
        if r["trajectory_labels"]["has_phantom_merge"] or r.get("review", {}).get("priority") == "high"
    ]
    review_rows.sort(
        key=lambda r: (
            0 if r["trajectory_labels"]["has_phantom_merge"] else 1,
            -(len(r.get("claims") or [])),
        )
    )
    with review_path.open("w", encoding="utf-8") as f:
        for r in review_rows[:limit]:
            f.write(
                json.dumps(
                    {
                        "question_id": r["question_id"],
                        "primary_stratum": r.get("primary_stratum"),
                        "task_correct_llm": r.get("task_correct_llm"),
                        "trajectory_labels": r.get("trajectory_labels"),
                        "claim_label_counts": r.get("claim_label_counts"),
                        "final_answer": r.get("final_answer"),
                        "anchor_ids": r.get("selected_anchor_resource_ids"),
                        "sample_claims": (r.get("claims") or [])[:5],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_task_correct(eval_path: Path) -> dict[str, int]:
    rows = json.loads(eval_path.read_text(encoding="utf-8"))
    return {str(r["question_id"]): int(r.get("answer_correctness", 0)) for r in rows}


def load_meta_csv(csv_path: Path) -> dict[str, dict]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    return {str(r["question_id"]): r.to_dict() for _, r in df.iterrows()}


def select_cohort(
    rows: list[dict],
    per_by_qid: dict[str, dict],
    task_correct: dict[str, int],
    cohort: str,
    max_rows: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)

    def pm_eligible(r: dict) -> bool:
        p = per_by_qid.get(str(r.get("question_id")), {})
        return p.get("qa_layer") == "pm_eligible"

    if cohort == "strict_bf":
        ids = [q for q, p in per_by_qid.items() if p.get("binding_failure_strict")]
        pool = [r for r in rows if str(r["question_id"]) in ids]
    elif cohort == "meeting":
        strict = [r for r in rows if per_by_qid.get(str(r["question_id"]), {}).get("binding_failure_strict")]
        correct = [
            r
            for r in rows
            if pm_eligible(r) and task_correct.get(str(r["question_id"])) == 1
        ]
        rng.shuffle(strict)
        rng.shuffle(correct)
        pool = strict[:80] + correct[:40]
    elif cohort == "pm_eligible_sample":
        pool = [r for r in rows if pm_eligible(r)]
    else:
        pool = [r for r in rows if pm_eligible(r)]

    if max_rows > 0 and len(pool) > max_rows:
        rng.shuffle(pool)
        pool = pool[:max_rows]
    return pool


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rollout",
        type=Path,
        default=None,
        help="FHIR agent rollout JSON (from FHIR-AgentBench; not shipped in this repo).",
    )
    ap.add_argument(
        "--eval-json",
        type=Path,
        default=None,
        help="Optional task-eval JSON for task_correct_llm.",
    )
    ap.add_argument(
        "--per-jsonl",
        type=Path,
        default=REPO_ROOT
        / "results/table1_characterization/fhir_qwen_n973/per_trajectory.jsonl",
        help="Existing per-trajectory output to aggregate (default: sealed Qwen cohort).",
    )
    ap.add_argument(
        "--rivals-csv",
        type=Path,
        default=None,
        help="Optional PM-subset task list CSV from FHIR-AgentBench.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "runs/fhir_pm_eval_out",
        help="Output directory when re-running judge (requires rollout + judge).",
    )
    ap.add_argument(
        "--cohort",
        choices=["pm_eligible", "strict_bf", "meeting", "pm_eligible_sample"],
        default="pm_eligible",
        help="pm_eligible=full eligible cohort; pm_eligible_sample=random subsample (needs --max-rows>0)",
    )
    ap.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="0 = no cap (full cohort); >0 = random subsample of that size",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-judge", action="store_true", help="Skip LLM judge (structure-only).")
    ap.add_argument("--judge-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--judge-base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--judge-max-tokens", type=int, default=4096)
    ap.add_argument("--judge-seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout-sec", type=float, default=180.0)
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    judge_enabled = not args.no_judge

    judge_cfg = JudgeConfig(
        enabled=judge_enabled,
        model=args.judge_model,
        base_url=args.judge_base_url,
        api_key_env=args.judge_api_key_env,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.judge_max_tokens,
        seed=args.judge_seed,
        timeout=args.timeout_sec,
        max_retries=3,
        evidence_chars=1200,
        min_confidence=args.min_confidence,
    )
    if judge_cfg.enabled and not judge_cfg.api_key:
        raise SystemExit(f"Missing {judge_cfg.api_key_env}; export OPENAI_API_KEY=EMPTY for vLLM.")

    if args.rollout is None:
        if not args.per_jsonl.is_file():
            raise SystemExit(
                "Provide --rollout for a new judge run, or an existing --per-jsonl "
                "(default: results/table1_characterization/fhir_qwen_n973/...)."
            )
        rows_out = [
            json.loads(line)
            for line in args.per_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary = aggregate_fhir(rows_out, calibration_sample=25, seed=args.seed)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    all_rows = json.loads(args.rollout.read_text(encoding="utf-8"))
    task_correct = load_task_correct(args.eval_json) if args.eval_json.exists() else {}
    per_by_qid = {}
    if args.per_jsonl.exists():
        for line in args.per_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                per_by_qid[r["question_id"]] = r
    meta_by_qid = load_meta_csv(args.rivals_csv) if args.rivals_csv.exists() else {}

    cohort_rows = select_cohort(
        all_rows, per_by_qid, task_correct, args.cohort, args.max_rows, args.seed
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_path = args.out_dir / "phantom_merge_v2_fhir_per_trajectory.jsonl"
    summary_path = args.out_dir / "phantom_merge_v2_fhir_summary.json"
    partial_summary_path = args.out_dir / "phantom_merge_v2_fhir_summary.partial.json"
    review_path = args.out_dir / "phantom_merge_v2_fhir_review_queue.jsonl"
    partial_review_path = args.out_dir / "phantom_merge_v2_fhir_review_queue.partial.jsonl"

    if args.fresh and per_path.exists():
        per_path.unlink()

    existing: dict[int, dict] = {}
    if per_path.exists():
        for line in per_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                existing[r["line_index"]] = r

    pending = [(i, row) for i, row in enumerate(cohort_rows) if i not in existing]
    print(
        f"FHIR PM judge: cohort={args.cohort} n={len(cohort_rows)} pending={len(pending)} "
        f"judge={judge_cfg.model} @ {judge_cfg.base_url}",
        flush=True,
    )

    def work(item: tuple[int, dict]) -> dict:
        i, row = item
        qid = str(row["question_id"])
        return analyze_one_fhir(
            i,
            row,
            task_correct.get(qid),
            judge_cfg,
            meta_by_qid.get(qid),
        )

    rows_out = list(existing.values())

    def enrich_summary(summary: dict[str, Any]) -> dict[str, Any]:
        summary["cohort"] = args.cohort
        summary["max_rows"] = args.max_rows
        summary["progress_completed"] = len(rows_out)
        summary["progress_total"] = len(cohort_rows)
        summary["rollout"] = str(args.rollout)
        summary["eval_json"] = str(args.eval_json)
        summary["judge"] = {
            "model": judge_cfg.model,
            "base_url": judge_cfg.base_url,
            "enabled": judge_cfg.enabled,
        }
        summary["full_eval_task_accuracy"] = (
            round(sum(task_correct.values()) / len(task_correct), 4) if task_correct else None
        )
        return summary

    def write_partial() -> None:
        if not rows_out:
            return
        sorted_rows = sorted(rows_out, key=lambda r: r["line_index"])
        partial = enrich_summary(aggregate_fhir(sorted_rows, calibration_sample=25, seed=args.seed))
        partial_summary_path.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
        write_review_queue(sorted_rows, partial_review_path)

    if pending:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(work, item) for item in pending]
            for fut in futures.as_completed(futs):
                row = fut.result()
                rows_out.append(row)
                with per_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_done = len(rows_out)
                tl = row["trajectory_labels"]
                print(
                    f"  done {n_done}/{len(cohort_rows)} "
                    f"pm={tl['has_phantom_merge']} "
                    f"cross={tl['has_cross_object_merge']} "
                    f"proj={tl['has_constraint_projection']} "
                    f"claims={len(row.get('claims') or [])}",
                    flush=True,
                )
                if n_done % 10 == 0 or n_done == len(cohort_rows):
                    write_partial()
                    pm_n = sum(r["trajectory_labels"]["has_phantom_merge"] for r in rows_out)
                    print(
                        f"  [partial] pm_rate={pm_n}/{n_done}="
                        f"{pm_n/n_done:.1%} -> {partial_summary_path.name}",
                        flush=True,
                    )

    rows_out.sort(key=lambda r: r["line_index"])
    summary = enrich_summary(aggregate_fhir(rows_out, calibration_sample=25, seed=args.seed))
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_review_queue(rows_out, review_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"Wrote {per_path}\n"
        f"      {summary_path}\n"
        f"      {partial_summary_path} (last partial)\n"
        f"      {review_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
