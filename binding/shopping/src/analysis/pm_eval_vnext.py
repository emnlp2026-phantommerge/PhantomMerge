"""
Phantom Merge VNEXT taxonomy — Shopping (and shared helpers).

See README.md for label definitions.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

PM_PIPELINE_VERSION = "shopping_pm_judge_vNEXT"

# Five paper-facing outcome labels + evidence_gap (evaluator abstain).
OUTCOME_LABELS = frozenset(
    {
        "correct_binding",
        "cross_object_merge",
        "constraint_projection",
        "anchored_hallucination",
        "pure_hallucination",
        "evidence_gap",
    }
)
PM_FAILURE_LABELS = frozenset(
    {
        "cross_object_merge",
        "constraint_projection",
        "anchored_hallucination",
    }
)
ANCHOR_STATES = frozenset({"supports", "contradicts", "absent", "unknown"})

PRIMARY_LABEL_PRIORITY = (
    "cross_object_merge",
    "constraint_projection",
    "anchored_hallucination",
    "pure_hallucination",
    "correct_binding",
    "evidence_gap",
)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "pm_slot_aliases.json"
_FHIR_CONFIG_PATH = Path(__file__).resolve().parents[2].parent / "fhir" / "config" / "pm_fhir_slot_aliases.json"

# Spans that are not checkable product facts (go to excluded_spans, not claims).
_SUBJECTIVE_RX = re.compile(
    r"\b(high quality|perfect|great choice|best|ideal|suitable for|recommended for you|"
    r"meets your (needs|requirements)|good choice|excellent|premium quality)\b",
    re.I,
)


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def load_slot_config(path: Path | None = None) -> tuple[frozenset[str], dict[str, str]]:
    p = path or _CONFIG_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    canonical = frozenset(_norm(x) for x in (data.get("canonical_slots") or []) if x)
    aliases = {_norm(k): _norm(v) for k, v in (data.get("aliases") or {}).items()}
    return canonical, aliases


def load_fhir_slot_config(path: Path | None = None) -> tuple[frozenset[str], dict[str, str]]:
    p = path or _FHIR_CONFIG_PATH
    data = json.loads(p.read_text(encoding="utf-8"))
    canonical = frozenset(_norm(x) for x in (data.get("canonical_slots") or []) if x)
    aliases = {_norm(k): _norm(v) for k, v in (data.get("aliases") or {}).items()}
    return canonical, aliases


def canonicalize_slot(slot_raw: str, aliases: dict[str, str], canonical: frozenset[str]) -> tuple[str, str]:
    raw = _norm(slot_raw) or "unknown"
    if raw in aliases:
        can = aliases[raw]
    elif raw in canonical:
        can = raw
    else:
        can = raw
    return raw, can


def derive_support_labels(
    anchor_evidence_state: str,
    cross_supported: bool,
    query_supported: bool,
) -> list[str]:
    """
    Deterministic label set per claim. Mutual exclusivity holds except
    cross_object_merge + constraint_projection may co-occur.
    """
    state = _norm(anchor_evidence_state)
    if state == "supports":
        return ["correct_binding"]
    if state in {"", "unknown", "evidence_gap"}:
        return ["evidence_gap"]

    labels: list[str] = []
    if cross_supported:
        labels.append("cross_object_merge")
    if query_supported:
        labels.append("constraint_projection")
    if labels:
        return labels

    if state == "contradicts":
        return ["anchored_hallucination"]
    if state == "absent":
        return ["pure_hallucination"]
    return ["evidence_gap"]


_CLINICAL_SUBJECTIVE_RX = re.compile(
    r"\b(based on (the )?information|no (records|documentation)|not documented|"
    r"appears to|seems to|likely|may have|unclear from|cannot determine from the answer alone)\b",
    re.I,
)


def is_pm_relevant_claim_fhir(raw: dict[str, Any]) -> tuple[bool, str]:
    """Clinical FHIR claims: must be checkable against anchor resource KB."""
    quote = str(raw.get("response_quote") or "").strip()
    claim = str(raw.get("claim") or "").strip()
    value = str(raw.get("value") or "").strip()
    slot = _norm(raw.get("slot") or raw.get("slot_raw") or "")

    if not quote and not claim:
        return False, "empty_claim"
    text = quote or claim
    if _CLINICAL_SUBJECTIVE_RX.search(text) and not value and len(claim) < 14:
        return False, "clinical_vague"
    if slot in {"", "unknown", "other", "general"} and not value:
        return False, "no_checkable_slot_or_value"
    if len(value) < 2 and len(claim) < 8:
        return False, "too_short_to_verify"
    return True, ""


def is_pm_relevant_claim(raw: dict[str, Any]) -> tuple[bool, str]:
    """
    Actionable = checkable factual assertion about the anchor product (PM-relevant).
    Returns (keep, drop_reason).
    """
    quote = str(raw.get("response_quote") or "").strip()
    claim = str(raw.get("claim") or "").strip()
    value = str(raw.get("value") or "").strip()
    slot = _norm(raw.get("slot") or raw.get("slot_raw") or "")

    if not quote and not claim:
        return False, "empty_claim"
    text = quote or claim
    if _SUBJECTIVE_RX.search(text) and not value and len(claim) < 12:
        return False, "subjective_or_vague"
    if slot in {"", "unknown", "other", "general"} and not value:
        return False, "no_checkable_slot_or_value"
    if len(value) < 2 and len(claim) < 6:
        return False, "too_short_to_verify"
    return True, ""


def evidence_gap_only_when(
    state: str,
    raw: dict[str, Any],
) -> str:
    """
    Anti-laziness: if judge said unknown but left verification hooks, downgrade to absent
    so derive can still reach pure/cross/proj (not a free-form dump bin).
    """
    if state != "unknown":
        return state
    ev = str(raw.get("supporting_anchor_evidence") or "").strip()
    if len(ev) >= 8:
        return "absent"
    if raw.get("supporting_other_product_ids"):
        return "absent"
    return "unknown"


def primary_support_label(support_labels: list[str]) -> str:
    for pref in PRIMARY_LABEL_PRIORITY:
        if pref in support_labels:
            return pref
    return support_labels[0] if support_labels else "evidence_gap"


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = _norm(value)
    if s in {"true", "yes", "1"}:
        return True
    if s in {"false", "no", "0"}:
        return False
    return None


def normalize_anchor_evidence_state(raw: Any) -> str:
    s = _norm(raw)
    if s in {"support", "supported", "supports", "match", "matches", "consistent"}:
        return "supports"
    if s in {"contradict", "contradicts", "contradiction", "conflict", "conflicts", "refute", "refutes"}:
        return "contradicts"
    if s in {"absent", "missing", "not_present", "no_evidence", "none"}:
        return "absent"
    if s in {"unknown", "unclear", "ambiguous", "unverifiable", "evidence_gap"}:
        return "unknown"
    return "unknown"


def infer_flags_from_legacy_label(
    support_label: str,
    supporting_other_ids: list[str],
    supporting_query_text: str,
    query: str,
    value: str,
    quote: str,
    kb: dict[str, dict[str, Any]],
    anchor_id: str,
    *,
    kb_blob_key: str = "evidence_blob",
) -> tuple[str, bool, bool]:
    """Fallback when judge omits structured fields."""
    label = _norm(support_label)
    other = [p for p in supporting_other_ids if p and p != anchor_id]
    cross = bool(other)
    if not cross and kb:
        needles = [n for n in {_norm(value), _norm(quote)} if len(n) >= 3]
        hits = []
        for rid, slot in kb.items():
            blob = _norm(slot.get(kb_blob_key, "") or slot.get("evidence_blob", ""))
            if any(n in blob for n in needles):
                hits.append(rid)
        cross = any(p != anchor_id for p in hits)
        if not other and cross:
            other = [p for p in hits if p != anchor_id]

    qtext = _norm(supporting_query_text)
    query_sup = bool(qtext) or (
        len(_norm(query)) >= 3
        and any(n in _norm(query) for n in {_norm(value), _norm(quote)} if len(n) >= 3)
    )

    if label == "correct_binding":
        return "supports", False, False
    if label == "cross_object_merge":
        return "contradicts", True, query_sup
    if label == "constraint_projection":
        return "absent", cross, True
    if label == "anchored_hallucination":
        return "contradicts", False, False
    if label == "pure_hallucination":
        return "absent", False, False
    return "unknown", cross, query_sup


def process_claim_record(
    raw: dict[str, Any],
    anchor_ids: list[str],
    query: str,
    kb: dict[str, dict[str, Any]],
    aliases: dict[str, str],
    canonical: frozenset[str],
    *,
    anchor_field: str = "anchor_pid",
    other_ids_field: str = "supporting_other_product_ids",
    kb_blob_key: str = "evidence_blob",
) -> dict[str, Any]:
    anchor = str(raw.get(anchor_field) or raw.get("anchor_pid") or raw.get("anchor_resource_id") or "").strip()
    if not anchor and len(anchor_ids) == 1:
        anchor = anchor_ids[0]

    slot_raw, slot_canonical = canonicalize_slot(str(raw.get("slot") or raw.get("slot_raw") or ""), aliases, canonical)

    state = normalize_anchor_evidence_state(raw.get("anchor_evidence_state"))
    state = evidence_gap_only_when(state, raw)
    cross_b = _parse_bool(raw.get("cross_supported"))
    query_b = _parse_bool(raw.get("query_supported"))

    if state == "unknown" and cross_b is None and query_b is None:
        legacy = _norm(raw.get("support_label"))
        if legacy == "unverifiable":
            legacy = "evidence_gap"
        state, cross_b, query_b = infer_flags_from_legacy_label(
            legacy or "evidence_gap",
            [str(x) for x in (raw.get(other_ids_field) or raw.get("supporting_other_product_ids") or [])],
            str(raw.get("supporting_query_text") or ""),
            query,
            str(raw.get("value") or ""),
            str(raw.get("response_quote") or raw.get("claim") or ""),
            kb,
            anchor,
            kb_blob_key=kb_blob_key,
        )
    cross_supported = bool(cross_b)
    query_supported = bool(query_b)

    support_labels = derive_support_labels(state, cross_supported, query_supported)
    primary = primary_support_label(support_labels)

    review_needed = bool(raw.get("review_needed", False))
    if state == "contradicts" and cross_supported and query_supported:
        review_needed = True

    out = {
        "anchor_pid": anchor,
        "anchor_resource_id": anchor,
        "claim": str(raw.get("claim") or "").strip(),
        "slot_raw": slot_raw,
        "slot": slot_canonical,
        "value": str(raw.get("value") or "").strip(),
        "response_quote": str(raw.get("response_quote") or "").strip(),
        "anchor_evidence_state": state,
        "cross_supported": cross_supported,
        "query_supported": query_supported,
        "support_labels": support_labels,
        "primary_support_label": primary,
        "support_label": primary,
        "supporting_anchor_evidence": str(raw.get("supporting_anchor_evidence") or "").strip(),
        "supporting_other_product_ids": [
            str(x) for x in (raw.get(other_ids_field) or raw.get("supporting_other_product_ids") or [])
        ],
        "supporting_other_resource_ids": [
            str(x) for x in (raw.get(other_ids_field) or raw.get("supporting_other_resource_ids") or [])
        ],
        "supporting_query_text": str(raw.get("supporting_query_text") or "").strip(),
        "confidence": float(raw.get("confidence") or 0.0) if str(raw.get("confidence", "")).strip() else 0.0,
        "review_needed": review_needed,
        "rationale": str(raw.get("rationale") or "").strip(),
        "judge_support_label_raw": str(raw.get("support_label") or "").strip(),
    }
    return out


def normalize_fhir_anchor_resolution(
    raw: dict[str, Any] | None,
    protocol_anchor_ids: list[str],
) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    protocol = protocol_anchor_ids[0] if protocol_anchor_ids else ""
    narrative = str(raw.get("narrative_referent_resource_id") or raw.get("narrative_referent_pid") or "").strip()
    matches = raw.get("protocol_matches_narrative")
    if matches is None:
        matches = (not narrative) or (narrative in protocol_anchor_ids)
    return {
        "protocol_anchor_resource_ids": protocol_anchor_ids,
        "protocol_anchor_resource_id": protocol,
        "narrative_referent_resource_id": narrative or protocol,
        "protocol_matches_narrative": bool(matches),
        "suspected_wrong_anchor": bool(raw.get("suspected_wrong_anchor", False)),
        "confidence": float(raw.get("confidence") or 0.0) if str(raw.get("confidence", "")).strip() else 0.0,
        "rationale": str(raw.get("rationale") or "").strip(),
    }


def process_fhir_claim_record(
    raw: dict[str, Any],
    anchor_ids: list[str],
    query: str,
    kb: dict[str, dict[str, Any]],
    aliases: dict[str, str],
    canonical: frozenset[str],
) -> dict[str, Any]:
    return process_claim_record(
        raw,
        anchor_ids,
        query,
        kb,
        aliases,
        canonical,
        anchor_field="anchor_resource_id",
        other_ids_field="supporting_other_resource_ids",
        kb_blob_key="evidence_blob",
    )


def compute_claim_structure_counts(claims: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {
        "n_claims_total": len(claims),
        "n_claims_cross_only": 0,
        "n_claims_projection_only": 0,
        "n_claims_cross_and_projection": 0,
        "n_claims_anchored_hall_only": 0,
        "n_claims_pure_only": 0,
        "n_claims_evidence_gap": 0,
        "n_claims_correct_only": 0,
        "n_claims_multi_pm_label": 0,
    }
    for c in claims:
        labels = set(c.get("support_labels") or [])
        pm = labels & PM_FAILURE_LABELS
        if len(pm) > 1:
            counts["n_claims_multi_pm_label"] += 1
        has_cross = "cross_object_merge" in labels
        has_proj = "constraint_projection" in labels
        if has_cross and has_proj:
            counts["n_claims_cross_and_projection"] += 1
        elif has_cross:
            counts["n_claims_cross_only"] += 1
        elif has_proj:
            counts["n_claims_projection_only"] += 1
        elif "anchored_hallucination" in labels:
            counts["n_claims_anchored_hall_only"] += 1
        elif "pure_hallucination" in labels:
            counts["n_claims_pure_only"] += 1
        elif "evidence_gap" in labels:
            counts["n_claims_evidence_gap"] += 1
        elif "correct_binding" in labels:
            counts["n_claims_correct_only"] += 1
    return counts


def trajectory_labels_from_claims(claims: list[dict[str, Any]]) -> dict[str, bool]:
    any_cross = any("cross_object_merge" in (c.get("support_labels") or []) for c in claims)
    any_proj = any("constraint_projection" in (c.get("support_labels") or []) for c in claims)
    any_anchored = any("anchored_hallucination" in (c.get("support_labels") or []) for c in claims)
    any_pure = any("pure_hallucination" in (c.get("support_labels") or []) for c in claims)
    any_gap = any("evidence_gap" in (c.get("support_labels") or []) for c in claims)
    return {
        "has_cross_object_merge": any_cross,
        "has_constraint_projection": any_proj,
        "has_anchored_hallucination": any_anchored,
        "has_pure_hallucination": any_pure,
        "has_evidence_gap": any_gap,
        "has_phantom_merge": any_cross or any_proj or any_anchored,
    }


def claim_label_counts_from_claims(claims: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    primary = Counter(c.get("primary_support_label") or c.get("support_label") for c in claims)
    multiset: Counter[str] = Counter()
    for c in claims:
        for lab in c.get("support_labels") or []:
            multiset[lab] += 1
    return dict(primary), dict(multiset)


def refresh_row_labels(row: dict[str, Any]) -> dict[str, Any]:
    claims = row.get("claims") or []
    row["trajectory_labels"] = trajectory_labels_from_claims(claims)
    primary, multiset = claim_label_counts_from_claims(claims)
    row["claim_label_counts"] = primary
    row["claim_label_multiset_counts"] = multiset
    row["claim_structure_counts"] = compute_claim_structure_counts(claims)
    return row


def normalize_anchor_resolution(
    raw: dict[str, Any] | None,
    protocol_pid: str,
    anchor_pids: list[str],
) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    narrative = str(raw.get("narrative_referent_pid") or raw.get("narrative_pid") or "").strip()
    text_pid = str(raw.get("narrative_text_pid") or "").strip() or None
    matches = raw.get("protocol_matches_narrative")
    if matches is None:
        matches = (not narrative) or (narrative == protocol_pid)
    suspected = bool(raw.get("suspected_wrong_anchor", False))
    return {
        "protocol_anchor_pid": protocol_pid,
        "protocol_anchor_pids": anchor_pids,
        "narrative_referent_pid": narrative or protocol_pid,
        "narrative_text_pid": text_pid,
        "protocol_matches_narrative": bool(matches),
        "suspected_wrong_anchor": suspected,
        "confidence": float(raw.get("confidence") or 0.0) if str(raw.get("confidence", "")).strip() else 0.0,
        "rationale": str(raw.get("rationale") or "").strip(),
    }


def detect_protocol_narrative_pid_mismatch(final_answer: str, protocol_pid: str) -> dict[str, Any]:
    pid_re = re.compile(r"\b(\d{8,12})\b")
    pids = pid_re.findall(final_answer or "")
    first = pids[0] if pids else None
    mismatch = bool(first and protocol_pid and first != protocol_pid)
    return {
        "final_answer_first_pid": first,
        "protocol_anchor_pid": protocol_pid,
        "protocol_text_pid_mismatch": mismatch,
    }


def qa_claim_label_consistency(claims: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    allowed_pairs = frozenset({frozenset({"cross_object_merge", "constraint_projection"})})
    for i, c in enumerate(claims):
        labels = set(c.get("support_labels") or [])
        state = c.get("anchor_evidence_state")
        if "correct_binding" in labels and state not in {"supports", ""}:
            issues.append(f"claim[{i}]: correct_binding but anchor_state={state}")
        if "pure_hallucination" in labels and ("cross_object_merge" in labels or "constraint_projection" in labels):
            issues.append(f"claim[{i}]: pure_hallucination with cross/projection")
        if "anchored_hallucination" in labels and ("cross_object_merge" in labels or "constraint_projection" in labels):
            issues.append(f"claim[{i}]: anchored_hallucination with cross/projection")
        if "evidence_gap" in labels and PM_FAILURE_LABELS & labels:
            issues.append(f"claim[{i}]: evidence_gap mixed with PM")
        if "correct_binding" in labels and len(labels) > 1:
            issues.append(f"claim[{i}]: correct_binding with other labels")
        if len(labels) > 2 or (len(labels) == 2 and labels not in allowed_pairs):
            issues.append(f"claim[{i}]: illegal label combination {sorted(labels)}")
        if state == "supports" and PM_FAILURE_LABELS & labels:
            issues.append(f"claim[{i}]: supports mixed with PM labels")
    return issues
