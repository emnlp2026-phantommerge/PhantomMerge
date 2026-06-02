"""FHIR-AgentBench binding helpers for Phantom Merge (anchor = FHIR resource id)."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

FINAL_ANSWER_RE = re.compile(r"the final answer is\s*:(.+)", re.I | re.S)
FHIR_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.I,
)
ENCOUNTER_CUE_RE = re.compile(
    r"\b(first|last|initial|earliest|latest)\s+(hospital|icu|emergency|ed)\s+(visit|stay|encounter)\b",
    re.I,
)


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def extract_final_answer(agent_answer: str) -> str:
    if not agent_answer:
        return ""
    m = FINAL_ANSWER_RE.search(str(agent_answer))
    if m:
        return m.group(1).strip()
    return str(agent_answer).strip()


def extract_query(row: dict[str, Any]) -> str:
    return str(row.get("question_with_context") or row.get("question") or "").strip()


def parse_fhir_resources(raw: Any) -> dict[str, list[str]]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        out: dict[str, list[str]] = {}
        for k, v in raw.items():
            if isinstance(v, list):
                out[str(k)] = [str(x) for x in v]
        return out
    if isinstance(raw, str):
        try:
            obj = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                return {}
        if isinstance(obj, dict):
            return {k: [str(x) for x in v] for k, v in obj.items() if isinstance(v, list)}
    return {}


def parse_true_fhir_ids(raw: Any) -> dict[str, list[str]]:
    return parse_fhir_resources(raw)


def _flatten_dict(d: Any, prefix: str = "") -> list[str]:
    parts: list[str] = []
    if d is None:
        return parts
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            parts.extend(_flatten_dict(v, p))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            parts.extend(_flatten_dict(v, f"{prefix}[{i}]"))
    else:
        parts.append(f"{prefix}={d}")
    return parts


def _codeable_text(obj: Any) -> str:
    if not obj:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        if obj.get("text"):
            return str(obj["text"])
        coding = obj.get("coding") or []
        if isinstance(coding, list):
            return " ".join(
                str(c.get("display") or c.get("code") or "") for c in coding if isinstance(c, dict)
            )
    return str(obj)[:300]


def _resource_evidence_blob(rtype: str, res: dict[str, Any]) -> str:
    bits = [rtype, str(res.get("id", ""))]
    if res.get("status"):
        bits.append(f"status={res['status']}")
    code_txt = _codeable_text(res.get("code"))
    if code_txt:
        bits.append(f"code={code_txt}")
    med_txt = _codeable_text(res.get("medicationCodeableConcept"))
    if med_txt:
        bits.append(f"medication={med_txt}")
    vq = res.get("valueQuantity") or {}
    if isinstance(vq, dict) and vq.get("value") is not None:
        bits.append(f"value={vq.get('value')}{vq.get('unit') or ''}")
    if res.get("valueString"):
        bits.append(f"valueString={res['valueString']}")
    for key in ("effectiveDateTime", "authoredOn", "recordedDate", "issued"):
        if res.get(key):
            bits.append(f"{key}={res[key]}")
    period = res.get("period") or res.get("effectivePeriod")
    if period:
        bits.append(f"period={period}")
    enc = res.get("encounter")
    if enc:
        bits.append(f"encounter={enc}")
    bits.extend(_flatten_dict(res.get("identifier")))
    return norm(" ".join(bits))


def _iter_compact_resources(payload: Any) -> list[dict[str, Any]]:
    """Extract slim FHIR dicts from tool message (dict or JSON string)."""
    if payload is None:
        return []
    if isinstance(payload, dict):
        obj = payload
    else:
        text = str(payload).strip()
        if not text:
            return []
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return []
    if not isinstance(obj, dict):
        return []

    out: list[dict[str, Any]] = []
    resources = obj.get("resources")
    if isinstance(resources, str):
        try:
            resources = json.loads(resources)
        except json.JSONDecodeError:
            try:
                resources = ast.literal_eval(resources)
            except (SyntaxError, ValueError):
                resources = None
    if isinstance(resources, dict):
        for rtype, items in resources.items():
            if not isinstance(items, list):
                continue
            for res in items:
                if isinstance(res, dict):
                    if not res.get("resourceType"):
                        res = {**res, "resourceType": rtype}
                    out.append(res)
    elif isinstance(resources, list):
        out.extend(r for r in resources if isinstance(r, dict))
    return out


def build_kb_from_row(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """resource_id -> {resource_type, evidence_blob, raw}."""
    kb: dict[str, dict[str, Any]] = {}
    query = extract_query(row)

    trace = row.get("trace")
    if isinstance(trace, list):
        for turn in trace:
            if not isinstance(turn, dict) or turn.get("role") != "tool":
                continue
            content = turn.get("content")
            for res in _iter_compact_resources(content):
                rid = str(res.get("id") or "").strip()
                rtype = str(res.get("resourceType") or "").strip()
                if not rid:
                    continue
                kb[rid] = {
                    "resource_id": rid,
                    "resource_type": rtype,
                    "evidence_blob": _resource_evidence_blob(rtype, res),
                    "raw": res,
                }

    for rtype, ids in parse_fhir_resources(row.get("agent_fhir_resources")).items():
        for rid in ids:
            rid = str(rid).strip()
            if not rid:
                continue
            slot = kb.setdefault(
                rid,
                {
                    "resource_id": rid,
                    "resource_type": rtype,
                    "evidence_blob": norm(f"{rtype} {rid}"),
                    "raw": None,
                },
            )
            if not slot.get("resource_type"):
                slot["resource_type"] = rtype

    # Enrich thin entries from assistant reasoning (medication names, dates).
    reasoning = ""
    if isinstance(trace, list):
        for turn in trace:
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                reasoning += " " + str(turn.get("content") or "")
    reasoning_n = norm(reasoning)
    for slot in kb.values():
        if len(slot.get("evidence_blob", "")) < 60 and reasoning_n:
            slot["evidence_blob"] = norm(
                f"{slot['evidence_blob']} {reasoning_n[:400]}"
            )
    return kb


def _pick_encounter_anchor(kb: dict[str, dict[str, Any]], query: str) -> str | None:
    encounters = [
        (rid, slot)
        for rid, slot in kb.items()
        if slot.get("resource_type") == "Encounter" and slot.get("raw")
    ]
    if not encounters:
        return None
    q = query.lower()
    want_first = any(w in q for w in ("first", "initial", "earliest"))
    want_last = any(w in q for w in ("last", "latest", "most recent"))

    def enc_start(item: tuple[str, dict]) -> str:
        raw = item[1].get("raw") or {}
        period = raw.get("period") or {}
        return str(period.get("start") or "")

    encounters.sort(key=enc_start)
    if want_last and encounters:
        return encounters[-1][0]
    if want_first and encounters:
        return encounters[0][0]
    return None


def infer_anchor_ids(
    response: str,
    kb: dict[str, dict[str, Any]],
    agent_fhir_resources: Any = None,
    true_fhir_ids: dict[str, list[str]] | None = None,
    query: str = "",
    max_anchors: int = 1,
) -> list[str]:
    """
    Prefer a single recoverable clinical anchor (paper-facing).
    Order: UUID in final answer → encounter cue → gold id in KB (binding target proxy).
    """
    kb_ids = set(kb.keys())
    cited: list[str] = []
    for uid in FHIR_UUID_RE.findall(response or ""):
        if uid in kb_ids and uid not in cited:
            cited.append(uid)
    if cited:
        return cited[:max_anchors]

    enc = _pick_encounter_anchor(kb, query)
    if enc:
        return [enc]

    if true_fhir_ids:
        for rtype in sorted(true_fhir_ids.keys(), key=lambda k: -len(true_fhir_ids[k])):
            for rid in true_fhir_ids[rtype]:
                if rid in kb_ids:
                    return [rid][:max_anchors]

    resources = parse_fhir_resources(agent_fhir_resources)
    for rtype in sorted(resources.keys(), key=lambda k: -len(resources[k])):
        for rid in resources[rtype][:5]:
            if rid in kb_ids:
                return [rid][:max_anchors]

    ranked = sorted(kb.items(), key=lambda x: -len(x[1].get("evidence_blob", "")))
    if ranked:
        return [ranked[0][0]][:max_anchors]
    return []


def fallback_claims_from_response(
    response: str, anchor_ids: list[str], query: str
) -> list[dict[str, Any]]:
    """Deterministic multi-claim seeds when judge returns few claims."""
    anchor = anchor_ids[0] if anchor_ids else ""
    claims: list[dict[str, Any]] = []
    text = response or ""

    if re.search(r"\b(yes|no)\b", text, re.I):
        val = "yes" if re.search(r"\byes\b", text, re.I) else "no"
        claims.append(
            {
                "anchor_resource_id": anchor,
                "claim": text[:200],
                "slot": "yes_no",
                "value": val,
                "response_quote": text[:200],
                "confidence": 0.75,
                "judge_rationale": "fallback_yes_no",
            }
        )

    for m in re.finditer(
        r"(\d+(?:\.\d+)?)\s*(?:mg|ml|mL|%|mmHg|bpm|meq/l|units?)?", text, re.I
    ):
        claims.append(
            {
                "anchor_resource_id": anchor,
                "claim": m.group(0),
                "slot": "numeric_value",
                "value": m.group(1),
                "response_quote": m.group(0),
                "confidence": 0.7,
                "judge_rationale": "fallback_numeric",
            }
        )

    if ENCOUNTER_CUE_RE.search(query) and not any(c["slot"] == "encounter_scope" for c in claims):
        claims.append(
            {
                "anchor_resource_id": anchor,
                "claim": "Encounter scope in query",
                "slot": "encounter_scope",
                "value": ENCOUNTER_CUE_RE.search(query).group(0),
                "response_quote": text[:120],
                "confidence": 0.65,
                "judge_rationale": "fallback_encounter_scope",
            }
        )

    if not claims and text.strip():
        claims.append(
            {
                "anchor_resource_id": anchor,
                "claim": text[:200],
                "slot": "clinical_attribute",
                "value": text[:80],
                "response_quote": text[:200],
                "confidence": 0.6,
                "judge_rationale": "fallback_whole_answer",
            }
        )
    return claims[:4]


def trajectory_integrity_fhir(row: dict[str, Any]) -> dict[str, Any]:
    ans = str(row.get("agent_answer") or "")
    trace = row.get("trace")
    issues: list[str] = []
    if not ans.strip():
        issues.append("empty_answer")
    if "Input tokens exceeded" in ans:
        issues.append("token_exceeded")
    if "the final answer is" not in ans.lower():
        issues.append("missing_final_marker")
    if not row.get("agent_fhir_resources"):
        issues.append("missing_fhir_resources")
    n_tool = 0
    if isinstance(trace, list):
        n_tool = sum(1 for t in trace if isinstance(t, dict) and t.get("role") == "tool")
        if n_tool == 0:
            issues.append("no_tool_usage")
    return {
        "trajectory_complete": "missing_final_marker" not in issues and "empty_answer" not in issues,
        "issues": issues,
        "n_tool_turns": n_tool,
    }


def compact_kb_fhir(
    kb: dict[str, dict[str, Any]],
    anchor_ids: list[str],
    true_fhir_ids: dict[str, list[str]] | None,
    limit: int,
    max_resources: int = 30,
) -> list[dict[str, Any]]:
    """Prioritize anchor, gold-overlap, then richest evidence for judge."""
    anchor_set = set(anchor_ids)
    gold_set: set[str] = set()
    if true_fhir_ids:
        for ids in true_fhir_ids.values():
            gold_set.update(ids)

    def priority(rid: str) -> tuple[int, int, str]:
        slot = kb[rid]
        blob_len = len(slot.get("evidence_blob", ""))
        return (
            0 if rid in anchor_set else 1,
            0 if rid in gold_set else 1,
            -blob_len,
            rid,
        )

    ordered_ids = sorted(kb.keys(), key=priority)
    out: list[dict[str, Any]] = []
    for rid in ordered_ids[:max_resources]:
        slot = kb[rid]
        blob = slot.get("evidence_blob", "")
        out.append(
            {
                "resource_id": rid,
                "resource_type": slot.get("resource_type", ""),
                "is_selected_anchor": rid in anchor_set,
                "is_gold_overlap": rid in gold_set,
                "evidence_excerpt": blob[:limit],
            }
        )
    return out


def lexical_support_ids(
    kb: dict[str, dict[str, Any]], value: str, quote: str
) -> list[str]:
    needles = []
    for candidate in {norm(value), norm(quote)}:
        if len(candidate) >= 3 and candidate not in {"yes", "no", "true", "false"}:
            needles.append(candidate)
    out: list[str] = []
    for rid, slot in kb.items():
        blob = norm(slot.get("evidence_blob", ""))
        if any(n in blob for n in needles):
            out.append(rid)
    return out


def lexical_query_support(query: str, value: str, quote: str) -> bool:
    q = norm(query)
    return any(n in q for n in {norm(value), norm(quote)} if len(n) >= 4)


infer_anchor_ids_fixed = infer_anchor_ids
