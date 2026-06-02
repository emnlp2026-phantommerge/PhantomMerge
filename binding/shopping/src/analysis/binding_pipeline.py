"""
Object-level evidence binding analysis for ShoppingBench rollouts.

Definitions (see repo-root proposal.txt):
- Anchored Composite: claims about anchored product A include facts supported by KB
  for some other product B != A.
- Out-of-Set Composite: a described object profile cannot be supported by any single
  KB product, but its facts are partitionable across multiple KB products.

Also classifies protocol / trajectory integrity (LLM empty, tool errors, incomplete)
separately from binding outcomes so incomplete runs can still receive binding labels
when evidence exists.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
import statistics


def load_rollout_lines(path: Path) -> list[list[dict[str, Any]]]:
    lines: list[list[dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            lines.append(json.loads(raw))
    return lines


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


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
            p = f"{prefix}[{i}]"
            parts.extend(_flatten_dict(v, p))
    else:
        parts.append(f"{prefix}={d}")
    return parts


def build_kb_from_steps(steps: list[dict[str, Any]], upto_step: int | None = None) -> dict[str, dict[str, Any]]:
    """
    Merge all find_product + view_product_information observations up to upto_step (inclusive).
    Returns product_id -> {title, find_snippet, detail_text, raw_find, raw_view}
    """
    kb: dict[str, dict[str, Any]] = {}
    end = len(steps) if upto_step is None else min(upto_step + 1, len(steps))

    for si in range(end):
        step = steps[si]
        msg = (step.get("completion") or {}).get("message") or {}
        obs_list = msg.get("obs") or []
        for obs in obs_list:
            results = obs.get("results")
            if isinstance(results, dict) and results.get("error"):
                continue
            if not isinstance(results, list):
                continue
            for item in results:
                if not isinstance(item, dict):
                    continue
                pid = str(item.get("product_id", "")).strip()
                if not pid:
                    continue
                slot = kb.setdefault(
                    pid,
                    {
                        "product_id": pid,
                        "title": "",
                        "find_snippet": "",
                        "detail_text": "",
                        "raw_find": None,
                        "raw_view": None,
                    },
                )
                if "title" in item and "price" in item:
                    title = str(item.get("title", ""))
                    if title and not slot["title"]:
                        slot["title"] = title
                    slot["raw_find"] = item
                    slot["find_snippet"] = _norm(
                        " ".join(
                            [
                                title,
                                str(item.get("price", "")),
                                ",".join(str(x) for x in (item.get("service") or [])),
                                str(item.get("shop_id", "")),
                                str(item.get("sold_count", "")),
                            ]
                        )
                    )
                else:
                    slot["raw_view"] = item
                    detail_bits = []
                    for key in ("short_description", "description"):
                        v = item.get(key)
                        if v:
                            detail_bits.append(str(v))
                    detail_bits.extend(_flatten_dict(item.get("attributes")))
                    detail_bits.extend(_flatten_dict(item.get("sku_options")))
                    slot["detail_text"] = _norm(" ".join(detail_bits))

    for pid, slot in kb.items():
        slot["evidence_blob"] = _norm(
            " ".join(
                [
                    slot.get("title") or "",
                    slot.get("find_snippet") or "",
                    slot.get("detail_text") or "",
                ]
            )
        )
    return kb


def extract_user_query(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return ""
    up = steps[0].get("prompt") or []
    for m in up:
        if m.get("role") == "user":
            u = m.get("content") or ""
            m_user = re.search(r"<user>(.*?)</user>", u, re.DOTALL)
            if m_user:
                return m_user.group(1).strip()
    ei = steps[0].get("extra_info") or {}
    return str(ei.get("query", ""))


def step_llm_degraded(step: dict[str, Any]) -> bool:
    comp = step.get("completion") or {}
    rc = comp.get("reasoning_content")
    content = comp.get("content")
    msg = comp.get("message") or {}
    if (rc is None or str(rc).strip() == "") and (content is None or str(content).strip() == ""):
        if not msg.get("think") and not msg.get("tool_call") and not msg.get("response"):
            return True
    return False


def step_has_tool_error(step: dict[str, Any]) -> bool:
    msg = (step.get("completion") or {}).get("message") or {}
    for obs in msg.get("obs") or []:
        res = obs.get("results")
        if isinstance(res, dict) and res.get("error"):
            return True
    return False


def last_recommend_product_ids(steps: list[dict[str, Any]]) -> tuple[list[str], int | None]:
    last_ids: list[str] = []
    last_step: int | None = None
    for step in steps:
        msg = (step.get("completion") or {}).get("message") or {}
        for tc in msg.get("tool_call") or []:
            if tc.get("name") == "recommend_product":
                pids = str((tc.get("parameters") or {}).get("product_ids", ""))
                last_ids = [x.strip() for x in pids.split(",") if x.strip()]
                last_step = int((step.get("extra_info") or {}).get("step", 0))
    return last_ids, last_step


_PID_IN_TEXT_RE = re.compile(
    r"(?:product[_\s-]*id\s*[:=]\s*)?(\d{8,12})\b",
    re.I,
)


def anchor_pids_from_response_text(text: str, kb: dict[str, dict[str, Any]] | None = None) -> list[str]:
    """NL fallback when recommend_product missing (Mistral response_only_end). Prefer ids in KB."""
    if not text:
        return []
    kb = kb or {}
    seen: list[str] = []
    for m in _PID_IN_TEXT_RE.finditer(text):
        pid = m.group(1)
        if pid not in seen:
            seen.append(pid)
    if kb:
        in_kb = [p for p in seen if p in kb]
        if in_kb:
            return in_kb[:1]
    return seen[:1]


def recover_anchor_pids(
    steps: list[dict[str, Any]],
    *,
    kb: dict[str, dict[str, Any]] | None = None,
    allow_nl_fallback: bool = True,
) -> tuple[list[str], int | None, str]:
    """
    PM anchor recovery: last recommend_product, else product_id in final response (in KB if possible).
    Returns (pids, step, source) with source in recommend_tool | nl_response | none.
    """
    pids, step = last_recommend_product_ids(steps)
    if pids:
        return pids, step, "recommend_tool"
    if allow_nl_fallback:
        response, rstep = last_nonempty_response_before_terminate(steps)
        nl = anchor_pids_from_response_text(response, kb)
        if nl:
            return nl, rstep, "nl_response"
    return [], None, "none"


def extract_step_response_text(step: dict[str, Any]) -> str:
    """Assistant user-visible text from message.response or tagged completion.content."""
    msg = (step.get("completion") or {}).get("message") or {}
    resp = (msg.get("response") or "").strip()
    if resp:
        return resp
    content = (step.get("completion") or {}).get("content") or ""
    match = re.search(r"<response>(.+?)</response>", content, flags=re.DOTALL | re.I)
    if match:
        return match.group(1).strip()
    if content.strip() and not (msg.get("tool_call") or []):
        return content.strip()
    return ""


def last_nonempty_response_before_terminate(steps: list[dict[str, Any]]) -> tuple[str, int | None]:
    """Longest assistant response text before final terminate (if any)."""
    best = ("", None)
    for step in steps:
        msg = (step.get("completion") or {}).get("message") or {}
        names = {tc.get("name") for tc in (msg.get("tool_call") or [])}
        resp = extract_step_response_text(step)
        if resp and len(resp) >= len(best[0]):
            best = (resp, int((step.get("extra_info") or {}).get("step", 0)))
        if "terminate" in names:
            break
    return best


def trajectory_integrity(steps: list[dict[str, Any]]) -> dict[str, Any]:
    degraded_steps = [i + 1 for i, s in enumerate(steps) if step_llm_degraded(s)]
    tool_error_steps = [i + 1 for i, s in enumerate(steps) if step_has_tool_error(s)]
    terminated = False
    term_status = None
    if steps:
        last_msg = (steps[-1].get("completion") or {}).get("message") or {}
        for tc in last_msg.get("tool_call") or []:
            if tc.get("name") == "terminate":
                terminated = True
                term_status = (tc.get("parameters") or {}).get("status")

    incomplete_reason: list[str] = []
    if degraded_steps:
        incomplete_reason.append("llm_empty_or_unparsed_step")
    if tool_error_steps:
        incomplete_reason.append("tool_execution_error")
    if not terminated:
        incomplete_reason.append("no_terminate_in_final_step")

    complete = bool(terminated) and not degraded_steps and not tool_error_steps
    return {
        "trajectory_complete": complete,
        "terminated": terminated,
        "terminate_status": term_status,
        "llm_degraded_steps": degraded_steps,
        "tool_error_steps": tool_error_steps,
        "incomplete_reasons": incomplete_reason,
        "num_steps": len(steps),
    }


def compute_exposure_stats(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Per-case product information exposure stats.
    """
    seen_find: set[str] = set()
    seen_view: set[str] = set()
    find_calls = 0
    view_calls = 0
    overflow_like_steps = 0

    for step in steps:
        msg = (step.get("completion") or {}).get("message") or {}
        tool_calls = msg.get("tool_call") or []
        for tc in tool_calls:
            name = tc.get("name")
            if name == "find_product":
                find_calls += 1
            elif name == "view_product_information":
                view_calls += 1

        obs_list = msg.get("obs") or []
        for obs in obs_list:
            res = obs.get("results")
            if isinstance(res, dict):
                err_msg = _norm(str(res.get("message", "")))
                if "maximum context length" in err_msg or "input tokens" in err_msg:
                    overflow_like_steps += 1
            if not isinstance(res, list):
                continue
            for item in res:
                if not isinstance(item, dict):
                    continue
                pid = str(item.get("product_id", "")).strip()
                if not pid:
                    continue
                if "title" in item and "price" in item:
                    seen_find.add(pid)
                else:
                    seen_view.add(pid)

    seen_union = seen_find | seen_view
    detail_coverage_ratio = (len(seen_view) / len(seen_union)) if seen_union else 0.0
    return {
        "num_find_calls": find_calls,
        "num_view_calls": view_calls,
        "unique_products_seen_from_find": len(seen_find),
        "unique_products_seen_with_detail": len(seen_view),
        "unique_products_seen_total": len(seen_union),
        "detail_coverage_ratio": detail_coverage_ratio,
        "overflow_like_obs_count": overflow_like_steps,
    }


SERVICE_PATTERNS = [
    (re.compile(r"free\s*shipping|freeshipping", re.I), "freeShipping"),
    (re.compile(r"\bCOD\b|cash\s*on\s*delivery", re.I), "COD"),
    (re.compile(r"\bofficial\b|lazmall", re.I), "official"),
    (re.compile(r"flash\s*sale|lazflash", re.I), "flashsale"),
]

MATERIAL_PATTERNS = [
    (re.compile(r"\bB20\b", re.I), "B20"),
    (re.compile(r"bronze", re.I), "bronze"),
    (re.compile(r"\bbrass\b", re.I), "brass"),
    (re.compile(r"\balloy\b", re.I), "alloy"),
    (re.compile(r"\bABS\b", re.I), "ABS"),
    (re.compile(r"cast\s+bronze|cast\s+brass", re.I), "cast_metal"),
    (re.compile(r"hybrid", re.I), "hybrid"),
    (re.compile(r"ghost", re.I), "ghost"),
]


@dataclass
class AtomicClaim:
    kind: str
    value: str
    span: str


def extract_atomic_claims(text: str) -> list[AtomicClaim]:
    claims: list[AtomicClaim] = []
    t = text or ""
    for rx, val in SERVICE_PATTERNS:
        for m in rx.finditer(t):
            claims.append(AtomicClaim("service", val, m.group(0)))
    for rx, val in MATERIAL_PATTERNS:
        for m in rx.finditer(t):
            claims.append(AtomicClaim("material_or_constraint", val, m.group(0)))
    return claims


def kb_supports_claim(blob: str, claim: AtomicClaim) -> bool:
    b = _norm(blob)
    if claim.kind == "service":
        needle = claim.value.lower()
        if needle == "freeshipping":
            return "freeshipping" in b or "free shipping" in b
        return needle in b
    return _norm(claim.span) in b or _norm(claim.value) in b


def query_supports_claim(query: str, claim: AtomicClaim) -> bool:
    q = _norm(query)
    if claim.kind == "service":
        return claim.value.lower() in q or claim.span.lower() in q
    return _norm(claim.span) in q or _norm(claim.value) in q


def best_support_pids(kb: dict[str, dict[str, Any]], claim: AtomicClaim) -> list[str]:
    hits: list[str] = []
    for pid, slot in kb.items():
        blob = slot.get("evidence_blob", "")
        if kb_supports_claim(blob, claim):
            hits.append(pid)
    return hits


def claim_source_label(
    anchor_pid: str,
    claim: AtomicClaim,
    kb: dict[str, dict[str, Any]],
    query: str,
) -> tuple[str, list[str]]:
    """
    Returns (label, support_pids) where label in:
    supported_on_anchor | from_other_object | from_query_only | hallucinated
    """
    anchor_blob = (kb.get(anchor_pid) or {}).get("evidence_blob", "")
    if anchor_pid in kb and kb_supports_claim(anchor_blob, claim):
        return "supported_on_anchor", [anchor_pid]
    others = [p for p in best_support_pids(kb, claim) if p != anchor_pid]
    if others:
        return "from_other_object", others
    if query_supports_claim(query, claim):
        return "from_query_only", []
    return "hallucinated", []


def split_numbered_segments(response: str) -> list[tuple[int, str]]:
    """Split '1. ... 2. ...' style segments (leading preamble allowed)."""
    text = (response or "").strip()
    if not text:
        return []
    matches = list(re.finditer(r"(?m)^\s*(\d+)\.\s+", text))
    if not matches:
        return [(0, text)]
    segs: list[tuple[int, str]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            segs.append((0, preamble))
    for i, m in enumerate(matches):
        idx = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            segs.append((idx, body))
    return segs


def match_segment_to_pid(
    seg: str,
    ordered_pids: list[str],
    kb: dict[str, dict[str, Any]],
) -> str | None:
    m = re.search(r"\b(\d{6,})\b", seg)
    if m:
        pid = m.group(1)
        if pid in kb:
            return pid
    if ordered_pids and len(ordered_pids) == 1:
        return ordered_pids[0]
    best_pid = None
    best_score = 0
    for pid in ordered_pids:
        title = (kb.get(pid) or {}).get("title") or ""
        if not title:
            continue
        tl = _norm(title)
        sc = 0
        for tok in tl.split()[:12]:
            if len(tok) > 3 and tok in _norm(seg):
                sc += 1
        if sc > best_score:
            best_score = sc
            best_pid = pid
    if best_score >= 2:
        return best_pid
    return ordered_pids[0] if len(ordered_pids) == 1 else None


@dataclass
class BindingReport:
    query: str = ""
    anchor_pids: list[str] = field(default_factory=list)
    anchored_composite: bool = False
    out_of_set_composite: bool = False
    anchored_claims: list[dict[str, Any]] = field(default_factory=list)
    out_of_set_profiles: list[dict[str, Any]] = field(default_factory=list)
    claim_source_counts: dict[str, int] = field(default_factory=dict)
    analysis_scope: str = "full"  # full | partial_last_good_step
    analysis_upto_step: int | None = None
    integrity: dict[str, Any] = field(default_factory=dict)
    exposure_stats: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    first_anchored_composite_step: int | None = None
    first_out_of_set_composite_step: int | None = None


def _prefix_recommend_ids(steps: list[dict[str, Any]], end_idx: int) -> list[str]:
    last_ids: list[str] = []
    for i in range(end_idx + 1):
        msg = (steps[i].get("completion") or {}).get("message") or {}
        for tc in msg.get("tool_call") or []:
            if tc.get("name") == "recommend_product":
                pids = str((tc.get("parameters") or {}).get("product_ids", ""))
                last_ids = [x.strip() for x in pids.split(",") if x.strip()]
    return last_ids


def localize_first_violations(
    steps: list[dict[str, Any]], query: str
) -> tuple[int | None, int | None]:
    first_anchor: int | None = None
    first_oos: int | None = None
    for i, step in enumerate(steps):
        kb = build_kb_from_steps(steps, upto_step=i)
        if not kb:
            continue
        msg = (step.get("completion") or {}).get("message") or {}
        resp = (msg.get("response") or "").strip()
        if not resp:
            continue
        rec_ids = _prefix_recommend_ids(steps, i)
        if not rec_ids:
            continue
        segments = split_numbered_segments(resp)
        for seg_idx, seg_body in segments:
            pid = match_segment_to_pid(seg_body, rec_ids, kb)
            if not pid:
                continue
            claims = extract_atomic_claims(seg_body)
            if len(claims) < 1:
                continue
            for c in claims:
                label, _ = claim_source_label(pid, c, kb, query)
                if label == "from_other_object" and first_anchor is None:
                    first_anchor = int((step.get("extra_info") or {}).get("step", i + 1))
            single_support = None
            for cand_pid, slot in kb.items():
                if all(kb_supports_claim(slot.get("evidence_blob", ""), c) for c in claims):
                    single_support = cand_pid
                    break
            if single_support is None and len(claims) >= 2:
                support_union: set[str] = set()
                for c in claims:
                    support_union.update(best_support_pids(kb, c))
                if len(support_union) >= 2 and first_oos is None:
                    first_oos = int((step.get("extra_info") or {}).get("step", i + 1))
        if first_anchor is not None and first_oos is not None:
            break
    return first_anchor, first_oos


def analyze_trajectory(steps: list[dict[str, Any]]) -> BindingReport:
    rep = BindingReport()
    rep.integrity = trajectory_integrity(steps)
    rep.exposure_stats = compute_exposure_stats(steps)
    rep.query = extract_user_query(steps)

    if not steps:
        rep.notes.append("empty_trajectory")
        return rep

    last_good_idx = len(steps) - 1
    while last_good_idx >= 0 and step_llm_degraded(steps[last_good_idx]):
        last_good_idx -= 1
    if last_good_idx < 0:
        rep.notes.append("no_non_degraded_step_binding_skipped")
        return rep

    rep.analysis_upto_step = int((steps[last_good_idx].get("extra_info") or {}).get("step", last_good_idx + 1))
    rep.analysis_scope = "full" if last_good_idx == len(steps) - 1 else "partial_last_good_step"

    kb = build_kb_from_steps(steps, upto_step=last_good_idx)
    if not kb:
        rep.notes.append("empty_kb_no_observations")
        fa, fo = localize_first_violations(steps[: last_good_idx + 1], rep.query)
        rep.first_anchored_composite_step = fa
        rep.first_out_of_set_composite_step = fo
        return rep

    rec_ids, _ = last_recommend_product_ids(steps[: last_good_idx + 1])
    response, resp_step = last_nonempty_response_before_terminate(steps[: last_good_idx + 1])
    anchor_list = rec_ids if rec_ids else ([] if not kb else [sorted(kb.keys())[0]])

    rep.anchor_pids = list(dict.fromkeys(anchor_list))

    segments = split_numbered_segments(response)
    if not segments:
        rep.notes.append("no_response_text_for_claims")
        fa, fo = localize_first_violations(steps[: last_good_idx + 1], rep.query)
        rep.first_anchored_composite_step = fa
        rep.first_out_of_set_composite_step = fo
        return rep

    for seg_idx, seg_body in segments:
        ordered = rep.anchor_pids
        pid = match_segment_to_pid(seg_body, ordered, kb)
        if not pid:
            rep.notes.append(f"unmapped_segment_index_{seg_idx}")
            continue
        claims = extract_atomic_claims(seg_body)
        if not claims:
            continue

        per_claim_reports: list[dict[str, Any]] = []
        for c in claims:
            label, support_pids = claim_source_label(pid, c, kb, rep.query)
            rep.claim_source_counts[label] = rep.claim_source_counts.get(label, 0) + 1
            entry = {
                "segment_index": seg_idx,
                "anchor_pid": pid,
                "claim_kind": c.kind,
                "claim_value": c.value,
                "claim_span": c.span,
                "source_label": label,
                "support_pids_other_than_anchor": [p for p in support_pids if p != pid],
            }
            per_claim_reports.append(entry)
            if label == "from_other_object":
                rep.anchored_composite = True
                rep.anchored_claims.append(entry)

        claim_values = [c.value for c in claims]
        support_union: set[str] = set()
        for c in claims:
            support_union.update(best_support_pids(kb, c))

        single_support = None
        for cand_pid, slot in kb.items():
            if all(kb_supports_claim(slot.get("evidence_blob", ""), c) for c in claims):
                single_support = cand_pid
                break

        if single_support is None and len(claims) >= 2 and support_union:
            if len(support_union) >= 2:
                rep.out_of_set_composite = True
                rep.out_of_set_profiles.append(
                    {
                        "segment_index": seg_idx,
                        "mapped_anchor_pid": pid,
                        "claims": claim_values,
                        "support_union_pids": sorted(support_union),
                    }
                )

    fa, fo = localize_first_violations(steps, rep.query)
    rep.first_anchored_composite_step = fa
    rep.first_out_of_set_composite_step = fo

    return rep


def report_to_dict(rep: BindingReport, line_index: int) -> dict[str, Any]:
    return {
        "line_index": line_index,
        "query": rep.query,
        "binding": {
            "anchored_composite": rep.anchored_composite,
            "out_of_set_composite": rep.out_of_set_composite,
            "anchor_pids": rep.anchor_pids,
            "anchored_claims": rep.anchored_claims,
            "out_of_set_profiles": rep.out_of_set_profiles,
            "claim_source_counts": rep.claim_source_counts,
            "first_anchored_composite_step": rep.first_anchored_composite_step,
            "first_out_of_set_composite_step": rep.first_out_of_set_composite_step,
        },
        "analysis": {
            "scope": rep.analysis_scope,
            "upto_step": rep.analysis_upto_step,
            "notes": rep.notes,
        },
        "integrity": rep.integrity,
        "exposure_stats": rep.exposure_stats,
    }


def _bucket_seen_products(n: int) -> str:
    if n <= 1:
        return "seen_0_1"
    if n <= 5:
        return "seen_2_5"
    if n <= 10:
        return "seen_6_10"
    if n <= 20:
        return "seen_11_20"
    return "seen_21_plus"


def _bucket_detail_ratio(r: float) -> str:
    if r <= 0:
        return "detail_0"
    if r <= 0.25:
        return "detail_0_0.25"
    if r <= 0.5:
        return "detail_0.25_0.5"
    if r <= 0.75:
        return "detail_0.5_0.75"
    return "detail_0.75_1"


def _bucket_steps(n: int) -> str:
    if n <= 3:
        return "steps_1_3"
    if n <= 6:
        return "steps_4_6"
    if n <= 10:
        return "steps_7_10"
    return "steps_11_plus"


def _bucket_binding_rates(
    reports: list[dict[str, Any]],
    bucket_key: str,
    value_getter,
) -> dict[str, dict[str, Any]]:
    agg: dict[str, dict[str, Any]] = {}
    for r in reports:
        b = value_getter(r)
        slot = agg.setdefault(
            b,
            {
                "num_cases": 0,
                "anchored_composite_cases": 0,
                "out_of_set_composite_cases": 0,
            },
        )
        slot["num_cases"] += 1
        if r["binding"]["anchored_composite"]:
            slot["anchored_composite_cases"] += 1
        if r["binding"]["out_of_set_composite"]:
            slot["out_of_set_composite_cases"] += 1
    for _, slot in agg.items():
        n = slot["num_cases"]
        slot["anchored_composite_rate"] = slot["anchored_composite_cases"] / n if n else 0.0
        slot["out_of_set_composite_rate"] = slot["out_of_set_composite_cases"] / n if n else 0.0
    return agg


def summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(reports)
    anchored = sum(1 for r in reports if r["binding"]["anchored_composite"])
    oos = sum(1 for r in reports if r["binding"]["out_of_set_composite"])
    complete = sum(1 for r in reports if r["integrity"]["trajectory_complete"])
    partial = n - complete
    complete_list = [r for r in reports if r["integrity"]["trajectory_complete"]]
    incomplete_list = [r for r in reports if not r["integrity"]["trajectory_complete"]]
    anchored_complete = sum(1 for r in complete_list if r["binding"]["anchored_composite"])
    anchored_incomplete = sum(1 for r in incomplete_list if r["binding"]["anchored_composite"])
    oos_complete = sum(1 for r in complete_list if r["binding"]["out_of_set_composite"])
    oos_incomplete = sum(1 for r in incomplete_list if r["binding"]["out_of_set_composite"])
    src_totals: dict[str, int] = {}
    for r in reports:
        for k, v in (r["binding"].get("claim_source_counts") or {}).items():
            src_totals[k] = src_totals.get(k, 0) + int(v)
    nc, ni = len(complete_list), len(incomplete_list)

    seen_totals = [int((r.get("exposure_stats") or {}).get("unique_products_seen_total", 0)) for r in reports]
    seen_find = [int((r.get("exposure_stats") or {}).get("unique_products_seen_from_find", 0)) for r in reports]
    seen_view = [int((r.get("exposure_stats") or {}).get("unique_products_seen_with_detail", 0)) for r in reports]
    detail_cov = [float((r.get("exposure_stats") or {}).get("detail_coverage_ratio", 0.0)) for r in reports]
    steps = [int((r.get("integrity") or {}).get("num_steps", 0)) for r in reports]

    exposure_summary = {
        "unique_products_seen_total_mean": statistics.mean(seen_totals) if seen_totals else 0.0,
        "unique_products_seen_total_median": statistics.median(seen_totals) if seen_totals else 0.0,
        "unique_products_seen_total_max": max(seen_totals) if seen_totals else 0,
        "unique_products_seen_from_find_mean": statistics.mean(seen_find) if seen_find else 0.0,
        "unique_products_seen_with_detail_mean": statistics.mean(seen_view) if seen_view else 0.0,
        "detail_coverage_ratio_mean": statistics.mean(detail_cov) if detail_cov else 0.0,
        "num_steps_mean": statistics.mean(steps) if steps else 0.0,
        "num_steps_median": statistics.median(steps) if steps else 0.0,
    }

    bucket_stats = {
        "by_seen_products_bucket": _bucket_binding_rates(
            reports,
            "seen_products",
            lambda r: _bucket_seen_products(
                int((r.get("exposure_stats") or {}).get("unique_products_seen_total", 0))
            ),
        ),
        "by_detail_coverage_bucket": _bucket_binding_rates(
            reports,
            "detail_coverage",
            lambda r: _bucket_detail_ratio(
                float((r.get("exposure_stats") or {}).get("detail_coverage_ratio", 0.0))
            ),
        ),
        "by_num_steps_bucket": _bucket_binding_rates(
            reports,
            "num_steps",
            lambda r: _bucket_steps(int((r.get("integrity") or {}).get("num_steps", 0))),
        ),
    }

    return {
        "num_trajectories": n,
        "trajectory_complete": complete,
        "trajectory_incomplete": partial,
        "anchored_composite_trajectories": anchored,
        "anchored_composite_rate": anchored / n if n else 0.0,
        "out_of_set_composite_trajectories": oos,
        "out_of_set_composite_rate": oos / n if n else 0.0,
        "anchored_composite_on_complete_trajectories": anchored_complete,
        "anchored_composite_on_complete_rate": anchored_complete / nc if nc else 0.0,
        "anchored_composite_on_incomplete_trajectories": anchored_incomplete,
        "anchored_composite_on_incomplete_rate": anchored_incomplete / ni if ni else 0.0,
        "out_of_set_composite_on_complete_trajectories": oos_complete,
        "out_of_set_composite_on_complete_rate": oos_complete / nc if nc else 0.0,
        "out_of_set_composite_on_incomplete_trajectories": oos_incomplete,
        "out_of_set_composite_on_incomplete_rate": oos_incomplete / ni if ni else 0.0,
        "claim_source_totals": src_totals,
        "exposure_summary": exposure_summary,
        "bucket_stats": bucket_stats,
    }


def run_pipeline(rollout_path: Path, per_line_out: Path | None = None) -> dict[str, Any]:
    lines = load_rollout_lines(rollout_path)
    reports: list[dict[str, Any]] = []
    for i, steps in enumerate(lines):
        rep = analyze_trajectory(steps)
        d = report_to_dict(rep, i)
        reports.append(d)
    summary = summarize(reports)
    summary["rollout_file"] = str(rollout_path)

    if per_line_out:
        per_line_out.parent.mkdir(parents=True, exist_ok=True)
        with per_line_out.open("w", encoding="utf-8") as f:
            for row in reports:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"summary": summary, "per_trajectory": reports}
