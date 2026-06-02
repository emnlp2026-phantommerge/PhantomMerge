"""Read and validate VNEXT per_trajectory.jsonl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .labels import y_com, y_cp, y_pm_from_labels
from .paths import VALID_PIPELINE_VERSIONS
from .templates import (
    MAX_OTHER_OBJECTS,
    build_bcp_prompt,
    format_other_evidence,
)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def validate_pipeline_version(row: dict[str, Any], expected: set[str] | None = None) -> str:
    pv = str(row.get("pipeline_version") or "")
    allowed = expected or VALID_PIPELINE_VERSIONS
    if pv not in allowed:
        raise ValueError(f"pipeline_version={pv!r} not in {sorted(allowed)}")
    return pv


def _task_correct(row: dict[str, Any]) -> int:
    if "task_correct_exact" in row:
        return int(bool(row.get("task_correct_exact")))
    if "task_correct_llm" in row:
        return int(bool(row.get("task_correct_llm")))
    return int(bool(row.get("task_correct")))


def _group_id(row: dict[str, Any], domain: str) -> str:
    if domain == "fhir":
        return str(row.get("question_id") or row.get("line_index"))
    return str(row.get("orig_index") if row.get("orig_index") is not None else row.get("line_index"))


def _anchor_id(claim: dict[str, Any], domain: str) -> str:
    if domain == "fhir":
        return str(claim.get("anchor_resource_id") or claim.get("anchor_pid") or "")
    return str(claim.get("anchor_pid") or claim.get("anchor_resource_id") or "")


def _source_ids(claim: dict[str, Any], domain: str) -> list[str]:
    if domain == "fhir":
        raw = claim.get("supporting_other_resource_ids") or []
    else:
        raw = claim.get("supporting_other_product_ids") or claim.get("supporting_other_resource_ids") or []
    return [str(x).strip() for x in raw if str(x).strip()]


def export_claim_rows(
    row: dict[str, Any],
    *,
    domain: str,
    model: str,
) -> list[dict[str, Any]]:
    validate_pipeline_version(row)
    gid = _group_id(row, domain)
    query = str(row.get("query") or "")
    final_answer = str(row.get("final_answer") or "")
    task_correct = _task_correct(row)
    traj_labels = dict(row.get("trajectory_labels") or {})
    out: list[dict[str, Any]] = []

    for idx, claim in enumerate(row.get("claims") or []):
        support_labels = list(claim.get("support_labels") or [])
        primary = str(claim.get("primary_support_label") or claim.get("support_label") or "")
        if primary == "unverifiable":
            raise ValueError(f"{gid} claim {idx}: deprecated unverifiable label")
        source_ids = _source_ids(claim, domain)
        anchor_ev = str(claim.get("supporting_anchor_evidence") or "")
        query_snip = str(claim.get("supporting_query_text") or "")
        rationale = str(claim.get("rationale") or "")
        other_ev = format_other_evidence(source_ids, anchor_ev, rationale)
        claim_text = str(claim.get("claim") or "")
        y_pm = y_pm_from_labels(support_labels, primary)
        y_com_v = y_com(support_labels)
        y_cp_v = y_cp(support_labels)

        oc_source_ok = y_com_v == 1 and 1 <= len(source_ids) <= MAX_OTHER_OBJECTS
        y_owner_positive = ""
        if y_com_v and oc_source_ok:
            y_owner_positive = source_ids[0]
        elif not y_pm and primary == "correct_binding":
            y_owner_positive = _anchor_id(claim, domain)

        bcp_text = build_bcp_prompt(
            query=query,
            anchor_evidence=anchor_ev,
            other_evidence=other_ev,
            claim_text=claim_text,
        )

        claim_id = f"{gid}::{idx}"
        out.append(
            {
                "claim_id": claim_id,
                "domain": domain,
                "model": model,
                "group_id": gid,
                "claim_index": idx,
                "anchor_id": _anchor_id(claim, domain),
                "source_ids": source_ids,
                "n_source_ids": len(source_ids),
                "query": query,
                "claim_text": claim_text,
                "slot": str(claim.get("slot") or claim.get("slot_raw") or ""),
                "value": str(claim.get("value") or ""),
                "response_quote": str(claim.get("response_quote") or ""),
                "anchor_evidence_text": anchor_ev,
                "other_evidence_text": other_ev,
                "query_snippet": query_snip,
                "support_labels": support_labels,
                "primary_support_label": primary,
                "y_pm": y_pm,
                "y_com": y_com_v,
                "y_cp": y_cp_v,
                "y_owner_positive": y_owner_positive,
                "oc_eligible": int(oc_source_ok),
                "oc_source_id": source_ids[0] if oc_source_ok else "",
                "task_correct": task_correct,
                "final_answer": final_answer,
                "trajectory_has_phantom_merge": int(bool(traj_labels.get("has_phantom_merge"))),
                "bcp_prompt": bcp_text,
                "pipeline_version": row.get("pipeline_version"),
                "line_index": row.get("line_index"),
            }
        )
    return out
