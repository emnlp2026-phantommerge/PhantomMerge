"""BCP / OC-BCP prompt builders (v1 templates)."""

from __future__ import annotations

import re
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

_BCP_TEMPLATE: str | None = None
_OC_STAGES: dict[str, str] | None = None

MAX_OTHER_OBJECTS = 3
MAX_OTHER_EVIDENCE_CHARS = 2000
MAX_QUERY_CHARS = 4000
MAX_ANCHOR_EVIDENCE_CHARS = 3000


def _load_bcp_template() -> str:
    global _BCP_TEMPLATE
    if _BCP_TEMPLATE is None:
        _BCP_TEMPLATE = (_CONFIG_DIR / "bcp_input_v1.txt").read_text(encoding="utf-8")
    return _BCP_TEMPLATE


def _load_oc_stages() -> dict[str, str]:
    global _OC_STAGES
    if _OC_STAGES is not None:
        return _OC_STAGES
    text = (_CONFIG_DIR / "oc_bcp_prompts_v1.txt").read_text(encoding="utf-8")
    stages: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current and buf:
                stages[current] = "\n".join(buf).strip() + "\n"
            current = line[3:].strip()
            buf = []
        elif line.startswith("#") or not line.strip():
            continue
        else:
            buf.append(line)
    if current and buf:
        stages[current] = "\n".join(buf).strip() + "\n"
    _OC_STAGES = stages
    return stages


def _clip(text: str, limit: int) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


def format_other_evidence(
    source_ids: list[str],
    supporting_anchor_evidence: str,
    rationale: str,
    *,
    max_objects: int = MAX_OTHER_OBJECTS,
) -> str:
    ids = [str(x).strip() for x in (source_ids or []) if str(x).strip()][:max_objects]
    if not ids:
        return ""
    parts = [f"Observed object(s): {', '.join(ids)}"]
    rat = _clip(rationale, 800)
    if rat and rat not in supporting_anchor_evidence:
        parts.append(rat)
    return _clip("\n".join(parts), MAX_OTHER_EVIDENCE_CHARS)


BCP_VARIANTS = ("full", "no_other", "no_anchor", "no_query")


def build_bcp_prompt(
    *,
    query: str,
    anchor_evidence: str,
    other_evidence: str,
    claim_text: str,
) -> str:
    return build_bcp_prompt_variant(
        "full",
        query=query,
        anchor_evidence=anchor_evidence,
        other_evidence=other_evidence,
        claim_text=claim_text,
    )


def build_bcp_prompt_variant(
    variant: str,
    *,
    query: str,
    anchor_evidence: str,
    other_evidence: str,
    claim_text: str,
) -> str:
    if variant not in BCP_VARIANTS:
        raise ValueError(f"Unknown BCP variant: {variant}")
    q = _clip(query, MAX_QUERY_CHARS)
    a = _clip(anchor_evidence, MAX_ANCHOR_EVIDENCE_CHARS) or "(none)"
    o = _clip(other_evidence, MAX_OTHER_EVIDENCE_CHARS) or "(none)"
    if variant == "no_query":
        q = "(none)"
    if variant == "no_anchor":
        a = "(none)"
    if variant == "no_other":
        o = "(none)"
    tpl = _load_bcp_template()
    return tpl.format(
        query=q,
        anchor_evidence=a,
        other_evidence=o,
        claim_text=_clip(claim_text, 1500),
    )


def build_oc_prompt(
    stage: str,
    *,
    anchor_evidence: str,
    source_evidence: str,
    claim: str,
    final_answer_prefix: str = "The final answer states:",
) -> str:
    stages = _load_oc_stages()
    if stage not in stages:
        raise KeyError(f"Unknown OC stage: {stage}")
    return stages[stage].format(
        anchor_evidence=_clip(anchor_evidence, MAX_ANCHOR_EVIDENCE_CHARS) or "(none)",
        source_evidence=_clip(source_evidence, MAX_OTHER_EVIDENCE_CHARS) or "(none)",
        claim=_clip(claim, 1500),
        final_answer_prefix=_clip(final_answer_prefix, 500),
    )


def final_answer_prefix_from_text(final_answer: str, *, max_chars: int = 280) -> str:
    fa = re.sub(r"\s+", " ", str(final_answer or "")).strip()
    if not fa:
        return '"Object A has the following property:"'
    snippet = fa[:max_chars]
    if len(fa) > max_chars:
        snippet += "..."
    return f'"{snippet}"'
