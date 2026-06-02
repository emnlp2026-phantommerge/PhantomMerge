#!/usr/bin/env python3
"""
P4 primary mitigation: anchor-only explanation regeneration + VNEXT judge.

Uses Qwen3-32B (transformers) for generation; 30B judge @ OpenAI-compatible API.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROBE_ROOT = Path(__file__).resolve().parent
REPO = PROBE_ROOT.parents[1]
BENCH_FHIR_SCRIPTS = REPO / "binding/fhir/scripts"
if str(PROBE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROBE_ROOT))
if str(BENCH_FHIR_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BENCH_FHIR_SCRIPTS))

from lib.hidden_extract import load_model_and_tokenizer  # noqa: E402
from lib.mitigation import (  # noqa: E402
    aggregate_anchor_evidence,
    build_anchor_only_user_prompt,
    format_anchor_kb_evidence,
    load_probe_split_map,
    pm_claim_count,
    select_mitigation_trajectories,
)
from fhir_binding_pipeline import build_kb_from_row  # noqa: E402
from lib.paths import DEFAULT_MODEL_DIR, FHIR_QWEN_VNEXT  # noqa: E402
from lib.per_trajectory import read_jsonl  # noqa: E402
from lib.io_utils import write_json  # noqa: E402

PM_FAILURE = frozenset(
    {"cross_object_merge", "constraint_projection", "anchored_hallucination"}
)

_CLAIM_SAVE_KEYS = (
    "claim",
    "slot",
    "value",
    "response_quote",
    "anchor_resource_id",
    "supporting_anchor_evidence",
    "supporting_other_resource_ids",
    "supporting_query_text",
    "rationale",
    "support_labels",
    "primary_support_label",
    "support_label",
    "cross_supported",
    "query_supported",
    "anchor_evidence_state",
)


def _compact_claims(claims: list[dict]) -> list[dict]:
    out: list[dict] = []
    for c in claims or []:
        row = {k: c.get(k) for k in _CLAIM_SAVE_KEYS if k in c}
        if row:
            out.append(row)
    return out


def _load_fhir_judge_api():
    shopping_v2 = REPO / "binding/shopping/scripts/analyze_phantom_merge_v2.py"
    spec = importlib.util.spec_from_file_location("shopping_pm_v2_judge", shopping_v2)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    analyze_path = BENCH_FHIR_SCRIPTS / "analyze_phantom_merge_fhir_v2.py"
    spec2 = importlib.util.spec_from_file_location("fhir_pm_v2", analyze_path)
    fhir = importlib.util.module_from_spec(spec2)
    assert spec2.loader is not None
    sys.modules[spec2.name] = fhir
    spec2.loader.exec_module(fhir)
    return fhir.analyze_one_fhir, mod.JudgeConfig


def _load_rollout_map(rollout_path: Path) -> dict[str, dict]:
    data = json.loads(rollout_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {str(r["question_id"]): r for r in data}
    return {str(k): v for k, v in data.items()}


def _generate_answer(
    model,
    tokenizer,
    user_prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": user_prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        text = user_prompt
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_ids = out[0, enc["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def run_anchor_only(
    *,
    per_path: Path,
    rollout_path: Path,
    out_dir: Path,
    model_path: str,
    max_cases: int,
    max_new_tokens: int,
    device_map: str,
    judge_base_url: str,
    judge_model: str,
    resume: bool,
    dry_run: bool,
    evidence_from_kb: bool = True,
    mitigation_judge_closure: bool = False,
    cohort: str = "pm_test",
    splits_path: Path | None = None,
) -> dict:
    analyze_one_fhir, JudgeConfig = _load_fhir_judge_api()
    per_rows = list(read_jsonl(per_path))
    split_map = None
    if cohort in ("pm_test", "pm") and splits_path and splits_path.is_file():
        split_map = load_probe_split_map(splits_path)
    prefer = "pm_test" if cohort == "pm_test" else ("pm" if cohort == "pm_all" else "com_or_cp")
    cap = max_cases if max_cases > 0 else 0
    cohort_rows = select_mitigation_trajectories(
        per_rows,
        max_cases=cap,
        prefer=prefer,
        split_map=split_map,
    )
    if rollout_path is None:
        if not dry_run:
            raise SystemExit(
                "Anchor-only mitigation requires --rollout (FHIR rollout JSON is not shipped). "
                "Use --dry-run to validate cohort selection only."
            )
        rollout: dict[str, dict] = {}
    else:
        rollout = _load_rollout_map(rollout_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "mitigation_anchor_only_per_case.jsonl"
    done_qids: set[str] = set()
    if resume and results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_qids.add(json.loads(line)["question_id"])

    if not dry_run and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "EMPTY"
    judge_cfg = JudgeConfig(
        enabled=not dry_run,
        model=judge_model,
        base_url=judge_base_url,
        api_key_env="OPENAI_API_KEY",
        temperature=0.0,
        top_p=1.0,
        max_tokens=4096,
        seed=42,
        timeout=180.0,
        max_retries=2,
        evidence_chars=12000,
        min_confidence=0.0,
    )

    model = tokenizer = None
    if not dry_run:
        model, tokenizer, _ = load_model_and_tokenizer(model_path, device_map=device_map)

    summary_cases: list[dict] = []
    t0 = time.time()

    with results_path.open("a", encoding="utf-8") as fout:
        for i, per_row in enumerate(cohort_rows):
            qid = str(per_row["question_id"])
            if qid in done_qids:
                continue
            query = str(per_row.get("query") or "")
            anchor_ids = [
                str(x).strip()
                for x in (per_row.get("selected_anchor_resource_ids") or [])
                if str(x).strip()
            ]
            anchor_ev = aggregate_anchor_evidence(per_row)
            if evidence_from_kb:
                base_row = rollout.get(qid)
                if base_row:
                    kb_full = build_kb_from_row(base_row)
                    allow = set(anchor_ids)
                    kb_anchor = {
                        rid: slot for rid, slot in kb_full.items() if rid in allow
                    } if allow else kb_full
                    anchor_ev = format_anchor_kb_evidence(kb_anchor, anchor_ids)
            user_prompt = build_anchor_only_user_prompt(query, anchor_ev)

            baseline_tl = per_row.get("trajectory_labels") or {}
            baseline_claims = per_row.get("claims") or []
            probe_split = (split_map or {}).get(qid, "")
            rec: dict[str, Any] = {
                "question_id": qid,
                "probe_split": probe_split,
                "cohort": cohort,
                "primary_stratum": per_row.get("primary_stratum"),
                "task_correct_llm": per_row.get("task_correct_llm"),
                "baseline_has_phantom_merge": bool(baseline_tl.get("has_phantom_merge")),
                "baseline_pm_claims": pm_claim_count(baseline_claims),
                "baseline_n_claims": len(baseline_claims),
                "mitigation_judge_closure": mitigation_judge_closure,
            }

            if dry_run:
                rec["anchor_only_answer"] = ""
                rec["skipped"] = "dry_run"
            else:
                assert model is not None and tokenizer is not None
                answer = _generate_answer(
                    model, tokenizer, user_prompt, max_new_tokens=max_new_tokens
                )
                rec["anchor_only_answer"] = answer[:2000]

                base_row = rollout.get(qid)
                if not base_row:
                    rec["error"] = "missing_rollout_row"
                else:
                    row = dict(base_row)
                    row["agent_answer"] = answer
                    row["question_id"] = qid
                    judged = analyze_one_fhir(
                        i,
                        row,
                        int(per_row.get("task_correct_llm") or 0),
                        judge_cfg,
                        meta={
                            "primary_stratum": per_row.get("primary_stratum"),
                            "kb_scope": "anchor_only",
                            "mitigation_judge_closure": mitigation_judge_closure,
                            "selected_anchor_resource_ids": anchor_ids,
                        },
                    )
                    mclaims = judged.get("claims") or []
                    rec["mitigation_version"] = (
                        "closure_eval" if mitigation_judge_closure else "standard_vnext"
                    )
                    rec["judge_error"] = (judged.get("judge") or {}).get("error")
                    rec["mitigated_has_phantom_merge"] = bool(
                        (judged.get("trajectory_labels") or {}).get("has_phantom_merge")
                    )
                    rec["mitigated_pm_claims"] = pm_claim_count(mclaims)
                    rec["mitigated_n_claims"] = len(mclaims)
                    rec["mitigated_claims"] = _compact_claims(mclaims)
                    rec["pm_reduced"] = rec["baseline_has_phantom_merge"] and not rec[
                        "mitigated_has_phantom_merge"
                    ]

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            summary_cases.append(rec)
            print(
                f"  [{i+1}/{len(cohort_rows)}] {qid} "
                f"baseline_pm={rec.get('baseline_has_phantom_merge')} "
                f"mitigated_pm={rec.get('mitigated_has_phantom_merge', 'n/a')}",
                flush=True,
            )

    # aggregate
    judged = [c for c in summary_cases if "mitigated_has_phantom_merge" in c]
    n = len(judged) or 1
    agg = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cohort": cohort,
        "mitigation_judge_closure": mitigation_judge_closure,
        "n_selected": len(cohort_rows),
        "n_judged_this_run": len(judged),
        "elapsed_sec": round(time.time() - t0, 1),
        "model_path": model_path,
        "judge": {"model": judge_model, "base_url": judge_base_url, "dry_run": dry_run},
        "baseline_pm_rate": sum(1 for c in judged if c["baseline_has_phantom_merge"]) / n,
        "mitigated_pm_rate": sum(1 for c in judged if c["mitigated_has_phantom_merge"]) / n,
        "pm_reduction_rate": sum(1 for c in judged if c.get("pm_reduced")) / n,
        "mean_baseline_pm_claims": float(
            sum(c.get("baseline_pm_claims", 0) for c in judged) / n
        ),
        "mean_mitigated_pm_claims": float(
            sum(c.get("mitigated_pm_claims", 0) for c in judged) / n
        ),
        "judge_errors": sum(1 for c in judged if c.get("judge_error")),
    }
    write_json(out_dir / "mitigation_anchor_only_summary.json", agg)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-path", type=Path, default=FHIR_QWEN_VNEXT)
    ap.add_argument(
        "--rollout",
        type=Path,
        default=None,
        help="FHIR rollout JSON (not shipped; required only when re-running anchor-only regen).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO / "runs/probe/fhir_anchor_only_out",
    )
    ap.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_DIR))
    ap.add_argument(
        "--cohort",
        choices=("com_or_cp", "pm_test", "pm_all"),
        default="pm_test",
        help="pm_test = probe test split PM trajectories (paper primary)",
    )
    ap.add_argument(
        "--splits",
        type=Path,
        default=REPO / "results/table3_representation/splits.json",
    )
    ap.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="0 = all in cohort (e.g. 52 for pm_test)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--device-map", type=str, default="none")
    ap.add_argument("--judge-base-url", type=str, default="http://127.0.0.1:8001/v1")
    ap.add_argument("--judge-model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Select cohort only, no GPU/judge")
    ap.add_argument(
        "--no-kb-evidence",
        action="store_true",
        help="Use baseline claim snippets for generation (v2; not recommended)",
    )
    ap.add_argument(
        "--judge-closure",
        action="store_true",
        help="Mitigation-eval only: disable COM/CP judge channels (not for main paper)",
    )
    args = ap.parse_args()

    agg = run_anchor_only(
        per_path=args.per_path,
        rollout_path=args.rollout,
        out_dir=args.out_dir,
        model_path=args.model_path,
        max_cases=args.max_cases,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        judge_base_url=args.judge_base_url,
        judge_model=args.judge_model,
        resume=args.resume,
        dry_run=args.dry_run,
        evidence_from_kb=not args.no_kb_evidence,
        mitigation_judge_closure=args.judge_closure,
        cohort=args.cohort,
        splits_path=args.splits,
    )
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
