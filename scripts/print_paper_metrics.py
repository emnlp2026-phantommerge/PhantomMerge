#!/usr/bin/env python3
"""Print sealed metrics cited in the paper (Table 1 + probe + MSPS). CPU only."""

from __future__ import annotations

import json

from repo_paths import (
    APPENDIX,
    BCP_DETECT,
    MSPS_ULTIMATE,
    PAPER_COUNTS,
    SUPPLEMENTAL,
)


def _load(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    print("=" * 60)
    print("TABLE 1 — table1_characterization/counts.json")
    print("=" * 60)
    data = _load(PAPER_COUNTS)
    for key, cell in data.get("cells", {}).items():
        n = cell["paper_denominator_n"]
        tc = cell["task_correct"]
        pm = cell["has_phantom_merge"]
        tcpm = cell["task_correct_and_pm"]
        print(
            f"  {key:24s} N={n:4d}  task_correct={tc:4d}  "
            f"has_pm={pm:4d}  TC∧PM={tcpm:3d}  "
            f"PM%={100*pm/n:5.1f}  TC∧PM%={100*tcpm/n:4.1f}"
        )

    print()
    print("=" * 60)
    print("BCP-Detect (FHIR Qwen test) — table3_representation/bcp_detect.json")
    print("=" * 60)
    p2 = _load(BCP_DETECT)
    primary = p2.get("primary_bcp_detect", {})
    test = (primary.get("splits") or {}).get("test", {})
    print(f"  label={primary.get('label')}  layer={primary.get('layer_name')}  n_test={test.get('n')}")
    print(f"  AUROC_test={test.get('auroc'):.4f}" if test.get("auroc") else "  AUROC_test=n/a")

    print()
    print("=" * 60)
    print("MSPS (FHIR Qwen test, n=146) — table4_mitigation/msps_ultimate_results.json")
    print("=" * 60)
    msps = _load(MSPS_ULTIMATE)
    t146 = msps.get("test_full_146", {})
    print(
        f"  baseline_pm_rate={t146.get('baseline_pm_rate', 0)*100:.1f}%  "
        f"gated_pm_rate={t146.get('gated_pm_rate', 0)*100:.1f}%  "
        f"pm_reduction={t146.get('pm_reduction', 0)*100:.1f} pp"
    )
    cfg = msps.get("val_selected_config", {})
    print(f"  taus: max={cfg.get('tau_max')} com={cfg.get('tau_com')} cp={cfg.get('tau_cp')}")

    cb_path = APPENDIX / "mitigation_cb_retention_audit.json"
    if cb_path.is_file():
        print()
        print("=" * 60)
        print("CB retention audit — appendix/mitigation_cb_retention_audit.json")
        print("=" * 60)
        cb = _load(cb_path).get("msps_tau0.45", {})
        print(
            f"  MSPS gated_pm_rate={cb.get('gated_pm_rate', 0)*100:.1f}%  "
            f"cb_retention={cb.get('cb_retention_rate', 0)*100:.1f}%"
        )

    util_path = SUPPLEMENTAL / "mitigation_utility_proxy.json"
    if util_path.is_file():
        print()
        print("=" * 60)
        print("Mitigation utility proxies — supplemental/mitigation_utility_proxy.json")
        print("=" * 60)
        util = _load(util_path)
        m = util.get("operating_points", {}).get("MSPS_aggressive_tau0.45", {})
        if m.get("task_unchanged_heuristic_rate_on_task_correct") is not None:
            print(
                f"  MSPS@0.45 anchor_unchanged={m.get('anchor_unchanged_rate', 0)*100:.1f}%  "
                f"task_heur|TC={m['task_unchanged_heuristic_rate_on_task_correct']*100:.1f}%"
            )

    print()
    print("Verify: bash reproduce/verify.sh")


if __name__ == "__main__":
    main()
