#!/usr/bin/env python3
"""P4: final release cleanup — remove cruft, dedupe docs, fix reproduce scripts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Safe to delete: duplicates, internal dev wrappers, repair utilities, unreleased pilots.
DELETE_FILES = [
    "11717_Phantom_Merge_When_Your_.pdf",
    "RELEASE_NOTES.txt",
    "probes/requirements.txt",
    "probes/summarize_mitigation_research.py",
    "probes/run_pgcs_onepass.sh",
    "probes/run_pgcs_onepass_v2.sh",
    "probes/run_pgcs_onepass.py",
    "probes/run_mitigation_v2.sh",
    "probes/run_mitigation_pm_all.sh",
    "probes/run_mitigation_ultimate.sh",
    "probes/run_p0_fhir.sh",
    "probes/run_p4.sh",
    "probes/backfill_mitigated_claims.py",
    "probes/reresolve_anchor_mitigation.py",
    "probes/lib/pgcs_onepass.py",
    "results/table4_mitigation/mitigation/anchor_regen_pilot50_v2_summary.json",
]


def _rm(rel: str) -> None:
    path = REPO / rel
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"DELETED {rel}")


def _fix_oc_bcp_script() -> None:
    path = REPO / "reproduce/table3_oc_bcp.sh"
    if not path.is_file():
        return
    path.write_text(
        """#!/usr/bin/env bash
# Optional: re-run OC-BCP mechanism exports (see probes/run_p3_suite.py).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
bash probes/run_p3.sh "$@"
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    print("FIXED reproduce/table3_oc_bcp.sh")


def _fix_table3_bcp_header() -> None:
    path = REPO / "reproduce/table3_bcp_detect.sh"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.startswith("# Tier 2 optional — Table 3 BCP")]
    if lines and not lines[0].startswith("#!"):
        lines.insert(0, "#!/usr/bin/env bash")
    if "# Optional:" not in text:
        lines.insert(1, "# Optional: re-run BCP-Detect (GPU; feature bank .npy not shipped).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _fix_table4_header() -> None:
    path = REPO / "reproduce/table4_mitigation.sh"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if not ln.startswith("# Tier 2 optional — Table 4")]
    if "# Optional:" not in text:
        lines.insert(1, "# Optional: re-run MSPS / gating (GPU; sealed Table 4 is authoritative).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _patch_p03_oc_header_bug() -> None:
    """Prevent p03 from re-triplicating OC-BCP header on re-run."""
    p03 = REPO / "scripts/p03_migrate_reproduce_layout.py"
    if not p03.is_file():
        return
    text = p03.read_text(encoding="utf-8")
    old = '    if path.name == "table3_oc_bcp.sh":\n        text = "# Tier 2 optional — Table 3 OC-BCP mechanism (CPU/GPU per probes/run_p3.sh).\\n" + text'
    new = '    if path.name == "table3_oc_bcp.sh" and "Optional: re-run OC-BCP" not in text:\n        text = "# Optional: re-run OC-BCP mechanism exports.\\n" + text'
    if old in text:
        text = text.replace(old, new)
        p03.write_text(text, encoding="utf-8")


def update_artifact_not_shipped() -> None:
    path = REPO / "ARTIFACT.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    extra = [
        "agent_rollouts",
        "feature_bank_tensors_npy",
        "human_label_audits",
        "qualitative_case_trajectories",
        "excluded_judge_failure_trajectories",
        "qa_label_auditor_scripts",
    ]
    data["not_shipped"] = extra
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print("UPDATED ARTIFACT.json not_shipped")


def main() -> int:
    for rel in DELETE_FILES:
        _rm(rel)
    _fix_oc_bcp_script()
    _fix_table3_bcp_header()
    _fix_table4_header()
    _patch_p03_oc_header_bug()
    update_artifact_not_shipped()
    print("\nP4 final cleanup completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
