#!/usr/bin/env python3
"""P3: unify reproduction entry points under reproduce/ (paper-aligned names)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPRODUCE = REPO / "reproduce"
LEGACY = REPO / "experiments"

MOVE_MAP = {
    "experiments/exp1_characterization/verify.sh": "reproduce/table1_characterization.sh",
    "experiments/exp2_bcp_detect/run.sh": "reproduce/table3_bcp_detect.sh",
    "experiments/exp3_oc_bcp/run.sh": "reproduce/table3_oc_bcp.sh",
    "experiments/exp4_mitigation/run.sh": "reproduce/table4_mitigation.sh",
    "experiments/INDEX.json": "reproduce/INDEX.json",
}


def _write_verify_sh() -> None:
    path = REPRODUCE / "verify.sh"
    path.write_text(
        """#!/usr/bin/env bash
# Tier 0 — verify sealed paper artifacts (CPU only; no judge / no GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

echo "=== Phantom Merge — sealed artifact verification ==="
"$PY" scripts/p01_migrate_results_layout.py
"$PY" scripts/p02_migrate_code_layout.py
"$PY" scripts/p03_migrate_reproduce_layout.py
"$PY" scripts/p04_finalize_release.py
"$PY" scripts/sanitize_results_metadata.py
"$PY" scripts/p05_harden_release.py
"$PY" scripts/check_scaffold.py

echo ""
echo "=== Paper metrics (sealed JSON) ==="
"$PY" scripts/print_paper_metrics.py
echo ""
echo "=== Done ==="
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    print(f"WROTE {path.relative_to(REPO)}")


def _patch_reproduce_shell(path: Path) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    replacements = [
        ('ROOT="$(cd "$(dirname "$0")/.." && pwd)"', 'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"'),
        ('ROOT="$(cd "$(dirname "$0")/../.." && pwd)"', 'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    if path.name == "table1_characterization.sh" and "Table 1" not in text:
        text = text.replace(
            "# Tier 1: verify sealed Table 1 counts (no judge re-run).",
            "# Tier 1 — Table 1 characterization (sealed counts; no judge re-run).",
        )
    if path.name == "table3_bcp_detect.sh" and "Table 3" not in text[:120]:
        text = "# Tier 2 optional — Table 3 BCP-Detect (GPU feature bank; not in release).\n" + text
    if path.name == "table3_oc_bcp.sh" and "Optional: re-run OC-BCP" not in text:
        text = "# Optional: re-run OC-BCP mechanism exports.\n" + text
    if path.name == "table4_mitigation.sh":
        text = "# Tier 2 optional — Table 4 MSPS / mitigation (GPU; sealed results authoritative).\n" + text
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _mv(src_rel: str, dst_rel: str) -> None:
    src = REPO / src_rel
    dst = REPO / dst_rel
    if not src.is_file():
        return
    if dst.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"MOVE {src_rel} -> {dst_rel}")
    _patch_reproduce_shell(dst)


def _write_index_json() -> None:
    payload = {
        "description": "Paper-aligned reproduction entry points. Sealed results/ are authoritative.",
        "tier0_verify": "reproduce/verify.sh",
        "tiers": {
            "table1": {
                "paper": "Table 1 — cross-domain PM characterization",
                "script": "reproduce/table1_characterization.sh",
                "results": "results/table1_characterization/",
                "note": "CPU only; checks sealed counts vs per_trajectory JSONL.",
            },
            "table3_bcp": {
                "paper": "Table 3 — BCP-Detect",
                "script": "reproduce/table3_bcp_detect.sh",
                "results": "results/table3_representation/bcp_detect.json",
                "note": "Optional re-run; requires GPU + feature bank tensors (not shipped).",
            },
            "table3_oc": {
                "paper": "Table 3 — OC-BCP mechanism",
                "script": "reproduce/table3_oc_bcp.sh",
                "results": "results/table3_representation/oc_bcp.json",
            },
            "table4": {
                "paper": "Table 4 — MSPS mitigation",
                "script": "reproduce/table4_mitigation.sh",
                "results": "results/table4_mitigation/msps_test146.json",
                "note": "Optional re-run; GPU recommended.",
            },
        },
        "shortcuts": {
            "run_diagnosis": "scripts/run_diagnosis.sh",
            "run_probe": "scripts/run_probe.sh",
            "run_mitigation": "scripts/run_mitigation.sh",
        },
    }
    (REPRODUCE / "INDEX.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print("WROTE reproduce/INDEX.json")


def _write_script_shortcuts() -> None:
    shortcuts = {
        "scripts/run_diagnosis.sh": """#!/usr/bin/env bash
# Tier 1 — verify sealed Table 1 (no LLM judge; no rollouts in this release).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/reproduce/table1_characterization.sh" "$@"
""",
        "scripts/run_probe.sh": """#!/usr/bin/env bash
# Tier 2 optional — re-run BCP-Detect pipeline (GPU; feature bank not shipped).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/reproduce/table3_bcp_detect.sh" "$@"
""",
        "scripts/run_mitigation.sh": """#!/usr/bin/env bash
# Tier 2 optional — re-run mitigation / MSPS suite (GPU; sealed Table 4 authoritative).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/reproduce/table4_mitigation.sh" "$@"
""",
        "scripts/verify_release.sh": """#!/usr/bin/env bash
# Backward-compatible wrapper — canonical entry is reproduce/verify.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/reproduce/verify.sh" "$@"
""",
    }
    for rel, body in shortcuts.items():
        path = REPO / rel
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
        print(f"WROTE {rel}")


def _remove_legacy_experiments() -> None:
    if not LEGACY.is_dir():
        return
    remaining = [p for p in LEGACY.rglob("*") if p.is_file()]
    if remaining:
        print(f"WARN experiments/ still has {len(remaining)} files")
        return
    shutil.rmtree(LEGACY)
    print("REMOVED experiments/")


def update_artifact_json() -> None:
    path = REPO / "ARTIFACT.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["verify"] = "bash reproduce/verify.sh"
    data["reproduce"] = {
        "index": "reproduce/INDEX.json",
        "tier0": "reproduce/verify.sh",
        "table1": "reproduce/table1_characterization.sh",
        "table3_bcp": "reproduce/table3_bcp_detect.sh",
        "table3_oc": "reproduce/table3_oc_bcp.sh",
        "table4": "reproduce/table4_mitigation.sh",
        "shortcuts": {
            "diagnosis": "scripts/run_diagnosis.sh",
            "probe": "scripts/run_probe.sh",
            "mitigation": "scripts/run_mitigation.sh",
        },
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print("UPDATED ARTIFACT.json (reproduce section)")


def migrate() -> None:
    REPRODUCE.mkdir(parents=True, exist_ok=True)
    _write_verify_sh()
    for src, dst in MOVE_MAP.items():
        _mv(src, dst)
    # Ensure table scripts exist even if already migrated
    for name in (
        "table1_characterization.sh",
        "table3_bcp_detect.sh",
        "table3_oc_bcp.sh",
        "table4_mitigation.sh",
    ):
        _patch_reproduce_shell(REPRODUCE / name)
    _write_index_json()
    _write_script_shortcuts()
    _remove_legacy_experiments()
    update_artifact_json()


def main() -> int:
    migrate()
    print("\nP3 reproduce layout completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
