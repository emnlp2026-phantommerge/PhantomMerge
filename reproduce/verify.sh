#!/usr/bin/env bash
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
