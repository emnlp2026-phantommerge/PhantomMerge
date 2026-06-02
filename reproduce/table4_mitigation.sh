#!/usr/bin/env bash
# Optional: re-run MSPS / gating (GPU; sealed Table 4 is authoritative).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python probes/run_p4_suite.py --skip-anchor-only "$@"
python probes/export_appendix_tables.py --out-dir results/appendix
