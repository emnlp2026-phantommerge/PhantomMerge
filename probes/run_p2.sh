#!/usr/bin/env bash
# P2 full suite (CPU linear probes + optional ModernBERT on GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-python}"

"$PY" probes/validate_p0.py

# BoW only (fast):  SKIP_BERT=1 bash probes/run_p2.sh
# Full validity:    bash probes/run_p2.sh
EXTRA=()
if [[ "${SKIP_BERT:-0}" == "1" ]]; then
  EXTRA+=(--skip-bert)
fi

"$PY" probes/run_p2_suite.py "${EXTRA[@]}"
echo "P2 DONE — see results/table3_representation/bcp_detect.json (or runs/probe/ if re-run)"
