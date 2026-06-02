#!/usr/bin/env bash
# Regenerate appendix figure PDFs/PNGs (CPU, matplotlib). Requires: pip install matplotlib pandas numpy.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
PY="${PYTHON:-python3}"
$PY make_fig4.py
$PY make_fig5_oc_margin.py
$PY make_fig6_mitigation.py
$PY appendix.py
echo "Outputs under $DIR/output/"
