#!/usr/bin/env bash
# Optional: re-run OC-BCP mechanism exports (see probes/run_p3_suite.py).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
bash probes/run_p3.sh "$@"
