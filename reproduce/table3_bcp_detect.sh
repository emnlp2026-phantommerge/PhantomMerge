#!/usr/bin/env bash
# Optional: re-run BCP-Detect (GPU; feature bank .npy not shipped).
# Re-run BCP-Detect (GPU feature bank required — not included in release).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
bash probes/run_p2.sh "$@"
