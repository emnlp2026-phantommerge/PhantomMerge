#!/usr/bin/env bash
# Tier 2 optional — re-run BCP-Detect pipeline (GPU; feature bank not shipped).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/reproduce/table3_bcp_detect.sh" "$@"
