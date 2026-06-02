#!/usr/bin/env bash
# Tier 2 optional — re-run mitigation / MSPS suite (GPU; sealed Table 4 authoritative).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/reproduce/table4_mitigation.sh" "$@"
