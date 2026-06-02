#!/usr/bin/env bash
# Tier 1 — verify sealed Table 1 (no LLM judge; no rollouts in this release).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/reproduce/table1_characterization.sh" "$@"
