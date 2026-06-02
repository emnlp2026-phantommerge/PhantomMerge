#!/usr/bin/env bash
# Tier 1: verify sealed Table 1 counts (no judge re-run).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python scripts/verify_paper_counts.py
python scripts/verify_results_integrity.py
