#!/usr/bin/env bash
# Backward-compatible wrapper — canonical entry is reproduce/verify.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/reproduce/verify.sh" "$@"
