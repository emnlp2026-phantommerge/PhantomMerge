#!/usr/bin/env bash
# Create a submission zip excluding git, venv, caches, and figure build outputs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT="$(dirname "$ROOT")"
NAME="PhantomMerge_emnlp2026_submission"
OUT="$PARENT/${NAME}.zip"

cd "$PARENT"
echo "Creating $OUT from $(basename "$ROOT") ..."

zip -r "$OUT" "$(basename "$ROOT")" \
  -x "$(basename "$ROOT")/.git/*" \
  -x "$(basename "$ROOT")/.git/**/*" \
  -x "*__pycache__/*" \
  -x "*.pyc" \
  -x "$(basename "$ROOT")/.venv/*" \
  -x "$(basename "$ROOT")/venv/*" \
  -x "$(basename "$ROOT")/figures/appendix/output/*" \
  -x "*.zip"

ls -lh "$OUT"
echo "Done. Upload or attach as supplementary material if allowed."
