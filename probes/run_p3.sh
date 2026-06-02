#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python probes/run_p3_suite.py
echo "P3 DONE — see results/supplemental/probe_auxiliary/p3_results_summary.json"
