#!/usr/bin/env python3
"""Verify sealed results match paper denominators and per-file checksums."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from repo_paths import COHORTS, PAPER_COUNTS, PER_TRAJECTORY, REPO_ROOT

PER_FILE = PER_TRAJECTORY


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def main() -> int:
    errors: list[str] = []
    counts = json.loads(PAPER_COUNTS.read_text(encoding="utf-8"))
    denominators = counts.get("paper_denominators", {})

    for cohort_key, expected_n in denominators.items():
        cohort_dir = COHORTS.get(cohort_key)
        if cohort_dir is None:
            errors.append(f"Unknown cohort {cohort_key}")
            continue
        per_path = cohort_dir / PER_FILE
        if not per_path.is_file():
            errors.append(f"Missing {per_path}")
            continue
        n_lines = count_jsonl(per_path)
        if n_lines != expected_n:
            errors.append(f"{cohort_key}: {n_lines} lines != paper N {expected_n}")

        manifest_path = cohort_dir / "cohort_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            sealed = manifest.get("sha256_per_sealed") or manifest.get("sha256_per_source")
            if sealed:
                actual = sha256_file(per_path)
                if actual != sealed:
                    errors.append(f"{cohort_key}: sha256 mismatch on per_trajectory")
            rel = manifest.get("source_per", "")
            if rel and not (REPO_ROOT / rel).resolve().samefile(per_path.resolve()):
                errors.append(f"{cohort_key}: source_per does not point to sealed file")

    if errors:
        print("verify_results_integrity FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("verify_results_integrity PASSED.")
    for cohort_key, expected_n in denominators.items():
        per_path = COHORTS[cohort_key] / PER_FILE
        print(f"  {cohort_key}: n={expected_n} sha256={sha256_file(per_path)[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
