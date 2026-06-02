#!/usr/bin/env python3
"""Ensure anonymous GitHub URL appears in submission-facing files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
URL = "https://github.com/emnlp2026-phantommerge/PhantomMerge"

REQUIRED_IN = [
    "README.md",
    "ARTIFACT.json",
    "results/MANIFEST.json",
]

FORBIDDEN_PLACEHOLDER = "anonymous.4open.science"


def main() -> int:
    errors: list[str] = []

    for rel in REQUIRED_IN:
        path = REPO / rel
        if not path.is_file():
            errors.append(f"Missing {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        if URL not in text:
            errors.append(f"{rel}: missing anonymous repo URL")
        if FORBIDDEN_PLACEHOLDER in text:
            errors.append(f"{rel}: still contains placeholder host {FORBIDDEN_PLACEHOLDER!r}")

    artifact = REPO / "ARTIFACT.json"
    if artifact.is_file():
        data = json.loads(artifact.read_text(encoding="utf-8"))
        if data.get("repository") != URL:
            errors.append("ARTIFACT.json: repository field mismatch")
        if data.get("verify") != "bash reproduce/verify.sh":
            errors.append("ARTIFACT.json: verify should point to reproduce/verify.sh")

    if errors:
        print("audit_submission_url FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"audit_submission_url PASSED ({URL})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
