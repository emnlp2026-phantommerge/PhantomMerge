#!/usr/bin/env python3
"""Print paper Table 1 counts from sealed results (no GPU)."""

from __future__ import annotations

import json

from repo_paths import PAPER_COUNTS


def main() -> None:
    if not PAPER_COUNTS.is_file():
        raise SystemExit(f"Missing {PAPER_COUNTS}")
    data = json.loads(PAPER_COUNTS.read_text(encoding="utf-8"))
    print("Paper denominators:", data.get("paper_denominators"))
    print()
    for key, cell in data.get("cells", {}).items():
        print(
            f"{key}: N={cell['paper_denominator_n']} "
            f"task_correct={cell['task_correct']} has_pm={cell['has_phantom_merge']} "
            f"task_correct∧pm={cell['task_correct_and_pm']}"
        )


if __name__ == "__main__":
    main()
