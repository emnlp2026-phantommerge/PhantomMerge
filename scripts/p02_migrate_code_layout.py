#!/usr/bin/env python3
"""P2: restructure code to binding/ + probes/ (paper-facing layout)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEGACY = REPO / "benchmarks"
BINDING = REPO / "binding"
PROBES = REPO / "probes"

MOVE_MAP = {
    "benchmarks/shoppingbench": "binding/shopping",
    "benchmarks/fhir_agentbench": "binding/fhir",
    "benchmarks/probe": "probes",
}


def _mv(src_rel: str, dst_rel: str) -> None:
    src = REPO / src_rel
    dst = REPO / dst_rel
    if not src.exists():
        return
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"MOVE {src_rel} -> {dst_rel}")


def _rm_pycache(root: Path) -> None:
    for cache in root.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
        print(f"REMOVED {cache.relative_to(REPO)}")


def _patch_file(path: Path, replacements: list[tuple[str, str]]) -> None:
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in replacements:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"PATCH {path.relative_to(REPO)}")


def _patch_probe_repo_depth() -> None:
    if not PROBES.is_dir():
        return
    for path in PROBES.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        new = text.replace("PROBE_ROOT.parents[2]", "PROBE_ROOT.parents[1]")
        if new != text:
            path.write_text(new, encoding="utf-8")
            print(f"PATCH depth {path.relative_to(REPO)}")


def _patch_binding_cross_refs() -> None:
    shopping_pm = BINDING / "shopping/src/analysis/pm_eval_vnext.py"
    _patch_file(
        shopping_pm,
        [
            (
                'parents[2].parent / "fhir_agentbench" / "config"',
                'parents[2].parent / "fhir" / "config"',
            ),
            ("See docs/METHODS.md", "See README.md"),
        ],
    )

    fhir_import = BINDING / "fhir/scripts/pm_eval_vnext_import.py"
    _patch_file(
        fhir_import,
        [
            ('parents[2] / "shoppingbench" / "src"', 'parents[2] / "shopping" / "src"'),
            ("docs/METHODS.md", "README.md"),
        ],
    )

    fhir_v2 = BINDING / "fhir/scripts/analyze_phantom_merge_fhir_v2.py"
    _patch_file(
        fhir_v2,
        [
            (
                'BENCH.parent / "shoppingbench" / "scripts"',
                'BENCH.parent / "shopping" / "scripts"',
            ),
            ("docs/METHODS.md", "README.md"),
        ],
    )

    shopping_v2 = BINDING / "shopping/scripts/analyze_phantom_merge_v2.py"
    _patch_file(shopping_v2, [("docs/METHODS.md", "README.md")])


def _patch_probes_paths_and_shells() -> None:
    paths_py = PROBES / "lib/paths.py"
    if paths_py.is_file():
        text = paths_py.read_text(encoding="utf-8")
        text = text.replace("parents[3]", "parents[2]")
        paths_py.write_text(text, encoding="utf-8")
        print(f"PATCH {paths_py.relative_to(REPO)}")

    bench_refs = [
        ('REPO / "benchmarks/fhir_agentbench/scripts"', 'REPO / "binding/fhir/scripts"'),
        (
            'REPO / "benchmarks/shoppingbench/scripts/analyze_phantom_merge_v2.py"',
            'REPO / "binding/shopping/scripts/analyze_phantom_merge_v2.py"',
        ),
    ]
    for path in PROBES.rglob("*.py"):
        _patch_file(path, bench_refs)

    shell_refs = [
        ("benchmarks/probe/", "probes/"),
        ("$ROOT/benchmarks/probe", "$ROOT/probes"),
        ("benchmarks/probe/README.md", "README.md"),
        ("docs/REPRODUCTION.md", "README.md"),
        ('"/../.."', '"/.."'),
        ("/../.. &&", "/.. &&"),
    ]
    for path in list(PROBES.rglob("*.sh")) + list((REPO / "experiments").rglob("*.sh")):
        _patch_file(path, shell_refs)


def migrate_directories() -> None:
    if BINDING.is_dir() and PROBES.is_dir() and not LEGACY.is_dir():
        print("P2 code layout already present.")
        return
    for src_rel, dst_rel in MOVE_MAP.items():
        _mv(src_rel, dst_rel)
    if LEGACY.is_dir():
        remaining = [p for p in LEGACY.rglob("*") if p.is_file()]
        if not remaining:
            LEGACY.rmdir()
            print("REMOVED empty benchmarks/")
        else:
            print(f"WARN benchmarks/ still has {len(remaining)} files")


def write_code_manifest_snippet() -> dict:
    return {
        "binding_shopping": "binding/shopping/",
        "binding_fhir": "binding/fhir/",
        "probes": "probes/",
        "pm_eval_core": "binding/shopping/src/analysis/pm_eval_vnext.py",
        "validate_p0": "probes/validate_p0.py",
    }


def update_artifact_json() -> None:
    path = REPO / "ARTIFACT.json"
    if not path.is_file():
        return
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    data["code"] = write_code_manifest_snippet()
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print("UPDATED ARTIFACT.json (code section)")


def main() -> int:
    migrate_directories()
    _rm_pycache(PROBES if PROBES.is_dir() else REPO)
    _patch_binding_cross_refs()
    _patch_probe_repo_depth()
    _patch_probes_paths_and_shells()
    update_artifact_json()
    print("\nP2 code layout migration completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
