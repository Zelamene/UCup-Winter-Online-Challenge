# OFFLINE ANALYSIS ONLY — never imported by the exporter
"""Check that Phase 13 analysis tools stay out of the submission path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CEILING_DIR = Path(__file__).resolve().parent
if str(CEILING_DIR) not in sys.path:
    sys.path.insert(0, str(CEILING_DIR))

from common import (  # noqa: E402
    OFFLINE_HEADER,
    OFFLINE_SYMBOLS,
    ROOT,
    check_offline_header,
    resolve_path,
)

EXPORTERS = (
    "tools/export_submission_cpp.py",
    "tools/export_submission_cpp_v2.py",
    "tools/export_submission_cpp_v3.py",
    "tools/validate_submission_v3.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    issues: list[str] = []
    tool_files = sorted(CEILING_DIR.glob("*.py"))
    if not tool_files:
        issues.append("no Phase 13 tools found")
    for path in tool_files:
        try:
            check_offline_header(path)
        except (OSError, ValueError) as exc:
            issues.append(str(exc))

    for relative in EXPORTERS:
        path = resolve_path(relative)
        if not path.exists():
            issues.append(f"missing isolation target: {relative}")
            continue
        lowered = path.read_text(encoding="utf-8", errors="ignore").lower()
        forbidden = [symbol for symbol in OFFLINE_SYMBOLS if symbol.lower() in lowered]
        if relative.endswith("validate_submission_v3.py"):
            # The validator is expected to name the symbols so it can reject them.
            continue
        if forbidden:
            issues.append(f"{relative} mentions offline symbols: {forbidden}")

    print(f"checked {len(tool_files)} Phase 13 tools and {len(EXPORTERS)} submission-path files")
    print(f"required header: {OFFLINE_HEADER}")
    if issues:
        for issue in issues:
            print(f"ERROR {issue}")
        raise SystemExit(1)
    print("PASS: offline analysis tools are isolated from exporters")
    if args.strict:
        print("NOTE: --strict currently adds no extra checks")


if __name__ == "__main__":
    main()
