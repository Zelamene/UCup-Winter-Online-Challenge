#!/usr/bin/env python3
"""Build a v18 manifest by overriding accepted-v17 cases with Phase 15 selections.

The selection file is a JSON object with a ``cases`` mapping from case index to
``{"trajectories": [...], "combine": ..., "mode": ...}``. Every trajectory is
copied verbatim, so scalar and per-step schedule configs are both supported.
Unchanged cases and public case 9 are copied byte-for-byte from v17.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import phase15_common as p  # noqa: E402

ROOT = TOOLS_DIR.parent
DEFAULT_BASE = p.V17_MANIFEST
DEFAULT_OUT = ROOT / "artifacts" / "phase15_case_manifest_v18.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    manifest = copy.deepcopy(json.loads(args.base.read_text(encoding="utf-8")))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selected = selection.get("cases", {})
    changed: list[int] = []
    for key, override in selected.items():
        case_index = int(key)
        case = p.CASE_SPECS[case_index - 1]
        _, n, k, _, snr = case
        if case_index == 9:
            raise ValueError("public case 9 must remain byte-identical to v17")
        matches = [item for item in manifest["cases"] if item["n"] == n and item["k"] == k]
        if len(matches) != 1:
            raise ValueError(f"base manifest does not have exactly one case {(n, k)}")
        trajectories = override["trajectories"]
        if not 1 <= len(trajectories) <= 6:
            raise ValueError(f"case {case_index}: expected 1..6 trajectories, got {len(trajectories)}")
        target = matches[0]
        target["trajectories"] = copy.deepcopy(trajectories)
        target["mode"] = override.get("mode", target.get("mode", "bsb"))
        target["combine"] = override.get("combine", target.get("combine", "energy"))
        target["steps"] = 50
        target["snr"] = float(snr)
        changed.append(case_index)

    manifest["base"] = str(args.base.relative_to(ROOT)) if args.base.is_relative_to(ROOT) else str(args.base)
    manifest["phase"] = 15
    manifest["changed_cases"] = sorted(changed)
    p.write_json(args.out, manifest)
    print(json.dumps({
        "out": str(args.out),
        "base": str(args.base),
        "changed_cases": sorted(changed),
        "case9_unchanged": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
