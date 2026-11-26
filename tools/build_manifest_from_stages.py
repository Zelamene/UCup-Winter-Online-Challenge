#!/usr/bin/env python3
"""Build a v14 manifest from a base manifest plus named Stage A/B results.

Every Stage A/B file replaces exactly the case named by its ``(n, k, snr)``.
All other cases are copied unchanged. The script prints the per-case diff and
refuses to write output if a non-target case differs from the base.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from build_v13_manifest import bsb_scheduled_trajectory, bsb_trajectory  # noqa: E402

ROOT = TOOLS_DIR.parent
DEFAULT_BASE = ROOT / "artifacts" / "phase9_case_manifest_v13c.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "phase10_case_manifest_v14.json"


def case_key(n: int, k: int, snr: float) -> tuple[int, int, float]:
    return int(n), int(k), round(float(snr), 4)


def validate_identity(artifact: dict, kind: str) -> tuple[int, int, float]:
    for key in ("n", "k", "snr"):
        if key not in artifact:
            raise ValueError(f"{kind} artifact has no {key!r} field")
    n, k, snr = int(artifact["n"]), int(artifact["k"]), float(artifact["snr"])
    if not 0 < k < n or not math.isfinite(snr):
        raise ValueError(f"{kind} artifact has invalid case identity: n={n}, k={k}, snr={snr}")
    return case_key(n, k, snr)


def case_from_stage_a(artifact: dict) -> dict:
    if not artifact.get("gate_passed", False):
        raise ValueError("Stage A gate did not pass")
    if artifact.get("needs_boundary_rerun", False):
        raise ValueError("Stage A best config is on a search bound; rerun with extended ranges")
    traj = bsb_trajectory(artifact["best_config"])
    return {
        "n": int(artifact["n"]),
        "k": int(artifact["k"]),
        "snr": float(artifact["snr"]),
        "mode": "bsb",
        "combine": "energy",
        "trajectories": [traj],
        "dt": float(traj["dt"]),
        "steps": 50,
    }


def case_from_stage_b(artifact: dict) -> dict:
    gate = artifact.get("gate", {})
    if not gate.get("all_seeds_ge_3000", False):
        raise ValueError("Stage B gate did not pass on both validation seeds")
    if not gate.get("schedules_beat_flat", True):
        raise ValueError("Stage B schedules did not beat flat Stage A; use the Stage A file")
    traj = bsb_scheduled_trajectory(artifact)
    dt = traj["dt"]
    if isinstance(dt, list):
        dt = sum(float(value) for value in dt) / len(dt)
    return {
        "n": int(artifact["n"]),
        "k": int(artifact["k"]),
        "snr": float(artifact["snr"]),
        "mode": "bsb",
        "combine": "energy",
        "trajectories": [traj],
        "dt": float(dt),
        "steps": 50,
    }


def changed_fields(before: dict, after: dict) -> list[str]:
    return sorted(
        key for key in set(before) | set(after) if before.get(key) != after.get(key)
    )


def build(base: dict, targets: dict[tuple[int, int, float], dict]) -> tuple[dict, list[dict]]:
    old_cases = copy.deepcopy(base.get("cases", []))
    new_cases: list[dict] = []
    seen: set[tuple[int, int, float]] = set()
    diffs: list[dict] = []

    for index, old_case in enumerate(old_cases, start=1):
        if "n" not in old_case or "k" not in old_case or "snr" not in old_case:
            raise ValueError(f"case {index} has no (n, k, snr) identity")
        key = case_key(old_case["n"], old_case["k"], old_case["snr"])
        if key in targets:
            new_case = copy.deepcopy(targets[key])
            fields = changed_fields(old_case, new_case)
            if not fields:
                raise ValueError(f"case {index} {key}: target is identical to the base case")
            seen.add(key)
            diffs.append({"index": index, "case": list(key), "changed_fields": fields})
        else:
            new_case = copy.deepcopy(old_case)
        new_cases.append(new_case)

    missing = sorted(set(targets) - seen)
    if missing:
        raise ValueError(f"target cases not found in base manifest: {missing}")

    for index, (old_case, new_case) in enumerate(zip(old_cases, new_cases), start=1):
        key = case_key(old_case["n"], old_case["k"], old_case["snr"])
        if key not in targets and old_case != new_case:
            raise ValueError(f"case {index} {key} changed but is not a target")

    manifest = copy.deepcopy(base)
    manifest["cases"] = new_cases
    return manifest, diffs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--stage-a", type=Path, action="append", default=[],
        help="flat Stage A result to install; repeat for multiple cases",
    )
    parser.add_argument(
        "--stage-b", type=Path, action="append", default=[],
        help="scheduled Stage B result to install; repeat for multiple cases",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    targets: dict[tuple[int, int, float], dict] = {}

    for kind, paths, converter in (
        ("Stage A", args.stage_a, case_from_stage_a),
        ("Stage B", args.stage_b, case_from_stage_b),
    ):
        for path in paths:
            artifact = json.loads(path.read_text(encoding="utf-8"))
            key = validate_identity(artifact, kind)
            if key in targets:
                raise ValueError(f"duplicate target case {key} in {path}")
            targets[key] = converter(artifact)
            print(f"{kind}: N={key[0]} K={key[1]} snr={key[2]:g} <- {path}")

    if not targets:
        raise SystemExit("no Stage A/B files supplied; refusing to write an unchanged manifest")

    manifest, diffs = build(base, targets)
    print("\nManifest diff:")
    for item in diffs:
        print(f"  case {item['index']:>2d} {tuple(item['case'])}: {', '.join(item['changed_fields'])}")
    print(f"  unchanged cases: {len(manifest['cases']) - len(diffs)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
