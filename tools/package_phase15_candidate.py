#!/usr/bin/env python3
"""Package a Phase 15 v18 candidate archive without marking it submitted."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import phase15_common as p  # noqa: E402

ROOT = TOOLS_DIR.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--version", default="v18a")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--final-gate", type=Path)
    parser.add_argument("--validation", type=Path, nargs="*", default=[])
    args = parser.parse_args()

    target = (args.output_dir or ROOT / "submissions" / f"{args.version}_unfolded_portfolio").resolve()
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty archive: {target}")
    target.mkdir(parents=True, exist_ok=True)

    manifest_path = target / "manifest.json"
    source_path = target / "submission.cpp"
    shutil.copyfile(args.manifest, manifest_path)
    shutil.copyfile(args.source, source_path)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    changed = [int(value) for value in manifest.get("changed_cases", [])]
    validation = []
    for path in args.validation:
        validation.append({"path": str(path.resolve()), "data": json.loads(path.read_text(encoding="utf-8"))})
    gate_passed = None
    if args.final_gate is not None and args.final_gate.exists():
        gate_passed = bool(json.loads(args.final_gate.read_text(encoding="utf-8")).get("all_passed", False))
        shutil.copyfile(args.final_gate, target / "final_gate.json")

    status = {
        "version": args.version,
        "status": "prepared_not_submitted",
        "base": "v17",
        "changed_cases": changed,
        "joint_pairs": True,
        "energy_portfolio": True,
        "final_gate_passed": gate_passed,
        "submission_authorized": False,
        "user_decision": "awaiting explicit approval before any judge submission",
        "source_sha256": sha256(source_path),
        "source_bytes": source_path.stat().st_size,
    }
    p.write_json(target / "candidate_status.json", status)

    lines = [
        f"# {args.version} — Phase 15 energy-selected tau/xi portfolio",
        "",
        "Prepared candidate; **not submitted**. It remains v17 except for the changed cases listed below.",
        "",
        "Base: accepted v17 (`submissions/v17_joint_pairs/`), judge score `2,038,794`.",
        "",
        f"Changed cases: {', '.join(str(value) for value in changed) or 'none'}.",
        "",
        "Each changed case uses up to six QAIA-compliant bSB trajectories with tuned `tau`/`xi`/`delta`/`lambd`",
        "settings and exact C++/Python energy selection. All trajectories keep 50 steps, at most six",
        "trajectories, no per-edge state, and unchanged `±1` Ising semantics. Public case 9 and every",
        "unchanged case remain byte-identical to v17.",
        "",
    ]
    for item in validation:
        raw_cases = item["data"].get("cases", {})
        if isinstance(raw_cases, dict):
            cases = [(int(key), value) for key, value in raw_cases.items()]
        else:
            cases = [(int(value["case_index"]), value) for value in raw_cases]
        if cases:
            frame_count = item['data'].get('frames', item['data'].get('samples'))
            lines.append(f"Validation from `{Path(item['path']).name}` (seed {item['data'].get('seed')}, "
                         f"{frame_count} frames):")
            for key, value in sorted(cases):
                selection = value.get("selection", value)
                ci = selection.get("gain_ci_95") or []
                lines.append(f"- case {key}: gain `{selection.get('gain', 0):+.1f}`, "
                             f"CI `[{ci[0]:.1f}, {ci[1]:.1f}]`" if len(ci) == 2 else
                             f"- case {key}: gain `{selection.get('gain', 0):+.1f}`")
            lines.append("")
    lines.extend([
        f"Source SHA-256: `{status['source_sha256']}`.",
        f"Final gate: `{gate_passed}`.",
        "",
        "Do not submit without explicit user approval.",
        "",
    ])
    (target / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(target),
        "changed_cases": changed,
        "source_sha256": status["source_sha256"],
        "source_bytes": status["source_bytes"],
        "final_gate_passed": gate_passed,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
