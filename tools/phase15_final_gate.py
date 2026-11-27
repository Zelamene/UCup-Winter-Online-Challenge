#!/usr/bin/env python3
"""Run the complete Phase 15 v18 acceptance checklist.

Every comparison is against the compiled accepted-v17 binary on identical
fixed-rule frames. The script writes a single ``final_gate.json`` with
``all_passed`` and refuses to report success if any step fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from binary_io import run_binary  # noqa: E402

ROOT = TOOLS_DIR.parent
GPP = Path(r"C:\Users\shazi\Downloads\UCup-Winter-Online-Challenge\artifacts\build\w64devkit\bin\g++.exe")
PUBLIC_INPUT = ROOT / "Problem statements" / "publish.in"
PUBLIC_SHA256 = "1BBF2BBAA56E6AD871BB6FFE7F0AC4D501C31F3CE7A684195CAFD8F1AAA8723A"
GUARD_MANIFEST = ROOT / "artifacts" / "phase14" / "guard_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_step(name: str, command: list[str], steps: list[dict], json_output: Path | None = None) -> bool:
    print(f"\n== {name} ==", flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    tail = (result.stdout + result.stderr).strip().splitlines()[-12:]
    ok = result.returncode == 0
    if ok and json_output is not None:
        try:
            payload = json.loads(json_output.read_text(encoding="utf-8"))
            ok = bool(payload.get("all_passed", False))
            tail = [json.dumps(payload.get("gate", payload), indent=2)[:4000]]
        except (OSError, ValueError) as exc:
            ok = False
            tail = [f"could not read {json_output}: {exc}"]
    steps.append({
        "name": name, "ok": ok, "returncode": result.returncode,
        "seconds": round(time.perf_counter() - started, 3), "tail": tail,
    })
    print("\n".join(tail), flush=True)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--v17-binary", type=Path, required=True)
    parser.add_argument("--changed-cases", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "phase15" / "final")
    parser.add_argument("--max-judge-seconds", type=float, default=60.0)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    gate_started = time.perf_counter()
    steps: list[dict] = []
    candidate = out / "v18_final.exe"
    public_output = out / "v18_public.out"

    if not GPP.exists():
        raise SystemExit(f"missing strict compiler: {GPP}")
    if not run_step(
        "strict C++17 compile",
        [str(GPP), "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic",
         str(args.source.resolve()), "-o", str(candidate)],
        steps,
    ):
        raise SystemExit("compile failed")

    run_step("unit tests", [sys.executable, "-m", "unittest", "discover", "-s", "tools/tests", "-p", "test_*.py"], steps)
    run_step("offline isolation", [sys.executable, "tools/ceiling/check_offline_isolation.py", "--strict"], steps)

    elapsed = run_binary(candidate, PUBLIC_INPUT, public_output)
    public_hash = sha256(public_output)
    ok_public = public_hash.upper() == PUBLIC_SHA256.upper()
    steps.append({
        "name": "public case 9 byte identity", "ok": ok_public,
        "tail": [f"hash={public_hash}", f"expected={PUBLIC_SHA256}", f"seconds={elapsed:.3f}"],
    })

    run_step(
        "manifest validator",
        [sys.executable, "tools/validate_submission_v3.py",
         "--manifest", str(args.manifest.resolve()), "--submission", str(args.source.resolve()),
         "--public-output", str(public_output), "--output", str(out / "validator.json")],
        steps, out / "validator.json",
    )
    run_step(
        "three-way bit identity",
        [sys.executable, "tools/phase15_equivalence.py",
         "--manifest", str(args.manifest.resolve()), "--binary", str(candidate),
         "--cases", *[str(value) for value in args.changed_cases], "--frames", "3",
         "--output", str(out / "equivalence.json")],
        steps, out / "equivalence.json",
    )

    guard_source = out / "guard.cpp"
    guard_binary = out / "guard.exe"
    if run_step(
        "export swapped-pair guard",
        [sys.executable, "tools/export_submission_cpp_v3.py",
         "--manifest", str(GUARD_MANIFEST), "--output", str(guard_source)],
        steps,
    ) and run_step(
        "strict compile guard binary",
        [str(GPP), "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic",
         str(guard_source), "-o", str(guard_binary)],
        steps,
    ):
        run_step(
            "swapped-pair consistency guard",
            [sys.executable, "tools/phase14_guard_check.py",
             "--public-input", str(PUBLIC_INPUT), "--joint-binary", str(guard_binary),
             "--reference-binary", str(args.v17_binary.resolve()),
             "--output", str(out / "guard_check.json")],
            steps, out / "guard_check.json",
        )

    run_step(
        "fresh seed 6060 vs v17",
        [sys.executable, "tools/phase14_fresh_seed_check.py",
         "--v16-binary", str(args.v17_binary.resolve()), "--v17-binary", str(candidate),
         "--changed-cases", *[str(value) for value in args.changed_cases],
         "--bootstrap-resamples", "1000",
         "--output", str(out / "fresh_seed_6060.json")],
        steps, out / "fresh_seed_6060.json",
    )
    run_step(
        "runtime vs v17",
        [sys.executable, "tools/phase14_runtime_check.py",
         "--v16-binary", str(args.v17_binary.resolve()), "--v17-binary", str(candidate),
         "--output", str(out / "runtime.json")],
        steps, out / "runtime.json",
    )
    runtime_payload = json.loads((out / "runtime.json").read_text(encoding="utf-8")) if (out / "runtime.json").exists() else {}
    worst_seconds = float(max((item.get("v17_seconds", 0.0) for item in runtime_payload.get("cases", [])), default=0.0))
    ok_judge = worst_seconds <= args.max_judge_seconds
    steps.append({
        "name": "absolute runtime judge estimate", "ok": ok_judge,
        "tail": [f"worst 500-frame case: {worst_seconds:.3f}s", f"cap: {args.max_judge_seconds:.1f}s"],
    })
    run_step(
        "SNR and schedule-extreme robustness",
        [sys.executable, "tools/phase15_robustness.py",
         "--v17-binary", str(args.v17_binary.resolve()), "--candidate-binary", str(candidate),
         "--changed-cases", *[str(value) for value in args.changed_cases],
         "--out", str(out / "robustness.json")],
        steps, out / "robustness.json",
    )

    cache_path = ROOT / "artifacts" / "phase15" / "step0_cache_check.json"
    ok_cache = cache_path.exists() and bool(json.loads(cache_path.read_text(encoding="utf-8")).get("all_passed", False))
    steps.append({
        "name": "v17 baseline cache identity", "ok": ok_cache,
        "tail": [f"{cache_path}: {'pass' if ok_cache else 'FAIL'}"],
    })

    payload = {
        "manifest": str(args.manifest.resolve()),
        "source": str(args.source.resolve()),
        "v17_binary": str(args.v17_binary.resolve()),
        "changed_cases": args.changed_cases,
        "steps": steps,
        "all_passed": all(item["ok"] for item in steps),
        "wall_seconds": round(time.perf_counter() - gate_started, 3),
    }
    gate_path = out / "final_gate.json"
    p_write(gate_path, payload)
    print(f"\nSaved {gate_path}")
    print(f"all_passed={payload['all_passed']}")
    if not payload["all_passed"]:
        raise SystemExit(1)


def p_write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
