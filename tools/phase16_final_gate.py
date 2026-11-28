#!/usr/bin/env python3
"""Run the complete v19 K-spin acceptance checklist against accepted v18b."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_step(
    name: str,
    command: list[str],
    steps: list[dict],
    json_output: Path | None = None,
    timeout: float = 1800.0,
) -> bool:
    print(f"\n== {name} ==", flush=True)
    started = time.perf_counter()
    result: subprocess.CompletedProcess[str] | None = None
    for attempt in (1, 2):
        attempt_started = time.perf_counter()
        try:
            result = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, timeout=timeout
            )
            break
        except subprocess.TimeoutExpired:
            tail = [f"attempt {attempt} timed out after {timeout:.0f}s"]
            if attempt == 1:
                print(tail[0] + "; retrying once", flush=True)
                continue
            steps.append({
                "name": name, "ok": False, "returncode": None,
                "seconds": round(time.perf_counter() - started, 3), "tail": tail,
            })
            print("\n".join(tail), flush=True)
            return False
    if result is None:
        raise AssertionError("unreachable")
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
    parser.add_argument("--v18-binary", type=Path, required=True)
    parser.add_argument("--changed-cases", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "phase16" / "final")
    parser.add_argument("--max-judge-seconds", type=float, default=60.0)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    steps: list[dict] = []
    candidate = out / "v19.exe"
    public_output = out / "v19_public.out"

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
    public_ok = public_hash.upper() == PUBLIC_SHA256.upper()
    steps.append({
        "name": "public case 9 byte identity", "ok": public_ok,
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
    run_step(
        "gate seeds 2024 and 31337 vs v18b",
        [sys.executable, "tools/phase16_case_gate.py",
         "--manifest", str(args.manifest.resolve()),
         "--seeds", "2024", "31337",
         "--min-gain", "2000",
         "--output", str(out / "case_gate.json")],
        steps, out / "case_gate.json",
    )

    gate_payload = json.loads((out / "case_gate.json").read_text(encoding="utf-8"))
    by_seed: dict[int, list[dict]] = {}
    for item in gate_payload.get("results", []):
        by_seed.setdefault(int(item["seed"]), []).append(item)
    total_gains = {
        seed: round(sum(float(item["gain"]) for item in items), 3)
        for seed, items in by_seed.items()
    }
    minimum_total_ok = all(value >= 15_000.0 for value in total_gains.values())
    cases_up_ok = all(
        item["gain"] > 0.0
        for item in gate_payload.get("results", [])
        if int(item["case"]) in (4, 10)
    )
    steps.append({
        "name": "minimum acceptance margin vs v18b",
        "ok": bool(minimum_total_ok and cases_up_ok),
        "tail": [f"total gains={total_gains}", f"cases 4 and 10 up={cases_up_ok}"],
    })

    run_step(
        "fresh seed 6060 vs v18b",
        [sys.executable, "tools/phase14_fresh_seed_check.py",
         "--v16-binary", str(args.v18_binary.resolve()), "--v17-binary", str(candidate),
         "--changed-cases", *[str(value) for value in args.changed_cases],
         "--bootstrap-resamples", "1000",
         "--output", str(out / "fresh_seed_6060.json")],
        steps, out / "fresh_seed_6060.json",
    )
    run_step(
        "runtime judge estimate",
        [sys.executable, "tools/phase16_runtime_estimate.py",
         "--baseline-binary", str(args.v18_binary.resolve()), "--candidate-binary", str(candidate),
         "--max-judge-seconds", str(args.max_judge_seconds),
         "--output", str(out / "runtime.json")],
        steps, out / "runtime.json",
    )

    payload = {
        "manifest": str(args.manifest.resolve()),
        "source": str(args.source.resolve()),
        "v18_binary": str(args.v18_binary.resolve()),
        "changed_cases": args.changed_cases,
        "total_gate_gains": total_gains,
        "steps": steps,
        "all_passed": all(item["ok"] for item in steps),
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    gate_path = out / "final_gate.json"
    gate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nSaved {gate_path}")
    print(f"all_passed={payload['all_passed']}")
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
