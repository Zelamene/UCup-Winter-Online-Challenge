#!/usr/bin/env python3
"""Run the complete Phase 14 acceptance checklist for a v17 submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

from binary_io import run_binary  # noqa: E402

ROOT = TOOLS_DIR.parent
GPP = Path(r"C:\Users\shazi\Downloads\UCup-Winter-Online-Challenge\artifacts\build\w64devkit\bin\g++.exe")
PUBLIC_INPUT = ROOT / "Problem statements" / "publish.in"
PUBLIC_V16_SHA256 = "1BBF2BBAA56E6AD871BB6FFE7F0AC4D501C31F3CE7A684195CAFD8F1AAA8723A"
GUARD_MANIFEST = ROOT / "artifacts" / "phase14" / "guard_manifest.json"
CACHE_DIR = ROOT / "artifacts" / "baseline_cache_v16"


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
    json_key: str = "all_passed",
) -> bool:
    print(f"\n== {name} ==", flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    tail = (result.stdout + result.stderr).strip().splitlines()[-12:]
    ok = result.returncode == 0
    if ok and json_output is not None:
        try:
            payload = json.loads(json_output.read_text(encoding="utf-8"))
            ok = bool(payload.get(json_key, False))
            tail = [json.dumps(payload.get("gate", payload), indent=2)]
        except (OSError, ValueError) as exc:
            ok = False
            tail = [f"could not read {json_output}: {exc}"]
    steps.append({
        "name": name,
        "ok": ok,
        "returncode": result.returncode,
        "seconds": round(time.perf_counter() - started, 3),
        "tail": tail,
    })
    print("\n".join(tail), flush=True)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--v16-binary", type=Path, required=True)
    parser.add_argument("--changed-cases", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "phase14")
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    steps: list[dict] = []
    v17_binary = out / "v17_final.exe"
    public_output = out / "v17_public.out"
    started = time.perf_counter()

    if not GPP.exists():
        raise SystemExit(f"missing strict compiler: {GPP}")

    ok_compile = run_step(
        "strict C++17 compile",
        [str(GPP), "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic",
         str(args.source.resolve()), "-o", str(v17_binary)],
        steps,
    )
    if not ok_compile:
        raise SystemExit("compile failed")

    ok_a0_shift = True
    shift_path = out / "bench_v2_shift.json"
    if shift_path.exists():
        payload = json.loads(shift_path.read_text(encoding="utf-8"))
        ok_a0_shift = bool(payload.get("gate", {}).get("all_within_1k", False))
        steps.append({"name": "A0 bench sigma shift", "ok": ok_a0_shift,
                      "tail": [json.dumps(payload["gate"], indent=2)]})
    else:
        ok_a0_shift = False
        steps.append({"name": "A0 bench sigma shift", "ok": False,
                      "tail": [f"missing {shift_path}"]})

    cache_files = sorted(CACHE_DIR.glob("*.npy"))
    ok_a0_cache = len(cache_files) == 48
    steps.append({"name": "A0 v16 cache inventory", "ok": ok_a0_cache,
                  "tail": [f"{len(cache_files)} cache files in {CACHE_DIR}; expected 48"]})

    run_step(
        "unit tests",
        [sys.executable, "-m", "unittest", "discover", "-s", "tools/tests",
         "-p", "test_*.py"],
        steps,
    )
    run_step(
        "offline isolation",
        [sys.executable, "tools/ceiling/check_offline_isolation.py", "--strict"],
        steps,
    )
    elapsed = run_binary(v17_binary, PUBLIC_INPUT, public_output)
    public_hash = sha256(public_output)
    ok_public = public_hash.upper() == PUBLIC_V16_SHA256.upper()

    run_step(
        "manifest validator",
        [sys.executable, "tools/validate_submission_v3.py",
         "--manifest", str(args.manifest.resolve()),
         "--submission", str(args.source.resolve()),
         "--public-output", str(public_output),
         "--output", str(out / "validator.json")],
        steps,
        out / "validator.json",
    )

    steps.append({
        "name": "public case byte identity",
        "ok": ok_public,
        "tail": [f"hash={public_hash}", f"expected={PUBLIC_V16_SHA256}",
                 f"seconds={elapsed:.3f}"],
    })

    run_step(
        "three-way bit identity",
        [sys.executable, "tools/phase14_equivalence.py",
         "--manifest", str(args.manifest.resolve()),
         "--binary", str(v17_binary),
         "--cases", *[str(value) for value in args.changed_cases],
         "--frames", "3",
         "--output", str(out / "equivalence.json")],
        steps,
        out / "equivalence.json",
    )

    guard_source = out / "guard_joint.cpp"
    guard_binary = out / "guard_joint.exe"
    ok_guard_source = run_step(
        "export guard manifest",
        [sys.executable, "tools/export_submission_cpp_v3.py",
         "--manifest", str(GUARD_MANIFEST), "--output", str(guard_source)],
        steps,
    )
    if ok_guard_source:
        ok_guard = run_step(
            "strict compile guard binary",
            [str(GPP), "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic",
             str(guard_source), "-o", str(guard_binary)],
            steps,
        )
        if ok_guard:
            run_step(
                "consistency guard",
                [sys.executable, "tools/phase14_guard_check.py",
                 "--public-input", str(PUBLIC_INPUT),
                 "--joint-binary", str(guard_binary),
                 "--reference-binary", str(args.v16_binary.resolve()),
                 "--output", str(out / "guard_check.json")],
                steps,
                out / "guard_check.json",
            )

    run_step(
        "fresh seed 6060",
        [sys.executable, "tools/phase14_fresh_seed_check.py",
         "--v16-binary", str(args.v16_binary.resolve()),
         "--v17-binary", str(v17_binary),
         "--changed-cases", *[str(value) for value in args.changed_cases],
         "--output", str(out / "fresh_seed_6060.json")],
        steps,
        out / "fresh_seed_6060.json",
    )
    run_step(
        "runtime guard",
        [sys.executable, "tools/phase14_runtime_check.py",
         "--v16-binary", str(args.v16_binary.resolve()),
         "--v17-binary", str(v17_binary),
         "--output", str(out / "runtime_check.json")],
        steps,
        out / "runtime_check.json",
    )

    payload = {
        "manifest": str(args.manifest.resolve()),
        "source": str(args.source.resolve()),
        "changed_cases": args.changed_cases,
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
