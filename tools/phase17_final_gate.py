#!/usr/bin/env python3
"""Run the complete Phase 17 acceptance checklist against accepted v19."""

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
V19_MANIFEST = ROOT / "artifacts" / "phase16_case_manifest_v19.json"
V19_JUDGE_RESULT = ROOT / "submissions" / "v19_kspin" / "judge_result.json"


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
    timeout: float = 3600.0,
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


def unchanged_identity(base: dict, candidate: dict, changed: set[int]) -> dict:
    """Confirm every unchanged case is exactly the v19 case object."""
    base_cases = base.get("cases", [])
    cand_cases = candidate.get("cases", [])
    differences = [
        index for index in range(1, len(base_cases) + 1)
        if index not in changed and base_cases[index - 1] != cand_cases[index - 1]
    ]
    return {
        "ok": not differences,
        "different_unchanged_cases": differences,
        "changed_cases": sorted(changed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--v19-binary", type=Path, required=True)
    parser.add_argument("--changed-cases", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "phase17" / "final")
    parser.add_argument("--base-manifest", type=Path, default=V19_MANIFEST)
    parser.add_argument("--judge-result", type=Path, default=V19_JUDGE_RESULT)
    parser.add_argument("--public-sha256", default=PUBLIC_SHA256)
    parser.add_argument(
        "--combo-reference", type=Path, default=None,
        help="better single-track manifest for the required +1000 combination gate",
    )
    parser.add_argument("--min-gain", type=float, default=1500.0)
    parser.add_argument("--max-judge-seconds", type=float, default=60.0)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    steps: list[dict] = []
    candidate = out / "candidate.exe"
    public_output = out / "candidate_public.out"

    if not GPP.exists():
        raise SystemExit(f"missing strict compiler: {GPP}")
    if not run_step(
        "strict C++17 compile",
        [str(GPP), "-std=c++17", "-O2", "-Wall", "-Wextra", "-pedantic",
         str(args.source.resolve()), "-o", str(candidate)],
        steps,
    ):
        raise SystemExit("compile failed")

    source_bytes = int(args.source.stat().st_size)
    steps.append({
        "name": "source size under 512 KiB",
        "ok": source_bytes < 512 * 1024,
        "tail": [f"bytes={source_bytes}"],
    })

    run_step("unit tests", [sys.executable, "-m", "unittest", "discover", "-s", "tools/tests", "-p", "test_*.py"], steps)
    run_step("offline isolation", [sys.executable, "tools/ceiling/check_offline_isolation.py", "--strict"], steps)

    elapsed = run_binary(candidate, PUBLIC_INPUT, public_output)
    public_hash = sha256(public_output)
    public_ok = public_hash.upper() == args.public_sha256.upper()
    steps.append({
        "name": "public case 9 output",
        "ok": public_ok,
        "tail": [f"hash={public_hash}", f"expected={args.public_sha256}", f"seconds={elapsed:.3f}"],
    })

    run_step(
        "manifest validator",
        [sys.executable, "tools/validate_submission_v3.py",
         "--manifest", str(args.manifest.resolve()), "--submission", str(args.source.resolve()),
         "--public-output", str(public_output), "--output", str(out / "validator.json")],
        steps, out / "validator.json",
    )
    unchanged = unchanged_identity(
        json.loads(args.base_manifest.read_text(encoding="utf-8")),
        json.loads(args.manifest.read_text(encoding="utf-8")),
        set(args.changed_cases),
    )
    steps.append({
        "name": "unchanged cases byte-identical to v19",
        "ok": unchanged["ok"],
        "tail": [json.dumps(unchanged, indent=2)],
    })
    run_step(
        "three-way bit identity",
        [sys.executable, "tools/phase15_equivalence.py",
         "--manifest", str(args.manifest.resolve()), "--binary", str(candidate),
         "--cases", *[str(value) for value in args.changed_cases], "--frames", "3",
         "--output", str(out / "equivalence.json")],
        steps, out / "equivalence.json",
    )
    run_step(
        f"gate seeds 2024 and 31337 vs v19 (>= {args.min_gain:+.0f})",
        [sys.executable, "tools/phase17_case_gate.py",
         "--manifest", str(args.manifest.resolve()),
         "--cache-dir", str(ROOT / "artifacts" / "baseline_cache_v19"),
         "--seeds", "2024", "31337",
         "--min-gain", str(args.min_gain),
         "--output", str(out / "case_gate.json")],
        steps, out / "case_gate.json",
    )
    if args.combo_reference is not None:
        run_step(
            "combination beats better single manifest (>= +1000)",
            [sys.executable, "tools/phase17_combo_gate.py",
             "--reference", str(args.combo_reference.resolve()),
             "--candidate", str(args.manifest.resolve()),
             "--cache-dir", str(ROOT / "artifacts" / "baseline_cache_v19"),
             "--seeds", "2024", "31337", "--min-gain", "1000",
             "--output", str(out / "combo_gate.json")],
            steps, out / "combo_gate.json",
        )
    run_step(
        "fresh seed 6060 vs v19",
        [sys.executable, "tools/phase14_fresh_seed_check.py",
         "--v16-binary", str(args.v19_binary.resolve()), "--v17-binary", str(candidate),
         "--changed-cases", *[str(value) for value in args.changed_cases],
         "--bootstrap-resamples", "1000",
         "--output", str(out / "fresh_seed_6060.json")],
        steps, out / "fresh_seed_6060.json",
    )
    run_step(
        "strict SNR robustness ahead of v19",
        [sys.executable, "tools/phase15_robustness.py",
         "--v17-binary", str(args.v19_binary.resolve()), "--candidate-binary", str(candidate),
         "--changed-cases", *[str(value) for value in args.changed_cases],
         "--out", str(out / "robustness.json")],
        steps, out / "robustness.json",
    )
    robustness = json.loads((out / "robustness.json").read_text(encoding="utf-8"))
    snr_failures = [
        item for item in robustness.get("snr_cases", []) if not float(item["gain"]) > 0.0
    ]
    steps.append({
        "name": "SNR offsets strictly ahead",
        "ok": not snr_failures,
        "tail": [json.dumps(snr_failures, indent=2)],
    })
    run_step(
        "NaN/Inf guard at learned extremes",
        [sys.executable, "tools/phase17_guard_check.py",
         "--manifest", str(args.manifest.resolve()),
         "--changed-cases", *[str(value) for value in args.changed_cases],
         "--output", str(out / "guard_check.json")],
        steps, out / "guard_check.json",
    )
    run_step(
        "runtime judge estimate",
        [sys.executable, "tools/phase16_runtime_estimate.py",
         "--baseline-binary", str(args.v19_binary.resolve()), "--candidate-binary", str(candidate),
         "--judge-result", str(args.judge_result.resolve()),
         "--max-judge-seconds", str(args.max_judge_seconds),
         "--output", str(out / "runtime.json")],
        steps, out / "runtime.json",
    )

    gate_payload = json.loads((out / "case_gate.json").read_text(encoding="utf-8"))
    total_gains = {
        int(seed): round(sum(
            float(item["gain"]) for item in gate_payload.get("results", [])
            if int(item["seed"]) == int(seed)
        ), 3)
        for seed in {int(item["seed"]) for item in gate_payload.get("results", [])}
    }
    payload = {
        "manifest": str(args.manifest.resolve()),
        "source": str(args.source.resolve()),
        "v19_binary": str(args.v19_binary.resolve()),
        "combo_reference": (
            str(args.combo_reference.resolve()) if args.combo_reference is not None else None
        ),
        "changed_cases": args.changed_cases,
        "total_gate_gains": total_gains,
        "steps": steps,
        "all_passed": all(item["ok"] for item in steps),
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    gate_path = out / "final_gate.json"
    pc_write_json(gate_path, payload)
    print(f"\nSaved {gate_path}")
    print(f"all_passed={payload['all_passed']}")
    if not payload["all_passed"]:
        raise SystemExit(1)


def pc_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
