#!/usr/bin/env python3
"""Judge-format input/output helpers shared by binary validation tools."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np


def write_input(case: dict, path: Path) -> None:
    """Write a case in judge format: header, H, LLR rows, G."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{case['n']} {case['k']} {case['b']} {case['snr_db']:g}"]
    lines.extend(" ".join(map(str, row.tolist())) for row in case["h"])
    lines.extend(" ".join(f"{value:.17g}" for value in row) for row in case["llr"])
    lines.extend(" ".join(map(str, row.tolist())) for row in case["g"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_binary(binary: Path, input_path: Path, output_path: Path) -> float:
    """Run a decoder binary on one input file and return elapsed seconds."""
    start = time.perf_counter()
    with input_path.open("rb") as fin, output_path.open("wb") as fout:
        proc = subprocess.run([str(binary)], stdin=fin, stdout=fout)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"{binary.name} exited with {proc.returncode} on {input_path.name}")
    return elapsed


def read_hard(output_path: Path, frames: int, n: int) -> np.ndarray:
    """Read one frame per line; bits may be space separated or contiguous."""
    lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != frames:
        raise ValueError(f"{output_path.name}: expected {frames} frames, found {len(lines)}")
    hard = np.empty((frames, n), dtype=np.int8)
    for index, line in enumerate(lines):
        digits = line.split() if len(line.split()) == n else list(line.strip())
        if len(digits) != n:
            raise ValueError(f"{output_path.name}: frame {index} has {len(digits)} bits, expected {n}")
        hard[index] = np.asarray([int(digit) for digit in digits], dtype=np.int8)
    return hard
