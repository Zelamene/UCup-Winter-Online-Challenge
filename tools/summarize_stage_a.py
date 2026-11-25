#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cases = [
    ("N=648",  ROOT / "artifacts/stage_a_N648_R0.5.json"),
    ("N=1296", ROOT / "artifacts/stage_a_N1296_R0.5.json"),
    ("N=1944", ROOT / "artifacts/stage_a_N1944_R0.5.json"),
    ("N=3888", ROOT / "artifacts/stage_a_N3888_R0.5.json"),
]

total_gain = 0.0
for label, path in cases:
    d = json.loads(path.read_text())
    c = d["best_config"]
    gain = d["gain"]
    total_gain += gain
    ci = d["ci_95"]
    print(
        f"{label}: base={d['base_score']:.1f} bsb={d['val_score']:.1f} "
        f"gain={gain:+.1f}  CI=[{ci[0]:.1f},{ci[1]:.1f}]  "
        f"tau={c['tau']:.4f} xi={c['xi']:.4f} lam={c['lambd']:.4f} "
        f"delta={c['delta']:.4f} dt={c['dt']:.4f}"
    )

print(f"\nTotal expected gain on R=1/2 cases: +{total_gain:.1f}")
