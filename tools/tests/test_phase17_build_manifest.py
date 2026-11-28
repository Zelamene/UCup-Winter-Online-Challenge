import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "tools" / "phase17_build_manifest.py"
BASE = ROOT / "artifacts" / "phase16_case_manifest_v19.json"


def build_manifest(configs: dict, directory: Path) -> dict:
    config_path = directory / "configs.json"
    output = directory / "manifest.json"
    config_path.write_text(json.dumps(configs), encoding="utf-8")
    subprocess.run(
        [sys.executable, str(BUILDER), "--configs", str(config_path),
         "--output", str(output), "--version", "test"],
        check=True, cwd=ROOT, capture_output=True, text=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


class Phase17BuildManifestTests(unittest.TestCase):
    def test_kspin_replacement_drops_bsb_only_parity_scale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_manifest({
                "5": {
                    "mode": "kspin",
                    "combine": "energy",
                    "dt": 0.4,
                    "trajectories": [{
                        "tau": 5.0,
                        "xi": 0.6,
                        "delta": 3.44,
                        "dt": 0.4,
                        "beta": 0.15,
                        "g": 0.98,
                        "clip": 6.0,
                        "joint_pairs": 0,
                        "parity_scale": 0.6375,
                    }],
                },
            }, Path(directory))
        self.assertEqual(manifest["changed_cases"], [5])
        trajectory = manifest["cases"][4]["trajectories"][0]
        self.assertNotIn("parity_scale", trajectory)
        self.assertEqual(trajectory["lambd"], 1.0)

    def test_unchanged_cases_stay_identical(self) -> None:
        base = json.loads(BASE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_manifest({
                "8": {
                    "mode": "kspin",
                    "trajectories": [{
                        "tau": 6.0, "xi": 0.6, "delta": 3.44, "dt": 0.4,
                        "beta": 0.15, "g": 1.05, "clip": 6.0,
                    }],
                },
            }, Path(directory))
        changed = set(manifest["changed_cases"])
        self.assertEqual(changed, {8})
        for index in range(12):
            if index + 1 not in changed:
                self.assertEqual(manifest["cases"][index], base["cases"][index])


if __name__ == "__main__":
    unittest.main()
