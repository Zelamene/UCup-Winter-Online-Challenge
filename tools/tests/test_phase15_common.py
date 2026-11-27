#!/usr/bin/env python3
"""Tests for Phase 15 frame-matched evaluation helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

import phase15_common as p  # noqa: E402
import tune_cma_schedule as schedules  # noqa: E402


class Phase15CommonTests(unittest.TestCase):
    def test_v17_config_reproduces_cached_frames_on_changed_cases(self) -> None:
        for case_index in (4, 7, 10):
            _, n, k, _, snr = p.CASE_SPECS[case_index - 1]
            workspace = p.CaseWorkspace(n, k, 911, 300, snr=snr, frame_slice=40)
            config = p.load_v17_config(n, k)
            errors = p.per_frame_errors(p.decode_hard(workspace, config), workspace.case)
            self.assertTrue(np.array_equal(errors, workspace.baseline_errors), f"case {case_index}")

    def test_portfolio_selector_matches_repo_ensemble(self) -> None:
        workspace = p.CaseWorkspace(1296, 648, 911, 300, frame_slice=40)
        configs = [
            p.load_v17_config(1296, 648),
            p.with_overrides(p.load_v17_config(1296, 648), tau=2.0, pair_gamma=0.22),
            p.with_overrides(p.load_v17_config(1296, 648), joint_pairs=0, energy_lambda=6.2),
        ]
        hards = [p.decode_hard(workspace, config) for config in configs]
        mine = p.portfolio_from_hards(workspace, hards, configs)
        reference = workspace.sim.evaluate_ensemble(
            workspace.case["llr"], workspace.case["answers"], workspace.k,
            configs, combine="energy", steps=50, snr=workspace.snr,
        )
        self.assertTrue(np.array_equal(mine["selected_errors"], p.per_frame_errors(
            np.stack(hards)[np.argmin(np.stack([
                p.trajectory_energies(workspace, hard, config)
                for hard, config in zip(hards, configs)
            ]), axis=0), np.arange(workspace.frames)], workspace.case
        )))
        self.assertEqual(int(mine["selected_errors"].sum()), reference[2])

    def test_zero_log_params_reproduce_v17(self) -> None:
        workspace = p.CaseWorkspace(1296, 648, 911, 300, frame_slice=40)
        base = p.load_v17_config(1296, 648)
        config = schedules.config_from_log_params(base, np.zeros(schedules.DIM))
        errors = p.per_frame_errors(p.decode_hard_schedules(workspace, config), workspace.case)
        self.assertEqual(schedules.DIM, 33)
        self.assertTrue(np.array_equal(errors, workspace.baseline_errors))


if __name__ == "__main__":
    unittest.main()
