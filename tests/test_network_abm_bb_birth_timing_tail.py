"""Check equal post-growth budgets and signal/exposure measurements."""

import json
from pathlib import Path
import subprocess
import sys
import unittest


class BBBirthTimingTailTests(unittest.TestCase):
    def test_cli_samples_real_tail_with_hand_derived_first_round(self):
        process = subprocess.run(
            [sys.executable, "-m", "examples.bb_birth_timing_tail"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads(process.stdout)
        checkpoints = report["checkpoints"]
        self.assertEqual([c["extra_rounds"] for c in checkpoints],
                         [0, 1, 2, 4, 8, 16, 32, 64])
        self.assertEqual(checkpoints[0]["max_between_history_gap"], 0.25)
        first = checkpoints[1]
        self.assertEqual(first["global_tick"], 5)
        self.assertEqual(first["max_between_history_gap"], 0.15625)
        a, b, c = first["scenarios"]
        self.assertEqual(a["beliefs"], [0.71875, 0.6875, 0.65625, 0.4375])
        self.assertEqual(b["beliefs"], a["beliefs"])
        self.assertEqual(c["beliefs"], [0.75, 0.703125, 0.65625, 0.28125])
        self.assertEqual(a["within_history_spread"], 0.28125)
        for case in (a, b, c):
            self.assertEqual(case["new_exposures"], [1, 2, 1, 1])
            self.assertEqual(case["broadcasting"], [True, True, True, False])
        for point in checkpoints:
            self.assertEqual(point["global_tick"], 4 + point["extra_rounds"])
            self.assertEqual([s["schedule"] for s in point["scenarios"]],
                             ["BBII", "BIBI", "IIBB"])
            for case in point["scenarios"]:
                self.assertEqual(case["trace_mass"], "1/8")
                self.assertEqual(case["edges"], [[0, 1], [1, 2], [2, 3]])
                self.assertEqual(case["birth_count"], 2)
            self.assertEqual(point["scenarios"][0]["beliefs"],
                             point["scenarios"][1]["beliefs"])
        for case in checkpoints[0]["scenarios"]:
            self.assertEqual(case["new_exposures"], [0, 0, 0, 0])
        self.assertTrue(all(checkpoints[2]["scenarios"][0]["broadcasting"]))
        self.assertFalse(checkpoints[2]["scenarios"][2]["broadcasting"][-1])
        for case in checkpoints[3]["scenarios"]:
            self.assertTrue(all(case["broadcasting"]))
        last = checkpoints[-1]
        self.assertGreater(last["max_between_history_gap"], 0.018)
        self.assertLess(last["max_between_history_gap"], 0.019)
        for case, weighted_mean in zip(last["scenarios"], (127/192, 127/192, 87/128)):
            self.assertLess(case["within_history_spread"], 3e-9)
            # Independent degree-weighted means when all nodes first broadcast.
            values = case["beliefs"]
            self.assertAlmostEqual(
                (values[0] + 2*values[1] + 2*values[2] + values[3])/6,
                weighted_mean, places=13,
            )

    def test_extended_replay_keeps_original_prefix_and_counts_all_rounds(self):
        from examples.bb_birth_timing import run_experiment
        original = run_experiment()
        extended = run_experiment(extra_idle_rounds=2)
        for old, new in zip(original["scenarios"], extended["scenarios"]):
            self.assertEqual(new["frames"][:5], old["frames"])
            self.assertEqual(len(new["frames"]), 7)
            self.assertEqual(new["propagation_rounds_by_agent"],
                             [r + 2 for r in old["propagation_rounds_by_agent"]])
        for invalid in (-1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                run_experiment(extra_idle_rounds=invalid)


if __name__ == "__main__":
    unittest.main()
