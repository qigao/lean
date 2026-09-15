"""Catch changed schedules, extra propagation, and misreported per-agent results."""

import json
import subprocess
import sys
import unittest
from pathlib import Path


class BBBirthTimingExperimentTests(unittest.TestCase):
    def test_cli_compares_controlled_histories_with_hand_derived_results(self):
        process = subprocess.run(
            [sys.executable, "-m", "examples.bb_birth_timing"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        report = json.loads(process.stdout)
        cases = report["scenarios"]
        self.assertEqual([case["schedule"] for case in cases],
                         ["BBII", "BIBI", "IIBB"])
        expected = (
            ([0.75, 0.6875, 0.625, 0.25], [3, 5, 3, 1], [4, 4, 4, 3]),
            ([0.75, 0.6875, 0.625, 0.25], [3, 5, 3, 1], [4, 4, 4, 2]),
            ([0.75, 0.75, 0.5625, 0.0], [3, 4, 2, 0], [4, 4, 2, 1]),
        )
        for case, (beliefs, exposures, rounds) in zip(cases, expected):
            with self.subTest(schedule=case["schedule"]):
                self.assertEqual(case["final_edges"], [[0, 1], [1, 2], [2, 3]])
                self.assertEqual(case["fitness"], ["1"] * 4)
                self.assertEqual(case["trace_mass"], "1/8")
                self.assertEqual(case["birth_targets"], [[1], [2]])
                self.assertEqual(case["propagation_rounds_by_agent"], rounds)
                self.assertEqual([f["tick"] for f in case["frames"]], list(range(5)))
                self.assertEqual(case["frames"][0], {
                    "tick": 0, "epoch": 0, "local_round": 0,
                    "agent_ids": ["0", "1"], "beliefs": [1.0, 0.0],
                    "exposures": [0, 0],
                })
                final = case["frames"][-1]
                self.assertEqual(final["agent_ids"], ["0", "1", "2", "3"])
                self.assertEqual(final["beliefs"], beliefs)
                self.assertEqual(final["exposures"], exposures)
        # Identical final profiles and network; timings alone may differ.
        self.assertEqual(cases[0]["final_profiles"], cases[1]["final_profiles"])
        self.assertEqual(cases[0]["final_profiles"], cases[2]["final_profiles"])
        self.assertEqual(cases[2]["frames"][3], {
            "tick": 3, "epoch": 1, "local_round": 1,
            "agent_ids": ["0", "1", "2"],
            "beliefs": [0.75, 0.75, 0.375], "exposures": [2, 3, 1],
        })
        self.assertEqual(
            [(f["epoch"], f["local_round"]) for f in cases[2]["frames"]],
            [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
        )


if __name__ == "__main__":
    unittest.main()
