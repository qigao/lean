"""Coverage gate regressions for public Lean mesh theorem reports."""
import unittest
from tools.check_mesh_proof_coverage import required_theorems

class MeshProofCoverageTests(unittest.TestCase):
    source = ("namespace NarrativeDynamics\n"
              "theorem left : True := by trivial\n"
              "theorem MeshWalk.right : True := by trivial\n"
              "end NarrativeDynamics\n")
    reports = ("#print axioms NarrativeDynamics.left\n"
               "#print axioms NarrativeDynamics.MeshWalk.right\n")

    def test_complete_coverage(self):
        self.assertEqual(required_theorems([self.source], self.reports),
                         ("NarrativeDynamics.left", "NarrativeDynamics.MeshWalk.right"))

    def test_missing_new_theorem_is_red(self):
        with self.assertRaisesRegex(ValueError, "missing="):
            required_theorems(
                [self.source + "theorem future : True := by trivial\n"], self.reports
            )

    def test_stale_report_is_red(self):
        with self.assertRaisesRegex(ValueError, "stale="):
            required_theorems([self.source],
                self.reports + "#print axioms NarrativeDynamics.deleted\n")

    def test_duplicate_report_is_red(self):
        with self.assertRaisesRegex(ValueError, "duplicate Lean"):
            required_theorems([self.source],
                self.reports + "#print axioms NarrativeDynamics.left\n")

    def test_duplicate_theorem_is_red(self):
        with self.assertRaisesRegex(ValueError, "duplicate public"):
            required_theorems([self.source, self.source], self.reports)

    def test_empty_source_is_red(self):
        with self.assertRaisesRegex(ValueError, "no declarations"):
            required_theorems(["def unused : Nat := 0\n"], "")
