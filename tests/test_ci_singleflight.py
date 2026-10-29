"""Static CI guard: one authoritative, cache-aware, superseding workflow."""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/proof.yml"


class SingleFlightCIContractTests(unittest.TestCase):
    def setUp(self):
        self.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_workflow_owns_all_required_stages(self):
        for stage in ("mesh-projection:", "python-tests:", "lean-proof:",
                      "verify:", "V24 finite runtime mesh Lean conformance"):
            with self.subTest(stage=stage):
                self.assertIn(stage, self.workflow)
        self.assertFalse((ROOT / ".github/workflows/world-studio.yml").exists())

    def test_one_pr_only_latest_commit(self):
        self.assertIn("  cancel-in-progress: true", self.workflow)
        self.assertIn("github.head_ref || github.ref_name", self.workflow)
        self.assertIn("  pull_request:", self.workflow)
        self.assertIn("    branches:\n      - master\n      - proof/narrative-dynamics-v0", self.workflow)
        self.assertNotIn("proof-event:", self.workflow)

    def test_one_global_runner_fifo_with_no_dropped_prs(self):
        self.assertEqual(self.workflow.count("      queue: max"), 4)
        self.assertEqual(self.workflow.count("      cancel-in-progress: false"), 4)
        self.assertIn("    needs: mesh-projection", self.workflow)
        self.assertIn("    needs: python-tests", self.workflow)
        self.assertIn("    needs: lean-proof", self.workflow)

    def test_all_cache_sources_present(self):
        for entry in ("cache: pip", "cache: npm", "~/.elan",
                      ".lake/packages", ".lake/build", "~/.cache/ms-playwright"):
            with self.subTest(entry=entry):
                self.assertIn(entry, self.workflow)

    def test_finite_lean_conformance_fails_before_expensive_bb_stages(self):
        self.assertLess(
            self.workflow.index("V24 finite runtime mesh Lean conformance"),
            self.workflow.index("BB ABM joint evolution contracts"),
        )

    def test_no_test_gate_weakening(self):
        for entry in ("python -m unittest discover -s tests -v",
                      "python -m unittest", "lake build",
                      "tools/check_mesh_proof_trust.sh",
                      "tools/check_mesh_projection_trust.sh",
                      "tools/check_mesh_evidence_lean.sh",
                      "npm test -- --run", "npm run typecheck",
                      "npx playwright test e2e/law-firm.spec.ts"):
            with self.subTest(entry=entry):
                self.assertIn(entry, self.workflow)
        self.assertNotIn("continue-on-error:", self.workflow)
