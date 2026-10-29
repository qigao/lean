"""CI policy regression: exactly one serial superseding workflow."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/proof.yml"

class SingleflightCIContractTests(unittest.TestCase):
    def setUp(self):
        self.content = WORKFLOW.read_text(encoding="utf-8")

    def test_unique_authoritative_workflow(self):
        self.assertFalse((ROOT / ".github/workflows/world-studio.yml").exists())
        self.assertEqual(self.content.count("name: proof"), 1)
        self.assertNotIn("  proof-event:", self.content)

    def test_one_latest_head_per_pr(self):
        self.assertIn("  cancel-in-progress: true", self.content)
        self.assertIn("github.head_ref || github.ref_name", self.content)
        self.assertIn("  pull_request:", self.content)
        self.assertIn("    branches:\n      - master\n      - proof/narrative-dynamics-v0", self.content)

    def test_repo_runner_queue_preserves_other_prs(self):
        self.assertEqual(self.content.count("      queue: max"), 4)
        self.assertEqual(self.content.count("      cancel-in-progress: false"), 4)
        self.assertIn("    needs: mesh-projection", self.content)
        self.assertIn("    needs: python-tests", self.content)
        self.assertIn("    needs: lean-proof", self.content)

    def test_warm_cache_inputs(self):
        for name in ("cache: pip", "cache: npm", "~/.elan",
                     ".lake/packages", ".lake/build", "~/.cache/ms-playwright"):
            with self.subTest(name=name):
                self.assertIn(name, self.content)


    def test_preserves_design_doc_exclusions(self):
        self.assertEqual(self.content.count('      - "docs/superpowers/specs/**"'), 2)
        self.assertEqual(self.content.count('      - "docs/superpowers/plans/**"'), 2)

    def test_warm_cache_logs_and_build_wall_time(self):
        for marker in ("id: lean_products_cache",
                       "steps.lean_products_cache.outputs.cache-hit",
                       "Lean full library build elapsed=",
                       "id: chromium_cache",
                       "steps.chromium_cache.outputs.cache-hit"):
            with self.subTest(marker=marker):
                self.assertIn(marker, self.content)

    def test_runtime_finite_lean_gate_runs_before_expensive_bb(self):
        self.assertIn("V24 finite runtime mesh Lean conformance", self.content)
        self.assertIn("tools/check_mesh_evidence_lean.sh", self.content)
        self.assertLess(
            self.content.index("V24 finite runtime mesh Lean conformance"),
            self.content.index("BB ABM joint evolution contracts"),
        )

    def test_full_gate_unchanged(self):
        self.assertIn("python -m unittest tests.test_mesh_projection tests.test_mesh_evidence -v", self.content)
        self.assertIn("tools/check_mesh_proof_trust.sh", self.content)
        self.assertIn("tools/check_mesh_projection_trust.sh", self.content)
        for command in ("python -m unittest discover -s tests -v",
                        "lake build", "tools/check_fitness_abm.sh",
                        "tools/check_fitness_abm_runtime.sh",
                        "tools/check_fitness_abm_pathn.sh",
                        "npm test -- --run", "npm run typecheck",
                        "npx playwright test e2e/law-firm.spec.ts"):
            with self.subTest(command=command):
                self.assertIn(command, self.content)
        self.assertNotIn("continue-on-error:", self.content)
