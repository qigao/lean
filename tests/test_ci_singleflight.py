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
        self.assertEqual(self.content.count("      queue: max"), 3)
        self.assertEqual(self.content.count("      cancel-in-progress: false"), 3)
        self.assertIn("    needs: python-tests", self.content)
        self.assertIn("    needs: lean-proof", self.content)

    def test_warm_cache_inputs(self):
        for name in ("cache: pip", "cache: npm", "~/.elan",
                     ".lake/packages", ".lake/build", "~/.cache/ms-playwright"):
            with self.subTest(name=name):
                self.assertIn(name, self.content)

    def test_full_gate_unchanged(self):
        for command in ("python -m unittest discover -s tests -v",
                        "lake build", "tools/check_fitness_abm.sh",
                        "tools/check_fitness_abm_runtime.sh",
                        "tools/check_fitness_abm_pathn.sh",
                        "npm test -- --run", "npm run typecheck",
                        "npx playwright test e2e/law-firm.spec.ts"):
            with self.subTest(command=command):
                self.assertIn(command, self.content)
        self.assertNotIn("continue-on-error:", self.content)
