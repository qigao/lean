"""CI policy: one superseding workflow and two bounded parallel runner lanes."""
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

    def test_two_parallel_bounded_runner_lanes_and_join(self):
        # Both lanes preserve unrelated PR jobs in their respective queues.
        self.assertEqual(self.content.count("      queue: max"), 4)
        self.assertEqual(self.content.count("      cancel-in-progress: false"), 4)
        self.assertEqual(
            self.content.count("group: lean-ci-runner-${{ github.repository }}"), 3
        )
        self.assertEqual(
            self.content.count("group: lean-ci-runner-lean-${{ github.repository }}"), 1
        )
        self.assertIn(
            "  verify:\n    # Join both independent test families even when one finishes RED.\n"
            "    needs: [python-tests, lean-proof]",
            self.content,
        )
        self.assertIn("  lean-proof:\n    name: Lean proof\n", self.content)
        self.assertEqual(self.content.count("    needs: mesh-projection"), 2)
        self.assertNotIn("    needs: python-tests\n", self.content)
        self.assertIn("    if: ${{ !cancelled() }}", self.content)

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

    def test_v24_math_proof_gates(self):
        for required in (
            "Runtime-derived mesh regression",
            "tests.test_mesh_projection",
            "V24 Lean mesh projection tests",
            "Mesh theorem trust audit",
            "V24 projection theorem trust audit",
        ):
            with self.subTest(required=required):
                self.assertIn(required, self.content)

    def test_extended_mesh_pin_and_negative_kernel_checks(self):
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("mesh_reach_lean.sha256", script)
        self.assertIn("sha256sum", script)
        self.assertIn("tools.mutate_mesh_reach_lean", script)
        self.assertIn("stale-path identity", script)
        self.assertIn("expected_failure", script)
        self.assertNotIn("sha256sum \"$paths\" > tests/fixtures", script)

    def test_runtime_review_vectors_gate_precedes_mesh_regressions(self):
        self.assertIn("Canonical V24 runtime vector conformance", self.content)
        self.assertIn("tools.render_mesh_proof_vectors", self.content)
        self.assertIn("conformance/mesh_runtime_vectors_v1.json", self.content)
        self.assertNotIn("--transport", self.content)
        self.assertIn('cmp -s "$pinned" "$generated"', self.content)
        self.assertLess(
            self.content.index("Canonical V24 runtime vector conformance"),
            self.content.index("Runtime-derived mesh regression"),
        )

    def test_r11_reviewed_source_checkpoint_vectors(self):
        self.assertIn("tools.render_social_topology_checkpoint_vectors", self.content)
        self.assertIn("conformance/social_topology_checkpoint_v1.json", self.content)
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("social_topology_checkpoint_v1.json", script)
        self.assertIn("cmp -s", script)

    def test_r11_real_social_memory_checkpoint_gate(self):
        self.assertIn("tests.test_social_topology_checkpoint", self.content)
        self.assertIn("tests.test_mesh_topology_revisions", self.content)
        self.assertLess(
            self.content.index("R1 immutable topology replay regression"),
            self.content.index("Runtime-derived mesh regression"),
        )

    def test_r1_source_revision_and_replay_gate(self):
        self.assertIn("R1 immutable topology replay regression", self.content)
        self.assertIn("tools.render_topology_revision_vectors", self.content)
        self.assertIn("tests.test_mesh_topology_revisions", self.content)
        self.assertIn("conformance/topology_revision_v1.json", self.content)
        self.assertLess(
            self.content.index("R1 immutable topology replay regression"),
            self.content.index("Runtime-derived mesh regression"),
        )
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("tools.render_topology_revision_vectors", script)
        self.assertIn("topology_revision_v1.json", script)
        self.assertIn("cmp -s", script)

    def test_runtime_generated_lean_gate(self):
        self.assertIn("python -m unittest tests.test_mesh_projection tests.test_mesh_evidence -v", self.content)
        self.assertIn("V24 finite runtime mesh Lean conformance", self.content)
        self.assertIn("tools/check_mesh_evidence_lean.sh", self.content)
        self.assertLess(
            self.content.index("V24 finite runtime mesh Lean conformance"),
            self.content.index("BB ABM joint evolution contracts"),
        )

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
