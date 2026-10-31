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
            "  verify:\n    # Join independently accepted Python and Lean families; skip Studio on upstream RED.\n"
            "    needs: [python-tests, lean-proof]",
            self.content,
        )
        self.assertIn("  lean-proof:\n    name: Lean proof\n", self.content)
        self.assertEqual(self.content.count("    needs: mesh-projection"), 2)
        self.assertNotIn("    needs: python-tests\n", self.content)
        self.assertIn("    if: ${{ success() }}", self.content)
        self.assertEqual(self.content.count("if: ${{ success()"), 3)
        self.assertNotIn("if: ${{ !cancelled()", self.content)

    def test_failed_mesh_source_gate_does_not_start_expensive_followups(self):
        self.assertIn("  python-tests:\n    needs: mesh-projection\n", self.content)
        self.assertIn("  lean-proof:\n    name: Lean proof\n    needs: mesh-projection\n", self.content)
        self.assertIn("  verify:\n", self.content)
        self.assertEqual(self.content.count("if: ${{ success()"), 3)
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

    def test_r2_m5g0_native_neutral_audit_trust_but_not_target_belief(self):
        self.assertIn(
            "R2 M5g0 V19 neutral audit trust learning without target-belief change",
            self.content,
        )
        self.assertIn("tests.test_network_neutral_audit_training", self.content)
        self.assertIn("tools.render_native_neutral_audit_vectors", self.content)
        self.assertIn("conformance/network_native_neutral_audit_v1.json", self.content)
        self.assertLess(
            self.content.index(
                "R2 M5g0 V19 neutral audit trust learning without target-belief change"),
            self.content.index("Runtime-derived mesh regression"),
        )
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("tools.render_native_neutral_audit_vectors", script)
        self.assertIn("network_native_neutral_audit_v1.json", script)
        self.assertIn("cmp -s", script)

    def test_r2_m5f_full_native_response_cost_grid_scope_and_ci(self):
        self.assertIn(
            "R2 M5f complete exploratory V19 speaking-cost response grid",
            self.content,
        )
        self.assertIn("tests.test_network_native_response_grid", self.content)
        self.assertIn("tools.render_native_speaking_cost_grid_vectors", self.content)
        self.assertIn("conformance/network_native_speaking_cost_grid_v1.json", self.content)
        self.assertLess(
            self.content.index("R2 M5f complete exploratory V19 speaking-cost response grid"),
            self.content.index("Runtime-derived mesh regression"),
        )
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("tools.render_native_speaking_cost_grid_vectors", script)
        self.assertIn("network_native_speaking_cost_grid_v1.json", script)
        self.assertIn("cmp -s", script)

    def test_r2_m5e2_actual_native_two_choice_not_threshold_inference(self):
        self.assertIn("R2 M5e2 source native two-choice MAP and reward integrity", self.content)
        self.assertIn("tests.test_network_native_two_choice", self.content)
        self.assertIn("tools.render_native_two_choice_vectors", self.content)
        self.assertIn("conformance/network_native_two_choice_v1.json", self.content)
        self.assertLess(
            self.content.index("R2 M5e2 source native two-choice MAP and reward integrity"),
            self.content.index("Runtime-derived mesh regression"),
        )
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("tools.render_native_two_choice_vectors", script)
        self.assertIn("network_native_two_choice_v1.json", script)
        self.assertIn("cmp -s", script)

    def test_r2_m5e1_learned_trust_then_later_native_tell(self):
        self.assertIn("R2 M5e1 source-learned trust before new native TELL", self.content)
        self.assertIn("tests.test_network_trust_history_new_tell", self.content)
        self.assertIn("tools.render_native_verified_trust_new_tell_vectors", self.content)
        self.assertIn("conformance/network_native_verified_trust_new_tell_v1.json", self.content)
        self.assertLess(
            self.content.index("R2 M5e1 source-learned trust before new native TELL"),
            self.content.index("Runtime-derived mesh regression"),
        )
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("tools.render_native_verified_trust_new_tell_vectors", script)
        self.assertIn("network_native_verified_trust_new_tell_v1.json", script)
        self.assertIn("cmp -s", script)

    def test_r2_m5e0_direct_tell_social_trust_optin(self):
        self.assertIn("R2 M5e0 direct V19 EXACT TELL source trust protocol", self.content)
        self.assertIn("tests.test_network_direct_tell_trust", self.content)
        self.assertIn("tools.render_native_direct_tell_trust_vectors", self.content)
        self.assertIn("conformance/network_native_direct_tell_trust_v1.json", self.content)
        self.assertLess(
            self.content.index("R2 M5e0 direct V19 EXACT TELL source trust protocol"),
            self.content.index("Runtime-derived mesh regression"),
        )
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("tools.render_native_direct_tell_trust_vectors", script)
        self.assertIn("network_native_direct_tell_trust_v1.json", script)
        self.assertIn("cmp -s", script)

    def test_r2_m5d_native_post_evidence_action_causal_firewall(self):
        self.assertIn("R2 M5d original V19 post-evidence actions and trust-causal firewall", self.content)
        self.assertIn("tests.test_network_post_evidence_actions", self.content)
        self.assertIn("tools.render_native_post_evidence_action_vectors", self.content)
        self.assertIn("conformance/network_post_evidence_actual_actions_v1.json", self.content)
        self.assertLess(
            self.content.index("R2 M5d original V19 post-evidence actions and trust-causal firewall"),
            self.content.index("Runtime-derived mesh regression"),
        )
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("tools.render_native_post_evidence_action_vectors", script)
        self.assertIn("network_post_evidence_actual_actions_v1.json", script)
        self.assertIn("cmp -s", script)

    def test_r2_m5c_source_owned_epistemic_trust_is_not_structural_power(self):
        self.assertIn(
            "R2 M5c source-owned TELL verification and directed-trust acceptance",
            self.content,
        )
        self.assertIn("tests.test_network_epistemic_influence", self.content)
        self.assertIn("tools.render_source_epistemic_influence_vectors", self.content)
        self.assertIn("conformance/network_source_epistemic_influence_v1.json", self.content)
        self.assertLess(
            self.content.index("R2 M5c source-owned TELL verification and directed-trust acceptance"),
            self.content.index("Runtime-derived mesh regression"),
        )
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("tools.render_source_epistemic_influence_vectors", script)
        self.assertIn("network_source_epistemic_influence_v1.json", script)
        self.assertIn("cmp -s", script)

    def test_r2_m5b_native_structural_source_is_scope_bound(self):
        self.assertIn("R2 M5b V19/V24 selected structural key-player qualification", self.content)
        self.assertIn("tests.test_network_structural_influence", self.content)
        self.assertIn("tools.render_source_structural_influence_vectors", self.content)
        self.assertIn("conformance/network_source_structural_influence_v1.json", self.content)
        self.assertLess(
            self.content.index("R2 M5b V19/V24 selected structural key-player qualification"),
            self.content.index("Runtime-derived mesh regression"),
        )
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("tools.render_source_structural_influence_vectors", script)
        self.assertIn("network_source_structural_influence_v1.json", script)

    def test_r2_m5a_authority_source_is_not_role_or_centrality(self):
        self.assertIn("R2 M5a source-owned passage decision authority", self.content)
        self.assertIn("tests.test_network_power_authority", self.content)
        self.assertIn("tools.render_passage_authority_vectors", self.content)
        self.assertIn("conformance/network_passage_authority_v1.json", self.content)
        self.assertLess(
            self.content.index("R2 M5a source-owned passage decision authority"),
            self.content.index("Runtime-derived mesh regression"),
        )
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("tools.render_passage_authority_vectors", script)
        self.assertIn("network_passage_authority_v1.json", script)
        self.assertIn("cmp -s", script)

    def test_r2_m4b_native_family_nonenquivalence_gate(self):
        self.assertIn("R2 M4b BB rewiring evolving native-family qualification", self.content)
        self.assertIn("tests.test_network_model_family_comparison", self.content)
        self.assertIn("tools.render_model_family_comparison_vectors", self.content)
        self.assertIn("conformance/network_model_family_comparison_v1.json", self.content)
        self.assertLess(
            self.content.index("R2 M4b BB rewiring evolving native-family qualification"),
            self.content.index("Runtime-derived mesh regression"),
        )
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("tools.render_model_family_comparison_vectors", script)
        self.assertIn("network_model_family_comparison_v1.json", script)
        self.assertIn("cmp -s", script)

    def test_r2_m4a_controlled_coupled_source_evidence_gate(self):
        self.assertIn("R2 M4a same-origin coupled V19 control and intervention", self.content)
        self.assertIn("tests.test_network_coupled_evolution", self.content)
        self.assertIn("tools.render_network_coupled_evolution_vectors", self.content)
        self.assertIn("conformance/network_coupled_evolution_v1.json", self.content)
        self.assertLess(
            self.content.index("R2 M4a same-origin coupled V19 control and intervention"),
            self.content.index("Runtime-derived mesh regression"),
        )
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("tools.render_network_coupled_evolution_vectors", script)
        self.assertIn("network_coupled_evolution_v1.json", script)
        self.assertIn("cmp -s", script)

    def test_r2_m3_exact_percept_content_evidence_gate(self):
        self.assertIn("R2 M3 exact V19 multi-Agent content evidence", self.content)
        self.assertIn("tests.test_agent_content_evidence", self.content)
        self.assertIn("tools.render_agent_content_evidence_vectors", self.content)
        self.assertIn("conformance/agent_content_evidence_v1.json", self.content)
        self.assertLess(
            self.content.index("R2 M3 exact V19 multi-Agent content evidence"),
            self.content.index("Runtime-derived mesh regression"),
        )
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("tools.render_agent_content_evidence_vectors", script)
        self.assertIn("agent_content_evidence_v1.json", script)
        self.assertIn("cmp -s", script)

    def test_r2_m2_real_environment_event_trace_gate(self):
        self.assertIn("R2 M2 real V19 environment-action trace", self.content)
        self.assertIn("tests.test_network_environment_actions", self.content)
        self.assertIn("tools.render_network_environment_vectors", self.content)
        self.assertIn("conformance/network_environment_action_v1.json", self.content)
        self.assertLess(
            self.content.index("R2 M2 real V19 environment-action trace"),
            self.content.index("Runtime-derived mesh regression"),
        )
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("tools.render_network_environment_vectors", script)
        self.assertIn("network_environment_action_v1.json", script)
        self.assertIn("cmp -s", script)

    def test_r2_m1_physical_passage_events_gate(self):
        self.assertIn("R2 M1 physical V19 passage-boundary events", self.content)
        self.assertIn("tests.test_network_boundary_events", self.content)
        self.assertIn("tools.render_network_boundary_events", self.content)
        self.assertIn("conformance/network_boundary_events_v1.json", self.content)
        self.assertLess(
            self.content.index("R2 M1 physical V19 passage-boundary events"),
            self.content.index("Runtime-derived mesh regression"),
        )
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("tools.render_network_boundary_events", script)
        self.assertIn("conformance/network_boundary_events_v1.json", script)
        self.assertIn("cmp -s", script)

    def test_r2_three_driver_source_observation_gate(self):
        self.assertIn("R2 three-driver V19/V24 observed-network regression", self.content)
        self.assertIn("tests.test_network_evolution_observation", self.content)
        self.assertIn("tools.render_network_evolution_vectors", self.content)
        self.assertIn("conformance/network_evolution_observation_v1.json", self.content)
        self.assertLess(
            self.content.index("R2 three-driver V19/V24 observed-network regression"),
            self.content.index("Runtime-derived mesh regression"),
        )
        script = (ROOT / "tools/check_mesh_evidence_lean.sh").read_text(encoding="utf-8")
        self.assertIn("tools.render_network_evolution_vectors", script)
        self.assertIn("network_evolution_observation_v1.json", script)
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
