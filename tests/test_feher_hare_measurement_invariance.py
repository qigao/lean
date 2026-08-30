from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from narrative_dynamics.candidate_execution import SequentialCandidateExecutor
from narrative_dynamics.contracts import Scenario
from narrative_dynamics.measurement_validity import (
    ExactInvarianceFinding,
    MeasurementValidityStatus,
)
import narrative_dynamics.studies.feher_hare_measurement_validity_v1 as measurement_study
from narrative_dynamics.studies.feher_hare_measurement_validity_v1 import (
    FeherHareSemanticFixture,
    combine_feher_hare_exact_invariance,
    evaluate_feher_hare_empirical_invariance,
    evaluate_feher_hare_semantic_invariance,
    execute_feher_hare_measurement_predictions,
    frozen_feher_hare_semantic_fixtures,
)

from tests.test_feher_hare_measurement_prediction import _fixture


class FeherHareMeasurementInvarianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.identity, cls.audit_input, cls.protocol = _fixture()
        cls.artifact = execute_feher_hare_measurement_predictions(
            cls.identity,
            cls.audit_input,
            cls.protocol,
            SequentialCandidateExecutor(),
        )

    def test_frozen_fixtures_cover_every_semantic_transformation(self) -> None:
        fixtures = frozen_feher_hare_semantic_fixtures()
        self.assertTrue(
            all(isinstance(row, FeherHareSemanticFixture) for row in fixtures)
        )
        self.assertEqual(
            tuple(row.name for row in fixtures),
            (
                "coherent_action_state_relabel",
                "second_stage_action_relabel",
                "magic_carpet_counterbalance",
                "spaceship_symbol_order",
                "post_choice_outcome_exclusion",
            ),
        )
        self.assertEqual(
            len({row.content_hash for row in fixtures}),
            len(fixtures),
        )

        coherent = fixtures[0]
        original_history = coherent.original_scenario.payload["history"]
        transformed_history = coherent.transformed_scenario.payload["history"]
        action_swap = {"action_0": "action_1", "action_1": "action_0"}
        state_swap = {"state_0": "state_1", "state_1": "state_0"}
        for original, transformed in zip(
            original_history,
            transformed_history,
            strict=True,
        ):
            self.assertEqual(
                transformed["first_stage_action"],
                action_swap[original["first_stage_action"]],
            )
            self.assertEqual(
                transformed["second_stage_action"],
                action_swap[original["second_stage_action"]],
            )
            self.assertEqual(
                transformed["final_state"],
                state_swap[original["final_state"]],
            )
        self.assertEqual(
            dict(coherent.inverse_metric_permutation),
            {
                "first_stage.action_0": "first_stage.action_1",
                "first_stage.action_1": "first_stage.action_0",
            },
        )

        magic = fixtures[2]
        self.assertNotEqual(
            magic.original_scenario.payload["first_stage_configuration"],
            magic.transformed_scenario.payload["first_stage_configuration"],
        )
        spaceship = fixtures[3]
        self.assertNotEqual(
            spaceship.original_scenario.payload["first_stage_configuration"],
            spaceship.transformed_scenario.payload["first_stage_configuration"],
        )
        post_choice = fixtures[4]
        self.assertEqual(
            post_choice.original_scenario.payload,
            post_choice.transformed_scenario.payload,
        )
        self.assertNotEqual(
            post_choice.original_scenario.id,
            post_choice.transformed_scenario.id,
        )

    def test_all_families_meet_semantic_and_spawn_invariance(self) -> None:
        findings = evaluate_feher_hare_semantic_invariance(
            self.identity,
            self.audit_input.frozen_candidates,
        )
        self.assertEqual(
            tuple(row.check_name for row in findings),
            ("task_canonicalization", "semantic_executor_invariance"),
        )
        self.assertTrue(all(isinstance(row, ExactInvarianceFinding) for row in findings))
        self.assertTrue(
            all(
                row.status is MeasurementValidityStatus.EXACT_INVARIANCE_MET
                for row in findings
            )
        )
        self.assertTrue(
            all(row.original_hash == row.transformed_hash for row in findings)
        )

    def test_empirical_coordinate_order_and_batch_invariance_are_exact(self) -> None:
        findings = evaluate_feher_hare_empirical_invariance(
            self.audit_input,
            self.artifact,
            self.protocol,
        )
        self.assertEqual(
            tuple(row.check_name for row in findings),
            (
                "categorical_coordinate_invariance",
                "record_order_batch_invariance",
            ),
        )
        self.assertTrue(
            all(
                row.status is MeasurementValidityStatus.EXACT_INVARIANCE_MET
                for row in findings
            )
        )
        self.assertTrue(
            all(row.original_hash == row.transformed_hash for row in findings)
        )

        semantic = evaluate_feher_hare_semantic_invariance(
            self.identity,
            self.audit_input.frozen_candidates,
        )
        gate = combine_feher_hare_exact_invariance(semantic, findings)
        self.assertIs(
            gate.status,
            MeasurementValidityStatus.EXACT_INVARIANCE_MET,
        )
        self.assertEqual(gate.original_hash, gate.transformed_hash)

    def test_protocol_or_artifact_binding_drift_fails_before_evaluation(self) -> None:
        with self.assertRaisesRegex(ValueError, "protocol"):
            evaluate_feher_hare_empirical_invariance(
                self.audit_input,
                replace(
                    self.artifact,
                    protocol_hash="sha256:" + "0" * 64,
                    manifest=None,
                ),
                self.protocol,
            )
        with self.assertRaisesRegex(ValueError, "audit input"):
            evaluate_feher_hare_empirical_invariance(
                replace(
                    self.audit_input,
                    dataset_hash="sha256:" + "9" * 64,
                ),
                self.artifact,
                self.protocol,
            )

    def test_broken_history_relabel_fails_hard_gate_without_robustness(self) -> None:
        fixtures = list(frozen_feher_hare_semantic_fixtures())
        coherent = fixtures[0]
        payload = dict(coherent.transformed_scenario.payload)
        history = [dict(row) for row in payload["history"]]
        history[-1]["first_stage_action"] = (
            "action_1"
            if history[-1]["first_stage_action"] == "action_0"
            else "action_0"
        )
        payload["history"] = tuple(history)
        fixtures[0] = replace(
            coherent,
            transformed_scenario=Scenario(
                id=coherent.transformed_scenario.id,
                payload=payload,
            ),
        )
        with patch.object(
            measurement_study,
            "frozen_feher_hare_semantic_fixtures",
            return_value=tuple(fixtures),
        ):
            semantic = evaluate_feher_hare_semantic_invariance(
                self.identity,
                self.audit_input.frozen_candidates,
            )
        task_finding = next(
            row for row in semantic if row.check_name == "task_canonicalization"
        )
        self.assertIs(
            task_finding.status,
            MeasurementValidityStatus.EXACT_INVARIANCE_FAILED,
        )
        empirical = evaluate_feher_hare_empirical_invariance(
            self.audit_input,
            self.artifact,
            self.protocol,
        )
        with patch.object(
            measurement_study,
            "build_measurement_robustness_profile",
            side_effect=AssertionError("robustness must not run after a hard-gate failure"),
        ) as robustness:
            gate = combine_feher_hare_exact_invariance(semantic, empirical)
        robustness.assert_not_called()
        self.assertIs(
            gate.status,
            MeasurementValidityStatus.EXACT_INVARIANCE_FAILED,
        )
        self.assertNotEqual(gate.original_hash, gate.transformed_hash)


if __name__ == "__main__":
    unittest.main()
