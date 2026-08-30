from __future__ import annotations

from dataclasses import replace
import json
import unittest

from narrative_dynamics import (
    MeasurementAuditAttempt,
    MeasurementValidityReport,
    measurement_report_payload,
)
from narrative_dynamics.adapters.two_stage_metrics import (
    two_stage_brier_loss,
    two_stage_log_loss,
)
from narrative_dynamics.candidate_execution import SequentialCandidateExecutor
from narrative_dynamics.contracts import ExperimentStage, stable_content_hash
from narrative_dynamics.measurement_validity import (
    ExactInvarianceFinding,
    MEASUREMENT_CLAIM_SCOPE,
    MeasurementTerminalClass,
    MeasurementValidityStatus,
    build_measurement_robustness_profile,
    score_measurement_predictions,
)
from narrative_dynamics.report_artifact import (
    AggregateReportArtifact,
    attest_report,
)
from narrative_dynamics.studies.feher_hare_measurement_validity_v1 import (
    assemble_feher_hare_measurement_validity_report,
    build_feher_hare_stay_switch_diagnostics,
    execute_feher_hare_measurement_predictions,
)

from tests.test_feher_hare_measurement_prediction import _fixture


_EXACT_CHECKS = (
    "task_canonicalization",
    "semantic_executor_invariance",
    "categorical_coordinate_invariance",
    "record_order_batch_invariance",
    "feher_hare_exact_invariance_gate",
)


def _exact_finding(
    name: str,
    *,
    failed: bool = False,
) -> ExactInvarianceFinding:
    original_hash = stable_content_hash(("reporting-exact-fixture", name, "expected"))
    transformed_hash = (
        stable_content_hash(("reporting-exact-fixture", name, "failed"))
        if failed
        else original_hash
    )
    return ExactInvarianceFinding(
        check_name=name,
        status=(
            MeasurementValidityStatus.EXACT_INVARIANCE_FAILED
            if failed
            else MeasurementValidityStatus.EXACT_INVARIANCE_MET
        ),
        original_hash=original_hash,
        transformed_hash=transformed_hash,
        details_hash=stable_content_hash(("reporting-exact-details", name)),
    )


def _complete_findings(
    *,
    failed: bool = False,
) -> tuple[ExactInvarianceFinding, ...]:
    return tuple(
        _exact_finding(
            name,
            failed=(
                failed
                and name
                in {
                    "task_canonicalization",
                    "feher_hare_exact_invariance_gate",
                }
            ),
        )
        for name in _EXACT_CHECKS
    )


class MeasurementValidityReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.identity, cls.audit_input, cls.protocol = _fixture()
        cls.prediction_artifact = execute_feher_hare_measurement_predictions(
            cls.identity,
            cls.audit_input,
            cls.protocol,
            SequentialCandidateExecutor(),
        )
        cls.case_losses = score_measurement_predictions(
            cls.audit_input,
            cls.prediction_artifact,
            (two_stage_brier_loss(), two_stage_log_loss()),
        )
        cls.profile = build_measurement_robustness_profile(
            cls.case_losses,
            cls.protocol,
        )
        cls.diagnostics = build_feher_hare_stay_switch_diagnostics(
            cls.audit_input,
            cls.prediction_artifact,
        )
        cls.findings = _complete_findings()

    def _green_report(self) -> MeasurementValidityReport:
        return assemble_feher_hare_measurement_validity_report(
            audit_input=self.audit_input,
            protocol=self.protocol,
            prediction_artifact=self.prediction_artifact,
            exact_invariance_findings=self.findings,
            robustness_profile=self.profile,
            stay_switch_diagnostics=self.diagnostics,
        )

    def test_green_report_binds_aggregate_evidence_and_attests(self) -> None:
        report = self._green_report()
        self.assertIsInstance(report, MeasurementValidityReport)
        self.assertIs(report.terminal_class, MeasurementTerminalClass.GREEN)
        self.assertEqual(report.claim_scope, MEASUREMENT_CLAIM_SCOPE)
        self.assertEqual(report.protocol_hash, self.protocol.content_hash)
        self.assertEqual(report.audit_input_hash, self.audit_input.content_hash)
        self.assertEqual(
            report.empirical_anchor_hash,
            self.audit_input.empirical_anchor_hash,
        )
        self.assertEqual(
            report.allowed_partition_hashes,
            tuple(
                (role.value, value)
                for role, value in self.audit_input.allowed_partition_hashes
            ),
        )
        self.assertEqual(
            report.allowed_target_report_hashes,
            tuple(
                (role.value, value)
                for role, value in self.audit_input.allowed_target_report_hashes
            ),
        )
        self.assertEqual(
            report.candidate_hashes,
            tuple(
                candidate.content_hash
                for candidate in self.audit_input.frozen_candidates
            ),
        )
        self.assertEqual(
            report.prediction_artifact_hash,
            self.prediction_artifact.content_hash,
        )
        self.assertEqual(
            report.execution_manifest_hashes,
            tuple(sorted(self.prediction_artifact.execution_manifest_hashes)),
        )
        self.assertEqual(report.exact_invariance_findings, self.findings)
        self.assertEqual(report.robustness_profile, self.profile)
        self.assertEqual(report.stay_switch_diagnostics, self.diagnostics)
        self.assertEqual(
            sum(row[2] for row in report.record_counts),
            len(self.audit_input.cases),
        )
        self.assertEqual(len(report.record_counts), 4)
        self.assertEqual(len(report.participant_counts), 4)
        self.assertEqual(len(report.diagnostic_cell_counts), 24)
        self.assertEqual(
            sum(row[4] for row in report.diagnostic_cell_counts),
            len(self.audit_input.cases) * 3,
        )
        self.assertFalse(report.parameter_training_performed)
        self.assertFalse(report.parameter_selection_performed)
        self.assertFalse(report.final_test_values_exposed_to_audit)
        self.assertFalse(report.final_test_outcomes_analyzed)
        self.assertFalse(report.final_model_execution)
        self.assertIs(report.manifest.stage, ExperimentStage.MEASUREMENT_AUDIT)

        artifact = AggregateReportArtifact.from_report(report)
        attested = attest_report(report)
        self.assertEqual(attested.artifact, artifact)
        self.assertIs(attested.require_integrity(), report)
        self.assertEqual(report.content_hash, stable_content_hash(report.identity_payload()))

    def test_serialized_report_contains_no_raw_rows_or_direct_identities(self) -> None:
        report = self._green_report()
        payload = measurement_report_payload(report)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        for forbidden in (
            "source_participant_id",
            "participant_id",
            "participant_group_hash",
            "source_path",
            "first_stage_action",
            "target_map",
            "prediction_rows",
            "final_test_cases",
        ):
            self.assertNotIn(forbidden, encoded)
        for case in self.audit_input.cases:
            self.assertNotIn(case.scenario.id, encoded)
            self.assertNotIn(case.record_hash, encoded)
        self.assertFalse(hasattr(report, "conclusion"))
        self.assertFalse(hasattr(report, "conclusions"))
        self.assertFalse(hasattr(report, "scientific_conclusion"))

    def test_pre_execution_exact_failure_is_attested_scientific_red(self) -> None:
        findings = (
            _exact_finding("task_canonicalization", failed=True),
            _exact_finding("semantic_executor_invariance"),
        )
        report = assemble_feher_hare_measurement_validity_report(
            audit_input=self.audit_input,
            protocol=self.protocol,
            prediction_artifact=None,
            exact_invariance_findings=findings,
            robustness_profile=None,
            stay_switch_diagnostics=(),
        )
        self.assertIs(
            report.terminal_class,
            MeasurementTerminalClass.SCIENTIFIC_RED,
        )
        self.assertIsNone(report.prediction_artifact_hash)
        self.assertEqual(report.execution_manifest_hashes, ())
        self.assertIsNone(report.robustness_profile)
        self.assertEqual(report.stay_switch_diagnostics, ())
        self.assertIs(attest_report(report).require_integrity(), report)

    def test_post_prediction_exact_failure_binds_execution_but_omits_robustness(self) -> None:
        report = assemble_feher_hare_measurement_validity_report(
            audit_input=self.audit_input,
            protocol=self.protocol,
            prediction_artifact=self.prediction_artifact,
            exact_invariance_findings=_complete_findings(failed=True),
            robustness_profile=None,
            stay_switch_diagnostics=(),
        )
        self.assertIs(
            report.terminal_class,
            MeasurementTerminalClass.SCIENTIFIC_RED,
        )
        self.assertEqual(
            report.prediction_artifact_hash,
            self.prediction_artifact.content_hash,
        )
        self.assertEqual(
            report.execution_manifest_hashes,
            self.prediction_artifact.execution_manifest_hashes,
        )
        self.assertIsNone(report.robustness_profile)
        with self.assertRaisesRegex(ValueError, "robustness"):
            assemble_feher_hare_measurement_validity_report(
                audit_input=self.audit_input,
                protocol=self.protocol,
                prediction_artifact=self.prediction_artifact,
                exact_invariance_findings=_complete_findings(failed=True),
                robustness_profile=self.profile,
                stay_switch_diagnostics=self.diagnostics,
            )

    def test_material_dependence_is_a_green_finding_not_a_hard_failure(self) -> None:
        dependence_rows = (
            self.profile.task_findings
            + self.profile.aggregation_findings
            + self.profile.score_findings
            + self.profile.participant_influence
        )
        self.assertTrue(
            any(
                row.status
                is MeasurementValidityStatus.MATERIALLY_MEASUREMENT_DEPENDENT
                for row in dependence_rows
            )
        )
        report = self._green_report()
        self.assertIs(report.terminal_class, MeasurementTerminalClass.GREEN)
        self.assertEqual(report.robustness_profile, self.profile)

    def test_infrastructure_exception_is_an_attempt_not_a_scientific_report(self) -> None:
        attempt = MeasurementAuditAttempt(
            attempt_id="measurement-attempt-1",
            scientific_revision="a" * 40,
            orchestration_revision="b" * 40,
            started_at_utc="2026-08-30T05:00:00Z",
            terminal_class=MeasurementTerminalClass.INFRASTRUCTURE_INCOMPLETE,
            protocol_hash=self.protocol.content_hash,
            report_hash=None,
            error_type="CandidateExecutionError",
            error_message_hash=stable_content_hash("injected executor failure"),
            artifact_file_hashes=(),
        )
        self.assertIs(
            attempt.terminal_class,
            MeasurementTerminalClass.INFRASTRUCTURE_INCOMPLETE,
        )
        self.assertIsNone(attempt.report_hash)
        self.assertIsNotNone(attempt.error_message_hash)
        self.assertTrue(attempt.content_hash.startswith("sha256:"))
        with self.assertRaisesRegex(ValueError, "infrastructure"):
            replace(
                self._green_report(),
                terminal_class=MeasurementTerminalClass.INFRASTRUCTURE_INCOMPLETE,
                manifest=None,
            )

    def test_report_rejects_boundary_flags_raw_objects_and_inconsistent_red(self) -> None:
        report = self._green_report()
        for field_name in (
            "parameter_training_performed",
            "parameter_selection_performed",
            "final_test_values_exposed_to_audit",
            "final_test_outcomes_analyzed",
            "final_model_execution",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, field_name):
                    replace(report, **{field_name: True, "manifest": None})
        with self.assertRaises((TypeError, ValueError)):
            replace(
                report,
                exact_invariance_findings=(self.prediction_artifact,),
                manifest=None,
            )
        with self.assertRaises((TypeError, ValueError)):
            replace(
                report,
                stay_switch_diagnostics=(self.audit_input.cases[0],),
                manifest=None,
            )
        with self.assertRaises((TypeError, ValueError)):
            replace(
                report,
                robustness_profile=self.prediction_artifact,
                manifest=None,
            )
        with self.assertRaisesRegex(ValueError, "prediction"):
            replace(
                report,
                terminal_class=MeasurementTerminalClass.SCIENTIFIC_RED,
                prediction_artifact_hash=None,
                exact_invariance_findings=_complete_findings(failed=True),
                robustness_profile=None,
                stay_switch_diagnostics=(),
                manifest=None,
            )

    def test_attestation_detects_any_report_field_mutation(self) -> None:
        report = self._green_report()
        attested = attest_report(report)
        original_hash = attested.content_hash
        object.__setattr__(report, "final_model_execution", True)
        with self.assertRaises(RuntimeError):
            attested.require_integrity()
        self.assertEqual(attested.content_hash, original_hash)


if __name__ == "__main__":
    unittest.main()
