from __future__ import annotations

import json
import unittest

from narrative_dynamics.cross_dataset_release import CARRY_FORWARD_REQUIREMENT_IDS
from narrative_dynamics.cross_dataset_reporting import (
    CarryForwardSensitivityStatus,
    CrossDatasetTransferReport,
    TransferStudyTerminal,
    classify_family_finding,
    score_and_report_transfer,
)
from tests.cross_dataset_transfer_fixtures import (
    carry_forward_statuses,
    digest,
    valid_negative_scoring_input,
)


class CrossDatasetReportingTests(unittest.TestCase):
    def test_family_finding_matrix_is_exact(self) -> None:
        self.assertEqual(
            classify_family_finding(True, True, False),
            "PARAMETER_AND_FAMILY_TRANSFER_EVIDENCE",
        )
        self.assertEqual(
            classify_family_finding(False, True, False),
            "FAMILY_TRANSFER_ONLY_RETRAINING_REQUIRED",
        )
        self.assertEqual(
            classify_family_finding(True, False, False),
            "ZERO_SHOT_TRANSFER_REFIT_NOT_ESTABLISHED",
        )
        self.assertEqual(
            classify_family_finding(False, False, False),
            "NO_TRANSFER_EVIDENCE_UNDER_PROTOCOL",
        )
        self.assertEqual(
            classify_family_finding(True, True, True),
            "MIXED_OR_INCONCLUSIVE_EVIDENCE",
        )

    def test_valid_negative_family_results_still_form_green_study(self) -> None:
        report = score_and_report_transfer(**valid_negative_scoring_input())
        self.assertIs(report.terminal, TransferStudyTerminal.GREEN)
        self.assertTrue(
            all(
                finding.finding == "NO_TRANSFER_EVIDENCE_UNDER_PROTOCOL"
                for finding in report.family_findings
            )
        )

    def test_report_is_aggregate_hash_only(self) -> None:
        report = score_and_report_transfer(**valid_negative_scoring_input())
        payload = json.dumps(report.to_payload(), sort_keys=True).lower()
        for forbidden in (
            "participant_id",
            "participant_token",
            "case_prediction",
            "final_target",
            "source_path",
            "private-p1",
            "private-p2",
        ):
            self.assertNotIn(forbidden, payload)

    def test_brier_log_disagreement_maps_to_mixed_finding(self) -> None:
        self.assertEqual(
            classify_family_finding(True, False, True),
            "MIXED_OR_INCONCLUSIVE_EVIDENCE",
        )

    def test_participant_equal_is_primary_and_trial_equal_is_diagnostic_only(self) -> None:
        report = score_and_report_transfer(**valid_negative_scoring_input())
        self.assertEqual(report.primary_aggregation, "PARTICIPANT_EQUAL")
        self.assertEqual(report.diagnostic_aggregation, "TRIAL_EQUAL")
        primary = tuple(
            evidence
            for evidence in report.inference_evidence
            if evidence.aggregation == "PARTICIPANT_EQUAL"
        )
        diagnostic = tuple(
            evidence
            for evidence in report.inference_evidence
            if evidence.aggregation == "TRIAL_EQUAL"
        )
        self.assertTrue(primary)
        self.assertTrue(diagnostic)
        self.assertEqual(len(primary), len(diagnostic))

    def test_report_contains_task_condition_and_participant_influence_diagnostics(self) -> None:
        report = score_and_report_transfer(**valid_negative_scoring_input())
        self.assertEqual(report.task_condition_strata, ("s0", "synthetic-condition"))
        self.assertEqual(
            report.participant_influence_hash,
            digest("participant-influence"),
        )
        self.assertTrue(all(hasattr(row, "refit_material_gain") for row in report.family_findings))

    def test_all_carry_forward_ids_have_comparable_or_not_established_status(self) -> None:
        values = valid_negative_scoring_input()
        statuses = tuple(
            (requirement_id, CarryForwardSensitivityStatus.NOT_ESTABLISHED)
            if index % 2
            else (requirement_id, CarryForwardSensitivityStatus.COMPARABLE)
            for index, requirement_id in enumerate(CARRY_FORWARD_REQUIREMENT_IDS)
        )
        report = score_and_report_transfer(**{**values, "carry_forward_statuses": statuses})
        self.assertEqual(
            tuple(row.requirement_id for row in report.carry_forward_sensitivities),
            CARRY_FORWARD_REQUIREMENT_IDS,
        )
        self.assertEqual(
            {row.status for row in report.carry_forward_sensitivities},
            {
                CarryForwardSensitivityStatus.COMPARABLE,
                CarryForwardSensitivityStatus.NOT_ESTABLISHED,
            },
        )

    def test_semantic_gate_failure_is_scientific_red(self) -> None:
        values = valid_negative_scoring_input()
        report = score_and_report_transfer(
            **{
                **values,
                "semantic_invariance_pass": False,
                "scientific_failure_hash": digest("semantic-failure"),
            }
        )
        self.assertIs(report.terminal, TransferStudyTerminal.SCIENTIFIC_RED)
        self.assertEqual(report.family_findings, ())

    def test_infrastructure_incomplete_is_exclusive(self) -> None:
        values = valid_negative_scoring_input()
        values["artifact"] = None
        report = score_and_report_transfer(
            **values,
            infrastructure_failure_hash=digest("infrastructure-failure"),
        )
        self.assertIs(report.terminal, TransferStudyTerminal.INFRASTRUCTURE_INCOMPLETE)
        with self.assertRaisesRegex(ValueError, "exclusive"):
            score_and_report_transfer(
                **values,
                infrastructure_failure_hash=digest("infrastructure-failure"),
                scientific_failure_hash=digest("scientific-failure"),
            )

    def test_aggregate_intervals_are_complete_and_content_addressed(self) -> None:
        report = score_and_report_transfer(**valid_negative_scoring_input())
        self.assertGreaterEqual(len(report.inference_evidence), 24)
        for evidence in report.inference_evidence:
            self.assertLessEqual(evidence.lower_95, evidence.upper_95)
            self.assertTrue(evidence.content_hash.startswith("sha256:"))

    def test_report_payload_is_strict_and_archive_scan_rejects_forbidden_value(self) -> None:
        report = score_and_report_transfer(**valid_negative_scoring_input())
        payload = report.to_payload()
        self.assertEqual(CrossDatasetTransferReport.from_payload(payload), report)
        with self.assertRaises(ValueError):
            CrossDatasetTransferReport.from_payload({**payload, "unknown": True})

        values = valid_negative_scoring_input()
        with self.assertRaisesRegex(ValueError, "forbidden"):
            score_and_report_transfer(
                **{
                    **values,
                    "task_condition_strata": ("s0", "private-secret"),
                    "forbidden_archive_values": ("private-secret",),
                }
            )


if __name__ == "__main__":
    unittest.main()
