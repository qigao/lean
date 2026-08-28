from __future__ import annotations

from dataclasses import replace
import unittest

from narrative_dynamics.adapters.prison_metrics import prison_initial_action_metrics
from narrative_dynamics.contracts import ExperimentStage
from narrative_dynamics.report_artifact import attest_report
from narrative_dynamics.simulation import SimulationRunner

from tests.test_narrative_identification_protocol import (
    FINAL_SEEDS,
    SELECTION_SEEDS,
    TRAIN_SEEDS,
    TRUTH,
    brier_loss,
    case_by_name,
    load_fixture,
    log_loss,
    target_spec,
)
from tests.test_narrative_identification_recovery import (
    METRIC_KEYS,
    joint_experiment,
    scale_experiment,
)
from tests.test_narrative_identification_interventions import (
    family_parameters,
    future_pair,
    memory_pair,
    sources as intervention_sources,
)
from tests.test_narrative_identification_model_comparison import execute_final_comparisons

_IDENTIFICATION_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.identification import (
        IdentificationReportError,
        IdentificationStatus,
        SyntheticIdentificationProtocol,
        build_synthetic_identification_report,
        certify_observational_equivalence,
        evaluate_information_intervention,
        evaluate_parameter_recovery_experiment,
    )
except ImportError as error:
    _IDENTIFICATION_IMPORT_ERROR = error
    IdentificationReportError = None
    IdentificationStatus = None
    SyntheticIdentificationProtocol = None
    build_synthetic_identification_report = None
    certify_observational_equivalence = None
    evaluate_information_intervention = None
    evaluate_parameter_recovery_experiment = None


HASH0 = "sha256:" + "0" * 64


def build_protocol_and_evidence():
    dataset = load_fixture()
    source_map = intervention_sources()

    joint = joint_experiment()
    pressure = scale_experiment(
        name="pressure",
        pressure_values=(0.5, 1.0, 2.0),
        instrumentality_values=(1.0,),
        target_coordinates=("goal_pressure_scale",),
        expected_status=IdentificationStatus.IDENTIFIED_UNDER_PROTOCOL,
    )
    instrumentality = scale_experiment(
        name="instrumentality",
        pressure_values=(1.0,),
        instrumentality_values=(0.5, 1.0, 2.0),
        target_coordinates=("instrumentality_scale",),
        expected_status=IdentificationStatus.IDENTIFIED_UNDER_PROTOCOL,
    )
    confounded = scale_experiment(
        name="scale-confounded",
        pressure_values=(0.5, 1.0, 2.0),
        instrumentality_values=(0.5, 1.0, 2.0),
        target_coordinates=("goal_pressure_scale", "instrumentality_scale"),
        expected_status=IdentificationStatus.NOT_IDENTIFIED_UNDER_PROTOCOL,
    )
    memory = memory_pair()
    future = future_pair()

    protocol = SyntheticIdentificationProtocol.create(
        name="narrative-synthetic-identification-v1",
        version="1",
        claim_scope="synthetic_protocol_only",
        dataset=dataset,
        adapter_sources=tuple(source_map[name] for name in sorted(source_map)),
        recovery_experiments=(joint, pressure, instrumentality, confounded),
        intervention_pairs=(memory, future),
        target_spec=target_spec(),
        extractor=prison_initial_action_metrics,
        brier_loss=brier_loss(),
        log_loss=log_loss(),
        brier_training_seeds=TRAIN_SEEDS,
        brier_selection_seeds=SELECTION_SEEDS,
        final_seeds=FINAL_SEEDS,
        equivalence_tolerance=1e-12,
    )

    runner = SimulationRunner()
    parameter_findings = tuple(
        evaluate_parameter_recovery_experiment(
            runner=runner,
            model=source_map["intentional"],
            experiment=experiment,
            extractor=prison_initial_action_metrics,
            loss=brier_loss(),
        )
        for experiment in (joint, pressure, instrumentality, confounded)
    )

    equivalence = certify_observational_equivalence(
        name="latent-observational-equivalence",
        runner=runner,
        left_model=source_map["intentional"],
        right_model=source_map["intentional"],
        left_parameters=dict(TRUTH, beta_goal=0.5, beta_action=0.5),
        right_parameters=dict(TRUTH, beta_goal=4.0, beta_action=4.0),
        cases=(case_by_name("final-equivalence").scenario,),
        seeds=FINAL_SEEDS,
        extractor=prison_initial_action_metrics,
        action_keys=METRIC_KEYS,
        tolerance=1e-12,
    )

    memory_finding = evaluate_information_intervention(
        pair=memory,
        runner=runner,
        families=(
            ("reactive", source_map["reactive"], family_parameters("reactive")),
            ("intentional", source_map["intentional"], family_parameters("intentional")),
        ),
        seeds=FINAL_SEEDS,
        extractor=prison_initial_action_metrics,
        action_keys=METRIC_KEYS,
        tolerance=1e-12,
    )
    future_finding = evaluate_information_intervention(
        pair=future,
        runner=runner,
        families=(
            ("intentional", source_map["intentional"], family_parameters("intentional")),
            ("planning", source_map["planning"], family_parameters("planning")),
        ),
        seeds=FINAL_SEEDS,
        extractor=prison_initial_action_metrics,
        action_keys=METRIC_KEYS,
        tolerance=1e-12,
    )

    (
        _dataset,
        _sources,
        _frozen,
        _brier_protocol,
        _log_protocol,
        _final_targets,
        _brier_report,
        _log_report,
        brier_finding,
        log_finding,
    ) = execute_final_comparisons()

    report = build_synthetic_identification_report(
        protocol=protocol,
        parameter_findings=parameter_findings,
        equivalence_findings=(equivalence,),
        intervention_findings=(memory_finding, future_finding),
        brier_comparison=brier_finding,
        log_comparison=log_finding,
    )
    return {
        "protocol": protocol,
        "parameter_findings": parameter_findings,
        "equivalence": equivalence,
        "interventions": (memory_finding, future_finding),
        "brier": brier_finding,
        "log": log_finding,
        "report": report,
    }


class NarrativeIdentificationReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _IDENTIFICATION_IMPORT_ERROR is not None:
            raise AssertionError(
                f"synthetic identification reporting API is missing: {_IDENTIFICATION_IMPORT_ERROR}"
            )
        cls.evidence = build_protocol_and_evidence()

    def test_final_report_requires_all_eight_roadmap_evidence_categories(self):
        report = self.evidence["report"]
        self.assertEqual(len(report.parameter_findings), 4)
        self.assertEqual(len(report.equivalence_findings), 1)
        self.assertEqual(len(report.intervention_findings), 2)
        self.assertIsNotNone(report.brier_comparison)
        self.assertIsNotNone(report.log_comparison)
        statuses = {finding.status for finding in report.parameter_findings}
        self.assertEqual(
            statuses,
            {
                IdentificationStatus.IDENTIFIED_UNDER_PROTOCOL,
                IdentificationStatus.NOT_IDENTIFIED_UNDER_PROTOCOL,
            },
        )

    def test_final_report_manifest_stage_is_synthetic_identification_and_parents_are_complete(self):
        report = self.evidence["report"]
        self.assertEqual(report.manifest.stage, ExperimentStage.SYNTHETIC_IDENTIFICATION)
        expected = set()
        for finding in report.parameter_findings:
            expected.update(finding.parent_manifest_hashes)
        for finding in report.equivalence_findings:
            expected.update(finding.parent_manifest_hashes)
        for finding in report.intervention_findings:
            expected.update(finding.parent_manifest_hashes)
        expected.add(report.brier_comparison.parent_comparison_manifest_hash)
        expected.add(report.log_comparison.parent_comparison_manifest_hash)
        self.assertEqual(set(report.manifest.parent_hashes), expected)

    def test_attested_report_round_trip_requires_integrity(self):
        report = self.evidence["report"]
        self.assertIs(attest_report(report).require_integrity(), report)

    def test_parameter_nonidentification_language_is_protocol_scoped(self):
        conclusions = tuple(self.evidence["report"].conclusions)
        matching = tuple(line for line in conclusions if "not identified" in line.lower())
        self.assertTrue(matching)
        self.assertTrue(
            all("under this frozen synthetic protocol" in line.lower() for line in matching)
        )
        self.assertTrue(all("structural" not in line.lower() for line in conclusions))
        self.assertTrue(all("empirical" not in line.lower() for line in conclusions))

    def test_model_family_language_uses_equivalent_or_separated_not_identified(self):
        conclusions = tuple(self.evidence["report"].conclusions)
        family_lines = tuple(
            line
            for line in conclusions
            if any(word in line.lower() for word in ("reactive", "intentional", "planning"))
        )
        self.assertTrue(family_lines)
        self.assertTrue(
            any(
                "equivalent" in line.lower()
                or "separated" in line.lower()
                or "discriminating" in line.lower()
                for line in family_lines
            )
        )
        self.assertTrue(
            all(
                "is identified" not in line.lower() and "not identified" not in line.lower()
                for line in family_lines
            )
        )

    def test_forged_protocol_hash_is_rejected(self):
        report = self.evidence["report"]
        with self.assertRaises(IdentificationReportError):
            replace(report, protocol_hash=HASH0)

    def test_forged_parent_manifest_set_is_rejected(self):
        report = self.evidence["report"]
        forged_manifest = replace(report.manifest, parent_hashes=(HASH0,))
        with self.assertRaises(IdentificationReportError):
            replace(report, manifest=forged_manifest)

    def test_missing_required_finding_prevents_completion(self):
        with self.assertRaises(IdentificationReportError):
            build_synthetic_identification_report(
                protocol=self.evidence["protocol"],
                parameter_findings=self.evidence["parameter_findings"][:-1],
                equivalence_findings=(self.evidence["equivalence"],),
                intervention_findings=self.evidence["interventions"],
                brier_comparison=self.evidence["brier"],
                log_comparison=self.evidence["log"],
            )


if __name__ == "__main__":
    unittest.main()
