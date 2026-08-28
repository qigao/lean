from __future__ import annotations

from dataclasses import replace
import unittest

from narrative_dynamics.contracts import ExperimentStage
from narrative_dynamics.report_artifact import attest_report
from narrative_dynamics.simulation import SimulationRunner

from tests.external_validation_fixtures import (
    METHOD_HASH,
    ProbabilityModel,
    brier_loss,
    build_constraint_plan,
    build_external_declaration,
    build_preregistration,
    final_targets,
    log_loss,
    make_release,
    policy_metrics,
    runtime_models,
    selection_targets,
    sibling_protocols,
    verify_release,
)

try:
    from narrative_dynamics.external_validation import (
        EXTERNAL_CLAIM_SCOPE,
        ExternalValidationError,
        build_external_validation_report,
        evaluate_external_constraint,
        evaluate_external_final,
    )
except ImportError as error:
    EXTERNAL_CLAIM_SCOPE = None
    ExternalValidationError = None
    build_external_validation_report = None
    evaluate_external_constraint = None
    evaluate_external_final = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None

HASH0 = "sha256:" + "0" * 64


class ExternalValidationReportingTests(unittest.TestCase):
    def _evidence(self, *, with_constraint=True):
        if evaluate_external_final is None or build_external_validation_report is None:
            self.fail(f"external reporting API is missing: {IMPORT_ERROR}")
        data, brier, log = sibling_protocols()
        declaration = build_external_declaration(data)
        plans = (build_constraint_plan(),) if with_constraint else ()
        prereg = build_preregistration(
            dataset=data,
            brier_protocol=brier,
            log_protocol=log,
            constraint_plans=plans,
            method_validation_hashes=(METHOD_HASH,),
        )
        brier_release = make_release(
            brier,
            evidence_hash=declaration.content_hash,
            preregistration_hash=prereg.content_hash,
            score_role="brier",
        )
        log_release = make_release(
            log,
            evidence_hash=declaration.content_hash,
            preregistration_hash=prereg.content_hash,
            score_role="log",
        )
        brier_models, _ = runtime_models(brier.candidates)
        log_models, _ = runtime_models(log.candidates)
        final_evaluation = evaluate_external_final(
            runner=SimulationRunner(),
            preregistration=prereg,
            evidence=declaration,
            brier_protocol=brier,
            brier_release=brier_release,
            brier_verified=verify_release(brier_release, brier),
            brier_models=brier_models,
            brier_loss=brier_loss(),
            log_protocol=log,
            log_release=log_release,
            log_verified=verify_release(log_release, log),
            log_models=log_models,
            log_loss=log_loss(),
            final_targets=final_targets(data),
            extractor=policy_metrics,
        )
        constraint_findings = ()
        if with_constraint:
            constraint_findings = (
                evaluate_external_constraint(
                    plan=plans[0],
                    runner=SimulationRunner(),
                    model=ProbabilityModel(),
                    selection_targets=selection_targets(data),
                    extractor=policy_metrics,
                    brier_loss=brier_loss(),
                    log_loss=log_loss(),
                ),
            )
        report = build_external_validation_report(
            preregistration=prereg,
            evidence=declaration,
            final_evaluation=final_evaluation,
            constraint_findings=constraint_findings,
        )
        return {
            "declaration": declaration,
            "prereg": prereg,
            "final_evaluation": final_evaluation,
            "constraint_findings": constraint_findings,
            "report": report,
        }

    def test_manifest_stage_is_external_validation(self):
        report = self._evidence()["report"]
        self.assertEqual(report.manifest.stage, ExperimentStage.EXTERNAL_VALIDATION)

    def test_report_binds_evidence_preregistration_both_release_verifications_and_child_manifests(self):
        evidence = self._evidence()
        report = evidence["report"]
        evaluation = evidence["final_evaluation"]
        self.assertEqual(
            report.evidence_declaration_hash,
            evidence["declaration"].content_hash,
        )
        self.assertEqual(report.preregistration_hash, evidence["prereg"].content_hash)
        self.assertEqual(
            report.brier_release_hash,
            evaluation.preflight.brier_release_hash,
        )
        self.assertEqual(
            report.log_release_hash,
            evaluation.preflight.log_release_hash,
        )
        self.assertEqual(
            report.brier_verification_hash,
            evaluation.preflight.brier_verification_hash,
        )
        self.assertEqual(
            report.log_verification_hash,
            evaluation.preflight.log_verification_hash,
        )
        self.assertEqual(
            report.brier_comparison_manifest_hash,
            evaluation.brier_report.manifest.content_hash,
        )
        self.assertEqual(
            report.log_comparison_manifest_hash,
            evaluation.log_report.manifest.content_hash,
        )

    def test_report_preserves_method_validation_hashes_as_opaque_lineage_only(self):
        report = self._evidence()["report"]
        self.assertEqual(report.method_validation_hashes, (METHOD_HASH,))
        self.assertFalse(hasattr(report, "method_validation_statuses"))

    def test_report_has_no_identification_status_or_free_text_conclusions_field(self):
        report = self._evidence()["report"]
        self.assertFalse(hasattr(report, "identification_status"))
        self.assertFalse(hasattr(report, "conclusions"))
        self.assertFalse(hasattr(report, "conclusion"))

    def test_report_claim_scope_is_exact(self):
        report = self._evidence()["report"]
        self.assertEqual(report.claim_scope, EXTERNAL_CLAIM_SCOPE)

    def test_report_requires_declared_constraint_findings_exactly(self):
        evidence = self._evidence()
        with self.assertRaises(ExternalValidationError):
            build_external_validation_report(
                preregistration=evidence["prereg"],
                evidence=evidence["declaration"],
                final_evaluation=evidence["final_evaluation"],
                constraint_findings=(),
            )

    def test_report_attests_and_tampering_changes_artifact_identity(self):
        report = self._evidence()["report"]
        attested = attest_report(report)
        self.assertIs(attested.require_integrity(), report)
        original_hash = attested.content_hash
        object.__setattr__(
            report,
            "claim_scope",
            "external_observational_predictive_only:tampered",
        )
        with self.assertRaises(RuntimeError):
            attested.require_integrity()
        self.assertEqual(attested.content_hash, original_hash)

    def test_forged_or_incomplete_parent_lineage_is_rejected(self):
        report = self._evidence()["report"]
        forged_manifest = replace(report.manifest, parent_hashes=(HASH0,))
        with self.assertRaises(ExternalValidationError):
            replace(report, manifest=forged_manifest)


if __name__ == "__main__":
    unittest.main()
