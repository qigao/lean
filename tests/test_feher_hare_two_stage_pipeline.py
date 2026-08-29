from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from narrative_dynamics.model_comparison import ComparisonModel
from narrative_dynamics.observations.release import ProtocolRelease, WitnessReceipt, verify_protocol_release
from narrative_dynamics.report_artifact import attest_report
from narrative_dynamics.simulation import SimulationRunner
from tests.test_two_stage_source import _files
from tests.two_stage_test_support import build_synthetic_two_stage_checkout

_PIPELINE_IMPORT_ERROR: Exception | None = None
try:
    from narrative_dynamics.adapters.narrative_two_stage import (
        create_narrative_two_stage_intentional_source,
        create_narrative_two_stage_planning_source,
        create_narrative_two_stage_reactive_source,
    )
    from narrative_dynamics.studies.feher_hare_two_stage_v1 import (
        TwoStageFinalAttemptLedger,
        build_feher_hare_protocol_bundle,
        fit_and_freeze_feher_hare_models,
        prepare_feher_hare_two_stage_v1,
        run_locked_feher_hare_final,
    )
    from narrative_dynamics.studies.two_stage_osf import (
        OSFRegistrationProof,
        OSFRegistrationVerifier,
        TwoStageOSFBundle,
    )
    from narrative_dynamics.studies.two_stage_source import TwoStageSourceManifest
except Exception as error:
    _PIPELINE_IMPORT_ERROR = error


class FeherHarePipelineTests(unittest.TestCase):
    def require_pipeline(self):
        self.assertIsNone(
            _PIPELINE_IMPORT_ERROR,
            f"full two-stage study pipeline is missing: {_PIPELINE_IMPORT_ERROR}",
        )

    def test_complete_synthetic_external_study_uses_one_final_prediction_pass(self):
        self.require_pipeline()
        with tempfile.TemporaryDirectory() as tmp:
            root, revision = build_synthetic_two_stage_checkout(Path(tmp), magic_n=6, spaceship_n=6)
            manifest = TwoStageSourceManifest(
                name="synthetic-feher-hare",
                version="1",
                repository="test/synthetic-two-stage",
                revision=revision,
                license_reference="test-only",
                files=_files(root),
            )
            runner = SimulationRunner()
            prepared = prepare_feher_hare_two_stage_v1(root=root, manifest=manifest)
            frozen = fit_and_freeze_feher_hare_models(runner=runner, prepared=prepared)
            protocols = build_feher_hare_protocol_bundle(prepared=prepared, frozen_models=frozen)

            common_revision = "synthetic-study-head"
            def release(protocol, role):
                return ProtocolRelease.create(
                    name=f"synthetic:{role}:release",
                    version="1",
                    protocol=protocol,
                    source_revision={
                        "repository_revision": common_revision,
                        "external_evidence_declaration_hash": prepared.evidence.content_hash,
                        "external_validation_preregistration_hash": protocols.preregistration.content_hash,
                        "score_role": role,
                    },
                )
            brier_release = release(protocols.brier_protocol, "brier")
            log_release = release(protocols.log_protocol, "log")

            osf_bundle = TwoStageOSFBundle(
                source_manifest_hash=manifest.content_hash,
                source_snapshot_hash=prepared.transform_report.source_snapshot_hash,
                transform_hash=prepared.transform_report.content_hash,
                participant_assignment_hash=prepared.assignment.content_hash,
                dataset_hash=prepared.dataset.content_hash,
                final_target_hash=prepared.final_targets.content_hash,
                frozen_candidate_hashes=tuple(candidate.content_hash for candidate in frozen.frozen_candidates),
                brier_protocol_hash=protocols.brier_protocol.content_hash,
                log_protocol_hash=protocols.log_protocol.content_hash,
                external_preregistration_hash=protocols.preregistration.content_hash,
                brier_release_hash=brier_release.content_hash,
                log_release_hash=log_release.content_hash,
                repository_revision=common_revision,
                claim_scope="external_observational_predictive_only",
                scientific_contract={"test_only": True, "constraint_plans": ()},
            )
            proof = OSFRegistrationProof(
                registration_reference="https://osf.io/registrations/synthetic-test",
                registered_at="2026-08-29T00:00:00Z",
                bundle_hash=osf_bundle.content_hash,
            )
            verifier = OSFRegistrationVerifier(proof)
            def verified(release, protocol):
                receipt = WitnessReceipt.create(
                    provider="osf-registration",
                    authority="public-registration",
                    subject_hash=release.content_hash,
                    reference=proof.registration_reference,
                    claimed_at=proof.registered_at,
                    proof={"bundle_hash": proof.bundle_hash},
                )
                return verify_protocol_release(
                    release,
                    protocol=protocol,
                    receipts=(receipt,),
                    verifier=verifier,
                )
            brier_verified = verified(brier_release, protocols.brier_protocol)
            log_verified = verified(log_release, protocols.log_protocol)

            source_by_name = {
                "reactive": create_narrative_two_stage_reactive_source(),
                "intentional": create_narrative_two_stage_intentional_source(),
                "planning": create_narrative_two_stage_planning_source(),
            }
            runtime_models = tuple(
                ComparisonModel(
                    frozen=candidate,
                    model=source_by_name[candidate.name],
                )
                for candidate in frozen.frozen_candidates
            )
            result = run_locked_feher_hare_final(
                runner=runner,
                prepared=prepared,
                protocols=protocols,
                brier_release=brier_release,
                brier_verified=brier_verified,
                log_release=log_release,
                log_verified=log_verified,
                runtime_models=runtime_models,
                repository_revision=common_revision,
                attempt_ledger=TwoStageFinalAttemptLedger(()),
                attempt_id="synthetic-final-1",
                started_at="2026-08-29T00:00:01Z",
            )
            self.assertEqual(result.report.claim_scope, "external_observational_predictive_only")
            self.assertEqual(result.report.constraint_findings, ())
            attest_report(result.report).require_integrity()
            self.assertEqual(result.attempt_ledger.attempts[-1].status.value, "completed")
            self.assertIn(result.brier_report.comparison.ranking[0].name, source_by_name)
            self.assertIn(result.log_report.comparison.ranking[0].name, source_by_name)
            self.assertEqual(
                result.external_evaluation.preflight.content_hash,
                result.prediction_artifact.preflight_hash,
            )


if __name__ == "__main__":
    unittest.main()
