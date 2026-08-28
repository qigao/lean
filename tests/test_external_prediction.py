from __future__ import annotations

from dataclasses import replace
import unittest

from narrative_dynamics.external_validation import preflight_external_releases
from narrative_dynamics.observations.release import compare_released_models
from narrative_dynamics.simulation import SimulationRunner
from tests.external_validation_fixtures import (
    brier_loss,
    build_external_declaration,
    build_preregistration,
    final_targets,
    log_loss,
    make_release,
    policy_metrics,
    runtime_models,
    sibling_protocols,
    verify_release,
)

_PREDICTION_IMPORT_ERROR: Exception | None = None
try:
    from narrative_dynamics.contracts import ExperimentStage
    from narrative_dynamics.external_prediction import (
        ExternalFinalPredictionArtifact,
        predict_external_final_once,
        score_external_prediction_artifact,
    )
    from narrative_dynamics.external_validation import (
        assemble_external_final_from_reports,
    )
except Exception as error:
    _PREDICTION_IMPORT_ERROR = error


def build_bundle():
    data, brier_protocol, log_protocol = sibling_protocols()
    evidence = build_external_declaration(data)
    preregistration = build_preregistration(
        dataset=data,
        brier_protocol=brier_protocol,
        log_protocol=log_protocol,
    )
    brier_release = make_release(
        brier_protocol,
        evidence_hash=evidence.content_hash,
        preregistration_hash=preregistration.content_hash,
        score_role="brier",
    )
    log_release = make_release(
        log_protocol,
        evidence_hash=evidence.content_hash,
        preregistration_hash=preregistration.content_hash,
        score_role="log",
    )
    brier_verified = verify_release(brier_release, brier_protocol)
    log_verified = verify_release(log_release, log_protocol)
    preflight = preflight_external_releases(
        preregistration=preregistration,
        evidence=evidence,
        brier_protocol=brier_protocol,
        brier_release=brier_release,
        brier_verified=brier_verified,
        log_protocol=log_protocol,
        log_release=log_release,
        log_verified=log_verified,
    )
    models, counters = runtime_models(brier_protocol.candidates)
    return {
        "data": data,
        "preregistration": preregistration,
        "evidence": evidence,
        "brier_protocol": brier_protocol,
        "log_protocol": log_protocol,
        "brier_release": brier_release,
        "log_release": log_release,
        "brier_verified": brier_verified,
        "log_verified": log_verified,
        "preflight": preflight,
        "models": models,
        "counters": counters,
        "targets": final_targets(data),
    }


class ExternalPredictionTests(unittest.TestCase):
    def require_prediction(self):
        self.assertIsNone(
            _PREDICTION_IMPORT_ERROR,
            f"sealed external prediction boundary is missing: {_PREDICTION_IMPORT_ERROR}",
        )

    def _artifact(self):
        self.require_prediction()
        bundle = build_bundle()
        artifact = predict_external_final_once(
            runner=SimulationRunner(),
            preregistration=bundle["preregistration"],
            preflight=bundle["preflight"],
            brier_protocol=bundle["brier_protocol"],
            log_protocol=bundle["log_protocol"],
            models=bundle["models"],
            final_targets=bundle["targets"],
            extractor=policy_metrics,
            repository_revision="test-repository-revision",
        )
        return bundle, artifact

    def test_external_prediction_stage_exists(self):
        self.require_prediction()
        self.assertEqual(ExperimentStage.EXTERNAL_PREDICTION.value, "external_prediction")

    def test_dual_protocol_preflight_happens_before_any_model_execution(self):
        self.require_prediction()
        bundle = build_bundle()
        forged = replace(bundle["preflight"], log_protocol_hash=bundle["brier_protocol"].content_hash)
        with self.assertRaises(ValueError):
            predict_external_final_once(
                runner=SimulationRunner(),
                preregistration=bundle["preregistration"],
                preflight=forged,
                brier_protocol=bundle["brier_protocol"],
                log_protocol=bundle["log_protocol"],
                models=bundle["models"],
                final_targets=bundle["targets"],
                extractor=policy_metrics,
                repository_revision="test-repository-revision",
            )
        self.assertEqual(tuple(model.calls for model in bundle["counters"]), (0, 0))

    def test_final_prediction_artifact_executes_each_model_case_seed_once(self):
        bundle, artifact = self._artifact()
        expected = len(bundle["targets"].cases) * len(bundle["brier_protocol"].simulation_seeds)
        self.assertEqual(tuple(model.calls for model in bundle["counters"]), (expected, expected))
        for model_prediction in artifact.model_predictions:
            self.assertEqual(len(model_prediction.predictions), expected)

    def test_prediction_artifact_binds_preflight_both_protocols_target_candidate_metric_seed_and_run_lineage(self):
        bundle, artifact = self._artifact()
        self.assertIsInstance(artifact, ExternalFinalPredictionArtifact)
        self.assertEqual(artifact.preflight_hash, bundle["preflight"].content_hash)
        self.assertEqual(artifact.brier_protocol_hash, bundle["brier_protocol"].content_hash)
        self.assertEqual(artifact.log_protocol_hash, bundle["log_protocol"].content_hash)
        self.assertEqual(artifact.final_target_hash, bundle["targets"].content_hash)
        self.assertEqual(artifact.simulation_seeds, bundle["brier_protocol"].simulation_seeds)
        self.assertEqual(artifact.manifest.stage, ExperimentStage.EXTERNAL_PREDICTION)
        run_hashes = {
            item.run_manifest_hash
            for model in artifact.model_predictions
            for item in model.predictions
        }
        self.assertTrue(run_hashes)
        self.assertTrue(run_hashes.issubset(set(artifact.manifest.parent_hashes)))

    def test_brier_and_log_scoring_consume_same_artifact_without_runner(self):
        bundle, artifact = self._artifact()
        calls = tuple(model.calls for model in bundle["counters"])
        brier_report = score_external_prediction_artifact(
            artifact=artifact,
            verified_release=bundle["brier_verified"],
            protocol=bundle["brier_protocol"],
            target_set=bundle["targets"],
            loss=brier_loss(),
        )
        log_report = score_external_prediction_artifact(
            artifact=artifact,
            verified_release=bundle["log_verified"],
            protocol=bundle["log_protocol"],
            target_set=bundle["targets"],
            loss=log_loss(),
        )
        self.assertEqual(tuple(model.calls for model in bundle["counters"]), calls)
        self.assertEqual(set(brier_report.entry_map), set(log_report.entry_map))

    def test_precomputed_scoring_matches_direct_comparator_losses_on_synthetic_fixture(self):
        bundle, artifact = self._artifact()
        precomputed = score_external_prediction_artifact(
            artifact=artifact,
            verified_release=bundle["brier_verified"],
            protocol=bundle["brier_protocol"],
            target_set=bundle["targets"],
            loss=brier_loss(),
        )
        direct_models, _ = runtime_models(bundle["brier_protocol"].candidates)
        direct = compare_released_models(
            runner=SimulationRunner(),
            verified_release=bundle["brier_verified"],
            protocol=bundle["brier_protocol"],
            models=direct_models,
            target_set=bundle["targets"],
            extractor=policy_metrics,
            loss=brier_loss(),
        )
        self.assertEqual(tuple(item.name for item in precomputed.comparison.ranking), tuple(item.name for item in direct.comparison.ranking))
        for name in precomputed.entry_map:
            left = precomputed.entry_map[name]
            right = direct.entry_map[name]
            self.assertEqual(left.mean_loss, right.mean_loss)
            self.assertEqual(left.worst_loss, right.worst_loss)
            self.assertEqual(left.adequate, right.adequate)

    def test_artifact_or_protocol_identity_drift_is_rejected_before_scoring(self):
        bundle, artifact = self._artifact()
        changed = replace(artifact, final_target_hash=bundle["preflight"].content_hash)
        with self.assertRaises(ValueError):
            score_external_prediction_artifact(
                artifact=changed,
                verified_release=bundle["brier_verified"],
                protocol=bundle["brier_protocol"],
                target_set=bundle["targets"],
                loss=brier_loss(),
            )

    def test_external_final_can_be_assembled_from_precomputed_released_reports(self):
        bundle, artifact = self._artifact()
        brier_report = score_external_prediction_artifact(
            artifact=artifact,
            verified_release=bundle["brier_verified"],
            protocol=bundle["brier_protocol"],
            target_set=bundle["targets"],
            loss=brier_loss(),
        )
        log_report = score_external_prediction_artifact(
            artifact=artifact,
            verified_release=bundle["log_verified"],
            protocol=bundle["log_protocol"],
            target_set=bundle["targets"],
            loss=log_loss(),
        )
        evaluation = assemble_external_final_from_reports(
            preregistration=bundle["preregistration"],
            evidence=bundle["evidence"],
            brier_protocol=bundle["brier_protocol"],
            brier_release=bundle["brier_release"],
            brier_verified=bundle["brier_verified"],
            brier_report=brier_report,
            log_protocol=bundle["log_protocol"],
            log_release=bundle["log_release"],
            log_verified=bundle["log_verified"],
            log_report=log_report,
            final_targets=bundle["targets"],
        )
        self.assertEqual(evaluation.preflight.content_hash, bundle["preflight"].content_hash)
        self.assertIs(evaluation.brier_report, brier_report)
        self.assertIs(evaluation.log_report, log_report)

    def test_precomputed_child_release_or_verification_drift_is_rejected(self):
        bundle, artifact = self._artifact()
        brier_report = score_external_prediction_artifact(
            artifact=artifact,
            verified_release=bundle["brier_verified"],
            protocol=bundle["brier_protocol"],
            target_set=bundle["targets"],
            loss=brier_loss(),
        )
        log_report = score_external_prediction_artifact(
            artifact=artifact,
            verified_release=bundle["log_verified"],
            protocol=bundle["log_protocol"],
            target_set=bundle["targets"],
            loss=log_loss(),
        )
        forged = replace(log_report, release_hash=bundle["brier_release"].content_hash)
        with self.assertRaises(ValueError):
            assemble_external_final_from_reports(
                preregistration=bundle["preregistration"],
                evidence=bundle["evidence"],
                brier_protocol=bundle["brier_protocol"],
                brier_release=bundle["brier_release"],
                brier_verified=bundle["brier_verified"],
                brier_report=brier_report,
                log_protocol=bundle["log_protocol"],
                log_release=bundle["log_release"],
                log_verified=bundle["log_verified"],
                log_report=forged,
                final_targets=bundle["targets"],
            )


if __name__ == "__main__":
    unittest.main()
