from __future__ import annotations

import unittest

from narrative_dynamics.simulation import SimulationRunner

from tests.external_validation_fixtures import (
    FINAL_SEEDS,
    brier_loss,
    build_external_declaration,
    build_preregistration,
    default_strata,
    final_targets,
    log_loss,
    make_release,
    policy_metrics,
    runtime_models,
    sibling_protocols,
    verify_release,
)

try:
    from narrative_dynamics.external_validation import (
        ExternalStratum,
        PairwiseSeparationRule,
        PredictiveAdequacyStatus,
        PredictiveSeparationStatus,
        evaluate_external_final,
    )
except ImportError as error:
    ExternalStratum = None
    PairwiseSeparationRule = None
    PredictiveAdequacyStatus = None
    PredictiveSeparationStatus = None
    evaluate_external_final = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


class ExternalValidationFinalTests(unittest.TestCase):
    def _execute(self, *, separation_rule=None, strata=None):
        if evaluate_external_final is None:
            self.fail(f"external final API is missing: {IMPORT_ERROR}")
        data, brier, log = sibling_protocols()
        declaration = build_external_declaration(data)
        prereg = build_preregistration(
            dataset=data,
            brier_protocol=brier,
            log_protocol=log,
            separation_rule=separation_rule,
            strata=default_strata() if strata is None else strata,
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
        brier_models, brier_sources = runtime_models(brier.candidates)
        log_models, log_sources = runtime_models(log.candidates)
        evaluation = evaluate_external_final(
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
        return {
            "dataset": data,
            "prereg": prereg,
            "evaluation": evaluation,
            "sources": brier_sources + log_sources,
        }

    def test_dual_final_uses_existing_released_comparisons_for_both_scores(self):
        evidence = self._execute()
        evaluation = evidence["evaluation"]
        self.assertEqual(
            evaluation.brier_report.release_hash,
            evaluation.preflight.brier_release_hash,
        )
        self.assertEqual(
            evaluation.log_report.release_hash,
            evaluation.preflight.log_release_hash,
        )
        self.assertEqual(
            evaluation.brier_report.manifest.stage.value,
            "released_model_comparison",
        )
        self.assertEqual(
            evaluation.log_report.manifest.stage.value,
            "released_model_comparison",
        )

    def test_per_score_and_aggregate_predictive_adequacy_are_claim_safe(self):
        evaluation = self._execute()["evaluation"]
        self.assertEqual(len(evaluation.adequacy_findings), 4)
        self.assertEqual(len(evaluation.aggregate_adequacy), 2)
        allowed = {
            PredictiveAdequacyStatus.MET,
            PredictiveAdequacyStatus.NOT_MET,
        }
        self.assertTrue(
            all(item.status in allowed for item in evaluation.adequacy_findings)
        )
        self.assertTrue(
            all(status in allowed for _, status in evaluation.aggregate_adequacy)
        )
        self.assertFalse(hasattr(evaluation, "identification_status"))

    def test_pairwise_separation_requires_same_direction_and_both_delta_thresholds(self):
        low = self._execute(
            separation_rule=PairwiseSeparationRule(
                min_mean_loss_delta_brier=1e-6,
                min_mean_loss_delta_log=1e-6,
            )
        )["evaluation"]
        self.assertEqual(
            low.separation_findings[0].status,
            PredictiveSeparationStatus.SEPARATED,
        )
        high = self._execute(
            separation_rule=PairwiseSeparationRule(
                min_mean_loss_delta_brier=100.0,
                min_mean_loss_delta_log=100.0,
            )
        )["evaluation"]
        self.assertEqual(
            high.separation_findings[0].status,
            PredictiveSeparationStatus.NOT_SEPARATED,
        )

    def test_nonseparation_preserves_both_raw_score_deltas(self):
        evaluation = self._execute(
            separation_rule=PairwiseSeparationRule(
                min_mean_loss_delta_brier=100.0,
                min_mean_loss_delta_log=100.0,
            )
        )["evaluation"]
        finding = evaluation.separation_findings[0]
        self.assertNotEqual(finding.brier_mean_loss_delta, 0.0)
        self.assertNotEqual(finding.log_mean_loss_delta, 0.0)
        self.assertEqual(
            finding.status,
            PredictiveSeparationStatus.NOT_SEPARATED,
        )

    def test_stratum_scores_use_existing_case_losses_without_resimulation(self):
        evidence = self._execute()
        expected_calls_per_model = len(final_targets(evidence["dataset"]).cases) * len(
            FINAL_SEEDS
        )
        self.assertEqual(
            tuple(model.calls for model in evidence["sources"]),
            (expected_calls_per_model,) * 4,
        )
        self.assertEqual(
            {score.stratum_name for score in evidence["evaluation"].stratum_scores},
            {"early", "late"},
        )

    def test_global_scores_are_unchanged_by_stratum_declarations(self):
        first = self._execute(
            strata=(ExternalStratum("all", ("ext:final:1", "ext:final:2")),)
        )["evaluation"]
        second = self._execute(strata=default_strata())["evaluation"]
        first_values = {
            (item.model_name, item.score_role.value): (item.mean_loss, item.worst_loss)
            for item in first.adequacy_findings
        }
        second_values = {
            (item.model_name, item.score_role.value): (item.mean_loss, item.worst_loss)
            for item in second.adequacy_findings
        }
        self.assertEqual(first_values, second_values)


if __name__ == "__main__":
    unittest.main()
