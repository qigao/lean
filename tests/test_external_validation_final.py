from __future__ import annotations

import unittest

from narrative_dynamics.simulation import SimulationRunner

from tests.external_validation_fixtures import (
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
        build_pairwise_separation_findings,
        build_predictive_adequacy_findings,
        build_stratum_scores,
        execute_external_final_comparisons,
    )
except ImportError as error:
    ExternalStratum = None
    PairwiseSeparationRule = None
    PredictiveAdequacyStatus = None
    PredictiveSeparationStatus = None
    build_pairwise_separation_findings = None
    build_predictive_adequacy_findings = None
    build_stratum_scores = None
    execute_external_final_comparisons = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


class ExternalValidationFinalTests(unittest.TestCase):
    def _execute(self, *, separation_rule=None, strata=None):
        if execute_external_final_comparisons is None:
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
        bundle = execute_external_final_comparisons(
            runner=SimulationRunner(),
            preregistration=prereg,
            brier_protocol=brier,
            log_protocol=log,
            brier_release=brier_release,
            log_release=log_release,
            brier_verified=verify_release(brier_release, brier),
            log_verified=verify_release(log_release, log),
            brier_models=brier_models,
            log_models=log_models,
            target_set=final_targets(data),
            extractor=policy_metrics,
            brier_loss=brier_loss(),
            log_loss=log_loss(),
        )
        return {
            "dataset": data,
            "prereg": prereg,
            "bundle": bundle,
            "sources": brier_sources + log_sources,
        }

    def test_dual_final_uses_existing_released_comparisons_for_both_scores(self):
        evidence = self._execute()
        bundle = evidence["bundle"]
        self.assertEqual(
            bundle.brier_comparison.release_hash,
            bundle.preflight.brier_release_hash,
        )
        self.assertEqual(
            bundle.log_comparison.release_hash,
            bundle.preflight.log_release_hash,
        )
        self.assertEqual(
            bundle.brier_comparison.manifest.stage.value,
            "released_model_comparison",
        )
        self.assertEqual(
            bundle.log_comparison.manifest.stage.value,
            "released_model_comparison",
        )

    def test_per_score_and_aggregate_predictive_adequacy_are_claim_safe(self):
        evidence = self._execute()
        findings = build_predictive_adequacy_findings(
            evidence["prereg"], evidence["bundle"]
        )
        self.assertEqual(len(findings.per_score), 4)
        self.assertEqual(len(findings.aggregate), 2)
        allowed = {
            PredictiveAdequacyStatus.PREDICTIVE_ADEQUACY_MET,
            PredictiveAdequacyStatus.PREDICTIVE_ADEQUACY_NOT_MET,
        }
        self.assertTrue(all(item.status in allowed for item in findings.per_score))
        self.assertTrue(all(item.status in allowed for item in findings.aggregate))
        self.assertFalse(hasattr(findings, "identification_status"))

    def test_pairwise_separation_requires_same_direction_and_both_delta_thresholds(self):
        low = self._execute(
            separation_rule=PairwiseSeparationRule(
                min_mean_loss_delta_brier=1e-6,
                min_mean_loss_delta_log=1e-6,
            )
        )
        low_finding = build_pairwise_separation_findings(
            low["prereg"], low["bundle"]
        )[0]
        self.assertEqual(
            low_finding.status,
            PredictiveSeparationStatus.PREDICTIVELY_SEPARATED_UNDER_PROTOCOL,
        )
        high = self._execute(
            separation_rule=PairwiseSeparationRule(
                min_mean_loss_delta_brier=100.0,
                min_mean_loss_delta_log=100.0,
            )
        )
        high_finding = build_pairwise_separation_findings(
            high["prereg"], high["bundle"]
        )[0]
        self.assertEqual(
            high_finding.status,
            PredictiveSeparationStatus.NOT_PREDICTIVELY_SEPARATED_UNDER_PROTOCOL,
        )

    def test_nonseparation_preserves_both_raw_score_deltas(self):
        evidence = self._execute(
            separation_rule=PairwiseSeparationRule(
                min_mean_loss_delta_brier=100.0,
                min_mean_loss_delta_log=100.0,
            )
        )
        finding = build_pairwise_separation_findings(
            evidence["prereg"], evidence["bundle"]
        )[0]
        self.assertNotEqual(finding.brier_mean_loss_delta, 0.0)
        self.assertNotEqual(finding.log_mean_loss_delta, 0.0)
        self.assertEqual(
            finding.status,
            PredictiveSeparationStatus.NOT_PREDICTIVELY_SEPARATED_UNDER_PROTOCOL,
        )

    def test_stratum_scores_use_existing_case_losses_without_resimulation(self):
        evidence = self._execute()
        before = tuple(model.calls for model in evidence["sources"])
        scores = build_stratum_scores(evidence["prereg"], evidence["bundle"])
        after = tuple(model.calls for model in evidence["sources"])
        self.assertEqual(before, after)
        self.assertEqual({score.stratum_name for score in scores}, {"early", "late"})

    def test_global_scores_are_unchanged_by_stratum_declarations(self):
        first = self._execute(
            strata=(ExternalStratum("all", ("ext:final:1", "ext:final:2")),)
        )
        second = self._execute(strata=default_strata())
        first_findings = build_predictive_adequacy_findings(
            first["prereg"], first["bundle"]
        )
        second_findings = build_predictive_adequacy_findings(
            second["prereg"], second["bundle"]
        )
        first_values = {
            (item.model_name, item.score_role.value): (item.mean_loss, item.worst_loss)
            for item in first_findings.per_score
        }
        second_values = {
            (item.model_name, item.score_role.value): (item.mean_loss, item.worst_loss)
            for item in second_findings.per_score
        }
        self.assertEqual(first_values, second_values)


if __name__ == "__main__":
    unittest.main()
