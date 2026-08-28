from __future__ import annotations

import unittest

from narrative_dynamics.observations import AdequacyThresholds

from tests.external_validation_fixtures import (
    ALTERNATIVE_SELECTION_HASH,
    METHOD_HASH,
    build_constraint_plan,
    build_external_declaration,
    build_preregistration,
    default_separation_rule,
    external_dataset,
    final_targets,
    frozen_candidates,
    policy_metrics_v2,
    sibling_protocols,
)

try:
    from narrative_dynamics.external_validation import (
        ExternalScoreRole,
        ExternalStratum,
        ExternalValidationError,
        ExternalValidationPreregistration,
        PairwiseSeparationRule,
    )
except ImportError as error:
    ExternalScoreRole = None
    ExternalStratum = None
    ExternalValidationError = None
    ExternalValidationPreregistration = None
    PairwiseSeparationRule = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


class ExternalValidationPreregistrationTests(unittest.TestCase):
    def _api(self):
        if ExternalValidationPreregistration is None:
            self.fail(f"external preregistration API is missing: {IMPORT_ERROR}")

    def _create(self, **kwargs):
        self._api()
        return build_preregistration(**kwargs)

    def test_score_roles_are_exactly_brier_then_log(self):
        self._api()
        self.assertEqual(
            tuple(role.value for role in ExternalScoreRole),
            ("brier", "log"),
        )

    def test_preregistration_binds_evidence_both_protocols_and_score_specific_thresholds(self):
        data, brier, log = sibling_protocols()
        declaration = build_external_declaration(data)
        prereg = self._create(
            dataset=data,
            brier_protocol=brier,
            log_protocol=log,
        )
        self.assertEqual(
            prereg.evidence_declaration_hash,
            declaration.content_hash,
        )
        self.assertEqual(prereg.brier_protocol_hash, brier.content_hash)
        self.assertEqual(prereg.log_protocol_hash, log.content_hash)
        self.assertEqual(
            tuple(role for role, _ in prereg.adequacy_thresholds_by_score),
            (ExternalScoreRole.BRIER, ExternalScoreRole.LOG),
        )
        self.assertEqual(prereg.method_validation_hashes, (METHOD_HASH,))

    def test_preregistration_accepts_different_brier_log_threshold_numbers_when_each_matches_its_protocol(self):
        _, brier, log = sibling_protocols(
            brier_thresholds=AdequacyThresholds(0.5, 0.75),
            log_thresholds=AdequacyThresholds(2.5, 2.75),
        )
        prereg = self._create(brier_protocol=brier, log_protocol=log)
        values = tuple(
            thresholds.identity_payload()
            for _, thresholds in prereg.adequacy_thresholds_by_score
        )
        self.assertNotEqual(values[0], values[1])

    def test_duplicate_or_unknown_score_family_is_rejected(self):
        self._api()
        data, brier, log = sibling_protocols()
        declaration = build_external_declaration(data)
        common = dict(
            name="external-validation-v1",
            version="1",
            evidence_declaration=declaration,
            brier_protocol=brier,
            log_protocol=log,
            final_target_set=final_targets(data),
            strata=(
                ExternalStratum("early", ("ext:final:1",)),
                ExternalStratum("late", ("ext:final:2",)),
            ),
            separation_rule=default_separation_rule(),
            constraint_plans=(),
            method_validation_hashes=(METHOD_HASH,),
        )
        with self.assertRaises(ExternalValidationError):
            ExternalValidationPreregistration.create(
                **common,
                adequacy_thresholds_by_score=(
                    (ExternalScoreRole.BRIER, brier.thresholds),
                    (ExternalScoreRole.BRIER, log.thresholds),
                ),
            )
        with self.assertRaises((ExternalValidationError, ValueError)):
            ExternalValidationPreregistration.create(
                **common,
                adequacy_thresholds_by_score=(
                    (ExternalScoreRole.BRIER, brier.thresholds),
                    ("unknown", log.thresholds),
                ),
            )

    def test_target_seed_metric_baseline_candidate_parameter_or_selection_lineage_drift_is_rejected(self):
        self._api()
        data, brier, _ = sibling_protocols()
        drifts = []
        different_data = external_dataset(swap_train_selection=True)
        _, _, target_drift = sibling_protocols(dataset=different_data)
        drifts.append(("target/dataset", target_drift))
        _, _, seed_drift = sibling_protocols(log_seeds=(41, 43))
        drifts.append(("seed", seed_drift))
        _, _, metric_drift = sibling_protocols(log_extractor=policy_metrics_v2)
        drifts.append(("metric", metric_drift))
        _, _, baseline_drift = sibling_protocols(log_baseline="alternative")
        drifts.append(("baseline", baseline_drift))
        parameter_candidates = frozen_candidates(alternative_p=0.3)
        _, _, parameter_drift = sibling_protocols(log_candidates=parameter_candidates)
        drifts.append(("parameter", parameter_drift))
        lineage_candidates = frozen_candidates(
            alternative_selection_hash=ALTERNATIVE_SELECTION_HASH
        )
        _, _, lineage_drift = sibling_protocols(log_candidates=lineage_candidates)
        drifts.append(("selection-lineage", lineage_drift))
        for label, drifted_log in drifts:
            with self.subTest(label=label):
                with self.assertRaises(ExternalValidationError):
                    self._create(
                        dataset=data,
                        brier_protocol=brier,
                        log_protocol=drifted_log,
                    )

    def test_score_specific_threshold_drift_is_rejected(self):
        self._api()
        _, brier, log = sibling_protocols()
        with self.assertRaises(ExternalValidationError):
            self._create(
                brier_protocol=brier,
                log_protocol=log,
                brier_thresholds=AdequacyThresholds(0.9, 1.0),
            )

    def test_strata_cover_every_final_case_exactly_once(self):
        self._api()
        prereg = self._create(
            strata=(
                ExternalStratum("b", ("ext:final:2",)),
                ExternalStratum("a", ("ext:final:1",)),
            )
        )
        self.assertEqual(tuple(stratum.name for stratum in prereg.strata), ("a", "b"))

    def test_strata_reject_missing_extra_or_overlapping_final_cases(self):
        self._api()
        invalid = (
            (ExternalStratum("only", ("ext:final:1",)),),
            (
                ExternalStratum("a", ("ext:final:1",)),
                ExternalStratum("b", ("ext:final:1", "ext:final:2")),
            ),
            (
                ExternalStratum("a", ("ext:final:1",)),
                ExternalStratum("b", ("ext:final:2", "missing")),
            ),
        )
        for strata in invalid:
            with self.subTest(strata=strata):
                with self.assertRaises(ExternalValidationError):
                    self._create(strata=strata)

    def test_pairwise_separation_rule_requires_positive_finite_deltas_and_direction_agreement(self):
        self._api()
        valid = PairwiseSeparationRule(
            min_mean_loss_delta_brier=0.01,
            min_mean_loss_delta_log=0.02,
        )
        self.assertTrue(valid.require_direction_agreement)
        for kwargs in (
            {"min_mean_loss_delta_brier": 0.0, "min_mean_loss_delta_log": 0.1},
            {"min_mean_loss_delta_brier": -0.1, "min_mean_loss_delta_log": 0.1},
            {"min_mean_loss_delta_brier": float("inf"), "min_mean_loss_delta_log": 0.1},
            {"min_mean_loss_delta_brier": 0.1, "min_mean_loss_delta_log": float("nan")},
            {
                "min_mean_loss_delta_brier": 0.1,
                "min_mean_loss_delta_log": 0.1,
                "require_direction_agreement": False,
            },
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ExternalValidationError):
                    PairwiseSeparationRule(**kwargs)

    def test_constraint_plan_is_finite_selection_scoped_and_content_hashed(self):
        self._api()
        plan = build_constraint_plan()
        self.assertEqual(plan.target_coordinates, ("p",))
        self.assertEqual(
            plan.selection_case_names,
            ("ext:selection:1", "ext:selection:2"),
        )
        self.assertTrue(plan.content_hash.startswith("sha256:"))
        with self.assertRaises(ExternalValidationError):
            build_constraint_plan(parameter_grid={})
        with self.assertRaises(ExternalValidationError):
            build_constraint_plan(parameter_grid={"p": (0.2, float("nan"))})
        with self.assertRaises(ExternalValidationError):
            build_constraint_plan(selection_case_names=("ext:final:1",))


if __name__ == "__main__":
    unittest.main()
