from dataclasses import replace
import unittest

from narrative_dynamics.abm.calibration_contracts import (
    ABMCalibrationCandidate,
    ABMCalibrationWeights,
    CalibrationSplit,
    EmpiricalABMCase,
    EmpiricalABMDataset,
    ObservedABMSnapshot,
)
from tests.abm_calibration_fixtures import empirical_case, snapshot


class EmpiricalABMCalibrationContractTests(unittest.TestCase):
    def test_dataset_requires_train_and_holdout_and_is_canonical(self):
        train = empirical_case("train", CalibrationSplit.TRAIN)
        holdout = empirical_case("holdout", CalibrationSplit.HOLDOUT)
        first = EmpiricalABMDataset("observations", "1", (train, holdout))
        second = EmpiricalABMDataset("observations", "1", (holdout, train))

        self.assertEqual(first.cases[0].case_id, "holdout")
        self.assertEqual(first, second)
        self.assertEqual(first.content_hash, second.content_hash)
        with self.assertRaisesRegex(ValueError, "TRAIN and HOLDOUT"):
            EmpiricalABMDataset("bad", "1", (train,))

    def test_case_canonicalizes_beliefs_and_binds_every_schedule_round(self):
        forward = empirical_case("train", CalibrationSplit.TRAIN)
        reverse = empirical_case(
            "train",
            CalibrationSplit.TRAIN,
            reverse_beliefs=True,
        )

        self.assertEqual(forward, reverse)
        self.assertEqual(forward.initial_beliefs, (("a", 1.0), ("c", 0.0)))
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            replace(forward, truth_observation_schedule=())
        with self.assertRaisesRegex(ValueError, "consecutive rounds"):
            replace(forward, observations=(replace(snapshot(), round_index=2),))

    def test_observed_snapshot_metrics_are_bounded_and_hashable(self):
        value = snapshot()
        self.assertTrue(value.content_hash.startswith("sha256:"))
        with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
            replace(value, mean_trust=1.1)
        with self.assertRaisesRegex(ValueError, "positive"):
            replace(value, round_index=0)

    def test_candidate_validates_ordered_rewiring_thresholds(self):
        candidate = ABMCalibrationCandidate(0.5, 0.2, 0.8)
        self.assertEqual(candidate.canonical_tuple, (0.5, 0.2, 0.8))
        with self.assertRaisesRegex(ValueError, "strictly below"):
            ABMCalibrationCandidate(0.5, 0.8, 0.8)

    def test_weights_require_finite_nonnegative_and_some_positive_weight(self):
        weights = ABMCalibrationWeights.uniform()
        self.assertEqual(len(weights.metric_items), 10)
        with self.assertRaisesRegex(ValueError, "at least one positive"):
            ABMCalibrationWeights(*([0.0] * 10))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            replace(weights, mean_trust=-1.0)

    def test_duplicate_case_ids_and_initial_belief_ids_fail_closed(self):
        train = empirical_case("same", CalibrationSplit.TRAIN)
        holdout = empirical_case("same", CalibrationSplit.HOLDOUT)
        with self.assertRaisesRegex(ValueError, "case ids must be unique"):
            EmpiricalABMDataset("bad", "1", (train, holdout))
        with self.assertRaisesRegex(ValueError, "belief agent ids must be unique"):
            EmpiricalABMCase(
                "bad",
                CalibrationSplit.TRAIN,
                (("a", 0.5), ("a", 0.6)),
                train.environment_event_schedule,
                train.truth_observation_schedule,
                train.observations,
            )

    def test_contract_values_are_immutable(self):
        dataset = EmpiricalABMDataset(
            "observations",
            "1",
            (
                empirical_case("train", CalibrationSplit.TRAIN),
                empirical_case("holdout", CalibrationSplit.HOLDOUT),
            ),
        )
        with self.assertRaises(Exception):
            dataset.version = "2"


if __name__ == "__main__":
    unittest.main()
