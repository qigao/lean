from __future__ import annotations

import json
import math
import pickle
import unittest

from narrative_dynamics.cross_dataset_inference import (
    _ParticipantLossBlock,
    _linear_percentile,
    fit_train_base_rate,
    paired_participant_bootstrap,
    participant_equal_loss,
    trial_equal_loss,
)
from tests.cross_dataset_transfer_fixtures import (
    digest,
    participant_blocks,
    train_projection,
)


class CrossDatasetInferenceTests(unittest.TestCase):
    def test_train_base_rate_uses_laplace_one_per_stratum(self) -> None:
        baseline = fit_train_base_rate(
            train_projection(("action_1", "action_1", "action_0"))
        )
        self.assertEqual(len(baseline.cells), 1)
        self.assertEqual(baseline.cells[0].probability_action_1, 3 / 5)
        self.assertEqual(baseline.cells[0].probability_action_0, 2 / 5)

    def test_unknown_baseline_stratum_fails_closed(self) -> None:
        baseline = fit_train_base_rate(train_projection())
        with self.assertRaisesRegex(KeyError, "unknown source stratum"):
            baseline.probabilities_for("absent")

    def test_participant_equal_and_trial_equal_reference_vectors(self) -> None:
        blocks = (
            _ParticipantLossBlock(token=digest("p1"), losses=(0.0, 0.0)),
            _ParticipantLossBlock(token=digest("p2"), losses=(1.0,)),
        )
        self.assertEqual(participant_equal_loss(blocks), 0.5)
        self.assertEqual(trial_equal_loss(blocks), 1 / 3)

    def test_paired_bootstrap_is_exact_and_aggregate_only(self) -> None:
        evidence = paired_participant_bootstrap(
            candidate=participant_blocks((0.10, 0.20, 0.30)),
            comparator=participant_blocks((0.20, 0.30, 0.40)),
            score="BRIER",
            aggregation="PARTICIPANT_EQUAL",
            candidate_identity=digest("candidate"),
            comparator_identity=digest("baseline"),
            seed=43001,
            replicates=10000,
        )
        repeated = paired_participant_bootstrap(
            candidate=participant_blocks((0.10, 0.20, 0.30)),
            comparator=participant_blocks((0.20, 0.30, 0.40)),
            score="BRIER",
            aggregation="PARTICIPANT_EQUAL",
            candidate_identity=digest("candidate"),
            comparator_identity=digest("baseline"),
            seed=43001,
            replicates=10000,
        )
        self.assertEqual(evidence, repeated)
        self.assertEqual(evidence.bootstrap_seed, 43001)
        self.assertEqual(evidence.bootstrap_replicates, 10000)
        self.assertEqual(
            evidence.pass_status,
            evidence.relative_improvement >= 0.01 and evidence.lower_95 > 0.0,
        )
        durable = evidence.to_payload()
        self.assertEqual(durable["participant_count"], 3)
        payload = json.dumps(durable, sort_keys=True).lower()
        self.assertNotIn("token", payload)
        self.assertNotIn("blocks", payload)
        self.assertNotIn("losses", payload)

    def test_linear_percentile_interpolation_is_frozen(self) -> None:
        self.assertEqual(_linear_percentile((0.0, 10.0, 20.0, 30.0), 0.25), 7.5)
        self.assertEqual(_linear_percentile((0.0, 10.0, 20.0, 30.0), 0.975), 29.25)

    def test_brier_and_clipped_log_use_the_same_paired_protocol(self) -> None:
        for score in ("BRIER", "CLIPPED_LOG"):
            with self.subTest(score=score):
                evidence = paired_participant_bootstrap(
                    candidate=participant_blocks((0.2, 0.3)),
                    comparator=participant_blocks((0.4, 0.5)),
                    score=score,
                    aggregation="PARTICIPANT_EQUAL",
                    candidate_identity=digest(f"{score}-candidate"),
                    comparator_identity=digest(f"{score}-comparator"),
                    seed=43001,
                    replicates=10000,
                )
                self.assertEqual(evidence.score, score)
                self.assertTrue(evidence.pass_status)

    def test_exact_threshold_and_zero_lower_endpoint(self) -> None:
        threshold = paired_participant_bootstrap(
            candidate=participant_blocks((0.99, 0.99)),
            comparator=participant_blocks((1.0, 1.0)),
            score="BRIER",
            aggregation="PARTICIPANT_EQUAL",
            candidate_identity=digest("threshold-candidate"),
            comparator_identity=digest("threshold-comparator"),
            seed=43001,
            replicates=10000,
        )
        self.assertGreaterEqual(threshold.relative_improvement, 0.01)
        self.assertTrue(threshold.pass_status)

        zero_lower = paired_participant_bootstrap(
            candidate=participant_blocks((1.0, 0.98)),
            comparator=participant_blocks((1.0, 1.0)),
            score="BRIER",
            aggregation="PARTICIPANT_EQUAL",
            candidate_identity=digest("zero-candidate"),
            comparator_identity=digest("zero-comparator"),
            seed=43001,
            replicates=10000,
        )
        self.assertEqual(zero_lower.lower_95, 0.0)
        self.assertFalse(zero_lower.pass_status)

    def test_private_blocks_reject_empty_duplicate_nonfinite_and_serialization(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            _ParticipantLossBlock(token=digest("empty"), losses=())
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    _ParticipantLossBlock(token=digest("bad"), losses=(value,))
        block = _ParticipantLossBlock(token=digest("private"), losses=(0.2,))
        with self.assertRaises(TypeError):
            pickle.dumps(block)
        with self.assertRaisesRegex(ValueError, "duplicate token"):
            participant_equal_loss((block, block))

    def test_paired_bootstrap_rejects_unequal_tokens_and_protocol_drift(self) -> None:
        candidate = participant_blocks((0.1, 0.2))
        comparator = participant_blocks((0.2, 0.3))
        mismatched = (
            comparator[0],
            _ParticipantLossBlock(token=digest("different"), losses=(0.3,)),
        )
        with self.assertRaisesRegex(ValueError, "same private tokens"):
            paired_participant_bootstrap(
                candidate=candidate,
                comparator=mismatched,
                score="BRIER",
                aggregation="PARTICIPANT_EQUAL",
                candidate_identity=digest("candidate"),
                comparator_identity=digest("comparator"),
                seed=43001,
                replicates=10000,
            )
        for field, value in (("seed", 1), ("replicates", 9999)):
            with self.subTest(field=field):
                kwargs = {"seed": 43001, "replicates": 10000, field: value}
                with self.assertRaisesRegex(ValueError, "frozen"):
                    paired_participant_bootstrap(
                        candidate=candidate,
                        comparator=comparator,
                        score="BRIER",
                        aggregation="PARTICIPANT_EQUAL",
                        candidate_identity=digest("candidate"),
                        comparator_identity=digest("comparator"),
                        **kwargs,
                    )


if __name__ == "__main__":
    unittest.main()
