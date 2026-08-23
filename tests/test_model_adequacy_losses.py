from __future__ import annotations

import unittest

from narrative_dynamics.calibration import calibrate_grid
from narrative_dynamics.contracts import ModelRun, Scenario
from narrative_dynamics.losses import (
    CategoricalBrierLoss,
    CategoricalLogLoss,
    CategoricalMetricGroup,
    WeightedSquaredErrorLoss,
)
from narrative_dynamics.simulation import SimulationRunner


class AdequacyProbabilityModel:
    name = "adequacy-probability"

    def simulate(self, scenario, parameters, rng):
        probability = float(parameters["p"])
        return ModelRun(
            events=(),
            outcome={
                "policy": {
                    "a": probability,
                    "b": 1.0 - probability,
                }
            },
        )


def policy_metrics(trace):
    policy = trace.outcome["policy"]
    return {
        "choice.a": float(policy["a"]),
        "choice.b": float(policy["b"]),
    }


policy_metrics.version = "1.0.0"


class ProperScoringLossTests(unittest.TestCase):
    def setUp(self):
        self.group = CategoricalMetricGroup(
            name="choice",
            keys=("choice.a", "choice.b"),
        )
        self.target = {"choice.a": 0.8, "choice.b": 0.2}

    def test_brier_and_log_losses_rank_true_probability_and_enter_manifest(self):
        for loss in (
            CategoricalBrierLoss(groups=(self.group,)),
            CategoricalLogLoss(groups=(self.group,)),
        ):
            with self.subTest(loss=loss.__class__.__name__):
                report = calibrate_grid(
                    runner=SimulationRunner(),
                    model=AdequacyProbabilityModel(),
                    scenario=Scenario(id="probabilities", payload={}),
                    parameter_grid={"p": (0.2, 0.8)},
                    seeds=(1,),
                    extractor=policy_metrics,
                    target=self.target,
                    loss=loss,
                )

                self.assertEqual(report.best.parameters, (("p", 0.8),))
                self.assertEqual(
                    report.manifest.inputs["loss"]["content_hash"],
                    loss.content_hash,
                )

    def test_default_loss_matches_explicit_weighted_squared_error(self):
        common = dict(
            runner=SimulationRunner(),
            model=AdequacyProbabilityModel(),
            scenario=Scenario(id="default-loss", payload={}),
            parameter_grid={"p": (0.2, 0.8)},
            seeds=(1,),
            extractor=policy_metrics,
            target=self.target,
        )
        implicit = calibrate_grid(**common)
        explicit = calibrate_grid(
            **common,
            loss=WeightedSquaredErrorLoss(),
        )

        self.assertEqual(implicit.ranking, explicit.ranking)
        self.assertEqual(
            implicit.manifest.inputs["loss"],
            explicit.manifest.inputs["loss"],
        )

    def test_categorical_definitions_and_probability_vectors_fail_closed(self):
        with self.assertRaises(ValueError):
            CategoricalMetricGroup(name="empty", keys=())
        with self.assertRaises(ValueError):
            CategoricalBrierLoss(
                groups=(
                    CategoricalMetricGroup("first", ("choice.a",)),
                    CategoricalMetricGroup("second", ("choice.a", "choice.b")),
                )
            )

        brier = CategoricalBrierLoss(groups=(self.group,))
        with self.assertRaises(ValueError):
            brier(
                {"choice.a": 0.4, "choice.b": 0.4},
                self.target,
            )
        with self.assertRaises(ValueError):
            brier(
                {"choice.a": 0.8, "choice.b": 0.2},
                self.target,
                weights={"choice.a": 1.0},
            )

        log_loss = CategoricalLogLoss(groups=(self.group,))
        with self.assertRaises(ValueError):
            log_loss(
                {"choice.a": 0.0, "choice.b": 1.0},
                self.target,
            )


if __name__ == "__main__":
    unittest.main()
