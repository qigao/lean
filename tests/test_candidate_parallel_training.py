from __future__ import annotations

from copy import deepcopy
import unittest

from narrative_dynamics.adapters.prison_metrics import prison_initial_action_metrics
from narrative_dynamics.observations.dataset import ObservationPartitionRole
from narrative_dynamics.simulation import SimulationRunner
from tests.test_observational_training_fit import (
    TrainingProbabilityModel,
    brier_loss,
    targets_for,
)


_IMPORT_ERROR: Exception | None = None
try:
    from narrative_dynamics.observations.training_shards import (
        TrainingCandidateShard,
        assemble_training_fit_report,
        evaluate_training_candidate,
    )
except Exception as error:
    _IMPORT_ERROR = error


CANDIDATES = (
    (("p", 0.2),),
    (("p", 0.5),),
    (("p", 0.8),),
)
SEEDS = (101, 102)


def _reference():
    from narrative_dynamics.observations.training import fit_training_target_grid

    return fit_training_target_grid(
        runner=SimulationRunner(),
        model=TrainingProbabilityModel(),
        target_report=targets_for(ObservationPartitionRole.TRAIN),
        parameter_grid={"p": (0.2, 0.5, 0.8)},
        simulation_seeds=SEEDS,
        extractor=prison_initial_action_metrics,
        loss=brier_loss(),
    )


def _shards():
    target_report = targets_for(ObservationPartitionRole.TRAIN)
    model = TrainingProbabilityModel()
    return target_report, model, tuple(
        evaluate_training_candidate(
            runner=SimulationRunner(),
            model=model,
            target_report=target_report,
            parameters=parameters,
            simulation_seeds=SEEDS,
            extractor=prison_initial_action_metrics,
            loss=brier_loss(),
        )
        for parameters in CANDIDATES
    )


@unittest.skipIf(_IMPORT_ERROR is not None, "training shard API is not implemented yet")
class CandidateParallelTrainingContractTests(unittest.TestCase):
    def test_candidate_shards_reassemble_exact_reference_report(self):
        target_report, model, shards = _shards()
        assembled = assemble_training_fit_report(
            model=model,
            target_report=target_report,
            parameter_candidates=CANDIDATES,
            simulation_seeds=SEEDS,
            extractor=prison_initial_action_metrics,
            loss=brier_loss(),
            shards=tuple(reversed(shards)),
        )
        reference = _reference()

        self.assertEqual(assembled.ranking, reference.ranking)
        self.assertEqual(assembled.manifest, reference.manifest)
        self.assertEqual(assembled.manifest.content_hash, reference.manifest.content_hash)

    def test_shard_payload_round_trip_recomputes_declared_hash(self):
        _, _, shards = _shards()
        shard = shards[0]
        payload = shard.to_payload()

        self.assertEqual(TrainingCandidateShard.from_payload(payload), shard)
        self.assertEqual(payload["content_hash"], shard.content_hash)

        tampered = deepcopy(payload)
        tampered["parameters"] = [["p", 0.5]]
        with self.assertRaises(ValueError):
            TrainingCandidateShard.from_payload(tampered)

    def test_assembly_rejects_missing_duplicate_and_undeclared_candidates(self):
        target_report, model, shards = _shards()

        def assemble(values, candidates=CANDIDATES):
            return assemble_training_fit_report(
                model=model,
                target_report=target_report,
                parameter_candidates=candidates,
                simulation_seeds=SEEDS,
                extractor=prison_initial_action_metrics,
                loss=brier_loss(),
                shards=values,
            )

        with self.assertRaises(ValueError):
            assemble(shards[:-1])
        with self.assertRaises(ValueError):
            assemble(shards + (shards[0],))

        undeclared = evaluate_training_candidate(
            runner=SimulationRunner(),
            model=model,
            target_report=target_report,
            parameters=(("p", 0.9),),
            simulation_seeds=SEEDS,
            extractor=prison_initial_action_metrics,
            loss=brier_loss(),
        )
        with self.assertRaises(ValueError):
            assemble(shards + (undeclared,))


class CandidateParallelTrainingRedTests(unittest.TestCase):
    def test_training_candidate_shard_api_exists(self):
        self.assertIsNone(
            _IMPORT_ERROR,
            f"candidate-parallel training shard API is missing: {_IMPORT_ERROR}",
        )


if __name__ == "__main__":
    unittest.main()
