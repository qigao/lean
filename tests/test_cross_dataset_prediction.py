from __future__ import annotations

from dataclasses import replace
import math
import pickle
import unittest

from narrative_dynamics.cross_dataset_prediction import (
    SealedTransferPredictionArtifact,
    execute_transfer_final_predictions,
)
from narrative_dynamics.cross_dataset_capabilities import FinalWorkerProjection
from narrative_dynamics.cross_dataset_source import CanonicalTransferTrial
from tests.cross_dataset_transfer_fixtures import (
    digest,
    final_worker_projection,
    recording_evaluator,
    recording_progress,
    refit_freeze,
    sibling_releases,
    zero_shot_freeze,
)


class CrossDatasetPredictionTests(unittest.TestCase):
    def _execute(self, **overrides: object):
        calls: list[tuple[object, ...]] = []
        projection = final_worker_projection(4)
        values: dict[str, object] = {
            "projection": projection,
            "zero_shot": zero_shot_freeze(),
            "refit": refit_freeze(),
            "seeds": (301, 302),
            "evaluator": recording_evaluator(calls),
            "progress": recording_progress(calls),
            "expected_final_commitment_hash": projection.commitment_hash,
            "prediction_artifact_identity": sibling_releases()[0].prediction_artifact_identity,
        }
        values.update(overrides)
        return execute_transfer_final_predictions(**values), calls

    def test_every_final_case_runs_six_candidates_by_two_seeds_once(self) -> None:
        artifact, calls = self._execute()
        self.assertEqual(len(calls), 4 * 6 * 2)
        self.assertEqual(len(set(calls)), len(calls))
        self.assertEqual(artifact.case_count, 4)
        self.assertEqual(artifact.completed_model_runs, 48)

    def test_prediction_artifact_has_no_durable_serializer(self) -> None:
        artifact, _calls = self._execute()
        self.assertIsInstance(artifact, SealedTransferPredictionArtifact)
        self.assertFalse(hasattr(artifact, "to_payload"))
        with self.assertRaises(TypeError):
            pickle.dumps(artifact)

    def test_seed_commitment_and_artifact_identity_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "FINAL seeds"):
            self._execute(seeds=(301,))
        with self.assertRaisesRegex(ValueError, "commitment"):
            self._execute(expected_final_commitment_hash=digest("wrong-final"))
        with self.assertRaisesRegex(ValueError, "artifact identity"):
            self._execute(prediction_artifact_identity=digest("wrong-artifact-schema"))

    def test_missing_duplicate_or_wrong_path_candidate_is_rejected(self) -> None:
        refit = refit_freeze()
        mutations = (
            refit.selected_candidates[:-1],
            refit.selected_candidates + (refit.selected_candidates[0],),
            (zero_shot_freeze().candidates[0],) + refit.selected_candidates[1:],
        )
        for index, selected in enumerate(mutations):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, "exact six candidates"):
                    self._execute(refit=replace(refit, selected_candidates=selected))

    def test_projection_is_single_use(self) -> None:
        calls: list[tuple[object, ...]] = []
        projection = final_worker_projection(1)
        kwargs = {
            "projection": projection,
            "zero_shot": zero_shot_freeze(),
            "refit": refit_freeze(),
            "seeds": (301, 302),
            "evaluator": recording_evaluator(calls),
            "progress": recording_progress(calls),
            "expected_final_commitment_hash": projection.commitment_hash,
            "prediction_artifact_identity": sibling_releases()[0].prediction_artifact_identity,
        }
        execute_transfer_final_predictions(**kwargs)
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            execute_transfer_final_predictions(**kwargs)

    def test_evaluator_exception_propagates_after_prior_progress_is_persisted(self) -> None:
        calls: list[tuple[object, ...]] = []
        projection = final_worker_projection(1)

        class FailOnSecond:
            def __call__(self, model_input, candidate, seed):
                calls.append((model_input.case_token, candidate.path, seed))
                if len(calls) == 2:
                    raise RuntimeError("synthetic evaluator failure")
                return (0.4, 0.6)

        progress = recording_progress(calls)
        with self.assertRaisesRegex(RuntimeError, "synthetic evaluator failure"):
            execute_transfer_final_predictions(
                projection=projection,
                zero_shot=zero_shot_freeze(),
                refit=refit_freeze(),
                seeds=(301, 302),
                evaluator=FailOnSecond(),
                progress=progress,
                expected_final_commitment_hash=projection.commitment_hash,
                prediction_artifact_identity=sibling_releases()[0].prediction_artifact_identity,
            )
        self.assertEqual(len(progress.rows), 1)
        self.assertEqual(progress.rows[0].completed_model_runs, 1)

    def test_evaluator_receives_no_target_or_postchoice_fields(self) -> None:
        calls: list[tuple[object, ...]] = []
        evaluator = recording_evaluator(calls)
        self._execute(evaluator=evaluator, progress=recording_progress(calls))
        forbidden = {
            "first_stage_action",
            "transition_common",
            "final_state",
            "second_stage_action",
            "reward",
            "target",
        }
        for model_input in evaluator.inputs:
            self.assertTrue(forbidden.isdisjoint(model_input.__dataclass_fields__))

    def test_nonfinite_or_incomplete_probabilities_fail_closed(self) -> None:
        invalid = (
            (math.nan, 0.0),
            (math.inf, 0.0),
            (0.2, 0.2),
            (0.4,),
            (-0.1, 1.1),
        )
        for probabilities in invalid:
            with self.subTest(probabilities=probabilities):
                calls: list[tuple[object, ...]] = []
                with self.assertRaisesRegex(ValueError, "probabilities"):
                    self._execute(
                        evaluator=recording_evaluator(calls, invalid=probabilities),
                        progress=recording_progress(calls),
                    )

    def test_one_artifact_identity_is_shared_by_both_score_siblings(self) -> None:
        artifact, _calls = self._execute()
        brier, log = sibling_releases()
        self.assertEqual(
            artifact.prediction_artifact_identity,
            brier.prediction_artifact_identity,
        )
        self.assertEqual(
            artifact.prediction_artifact_identity,
            log.prediction_artifact_identity,
        )

    def test_private_rows_preserve_participant_grouping_without_evaluator_leakage(self) -> None:
        rows = tuple(
            CanonicalTransferTrial(
                participant_key=participant,
                trial_id=trial_id,
                source_stratum="s0",
                first_stage_action="action_1",
                transition_common=True,
                final_state="state_0",
                second_stage_action="second_0",
                reward=1,
                row_commitment=digest(f"{participant}-{trial_id}"),
            )
            for participant, trial_id in (("private-p1", 1), ("private-p1", 2), ("private-p2", 1))
        )
        projection = FinalWorkerProjection(
            rows=rows,
            commitment_hash=digest("grouped-final"),
        )
        calls: list[tuple[object, ...]] = []
        evaluator = recording_evaluator(calls)
        artifact = execute_transfer_final_predictions(
            projection=projection,
            zero_shot=zero_shot_freeze(),
            refit=refit_freeze(),
            seeds=(301, 302),
            evaluator=evaluator,
            progress=recording_progress(calls),
            expected_final_commitment_hash=projection.commitment_hash,
            prediction_artifact_identity=sibling_releases()[0].prediction_artifact_identity,
        )
        private_rows = artifact.consume_for_scoring_once()
        groups_by_case = {
            row.case_token: row.participant_token for row in private_rows
        }
        self.assertEqual(len(groups_by_case), 3)
        self.assertEqual(len(set(groups_by_case.values())), 2)
        self.assertTrue(
            all("participant_token" not in item.__dataclass_fields__ for item in evaluator.inputs)
        )


if __name__ == "__main__":
    unittest.main()
