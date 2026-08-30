from __future__ import annotations

from dataclasses import replace
import json
import math
import unittest

from narrative_dynamics.adapters.two_stage_metrics import (
    two_stage_brier_loss,
    two_stage_log_loss,
)
from narrative_dynamics.contracts import ExperimentStage, stable_content_hash
from narrative_dynamics.external_prediction import ExternalFinalPredictionArtifact
from narrative_dynamics.measurement_validity import (
    MeasurementCaseLoss,
    MeasurementModelPrediction,
    MeasurementPredictionArtifact,
    MeasurementScore,
    MeasurementSeedPrediction,
    average_seed_metrics,
    score_measurement_predictions,
)
from narrative_dynamics.observations import ObservationPartitionRole

from tests.measurement_validity_fixtures import (
    digest,
    measurement_input,
    measurement_protocol,
)


def _run_hash(model_name: str, case_hash: str, seed: int) -> str:
    return stable_content_hash(
        {
            "fixture": "measurement-prediction-run",
            "model_name": model_name,
            "case_hash": case_hash,
            "seed": seed,
        }
    )


def _metrics(seed_index: int) -> tuple[tuple[str, float], ...]:
    action_0 = (0.7, 0.9)[seed_index]
    return (
        ("first_stage.action_0", action_0),
        ("first_stage.action_1", 1.0 - action_0),
    )


def _model_prediction(
    *,
    audit_input,
    protocol,
    model_name: str,
    rows_transform=lambda rows: rows,
) -> MeasurementModelPrediction:
    candidate = next(
        item for item in audit_input.frozen_candidates if item.name == model_name
    )
    seeds_by_role = dict(protocol.seeds_by_role)
    rows = tuple(
        MeasurementSeedPrediction(
            case_hash=case.case_hash,
            scenario_hash=case.scenario.content_hash,
            role=case.role,
            model_name=model_name,
            seed=seed,
            metrics=_metrics(seed_index),
            run_manifest_hash=_run_hash(model_name, case.case_hash, seed),
        )
        for case in audit_input.cases
        for seed_index, seed in enumerate(seeds_by_role[case.role])
    )
    return MeasurementModelPrediction(
        model_name=model_name,
        candidate_hash=candidate.content_hash,
        rows=rows_transform(rows),
    )


def _artifact(*, models_transform=lambda models: models) -> tuple[object, object, MeasurementPredictionArtifact]:
    audit_input = measurement_input()
    protocol = measurement_protocol(audit_input)
    models = tuple(
        _model_prediction(
            audit_input=audit_input,
            protocol=protocol,
            model_name=model_name,
        )
        for model_name in ("reactive", "intentional", "planning")
    )
    transformed_models = models_transform(models)
    manifests = tuple(
        row.run_manifest_hash
        for model in transformed_models
        for row in model.rows
    )
    return (
        audit_input,
        protocol,
        MeasurementPredictionArtifact(
            protocol_hash=protocol.content_hash,
            audit_input_hash=audit_input.content_hash,
            models=transformed_models,
            execution_manifest_hashes=tuple(reversed(manifests)),
        ),
    )


class MeasurementPredictionTests(unittest.TestCase):
    def test_average_seed_metrics_precedes_loss_scoring(self) -> None:
        seed_metrics = (
            {
                "first_stage.action_0": 0.7,
                "first_stage.action_1": 0.3,
            },
            {
                "first_stage.action_0": 0.9,
                "first_stage.action_1": 0.1,
            },
        )
        self.assertEqual(
            average_seed_metrics(seed_metrics),
            (
                ("first_stage.action_0", 0.8),
                ("first_stage.action_1", 0.2),
            ),
        )

        audit_input, _protocol, artifact = _artifact()
        scored = score_measurement_predictions(
            audit_input,
            artifact,
            (two_stage_log_loss(), two_stage_brier_loss()),
        )
        action_0_case = next(
            case
            for case in audit_input.cases
            if dict(case.target)["first_stage.action_0"] == 1.0
        )
        by_score = {
            row.score: row
            for row in scored
            if row.case_hash == action_0_case.case_hash
            and row.model_name == "reactive"
        }
        self.assertEqual(set(by_score), set(MeasurementScore))
        self.assertEqual(by_score[MeasurementScore.BRIER].value, 0.07999999999999999)
        self.assertEqual(by_score[MeasurementScore.LOG].value, -math.log(0.8))
        self.assertTrue(all(isinstance(row, MeasurementCaseLoss) for row in scored))
        self.assertEqual(
            len(scored),
            len(audit_input.cases)
            * len(audit_input.frozen_candidates)
            * len(MeasurementScore),
        )

    def test_prediction_rows_and_artifact_are_canonical_and_sealed(self) -> None:
        audit_input, protocol, artifact = _artifact(
            models_transform=lambda models: tuple(reversed(models))
        )
        independently_reversed = tuple(
            MeasurementModelPrediction(
                model_name=model.model_name,
                candidate_hash=model.candidate_hash,
                rows=tuple(reversed(model.rows)),
            )
            for model in reversed(artifact.models)
        )
        canonical = MeasurementPredictionArtifact(
            protocol_hash=protocol.content_hash,
            audit_input_hash=audit_input.content_hash,
            models=independently_reversed,
            execution_manifest_hashes=tuple(
                reversed(artifact.execution_manifest_hashes)
            ),
        )

        self.assertEqual(
            tuple(model.model_name for model in artifact.models),
            ("reactive", "intentional", "planning"),
        )
        self.assertEqual(artifact.content_hash, canonical.content_hash)
        self.assertEqual(artifact.models, canonical.models)
        self.assertEqual(
            len(artifact.execution_manifest_hashes),
            len(audit_input.cases) * 2 * 3,
        )
        for model in artifact.models:
            self.assertEqual(len(model.rows), len(audit_input.cases) * 2)
            self.assertEqual(
                len({(row.case_hash, row.seed) for row in model.rows}),
                len(model.rows),
            )
        self.assertEqual(
            len(
                {
                    (row.model_name, row.case_hash, row.seed)
                    for model in artifact.models
                    for row in model.rows
                }
            ),
            len(artifact.execution_manifest_hashes),
        )

    def test_artifact_is_measurement_only_and_excludes_targets_and_identity(self) -> None:
        _audit_input, _protocol, artifact = _artifact()
        self.assertNotIsInstance(artifact, ExternalFinalPredictionArtifact)
        self.assertIs(artifact.manifest.stage, ExperimentStage.MEASUREMENT_AUDIT)
        self.assertIsNot(artifact.manifest.stage, ExperimentStage.FINAL_TEST)

        encoded = json.dumps(artifact.identity_payload(), sort_keys=True)
        self.assertNotIn('"target"', encoded)
        self.assertNotIn("participant_id", encoded)
        self.assertNotIn("source_participant_id", encoded)
        self.assertNotIn(ObservationPartitionRole.FINAL_TEST.value, encoded)

    def test_score_rejects_missing_or_extra_case_and_seed_coverage(self) -> None:
        audit_input, protocol, artifact = _artifact()

        def rebuild(models):
            manifests = tuple(
                row.run_manifest_hash for model in models for row in model.rows
            )
            return MeasurementPredictionArtifact(
                protocol_hash=protocol.content_hash,
                audit_input_hash=audit_input.content_hash,
                models=models,
                execution_manifest_hashes=manifests,
            )

        missing_case_hash = audit_input.cases[0].case_hash
        missing_case_models = tuple(
            replace(
                model,
                rows=tuple(
                    row for row in model.rows if row.case_hash != missing_case_hash
                ),
            )
            for model in artifact.models
        )

        missing_seed_models = tuple(
            replace(
                model,
                rows=tuple(
                    row
                    for row in model.rows
                    if not (
                        row.case_hash == audit_input.cases[0].case_hash
                        and row.seed
                        == dict(protocol.seeds_by_role)[audit_input.cases[0].role][0]
                    )
                ),
            )
            for model in artifact.models
        )

        extra_case_hash = stable_content_hash("extra-measurement-case")
        extra_scenario_hash = stable_content_hash("extra-measurement-scenario")
        extra_case_models = tuple(
            replace(
                model,
                rows=model.rows
                + tuple(
                    MeasurementSeedPrediction(
                        case_hash=extra_case_hash,
                        scenario_hash=extra_scenario_hash,
                        role=ObservationPartitionRole.TRAIN,
                        model_name=model.model_name,
                        seed=seed,
                        metrics=_metrics(index),
                        run_manifest_hash=_run_hash(
                            model.model_name,
                            extra_case_hash,
                            seed,
                        ),
                    )
                    for index, seed in enumerate(
                        dict(protocol.seeds_by_role)[ObservationPartitionRole.TRAIN]
                    )
                ),
            )
            for model in artifact.models
        )

        extra_seed = 103
        extra_seed_models = tuple(
            replace(
                model,
                rows=model.rows
                + (
                    MeasurementSeedPrediction(
                        case_hash=audit_input.cases[0].case_hash,
                        scenario_hash=audit_input.cases[0].scenario.content_hash,
                        role=audit_input.cases[0].role,
                        model_name=model.model_name,
                        seed=extra_seed,
                        metrics=_metrics(0),
                        run_manifest_hash=_run_hash(
                            model.model_name,
                            audit_input.cases[0].case_hash,
                            extra_seed,
                        ),
                    ),
                ),
            )
            for model in artifact.models
        )

        for models in (
            missing_case_models,
            missing_seed_models,
            extra_case_models,
            extra_seed_models,
        ):
            with self.subTest(row_count=sum(len(model.rows) for model in models)):
                with self.assertRaisesRegex(ValueError, "coverage"):
                    score_measurement_predictions(
                        audit_input,
                        rebuild(models),
                        (two_stage_brier_loss(), two_stage_log_loss()),
                    )

    def test_score_rejects_model_candidate_and_binding_drift(self) -> None:
        audit_input, _protocol, artifact = _artifact()
        alien = replace(
            artifact.models[0],
            model_name="alien",
            rows=tuple(
                replace(row, model_name="alien")
                for row in artifact.models[0].rows
            ),
        )
        alien_models = (alien,) + artifact.models[1:]
        alien_artifact = MeasurementPredictionArtifact(
            protocol_hash=artifact.protocol_hash,
            audit_input_hash=artifact.audit_input_hash,
            models=alien_models,
            execution_manifest_hashes=tuple(
                row.run_manifest_hash for model in alien_models for row in model.rows
            ),
        )
        wrong_candidate_models = (
            replace(artifact.models[0], candidate_hash=digest("e")),
        ) + artifact.models[1:]
        wrong_candidate_artifact = MeasurementPredictionArtifact(
            protocol_hash=artifact.protocol_hash,
            audit_input_hash=artifact.audit_input_hash,
            models=wrong_candidate_models,
            execution_manifest_hashes=artifact.execution_manifest_hashes,
        )
        changed_input = replace(audit_input, dataset_hash=digest("f"))

        for candidate in (alien_artifact, wrong_candidate_artifact):
            with self.subTest(candidate=candidate.content_hash):
                with self.assertRaisesRegex(ValueError, "candidate"):
                    score_measurement_predictions(
                        audit_input,
                        candidate,
                        (two_stage_brier_loss(), two_stage_log_loss()),
                    )
        with self.assertRaisesRegex(ValueError, "audit input"):
            score_measurement_predictions(
                changed_input,
                artifact,
                (two_stage_brier_loss(), two_stage_log_loss()),
            )
        with self.assertRaisesRegex(ValueError, "protocol"):
            replace(artifact, protocol_hash=digest("0"))

    def test_contracts_reject_duplicate_manifests_and_nonfinite_metrics(self) -> None:
        _audit_input, _protocol, artifact = _artifact()
        row = artifact.models[0].rows[0]
        with self.assertRaises((TypeError, ValueError)):
            replace(
                row,
                metrics=(
                    ("first_stage.action_0", math.nan),
                    ("first_stage.action_1", 0.0),
                ),
            )
        with self.assertRaisesRegex(ValueError, "case/seed"):
            replace(
                artifact.models[0],
                rows=artifact.models[0].rows + (artifact.models[0].rows[0],),
            )
        with self.assertRaisesRegex(ValueError, "execution manifest"):
            replace(
                artifact,
                execution_manifest_hashes=artifact.execution_manifest_hashes
                + (artifact.execution_manifest_hashes[0],),
            )
        duplicate_run_models = (
            replace(
                artifact.models[0],
                rows=(
                    artifact.models[0].rows[0],
                    replace(
                        artifact.models[0].rows[1],
                        run_manifest_hash=artifact.models[0].rows[0].run_manifest_hash,
                    ),
                )
                + artifact.models[0].rows[2:],
            ),
        ) + artifact.models[1:]
        with self.assertRaisesRegex(ValueError, "run manifest"):
            MeasurementPredictionArtifact(
                protocol_hash=artifact.protocol_hash,
                audit_input_hash=artifact.audit_input_hash,
                models=duplicate_run_models,
                execution_manifest_hashes=artifact.execution_manifest_hashes,
            )


if __name__ == "__main__":
    unittest.main()
