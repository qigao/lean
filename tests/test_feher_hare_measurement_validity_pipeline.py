from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import narrative_dynamics
from narrative_dynamics.adapters.two_stage_metrics import (
    two_stage_brier_loss,
    two_stage_log_loss,
)
from narrative_dynamics.candidate_execution import (
    CandidateExecutionError,
    SequentialCandidateExecutor,
)
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.manifest import callable_identity
from narrative_dynamics.losses import metric_loss_identity
from narrative_dynamics.measurement_validity import (
    ExactInvarianceFinding,
    MeasurementAuditAttempt,
    MeasurementTerminalClass,
    MeasurementValidityStatus,
)
from narrative_dynamics.report_artifact import AggregateReportArtifact
import narrative_dynamics.studies as studies
import narrative_dynamics.studies.feher_hare_measurement_validity_v1 as measurement_study
from narrative_dynamics.studies.feher_hare_measurement_validity_v1 import (
    FEHER_HARE_R3_LOCK_COMMIT,
    FeherHareMeasurementValidityResult,
    build_feher_hare_measurement_protocol,
    freeze_feher_hare_measurement_candidates,
    load_feher_hare_r3_measurement_anchor,
    run_feher_hare_measurement_validity_v1,
)
from narrative_dynamics.studies.two_stage_source import TwoStageSourceManifest

from tests.test_feher_hare_measurement_prediction import _fixture
from tests.test_feher_hare_measurement_validity_input import (
    _lock_payload,
    _write_lock,
)


def _finding(name: str, *, failed: bool = False) -> ExactInvarianceFinding:
    original = stable_content_hash(("pipeline-invariance", name, "expected"))
    transformed = (
        stable_content_hash(("pipeline-invariance", name, "failed"))
        if failed
        else original
    )
    return ExactInvarianceFinding(
        check_name=name,
        status=(
            MeasurementValidityStatus.EXACT_INVARIANCE_FAILED
            if failed
            else MeasurementValidityStatus.EXACT_INVARIANCE_MET
        ),
        original_hash=original,
        transformed_hash=transformed,
        details_hash=stable_content_hash(("pipeline-invariance-details", name)),
    )


def _semantic_findings(*, failed: bool = False):
    return (
        _finding("task_canonicalization", failed=failed),
        _finding("semantic_executor_invariance"),
    )


def _empirical_findings(*, failed: bool = False):
    return (
        _finding("categorical_coordinate_invariance", failed=failed),
        _finding("record_order_batch_invariance"),
    )


class FailingExecutor:
    def execute(self, function, tasks, *, initializer=None, initargs=()):
        raise CandidateExecutionError("injected pipeline executor failure")


class FeherHareMeasurementValidityPipelineTests(unittest.TestCase):
    def _context(self, root: Path):
        lock_path = _write_lock(root, _lock_payload())
        anchor = load_feher_hare_r3_measurement_anchor(
            lock_path,
            lock_commit=FEHER_HARE_R3_LOCK_COMMIT,
        )
        identity, base_input, _base_protocol = _fixture()
        candidates = freeze_feher_hare_measurement_candidates(anchor)
        audit_input = replace(
            base_input,
            source_manifest_hash=anchor.source_manifest_hash,
            source_snapshot_hash=anchor.source_snapshot_hash,
            transform_hash=anchor.transform_hash,
            participant_assignment_hash=anchor.participant_assignment_hash,
            dataset_hash=anchor.dataset_hash,
            target_spec_hash=anchor.target_spec_hash,
            allowed_partition_hashes=(
                ("train", anchor.train_partition_hash),
                ("selection_validation", anchor.selection_partition_hash),
            ),
            frozen_candidates=candidates,
            excluded_final_partition_hash=anchor.excluded_final_partition_hash,
            excluded_final_target_hash=anchor.excluded_final_target_hash,
        )
        manifest = TwoStageSourceManifest(
            name="synthetic-measurement-pipeline",
            version="1",
            repository="test/synthetic-measurement-pipeline",
            revision=anchor.upstream_revision,
            license_reference="test-only",
            files=(),
        )
        source_root = root / "source-root-is-owned-by-provisioner"
        return source_root, manifest, lock_path, anchor, identity, audit_input

    def test_complete_pipeline_is_deterministic_and_uses_no_fit_or_final_api(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root, manifest, lock_path, anchor, identity, audit_input = (
                self._context(root)
            )
            reordered_input = replace(
                audit_input,
                cases=tuple(reversed(audit_input.cases)),
            )
            provisioned_inputs = iter((audit_input, reordered_input))
            provision_calls = []

            def provision(source, selected_manifest, selected_anchor):
                provision_calls.append(
                    (source, selected_manifest, selected_anchor)
                )
                return next(provisioned_inputs)

            forbidden = (
                "narrative_dynamics.calibration._grid_candidates",
                "narrative_dynamics.observations.training.fit_training_target_grid",
                "narrative_dynamics.validation.select_on_validation_suite",
                "narrative_dynamics.uncertainty.ParameterAcceptanceSet.from_parameters",
                "narrative_dynamics.observations.preregistration.FrozenModelSpec.from_selection",
                "narrative_dynamics.external_prediction.predict_external_final_once",
            )
            with ExitStack() as stack:
                forbidden_mocks = tuple(
                    stack.enter_context(
                        patch(path, side_effect=AssertionError(f"forbidden call: {path}"))
                    )
                    for path in forbidden
                )
                stack.enter_context(
                    patch.object(
                        measurement_study,
                        "provision_feher_hare_measurement_input",
                        new=provision,
                    )
                )
                first = run_feher_hare_measurement_validity_v1(
                    root=source_root,
                    manifest=manifest,
                    lock_path=lock_path,
                    lock_commit=FEHER_HARE_R3_LOCK_COMMIT,
                    repository_identity=identity,
                    executor=SequentialCandidateExecutor(),
                )
                second = run_feher_hare_measurement_validity_v1(
                    root=source_root,
                    manifest=manifest,
                    lock_path=lock_path,
                    lock_commit=FEHER_HARE_R3_LOCK_COMMIT,
                    repository_identity=identity,
                    executor=SequentialCandidateExecutor(),
                )
            for mocked in forbidden_mocks:
                mocked.assert_not_called()
            self.assertEqual(len(provision_calls), 2)
            self.assertTrue(
                all(call[0] == source_root for call in provision_calls)
            )

        self.assertIsInstance(first, FeherHareMeasurementValidityResult)
        self.assertEqual(first, second)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertIs(first.report.terminal_class, MeasurementTerminalClass.GREEN)
        self.assertEqual(first.audit_input_hash, audit_input.content_hash)
        self.assertEqual(first.prediction_hash, first.report.prediction_artifact_hash)
        self.assertIsInstance(first.artifact, AggregateReportArtifact)
        self.assertEqual(first.artifact, first.attestation.artifact)
        self.assertIs(first.attestation.require_integrity(), first.report)
        self.assertFalse(first.report.parameter_training_performed)
        self.assertFalse(first.report.parameter_selection_performed)
        self.assertFalse(first.report.final_test_values_exposed_to_audit)
        self.assertFalse(first.report.final_test_outcomes_analyzed)
        self.assertFalse(first.report.final_model_execution)

    def test_semantic_failure_stops_before_prediction_and_robustness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root, manifest, lock_path, anchor, identity, audit_input = (
                self._context(root)
            )
            forbidden_calls = {"prediction": 0, "robustness": 0}

            def provision(source, selected_manifest, selected_anchor):
                return audit_input

            def semantic(repository_identity, candidates):
                return _semantic_findings(failed=True)

            def forbidden_prediction(*args, **kwargs):
                forbidden_calls["prediction"] += 1
                raise AssertionError("prediction must not run")

            def forbidden_robustness(*args, **kwargs):
                forbidden_calls["robustness"] += 1
                raise AssertionError("robustness must not run")

            with (
                patch.object(
                    measurement_study,
                    "provision_feher_hare_measurement_input",
                    new=provision,
                ),
                patch.object(
                    measurement_study,
                    "evaluate_feher_hare_semantic_invariance",
                    new=semantic,
                ),
                patch.object(
                    measurement_study,
                    "execute_feher_hare_measurement_predictions",
                    new=forbidden_prediction,
                ),
                patch.object(
                    measurement_study,
                    "build_measurement_robustness_profile",
                    new=forbidden_robustness,
                ),
            ):
                result = run_feher_hare_measurement_validity_v1(
                    root=source_root,
                    manifest=manifest,
                    lock_path=lock_path,
                    lock_commit=FEHER_HARE_R3_LOCK_COMMIT,
                    repository_identity=identity,
                    executor=SequentialCandidateExecutor(),
                )
        self.assertEqual(forbidden_calls, {"prediction": 0, "robustness": 0})
        self.assertIs(
            result.report.terminal_class,
            MeasurementTerminalClass.SCIENTIFIC_RED,
        )
        self.assertIsNone(result.prediction_hash)
        self.assertIsNone(result.report.robustness_profile)
        self.assertIs(result.attestation.require_integrity(), result.report)

    def test_empirical_failure_binds_prediction_and_stops_final_robustness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root, manifest, lock_path, _anchor, identity, audit_input = (
                self._context(root)
            )
            forbidden_calls = {"score": 0, "robustness": 0}

            def provision(source, selected_manifest, selected_anchor):
                return audit_input

            def semantic(repository_identity, candidates):
                return _semantic_findings()

            def empirical(selected_input, prediction, selected_protocol):
                return _empirical_findings(failed=True)

            def forbidden_score(*args, **kwargs):
                forbidden_calls["score"] += 1
                raise AssertionError("final scoring must not run")

            def forbidden_robustness(*args, **kwargs):
                forbidden_calls["robustness"] += 1
                raise AssertionError("robustness must not run")

            with (
                patch.object(
                    measurement_study,
                    "provision_feher_hare_measurement_input",
                    new=provision,
                ),
                patch.object(
                    measurement_study,
                    "evaluate_feher_hare_semantic_invariance",
                    new=semantic,
                ),
                patch.object(
                    measurement_study,
                    "evaluate_feher_hare_empirical_invariance",
                    new=empirical,
                ),
                patch.object(
                    measurement_study,
                    "score_measurement_predictions",
                    new=forbidden_score,
                ),
                patch.object(
                    measurement_study,
                    "build_measurement_robustness_profile",
                    new=forbidden_robustness,
                ),
            ):
                result = run_feher_hare_measurement_validity_v1(
                    root=source_root,
                    manifest=manifest,
                    lock_path=lock_path,
                    lock_commit=FEHER_HARE_R3_LOCK_COMMIT,
                    repository_identity=identity,
                    executor=SequentialCandidateExecutor(),
                )
        self.assertEqual(forbidden_calls, {"score": 0, "robustness": 0})
        self.assertIs(
            result.report.terminal_class,
            MeasurementTerminalClass.SCIENTIFIC_RED,
        )
        self.assertIsNotNone(result.prediction_hash)
        self.assertEqual(
            result.prediction_hash,
            result.report.prediction_artifact_hash,
        )
        self.assertIsNone(result.report.robustness_profile)

    def test_executor_exception_propagates_for_infrastructure_attempt_recording(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root, manifest, lock_path, anchor, identity, audit_input = (
                self._context(root)
            )
            protocol_hash = build_feher_hare_measurement_protocol(anchor).content_hash
            assemble_calls = []

            def provision(source, selected_manifest, selected_anchor):
                return audit_input

            def semantic(repository_identity, candidates):
                return _semantic_findings()

            def forbidden_assemble(*args, **kwargs):
                assemble_calls.append((args, kwargs))
                raise AssertionError("scientific report must not be assembled")

            with (
                patch.object(
                    measurement_study,
                    "provision_feher_hare_measurement_input",
                    new=provision,
                ),
                patch.object(
                    measurement_study,
                    "evaluate_feher_hare_semantic_invariance",
                    new=semantic,
                ),
                patch.object(
                    measurement_study,
                    "assemble_feher_hare_measurement_validity_report",
                    new=forbidden_assemble,
                ),
            ):
                with self.assertRaisesRegex(
                    CandidateExecutionError,
                    "injected pipeline executor failure",
                ) as raised:
                    run_feher_hare_measurement_validity_v1(
                        root=source_root,
                        manifest=manifest,
                        lock_path=lock_path,
                        lock_commit=FEHER_HARE_R3_LOCK_COMMIT,
                        repository_identity=identity,
                        executor=FailingExecutor(),
                    )
        self.assertEqual(assemble_calls, [])
        attempt = MeasurementAuditAttempt(
            attempt_id="pipeline-infrastructure-attempt",
            scientific_revision="a" * 40,
            orchestration_revision="b" * 40,
            started_at_utc="2026-08-30T06:00:00Z",
            terminal_class=MeasurementTerminalClass.INFRASTRUCTURE_INCOMPLETE,
            protocol_hash=protocol_hash,
            report_hash=None,
            error_type=type(raised.exception).__name__,
            error_message_hash=stable_content_hash(str(raised.exception)),
            artifact_file_hashes=(),
        )
        self.assertIs(
            attempt.terminal_class,
            MeasurementTerminalClass.INFRASTRUCTURE_INCOMPLETE,
        )
        self.assertIsNone(attempt.report_hash)

    def test_protocol_freezes_every_invoked_callable_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _source, _manifest, _lock, anchor, _identity, audit_input = (
                self._context(root)
            )
        protocol = build_feher_hare_measurement_protocol(anchor)
        expected = {
            "anchor_loader": callable_identity(
                measurement_study.load_feher_hare_r3_measurement_anchor
            ),
            "provisioner": callable_identity(
                measurement_study.provision_feher_hare_measurement_input
            ),
            "candidate_freezer": callable_identity(
                measurement_study.freeze_feher_hare_measurement_candidates
            ),
            "semantic_evaluator": callable_identity(
                measurement_study.evaluate_feher_hare_semantic_invariance
            ),
            "prediction_executor": callable_identity(
                measurement_study.execute_feher_hare_measurement_predictions
            ),
            "empirical_invariance": callable_identity(
                measurement_study.evaluate_feher_hare_empirical_invariance
            ),
            "invariance_combiner": callable_identity(
                measurement_study.combine_feher_hare_exact_invariance
            ),
            "prediction_scorer": callable_identity(
                measurement_study.score_measurement_predictions
            ),
            "robustness_builder": callable_identity(
                measurement_study.build_measurement_robustness_profile
            ),
            "diagnostic_builder": callable_identity(
                measurement_study.build_feher_hare_stay_switch_diagnostics
            ),
            "report_assembler": callable_identity(
                measurement_study.assemble_feher_hare_measurement_validity_report
            ),
            "report_attestor": callable_identity(measurement_study.attest_report),
            "brier_loss": metric_loss_identity(two_stage_brier_loss()),
            "log_loss": metric_loss_identity(two_stage_log_loss()),
        }
        self.assertEqual(dict(protocol.implementation_identities), expected)
        self.assertEqual(protocol.empirical_anchor_hash, audit_input.empirical_anchor_hash)
        self.assertEqual(
            protocol.semantic_fixture_hashes,
            tuple(
                sorted(
                    fixture.content_hash
                    for fixture in measurement_study.frozen_feher_hare_semantic_fixtures()
                )
            ),
        )
        self.assertEqual(
            protocol.build_manifest(audit_input).parent_hashes,
            (audit_input.content_hash,),
        )

    def test_studies_exports_runner_but_package_root_exports_only_contracts(self) -> None:
        self.assertIs(
            studies.run_feher_hare_measurement_validity_v1,
            run_feher_hare_measurement_validity_v1,
        )
        self.assertIn("run_feher_hare_measurement_validity_v1", studies.__all__)
        self.assertFalse(
            hasattr(narrative_dynamics, "run_feher_hare_measurement_validity_v1")
        )
        self.assertTrue(hasattr(narrative_dynamics, "MeasurementValidityReport"))
        self.assertTrue(hasattr(narrative_dynamics, "MeasurementAuditAttempt"))


if __name__ == "__main__":
    unittest.main()
