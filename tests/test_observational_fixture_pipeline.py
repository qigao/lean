from __future__ import annotations

from pathlib import Path
import unittest

import narrative_dynamics
import narrative_dynamics.observations as observations
from narrative_dynamics.adapters.prison_metrics import prison_initial_action_metrics
from narrative_dynamics.adapters.prison_pomdp import create_prison_pomdp_model
from narrative_dynamics.adapters.prison_reactive import create_prison_reactive_model
from narrative_dynamics.losses import CategoricalBrierLoss, CategoricalMetricGroup
from narrative_dynamics.model_comparison import ComparisonModel
from narrative_dynamics.observations import (
    AdequacyThresholds,
    CategoricalTargetSpec,
    FrozenModelCandidate,
    ObservationPartitionRole,
    PreregisteredEvaluationProtocol,
    construct_categorical_targets,
    load_observation_dataset,
)
from narrative_dynamics.report_artifact import attest_report
from narrative_dynamics.simulation import ModelFactory, SimulationRunner
from narrative_dynamics.uncertainty import ParameterAcceptanceSet
from narrative_dynamics.validation import (
    EvaluationRole,
    HeldOutCase,
    HeldOutSuite,
    select_on_validation_suite,
)


FIXTURE = Path("fixtures/observations/prison_initial_choice_v1.json")
_GRID = {"beta": (0.5, 1.0, 2.0, 4.0)}
_TRAIN_SEEDS = (101, 102)
_SELECTION_SEEDS = (201, 202)
_FINAL_SEEDS = (301, 302)


class FixtureWitnessVerifier:
    name = "fixture-witness-verifier"
    version = "1"

    def verify(self, release, receipt):
        return (
            receipt.provider == "test-fixture"
            and receipt.subject_hash == release.content_hash
            and dict(receipt.proof) == {"nonce": "release-v1"}
        )


def changed_prison_initial_action_metrics(trace):
    return prison_initial_action_metrics(trace)


changed_prison_initial_action_metrics.version = "2.0.0"


class PrisonObservationFixtureTests(unittest.TestCase):
    def test_committed_fixture_is_synthetic_versioned_and_role_complete(self):
        self.assertTrue(FIXTURE.is_file(), "versioned synthetic prison fixture is missing")
        dataset = load_observation_dataset(FIXTURE)

        self.assertEqual(dataset.name, "prison-initial-choice")
        self.assertEqual(dataset.version, "1.0.0")
        self.assertEqual(dataset.source["kind"], "synthetic_fixture")
        self.assertEqual(dataset.source["purpose"], "protocol_integration_test")
        self.assertFalse(dataset.provenance["empirical_human_data"])
        self.assertFalse(dataset.provenance["population_representative"])
        self.assertEqual(
            {partition.role for partition in dataset.partitions},
            set(ObservationPartitionRole),
        )
        for partition in dataset.partitions:
            self.assertTrue(partition.records)
            for record in partition.records:
                self.assertEqual(
                    set(record.counts),
                    {"scout", "escape", "submit"},
                )

    def test_committed_counts_are_fixed_and_persistence_pair_is_discriminating(self):
        self.assertTrue(FIXTURE.is_file(), "versioned synthetic prison fixture is missing")
        dataset = load_observation_dataset(FIXTURE)
        final_partition = dataset.partition(ObservationPartitionRole.FINAL_TEST)
        final_by_id = {record.id: record for record in final_partition.records}

        self.assertEqual(
            final_by_id["final-1"].count_map,
            {"scout": 18, "escape": 66, "submit": 16},
        )
        self.assertEqual(
            final_by_id["final-2"].count_map,
            {"scout": 43, "escape": 45, "submit": 12},
        )
        first = dict(final_by_id["final-1"].scenario.payload)
        second = dict(final_by_id["final-2"].scenario.payload)
        self.assertEqual(
            {key: value for key, value in first.items() if key != "guard_persistence"},
            {key: value for key, value in second.items() if key != "guard_persistence"},
        )
        self.assertEqual(first["guard_persistence"], 0.2)
        self.assertEqual(second["guard_persistence"], 0.9)


class ReleasedPrisonPipelineTests(unittest.TestCase):
    def _public_pipeline_api(self):
        required = (
            "TrainingCaseFit",
            "TrainingCandidateFit",
            "TrainingFitReport",
            "fit_training_target_grid",
            "ProtocolRelease",
            "WitnessReceipt",
            "VerifiedProtocolRelease",
            "ProtocolReleaseVerificationError",
            "ReleasedModelComparisonReport",
            "verify_protocol_release",
            "compare_released_models",
        )
        missing_observations = tuple(
            name for name in required if not hasattr(observations, name)
        )
        missing_root = tuple(
            name for name in required if not hasattr(narrative_dynamics, name)
        )
        self.assertEqual(missing_observations, (), "observational public API is incomplete")
        self.assertEqual(missing_root, (), "package-root public API is incomplete")
        return {name: getattr(observations, name) for name in required}

    def _prepare_release(self):
        api = self._public_pipeline_api()
        dataset = load_observation_dataset(FIXTURE)
        spec = CategoricalTargetSpec(
            name="prison-initial-choice-target",
            version="1",
            categories=("scout", "escape", "submit"),
            metric_prefix="initial",
        )
        loss = CategoricalBrierLoss(
            (
                CategoricalMetricGroup(
                    "initial-action",
                    ("initial.scout", "initial.escape", "initial.submit"),
                ),
            )
        )
        factory_calls = {"finite-prison-pomdp": 0, "finite-prison-reactive": 0}

        def create_baseline():
            factory_calls["finite-prison-pomdp"] += 1
            return create_prison_pomdp_model()

        def create_reactive():
            factory_calls["finite-prison-reactive"] += 1
            return create_prison_reactive_model()

        baseline_factory = ModelFactory(
            name="finite-prison-pomdp",
            create=create_baseline,
            version="1.0.0",
            implementation_revision="prison-pomdp-v1",
        )
        reactive_factory = ModelFactory(
            name="finite-prison-reactive",
            create=create_reactive,
            version="1.0.0",
            implementation_revision="prison-reactive-v1",
        )
        runner = SimulationRunner()
        train_targets = construct_categorical_targets(
            dataset,
            role=ObservationPartitionRole.TRAIN,
            spec=spec,
        )

        training_reports = {}
        acceptance_sets = {}
        for name, factory in (
            ("finite-prison-pomdp", baseline_factory),
            ("finite-prison-reactive", reactive_factory),
        ):
            report = api["fit_training_target_grid"](
                runner=runner,
                model=factory,
                target_report=train_targets,
                parameter_grid=_GRID,
                simulation_seeds=_TRAIN_SEEDS,
                extractor=prison_initial_action_metrics,
                loss=loss,
            )
            training_reports[name] = report
            acceptance = ParameterAcceptanceSet.from_parameters(
                report.candidate_parameters,
                source_manifest_hashes=(report.manifest.content_hash,),
            )
            self.assertEqual(
                acceptance.source_manifest_hashes,
                (report.manifest.content_hash,),
            )
            acceptance_sets[name] = acceptance

        selection_targets = construct_categorical_targets(
            dataset,
            role=ObservationPartitionRole.SELECTION_VALIDATION,
            spec=spec,
        )
        selection_suite = HeldOutSuite(
            name="prison-model-selection-v1",
            role=EvaluationRole.SELECTION_VALIDATION,
            cases=tuple(
                HeldOutCase(
                    scenario=case.scenario,
                    seeds=_SELECTION_SEEDS,
                    target=case.target_map,
                    name=case.name,
                )
                for case in selection_targets.cases
            ),
        )
        baseline_selection = select_on_validation_suite(
            runner=runner,
            model=baseline_factory,
            accepted_parameters=acceptance_sets["finite-prison-pomdp"],
            suite=selection_suite,
            extractor=prison_initial_action_metrics,
            loss=loss,
        )
        reactive_selection = select_on_validation_suite(
            runner=runner,
            model=reactive_factory,
            accepted_parameters=acceptance_sets["finite-prison-reactive"],
            suite=selection_suite,
            extractor=prison_initial_action_metrics,
            loss=loss,
        )
        baseline_frozen = FrozenModelCandidate.from_selection(
            "finite-prison-pomdp",
            baseline_factory,
            baseline_selection,
        )
        reactive_frozen = FrozenModelCandidate.from_selection(
            "finite-prison-reactive",
            reactive_factory,
            reactive_selection,
        )
        self.assertEqual(
            baseline_frozen.selection_manifest_hash,
            baseline_selection.manifest.content_hash,
        )
        self.assertEqual(
            reactive_frozen.selection_manifest_hash,
            reactive_selection.manifest.content_hash,
        )

        protocol = PreregisteredEvaluationProtocol.create(
            name="prison-model-comparison-v1",
            version="1",
            dataset=dataset,
            target_spec=spec,
            extractor=prison_initial_action_metrics,
            loss=loss,
            simulation_seeds=_FINAL_SEEDS,
            baseline_name="finite-prison-pomdp",
            candidates=(baseline_frozen, reactive_frozen),
            thresholds=AdequacyThresholds(
                max_mean_loss=0.35,
                max_worst_loss=0.50,
            ),
        )
        release = api["ProtocolRelease"].create(
            name="prison-model-comparison-release",
            version="1",
            protocol=protocol,
            source_revision={
                "kind": "declared_revision",
                "repository": "qigao/lean",
                "revision": "prison-alternative-release-v1",
            },
        )
        receipt = api["WitnessReceipt"].create(
            provider="test-fixture",
            authority="integration-test",
            subject_hash=release.content_hash,
            reference="prison-release-v1",
            claimed_at="2026-08-23T00:00:00Z",
            proof={"nonce": "release-v1"},
        )
        verified = api["verify_protocol_release"](
            release,
            protocol=protocol,
            receipts=(receipt,),
            verifier=FixtureWitnessVerifier(),
        )
        final_targets = construct_categorical_targets(
            dataset,
            role=ObservationPartitionRole.FINAL_TEST,
            spec=spec,
        )
        models = (
            ComparisonModel(frozen=baseline_frozen, model=baseline_factory),
            ComparisonModel(frozen=reactive_frozen, model=reactive_factory),
        )
        return {
            "api": api,
            "dataset": dataset,
            "spec": spec,
            "loss": loss,
            "runner": runner,
            "training_reports": training_reports,
            "selection_targets": selection_targets,
            "baseline_factory": baseline_factory,
            "reactive_factory": reactive_factory,
            "baseline_frozen": baseline_frozen,
            "reactive_frozen": reactive_frozen,
            "protocol": protocol,
            "release": release,
            "verified": verified,
            "final_targets": final_targets,
            "models": models,
            "factory_calls": factory_calls,
        }

    def test_full_fixture_train_selection_witness_final_pipeline(self):
        context = self._prepare_release()
        report = context["api"]["compare_released_models"](
            runner=context["runner"],
            verified_release=context["verified"],
            protocol=context["protocol"],
            models=context["models"],
            target_set=context["final_targets"],
            extractor=prison_initial_action_metrics,
            loss=context["loss"],
        )
        self.assertEqual(
            {entry.name for entry in report.comparison.ranking},
            {"finite-prison-pomdp", "finite-prison-reactive"},
        )
        self.assertEqual(
            report.manifest.inputs["release_hash"],
            context["release"].content_hash,
        )
        self.assertIs(attest_report(report).require_integrity(), report)
        for entry in report.comparison.ranking:
            cases = entry.final_test.validation.manifest.inputs["cases"]
            self.assertTrue(cases)
            for case in cases:
                self.assertEqual(case["seeds"], _FINAL_SEEDS)

    def test_final_gate_rejects_role_and_identity_drift_before_execution(self):
        context = self._prepare_release()
        compare = context["api"]["compare_released_models"]

        def assert_preflight_rejects(
            expected_exception,
            *,
            verified_release=None,
            protocol=None,
            models=None,
            target_set=None,
            extractor=None,
            loss=None,
        ):
            before = dict(context["factory_calls"])
            with self.assertRaises(expected_exception):
                compare(
                    runner=context["runner"],
                    verified_release=(
                        context["verified"]
                        if verified_release is None
                        else verified_release
                    ),
                    protocol=(
                        context["protocol"]
                        if protocol is None
                        else protocol
                    ),
                    models=context["models"] if models is None else models,
                    target_set=(
                        context["final_targets"]
                        if target_set is None
                        else target_set
                    ),
                    extractor=(
                        prison_initial_action_metrics
                        if extractor is None
                        else extractor
                    ),
                    loss=context["loss"] if loss is None else loss,
                )
            self.assertEqual(context["factory_calls"], before)

        assert_preflight_rejects(
            ValueError,
            target_set=context["selection_targets"],
        )
        assert_preflight_rejects(
            ValueError,
            extractor=changed_prison_initial_action_metrics,
        )
        changed_loss = CategoricalBrierLoss(
            (
                CategoricalMetricGroup(
                    "changed-initial-action",
                    ("initial.scout", "initial.escape", "initial.submit"),
                ),
            )
        )
        assert_preflight_rejects(ValueError, loss=changed_loss)

        changed_seed_protocol = PreregisteredEvaluationProtocol.create(
            name=context["protocol"].name,
            version=context["protocol"].version,
            dataset=context["dataset"],
            target_spec=context["spec"],
            extractor=prison_initial_action_metrics,
            loss=context["loss"],
            simulation_seeds=(303, 304),
            baseline_name=context["protocol"].baseline_name,
            candidates=context["protocol"].candidates,
            thresholds=context["protocol"].thresholds,
        )
        assert_preflight_rejects(
            context["api"]["ProtocolReleaseVerificationError"],
            protocol=changed_seed_protocol,
        )

        selected_beta = dict(context["baseline_frozen"].parameters)["beta"]
        substituted_beta = 4.0 if selected_beta != 4.0 else 0.5
        substituted_baseline = FrozenModelCandidate.freeze(
            name="finite-prison-pomdp",
            model=context["baseline_factory"],
            parameters={"beta": substituted_beta},
            selection_manifest_hash=context["baseline_frozen"].selection_manifest_hash,
        )
        substituted_models = (
            ComparisonModel(
                frozen=substituted_baseline,
                model=context["baseline_factory"],
            ),
            context["models"][1],
        )
        assert_preflight_rejects(ValueError, models=substituted_models)

        assert_preflight_rejects(
            TypeError,
            verified_release=context["release"],
        )

        other_protocol = PreregisteredEvaluationProtocol.create(
            name="other-prison-model-comparison-v1",
            version="1",
            dataset=context["dataset"],
            target_spec=context["spec"],
            extractor=prison_initial_action_metrics,
            loss=context["loss"],
            simulation_seeds=_FINAL_SEEDS,
            baseline_name=context["protocol"].baseline_name,
            candidates=context["protocol"].candidates,
            thresholds=context["protocol"].thresholds,
        )
        other_release = context["api"]["ProtocolRelease"].create(
            name="other-prison-model-comparison-release",
            version="1",
            protocol=other_protocol,
            source_revision={
                "kind": "declared_revision",
                "repository": "qigao/lean",
                "revision": "prison-alternative-release-v1",
            },
        )
        other_receipt = context["api"]["WitnessReceipt"].create(
            provider="test-fixture",
            authority="integration-test",
            subject_hash=other_release.content_hash,
            reference="other-prison-release-v1",
            claimed_at="2026-08-23T00:00:00Z",
            proof={"nonce": "release-v1"},
        )
        other_verified = context["api"]["verify_protocol_release"](
            other_release,
            protocol=other_protocol,
            receipts=(other_receipt,),
            verifier=FixtureWitnessVerifier(),
        )
        assert_preflight_rejects(
            context["api"]["ProtocolReleaseVerificationError"],
            verified_release=other_verified,
        )

    def test_training_and_release_apis_are_exported_from_observations_and_root(self):
        self._public_pipeline_api()


if __name__ == "__main__":
    unittest.main()
