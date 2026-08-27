from __future__ import annotations

import inspect
import math
from pathlib import Path
import random
import unittest

from narrative_dynamics.adapters.prison_metrics import prison_initial_action_metrics
from narrative_dynamics.adapters.prison_pomdp import create_prison_pomdp_model
from narrative_dynamics.adapters.prison_reactive import create_prison_reactive_model
from narrative_dynamics.losses import CategoricalBrierLoss, CategoricalMetricGroup
from narrative_dynamics.manifest import component_identity
from narrative_dynamics.model_comparison import ComparisonModel, compare_models_on_final_partition
from narrative_dynamics.observations import (
    AdequacyThresholds,
    CategoricalTargetSpec,
    FrozenModelCandidate,
    ObservationPartitionRole,
    PreregisteredEvaluationProtocol,
    construct_categorical_targets,
    fit_training_target_grid,
    load_observation_dataset,
)
from narrative_dynamics.report_artifact import attest_report
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.uncertainty import ParameterAcceptanceSet
from narrative_dynamics.validation import (
    EvaluationRole,
    HeldOutCase,
    HeldOutSuite,
    select_on_validation_suite,
)


_ADAPTER_IMPORT_ERROR: ImportError | None = None
try:
    import narrative_dynamics.adapters.narrative_prison as narrative_prison_module
    from narrative_dynamics.adapters.narrative_prison import (
        NarrativePrisonModelSource,
        create_narrative_prison_intentional_source,
        create_narrative_prison_planning_source,
        create_narrative_prison_reactive_source,
    )
except ImportError as error:
    _ADAPTER_IMPORT_ERROR = error


FIXTURE = Path("fixtures/observations/prison_initial_choice_v1.json")
GRID = {"beta": (0.5, 1.0, 2.0, 4.0)}
TRAIN_SEEDS = (101, 102)
SELECTION_SEEDS = (201, 202)
FINAL_SEEDS = (301, 302)
TOL = 1e-12


def target_spec() -> CategoricalTargetSpec:
    return CategoricalTargetSpec(
        name="prison-initial-choice-target",
        version="1",
        categories=("scout", "escape", "submit"),
        metric_prefix="initial",
    )


def brier_loss() -> CategoricalBrierLoss:
    return CategoricalBrierLoss(
        (
            CategoricalMetricGroup(
                "initial-action",
                ("initial.scout", "initial.escape", "initial.submit"),
            ),
        )
    )


def policy(trace) -> dict[str, float]:
    return {
        action: float(trace.outcome["initial_policy"][action])
        for action in ("scout", "escape", "submit")
    }


class NarrativeHeldOutModelComparisonTests(unittest.TestCase):
    def require_adapter(self) -> None:
        if _ADAPTER_IMPORT_ERROR is not None:
            self.fail(
                "generic narrative prison benchmark adapter is missing: "
                f"{_ADAPTER_IMPORT_ERROR}"
            )

    def assert_policy_close(self, left, right) -> None:
        self.assertEqual(set(left), set(right))
        for action in left:
            self.assertTrue(
                math.isclose(
                    float(left[action]),
                    float(right[action]),
                    rel_tol=0.0,
                    abs_tol=TOL,
                ),
                (action, left[action], right[action]),
            )

    def test_sources_are_fresh_per_batch_and_bind_family_implementation_identity(self):
        self.require_adapter()
        sources = (
            create_narrative_prison_reactive_source(),
            create_narrative_prison_intentional_source(),
            create_narrative_prison_planning_source(),
        )
        self.assertTrue(
            all(isinstance(source, NarrativePrisonModelSource) for source in sources)
        )
        self.assertEqual(
            tuple(source.family for source in sources),
            ("reactive", "intentional", "planning"),
        )
        self.assertEqual(len({source.name for source in sources}), 3)
        for source in sources:
            self.assertEqual(source.lifecycle, "fresh_per_batch")
            first = source.instantiate()
            second = source.instantiate()
            self.assertIsNot(first, second)
            self.assertEqual(first.name, source.name)
            identity = component_identity(source)
            self.assertEqual(identity["family"], source.family)
            self.assertEqual(identity["lifecycle"], "fresh_per_batch")
            for key in (
                "source_implementation_identity",
                "model_implementation_identity",
                "benchmark_builder_implementation_identity",
                "family_builder_implementation_identity",
                "dispatch_implementation_identity",
            ):
                self.assertIn(key, identity)

    def test_adapter_uses_only_unified_dispatch_and_not_comparison_protocol(self):
        self.require_adapter()
        source = inspect.getsource(narrative_prison_module)
        for banned in (
            "narrative_dynamics.model_comparison",
            "narrative_dynamics.observations",
            "compare_models_on_final_partition",
            "PreregisteredEvaluationProtocol",
            "fit_training_target_grid",
            "run_runtime_reactive_decision",
            "run_runtime_intentional_decision",
            "run_runtime_planning_decision",
        ):
            self.assertNotIn(banned, source)
        self.assertIn("run_runtime_decision", source)

    def test_horizon_one_zero_fills_scout_and_output_binds_runtime_dispatch(self):
        self.require_adapter()
        dataset = load_observation_dataset(FIXTURE)
        scenario = next(
            record.scenario
            for record in dataset.partition(ObservationPartitionRole.TRAIN).records
            if record.id == "train-2"
        )
        runner = SimulationRunner()
        for source in (
            create_narrative_prison_reactive_source(),
            create_narrative_prison_intentional_source(),
            create_narrative_prison_planning_source(),
        ):
            with self.subTest(source=source.name):
                trace = runner.run_once(source, scenario, {"beta": 2.0}, seed=17)
                self.assertEqual(
                    set(trace.outcome["initial_policy"]),
                    {"scout", "escape", "submit"},
                )
                self.assertEqual(trace.outcome["initial_policy"]["scout"], 0.0)
                dispatch = trace.outcome["runtime_dispatch"]
                self.assertEqual(
                    trace.outcome["selected_action"],
                    dispatch["selected_action"],
                )
                self.assertEqual(
                    {
                        key: trace.outcome["initial_policy"][key]
                        for key in ("escape", "submit")
                    },
                    {
                        key: dispatch["action_policy"][key]
                        for key in ("escape", "submit")
                    },
                )

    def test_generic_reactive_matches_finite_reactive_on_all_fixture_cases_and_betas(self):
        self.require_adapter()
        dataset = load_observation_dataset(FIXTURE)
        runner = SimulationRunner()
        generic = create_narrative_prison_reactive_source()
        legacy = create_prison_reactive_model()
        for partition in dataset.partitions:
            for record in partition.records:
                for beta in GRID["beta"]:
                    with self.subTest(record=record.id, beta=beta):
                        actual = policy(
                            runner.run_once(
                                generic,
                                record.scenario,
                                {"beta": beta},
                                seed=1,
                            )
                        )
                        expected_run = legacy.simulate(
                            record.scenario,
                            {"beta": beta},
                            random.Random(1),
                        )
                        expected = {
                            key: float(expected_run.outcome["initial_policy"][key])
                            for key in ("scout", "escape", "submit")
                        }
                        self.assert_policy_close(actual, expected)

    def test_generic_planning_matches_finite_pomdp_on_all_fixture_cases_and_betas(self):
        self.require_adapter()
        dataset = load_observation_dataset(FIXTURE)
        runner = SimulationRunner()
        generic = create_narrative_prison_planning_source()
        legacy = create_prison_pomdp_model()
        for partition in dataset.partitions:
            for record in partition.records:
                for beta in GRID["beta"]:
                    with self.subTest(record=record.id, beta=beta):
                        actual = policy(
                            runner.run_once(
                                generic,
                                record.scenario,
                                {"beta": beta},
                                seed=1,
                            )
                        )
                        expected_run = legacy.simulate(
                            record.scenario,
                            {"beta": beta},
                            random.Random(1),
                        )
                        expected = {
                            key: float(expected_run.outcome["initial_policy"][key])
                            for key in ("scout", "escape", "submit")
                        }
                        self.assert_policy_close(actual, expected)

    def test_persistence_pair_separates_planning_from_non_lookahead_families(self):
        self.require_adapter()
        dataset = load_observation_dataset(FIXTURE)
        records = {
            record.id: record
            for record in dataset.partition(ObservationPartitionRole.FINAL_TEST).records
        }
        first = records["final-1"].scenario
        second = records["final-2"].scenario
        runner = SimulationRunner()
        for source in (
            create_narrative_prison_reactive_source(),
            create_narrative_prison_intentional_source(),
        ):
            left = policy(
                runner.run_once(source, first, {"beta": 2.0}, seed=301)
            )
            right = policy(
                runner.run_once(source, second, {"beta": 2.0}, seed=301)
            )
            self.assert_policy_close(left, right)

        planning = create_narrative_prison_planning_source()
        left = policy(runner.run_once(planning, first, {"beta": 2.0}, seed=301))
        right = policy(runner.run_once(planning, second, {"beta": 2.0}, seed=301))
        self.assertTrue(any(abs(left[key] - right[key]) > TOL for key in left))

    def test_seed_changes_lineage_but_not_predicted_policy(self):
        self.require_adapter()
        dataset = load_observation_dataset(FIXTURE)
        scenario = dataset.partition(ObservationPartitionRole.FINAL_TEST).records[0].scenario
        runner = SimulationRunner()
        for source in (
            create_narrative_prison_reactive_source(),
            create_narrative_prison_intentional_source(),
            create_narrative_prison_planning_source(),
        ):
            with self.subTest(source=source.name):
                first = runner.run_once(source, scenario, {"beta": 2.0}, seed=301)
                second = runner.run_once(source, scenario, {"beta": 2.0}, seed=302)
                self.assertEqual(policy(first), policy(second))
                self.assertNotEqual(
                    first.manifest.content_hash,
                    second.manifest.content_hash,
                )

    def prepare_three_family_protocol(self):
        self.require_adapter()
        dataset = load_observation_dataset(FIXTURE)
        spec = target_spec()
        loss = brier_loss()
        sources = {
            source.name: source
            for source in (
                create_narrative_prison_reactive_source(),
                create_narrative_prison_intentional_source(),
                create_narrative_prison_planning_source(),
            )
        }
        runner = SimulationRunner()
        train_targets = construct_categorical_targets(
            dataset,
            role=ObservationPartitionRole.TRAIN,
            spec=spec,
        )
        selection_targets = construct_categorical_targets(
            dataset,
            role=ObservationPartitionRole.SELECTION_VALIDATION,
            spec=spec,
        )
        selection_suite = HeldOutSuite(
            name="generic-narrative-prison-selection-v1",
            role=EvaluationRole.SELECTION_VALIDATION,
            cases=tuple(
                HeldOutCase(
                    scenario=case.scenario,
                    seeds=SELECTION_SEEDS,
                    target=case.target_map,
                    name=case.name,
                )
                for case in selection_targets.cases
            ),
        )
        frozen = {}
        for name, source in sources.items():
            training = fit_training_target_grid(
                runner=runner,
                model=source,
                target_report=train_targets,
                parameter_grid=GRID,
                simulation_seeds=TRAIN_SEEDS,
                extractor=prison_initial_action_metrics,
                loss=loss,
            )
            accepted = ParameterAcceptanceSet.from_parameters(
                training.candidate_parameters,
                source_manifest_hashes=(training.manifest.content_hash,),
            )
            selection = select_on_validation_suite(
                runner=runner,
                model=source,
                accepted_parameters=accepted,
                suite=selection_suite,
                extractor=prison_initial_action_metrics,
                loss=loss,
            )
            frozen[name] = FrozenModelCandidate.from_selection(
                name,
                source,
                selection,
            )

        planning_name = create_narrative_prison_planning_source().name
        protocol = PreregisteredEvaluationProtocol.create(
            name="generic-narrative-prison-three-family-v1",
            version="1",
            dataset=dataset,
            target_spec=spec,
            extractor=prison_initial_action_metrics,
            loss=loss,
            simulation_seeds=FINAL_SEEDS,
            baseline_name=planning_name,
            candidates=tuple(frozen[name] for name in sorted(frozen)),
            thresholds=AdequacyThresholds(2.0, 2.0),
        )
        final_targets = construct_categorical_targets(
            dataset,
            role=ObservationPartitionRole.FINAL_TEST,
            spec=spec,
        )
        return runner, sources, frozen, protocol, final_targets, loss

    def test_three_generic_families_complete_preregistered_final_comparison(self):
        (
            runner,
            sources,
            frozen,
            protocol,
            final_targets,
            loss,
        ) = self.prepare_three_family_protocol()
        report = compare_models_on_final_partition(
            runner=runner,
            models=tuple(
                ComparisonModel(frozen=frozen[name], model=sources[name])
                for name in sorted(sources)
            ),
            target_set=final_targets,
            extractor=prison_initial_action_metrics,
            loss=loss,
            simulation_seeds=FINAL_SEEDS,
            protocol=protocol,
        )
        self.assertEqual({entry.name for entry in report.ranking}, set(sources))
        self.assertEqual(
            report.baseline_name,
            create_narrative_prison_planning_source().name,
        )
        self.assertIs(attest_report(report).require_integrity(), report)
        for entry in report.ranking:
            self.assertTrue(math.isfinite(entry.mean_loss))
            self.assertTrue(math.isfinite(entry.worst_loss))
            cases = entry.final_test.validation.manifest.inputs["cases"]
            self.assertTrue(cases)
            self.assertTrue(all(case["seeds"] == FINAL_SEEDS for case in cases))


if __name__ == "__main__":
    unittest.main()
