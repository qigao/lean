from __future__ import annotations

import inspect
import unittest

from narrative_dynamics.adapters.prison_metrics import prison_initial_action_metrics
from narrative_dynamics.model_comparison import ComparisonModel, compare_models_on_final_partition
from narrative_dynamics.observations import (
    AdequacyThresholds,
    FrozenModelCandidate,
    ObservationPartitionRole,
    PreregisteredEvaluationProtocol,
    construct_categorical_targets,
    fit_training_target_grid,
    load_observation_dataset,
)
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.uncertainty import ParameterAcceptanceSet
from narrative_dynamics.validation import EvaluationRole, HeldOutCase, HeldOutSuite, select_on_validation_suite

from tests.test_narrative_identification_protocol import (
    FINAL_SEEDS,
    FIXTURE,
    SELECTION_SEEDS,
    TRAIN_SEEDS,
    TRUTH,
    brier_loss,
    log_loss,
    require_identification,
    target_spec,
)

_IDENTIFICATION_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.identification import (
        IdentificationComparisonError,
        score_model_comparison_by_stratum,
        validate_sibling_final_protocols,
    )
except ImportError as error:
    _IDENTIFICATION_IMPORT_ERROR = error
    IdentificationComparisonError = None
    score_model_comparison_by_stratum = None
    validate_sibling_final_protocols = None

_ADAPTER_IMPORT_ERROR: ImportError | None = None
try:
    import narrative_dynamics.adapters.narrative_prison_identification as identification_adapter_module
    from narrative_dynamics.adapters.narrative_prison_identification import (
        NarrativePrisonIdentificationSource,
        create_narrative_identification_intentional_source,
        create_narrative_identification_planning_source,
        create_narrative_identification_reactive_source,
    )
except ImportError as error:
    _ADAPTER_IMPORT_ERROR = error
    identification_adapter_module = None
    NarrativePrisonIdentificationSource = None
    create_narrative_identification_intentional_source = None
    create_narrative_identification_planning_source = None
    create_narrative_identification_reactive_source = None


BETA_GRID = (0.5, 1.0, 2.0, 4.0)


def require_adapter(test: unittest.TestCase) -> None:
    if _ADAPTER_IMPORT_ERROR is not None:
        test.fail(f"synthetic identification adapter is missing: {_ADAPTER_IMPORT_ERROR}")


def source_map():
    return {
        "reactive": create_narrative_identification_reactive_source(),
        "intentional": create_narrative_identification_intentional_source(),
        "planning": create_narrative_identification_planning_source(),
    }


def parameters_for(family: str):
    if family == "intentional":
        return dict(TRUTH)
    return {"beta_action": TRUTH["beta_action"]}


def grid_for(family: str):
    if family == "intentional":
        return {
            "beta_goal": BETA_GRID,
            "beta_action": BETA_GRID,
            "goal_pressure_scale": (1.0,),
            "instrumentality_scale": (1.0,),
        }
    return {"beta_action": BETA_GRID}


def prepare_selected_candidates():
    dataset = load_observation_dataset(FIXTURE)
    sources = source_map()
    runner = SimulationRunner()
    spec = target_spec()
    loss = brier_loss()
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
        name="narrative-synthetic-identification-selection-v1",
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
    training = {}
    selection = {}
    for family, source in sources.items():
        training[family] = fit_training_target_grid(
            runner=runner,
            model=source,
            target_report=train_targets,
            parameter_grid=grid_for(family),
            simulation_seeds=TRAIN_SEEDS,
            extractor=prison_initial_action_metrics,
            loss=loss,
        )
        accepted = ParameterAcceptanceSet.from_parameters(
            training[family].candidate_parameters,
            source_manifest_hashes=(training[family].manifest.content_hash,),
        )
        selection[family] = select_on_validation_suite(
            runner=runner,
            model=source,
            accepted_parameters=accepted,
            suite=selection_suite,
            extractor=prison_initial_action_metrics,
            loss=loss,
        )
        frozen[family] = FrozenModelCandidate.from_selection(
            source.name,
            source,
            selection[family],
        )
    return dataset, sources, frozen, training, selection


def prepare_final_protocols():
    dataset, sources, frozen, training, selection = prepare_selected_candidates()
    candidates = tuple(frozen[family] for family in sorted(frozen))
    intentional_name = sources["intentional"].name
    brier_protocol = PreregisteredEvaluationProtocol.create(
        name="narrative-synthetic-identification-brier-v1",
        version="1",
        dataset=dataset,
        target_spec=target_spec(),
        extractor=prison_initial_action_metrics,
        loss=brier_loss(),
        simulation_seeds=FINAL_SEEDS,
        baseline_name=intentional_name,
        candidates=candidates,
        thresholds=AdequacyThresholds(2.0, 2.0),
    )
    log_protocol = PreregisteredEvaluationProtocol.create(
        name="narrative-synthetic-identification-log-v1",
        version="1",
        dataset=dataset,
        target_spec=target_spec(),
        extractor=prison_initial_action_metrics,
        loss=log_loss(),
        simulation_seeds=FINAL_SEEDS,
        baseline_name=intentional_name,
        candidates=candidates,
        thresholds=AdequacyThresholds(20.0, 20.0),
    )
    final_targets = construct_categorical_targets(
        dataset,
        role=ObservationPartitionRole.FINAL_TEST,
        spec=target_spec(),
    )
    models = tuple(
        ComparisonModel(frozen=frozen[family], model=sources[family])
        for family in sorted(sources)
    )
    return (
        dataset,
        sources,
        frozen,
        training,
        selection,
        brier_protocol,
        log_protocol,
        final_targets,
        models,
    )


def execute_final_comparisons():
    (
        dataset,
        sources,
        frozen,
        training,
        selection,
        brier_protocol,
        log_protocol,
        final_targets,
        models,
    ) = prepare_final_protocols()
    validate_sibling_final_protocols(brier_protocol, log_protocol)
    runner = SimulationRunner()
    brier_report = compare_models_on_final_partition(
        runner=runner,
        models=models,
        target_set=final_targets,
        extractor=prison_initial_action_metrics,
        loss=brier_loss(),
        simulation_seeds=FINAL_SEEDS,
        protocol=brier_protocol,
    )
    log_report = compare_models_on_final_partition(
        runner=runner,
        models=models,
        target_set=final_targets,
        extractor=prison_initial_action_metrics,
        loss=log_loss(),
        simulation_seeds=FINAL_SEEDS,
        protocol=log_protocol,
    )
    brier_finding = score_model_comparison_by_stratum(
        dataset=dataset,
        protocol=brier_protocol,
        report=brier_report,
    )
    log_finding = score_model_comparison_by_stratum(
        dataset=dataset,
        protocol=log_protocol,
        report=log_report,
    )
    return (
        dataset,
        sources,
        frozen,
        brier_protocol,
        log_protocol,
        final_targets,
        brier_report,
        log_report,
        brier_finding,
        log_finding,
    )


class NarrativeIdentificationModelComparisonTests(unittest.TestCase):
    def setUp(self):
        require_identification(self)
        require_adapter(self)

    def test_identification_adapter_uses_unified_dispatch_and_not_identification_or_comparison_protocol(self):
        source = inspect.getsource(identification_adapter_module)
        for banned in (
            "narrative_dynamics.identification",
            "narrative_dynamics.model_comparison",
            "PreregisteredEvaluationProtocol",
            "compare_models_on_final_partition",
            "run_runtime_reactive_decision",
            "run_runtime_intentional_decision",
            "run_runtime_planning_decision",
        ):
            self.assertNotIn(banned, source)
        self.assertIn("run_runtime_decision", source)

    def test_source_parameter_schemas_match_real_family_semantics(self):
        dataset = load_observation_dataset(FIXTURE)
        scenario = dataset.partition(ObservationPartitionRole.TRAIN).cases[0].scenario
        runner = SimulationRunner()
        for family, source in source_map().items():
            with self.subTest(family=family):
                runner.run_once(source, scenario, parameters_for(family), seed=1)
        with self.assertRaises((TypeError, ValueError)):
            runner.run_once(
                source_map()["reactive"],
                scenario,
                {"beta_action": 1.0, "beta_goal": 2.0},
                seed=1,
            )
        with self.assertRaises((TypeError, ValueError)):
            runner.run_once(
                source_map()["intentional"],
                scenario,
                {"beta_action": 1.0},
                seed=1,
            )

    def test_all_sources_accept_every_fixture_mode(self):
        dataset = load_observation_dataset(FIXTURE)
        runner = SimulationRunner()
        for partition in dataset.partitions:
            for case in partition.cases:
                for family, source in source_map().items():
                    with self.subTest(case=case.name, family=family):
                        trace = runner.run_once(
                            source,
                            case.scenario,
                            parameters_for(family),
                            seed=1,
                        )
                        self.assertEqual(
                            set(trace.outcome["initial_policy"]),
                            {"scout", "escape", "submit"},
                        )

    def test_binary_modes_zero_fill_scout_without_changing_dispatch_policy(self):
        runner = SimulationRunner()
        binary_cases = (
            "train-temp-anchor",
            "train-goal-low",
            "train-memory-hidden",
            "selection-memory-revealed",
            "final-equivalence",
        )
        dataset = load_observation_dataset(FIXTURE)
        cases = {
            case.name: case
            for partition in dataset.partitions
            for case in partition.cases
        }
        for name in binary_cases:
            for family, source in source_map().items():
                with self.subTest(case=name, family=family):
                    trace = runner.run_once(
                        source,
                        cases[name].scenario,
                        parameters_for(family),
                        seed=1,
                    )
                    self.assertEqual(trace.outcome["initial_policy"]["scout"], 0.0)
                    dispatch_policy = trace.outcome["runtime_dispatch"]["action_policy"]
                    self.assertNotIn("scout", dispatch_policy)
                    self.assertEqual(
                        {
                            action: trace.outcome["initial_policy"][action]
                            for action in ("escape", "submit")
                        },
                        dict(dispatch_policy),
                    )

    def test_brier_training_and_selection_freeze_one_candidate_per_family(self):
        dataset, sources, frozen, training, selection = prepare_selected_candidates()
        self.assertEqual(set(frozen), {"reactive", "intentional", "planning"})
        for family in frozen:
            self.assertEqual(
                frozen[family].parameters,
                selection[family].selected_parameters,
            )
            self.assertTrue(training[family].candidate_parameters)

    def test_brier_final_protocol_completes_on_exact_frozen_candidates(self):
        (
            dataset,
            sources,
            frozen,
            brier_protocol,
            log_protocol,
            final_targets,
            brier_report,
            log_report,
            brier_finding,
            log_finding,
        ) = execute_final_comparisons()
        self.assertEqual(
            {entry.name for entry in brier_report.ranking},
            {source.name for source in sources.values()},
        )
        self.assertEqual(
            tuple(candidate.content_hash for candidate in brier_protocol.candidates),
            tuple(candidate.content_hash for candidate in log_protocol.candidates),
        )

    def test_log_sibling_protocol_reuses_exact_brier_candidates_targets_and_seeds(self):
        (
            dataset,
            sources,
            frozen,
            brier_protocol,
            log_protocol,
            final_targets,
            brier_report,
            log_report,
            brier_finding,
            log_finding,
        ) = execute_final_comparisons()
        self.assertIsNone(validate_sibling_final_protocols(brier_protocol, log_protocol))
        self.assertEqual(brier_protocol.candidates, log_protocol.candidates)
        self.assertEqual(brier_protocol.final_target_hash, log_protocol.final_target_hash)
        self.assertEqual(brier_protocol.simulation_seeds, log_protocol.simulation_seeds)
        self.assertEqual(brier_finding.candidate_hashes, log_finding.candidate_hashes)

    def test_log_candidate_or_seed_drift_is_rejected_before_final_scoring(self):
        (
            dataset,
            sources,
            frozen,
            training,
            selection,
            brier_protocol,
            log_protocol,
            final_targets,
            models,
        ) = prepare_final_protocols()
        drift = PreregisteredEvaluationProtocol.create(
            name="narrative-synthetic-identification-log-drift",
            version="1",
            dataset=dataset,
            target_spec=target_spec(),
            extractor=prison_initial_action_metrics,
            loss=log_loss(),
            simulation_seeds=(999, 1000),
            baseline_name=sources["intentional"].name,
            candidates=log_protocol.candidates,
            thresholds=AdequacyThresholds(20.0, 20.0),
        )
        with self.assertRaises(IdentificationComparisonError):
            validate_sibling_final_protocols(brier_protocol, drift)

    def test_scored_findings_expose_equivalence_memory_and_future_strata(self):
        *_, brier_finding, log_finding = execute_final_comparisons()
        for finding in (brier_finding, log_finding):
            self.assertEqual(
                {score.stratum for score in finding.stratum_scores},
                {
                    "observational_equivalence",
                    "memory_evidence",
                    "future_information",
                },
            )

    def test_comparison_does_not_require_one_family_to_win_every_stratum(self):
        *_, brier_finding, log_finding = execute_final_comparisons()
        scores = {}
        for item in brier_finding.stratum_scores:
            scores.setdefault(item.stratum, []).append(item)
        winners = {
            min(
                items,
                key=lambda score: (score.mean_loss, score.worst_loss, score.model_name),
            ).model_name
            for items in scores.values()
        }
        self.assertGreaterEqual(len(winners), 2)


if __name__ == "__main__":
    unittest.main()
