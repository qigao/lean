from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import unittest

from narrative_dynamics.adapters.prison_metrics import prison_initial_action_metrics
from narrative_dynamics.contracts import ExperimentStage, Scenario
from narrative_dynamics.losses import (
    CategoricalBrierLoss,
    CategoricalLogLoss,
    CategoricalMetricGroup,
)
from narrative_dynamics.manifest import callable_identity, component_identity
from narrative_dynamics.observations import (
    AdequacyThresholds,
    CategoricalTargetSpec,
    FrozenModelCandidate,
    ObservationPartitionRole,
    PreregisteredEvaluationProtocol,
    load_observation_dataset,
)

_IDENTIFICATION_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.identification import (
        IdentificationComparisonError,
        IdentificationProtocolError,
        IdentificationStatus,
        InformationInterventionPair,
        ParameterRecoveryExperiment,
        SyntheticIdentificationProtocol,
        validate_sibling_final_protocols,
    )
except ImportError as error:
    _IDENTIFICATION_IMPORT_ERROR = error
    IdentificationComparisonError = None
    IdentificationProtocolError = None
    IdentificationStatus = None
    InformationInterventionPair = None
    ParameterRecoveryExperiment = None
    SyntheticIdentificationProtocol = None
    validate_sibling_final_protocols = None


FIXTURE = Path("fixtures/observations/narrative_identification_v1.json")
BETA_GRID = (0.5, 1.0, 2.0, 4.0)
SCALE_GRID = (0.5, 1.0, 2.0)
TRUTH = {
    "beta_goal": 2.0,
    "beta_action": 1.0,
    "goal_pressure_scale": 1.0,
    "instrumentality_scale": 1.0,
}
GENERATION_SEEDS = (11, 12)
CALIBRATION_BLOCKS = ((101, 102), (111, 112))
TRAIN_SEEDS = (201, 202)
SELECTION_SEEDS = (301, 302)
FINAL_SEEDS = (401, 402)
TOL = 1e-12
HASH1 = "sha256:" + "1" * 64


class DummySource:
    version = "1"

    def __init__(self, name: str):
        self.name = name

    @property
    def lifecycle(self) -> str:
        return "fresh_per_batch"

    def manifest_identity(self):
        return {"name": self.name, "version": self.version, "lifecycle": self.lifecycle}


def require_identification(test: unittest.TestCase) -> None:
    if _IDENTIFICATION_IMPORT_ERROR is not None:
        test.fail(f"synthetic identification API is missing: {_IDENTIFICATION_IMPORT_ERROR}")


def load_fixture():
    return load_observation_dataset(FIXTURE)


def target_spec() -> CategoricalTargetSpec:
    return CategoricalTargetSpec(
        name="narrative-identification-initial-action",
        version="1",
        categories=("scout", "escape", "submit"),
        metric_prefix="initial",
    )


def metric_group() -> CategoricalMetricGroup:
    return CategoricalMetricGroup(
        "initial-action",
        ("initial.scout", "initial.escape", "initial.submit"),
    )


def brier_loss() -> CategoricalBrierLoss:
    return CategoricalBrierLoss((metric_group(),))


def log_loss() -> CategoricalLogLoss:
    return CategoricalLogLoss((metric_group(),))


def case_by_name(name: str):
    dataset = load_fixture()
    for partition in dataset.partitions:
        for case in partition.cases:
            if case.name == name:
                return case
    raise KeyError(name)


def make_pair(
    baseline_name: str = "final-memory-hidden",
    intervention_name: str = "final-memory-revealed",
    *,
    allowed_key: str = "memory_evidence_available",
    expected_relationship: str = "intentional_vs_reactive",
):
    if InformationInterventionPair is None:
        raise RuntimeError("identification API unavailable")
    baseline_case = case_by_name(baseline_name)
    intervention_case = case_by_name(intervention_name)
    baseline = baseline_case.scenario
    intervention = intervention_case.scenario
    frozen = tuple((key,) for key in sorted(baseline.payload) if key != allowed_key)
    return InformationInterventionPair(
        name=f"{baseline_name}--{intervention_name}",
        baseline=baseline,
        intervention=intervention,
        allowed_information_path=(allowed_key,),
        frozen_paths=frozen,
        stratum=baseline_case.provenance["stratum"],
        expected_relationship=expected_relationship,
    )


def make_recovery_experiment(
    *,
    name: str = "joint-beta",
    cases=None,
    parameter_grid=None,
    true_parameters=None,
    target_coordinates=("beta_goal", "beta_action"),
    expected_status=None,
    model_source=None,
):
    if ParameterRecoveryExperiment is None:
        raise RuntimeError("identification API unavailable")
    source = DummySource("dummy-intentional") if model_source is None else model_source
    if cases is None:
        cases = (
            case_by_name("train-temp-anchor").scenario,
            case_by_name("train-goal-low").scenario,
            case_by_name("selection-goal-high").scenario,
        )
    if parameter_grid is None:
        parameter_grid = {
            "beta_goal": BETA_GRID,
            "beta_action": BETA_GRID,
            "goal_pressure_scale": (1.0,),
            "instrumentality_scale": (1.0,),
        }
    if true_parameters is None:
        true_parameters = TRUTH
    if expected_status is None and IdentificationStatus is not None:
        expected_status = IdentificationStatus.IDENTIFIED_UNDER_PROTOCOL
    return ParameterRecoveryExperiment(
        name=name,
        model_identity=component_identity(source),
        true_parameters=true_parameters,
        parameter_grid=parameter_grid,
        target_coordinates=target_coordinates,
        cases=tuple(cases),
        generation_seeds=GENERATION_SEEDS,
        calibration_seed_blocks=CALIBRATION_BLOCKS,
        acceptance_loss_delta=0.0,
        min_acceptance_fraction=1.0,
        metric_identity=callable_identity(prison_initial_action_metrics),
        loss_identity=brier_loss().manifest_identity(),
        expected_status=expected_status,
    )


def dummy_candidates():
    return (
        FrozenModelCandidate.freeze(
            name="reactive",
            model=DummySource("reactive"),
            parameters={"beta_action": 1.0},
            selection_manifest_hash=HASH1,
        ),
        FrozenModelCandidate.freeze(
            name="intentional",
            model=DummySource("intentional"),
            parameters={
                "beta_goal": 2.0,
                "beta_action": 1.0,
                "goal_pressure_scale": 1.0,
                "instrumentality_scale": 1.0,
            },
            selection_manifest_hash=HASH1,
        ),
        FrozenModelCandidate.freeze(
            name="planning",
            model=DummySource("planning"),
            parameters={"beta_action": 1.0},
            selection_manifest_hash=HASH1,
        ),
    )


def make_final_protocol(*, loss, name: str, seeds=FINAL_SEEDS, candidates=None, baseline="intentional", thresholds=None):
    dataset = load_fixture()
    if candidates is None:
        candidates = dummy_candidates()
    if thresholds is None:
        thresholds = (
            AdequacyThresholds(2.0, 2.0)
            if isinstance(loss, CategoricalBrierLoss)
            else AdequacyThresholds(20.0, 20.0)
        )
    return PreregisteredEvaluationProtocol.create(
        name=name,
        version="1",
        dataset=dataset,
        target_spec=target_spec(),
        extractor=prison_initial_action_metrics,
        loss=loss,
        simulation_seeds=tuple(seeds),
        baseline_name=baseline,
        candidates=tuple(candidates),
        thresholds=thresholds,
    )


def make_synthetic_protocol(*, recovery_experiments=None, intervention_pairs=None):
    if SyntheticIdentificationProtocol is None:
        raise RuntimeError("identification API unavailable")
    dataset = load_fixture()
    if recovery_experiments is None:
        recovery_experiments = (make_recovery_experiment(),)
    if intervention_pairs is None:
        intervention_pairs = (
            make_pair(),
            make_pair(
                "final-future-off",
                "final-future-on",
                allowed_key="future_information_available",
                expected_relationship="planning_vs_intentional",
            ),
        )
    return SyntheticIdentificationProtocol.create(
        name="narrative-synthetic-identification-v1",
        version="1",
        claim_scope="synthetic_protocol_only",
        dataset=dataset,
        adapter_sources=(
            DummySource("reactive"),
            DummySource("intentional"),
            DummySource("planning"),
        ),
        recovery_experiments=tuple(recovery_experiments),
        intervention_pairs=tuple(intervention_pairs),
        target_spec=target_spec(),
        extractor=prison_initial_action_metrics,
        brier_loss=brier_loss(),
        log_loss=log_loss(),
        brier_training_seeds=TRAIN_SEEDS,
        brier_selection_seeds=SELECTION_SEEDS,
        final_seeds=FINAL_SEEDS,
        equivalence_tolerance=TOL,
    )


class NarrativeIdentificationProtocolTests(unittest.TestCase):
    def test_manifest_stage_is_additive_and_exact(self):
        self.assertEqual(
            ExperimentStage.SYNTHETIC_IDENTIFICATION.value,
            "synthetic_identification",
        )

    def test_fixture_is_synthetic_partitioned_and_pair_linked(self):
        dataset = load_fixture()
        self.assertEqual(dataset.provenance["claim_scope"], "synthetic_protocol_only")
        self.assertIs(dataset.provenance["synthetic_non_empirical"], True)
        self.assertEqual(
            {partition.role for partition in dataset.partitions},
            set(ObservationPartitionRole),
        )
        cases = {
            case.name: case
            for partition in dataset.partitions
            for case in partition.cases
        }
        for name in (
            "final-memory-hidden",
            "final-memory-revealed",
            "final-future-off",
            "final-future-on",
        ):
            self.assertIn("paired_case_id", cases[name].provenance)
            self.assertIs(cases[name].provenance["synthetic_non_empirical"], True)
        self.assertEqual(
            cases["final-memory-hidden"].provenance["paired_case_id"],
            "final-memory-revealed",
        )
        self.assertEqual(
            cases["final-future-on"].provenance["paired_case_id"],
            "final-future-off",
        )

    def test_protocol_create_binds_case_hashes_and_canonical_hash(self):
        require_identification(self)
        first = make_synthetic_protocol()
        second = make_synthetic_protocol(
            recovery_experiments=tuple(reversed(first.recovery_experiments)),
            intervention_pairs=tuple(reversed(first.intervention_pairs)),
        )
        self.assertEqual(first.content_hash, second.content_hash)
        dataset = load_fixture()
        expected = {}
        for role in ObservationPartitionRole:
            expected[role] = tuple(
                sorted(case.scenario.content_hash for case in dataset.partition(role).cases)
            )
        self.assertEqual(first.train_case_hashes, expected[ObservationPartitionRole.TRAIN])
        self.assertEqual(
            first.selection_case_hashes,
            expected[ObservationPartitionRole.SELECTION_VALIDATION],
        )
        self.assertEqual(first.final_case_hashes, expected[ObservationPartitionRole.FINAL_TEST])

    def test_protocol_claim_scope_is_closed(self):
        require_identification(self)
        good = make_synthetic_protocol()
        with self.assertRaises(IdentificationProtocolError):
            replace(good, claim_scope="empirical_human_cognition")

    def test_recovery_truth_must_be_in_grid(self):
        require_identification(self)
        bad_truth = dict(TRUTH)
        bad_truth["beta_goal"] = 8.0
        with self.assertRaises(IdentificationProtocolError):
            make_recovery_experiment(true_parameters=bad_truth)

    def test_protocol_rejects_duplicate_experiment_and_pair_names(self):
        require_identification(self)
        experiment = make_recovery_experiment()
        pair = make_pair()
        with self.assertRaises(IdentificationProtocolError):
            make_synthetic_protocol(recovery_experiments=(experiment, experiment))
        with self.assertRaises(IdentificationProtocolError):
            make_synthetic_protocol(intervention_pairs=(pair, pair))

    def test_protocol_rejects_partition_overlap(self):
        require_identification(self)
        good = make_synthetic_protocol()
        with self.assertRaises(IdentificationProtocolError):
            replace(good, selection_case_hashes=good.train_case_hashes)

    def test_sibling_final_protocols_allow_only_name_loss_threshold_and_derived_hash_difference(self):
        require_identification(self)
        brier = make_final_protocol(loss=brier_loss(), name="brier")
        log = make_final_protocol(loss=log_loss(), name="log")
        self.assertIsNone(validate_sibling_final_protocols(brier, log))

    def test_sibling_final_protocols_reject_candidate_target_metric_seed_or_baseline_drift(self):
        require_identification(self)
        brier = make_final_protocol(loss=brier_loss(), name="brier")
        with self.subTest("seed"):
            drift = make_final_protocol(loss=log_loss(), name="log-seed", seeds=(499, 500))
            with self.assertRaises(IdentificationComparisonError):
                validate_sibling_final_protocols(brier, drift)
        with self.subTest("baseline"):
            drift = make_final_protocol(loss=log_loss(), name="log-baseline", baseline="planning")
            with self.assertRaises(IdentificationComparisonError):
                validate_sibling_final_protocols(brier, drift)
        with self.subTest("candidate"):
            changed = list(dummy_candidates())
            changed[0] = FrozenModelCandidate.freeze(
                name="reactive",
                model=DummySource("reactive"),
                parameters={"beta_action": 2.0},
                selection_manifest_hash=HASH1,
            )
            drift = make_final_protocol(
                loss=log_loss(),
                name="log-candidate",
                candidates=tuple(changed),
            )
            with self.assertRaises(IdentificationComparisonError):
                validate_sibling_final_protocols(brier, drift)

    def test_protocol_rejects_nonfinite_grid_tolerance_and_acceptance_values(self):
        require_identification(self)
        with self.assertRaises(IdentificationProtocolError):
            make_recovery_experiment(
                parameter_grid={
                    "beta_goal": (0.5, math.nan),
                    "beta_action": (1.0,),
                    "goal_pressure_scale": (1.0,),
                    "instrumentality_scale": (1.0,),
                }
            )
        experiment = make_recovery_experiment()
        with self.assertRaises(IdentificationProtocolError):
            replace(experiment, acceptance_loss_delta=math.inf)
        protocol = make_synthetic_protocol()
        with self.assertRaises(IdentificationProtocolError):
            replace(protocol, equivalence_tolerance=math.nan)


if __name__ == "__main__":
    unittest.main()
