from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

from narrative_dynamics.contracts import Scenario, stable_content_hash
from narrative_dynamics.measurement_validity import (
    MeasurementAuditCase,
    MeasurementModelPrediction,
    MeasurementPredictionArtifact,
    MeasurementSeedPrediction,
    MeasurementValidityStatus,
)
from narrative_dynamics.observations import ObservationPartitionRole
from narrative_dynamics.studies.feher_hare_measurement_validity_v1 import (
    StaySwitchCell,
    StaySwitchDiagnostic,
    build_feher_hare_stay_switch_diagnostics,
)

from tests.measurement_validity_fixtures import (
    measurement_input,
    measurement_protocol,
)


def _hash(*parts: object) -> str:
    return stable_content_hash(("measurement-diagnostic-fixture",) + parts)


def _history(
    *,
    trial: int,
    action: str,
    reward: int,
    common: bool,
) -> dict[str, object]:
    return {
        "trial_id": trial,
        "first_stage_action": action,
        "transition_common": common,
        "final_state": "state_0",
        "second_stage_action": "action_0",
        "reward": reward,
    }


def _case(
    *,
    role: ObservationPartitionRole,
    task: str,
    participant: str,
    trial: int,
    current_action: str,
    previous: object | None,
) -> MeasurementAuditCase:
    return MeasurementAuditCase(
        role=role,
        scenario=Scenario(
            id=f"diagnostic-{task}-{participant}-{trial}",
            payload={
                "task_variant": task,
                "first_stage_configuration": (
                    ("action_0_position", "left"),
                    ("action_1_position", "right"),
                ),
                "history": () if previous is None else (previous,),
            },
        ),
        target=(
            (
                "first_stage.action_0",
                1.0 if current_action == "action_0" else 0.0,
            ),
            (
                "first_stage.action_1",
                1.0 if current_action == "action_1" else 0.0,
            ),
        ),
        task_variant=task,
        participant_group_hash=_hash(task, participant),
        trial_index=trial,
        record_hash=_hash(task, participant, trial, "record"),
    )


def _diagnostic_cases() -> tuple[MeasurementAuditCase, ...]:
    return (
        _case(
            role=ObservationPartitionRole.TRAIN,
            task="magic_carpet",
            participant="p1",
            trial=0,
            current_action="action_0",
            previous=None,
        ),
        _case(
            role=ObservationPartitionRole.TRAIN,
            task="magic_carpet",
            participant="p1",
            trial=2,
            current_action="action_0",
            previous=_history(
                trial=0,
                action="action_0",
                reward=1,
                common=True,
            ),
        ),
        _case(
            role=ObservationPartitionRole.TRAIN,
            task="magic_carpet",
            participant="p1",
            trial=5,
            current_action="action_1",
            previous=_history(
                trial=2,
                action="action_0",
                reward=0,
                common=True,
            ),
        ),
        _case(
            role=ObservationPartitionRole.TRAIN,
            task="magic_carpet",
            participant="p1",
            trial=9,
            current_action="action_1",
            previous=_history(
                trial=5,
                action="action_1",
                reward=1,
                common=False,
            ),
        ),
        _case(
            role=ObservationPartitionRole.TRAIN,
            task="magic_carpet",
            participant="p1",
            trial=12,
            current_action="action_1",
            previous=_history(
                trial=9,
                action="action_1",
                reward=0,
                common=False,
            ),
        ),
        _case(
            role=ObservationPartitionRole.TRAIN,
            task="magic_carpet",
            participant="p1",
            trial=20,
            current_action="action_0",
            previous=_history(
                trial=12,
                action="action_1",
                reward=0,
                common=False,
            ),
        ),
        _case(
            role=ObservationPartitionRole.SELECTION_VALIDATION,
            task="magic_carpet",
            participant="p2",
            trial=1,
            current_action="action_1",
            previous=None,
        ),
        _case(
            role=ObservationPartitionRole.SELECTION_VALIDATION,
            task="spaceship",
            participant="p3",
            trial=0,
            current_action="action_0",
            previous=None,
        ),
        _case(
            role=ObservationPartitionRole.SELECTION_VALIDATION,
            task="spaceship",
            participant="p3",
            trial=3,
            current_action="action_0",
            previous=_history(
                trial=0,
                action="action_0",
                reward=1,
                common=True,
            ),
        ),
        _case(
            role=ObservationPartitionRole.SELECTION_VALIDATION,
            task="spaceship",
            participant="p3",
            trial=7,
            current_action="action_1",
            previous=_history(
                trial=3,
                action="action_0",
                reward=0,
                common=True,
            ),
        ),
        _case(
            role=ObservationPartitionRole.SELECTION_VALIDATION,
            task="spaceship",
            participant="p3",
            trial=11,
            current_action="action_1",
            previous=_history(
                trial=7,
                action="action_1",
                reward=1,
                common=False,
            ),
        ),
    )


def _expected_stay_probability(case: MeasurementAuditCase) -> float:
    history = case.scenario.payload.get("history")
    if not isinstance(history, tuple) or not history:
        return 0.5
    latest = history[-1]
    if not isinstance(latest, dict) and not hasattr(latest, "get"):
        return 0.5
    reward = latest.get("reward")
    common = latest.get("transition_common")
    if (
        not isinstance(reward, int)
        or isinstance(reward, bool)
        or reward not in (0, 1)
    ):
        return 0.5
    if not isinstance(common, bool):
        return 0.5
    by_cell = {
        "magic_carpet": {
            (1, True): 0.8,
            (0, True): 0.3,
            (1, False): 0.4,
            (0, False): 0.4,
        },
        "spaceship": {
            (1, True): 0.6,
            (0, True): 0.2,
            (1, False): 0.3,
            (0, False): 0.5,
        },
    }
    return by_cell[case.task_variant][(reward, common)]


def _artifact(audit_input) -> MeasurementPredictionArtifact:
    protocol = measurement_protocol(audit_input)
    seeds_by_role = dict(protocol.seeds_by_role)
    models = []
    for candidate in audit_input.frozen_candidates:
        rows = []
        for case in audit_input.cases:
            history = case.scenario.payload.get("history")
            latest = history[-1] if isinstance(history, tuple) and history else None
            previous_action = (
                latest.get("first_stage_action")
                if hasattr(latest, "get")
                else "action_0"
            )
            if previous_action not in ("action_0", "action_1"):
                previous_action = "action_0"
            stay_probability = _expected_stay_probability(case)
            action_0_probability = (
                stay_probability
                if previous_action == "action_0"
                else 1.0 - stay_probability
            )
            for seed in seeds_by_role[case.role]:
                rows.append(
                    MeasurementSeedPrediction(
                        case_hash=case.case_hash,
                        scenario_hash=case.scenario.content_hash,
                        role=case.role,
                        model_name=candidate.name,
                        seed=seed,
                        metrics=(
                            ("first_stage.action_0", action_0_probability),
                            (
                                "first_stage.action_1",
                                1.0 - action_0_probability,
                            ),
                        ),
                        run_manifest_hash=_hash(
                            candidate.name,
                            case.case_hash,
                            seed,
                            "run",
                        ),
                    )
                )
        models.append(
            MeasurementModelPrediction(
                model_name=candidate.name,
                candidate_hash=candidate.content_hash,
                rows=tuple(rows),
            )
        )
    return MeasurementPredictionArtifact(
        protocol_hash=protocol.content_hash,
        audit_input_hash=audit_input.content_hash,
        models=tuple(models),
        execution_manifest_hashes=tuple(
            row.run_manifest_hash for model in models for row in model.rows
        ),
    )


def _diagnostic(rows, *, task: str, model: str) -> StaySwitchDiagnostic:
    matches = tuple(
        row for row in rows if row.task_variant == task and row.model_name == model
    )
    if len(matches) != 1:
        raise AssertionError(f"expected one diagnostic, got {len(matches)}")
    return matches[0]


class MeasurementStaySwitchDiagnosticTests(unittest.TestCase):
    def test_interaction_uses_previous_reward_transition_and_current_choice(self) -> None:
        audit_input = measurement_input(cases=_diagnostic_cases())
        artifact = _artifact(audit_input)
        diagnostics = build_feher_hare_stay_switch_diagnostics(
            audit_input,
            artifact,
        )
        self.assertEqual(len(diagnostics), 6)
        self.assertTrue(
            all(isinstance(row, StaySwitchDiagnostic) for row in diagnostics)
        )

        magic = _diagnostic(
            diagnostics,
            task="magic_carpet",
            model="planning",
        )
        self.assertIs(
            magic.status,
            MeasurementValidityStatus.STABLE_UNDER_FROZEN_AUDIT,
        )
        self.assertEqual(magic.observed_interaction, 0.5)
        self.assertEqual(magic.model_interaction, 0.5)
        self.assertEqual(len(magic.cells), 4)
        by_cell = {(cell.reward, cell.transition_common): cell for cell in magic.cells}
        self.assertEqual(by_cell[(1, True)].observed_stay_probability, 1.0)
        self.assertEqual(by_cell[(0, True)].observed_stay_probability, 0.0)
        self.assertEqual(by_cell[(1, False)].observed_stay_probability, 1.0)
        self.assertEqual(by_cell[(0, False)].observed_stay_probability, 0.5)
        self.assertEqual(by_cell[(0, False)].count, 2)
        self.assertEqual(by_cell[(1, True)].model_expected_stay_probability, 0.8)
        self.assertTrue(all(isinstance(cell, StaySwitchCell) for cell in magic.cells))

    def test_participant_boundaries_first_trials_and_trial_gaps_are_respected(self) -> None:
        audit_input = measurement_input(cases=tuple(reversed(_diagnostic_cases())))
        diagnostics = build_feher_hare_stay_switch_diagnostics(
            audit_input,
            _artifact(audit_input),
        )
        magic = _diagnostic(
            diagnostics,
            task="magic_carpet",
            model="reactive",
        )
        self.assertEqual(sum(cell.count for cell in magic.cells), 5)
        self.assertEqual(
            diagnostics,
            build_feher_hare_stay_switch_diagnostics(
                audit_input,
                _artifact(audit_input),
            ),
        )

    def test_empty_cell_is_emitted_and_summary_is_not_established(self) -> None:
        audit_input = measurement_input(cases=_diagnostic_cases())
        diagnostics = build_feher_hare_stay_switch_diagnostics(
            audit_input,
            _artifact(audit_input),
        )
        spaceship = _diagnostic(
            diagnostics,
            task="spaceship",
            model="intentional",
        )
        self.assertEqual(len(spaceship.cells), 4)
        empty = next(
            cell
            for cell in spaceship.cells
            if (cell.reward, cell.transition_common) == (0, False)
        )
        self.assertEqual(empty.count, 0)
        self.assertIsNone(empty.observed_stay_probability)
        self.assertIsNone(empty.model_expected_stay_probability)
        self.assertIs(
            empty.status,
            MeasurementValidityStatus.NOT_ESTABLISHED,
        )
        self.assertIsNone(spaceship.observed_interaction)
        self.assertIsNone(spaceship.model_interaction)
        self.assertIs(
            spaceship.status,
            MeasurementValidityStatus.NOT_ESTABLISHED,
        )

    def test_malformed_retained_history_fails_closed(self) -> None:
        cases = _diagnostic_cases()
        target_index = next(
            index
            for index, case in enumerate(cases)
            if case.task_variant == "magic_carpet" and case.trial_index == 2
        )
        malformed_histories = (
            "not-a-sequence",
            ("not-a-mapping",),
            ({"trial_id": 0, "reward": 1, "transition_common": True},),
            (
                {
                    "trial_id": 0,
                    "first_stage_action": "action_0",
                    "reward": 2,
                    "transition_common": True,
                },
            ),
            (
                {
                    "trial_id": 0,
                    "first_stage_action": "action_0",
                    "reward": 1,
                    "transition_common": 1,
                },
            ),
        )
        for malformed in malformed_histories:
            with self.subTest(malformed=malformed):
                original = cases[target_index]
                changed = replace(
                    original,
                    scenario=Scenario(
                        id=original.scenario.id,
                        payload={
                            **dict(original.scenario.payload),
                            "history": malformed,
                        },
                    ),
                )
                changed_cases = (
                    cases[:target_index] + (changed,) + cases[target_index + 1 :]
                )
                audit_input = measurement_input(cases=changed_cases)
                with self.assertRaises((TypeError, ValueError)):
                    build_feher_hare_stay_switch_diagnostics(
                        audit_input,
                        _artifact(audit_input),
                    )

    def test_api_consumes_only_frozen_input_and_predictions(self) -> None:
        signature = inspect.signature(build_feher_hare_stay_switch_diagnostics)
        self.assertEqual(
            tuple(signature.parameters),
            ("audit_input", "prediction_artifact"),
        )
        forbidden = (
            "runner",
            "executor",
            "parameter",
            "candidate",
            "threshold",
        )
        self.assertTrue(
            all(
                token not in parameter.lower()
                for parameter in signature.parameters
                for token in forbidden
            )
        )


if __name__ == "__main__":
    unittest.main()
