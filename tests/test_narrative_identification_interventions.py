from __future__ import annotations

from dataclasses import replace
import unittest

from narrative_dynamics.adapters.prison_metrics import prison_initial_action_metrics
from narrative_dynamics.contracts import Scenario
from narrative_dynamics.simulation import SimulationRunner

from tests.test_narrative_identification_protocol import (
    FINAL_SEEDS,
    TRUTH,
    make_pair,
    require_identification,
)

_IDENTIFICATION_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.identification import (
        InterventionCertificationError,
        certify_information_intervention,
        evaluate_information_intervention,
    )
except ImportError as error:
    _IDENTIFICATION_IMPORT_ERROR = error
    InterventionCertificationError = None
    certify_information_intervention = None
    evaluate_information_intervention = None

_ADAPTER_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.adapters.narrative_prison_identification import (
        create_narrative_identification_intentional_source,
        create_narrative_identification_planning_source,
        create_narrative_identification_reactive_source,
    )
except ImportError as error:
    _ADAPTER_IMPORT_ERROR = error
    create_narrative_identification_intentional_source = None
    create_narrative_identification_planning_source = None
    create_narrative_identification_reactive_source = None


METRIC_KEYS = ("initial.scout", "initial.escape", "initial.submit")


def require_adapter(test: unittest.TestCase) -> None:
    if _ADAPTER_IMPORT_ERROR is not None:
        test.fail(f"synthetic identification adapter is missing: {_ADAPTER_IMPORT_ERROR}")


def sources():
    return {
        "reactive": create_narrative_identification_reactive_source(),
        "intentional": create_narrative_identification_intentional_source(),
        "planning": create_narrative_identification_planning_source(),
    }


def family_parameters(name: str):
    if name == "intentional":
        return dict(TRUTH)
    return {"beta_action": TRUTH["beta_action"]}


def memory_pair():
    return make_pair()


def future_pair():
    return make_pair(
        "final-future-off",
        "final-future-on",
        allowed_key="future_information_available",
        expected_relationship="planning_vs_intentional",
    )


def replace_scenario_payload(scenario: Scenario, **changes) -> Scenario:
    payload = dict(scenario.payload)
    payload.update(changes)
    return Scenario(scenario.id + "-drift", payload)


class NarrativeIdentificationInterventionTests(unittest.TestCase):
    def setUp(self):
        require_identification(self)
        require_adapter(self)

    def test_memory_pair_diff_is_exactly_memory_availability(self):
        diff = certify_information_intervention(memory_pair())
        self.assertEqual(diff[0], ("memory_evidence_available",))
        self.assertEqual(diff[1:], (False, True))

    def test_future_pair_diff_is_exactly_future_information_availability(self):
        diff = certify_information_intervention(future_pair())
        self.assertEqual(diff[0], ("future_information_available",))
        self.assertEqual(diff[1:], (False, True))

    def test_reward_drift_is_rejected(self):
        pair = memory_pair()
        intervention = replace_scenario_payload(
            pair.intervention,
            escape_reward=float(pair.intervention.payload["escape_reward"]) + 1.0,
        )
        with self.assertRaises(InterventionCertificationError):
            certify_information_intervention(replace(pair, intervention=intervention))

    def test_action_semantic_drift_is_rejected(self):
        pair = memory_pair()
        intervention = replace_scenario_payload(
            pair.intervention,
            latent_structure="mirror",
        )
        with self.assertRaises(InterventionCertificationError):
            certify_information_intervention(replace(pair, intervention=intervention))

    def test_multiple_changed_information_fields_are_rejected(self):
        pair = memory_pair()
        intervention = replace_scenario_payload(
            pair.intervention,
            memory_signal="strong",
        )
        with self.assertRaises(InterventionCertificationError):
            certify_information_intervention(replace(pair, intervention=intervention))

    def test_memory_evidence_intervention_separates_intentional_from_reactive(self):
        source_map = sources()
        finding = evaluate_information_intervention(
            pair=memory_pair(),
            runner=SimulationRunner(),
            families=(
                ("reactive", source_map["reactive"], family_parameters("reactive")),
                ("intentional", source_map["intentional"], family_parameters("intentional")),
            ),
            seeds=FINAL_SEEDS,
            extractor=prison_initial_action_metrics,
            action_keys=METRIC_KEYS,
            tolerance=1e-12,
        )
        deltas = dict(finding.within_family_max_deltas)
        self.assertTrue(finding.discriminating)
        self.assertLessEqual(deltas["reactive"], 1e-12)
        self.assertGreater(deltas["intentional"], 1e-12)

    def test_future_information_intervention_separates_planning_from_intentional(self):
        source_map = sources()
        finding = evaluate_information_intervention(
            pair=future_pair(),
            runner=SimulationRunner(),
            families=(
                ("intentional", source_map["intentional"], family_parameters("intentional")),
                ("planning", source_map["planning"], family_parameters("planning")),
            ),
            seeds=FINAL_SEEDS,
            extractor=prison_initial_action_metrics,
            action_keys=METRIC_KEYS,
            tolerance=1e-12,
        )
        deltas = dict(finding.within_family_max_deltas)
        self.assertTrue(finding.discriminating)
        self.assertLessEqual(deltas["intentional"], 1e-12)
        self.assertGreater(deltas["planning"], 1e-12)

    def test_intervention_finding_replays_and_canonicalizes_family_order(self):
        source_map = sources()
        first = evaluate_information_intervention(
            pair=memory_pair(),
            runner=SimulationRunner(),
            families=(
                ("reactive", source_map["reactive"], family_parameters("reactive")),
                ("intentional", source_map["intentional"], family_parameters("intentional")),
            ),
            seeds=FINAL_SEEDS,
            extractor=prison_initial_action_metrics,
            action_keys=METRIC_KEYS,
            tolerance=1e-12,
        )
        second = evaluate_information_intervention(
            pair=memory_pair(),
            runner=SimulationRunner(),
            families=(
                ("intentional", source_map["intentional"], family_parameters("intentional")),
                ("reactive", source_map["reactive"], family_parameters("reactive")),
            ),
            seeds=FINAL_SEEDS,
            extractor=prison_initial_action_metrics,
            action_keys=METRIC_KEYS,
            tolerance=1e-12,
        )
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(first.family_policies, second.family_policies)


if __name__ == "__main__":
    unittest.main()
