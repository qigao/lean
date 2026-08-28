from __future__ import annotations

import inspect
import unittest

from narrative_dynamics.contracts import Scenario
from narrative_dynamics.manifest import component_identity
from narrative_dynamics.simulation import SimulationRunner
from tests.two_stage_test_support import canonical_history

_ADAPTER_IMPORT_ERROR: Exception | None = None
try:
    import narrative_dynamics.adapters.narrative_two_stage as two_stage_module
    from narrative_dynamics.adapters.narrative_two_stage import (
        NarrativeTwoStageModelSource,
        create_narrative_two_stage_intentional_source,
        create_narrative_two_stage_planning_source,
        create_narrative_two_stage_reactive_source,
    )
except Exception as error:
    _ADAPTER_IMPORT_ERROR = error


def scenario(history=()):
    return Scenario(
        id="two-stage-test",
        payload={
            "task_variant": "magic_carpet",
            "first_stage_configuration": (
                ("action_0", "left"),
                ("action_1", "right"),
            ),
            "history": tuple(
                {"trial_id": index, **item}
                for index, item in enumerate(history)
            ),
        },
    )


class NarrativeTwoStageAdapterTests(unittest.TestCase):
    def require_adapter(self):
        self.assertIsNone(
            _ADAPTER_IMPORT_ERROR,
            f"two-stage narrative adapter is missing: {_ADAPTER_IMPORT_ERROR}",
        )

    def test_three_sources_are_fresh_and_bind_family_identity(self):
        self.require_adapter()
        sources = (
            create_narrative_two_stage_reactive_source(),
            create_narrative_two_stage_intentional_source(),
            create_narrative_two_stage_planning_source(),
        )
        self.assertTrue(all(isinstance(source, NarrativeTwoStageModelSource) for source in sources))
        self.assertEqual(tuple(source.family for source in sources), ("reactive", "intentional", "planning"))
        for source in sources:
            self.assertEqual(source.lifecycle, "fresh_per_batch")
            self.assertIsNot(source.instantiate(), source.instantiate())
            self.assertEqual(component_identity(source)["family"], source.family)

    def test_adapter_uses_unified_dispatch_and_not_observation_or_external_protocol_modules(self):
        self.require_adapter()
        source = inspect.getsource(two_stage_module)
        for banned in (
            "narrative_dynamics.observations",
            "narrative_dynamics.external_validation",
            "PreregisteredEvaluationProtocol",
            "run_runtime_reactive_decision",
            "run_runtime_intentional_decision",
            "run_runtime_planning_decision",
        ):
            self.assertNotIn(banned, source)
        self.assertIn("run_runtime_decision", source)

    def test_reactive_is_neutral_with_empty_history(self):
        self.require_adapter()
        trace = SimulationRunner().run_once(
            create_narrative_two_stage_reactive_source(),
            scenario(),
            {"beta": 2.0},
            seed=1,
        )
        self.assertEqual(trace.outcome["first_stage_policy"], {"action_0": 0.5, "action_1": 0.5})

    def test_reactive_uses_latest_rewarded_stay_and_unrewarded_switch_only(self):
        self.require_adapter()
        runner = SimulationRunner()
        source = create_narrative_two_stage_reactive_source()
        rewarded = runner.run_once(
            source,
            scenario((canonical_history(first_stage_action="action_0", reward=1),)),
            {"beta": 2.0},
            seed=1,
        ).outcome["first_stage_policy"]
        unrewarded = runner.run_once(
            source,
            scenario((canonical_history(first_stage_action="action_0", reward=0),)),
            {"beta": 2.0},
            seed=1,
        ).outcome["first_stage_policy"]
        self.assertGreater(rewarded["action_0"], rewarded["action_1"])
        self.assertLess(unrewarded["action_0"], unrewarded["action_1"])

    def test_reactive_ignores_earlier_history_and_previous_transition_identity(self):
        self.require_adapter()
        runner = SimulationRunner()
        source = create_narrative_two_stage_reactive_source()
        latest = canonical_history(first_stage_action="action_1", reward=1, transition_common=True)
        left = runner.run_once(
            source,
            scenario((canonical_history(first_stage_action="action_0", reward=0), latest)),
            {"beta": 1.0},
            seed=3,
        ).outcome["first_stage_policy"]
        right = runner.run_once(
            source,
            scenario((canonical_history(first_stage_action="action_1", reward=1), {**latest, "transition_common": False})),
            {"beta": 1.0},
            seed=3,
        ).outcome["first_stage_policy"]
        self.assertEqual(left, right)

    def test_intentional_uses_full_reward_history_and_beta1_smoothing_without_planning_hooks(self):
        self.require_adapter()
        runner = SimulationRunner()
        source = create_narrative_two_stage_intentional_source()
        favorable = scenario((
            canonical_history(first_stage_action="action_0", reward=1),
            canonical_history(first_stage_action="action_0", reward=1),
            canonical_history(first_stage_action="action_1", reward=0),
        ))
        unfavorable = scenario((
            canonical_history(first_stage_action="action_0", reward=0),
            canonical_history(first_stage_action="action_0", reward=0),
            canonical_history(first_stage_action="action_1", reward=1),
        ))
        left = runner.run_once(source, favorable, {"beta": 2.0, "memory_decay": 1.0}, seed=1)
        right = runner.run_once(source, unfavorable, {"beta": 2.0, "memory_decay": 1.0}, seed=1)
        self.assertGreater(left.outcome["first_stage_policy"]["action_0"], right.outcome["first_stage_policy"]["action_0"])

    def test_intentional_binds_beta_goal_and_beta_action_to_same_beta(self):
        self.require_adapter()
        trace = SimulationRunner().run_once(
            create_narrative_two_stage_intentional_source(),
            scenario(),
            {"beta": 4.0, "memory_decay": 1.0},
            seed=1,
        )
        self.assertEqual(trace.outcome["runtime_dispatch"]["parameters"]["beta_goal"], 4.0)
        self.assertEqual(trace.outcome["runtime_dispatch"]["parameters"]["beta_action"], 4.0)

    def test_planning_uses_fixed_point_7_transition_and_second_stage_reward_beliefs(self):
        self.require_adapter()
        trace = SimulationRunner().run_once(
            create_narrative_two_stage_planning_source(),
            scenario((
                canonical_history(final_state="state_0", second_stage_action="action_0", reward=1),
                canonical_history(final_state="state_1", second_stage_action="action_0", reward=0),
            )),
            {"beta": 2.0, "memory_decay": 1.0},
            seed=1,
        )
        self.assertEqual(trace.outcome["planning_transition"]["common_probability"], 0.7)
        self.assertEqual(trace.outcome["planning_transition"]["rare_probability"], 0.3)

    def test_planning_discount_is_fixed_one(self):
        self.require_adapter()
        trace = SimulationRunner().run_once(
            create_narrative_two_stage_planning_source(),
            scenario(),
            {"beta": 1.0, "memory_decay": 1.0},
            seed=1,
        )
        self.assertEqual(trace.outcome["planning_discount"], 1.0)

    def test_seed_changes_lineage_but_not_policy_for_deterministic_adapter(self):
        self.require_adapter()
        runner = SimulationRunner()
        source = create_narrative_two_stage_planning_source()
        one = runner.run_once(source, scenario(), {"beta": 1.0, "memory_decay": 1.0}, seed=1)
        two = runner.run_once(source, scenario(), {"beta": 1.0, "memory_decay": 1.0}, seed=2)
        self.assertEqual(one.outcome["first_stage_policy"], two.outcome["first_stage_policy"])
        self.assertNotEqual(one.manifest.content_hash, two.manifest.content_hash)


if __name__ == "__main__":
    unittest.main()
