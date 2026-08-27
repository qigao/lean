from __future__ import annotations

import inspect
import unittest

from narrative_dynamics.narrative.domain import StateDelta, StateDeltaOp
from narrative_dynamics.narrative.ir import TypedValue
from narrative_dynamics.narrative.observation_projection import (
    ObservationCapabilitySpec,
    ObservationFact,
    ObservationProjectionModelSpec,
)
from narrative_dynamics.narrative.runtime_decision_dispatch import (
    RuntimeDecisionModelSpec,
)
from narrative_dynamics.narrative.simulation import (
    RuntimeAgentSpec,
    SimulationModelSpec,
    simulate_step,
    simulate_trajectory,
    simulation_state_from_story,
)
from narrative_dynamics.narrative.world import (
    ActionEffectSpec,
    ActionTransitionSpec,
    WorldTransitionModelSpec,
)
from tests.test_narrative_simulation import (
    ResponseTransition,
    _service_ref,
    alert_cell,
    make_intentional_models,
    make_scheduler_domain,
    make_scheduler_story,
)

_STOCHASTIC_SIMULATION_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.observation_projection import (
        ObservationOutcome,
        ObservationOutcomeDistribution,
        StochasticObserverProjectionSpec,
    )
    from narrative_dynamics.narrative.world import (
        StateDeltaDistribution,
        StateDeltaOutcome,
        StochasticActionTransitionSpec,
    )
except ImportError as error:
    _STOCHASTIC_SIMULATION_IMPORT_ERROR = error


class StochasticAlertTransition:
    def __call__(self, snapshot, decision, action, parameters):
        p = parameters["success_probability"]
        if action.arguments["raise"].value is False:
            return StateDeltaDistribution(
                (
                    StateDeltaOutcome("noop-a", 0.5, StateDelta(())),
                    StateDeltaOutcome("noop-b", 0.5, StateDelta(())),
                )
            )
        return StateDeltaDistribution(
            (
                StateDeltaOutcome("fail", 1.0 - p, StateDelta(())),
                StateDeltaOutcome(
                    "succeed",
                    p,
                    StateDelta(
                        (
                            StateDeltaOp(
                                "set",
                                "svc",
                                "service.alert",
                                TypedValue("AlertState", True),
                            ),
                        )
                    ),
                ),
            )
        )


class StochasticAlertProjection:
    def __call__(
        self,
        prior_visible,
        next_visible,
        observer,
        step_index,
        parameters,
    ):
        if observer.id != "a2":
            return ObservationOutcomeDistribution(
                (
                    ObservationOutcome("drop-a", 0.5, ()),
                    ObservationOutcome("drop-b", 0.5, ()),
                )
            )
        value = next_visible.get(alert_cell())
        facts = (
            ()
            if value is None
            else (ObservationFact(alert_cell(), "equals", value),)
        )
        p = parameters["emit_probability"]
        return ObservationOutcomeDistribution(
            (
                ObservationOutcome("drop", 1.0 - p, ()),
                ObservationOutcome("emit", p, facts),
            )
        )


def _stochastic_case():
    domain = make_scheduler_domain()
    story = make_scheduler_story(domain)
    a_model, b_model = make_intentional_models()
    agents = (
        RuntimeAgentSpec(
            "a1",
            "d-a1-scheduler",
            RuntimeDecisionModelSpec("intentional", a_model),
        ),
        RuntimeAgentSpec(
            "a2",
            "d-a2-scheduler",
            RuntimeDecisionModelSpec("intentional", b_model),
        ),
    )
    world = WorldTransitionModelSpec(
        "stochastic-scheduler-world",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        (
            ActionTransitionSpec(
                "response-action",
                (ActionEffectSpec("agent.phase", "actor"),),
                ResponseTransition(),
            ),
        ),
        None,
        (
            StochasticActionTransitionSpec(
                "alert-control-action",
                (
                    ActionEffectSpec(
                        "service.alert",
                        "argument",
                        "service",
                    ),
                ),
                {"success_probability": 0.5},
                StochasticAlertTransition(),
            ),
        ),
    )
    capability = ObservationCapabilitySpec("service.alert", "any")
    observation = ObservationProjectionModelSpec(
        "stochastic-scheduler-projection",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        (),
        stochastic_projections=(
            StochasticObserverProjectionSpec(
                "Agent",
                "vision",
                (capability,),
                (capability,),
                {"emit_probability": 0.5},
                StochasticAlertProjection(),
            ),
        ),
    )
    model = SimulationModelSpec(
        "stochastic-scheduler-model",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        agents,
        world,
        observation,
    )
    return domain, story, model


def _agent_step(result, agent_id: str):
    return next(item for item in result.agent_steps if item.agent_id == agent_id)


def _a2_evidence_semantics(step):
    return tuple(
        (
            item.channel,
            item.cell.to_dict(),
            item.relation,
            None if item.value is None else item.value.to_dict(),
        )
        for item in step.admission_result.evidence_batch.evidence
        if item.observer_id == "a2"
    )


class NarrativeStochasticSimulationTests(unittest.TestCase):
    def require_stochastic_simulation(self) -> None:
        if _STOCHASTIC_SIMULATION_IMPORT_ERROR is not None:
            self.fail(
                "stochastic simulation API is missing: "
                f"{_STOCHASTIC_SIMULATION_IMPORT_ERROR}"
            )

    def test_seeded_initialization_binds_world_and_ledger_to_same_initial_hash(self):
        self.require_stochastic_simulation()
        domain, story, model = _stochastic_case()
        state = simulation_state_from_story(
            story,
            domain,
            model,
            seed=101,
        )
        self.assertEqual(state.world_state.root_seed, 101)
        self.assertEqual(
            state.evidence_ledger.initial_world_state_hash,
            state.world_state.content_hash,
        )
        self.assertEqual(
            state.evidence_ledger.current_world_state_hash,
            state.world_state.content_hash,
        )

    def test_same_seed_replays_exact_multi_round_trajectory(self):
        self.require_stochastic_simulation()
        domain, story, model = _stochastic_case()
        first_state = simulation_state_from_story(
            story,
            domain,
            model,
            seed=101,
        )
        second_state = simulation_state_from_story(
            story,
            domain,
            model,
            seed=101,
        )
        first = simulate_trajectory(
            story,
            domain,
            first_state,
            model,
            rounds=2,
        )
        second = simulate_trajectory(
            story,
            domain,
            second_state,
            model,
            rounds=2,
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.content_hash, second.content_hash)

    def test_different_root_seed_changes_lineage_even_when_extensional_outcome_matches(self):
        self.require_stochastic_simulation()
        domain, story, model = _stochastic_case()
        by_outcome = {}
        pair = None
        for seed in range(256):
            initial = simulation_state_from_story(
                story,
                domain,
                model,
                seed=seed,
            )
            step = simulate_step(story, domain, initial, model)
            record = next(
                item
                for item in step.world_step.transitions
                if item.actor_id == "a1"
            )
            self.assertIsNotNone(record.stochastic_sample)
            outcome = record.stochastic_sample.sample_record.selected_outcome_id
            previous = by_outcome.get(outcome)
            if previous is not None:
                pair = (previous, (seed, step, record))
                break
            by_outcome[outcome] = (seed, step, record)
        self.assertIsNotNone(pair)
        assert pair is not None
        (seed_a, step_a, record_a), (seed_b, step_b, record_b) = pair
        self.assertNotEqual(seed_a, seed_b)
        self.assertEqual(
            record_a.stochastic_sample.sample_record.selected_outcome_id,
            record_b.stochastic_sample.sample_record.selected_outcome_id,
        )
        self.assertNotEqual(
            record_a.stochastic_sample.content_hash,
            record_b.stochastic_sample.content_hash,
        )
        self.assertNotEqual(
            step_a.world_step.next_state.content_hash,
            step_b.world_step.next_state.content_hash,
        )

    def test_step_and_trajectory_signatures_do_not_accept_replacement_seed(self):
        self.require_stochastic_simulation()
        self.assertNotIn("seed", inspect.signature(simulate_step).parameters)
        self.assertNotIn("seed", inspect.signature(simulate_trajectory).parameters)
        parameter = inspect.signature(simulation_state_from_story).parameters["seed"]
        self.assertIsNone(parameter.default)
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_decisions_are_seed_invariant_until_stochastic_history_diverges(self):
        self.require_stochastic_simulation()
        domain, story, model = _stochastic_case()
        first_initial = simulation_state_from_story(
            story,
            domain,
            model,
            seed=1,
        )
        second_initial = simulation_state_from_story(
            story,
            domain,
            model,
            seed=2,
        )
        first_step = simulate_step(story, domain, first_initial, model)
        second_step = simulate_step(story, domain, second_initial, model)
        for agent_id in ("a1", "a2"):
            left = _agent_step(first_step, agent_id).decision_result
            right = _agent_step(second_step, agent_id).decision_result
            self.assertEqual(left.action_policy, right.action_policy)
            self.assertEqual(left.selected_action, right.selected_action)

        branches = []
        for seed in range(256):
            initial = simulation_state_from_story(
                story,
                domain,
                model,
                seed=seed,
            )
            step1 = simulate_step(story, domain, initial, model)
            step2 = simulate_step(story, domain, step1.next_state, model)
            branches.append(
                (
                    _a2_evidence_semantics(step1),
                    _agent_step(step2, "a2").decision_result.action_policy,
                )
            )
        found = False
        for left_index, left in enumerate(branches):
            for right in branches[left_index + 1 :]:
                if left[0] != right[0] and left[1] != right[1]:
                    found = True
                    break
            if found:
                break
        self.assertTrue(found)

    def test_world_and_observation_namespaces_are_distinct(self):
        self.require_stochastic_simulation()
        domain, story, model = _stochastic_case()
        initial = simulation_state_from_story(
            story,
            domain,
            model,
            seed=77,
        )
        step = simulate_step(story, domain, initial, model)
        world_record = next(
            item
            for item in step.world_step.transitions
            if item.stochastic_sample is not None
        )
        observation_sample = next(
            item
            for item in step.admission_result.projection_result.stochastic_samples
            if item.observer_id == "a2"
        )
        world_sample = world_record.stochastic_sample.sample_record
        projection_sample = observation_sample.sample_record
        self.assertEqual(world_sample.namespace, "world.transition")
        self.assertEqual(
            projection_sample.namespace,
            "observation.projection",
        )
        self.assertNotEqual(world_sample.stream_hash, projection_sample.stream_hash)


if __name__ == "__main__":
    unittest.main()
