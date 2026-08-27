from __future__ import annotations

from dataclasses import replace
import unittest

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.conflict import ConflictResolverSpec
from narrative_dynamics.narrative.domain import StateDelta, StateDeltaOp
from narrative_dynamics.narrative.ir import EntityRef, TypedValue
from narrative_dynamics.narrative.world import (
    ActionEffectSpec,
    ActionTransitionSpec,
    WorldTransitionError,
    WorldTransitionModelSpec,
    advance_world_step,
    world_state_from_story,
)
from tests.test_narrative_conflict_resolution import (
    ClaimResolver,
    make_conflict_domain,
    make_conflict_story,
)
from tests.test_narrative_world_transition import (
    ActorPhaseTransitionHook,
    _agent_cell,
    intent,
    make_transition_model,
    make_world_domain,
    make_world_story,
)

_WORLD_STOCHASTIC_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.world import (
        StateDeltaDistribution,
        StateDeltaOutcome,
        StochasticActionTransitionSpec,
        StochasticTransitionSample,
    )
except ImportError as error:
    _WORLD_STOCHASTIC_IMPORT_ERROR = error


class StochasticPhaseHook:
    def __init__(self):
        self.calls = []

    def __call__(self, snapshot, decision, action, parameters):
        self.calls.append(
            (dict(snapshot), decision.id, action.id, dict(parameters))
        )
        p = parameters["ready_probability"]
        return StateDeltaDistribution(
            (
                StateDeltaOutcome(
                    "done",
                    1.0 - p,
                    StateDelta(
                        (
                            StateDeltaOp(
                                "set",
                                decision.actor_id,
                                "agent.phase",
                                TypedValue("PhaseState", "done"),
                            ),
                        )
                    ),
                ),
                StateDeltaOutcome(
                    "ready",
                    p,
                    StateDelta(
                        (
                            StateDeltaOp(
                                "set",
                                decision.actor_id,
                                "agent.phase",
                                TypedValue("PhaseState", "ready"),
                            ),
                        )
                    ),
                ),
            )
        )


class InvalidCandidateHook:
    def __call__(self, snapshot, decision, action, parameters):
        return StateDeltaDistribution(
            (
                StateDeltaOutcome(
                    "valid",
                    1.0 - 1e-12,
                    StateDelta(
                        (
                            StateDeltaOp(
                                "set",
                                decision.actor_id,
                                "agent.phase",
                                TypedValue("PhaseState", "ready"),
                            ),
                        )
                    ),
                ),
                StateDeltaOutcome(
                    "invalid",
                    1e-12,
                    StateDelta(
                        (
                            StateDeltaOp(
                                "set",
                                "svc",
                                "service.health",
                                TypedValue("HealthState", "healthy"),
                            ),
                        )
                    ),
                ),
            )
        )


class StochasticClaimHook:
    def __call__(self, snapshot, decision, action, parameters):
        service_id = action.arguments["service"].value.entity_id
        owner = TypedValue("AgentRef", EntityRef(decision.actor_id, "Agent"))
        p = parameters["write_probability"]
        return StateDeltaDistribution(
            (
                StateDeltaOutcome("noop", 1.0 - p, StateDelta(())),
                StateDeltaOutcome(
                    "write",
                    p,
                    StateDelta(
                        (
                            StateDeltaOp(
                                "set",
                                service_id,
                                "service.owner",
                                owner,
                            ),
                        )
                    ),
                ),
            )
        )


def _stochastic_phase_model(domain, hook=None):
    if hook is None:
        hook = StochasticPhaseHook()
    return WorldTransitionModelSpec(
        "stochastic-world",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        (),
        None,
        (
            StochasticActionTransitionSpec(
                "actor-phase-action",
                (ActionEffectSpec("agent.phase", "actor"),),
                {"ready_probability": 0.5},
                hook,
            ),
        ),
    )


def _selected_outcome(result, actor_id: str) -> str:
    record = next(item for item in result.transitions if item.actor_id == actor_id)
    assert record.stochastic_sample is not None
    return record.stochastic_sample.sample_record.selected_outcome_id


class NarrativeStochasticWorldTests(unittest.TestCase):
    def require_stochastic_world(self) -> None:
        if _WORLD_STOCHASTIC_IMPORT_ERROR is not None:
            self.fail(
                "stochastic world API is missing: "
                f"{_WORLD_STOCHASTIC_IMPORT_ERROR}"
            )

    def test_deterministic_world_payloads_and_hashes_remain_v1_shaped(self):
        self.require_stochastic_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain)
        model = make_transition_model()
        result = advance_world_step(
            story,
            domain,
            prior,
            model,
            (intent("d-bob-phase", "bob-ready"),),
        )

        model_payload = model.to_dict()
        self.assertNotIn("stochastic_transitions", model_payload)
        self.assertEqual(
            set(model_payload),
            {
                "model_id",
                "version",
                "domain_id",
                "domain_version",
                "domain_spec_hash",
                "transitions",
            },
        )
        self.assertEqual(model.content_hash, stable_content_hash(model_payload))

        prior_payload = prior.to_dict()
        self.assertNotIn("root_seed", prior_payload)
        self.assertEqual(
            set(prior_payload),
            {
                "domain_id",
                "domain_version",
                "domain_spec_hash",
                "source_story_hash",
                "source_at_time",
                "step_index",
                "parent_state_hash",
                "transition_batch_hash",
                "values",
            },
        )
        self.assertEqual(prior.content_hash, stable_content_hash(prior_payload))

        record_payload = result.transitions[0].to_dict()
        self.assertNotIn("stochastic_sample", record_payload)
        self.assertEqual(
            set(record_payload),
            {
                "intent",
                "actor_id",
                "action",
                "transition_spec_hash",
                "prior_state_hash",
                "delta",
            },
        )
        self.assertEqual(
            result.transitions[0].content_hash,
            stable_content_hash(record_payload),
        )

    def test_same_seed_replays_stochastic_transition_exactly(self):
        self.require_stochastic_world()
        story, domain = make_world_story(), make_world_domain()
        model = _stochastic_phase_model(domain)
        first_prior = world_state_from_story(story, domain, seed=17)
        second_prior = world_state_from_story(story, domain, seed=17)
        first = advance_world_step(
            story,
            domain,
            first_prior,
            model,
            (intent("d-bob-phase", "bob-ready"),),
        )
        second = advance_world_step(
            story,
            domain,
            second_prior,
            model,
            (intent("d-bob-phase", "bob-ready"),),
        )
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(first.next_state.content_hash, second.next_state.content_hash)

    def test_actor_order_and_unrelated_actor_do_not_perturb_component_sample(self):
        self.require_stochastic_world()
        story, domain = make_world_story(), make_world_domain()
        model = _stochastic_phase_model(domain)
        prior = world_state_from_story(story, domain, seed=91)
        bob = intent("d-bob-phase", "bob-ready")
        alice = intent("d-alice-phase", "alice-done", marker="2")
        first = advance_world_step(story, domain, prior, model, (bob, alice))
        second = advance_world_step(story, domain, prior, model, (alice, bob))
        third = advance_world_step(story, domain, prior, model, (bob,))

        def sample(result):
            record = next(item for item in result.transitions if item.actor_id == "bob")
            self.assertIsInstance(record.stochastic_sample, StochasticTransitionSample)
            return record.stochastic_sample.to_dict()

        self.assertEqual(sample(first), sample(second))
        self.assertEqual(sample(first), sample(third))

    def test_multiple_seeds_exercise_declared_world_outcomes(self):
        self.require_stochastic_world()
        story, domain = make_world_story(), make_world_domain()
        model = _stochastic_phase_model(domain)
        observed = set()
        for seed in range(256):
            prior = world_state_from_story(story, domain, seed=seed)
            result = advance_world_step(
                story,
                domain,
                prior,
                model,
                (intent("d-bob-phase", "bob-ready"),),
            )
            observed.add(_selected_outcome(result, "bob"))
            if observed == {"done", "ready"}:
                break
        self.assertEqual(observed, {"done", "ready"})

    def test_every_candidate_delta_is_validated_before_sampling(self):
        self.require_stochastic_world()
        story, domain = make_world_story(), make_world_domain()
        prior = world_state_from_story(story, domain, seed=1)
        model = _stochastic_phase_model(domain, hook=InvalidCandidateHook())
        with self.assertRaises(WorldTransitionError):
            advance_world_step(
                story,
                domain,
                prior,
                model,
                (intent("d-bob-phase", "bob-ready"),),
            )

    def test_stochastic_action_requires_seed_before_hook(self):
        self.require_stochastic_world()
        story, domain = make_world_story(), make_world_domain()
        hook = StochasticPhaseHook()
        model = _stochastic_phase_model(domain, hook=hook)
        prior = world_state_from_story(story, domain)
        with self.assertRaises(WorldTransitionError):
            advance_world_step(
                story,
                domain,
                prior,
                model,
                (intent("d-bob-phase", "bob-ready"),),
            )
        self.assertEqual(hook.calls, [])

    def test_realized_stochastic_writes_feed_existing_conflict_resolution(self):
        self.require_stochastic_world()
        domain = make_conflict_domain()
        story = make_conflict_story(domain)
        resolver_hook = ClaimResolver()
        resolver = ConflictResolverSpec(
            "claim-resolver",
            "1",
            domain.domain_id,
            domain.version,
            domain.content_hash,
            ("claim-service-action",),
            resolver_hook,
        )
        model = WorldTransitionModelSpec(
            "stochastic-claim-world",
            "1",
            domain.domain_id,
            domain.version,
            domain.content_hash,
            (),
            resolver,
            (
                StochasticActionTransitionSpec(
                    "claim-service-action",
                    (ActionEffectSpec("service.owner", "argument", "service"),),
                    {"write_probability": 0.5},
                    StochasticClaimHook(),
                ),
            ),
        )
        intents = (
            intent("d-alice-claim-svc", "alice-claim-svc"),
            intent("d-bob-claim-svc", "bob-claim-svc", marker="2"),
        )
        chosen = None
        for seed in range(1024):
            prior = world_state_from_story(story, domain, seed=seed)
            result = advance_world_step(story, domain, prior, model, intents)
            if len(result.conflict_resolutions) == 1:
                chosen = result
                break
        self.assertIsNotNone(chosen)
        assert chosen is not None
        self.assertEqual(len(resolver_hook.calls), 1)
        self.assertEqual(len(chosen.conflict_resolutions), 1)
        self.assertEqual(
            chosen.conflict_resolutions[0].context.conflict_cells[0].state_variable,
            "service.owner",
        )


if __name__ == "__main__":
    unittest.main()
