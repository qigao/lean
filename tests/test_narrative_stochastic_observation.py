from __future__ import annotations

import unittest

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.ir import TypedValue
from narrative_dynamics.narrative.observation_projection import (
    ObservationCapabilitySpec,
    ObservationFact,
    ObservationProjectionError,
    ObservationProjectionModelSpec,
    ObserverProjectionSpec,
    project_world_observations,
)
from narrative_dynamics.narrative.runtime_perception import (
    admit_world_percepts,
    runtime_evidence_ledger_from_story,
)
from narrative_dynamics.narrative.world import (
    ActionIntent,
    advance_world_step,
    world_state_from_story,
)
from tests.test_narrative_observation_projection import (
    OwnLocationHook,
    _cell,
    _spec,
    make_projection_domain,
    make_projection_story,
    make_transition_model,
    make_world_step,
    projection_model,
)

_OBSERVATION_STOCHASTIC_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.observation_projection import (
        ObservationOutcome,
        ObservationOutcomeDistribution,
        StochasticObservationSample,
        StochasticObserverProjectionSpec,
    )
except ImportError as error:
    _OBSERVATION_STOCHASTIC_IMPORT_ERROR = error


class StochasticRoomLightingHook:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        prior_visible,
        next_visible,
        observer,
        step_index,
        parameters,
    ):
        self.calls.append((observer.id, step_index, dict(parameters)))
        cell = _cell(observer.id, observer.type_name, "room.lighting")
        fact = ObservationFact(cell, "equals", next_visible[cell])
        p = parameters["emit_probability"]
        return ObservationOutcomeDistribution(
            (
                ObservationOutcome("drop", 1.0 - p, ()),
                ObservationOutcome("emit", p, (fact,)),
            )
        )


class InvalidLightingCandidateHook:
    def __call__(
        self,
        prior_visible,
        next_visible,
        observer,
        step_index,
        parameters,
    ):
        cell = _cell(observer.id, observer.type_name, "room.lighting")
        valid = ObservationFact(cell, "equals", next_visible[cell])
        wrong = (
            "bright"
            if next_visible[cell].value == "dark"
            else "dark"
        )
        invalid = ObservationFact(
            cell,
            "equals",
            TypedValue("LightState", wrong),
        )
        return ObservationOutcomeDistribution(
            (
                ObservationOutcome("valid", 1.0 - 1e-12, (valid,)),
                ObservationOutcome("invalid", 1e-12, (invalid,)),
            )
        )


def _seeded_world_step(seed: int | None):
    domain = make_projection_domain()
    story = make_projection_story(domain)
    prior = world_state_from_story(story, domain, seed=seed)
    step = advance_world_step(
        story,
        domain,
        prior,
        make_transition_model(domain),
        (
            ActionIntent(
                "d-a1-phase",
                "a1-active",
                "selection-model",
                "sha256:" + "1" * 64,
            ),
            ActionIntent(
                "d-a2-noop",
                "a2-wait",
                "selection-model",
                "sha256:" + "2" * 64,
            ),
        ),
    )
    return domain, story, step


def _lighting_spec(
    hook,
    *,
    channel: str = "vision",
    emit_probability: float = 0.5,
):
    capability = ObservationCapabilitySpec("room.lighting", "observer")
    return StochasticObserverProjectionSpec(
        "Room",
        channel,
        (capability,),
        (capability,),
        {"emit_probability": emit_probability},
        hook,
    )


def _stochastic_projection_model(domain, *specs):
    return ObservationProjectionModelSpec(
        "stochastic-projection",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        (),
        stochastic_projections=tuple(specs),
    )


def _sample_for(result, observer_id: str, channel: str):
    return next(
        item
        for item in result.stochastic_samples
        if item.observer_id == observer_id and item.channel == channel
    )


class NarrativeStochasticObservationTests(unittest.TestCase):
    def require_stochastic_observation(self) -> None:
        if _OBSERVATION_STOCHASTIC_IMPORT_ERROR is not None:
            self.fail(
                "stochastic observation API is missing: "
                f"{_OBSERVATION_STOCHASTIC_IMPORT_ERROR}"
            )

    def test_deterministic_projection_payloads_and_hashes_remain_v1_shaped(self):
        self.require_stochastic_observation()
        domain, story, world_step = make_world_step()
        capability = ObservationCapabilitySpec("agent.location", "observer")
        model = projection_model(
            domain,
            _spec(
                OwnLocationHook(),
                read=(capability,),
                emit=(capability,),
            ),
        )
        result = project_world_observations(story, domain, world_step, model)
        self.assertNotIn("stochastic_projections", model.to_dict())
        self.assertNotIn("stochastic_samples", result.to_dict())
        self.assertEqual(model.content_hash, stable_content_hash(model.to_dict()))
        self.assertEqual(result.content_hash, stable_content_hash(result.to_dict()))
        for item in result.observations:
            self.assertNotIn("stochastic_sample_hash", item.to_dict())
            self.assertEqual(item.content_hash, stable_content_hash(item.to_dict()))

    def test_same_seed_and_world_step_replay_projection_exactly(self):
        self.require_stochastic_observation()
        domain, story, world_step = _seeded_world_step(17)
        model = _stochastic_projection_model(
            domain,
            _lighting_spec(StochasticRoomLightingHook()),
        )
        first = project_world_observations(story, domain, world_step, model)
        second = project_world_observations(story, domain, world_step, model)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.content_hash, second.content_hash)

    def test_spec_order_and_unrelated_projection_do_not_perturb_existing_sample(self):
        self.require_stochastic_observation()
        domain, story, world_step = _seeded_world_step(23)
        vision = _lighting_spec(StochasticRoomLightingHook(), channel="vision")
        status = _lighting_spec(StochasticRoomLightingHook(), channel="status")
        only = project_world_observations(
            story,
            domain,
            world_step,
            _stochastic_projection_model(domain, vision),
        )
        both = project_world_observations(
            story,
            domain,
            world_step,
            _stochastic_projection_model(domain, vision, status),
        )
        reversed_both = project_world_observations(
            story,
            domain,
            world_step,
            _stochastic_projection_model(domain, status, vision),
        )
        expected = _sample_for(only, "room", "vision").to_dict()
        self.assertEqual(
            _sample_for(both, "room", "vision").to_dict(),
            expected,
        )
        self.assertEqual(
            _sample_for(reversed_both, "room", "vision").to_dict(),
            expected,
        )
        self.assertEqual(both.to_dict(), reversed_both.to_dict())

    def test_seed_scan_exercises_emit_and_dropout(self):
        self.require_stochastic_observation()
        observed = set()
        for seed in range(256):
            domain, story, world_step = _seeded_world_step(seed)
            model = _stochastic_projection_model(
                domain,
                _lighting_spec(StochasticRoomLightingHook()),
            )
            result = project_world_observations(story, domain, world_step, model)
            observed.add(
                _sample_for(
                    result,
                    "room",
                    "vision",
                ).sample_record.selected_outcome_id
            )
            if observed == {"drop", "emit"}:
                break
        self.assertEqual(observed, {"drop", "emit"})

    def test_empty_outcome_keeps_projection_and_evidence_batch_lineage(self):
        self.require_stochastic_observation()
        chosen = None
        for seed in range(256):
            domain, story, world_step = _seeded_world_step(seed)
            model = _stochastic_projection_model(
                domain,
                _lighting_spec(StochasticRoomLightingHook()),
            )
            result = project_world_observations(story, domain, world_step, model)
            sample = _sample_for(result, "room", "vision")
            if sample.sample_record.selected_outcome_id == "drop":
                chosen = (seed, domain, story, world_step, model, result, sample)
                break
        self.assertIsNotNone(chosen)
        assert chosen is not None
        seed, domain, story, world_step, model, result, sample = chosen
        self.assertEqual(result.observations, ())
        self.assertEqual(len(result.stochastic_samples), 1)
        ledger = runtime_evidence_ledger_from_story(story, domain, seed=seed)
        admission = admit_world_percepts(
            story,
            domain,
            world_step,
            model,
            ledger,
        )
        self.assertEqual(admission.evidence_batch.evidence, ())
        self.assertEqual(
            admission.evidence_batch.projection_sample_hashes,
            (sample.content_hash,),
        )

    def test_every_candidate_fact_is_validated_before_sampling(self):
        self.require_stochastic_observation()
        domain, story, world_step = _seeded_world_step(31)
        model = _stochastic_projection_model(
            domain,
            _lighting_spec(InvalidLightingCandidateHook()),
        )
        with self.assertRaises(ObservationProjectionError):
            project_world_observations(story, domain, world_step, model)

    def test_stochastic_projection_requires_seed_before_hook(self):
        self.require_stochastic_observation()
        domain, story, world_step = _seeded_world_step(None)
        hook = StochasticRoomLightingHook()
        model = _stochastic_projection_model(
            domain,
            _lighting_spec(hook),
        )
        with self.assertRaises(ObservationProjectionError):
            project_world_observations(story, domain, world_step, model)
        self.assertEqual(hook.calls, [])


if __name__ == "__main__":
    unittest.main()
