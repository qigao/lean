from __future__ import annotations

from dataclasses import replace
import unittest

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.ir import Observation, StateCellRef, EntityRef, TypedValue
from narrative_dynamics.narrative.observation_projection import (
    ObservationCapabilitySpec,
    ObservationFact,
    ObservationProjectionError,
    ObservationProjectionModelSpec,
    ObserverProjectionSpec,
)
from narrative_dynamics.narrative.world import (
    ActionIntent,
    advance_world_step,
    world_state_from_story,
)
from tests.test_narrative_observation_projection import (
    _forge,
    make_projection_domain,
    make_projection_story,
    make_transition_model,
    make_second_noop_step,
)

_PERCEPTION_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.runtime_perception import (
        RuntimeEpistemicEvidence,
        RuntimeEvidenceBatch,
        RuntimeEvidenceLedger,
        RuntimePerceptAdmissionError,
        RuntimePerceptAdmissionResult,
        RuntimePerceptView,
        admit_world_percepts,
        runtime_evidence_ledger_from_story,
    )
except ImportError as error:
    _PERCEPTION_IMPORT_ERROR = error


class ObserveAgentPhase:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        cell = StateCellRef(EntityRef(observer.id, observer.type_name), "agent.phase")
        value = next_visible.get(cell)
        return () if value is None else (ObservationFact(cell, "equals", value),)


class ObserveAgentPhaseV2:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        cell = StateCellRef(EntityRef(observer.id, observer.type_name), "agent.phase")
        if cell not in next_visible:
            return ()
        return (ObservationFact(cell, "equals", next_visible[cell]),)


class NoPercepts:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        return ()


class RecordingPhaseHook(ObserveAgentPhase):
    def __init__(self):
        self.calls = []

    def __call__(self, prior_visible, next_visible, observer, step_index):
        self.calls.append(
            (dict(prior_visible), dict(next_visible), observer.id, step_index)
        )
        return super().__call__(prior_visible, next_visible, observer, step_index)


class RaisingProjectionHook:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        raise RuntimeError("projection boom")


def _hash(label: str) -> str:
    return stable_content_hash({"runtime-test": label})


def make_runtime_domain():
    return make_projection_domain()


def make_runtime_story(domain):
    story = make_projection_story(domain)
    return replace(
        story,
        observations=(
            Observation("o-a1-phase", "a1", "e1"),
            Observation("o-a2-phase", "a2", "e2"),
        ),
    )


def make_world_transition_model(domain):
    return make_transition_model(domain)


def make_projection_model(domain, hook=None, *, reverse=False, blackout=False):
    if blackout:
        return ObservationProjectionModelSpec(
            "runtime-projection",
            "1",
            domain.domain_id,
            domain.version,
            domain.content_hash,
            (),
        )
    phase = ObservationCapabilitySpec("agent.phase", "observer")
    first = ObserverProjectionSpec(
        "Agent",
        "vision",
        (phase,),
        (phase,),
        ObserveAgentPhase() if hook is None else hook,
    )
    status = ObserverProjectionSpec(
        "Agent",
        "status",
        (phase,),
        (phase,),
        ObserveAgentPhaseV2(),
    )
    specs = (status, first) if reverse else (first, status)
    return ObservationProjectionModelSpec(
        "runtime-projection",
        "1",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        specs,
    )


def advance_runtime_world(
    story,
    domain,
    prior_state,
    transition_model,
    *,
    phase_value: str,
):
    action_id = "a1-ready" if phase_value == "ready" else "a1-active"
    return advance_world_step(
        story,
        domain,
        prior_state,
        transition_model,
        (
            ActionIntent(
                "d-a1-phase",
                action_id,
                "runtime-selection",
                _hash("selection-a1"),
            ),
            ActionIntent(
                "d-a2-noop",
                "a2-wait",
                "runtime-selection",
                _hash("selection-a2"),
            ),
        ),
    )


def make_first_world_step(*, phase_value: str = "active"):
    domain = make_runtime_domain()
    story = make_runtime_story(domain)
    prior = world_state_from_story(story, domain, at_time=11)
    step = advance_runtime_world(
        story,
        domain,
        prior,
        make_world_transition_model(domain),
        phase_value=phase_value,
    )
    return domain, story, step


class NarrativeRuntimePerceptionTests(unittest.TestCase):
    def require_perception(self) -> None:
        if _PERCEPTION_IMPORT_ERROR is not None:
            self.fail(
                "narrative runtime perception boundary is missing: "
                f"{_PERCEPTION_IMPORT_ERROR}"
            )

    def test_runtime_perception_record_constructors_are_canonical_data_not_certification(self):
        self.require_perception()
        cell = StateCellRef(EntityRef("a1", "Agent"), "agent.phase")
        value = TypedValue("PhaseState", "active")
        with self.assertRaises((TypeError, ValueError)):
            RuntimePerceptView("", "vision", cell, "equals", value, 1)
        with self.assertRaises((TypeError, ValueError)):
            RuntimePerceptView("a1", "vision", cell, "not_equals", value, 1)
        with self.assertRaises((TypeError, ValueError)):
            RuntimePerceptView("a1", "vision", cell, "equals", None, 1)
        with self.assertRaises((TypeError, ValueError)):
            RuntimePerceptView("a1", "vision", cell, "clear", value, 1)
        with self.assertRaises((TypeError, ValueError)):
            RuntimePerceptView("a1", "vision", cell, "equals", value, True)

        evidence = RuntimeEpistemicEvidence(
            observer_id="a1",
            channel="vision",
            cell=cell,
            relation="equals",
            value=value,
            step_index=1,
            projected_observation_hash=_hash("projected"),
            projection_result_hash=_hash("projection-result"),
            projection_model_hash=_hash("projection-model"),
            source_world_state_hash=_hash("world-state"),
            source_world_step_hash=_hash("world-step"),
            source_transition_hashes=(_hash("transition-b"), _hash("transition-a")),
            projection_spec_hash=_hash("projection-spec"),
        )
        self.assertEqual(
            evidence.source_transition_hashes,
            tuple(sorted(evidence.source_transition_hashes)),
        )
        self.assertEqual(evidence.percept_view.observer_id, evidence.observer_id)
        self.assertEqual(evidence.percept_view.channel, evidence.channel)
        self.assertEqual(evidence.percept_view.cell, evidence.cell)
        self.assertEqual(evidence.percept_view.relation, evidence.relation)
        self.assertEqual(evidence.percept_view.value, evidence.value)
        self.assertEqual(evidence.percept_view.step_index, evidence.step_index)
        self.assertFalse(hasattr(evidence.percept_view, "source_world_state_hash"))
        self.assertFalse(hasattr(evidence.percept_view, "source_world_step_hash"))
        self.assertEqual(evidence.content_hash, stable_content_hash(evidence.to_dict()))

    def test_empty_ledger_binds_exact_branch_point_and_initial_world_state(self):
        self.require_perception()
        domain = make_runtime_domain()
        story = make_runtime_story(domain)
        ledger = runtime_evidence_ledger_from_story(story, domain, at_time=11)
        initial = world_state_from_story(story, domain, at_time=11)
        self.assertEqual(ledger.domain_id, domain.domain_id)
        self.assertEqual(ledger.domain_version, domain.version)
        self.assertEqual(ledger.domain_spec_hash, domain.content_hash)
        self.assertEqual(ledger.source_story_hash, story.content_hash)
        self.assertEqual(ledger.source_at_time, 11)
        self.assertEqual(ledger.initial_world_state_hash, initial.content_hash)
        self.assertEqual(ledger.current_world_state_hash, initial.content_hash)
        self.assertEqual(ledger.current_step_index, 0)
        self.assertEqual(ledger.batches, ())

    def test_first_admission_executes_projection_and_binds_exact_lineage(self):
        self.require_perception()
        domain, story, world_step = make_first_world_step()
        ledger = runtime_evidence_ledger_from_story(story, domain, at_time=11)
        result = admit_world_percepts(
            story, domain, world_step, make_projection_model(domain), ledger
        )
        self.assertIsInstance(result, RuntimePerceptAdmissionResult)
        self.assertEqual(result.prior_ledger_hash, ledger.content_hash)
        self.assertEqual(result.evidence_batch.step_index, 1)
        self.assertEqual(
            result.evidence_batch.source_prior_world_state_hash,
            world_step.prior_state.content_hash,
        )
        self.assertEqual(
            result.evidence_batch.source_world_state_hash,
            world_step.next_state.content_hash,
        )
        self.assertEqual(
            result.evidence_batch.source_world_step_hash,
            world_step.content_hash,
        )
        self.assertEqual(
            result.evidence_batch.projection_result_hash,
            result.projection_result.content_hash,
        )
        projected_by_key = {
            (item.observer_id, item.channel, item.fact.cell): item
            for item in result.projection_result.observations
        }
        for item in result.evidence_batch.evidence:
            projected = projected_by_key[(item.observer_id, item.channel, item.cell)]
            self.assertEqual(item.projected_observation_hash, projected.content_hash)
            self.assertEqual(
                item.projection_result_hash, result.projection_result.content_hash
            )
            self.assertEqual(item.source_world_step_hash, world_step.content_hash)
            self.assertEqual(
                item.source_world_state_hash, world_step.next_state.content_hash
            )

    def test_blackout_appends_explicit_empty_batch_and_advances_step(self):
        self.require_perception()
        domain, story, world_step = make_first_world_step()
        ledger = runtime_evidence_ledger_from_story(story, domain, at_time=11)
        result = admit_world_percepts(
            story,
            domain,
            world_step,
            make_projection_model(domain, blackout=True),
            ledger,
        )
        self.assertEqual(result.evidence_batch.evidence, ())
        self.assertEqual(result.next_ledger.current_step_index, 1)
        self.assertEqual(
            result.next_ledger.current_world_state_hash,
            world_step.next_state.content_hash,
        )
        self.assertEqual(len(result.next_ledger.batches), 1)

    def test_consecutive_batches_bind_world_and_prior_ledger_hash_chain(self):
        self.require_perception()
        domain, story, first = make_first_world_step()
        model = make_projection_model(domain)
        initial = runtime_evidence_ledger_from_story(story, domain, at_time=11)
        first_result = admit_world_percepts(story, domain, first, model, initial)
        second = make_second_noop_step(domain, story, first.next_state)
        second_result = admit_world_percepts(
            story, domain, second, model, first_result.next_ledger
        )
        batches = second_result.next_ledger.batches
        self.assertEqual(tuple(item.step_index for item in batches), (1, 2))
        self.assertEqual(
            batches[0].source_prior_world_state_hash,
            initial.initial_world_state_hash,
        )
        self.assertEqual(
            batches[1].source_prior_world_state_hash,
            batches[0].source_world_state_hash,
        )
        self.assertEqual(batches[0].prior_ledger_hash, initial.content_hash)
        self.assertEqual(
            batches[1].prior_ledger_hash, first_result.next_ledger.content_hash
        )

    def test_duplicate_skipped_and_reordered_admission_reject_typed(self):
        self.require_perception()
        domain, story, first = make_first_world_step()
        model = make_projection_model(domain)
        initial = runtime_evidence_ledger_from_story(story, domain, at_time=11)
        first_result = admit_world_percepts(story, domain, first, model, initial)
        second = make_second_noop_step(domain, story, first.next_state)
        with self.assertRaises(RuntimePerceptAdmissionError):
            admit_world_percepts(story, domain, first, model, first_result.next_ledger)
        with self.assertRaises(RuntimePerceptAdmissionError):
            admit_world_percepts(story, domain, second, model, initial)
        with self.assertRaises(RuntimePerceptAdmissionError):
            admit_world_percepts(story, domain, first, model, first_result.next_ledger)

    def test_source_domain_story_cutoff_and_prior_state_mismatch_reject_before_projection(self):
        self.require_perception()
        domain, story, world_step = make_first_world_step()
        initial = runtime_evidence_ledger_from_story(story, domain, at_time=11)
        cases = (
            _forge(initial, domain_id="other-domain"),
            _forge(initial, source_story_hash=_hash("other-story")),
            _forge(initial, source_at_time=10),
            _forge(initial, current_world_state_hash=_hash("other-world")),
        )
        for bad_ledger in cases:
            hook = RecordingPhaseHook()
            model = make_projection_model(domain, hook)
            with self.assertRaises(RuntimePerceptAdmissionError) as caught:
                admit_world_percepts(story, domain, world_step, model, bad_ledger)
            self.assertEqual(hook.calls, [])
            self.assertNotEqual(
                str(caught.exception),
                "runtime percept admission execution is unavailable in this stage",
            )

    def test_evidence_and_ledger_identity_ignore_semantically_irrelevant_input_order(self):
        self.require_perception()
        domain, story, world_step = make_first_world_step()
        initial = runtime_evidence_ledger_from_story(story, domain, at_time=11)
        left = admit_world_percepts(
            story, domain, world_step, make_projection_model(domain), initial
        )
        right = admit_world_percepts(
            story,
            domain,
            world_step,
            make_projection_model(domain, reverse=True),
            initial,
        )
        self.assertEqual(
            tuple(item.to_dict() for item in left.evidence_batch.evidence),
            tuple(item.to_dict() for item in right.evidence_batch.evidence),
        )
        self.assertEqual(left.evidence_batch.content_hash, right.evidence_batch.content_hash)
        self.assertEqual(left.next_ledger.content_hash, right.next_ledger.content_hash)

    def test_immutable_old_ledger_can_seed_two_counterfactual_next_steps(self):
        self.require_perception()
        domain = make_runtime_domain()
        story = make_runtime_story(domain)
        initial = runtime_evidence_ledger_from_story(story, domain, at_time=11)
        prior = world_state_from_story(story, domain, at_time=11)
        transition_model = make_world_transition_model(domain)
        active = advance_runtime_world(
            story, domain, prior, transition_model, phase_value="active"
        )
        ready = advance_runtime_world(
            story, domain, prior, transition_model, phase_value="ready"
        )
        model = make_projection_model(domain)
        left = admit_world_percepts(story, domain, active, model, initial)
        right = admit_world_percepts(story, domain, ready, model, initial)
        self.assertEqual(initial.current_step_index, 0)
        self.assertEqual(initial.batches, ())
        self.assertNotEqual(
            left.next_ledger.current_world_state_hash,
            right.next_ledger.current_world_state_hash,
        )
        self.assertNotEqual(left.next_ledger.content_hash, right.next_ledger.content_hash)

    def test_projection_failure_is_preserved_as_typed_admission_failure(self):
        self.require_perception()
        domain, story, world_step = make_first_world_step()
        initial = runtime_evidence_ledger_from_story(story, domain, at_time=11)
        model = make_projection_model(domain, RaisingProjectionHook())
        with self.assertRaises(RuntimePerceptAdmissionError) as caught:
            admit_world_percepts(story, domain, world_step, model, initial)
        self.assertIsInstance(caught.exception.__cause__, ObservationProjectionError)


if __name__ == "__main__":
    unittest.main()