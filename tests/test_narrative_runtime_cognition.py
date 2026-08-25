from __future__ import annotations

import inspect
import math
import unittest

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.ir import StateCellRef, EntityRef, TypedValue
from narrative_dynamics.narrative.replay import epistemic_state
from narrative_dynamics.narrative.uncertain import (
    UncertainBeliefModelSpec,
    uncertain_epistemic_state,
)
from tests.test_narrative_runtime_perception import (
    make_runtime_domain,
    make_runtime_story,
)

_COGNITION_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.runtime_perception import (
        RuntimeEpistemicEvidence,
        RuntimeEvidenceBatch,
        RuntimeEvidenceLedger,
        runtime_evidence_ledger_from_story,
    )
    from narrative_dynamics.narrative.runtime_cognition import (
        RuntimeBeliefModelSpec,
        RuntimeBeliefResolutionError,
        RuntimeBeliefUpdateStep,
        RuntimeEpistemicCellView,
        RuntimeEpistemicResolutionError,
        RuntimeEpistemicState,
        RuntimeUncertainBeliefCellView,
        RuntimeUncertainBeliefState,
        runtime_epistemic_state,
        runtime_uncertain_belief_state,
    )
except ImportError as error:
    _COGNITION_IMPORT_ERROR = error


def _hash(label: str) -> str:
    return stable_content_hash({"runtime-cognition-test": label})


def _value_hash(value: TypedValue) -> str:
    return stable_content_hash(value.to_dict())


def phase_cell(agent_id: str = "a1") -> StateCellRef:
    return StateCellRef(EntityRef(agent_id, "Agent"), "agent.phase")


def alert_cell() -> StateCellRef:
    return StateCellRef(EntityRef("svc", "Service"), "service.alert")


class SeedPriorHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        mass = 1.0 / len(hypotheses)
        return {_value_hash(value): mass for value in hypotheses}


class SeedLikelihoodHook:
    def __call__(self, agent_id, evidence, hypotheses, parameters):
        if evidence.relation != "equals":
            return {_value_hash(value): 1.0 for value in hypotheses}
        return {
            _value_hash(value): (0.9 if value == evidence.value else 0.1)
            for value in hypotheses
        }


class SemanticRuntimeLikelihood:
    def __init__(self):
        self.inputs = []

    def __call__(self, agent_id, percept_view, hypotheses, parameters):
        self.inputs.append(percept_view)
        favored = percept_view.value
        return {
            _value_hash(value): (
                parameters["match"] if value == favored else parameters["mismatch"]
            )
            for value in hypotheses
        }


class SemanticRuntimeLikelihoodV2:
    def __call__(self, agent_id, percept_view, hypotheses, parameters):
        favored = percept_view.value
        pairs = [
            (
                _value_hash(value),
                parameters["match"] if value == favored else parameters["mismatch"],
            )
            for value in reversed(hypotheses)
        ]
        return dict(pairs)


class RecordingClearRuntimeLikelihood:
    def __init__(self):
        self.calls = []

    def __call__(self, agent_id, percept_view, hypotheses, parameters):
        self.calls.append((percept_view, tuple(hypotheses)))
        return {_value_hash(value): 1.0 for value in hypotheses}


class InvalidByModeRuntimeLikelihood:
    def __call__(self, agent_id, percept_view, hypotheses, parameters):
        mode = parameters["mode"]
        keys = [_value_hash(value) for value in hypotheses]
        if mode == "nonmapping":
            return []
        if mode == "missing":
            return {key: 1.0 for key in keys[:-1]}
        if mode == "extra":
            result = {key: 1.0 for key in keys}
            result[_hash("extra-hypothesis")] = 1.0
            return result
        bad = {
            "bool": True,
            "negative": -0.1,
            "high": 1.1,
            "nan": math.nan,
            "inf": math.inf,
        }[mode]
        return {key: bad for key in keys}


class ZeroRuntimeLikelihood:
    def __call__(self, agent_id, percept_view, hypotheses, parameters):
        return {_value_hash(value): 0.0 for value in hypotheses}


def make_seed_model(*, version: str = "1") -> UncertainBeliefModelSpec:
    return UncertainBeliefModelSpec(
        "runtime-seed",
        version,
        {},
        SeedPriorHook(),
        SeedLikelihoodHook(),
    )


def make_runtime_belief_model(seed_model=None, hook=None, *, parameters=None):
    if seed_model is None:
        seed_model = make_seed_model()
    if hook is None:
        hook = SemanticRuntimeLikelihood()
    if parameters is None:
        parameters = {"match": 0.9, "mismatch": 0.1, "clear": 0.5}
    return RuntimeBeliefModelSpec(
        "runtime-belief",
        "1",
        seed_model,
        parameters,
        hook,
    )


def empty_runtime_case():
    domain = make_runtime_domain()
    story = make_runtime_story(domain)
    ledger = runtime_evidence_ledger_from_story(story, domain, at_time=11)
    return domain, story, ledger


def extend_ledger(ledger, semantic_rows, *, branch: str):
    step = ledger.current_step_index + 1
    next_world_hash = _hash(f"{branch}-world-{step}")
    world_step_hash = _hash(f"{branch}-world-step-{step}")
    projection_model_hash = _hash(f"{branch}-projection-model-{step}")
    projection_result_hash = _hash(f"{branch}-projection-result-{step}")
    evidence = []
    for index, (observer_id, channel, cell, relation, value) in enumerate(semantic_rows):
        evidence.append(
            RuntimeEpistemicEvidence(
                observer_id=observer_id,
                channel=channel,
                cell=cell,
                relation=relation,
                value=value,
                step_index=step,
                projected_observation_hash=_hash(
                    f"{branch}-projected-{step}-{index}"
                ),
                projection_result_hash=projection_result_hash,
                projection_model_hash=projection_model_hash,
                source_world_state_hash=next_world_hash,
                source_world_step_hash=world_step_hash,
                source_transition_hashes=(
                    _hash(f"{branch}-transition-{step}-{index}"),
                ),
                projection_spec_hash=_hash(
                    f"{branch}-projection-spec-{channel}"
                ),
            )
        )
    batch = RuntimeEvidenceBatch(
        prior_ledger_hash=ledger.content_hash,
        step_index=step,
        source_prior_world_state_hash=ledger.current_world_state_hash,
        source_world_state_hash=next_world_hash,
        source_world_step_hash=world_step_hash,
        projection_model_hash=projection_model_hash,
        projection_result_hash=projection_result_hash,
        evidence=tuple(evidence),
    )
    return RuntimeEvidenceLedger(
        domain_id=ledger.domain_id,
        domain_version=ledger.domain_version,
        domain_spec_hash=ledger.domain_spec_hash,
        source_story_hash=ledger.source_story_hash,
        source_at_time=ledger.source_at_time,
        initial_world_state_hash=ledger.initial_world_state_hash,
        current_world_state_hash=next_world_hash,
        batches=ledger.batches + (batch,),
    )


def cognition_semantics(state):
    return {
        "cells": {
            cell: (
                view.status,
                view.resolved_value,
                view.constraints,
                view.basis,
                view.last_runtime_step,
            )
            for cell, view in state.cells.items()
        },
        "resolved_values": dict(state.resolved_values),
    }


class NarrativeRuntimeCognitionTests(unittest.TestCase):
    def require_cognition(self) -> None:
        if _COGNITION_IMPORT_ERROR is not None:
            self.fail(
                "narrative runtime cognition boundary is missing: "
                f"{_COGNITION_IMPORT_ERROR}"
            )

    def test_runtime_epistemic_seed_equals_authored_seed_and_keeps_clocks_separate(self):
        self.require_cognition()
        domain, story, ledger = empty_runtime_case()
        state = runtime_epistemic_state(story, domain, "a1", ledger)
        authored_seed = epistemic_state(
            story, domain, "a1", at_time=ledger.source_at_time
        )
        self.assertEqual(state.seed_state.to_dict(), authored_seed.to_dict())
        self.assertEqual(state.step_index, 0)
        self.assertEqual(state.source_at_time, 11)
        self.assertNotIn(
            "logical_time", RuntimeEpistemicEvidence.__dataclass_fields__
        )
        self.assertEqual(
            tuple(inspect.signature(runtime_epistemic_state).parameters),
            ("story", "domain", "agent_id", "ledger"),
        )
        self.assertEqual(
            tuple(inspect.signature(runtime_uncertain_belief_state).parameters),
            ("story", "domain", "agent_id", "ledger", "model", "tracked_cells"),
        )

    def test_runtime_equals_supersedes_stale_authored_cell_semantics(self):
        self.require_cognition()
        domain, story, ledger = empty_runtime_case()
        ledger = extend_ledger(
            ledger,
            (
                (
                    "a1",
                    "vision",
                    phase_cell(),
                    "equals",
                    TypedValue("PhaseState", "active"),
                ),
            ),
            branch="equals",
        )
        state = runtime_epistemic_state(story, domain, "a1", ledger)
        view = state.cells[phase_cell()]
        self.assertEqual(view.status, "resolved")
        self.assertEqual(view.resolved_value, TypedValue("PhaseState", "active"))
        self.assertEqual(view.constraints, ())
        self.assertEqual(view.basis, "runtime_perception")
        self.assertEqual(view.last_runtime_step, 1)
        self.assertEqual(len(view.supporting_runtime_evidence_hashes), 1)

    def test_runtime_clear_makes_cell_unknown_and_clears_stale_constraints(self):
        self.require_cognition()
        domain, story, ledger = empty_runtime_case()
        ledger = extend_ledger(
            ledger,
            (("a1", "vision", phase_cell(), "clear", None),),
            branch="clear",
        )
        state = runtime_epistemic_state(story, domain, "a1", ledger)
        view = state.cells[phase_cell()]
        self.assertEqual(view.status, "unknown")
        self.assertIsNone(view.resolved_value)
        self.assertEqual(view.constraints, ())
        self.assertEqual(view.basis, "runtime_perception")
        self.assertEqual(view.last_runtime_step, 1)
        self.assertNotIn(phase_cell(), state.resolved_values)

    def test_unperceived_cells_preserve_authored_seed_semantics(self):
        self.require_cognition()
        domain, story, ledger = empty_runtime_case()
        ledger = extend_ledger(
            ledger,
            (
                (
                    "a1",
                    "vision",
                    alert_cell(),
                    "equals",
                    TypedValue("AlertState", True),
                ),
            ),
            branch="new-cell",
        )
        state = runtime_epistemic_state(story, domain, "a1", ledger)
        phase = state.cells[phase_cell()]
        self.assertEqual(phase.basis, "authored_seed")
        self.assertEqual(phase.resolved_value, TypedValue("PhaseState", "ready"))
        self.assertNotIn(alert_cell(), state.seed_state.cells)
        self.assertIn(alert_cell(), state.cells)
        self.assertEqual(
            state.cells[alert_cell()].resolved_value,
            TypedValue("AlertState", True),
        )

    def test_multi_channel_same_cell_is_deterministic_and_keeps_all_runtime_support(self):
        self.require_cognition()
        domain, story, ledger = empty_runtime_case()
        value = TypedValue("PhaseState", "active")
        ledger = extend_ledger(
            ledger,
            (
                ("a1", "vision", phase_cell(), "equals", value),
                ("a1", "status", phase_cell(), "equals", value),
            ),
            branch="multi-channel",
        )
        state = runtime_epistemic_state(story, domain, "a1", ledger)
        view = state.cells[phase_cell()]
        self.assertEqual(view.status, "resolved")
        self.assertEqual(view.resolved_value, value)
        self.assertEqual(
            view.supporting_runtime_evidence_hashes,
            tuple(sorted(view.supporting_runtime_evidence_hashes)),
        )
        self.assertEqual(len(view.supporting_runtime_evidence_hashes), 2)

    def test_runtime_epistemic_state_filters_other_agents_evidence(self):
        self.require_cognition()
        domain, story, ledger = empty_runtime_case()
        active = TypedValue("PhaseState", "active")
        ledger = extend_ledger(
            ledger,
            (
                ("a1", "vision", phase_cell("a1"), "equals", active),
                ("a2", "vision", phase_cell("a2"), "equals", active),
            ),
            branch="isolation",
        )
        state = runtime_epistemic_state(story, domain, "a1", ledger)
        self.assertTrue(
            all(item.observer_id == "a1" for item in state.runtime_evidence_history)
        )
        self.assertFalse(
            any(item.observer_id == "a2" for item in state.runtime_evidence_history)
        )

    def test_same_semantic_percepts_preserve_cognition_semantics_across_provenance_branches(self):
        self.require_cognition()
        domain, story, initial = empty_runtime_case()
        row = (
            "a1",
            "vision",
            phase_cell(),
            "equals",
            TypedValue("PhaseState", "active"),
        )
        left_ledger = extend_ledger(initial, (row,), branch="left")
        right_ledger = extend_ledger(initial, (row,), branch="right")
        left = runtime_epistemic_state(story, domain, "a1", left_ledger)
        right = runtime_epistemic_state(story, domain, "a1", right_ledger)
        self.assertEqual(cognition_semantics(left), cognition_semantics(right))
        self.assertNotEqual(left.ledger_hash, right.ledger_hash)

    def test_runtime_belief_seed_equals_authored_uncertain_seed_without_runtime_evidence(self):
        self.require_cognition()
        domain, story, ledger = empty_runtime_case()
        model = make_runtime_belief_model()
        state = runtime_uncertain_belief_state(
            story, domain, "a1", ledger, model, (phase_cell(),)
        )
        seed = uncertain_epistemic_state(
            story,
            domain,
            "a1",
            model.seed_model,
            (phase_cell(),),
            at_time=ledger.source_at_time,
        )
        self.assertEqual(state.seed_belief_state.to_dict(), seed.to_dict())
        view = state.cells[phase_cell()]
        self.assertEqual(view.updates, ())
        self.assertEqual(view.posterior, view.seed_posterior)

    def test_runtime_belief_model_identity_binds_seed_parameters_and_hook(self):
        self.require_cognition()
        first = make_runtime_belief_model(hook=SemanticRuntimeLikelihood())
        same = make_runtime_belief_model(hook=SemanticRuntimeLikelihood())
        changed_seed = make_runtime_belief_model(
            seed_model=make_seed_model(version="2"),
            hook=SemanticRuntimeLikelihood(),
        )
        changed_parameters = make_runtime_belief_model(
            hook=SemanticRuntimeLikelihood(),
            parameters={"match": 0.8, "mismatch": 0.2, "clear": 0.5},
        )
        changed_hook = make_runtime_belief_model(hook=SemanticRuntimeLikelihoodV2())
        self.assertEqual(first.content_hash, same.content_hash)
        self.assertNotEqual(first.content_hash, changed_seed.content_hash)
        self.assertNotEqual(first.content_hash, changed_parameters.content_hash)
        self.assertNotEqual(first.content_hash, changed_hook.content_hash)

    def test_runtime_likelihood_receives_only_provenance_free_percept_view(self):
        self.require_cognition()
        domain, story, ledger = empty_runtime_case()
        ledger = extend_ledger(
            ledger,
            (
                (
                    "a1",
                    "vision",
                    phase_cell(),
                    "equals",
                    TypedValue("PhaseState", "active"),
                ),
            ),
            branch="hook-view",
        )
        hook = SemanticRuntimeLikelihood()
        model = make_runtime_belief_model(hook=hook)
        runtime_uncertain_belief_state(
            story, domain, "a1", ledger, model, (phase_cell(),)
        )
        self.assertEqual(len(hook.inputs), 1)
        view = hook.inputs[0]
        for name in (
            "projected_observation_hash",
            "projection_result_hash",
            "projection_model_hash",
            "source_world_state_hash",
            "source_world_step_hash",
            "source_transition_hashes",
            "projection_spec_hash",
            "ledger_hash",
        ):
            self.assertFalse(hasattr(view, name), name)

    def test_hidden_provenance_changes_cannot_change_posterior_semantics(self):
        self.require_cognition()
        domain, story, initial = empty_runtime_case()
        row = (
            "a1",
            "vision",
            phase_cell(),
            "equals",
            TypedValue("PhaseState", "active"),
        )
        left_ledger = extend_ledger(initial, (row,), branch="posterior-left")
        right_ledger = extend_ledger(initial, (row,), branch="posterior-right")
        model = make_runtime_belief_model(hook=SemanticRuntimeLikelihood())
        left = runtime_uncertain_belief_state(
            story, domain, "a1", left_ledger, model, (phase_cell(),)
        )
        right = runtime_uncertain_belief_state(
            story, domain, "a1", right_ledger, model, (phase_cell(),)
        )
        self.assertEqual(
            left.cells[phase_cell()].posterior,
            right.cells[phase_cell()].posterior,
        )
        self.assertNotEqual(left.ledger_hash, right.ledger_hash)

    def test_invalid_runtime_likelihood_shape_keys_and_probabilities_reject_typed(self):
        self.require_cognition()
        domain, story, ledger = empty_runtime_case()
        ledger = extend_ledger(
            ledger,
            (
                (
                    "a1",
                    "vision",
                    phase_cell(),
                    "equals",
                    TypedValue("PhaseState", "active"),
                ),
            ),
            branch="invalid-likelihood",
        )
        for mode in (
            "nonmapping",
            "missing",
            "extra",
            "bool",
            "negative",
            "high",
            "nan",
            "inf",
        ):
            with self.subTest(mode=mode):
                model = make_runtime_belief_model(
                    hook=InvalidByModeRuntimeLikelihood(),
                    parameters={
                        "match": 0.9,
                        "mismatch": 0.1,
                        "clear": 0.5,
                        "mode": mode,
                    },
                )
                with self.assertRaises(RuntimeBeliefResolutionError):
                    runtime_uncertain_belief_state(
                        story, domain, "a1", ledger, model, (phase_cell(),)
                    )

    def test_zero_runtime_posterior_mass_rejects_typed(self):
        self.require_cognition()
        domain, story, ledger = empty_runtime_case()
        ledger = extend_ledger(
            ledger,
            (
                (
                    "a1",
                    "vision",
                    phase_cell(),
                    "equals",
                    TypedValue("PhaseState", "active"),
                ),
            ),
            branch="zero",
        )
        model = make_runtime_belief_model(hook=ZeroRuntimeLikelihood())
        with self.assertRaises(RuntimeBeliefResolutionError):
            runtime_uncertain_belief_state(
                story, domain, "a1", ledger, model, (phase_cell(),)
            )

    def test_runtime_belief_updates_form_exact_chain_from_seed_posterior(self):
        self.require_cognition()
        domain, story, ledger = empty_runtime_case()
        ledger = extend_ledger(
            ledger,
            (
                (
                    "a1",
                    "vision",
                    phase_cell(),
                    "equals",
                    TypedValue("PhaseState", "active"),
                ),
            ),
            branch="chain-1",
        )
        ledger = extend_ledger(
            ledger,
            (
                (
                    "a1",
                    "vision",
                    phase_cell(),
                    "equals",
                    TypedValue("PhaseState", "ready"),
                ),
            ),
            branch="chain-2",
        )
        model = make_runtime_belief_model(hook=SemanticRuntimeLikelihood())
        state = runtime_uncertain_belief_state(
            story, domain, "a1", ledger, model, (phase_cell(),)
        )
        view = state.cells[phase_cell()]
        self.assertEqual(len(view.updates), 2)
        self.assertEqual(view.updates[0].prior, view.seed_posterior)
        self.assertEqual(view.updates[1].prior, view.updates[0].posterior)
        self.assertEqual(view.posterior, view.updates[-1].posterior)

    def test_runtime_clear_uses_existing_hypotheses_and_never_invents_absent(self):
        self.require_cognition()
        domain, story, ledger = empty_runtime_case()
        ledger = extend_ledger(
            ledger,
            (("a1", "vision", phase_cell(), "clear", None),),
            branch="belief-clear",
        )
        hook = RecordingClearRuntimeLikelihood()
        model = make_runtime_belief_model(hook=hook)
        state = runtime_uncertain_belief_state(
            story, domain, "a1", ledger, model, (phase_cell(),)
        )
        self.assertEqual(len(hook.calls), 1)
        percept, hypotheses = hook.calls[0]
        self.assertEqual(percept.relation, "clear")
        raw_values = {item.value for item in hypotheses}
        self.assertEqual(raw_values, {"ready", "active"})
        self.assertNotIn("absent", raw_values)
        self.assertEqual(
            {mass.value.value for mass in state.cells[phase_cell()].posterior.masses},
            {"ready", "active"},
        )

    def test_blackout_only_steps_preserve_cognition_and_posterior_while_advancing_step(self):
        self.require_cognition()
        domain, story, initial = empty_runtime_case()
        blackout = extend_ledger(initial, (), branch="blackout")
        before_epistemic = runtime_epistemic_state(story, domain, "a1", initial)
        after_epistemic = runtime_epistemic_state(story, domain, "a1", blackout)
        self.assertEqual(
            cognition_semantics(before_epistemic),
            cognition_semantics(after_epistemic),
        )
        self.assertEqual(before_epistemic.step_index, 0)
        self.assertEqual(after_epistemic.step_index, 1)

        model = make_runtime_belief_model(hook=SemanticRuntimeLikelihood())
        before_belief = runtime_uncertain_belief_state(
            story, domain, "a1", initial, model, (phase_cell(),)
        )
        after_belief = runtime_uncertain_belief_state(
            story, domain, "a1", blackout, model, (phase_cell(),)
        )
        self.assertEqual(
            before_belief.cells[phase_cell()].posterior,
            after_belief.cells[phase_cell()].posterior,
        )
        self.assertEqual(before_belief.step_index, 0)
        self.assertEqual(after_belief.step_index, 1)


if __name__ == "__main__":
    unittest.main()
