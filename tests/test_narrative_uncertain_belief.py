from __future__ import annotations

from dataclasses import replace
import math
import unittest

from hypothesis_competition import posterior_distribution
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative import uncertain
from narrative_dynamics.narrative.domain import (
    DomainSpec,
    EntityTypeSpec,
    StateVariableSpec,
    ValueTypeSpec,
)
from narrative_dynamics.narrative.ir import (
    Entity,
    EntityRef,
    GenericNarrative,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.replay import epistemic_state, objective_state
from tests.narrative_test_support import (
    make_test_domain,
    make_test_story,
    target_cell,
)


def _value_token(value: TypedValue) -> str:
    raw = value.value
    if isinstance(raw, EntityRef):
        return raw.entity_id
    if isinstance(raw, bool):
        return "true" if raw else "false"
    return str(raw)


def _value_hash(value: TypedValue) -> str:
    return stable_content_hash(value.to_dict())


class ParameterPriorHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        table = parameters["prior_by_value"]
        return {
            _value_hash(value): table[_value_token(value)]
            for value in hypotheses
        }


class ParameterLikelihoodHook:
    def __call__(self, agent_id, evidence, hypotheses, parameters):
        table = parameters["likelihood_by_evidence"].get(
            evidence.supporting_id,
            parameters["default_likelihood_by_value"],
        )
        return {
            _value_hash(value): table[_value_token(value)]
            for value in hypotheses
        }


class AlternateLikelihoodHook:
    def __call__(self, agent_id, evidence, hypotheses, parameters):
        table = parameters["likelihood_by_evidence"].get(
            evidence.supporting_id,
            parameters["default_likelihood_by_value"],
        )
        rows = [
            (_value_hash(value), table[_value_token(value)])
            for value in hypotheses
        ]
        return dict(reversed(rows))


def make_uncertain_model(*, parameters=None, likelihood_hook=None, prior_hook=None):
    if parameters is None:
        parameters = {
            "prior_by_value": {
                "healthy": 0.20,
                "failed": 0.50,
                "recovered": 0.30,
            },
            "default_likelihood_by_value": {
                "healthy": 1.0,
                "failed": 1.0,
                "recovered": 1.0,
            },
            "likelihood_by_evidence": {
                "c1": {
                    "healthy": 0.10,
                    "failed": 0.20,
                    "recovered": 0.90,
                },
                "c2": {
                    "healthy": 0.10,
                    "failed": 0.80,
                    "recovered": 0.20,
                },
            },
        }
    return uncertain.UncertainBeliefModelSpec(
        model_id="service-health-uncertain",
        version="1",
        parameters=parameters,
        prior_hook=(ParameterPriorHook() if prior_hook is None else prior_hook),
        likelihood_hook=(
            ParameterLikelihoodHook()
            if likelihood_hook is None
            else likelihood_hook
        ),
    )


class UncertainBeliefRecordModelTests(unittest.TestCase):
    def test_distribution_is_canonical_normalized_and_supports_exact_lookup(self):
        cell = target_cell()
        failed = TypedValue("HealthState", "failed")
        recovered = TypedValue("HealthState", "recovered")
        distribution = uncertain.BeliefDistribution(
            cell,
            (
                uncertain.BeliefMass(recovered, 0.25),
                uncertain.BeliefMass(failed, 0.75),
            ),
        )
        self.assertAlmostEqual(distribution.probability_of(failed), 0.75)
        self.assertEqual(
            tuple(mass.value for mass in distribution.masses),
            tuple(
                sorted(
                    (failed, recovered),
                    key=lambda value: stable_content_hash(value.to_dict()),
                )
            ),
        )
        with self.assertRaises(KeyError):
            distribution.probability_of(
                TypedValue("HealthState", "healthy")
            )

    def test_model_identity_binds_code_and_explicit_parameters(self):
        first = make_uncertain_model()
        same = make_uncertain_model()
        changed_parameters = make_uncertain_model(
            parameters={
                "prior_by_value": {
                    "healthy": 0.30,
                    "failed": 0.40,
                    "recovered": 0.30,
                },
                "default_likelihood_by_value": {
                    "healthy": 1.0,
                    "failed": 1.0,
                    "recovered": 1.0,
                },
                "likelihood_by_evidence": {},
            }
        )
        changed_hook = make_uncertain_model(
            likelihood_hook=AlternateLikelihoodHook()
        )
        self.assertEqual(first.content_hash, same.content_hash)
        self.assertNotEqual(first.content_hash, changed_parameters.content_hash)
        self.assertNotEqual(first.content_hash, changed_hook.content_hash)

    def test_parameters_are_detached_and_noncanonical_values_reject(self):
        raw = {
            "prior_by_value": {
                "healthy": 0.2,
                "failed": 0.5,
                "recovered": 0.3,
            },
            "default_likelihood_by_value": {
                "healthy": 1.0,
                "failed": 1.0,
                "recovered": 1.0,
            },
            "likelihood_by_evidence": {},
        }
        model = make_uncertain_model(parameters=raw)
        before = model.content_hash
        raw["prior_by_value"]["failed"] = 0.99
        self.assertEqual(model.content_hash, before)
        with self.assertRaises((TypeError, ValueError)):
            uncertain.UncertainBeliefModelSpec(
                "bad",
                "1",
                {"bad": {1, 2}},
                ParameterPriorHook(),
                ParameterLikelihoodHook(),
            )
        with self.assertRaises((TypeError, ValueError)):
            uncertain.UncertainBeliefModelSpec(
                "bad",
                "1",
                {"bad": math.nan},
                ParameterPriorHook(),
                ParameterLikelihoodHook(),
            )

    def test_probability_record_constructors_fail_closed(self):
        value = TypedValue("HealthState", "failed")
        for bad in (True, -0.01, 1.01, math.nan, math.inf):
            with self.subTest(record="mass", bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    uncertain.BeliefMass(value, bad)
            with self.subTest(record="likelihood", bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    uncertain.BeliefLikelihood(value, bad)
        with self.assertRaises((TypeError, ValueError)):
            uncertain.BeliefMass("failed", 0.5)
        with self.assertRaises((TypeError, ValueError)):
            uncertain.BeliefLikelihood("failed", 0.5)

    def test_distribution_constructor_rejects_invalid_support_or_mass(self):
        cell = target_cell()
        failed = TypedValue("HealthState", "failed")
        recovered = TypedValue("HealthState", "recovered")
        with self.assertRaises((TypeError, ValueError)):
            uncertain.BeliefDistribution(
                cell,
                (uncertain.BeliefMass(failed, 1.0),),
            )
        with self.assertRaises((TypeError, ValueError)):
            uncertain.BeliefDistribution(
                cell,
                (
                    uncertain.BeliefMass(failed, 0.5),
                    uncertain.BeliefMass(failed, 0.5),
                ),
            )
        with self.assertRaises((TypeError, ValueError)):
            uncertain.BeliefDistribution(
                cell,
                (
                    uncertain.BeliefMass(failed, 0.6),
                    uncertain.BeliefMass(recovered, 0.5),
                ),
            )


class NarrativeUncertainBeliefTests(unittest.TestCase):
    def test_prior_without_evidence_is_preserved_and_discrete_replay_stays_sparse(self):
        story = make_test_story(
            receive=False,
            bob_observes_failure=False,
        )
        domain = make_test_domain()
        state = uncertain.uncertain_epistemic_state(
            story,
            domain,
            "bob",
            make_uncertain_model(),
            (target_cell(),),
            at_time=3,
        )
        view = state.cells[target_cell()]
        self.assertEqual(view.updates, ())
        self.assertEqual(view.posterior, view.prior)
        self.assertAlmostEqual(
            view.posterior.probability_of(
                TypedValue("HealthState", "failed")
            ),
            0.50,
        )
        self.assertNotIn(
            target_cell(),
            epistemic_state(story, domain, "bob", at_time=3).cells,
        )

    def test_received_claim_updates_posterior_with_exact_provenance(self):
        story = make_test_story(
            receive=True,
            bob_observes_failure=False,
        )
        domain = make_test_domain()
        state = uncertain.uncertain_epistemic_state(
            story,
            domain,
            "bob",
            make_uncertain_model(),
            (target_cell(),),
            at_time=3,
        )
        view = state.cells[target_cell()]
        step = view.updates[0]
        self.assertEqual(step.evidence.supporting_id, "c1")
        self.assertEqual(step.evidence.source_agent, "alice")
        self.assertEqual(step.evidence.logical_time, 3)
        self.assertEqual(
            step.evidence.provenance_refs,
            ("c1", "e2"),
        )
        expected = posterior_distribution(
            {
                _value_hash(
                    TypedValue("HealthState", "healthy")
                ): 0.20,
                _value_hash(
                    TypedValue("HealthState", "failed")
                ): 0.50,
                _value_hash(
                    TypedValue("HealthState", "recovered")
                ): 0.30,
            },
            {
                _value_hash(
                    TypedValue("HealthState", "healthy")
                ): 0.10,
                _value_hash(
                    TypedValue("HealthState", "failed")
                ): 0.20,
                _value_hash(
                    TypedValue("HealthState", "recovered")
                ): 0.90,
            },
        )
        for mass in view.posterior.masses:
            self.assertAlmostEqual(
                mass.probability,
                expected[_value_hash(mass.value)],
            )

    def test_reception_changes_belief_without_changing_objective_state(self):
        received = make_test_story(
            receive=True,
            bob_observes_failure=False,
        )
        unreceived = make_test_story(
            receive=False,
            bob_observes_failure=False,
        )
        domain = make_test_domain()
        model = make_uncertain_model()
        left = uncertain.uncertain_epistemic_state(
            received,
            domain,
            "bob",
            model,
            (target_cell(),),
            at_time=3,
        )
        right = uncertain.uncertain_epistemic_state(
            unreceived,
            domain,
            "bob",
            model,
            (target_cell(),),
            at_time=3,
        )
        self.assertNotEqual(
            left.cells[target_cell()].posterior,
            right.cells[target_cell()].posterior,
        )
        self.assertEqual(
            objective_state(received, domain, at_time=3),
            objective_state(unreceived, domain, at_time=3),
        )

    def test_conflicting_claims_are_discrete_conflict_but_graded_uncertainty(self):
        story = make_test_story(
            claim_value="recovered",
            second_claim_value="failed",
            bob_observes_failure=False,
        )
        domain = make_test_domain()
        discrete = epistemic_state(
            story,
            domain,
            "bob",
            at_time=4,
        )
        self.assertEqual(
            discrete.cells[target_cell()].status,
            "conflicted",
        )
        view = uncertain.uncertain_epistemic_state(
            story,
            domain,
            "bob",
            make_uncertain_model(),
            (target_cell(),),
            at_time=4,
        ).cells[target_cell()]
        self.assertEqual(
            tuple(
                step.evidence.supporting_id
                for step in view.updates
            ),
            ("c1", "c2"),
        )
        self.assertAlmostEqual(
            sum(
                mass.probability
                for mass in view.posterior.masses
            ),
            1.0,
        )
        self.assertTrue(
            all(
                0.0 < mass.probability < 1.0
                for mass in view.posterior.masses
            )
        )

    def test_time_cutoff_excludes_later_claim_update(self):
        story = make_test_story(
            claim_value="recovered",
            second_claim_value="failed",
            bob_observes_failure=False,
        )
        domain = make_test_domain()
        model = make_uncertain_model()
        early = uncertain.uncertain_epistemic_state(
            story,
            domain,
            "bob",
            model,
            (target_cell(),),
            at_time=3,
        )
        late = uncertain.uncertain_epistemic_state(
            story,
            domain,
            "bob",
            model,
            (target_cell(),),
            at_time=4,
        )
        self.assertEqual(
            tuple(
                step.evidence.supporting_id
                for step in early.cells[target_cell()].updates
            ),
            ("c1",),
        )
        self.assertEqual(
            tuple(
                step.evidence.supporting_id
                for step in late.cells[target_cell()].updates
            ),
            ("c1", "c2"),
        )
        self.assertNotEqual(
            early.cells[target_cell()].posterior,
            late.cells[target_cell()].posterior,
        )


def make_finite_domain_and_story():
    domain = DomainSpec(
        domain_id="finite-belief-test",
        version="1",
        entity_types=(
            EntityTypeSpec("Agent"),
            EntityTypeSpec("Subject"),
            EntityTypeSpec("Candidate"),
        ),
        value_types=(
            ValueTypeSpec("Flag", "bool"),
            ValueTypeSpec(
                "CandidateRef",
                "entity_ref",
                entity_type="Candidate",
            ),
            ValueTypeSpec("Count", "integer"),
            ValueTypeSpec("Label", "text"),
        ),
        state_variables=(
            StateVariableSpec(
                "subject.flag",
                "Subject",
                "Flag",
            ),
            StateVariableSpec(
                "subject.target",
                "Subject",
                "CandidateRef",
            ),
            StateVariableSpec(
                "subject.count",
                "Subject",
                "Count",
            ),
            StateVariableSpec(
                "subject.label",
                "Subject",
                "Label",
            ),
        ),
        event_types=(),
        action_types=(),
        decision_types=(),
        semantic_hooks=(),
    )
    story = GenericNarrative(
        domain.domain_id,
        domain.version,
        domain.content_hash,
        (
            Entity("bob", "Agent"),
            Entity("subject", "Subject"),
            Entity("candidate-a", "Candidate"),
            Entity("candidate-b", "Candidate"),
        ),
        (),
        (),
        (),
        (),
        (),
    )
    return domain, story


def make_finite_model():
    return uncertain.UncertainBeliefModelSpec(
        "finite",
        "1",
        {
            "prior_by_value": {
                "false": 0.25,
                "true": 0.75,
                "candidate-a": 0.60,
                "candidate-b": 0.40,
            },
            "default_likelihood_by_value": {},
            "likelihood_by_evidence": {},
        },
        ParameterPriorHook(),
        ParameterLikelihoodHook(),
    )


class FiniteHypothesisTests(unittest.TestCase):
    def test_bool_hypotheses_are_enumerated_deterministically(self):
        domain, story = make_finite_domain_and_story()
        cell = StateCellRef(
            EntityRef("subject", "Subject"),
            "subject.flag",
        )
        view = uncertain.uncertain_epistemic_state(
            story,
            domain,
            "bob",
            make_finite_model(),
            (cell,),
        ).cells[cell]
        self.assertEqual(
            {mass.value.value for mass in view.prior.masses},
            {False, True},
        )

    def test_entity_ref_hypotheses_are_enumerated_deterministically(self):
        domain, story = make_finite_domain_and_story()
        cell = StateCellRef(
            EntityRef("subject", "Subject"),
            "subject.target",
        )
        view = uncertain.uncertain_epistemic_state(
            story,
            domain,
            "bob",
            make_finite_model(),
            (cell,),
        ).cells[cell]
        self.assertEqual(
            {
                mass.value.value.entity_id
                for mass in view.prior.masses
            },
            {"candidate-a", "candidate-b"},
        )

    def test_integer_and_text_spaces_fail_closed_before_prior_hook(self):
        domain, story = make_finite_domain_and_story()

        class ExplodingPriorHook:
            def __call__(
                self,
                agent_id,
                cell,
                hypotheses,
                parameters,
            ):
                raise AssertionError(
                    "unsupported space reached prior hook"
                )

        model = uncertain.UncertainBeliefModelSpec(
            "unsupported",
            "1",
            {},
            ExplodingPriorHook(),
            ParameterLikelihoodHook(),
        )
        for variable in ("subject.count", "subject.label"):
            with self.subTest(variable=variable):
                cell = StateCellRef(
                    EntityRef("subject", "Subject"),
                    variable,
                )
                with self.assertRaises(
                    uncertain.UncertainBeliefResolutionError
                ):
                    uncertain.uncertain_epistemic_state(
                        story,
                        domain,
                        "bob",
                        model,
                        (cell,),
                    )


class NonMappingPriorHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        return (0.2, 0.5, 0.3)


class MissingPriorKeyHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        return {
            _value_hash(value): 1.0 / len(hypotheses)
            for value in hypotheses[:-1]
        }


class ExtraPriorKeyHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        result = {
            _value_hash(value): 1.0 / len(hypotheses)
            for value in hypotheses
        }
        result["sha256:" + "0" * 64] = 0.0
        return result


class NonNormalizedPriorHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        return {
            _value_hash(value): 0.1
            for value in hypotheses
        }


class ParameterRawPriorHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        raw = parameters["raw_prior"]
        return {
            _value_hash(value): raw[_value_token(value)]
            for value in hypotheses
        }


class ParameterRawLikelihoodHook:
    def __call__(
        self,
        agent_id,
        evidence,
        hypotheses,
        parameters,
    ):
        raw = parameters["raw_likelihood"]
        return {
            _value_hash(value): raw[_value_token(value)]
            for value in hypotheses
        }


class NaNPriorHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        result = {
            _value_hash(value): 1.0 / len(hypotheses)
            for value in hypotheses
        }
        result[_value_hash(hypotheses[0])] = math.nan
        return result


class InfinitePriorHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        result = {
            _value_hash(value): 1.0 / len(hypotheses)
            for value in hypotheses
        }
        result[_value_hash(hypotheses[0])] = math.inf
        return result


class NaNLikelihoodHook:
    def __call__(self, agent_id, evidence, hypotheses, parameters):
        result = {
            _value_hash(value): 0.5
            for value in hypotheses
        }
        result[_value_hash(hypotheses[0])] = math.nan
        return result


class InfiniteLikelihoodHook:
    def __call__(self, agent_id, evidence, hypotheses, parameters):
        result = {
            _value_hash(value): 0.5
            for value in hypotheses
        }
        result[_value_hash(hypotheses[0])] = math.inf
        return result


class ZeroLikelihoodHook:
    def __call__(self, agent_id, evidence, hypotheses, parameters):
        return {
            _value_hash(value): 0.0
            for value in hypotheses
        }


class UncertainBeliefResolutionValidationTests(unittest.TestCase):
    def _state(self, model):
        story = make_test_story(
            receive=True,
            bob_observes_failure=False,
        )
        return uncertain.uncertain_epistemic_state(
            story,
            make_test_domain(),
            "bob",
            model,
            (target_cell(),),
            at_time=3,
        )

    def test_prior_shape_and_normalization_fail_closed(self):
        for hook in (
            NonMappingPriorHook(),
            MissingPriorKeyHook(),
            ExtraPriorKeyHook(),
            NonNormalizedPriorHook(),
        ):
            with self.subTest(hook=hook.__class__.__name__):
                with self.assertRaises(
                    uncertain.UncertainBeliefResolutionError
                ):
                    self._state(
                        make_uncertain_model(prior_hook=hook)
                    )

    def test_prior_numeric_boundaries_fail_closed(self):
        baseline = {
            "healthy": 0.20,
            "failed": 0.50,
            "recovered": 0.30,
        }
        for bad in (-0.01, 1.01):
            with self.subTest(bad=bad):
                raw = dict(baseline)
                raw["failed"] = bad
                model = make_uncertain_model(
                    parameters={
                        "raw_prior": raw,
                        "default_likelihood_by_value": {
                            "healthy": 1.0,
                            "failed": 1.0,
                            "recovered": 1.0,
                        },
                        "likelihood_by_evidence": {},
                    },
                    prior_hook=ParameterRawPriorHook(),
                )
                with self.assertRaises(
                    uncertain.UncertainBeliefResolutionError
                ):
                    self._state(model)
        for hook in (NaNPriorHook(), InfinitePriorHook()):
            with self.subTest(hook=hook.__class__.__name__):
                with self.assertRaises(
                    uncertain.UncertainBeliefResolutionError
                ):
                    self._state(make_uncertain_model(prior_hook=hook))

    def test_likelihood_numeric_boundaries_fail_closed(self):
        baseline = {
            "healthy": 0.10,
            "failed": 0.20,
            "recovered": 0.90,
        }
        for bad in (-0.01, 1.01):
            with self.subTest(bad=bad):
                raw = dict(baseline)
                raw["failed"] = bad
                parameters = {
                    "prior_by_value": {
                        "healthy": 0.20,
                        "failed": 0.50,
                        "recovered": 0.30,
                    },
                    "raw_likelihood": raw,
                    "likelihood_by_evidence": {},
                    "default_likelihood_by_value": {},
                }
                model = make_uncertain_model(
                    parameters=parameters,
                    likelihood_hook=ParameterRawLikelihoodHook(),
                )
                with self.assertRaises(
                    uncertain.UncertainBeliefResolutionError
                ):
                    self._state(model)
        for hook in (NaNLikelihoodHook(), InfiniteLikelihoodHook()):
            with self.subTest(hook=hook.__class__.__name__):
                with self.assertRaises(
                    uncertain.UncertainBeliefResolutionError
                ):
                    self._state(make_uncertain_model(likelihood_hook=hook))

    def test_zero_posterior_mass_fails_closed(self):
        with self.assertRaises(
            uncertain.UncertainBeliefResolutionError
        ):
            self._state(
                make_uncertain_model(
                    likelihood_hook=ZeroLikelihoodHook()
                )
            )

    def test_structural_domain_identity_error_keeps_existing_type(self):
        story = make_test_story(
            receive=True,
            bob_observes_failure=False,
        )
        domain = make_test_domain()
        bad_story = replace(
            story,
            domain_spec_hash="sha256:" + "0" * 64,
        )
        with self.assertRaisesRegex(
            ValueError,
            "domain identity",
        ):
            uncertain.uncertain_epistemic_state(
                bad_story,
                domain,
                "bob",
                make_uncertain_model(),
                (target_cell(),),
            )


if __name__ == "__main__":
    unittest.main()
