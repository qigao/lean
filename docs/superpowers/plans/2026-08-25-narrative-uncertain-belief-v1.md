# Narrative Uncertain Belief V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a narrative-native probabilistic epistemic projection that derives provenance-linked finite belief distributions from exactly the evidence available to one agent, without changing existing objective/discrete epistemic semantics.

**Architecture:** Keep `epistemic_state()` canonical and unchanged. Add `narrative_dynamics/narrative/uncertain.py` as a parallel projection that consumes the existing ordered `EpistemicEvidence` stream, enumerates finite hypotheses for explicitly tracked cells, applies explicit content-hashed prior/likelihood assumptions, and delegates Bayesian normalization to the existing `hypothesis_competition.posterior_distribution()` primitive. Do not integrate probability into decisions, interventions, persuasion, source reliability, memory, institutional rules, or Lean in V1.

**Tech Stack:** Python 3 stdlib (`dataclasses`, `math`, `collections.abc`, `types`), existing Generic Narrative Engine records and replay APIs, `narrative_dynamics.contracts.stable_content_hash`, `narrative_dynamics.attestation.measure_implementation`, existing `hypothesis_competition.posterior_distribution`, `unittest`, GitHub Actions proof workflow.

**Spec:** `docs/superpowers/specs/2026-08-25-narrative-uncertain-belief-v1-design.md`

## Global Constraints

- Preserve `GENERIC_NARRATIVE_SCHEMA_VERSION == 1`; do not add floats or probability fields to GenericNarrative IR.
- Do not modify the meaning or serialization of `objective_state()`, `direct_state()`, `epistemic_state()`, `EpistemicCellView`, existing decisions, analyses, or interventions.
- `uncertain_epistemic_state()` must consume `epistemic_state(...).evidence_history`; it must not call `objective_state()` or receive canonical truth through the official probability-hook interface.
- Priors and conditional likelihoods are explicit model assumptions. The engine must not hardcode testimony confidence, source trust, or a persuasion outcome.
- Priors must already be normalized; malformed priors reject rather than being auto-normalized.
- Conditional likelihoods are `P(evidence | hypothesis)`, each finite and in `[0, 1]`; the likelihood vector does not need to sum to one.
- V1 finite hypothesis spaces are exactly `enum`, `bool`, and `entity_ref`; `integer`, `text`, and finite spaces with fewer than two values fail closed.
- All behavior-affecting model configuration must be in explicit recursively frozen `parameters`; hidden mutable hook state is outside the V1 contract.
- Probability-model identity must change when hook identity or canonical parameters change.
- Export the new API from `narrative_dynamics.narrative` only; do not export it from top-level `narrative_dynamics`.
- Do not change `ir.py`, `domain.py`, `replay.py`, decision machinery, analysis/intervention machinery, film fixtures, or Lean unless a failing test proves the approved design cannot be implemented without such a change. If GenericNarrative schema, discrete epistemic semantics, or decision contracts would need to change, stop and reclassify instead of expanding V1.
- Use `_PROBABILITY_TOLERANCE = 1e-12`, `math.isclose(..., rel_tol=0.0, abs_tol=_PROBABILITY_TOLERANCE)` for normalized-distribution checks.
- Follow strict TDD: the first implementation commit after these design/plan commits is tests only; observe the intended RED in a fresh PR proof before creating `uncertain.py`.
- GREEN requires a fresh full PR proof: Lean conformance success, full Lean success, Lean theorem suite success, all Python tests success, StoryState success, Testimony success, and feature-tree == tested PR-checkout-tree.
- Do not advance `proof/narrative-dynamics-v0` or `master` as part of this plan. Stop at integration-ready.

---

## File Structure

- Create `tests/test_narrative_uncertain_belief.py` — all new uncertain-belief behavioral, validation, provenance, finite-hypothesis, time/reception, and model-identity tests. Keep new test fixtures local so existing shared narrative fixtures remain untouched.
- Modify `tests/test_narrative_trust_api.py` — enlarge the exact `narrative_dynamics.narrative` public-surface expectation by exactly the nine approved uncertain-belief symbols while preserving root-package isolation.
- Create `narrative_dynamics/narrative/uncertain.py` — immutable uncertain-belief records, canonical parameter freezing, model identity, finite hypothesis enumeration, probability-vector validation, sequential Bayesian replay, and typed resolution errors.
- Modify `narrative_dynamics/narrative/__init__.py` — import and export exactly the nine approved public symbols.
- Do not modify `narrative_dynamics/__init__.py`.

### Approved public symbols

```python
BeliefMass
BeliefLikelihood
BeliefDistribution
BeliefUpdateStep
UncertainBeliefCellView
UncertainBeliefState
UncertainBeliefModelSpec
UncertainBeliefResolutionError
uncertain_epistemic_state
```

---

### Task 1: Freeze the complete test-only RED and observe it in CI

**Files:**
- Create: `tests/test_narrative_uncertain_belief.py`
- Modify: `tests/test_narrative_trust_api.py`

**Interfaces:**
- Consumes: current `StateCellRef`, `TypedValue`, `EntityRef`, `GenericNarrative`, `DomainSpec`, `epistemic_state`, `objective_state`, `stable_content_hash`, `make_test_domain()`, `make_test_story()`, `target_cell()`, and existing `posterior_distribution()`.
- Produces: the frozen behavioral contract for all later tasks; no production interface exists yet.

- [ ] **Step 1: Add direct import of the missing uncertain module so discovery gives a clean module-level RED**

At the top of `tests/test_narrative_uncertain_belief.py`, import the submodule directly rather than wrapping ImportError:

```python
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
    Claim,
    Entity,
    EntityRef,
    GenericNarrative,
    Proposition,
    Reception,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.replay import epistemic_state, objective_state
from tests.narrative_test_support import make_test_domain, make_test_story, target_cell
```

This is intentional: before `uncertain.py` exists, unittest discovery should report one import error for this test module instead of many duplicate assertion failures.

- [ ] **Step 2: Add deterministic test hooks that use only explicit parameters**

Use callable classes so `measure_implementation()` can attest their module/class identity:

```python
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
        table = parameters["likelihood_by_evidence"].get(evidence.supporting_id)
        if table is None:
            table = parameters["default_likelihood_by_value"]
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
        return dict(
            sorted(
                (
                    (_value_hash(value), table[_value_token(value)])
                    for value in hypotheses
                ),
                reverse=True,
            )
        )
```

`AlternateLikelihoodHook` intentionally computes the same mapping through a distinct callable identity; the model-identity test uses it to prove hook identity is bound.

- [ ] **Step 3: Add one canonical model factory for the service-health enum fixture**

```python
def make_uncertain_model(*, parameters=None, likelihood_hook=None):
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
        prior_hook=ParameterPriorHook(),
        likelihood_hook=(
            ParameterLikelihoodHook()
            if likelihood_hook is None
            else likelihood_hook
        ),
    )
```

- [ ] **Step 4: Add prior-only and single-testimony behavior tests**

```python
class NarrativeUncertainBeliefTests(unittest.TestCase):
    def test_prior_without_evidence_is_preserved_and_discrete_replay_stays_sparse(self):
        story = make_test_story(receive=False, bob_observes_failure=False)
        domain = make_test_domain()
        model = make_uncertain_model()

        state = uncertain.uncertain_epistemic_state(
            story,
            domain,
            "bob",
            model,
            (target_cell(),),
            at_time=3,
        )
        view = state.cells[target_cell()]
        self.assertEqual(view.updates, ())
        self.assertEqual(view.posterior, view.prior)
        self.assertAlmostEqual(
            view.posterior.probability_of(TypedValue("HealthState", "failed")),
            0.50,
        )
        self.assertNotIn(
            target_cell(),
            epistemic_state(story, domain, "bob", at_time=3).cells,
        )

    def test_received_claim_updates_posterior_with_exact_provenance(self):
        story = make_test_story(receive=True, bob_observes_failure=False)
        domain = make_test_domain()
        model = make_uncertain_model()

        state = uncertain.uncertain_epistemic_state(
            story, domain, "bob", model, (target_cell(),), at_time=3
        )
        view = state.cells[target_cell()]
        expected = posterior_distribution(
            {
                _value_hash(TypedValue("HealthState", "healthy")): 0.20,
                _value_hash(TypedValue("HealthState", "failed")): 0.50,
                _value_hash(TypedValue("HealthState", "recovered")): 0.30,
            },
            {
                _value_hash(TypedValue("HealthState", "healthy")): 0.10,
                _value_hash(TypedValue("HealthState", "failed")): 0.20,
                _value_hash(TypedValue("HealthState", "recovered")): 0.90,
            },
        )
        self.assertEqual(len(view.updates), 1)
        step = view.updates[0]
        self.assertEqual(step.evidence.supporting_id, "c1")
        self.assertEqual(step.evidence.source_agent, "alice")
        self.assertEqual(step.evidence.logical_time, 3)
        self.assertEqual(step.evidence.provenance_refs, ("c1", "e2"))
        for value in ("healthy", "failed", "recovered"):
            typed = TypedValue("HealthState", value)
            self.assertAlmostEqual(
                view.posterior.probability_of(typed),
                expected[_value_hash(typed)],
            )
```

- [ ] **Step 5: Add reception, objective-isolation, conflicting-evidence, and cutoff tests**

```python
    def test_reception_changes_belief_without_changing_objective_state(self):
        received = make_test_story(receive=True, bob_observes_failure=False)
        unreceived = make_test_story(receive=False, bob_observes_failure=False)
        domain = make_test_domain()
        model = make_uncertain_model()

        left = uncertain.uncertain_epistemic_state(
            received, domain, "bob", model, (target_cell(),), at_time=3
        )
        right = uncertain.uncertain_epistemic_state(
            unreceived, domain, "bob", model, (target_cell(),), at_time=3
        )
        self.assertNotEqual(
            left.cells[target_cell()].posterior,
            right.cells[target_cell()].posterior,
        )
        self.assertEqual(
            objective_state(received, domain, at_time=3),
            objective_state(unreceived, domain, at_time=3),
        )

    def test_conflicting_claims_remain_discrete_conflict_but_receive_graded_posterior(self):
        story = make_test_story(
            claim_value="recovered",
            second_claim_value="failed",
            bob_observes_failure=False,
        )
        domain = make_test_domain()
        model = make_uncertain_model()

        discrete = epistemic_state(story, domain, "bob", at_time=4)
        self.assertEqual(discrete.cells[target_cell()].status, "conflicted")

        uncertain_state = uncertain.uncertain_epistemic_state(
            story, domain, "bob", model, (target_cell(),), at_time=4
        )
        view = uncertain_state.cells[target_cell()]
        self.assertEqual(
            tuple(step.evidence.supporting_id for step in view.updates),
            ("c1", "c2"),
        )
        probabilities = tuple(mass.probability for mass in view.posterior.masses)
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertTrue(all(0.0 < value < 1.0 for value in probabilities))

    def test_time_cutoff_excludes_later_claim_update(self):
        story = make_test_story(
            claim_value="recovered",
            second_claim_value="failed",
            bob_observes_failure=False,
        )
        domain = make_test_domain()
        model = make_uncertain_model()

        early = uncertain.uncertain_epistemic_state(
            story, domain, "bob", model, (target_cell(),), at_time=3
        )
        late = uncertain.uncertain_epistemic_state(
            story, domain, "bob", model, (target_cell(),), at_time=4
        )
        self.assertEqual(
            tuple(step.evidence.supporting_id for step in early.cells[target_cell()].updates),
            ("c1",),
        )
        self.assertEqual(
            tuple(step.evidence.supporting_id for step in late.cells[target_cell()].updates),
            ("c1", "c2"),
        )
        self.assertNotEqual(
            early.cells[target_cell()].posterior,
            late.cells[target_cell()].posterior,
        )
```

- [ ] **Step 6: Add a no-event finite-type fixture and enumeration tests**

Keep the fixture local to this test file:

```python
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
            ValueTypeSpec("CandidateRef", "entity_ref", entity_type="Candidate"),
            ValueTypeSpec("Count", "integer"),
            ValueTypeSpec("Label", "text"),
        ),
        state_variables=(
            StateVariableSpec("subject.flag", "Subject", "Flag"),
            StateVariableSpec("subject.target", "Subject", "CandidateRef"),
            StateVariableSpec("subject.count", "Subject", "Count"),
            StateVariableSpec("subject.label", "Subject", "Label"),
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
        (), (), (), (), (),
    )
    return domain, story
```

Then test bool/entity-ref enumeration and integer/text rejection:

```python
class FiniteHypothesisTests(unittest.TestCase):
    def test_bool_and_entity_ref_hypotheses_are_canonical(self):
        domain, story = make_finite_domain_and_story()
        flag_cell = StateCellRef(EntityRef("subject", "Subject"), "subject.flag")
        target = StateCellRef(EntityRef("subject", "Subject"), "subject.target")
        model = uncertain.UncertainBeliefModelSpec(
            "finite", "1",
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
        state = uncertain.uncertain_epistemic_state(
            story, domain, "bob", model, (flag_cell, target)
        )
        self.assertEqual(
            {_value_token(item.value) for item in state.cells[flag_cell].prior.masses},
            {"false", "true"},
        )
        self.assertEqual(
            tuple(
                sorted(_value_token(item.value) for item in state.cells[target].prior.masses)
            ),
            ("candidate-a", "candidate-b"),
        )

    def test_integer_and_text_cells_fail_closed(self):
        domain, story = make_finite_domain_and_story()
        model = make_uncertain_model()
        for variable in ("subject.count", "subject.label"):
            with self.subTest(variable=variable):
                cell = StateCellRef(EntityRef("subject", "Subject"), variable)
                with self.assertRaises(uncertain.UncertainBeliefResolutionError):
                    uncertain.uncertain_epistemic_state(
                        story, domain, "bob", model, (cell,)
                    )
```

During implementation, use a finite-model parameter table appropriate to the custom cells rather than letting `make_uncertain_model()` supply service-health priors; the test must fail for unsupported type before invoking the prior hook.

- [ ] **Step 7: Add record/model identity and probability-validation tests**

Tests must cover:

```python
class UncertainBeliefContractTests(unittest.TestCase):
    def test_model_identity_binds_code_and_explicit_parameters(self):
        first = make_uncertain_model()
        same = make_uncertain_model()
        changed_parameters = make_uncertain_model(
            parameters={
                **dict(first.parameters),
                "prior_by_value": {
                    "healthy": 0.30,
                    "failed": 0.40,
                    "recovered": 0.30,
                },
            }
        )
        changed_hook = make_uncertain_model(likelihood_hook=AlternateLikelihoodHook())
        self.assertEqual(first.content_hash, same.content_hash)
        self.assertNotEqual(first.content_hash, changed_parameters.content_hash)
        self.assertNotEqual(first.content_hash, changed_hook.content_hash)

    def test_model_parameters_are_detached_and_noncanonical_values_reject(self):
        raw = {
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
            "likelihood_by_evidence": {},
        }
        model = make_uncertain_model(parameters=raw)
        before = model.content_hash
        raw["prior_by_value"]["failed"] = 0.99
        self.assertEqual(model.content_hash, before)
        with self.assertRaises((TypeError, ValueError)):
            uncertain.UncertainBeliefModelSpec(
                "bad", "1", {"bad": {1, 2}}, ParameterPriorHook(), ParameterLikelihoodHook()
            )
        with self.assertRaises((TypeError, ValueError)):
            uncertain.UncertainBeliefModelSpec(
                "bad", "1", {"bad": math.nan}, ParameterPriorHook(), ParameterLikelihoodHook()
            )
```

Add malformed-hook classes and use `subTest` to lock exact failure classes for missing/extra keys, non-mapping returns, non-normalized priors, out-of-range/non-finite probabilities, out-of-range/non-finite likelihoods, and zero posterior mass. Every probabilistic failure after narrative validation must be `UncertainBeliefResolutionError`.

Example malformed hooks:

```python
class NonMappingPriorHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        return (0.2, 0.5, 0.3)


class MissingPriorKeyHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        values = list(hypotheses)
        return {_value_hash(value): 1.0 / len(values) for value in values[:-1]}


class ExtraPriorKeyHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        result = {_value_hash(value): 1.0 / len(hypotheses) for value in hypotheses}
        result["sha256:" + "0" * 64] = 0.0
        return result


class ZeroLikelihoodHook:
    def __call__(self, agent_id, evidence, hypotheses, parameters):
        return {_value_hash(value): 0.0 for value in hypotheses}
```

Also construct `BeliefMass`, `BeliefLikelihood`, and `BeliefDistribution` directly to lock immutability, deterministic ordering, range checks, duplicate-value rejection, and the `probability_of()` exact-value lookup.

- [ ] **Step 8: Enlarge the exact narrative public-surface expectation in the RED commit**

In `tests/test_narrative_trust_api.py`, add exactly these names to `_EXPECTED_PUBLIC_API`:

```python
    "BeliefMass",
    "BeliefLikelihood",
    "BeliefDistribution",
    "BeliefUpdateStep",
    "UncertainBeliefCellView",
    "UncertainBeliefState",
    "UncertainBeliefModelSpec",
    "UncertainBeliefResolutionError",
    "uncertain_epistemic_state",
```

Do not change the assertion that every narrative symbol is absent from top-level `narrative_dynamics`.

- [ ] **Step 9: Commit tests only**

```bash
git add tests/test_narrative_uncertain_belief.py tests/test_narrative_trust_api.py
git diff --cached --name-only
git commit -m "test: define narrative uncertain belief RED"
```

Expected staged names: exactly the two test files above. The approved spec/plan commits pre-exist on the branch; this new RED commit itself must contain no production file.

- [ ] **Step 10: Open a draft PR against the research branch and observe a fresh RED proof**

Use these exact PR semantics:

```text
base: proof/narrative-dynamics-v0
head: work/narrative-uncertain-belief-v1
draft: true
title: RED: add narrative uncertain belief v1
```

Expected fresh proof shape:

```text
Lean conformance: success
Full Lean build: success
Lean theorem suite: success
Python: failure only from
  1. test_narrative_uncertain_belief module import because narrative_dynamics.narrative.uncertain does not exist
  2. exact narrative public-surface test because the nine approved exports do not exist
No existing test regression
```

Do not create `uncertain.py` until that exact RED is observed and recorded in the PR body.

---

### Task 2: Implement immutable records, canonical parameters, and model identity

**Files:**
- Create: `narrative_dynamics/narrative/uncertain.py`
- Test: `tests/test_narrative_uncertain_belief.py`

**Interfaces:**
- Consumes: `StateCellRef`, `TypedValue`, `EpistemicEvidence`, `stable_content_hash`, `measure_implementation`.
- Produces: `BeliefMass`, `BeliefLikelihood`, `BeliefDistribution`, `BeliefUpdateStep`, `UncertainBeliefCellView`, `UncertainBeliefState`, `UncertainBeliefModelSpec`, `UncertainBeliefResolutionError`. Later tasks consume these exact names.

- [ ] **Step 1: Add module constants, text/probability validation, and canonical parameter freezing**

Start `uncertain.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType

from hypothesis_competition import posterior_distribution
from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import DomainSpec, validate_narrative
from narrative_dynamics.narrative.ir import (
    EntityRef,
    GenericNarrative,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.replay import EpistemicEvidence, epistemic_state


_PROBABILITY_TOLERANCE = 1e-12


class UncertainBeliefResolutionError(ValueError):
    """An explicit uncertain-belief model could not produce a valid state."""


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _probability(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return result


def _freeze_parameter(value: object, *, label: str) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} keys must be non-empty strings")
            frozen[key] = _freeze_parameter(item, label=f"{label}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_parameter(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{label} contains a non-canonical value")
```

- [ ] **Step 2: Implement the scalar and normalized-distribution records**

Use these exact fields:

```python
@dataclass(frozen=True)
class BeliefMass:
    value: TypedValue
    probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.value, TypedValue):
            raise TypeError("belief mass value must be TypedValue")
        object.__setattr__(
            self,
            "probability",
            _probability(self.probability, label="belief mass probability"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"value": self.value.to_dict(), "probability": self.probability}


@dataclass(frozen=True)
class BeliefLikelihood:
    value: TypedValue
    likelihood: float

    def __post_init__(self) -> None:
        if not isinstance(self.value, TypedValue):
            raise TypeError("belief likelihood value must be TypedValue")
        object.__setattr__(
            self,
            "likelihood",
            _probability(self.likelihood, label="belief likelihood"),
        )

    def to_dict(self) -> dict[str, object]:
        return {"value": self.value.to_dict(), "likelihood": self.likelihood}
```

For `BeliefDistribution`, canonicalize masses by `stable_content_hash(mass.value.to_dict())`, reject fewer than two or duplicate values, require total probability to equal 1 within the global tolerance, and expose:

```python
@dataclass(frozen=True)
class BeliefDistribution:
    cell: StateCellRef
    masses: tuple[BeliefMass, ...]

    def probability_of(self, value: TypedValue) -> float:
        for mass in self.masses:
            if mass.value == value:
                return mass.probability
        raise KeyError("belief distribution does not contain the requested value")
```

`to_dict()` must serialize `cell` plus the canonical mass sequence.

- [ ] **Step 3: Implement update/cell/state records with chain invariants**

`BeliefUpdateStep` must require one `EpistemicEvidence`, matching-cell prior/posterior distributions, one likelihood entry per prior hypothesis, and the same exact hypothesis set in prior/likelihood/posterior.

`UncertainBeliefCellView` must require prior/posterior for its own cell and verify update chaining:

```python
current = self.prior
for step in self.updates:
    if step.cell != self.cell or step.prior != current:
        raise ValueError("uncertain belief update chain is discontinuous")
    current = step.posterior
if current != self.posterior:
    raise ValueError("uncertain belief posterior must equal the update-chain tail")
```

`UncertainBeliefState` must freeze its `cells` mapping, require each mapping key to equal `view.cell`, and require `model_hash` to begin with `sha256:`.

- [ ] **Step 4: Implement `UncertainBeliefModelSpec` with explicit configuration identity**

```python
@dataclass(frozen=True)
class UncertainBeliefModelSpec:
    model_id: str
    version: str
    parameters: Mapping[str, object]
    prior_hook: object = field(compare=False, repr=False)
    likelihood_hook: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="uncertain model id"))
        object.__setattr__(self, "version", _text(self.version, label="uncertain model version"))
        if not isinstance(self.parameters, Mapping):
            raise TypeError("uncertain model parameters must be a mapping")
        frozen = _freeze_parameter(self.parameters, label="uncertain model parameters")
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "parameters", frozen)
        if not callable(self.prior_hook) or not callable(self.likelihood_hook):
            raise TypeError("uncertain model hooks must be callable")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "parameters": self.parameters,
            "prior_hook_identity": measure_implementation(self.prior_hook).manifest_identity(),
            "likelihood_hook_identity": measure_implementation(self.likelihood_hook).manifest_identity(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())
```

Do not hash hook instance attributes as an undocumented substitute for `parameters`.

- [ ] **Step 5: Run only record/model-contract tests**

```bash
python3 -m unittest -v \
  tests.test_narrative_uncertain_belief.UncertainBeliefContractTests
```

Expected: record/model identity and canonical-parameter tests pass. Replay-oriented tests may still fail because `uncertain_epistemic_state` is not implemented yet.

- [ ] **Step 6: Commit the independently reviewable records/model layer**

```bash
git add narrative_dynamics/narrative/uncertain.py
git commit -m "feat: add uncertain belief records and model identity"
```

---

### Task 3: Implement finite hypothesis enumeration and prior-only uncertain replay

**Files:**
- Modify: `narrative_dynamics/narrative/uncertain.py`
- Test: `tests/test_narrative_uncertain_belief.py`

**Interfaces:**
- Consumes: `DomainSpec._state_variable`, `DomainSpec._value_type`, declared story entities, `epistemic_state()`.
- Produces: `_hypotheses_for_cell(...)`, `_prior_distribution(...)`, and public `uncertain_epistemic_state(...)` with correct no-evidence behavior. Task 4 extends the same function with sequential evidence updates.

- [ ] **Step 1: Add tracked-cell and finite-hypothesis validation**

Implement:

```python
def _hypotheses_for_cell(
    story: GenericNarrative,
    domain: DomainSpec,
    cell: StateCellRef,
) -> tuple[TypedValue, ...]:
    entities = {entity.id: entity for entity in story.entities}
    subject = entities.get(cell.subject.entity_id)
    if subject is None or subject.type_name != cell.subject.entity_type:
        raise UncertainBeliefResolutionError("tracked cell subject is not declared exactly")
    try:
        variable = domain._state_variable(cell.state_variable)
    except ValueError as error:
        raise UncertainBeliefResolutionError("tracked state variable is undeclared") from error
    if subject.type_name != variable.subject_type:
        raise UncertainBeliefResolutionError("tracked cell subject type does not match state variable")
    value_type = domain._value_type(variable.value_type)

    if value_type.kind == "enum":
        hypotheses = tuple(
            TypedValue(value_type.name, raw)
            for raw in value_type.allowed_values
        )
    elif value_type.kind == "bool":
        hypotheses = (
            TypedValue(value_type.name, False),
            TypedValue(value_type.name, True),
        )
    elif value_type.kind == "entity_ref":
        hypotheses = tuple(
            TypedValue(
                value_type.name,
                EntityRef(entity.id, entity.type_name),
            )
            for entity in sorted(
                story.entities,
                key=lambda item: (item.type_name, item.id),
            )
            if entity.type_name == value_type.entity_type
        )
    else:
        raise UncertainBeliefResolutionError(
            "tracked state variable is not finitely enumerable in V1"
        )
    if len(hypotheses) < 2:
        raise UncertainBeliefResolutionError(
            "uncertain belief requires at least two finite hypotheses"
        )
    return tuple(
        sorted(hypotheses, key=lambda item: stable_content_hash(item.to_dict()))
    )
```

- [ ] **Step 2: Add exact-key vector validation and prior construction**

```python
def _hypothesis_map(hypotheses: tuple[TypedValue, ...]) -> dict[str, TypedValue]:
    return {stable_content_hash(value.to_dict()): value for value in hypotheses}


def _validated_vector(
    raw: object,
    hypotheses: tuple[TypedValue, ...],
    *,
    label: str,
) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise UncertainBeliefResolutionError(f"{label} must be a mapping")
    expected = _hypothesis_map(hypotheses)
    if set(raw) != set(expected):
        raise UncertainBeliefResolutionError(
            f"{label} keys must match the finite hypothesis set exactly"
        )
    result: dict[str, float] = {}
    for key in expected:
        try:
            result[key] = _probability(raw[key], label=f"{label} weight")
        except (TypeError, ValueError) as error:
            raise UncertainBeliefResolutionError(str(error)) from error
    return result
```

For a prior, additionally require:

```python
if not math.isclose(
    sum(vector.values()),
    1.0,
    rel_tol=0.0,
    abs_tol=_PROBABILITY_TOLERANCE,
):
    raise UncertainBeliefResolutionError("uncertain prior must sum to 1")
```

Construct the `BeliefDistribution` from the engine-supplied hypothesis objects rather than any hook-supplied values.

- [ ] **Step 3: Implement the public replay function through the prior-only path**

Use the exact public signature:

```python
def uncertain_epistemic_state(
    story: GenericNarrative,
    domain: DomainSpec,
    agent_id: str,
    model: UncertainBeliefModelSpec,
    tracked_cells: tuple[StateCellRef, ...],
    *,
    at_time: int | None = None,
) -> UncertainBeliefState:
```

Required first-half behavior:

```python
validate_narrative(story, domain)
if not isinstance(model, UncertainBeliefModelSpec):
    raise TypeError("uncertain replay requires UncertainBeliefModelSpec")
cells = tuple(tracked_cells)
if not cells or any(not isinstance(cell, StateCellRef) for cell in cells):
    raise TypeError("tracked cells must be a non-empty tuple of StateCellRef values")
if len(set(cells)) != len(cells):
    raise ValueError("tracked cells must be unique")

admitted = epistemic_state(story, domain, agent_id, at_time=at_time)
```

`epistemic_state()` remains responsible for agent declaration and cutoff validation. For each tracked cell, enumerate hypotheses, call `model.prior_hook(agent_id, cell, hypotheses, model.parameters)`, validate the prior exactly, and create a view with `posterior == prior` and `updates == ()` when no evidence for that cell is admitted.

The returned state must include the complete `admitted.evidence_history`, not only evidence for tracked cells.

- [ ] **Step 4: Run prior/enumeration tests**

```bash
python3 -m unittest -v \
  tests.test_narrative_uncertain_belief.NarrativeUncertainBeliefTests.test_prior_without_evidence_is_preserved_and_discrete_replay_stays_sparse \
  tests.test_narrative_uncertain_belief.FiniteHypothesisTests
```

Expected: prior-only, bool/entity-ref, and unsupported integer/text tests pass.

- [ ] **Step 5: Commit finite hypotheses and prior-only replay**

```bash
git add narrative_dynamics/narrative/uncertain.py
git commit -m "feat: add finite uncertain belief priors"
```

---

### Task 4: Add sequential Bayesian updates with exact narrative provenance

**Files:**
- Modify: `narrative_dynamics/narrative/uncertain.py`
- Test: `tests/test_narrative_uncertain_belief.py`

**Interfaces:**
- Consumes: Task 3 hypothesis/prior helpers, `EpistemicState.evidence_history`, `posterior_distribution()`.
- Produces: completed `uncertain_epistemic_state()` with ordered `BeliefUpdateStep` traces.

- [ ] **Step 1: Add likelihood validation without normalization**

Reuse `_validated_vector(..., label="uncertain likelihood")`. Do not check that likelihoods sum to one. Convert the validated vector to canonical `BeliefLikelihood` records using the engine-owned hypotheses.

- [ ] **Step 2: Apply only admitted evidence for the exact tracked cell**

Inside each cell replay:

```python
current = prior
updates: list[BeliefUpdateStep] = []
for evidence in admitted.evidence_history:
    if evidence.cell != cell:
        continue
    hypotheses = tuple(mass.value for mass in current.masses)
    raw = model.likelihood_hook(
        agent_id,
        evidence,
        hypotheses,
        model.parameters,
    )
    likelihood_vector = _validated_vector(
        raw,
        hypotheses,
        label="uncertain likelihood",
    )
    prior_vector = {
        stable_content_hash(mass.value.to_dict()): mass.probability
        for mass in current.masses
    }
    try:
        posterior_vector = posterior_distribution(prior_vector, likelihood_vector)
    except ValueError as error:
        raise UncertainBeliefResolutionError(
            "uncertain evidence update has zero posterior mass"
        ) from error
    posterior = _distribution_from_vector(cell, hypotheses, posterior_vector)
    likelihoods = tuple(
        BeliefLikelihood(value, likelihood_vector[stable_content_hash(value.to_dict())])
        for value in hypotheses
    )
    updates.append(BeliefUpdateStep(evidence, current, likelihoods, posterior))
    current = posterior
```

Do not inspect `evidence.relation` in the engine. The likelihood model sees the complete admitted `EpistemicEvidence` and decides how that evidence bears on each hypothesis.

- [ ] **Step 3: Preserve canonical evidence ordering rather than re-sorting uncertain updates**

`admitted.evidence_history` is already ordered by existing replay semantics. Iterate it directly. This ensures reception and `at_time` behavior cannot diverge from discrete replay.

- [ ] **Step 4: Run the behavior/provenance tests**

```bash
python3 -m unittest -v \
  tests.test_narrative_uncertain_belief.NarrativeUncertainBeliefTests
```

Expected: prior-only, single claim, reception/objective isolation, conflicting claims, and time-cutoff tests all pass. Existing discrete conflict assertions remain unchanged.

- [ ] **Step 5: Commit sequential uncertain replay**

```bash
git add narrative_dynamics/narrative/uncertain.py
git commit -m "feat: update uncertain beliefs from narrative evidence"
```

---

### Task 5: Close all probabilistic fail-closed boundaries

**Files:**
- Modify: `narrative_dynamics/narrative/uncertain.py`
- Test: `tests/test_narrative_uncertain_belief.py`

**Interfaces:**
- Consumes: Task 2/3/4 validation helpers and public replay API.
- Produces: approved `UncertainBeliefResolutionError` behavior for all probabilistic resolution failures.

- [ ] **Step 1: Convert malformed hook outputs to the typed uncertain error**

Wrap only model-output validation and Bayesian-resolution errors. Do not relabel `validate_narrative()` structural errors.

Use:

```python
def _call_prior_hook(model, agent_id, cell, hypotheses):
    try:
        raw = model.prior_hook(agent_id, cell, hypotheses, model.parameters)
    except (TypeError, ValueError) as error:
        raise UncertainBeliefResolutionError(
            "uncertain prior hook could not produce a valid vector"
        ) from error
    return raw


def _call_likelihood_hook(model, agent_id, evidence, hypotheses):
    try:
        raw = model.likelihood_hook(
            agent_id,
            evidence,
            hypotheses,
            model.parameters,
        )
    except (TypeError, ValueError) as error:
        raise UncertainBeliefResolutionError(
            "uncertain likelihood hook could not produce a valid vector"
        ) from error
    return raw
```

Do not catch arbitrary `BaseException`, `KeyboardInterrupt`, or `SystemExit`.

- [ ] **Step 2: Lock missing/extra keys, non-mapping values, prior normalization, ranges, non-finite values, and zero posterior mass**

Run the contract test class after each minimal fix:

```bash
python3 -m unittest -v \
  tests.test_narrative_uncertain_belief.UncertainBeliefContractTests
```

Expected: every malformed probabilistic configuration fails with the specified type and valid models remain deterministic.

- [ ] **Step 3: Verify structural narrative errors keep their existing types**

Add/retain a test that changes `story.domain_spec_hash` to a wrong valid SHA-256 and assert `uncertain_epistemic_state()` raises the same `ValueError` from `validate_narrative()`, not `UncertainBeliefResolutionError`.

Example:

```python
bad_story = replace(story, domain_spec_hash="sha256:" + "0" * 64)
with self.assertRaisesRegex(ValueError, "domain identity"):
    uncertain.uncertain_epistemic_state(
        bad_story, domain, "bob", model, (target_cell(),)
    )
```

- [ ] **Step 4: Run the entire new uncertain test module**

```bash
python3 -m unittest -v tests.test_narrative_uncertain_belief
```

Expected: all new uncertain-belief tests pass locally.

- [ ] **Step 5: Commit fail-closed validation**

```bash
git add narrative_dynamics/narrative/uncertain.py tests/test_narrative_uncertain_belief.py
git commit -m "fix: enforce uncertain belief resolution boundaries"
```

If a test expectation itself is wrong because it contradicts an already-documented engine contract, diagnose first and make any test-only correction in its own commit, exactly as with previous frozen-scope corrections; do not change production to satisfy an invalid assertion.

---

### Task 6: Publish the narrative-only API without root leakage

**Files:**
- Modify: `narrative_dynamics/narrative/__init__.py`
- Test: `tests/test_narrative_trust_api.py`
- Test: `tests/test_narrative_uncertain_belief.py`

**Interfaces:**
- Consumes: all public records/functions from `narrative_dynamics.narrative.uncertain`.
- Produces: exact enlarged `narrative_dynamics.narrative.__all__`; top-level package remains unchanged.

- [ ] **Step 1: Import the approved symbols into the narrative package**

Add:

```python
from narrative_dynamics.narrative.uncertain import (
    BeliefDistribution,
    BeliefLikelihood,
    BeliefMass,
    BeliefUpdateStep,
    UncertainBeliefCellView,
    UncertainBeliefModelSpec,
    UncertainBeliefResolutionError,
    UncertainBeliefState,
    uncertain_epistemic_state,
)
```

- [ ] **Step 2: Add exactly the same nine names to `narrative.__all__`**

Keep the existing ordering grouped under replay/epistemic functionality; do not modify `narrative_dynamics/__init__.py`.

- [ ] **Step 3: Run exact public-surface and root-isolation tests**

```bash
python3 -m unittest -v \
  tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation \
  tests.test_narrative_uncertain_belief
```

Expected: PASS; every new symbol exists under `narrative_dynamics.narrative` and is absent from top-level `narrative_dynamics`.

- [ ] **Step 4: Commit the public API boundary**

```bash
git add narrative_dynamics/narrative/__init__.py
git commit -m "feat: export narrative uncertain belief api"
```

---

### Task 7: Run compatibility regression locally before fresh full proof

**Files:**
- No intended file changes.

**Interfaces:**
- Consumes: completed feature tree.
- Produces: local evidence that existing discrete/movie behavior was not changed before spending a full CI proof.

- [ ] **Step 1: Run all Python tests**

```bash
python3 -m unittest discover -s tests -v
```

Expected: all tests pass, including Knives Out, The Matrix, Memento, existing replay, decision, analysis, trust, and new uncertain-belief tests.

- [ ] **Step 2: Run the existing Lean theorem gates used by proof**

```bash
export PATH="$HOME/.elan/bin:$PATH"
lake build \
  NarrativeDynamics.Core.Belief \
  NarrativeDynamics.Core.Drive \
  NarrativeDynamics.Core.Learning
lake build
lake env lean NarrativeDynamics/Tests/StoryState.lean
lake env lean NarrativeDynamics/Tests/Testimony.lean
```

Expected: all commands succeed. Do not edit Lean to silence an unrelated failure.

- [ ] **Step 3: Lock the final diff boundary**

```bash
git diff --name-status proof/narrative-dynamics-v0...HEAD
```

Expected implementation-scope names are limited to:

```text
docs/superpowers/specs/2026-08-25-narrative-uncertain-belief-v1-design.md
docs/superpowers/plans/2026-08-25-narrative-uncertain-belief-v1.md
narrative_dynamics/narrative/uncertain.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_uncertain_belief.py
tests/test_narrative_trust_api.py
```

If any other production/test path appears, explain and justify it before proceeding; do not silently widen scope.

---

### Task 8: Obtain final fresh PR proof, exact-tree evidence, and integration-ready lock

**Files:**
- PR metadata only; no intended code changes.

**Interfaces:**
- Consumes: the final feature head and existing draft PR.
- Produces: fresh proof evidence and an integration-ready PR; does not move research/master refs.

- [ ] **Step 1: Confirm the draft PR head is the exact final feature SHA before relying on CI**

Record:

```text
base ref and SHA
feature ref and SHA
master SHA
feature tree SHA
```

The base must still be `proof/narrative-dynamics-v0`; master must remain unchanged.

- [ ] **Step 2: Observe a fresh proof triggered by the final head**

Required job results:

```text
Resolve Lean dependencies: success
Lean Python conformance vectors: success
Build Lean library: success
Lean theorem tests: success
Python numerical tests: success
Narrative story theorem tests: success
Narrative testimony theorem tests: success
```

Read the job log and explicitly verify:

```text
all Python tests: OK
new uncertain-belief tests: explicitly ok
existing Knives Out tests: ok
existing The Matrix tests: ok
existing Memento tests: ok
Lean conformance build count: success
full Lean build count: success
StoryState: success
Testimony: success
```

Do not call final GREEN from a combined status alone.

- [ ] **Step 3: Prove exact-tree**

Read the PR-checkout merge commit from the successful workflow checkout log. Fetch both commits and compare tree SHAs:

```text
feature HEAD tree == tested PR merge-checkout tree
```

A matching commit SHA is not required; matching tree SHA is required because the workflow tests the PR merge ref.

- [ ] **Step 4: Reconfirm topology and pure feature ancestry**

Compare `proof/narrative-dynamics-v0...work/narrative-uncertain-belief-v1` and require:

```text
status: ahead
ahead_by > 0
behind_by: 0
```

Re-fetch master and verify it did not move.

- [ ] **Step 5: Update PR metadata with exact evidence**

Change title to:

```text
feat: add narrative uncertain belief v1
```

PR body must record:

```text
RED commit + proof run/job + exact intended failure shape
production commits
any test-contract correction commits, if they occurred
FINAL GREEN run/job
Python test total and OK
Lean conformance/full build results
StoryState/Testimony results
feature SHA/tree
PR checkout SHA/tree
exact-tree statement
diff boundary
research/master SHAs
```

Keep the PR draft and stop.

- [ ] **Step 6: Do not integrate**

The terminal state of this plan is:

```text
Narrative Uncertain Belief V1 = integration-ready
proof/narrative-dynamics-v0 = unchanged by this task
master = unchanged
```

A later explicit user authorization may perform a separately verified non-force fast-forward of the research ref.

---

## Self-Review Results

- **Spec coverage:** Every approved V1 requirement maps to a task: separate projection (Tasks 3–4), exact evidence reuse/provenance (Task 4), explicit priors/likelihoods and no hidden normalization (Tasks 2–5), model identity including parameters (Task 2), finite hypotheses (Task 3), fail-closed errors (Task 5), exact public surface/root isolation (Task 6), no decision/persuasion/institution integration (Global Constraints), and full exact-tree proof (Task 8).
- **Scope:** One coherent subsystem only: uncertain narrative replay. Persuasion, source reliability, action feasibility, motivation, counterfactual integration, memory, and Lean remain follow-on designs.
- **Type consistency:** Public signature and record names are identical across Tasks 1–8. `BeliefLikelihood` is distinct from normalized `BeliefMass`; `parameters` are always explicit and passed to both hooks.
- **Identity consistency:** Hook module/class attestation plus recursively frozen canonical parameters enter `UncertainBeliefModelSpec.content_hash`; no supported behavior-affecting configuration lives only in mutable hook instance state.
- **Compatibility:** Existing discrete replay and GenericNarrative schema are read-only dependencies. The only approved existing production file change is `narrative_dynamics/narrative/__init__.py`.
- **TDD discipline:** The RED commit changes tests only and must be observed in a fresh PR proof before `uncertain.py` is created. Tests remain frozen through implementation except for evidence-backed test-contract corrections made in isolated commits.
