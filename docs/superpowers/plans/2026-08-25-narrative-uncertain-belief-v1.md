# Narrative Uncertain Belief V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a narrative-native probabilistic epistemic projection that derives provenance-linked finite belief distributions from exactly the evidence available to one agent, without changing existing objective/discrete epistemic semantics.

**Architecture:** Keep `epistemic_state()` canonical and unchanged. Add `narrative_dynamics/narrative/uncertain.py` as a parallel projection that consumes the existing ordered `EpistemicEvidence` stream, enumerates finite hypotheses for explicitly tracked cells, applies explicit content-hashed prior/likelihood assumptions, and delegates Bayesian normalization to `hypothesis_competition.posterior_distribution()`. V1 does not integrate probability into decisions, persuasion, source reliability, memory, institutional rules, interventions, or Lean.

**Tech Stack:** Python 3 stdlib, existing Generic Narrative Engine IR/replay APIs, `stable_content_hash`, `measure_implementation`, `posterior_distribution`, `unittest`, GitHub Actions proof workflow.

**Spec:** `docs/superpowers/specs/2026-08-25-narrative-uncertain-belief-v1-design.md`

## Global Constraints

- Preserve `GENERIC_NARRATIVE_SCHEMA_VERSION == 1`; do not add floats/probabilities to GenericNarrative IR.
- Do not change `objective_state()`, `direct_state()`, `epistemic_state()`, `EpistemicCellView`, decision models, analysis/intervention semantics, movie fixtures, or Lean.
- `uncertain_epistemic_state()` must consume `epistemic_state(...).evidence_history`; it must not call `objective_state()`.
- Prior/likelihood hooks receive only agent/cell-or-evidence/hypotheses/explicit parameters. No canonical objective truth or decision outcome is passed.
- Priors must already be normalized; reject malformed priors instead of silently renormalizing.
- Conditional likelihoods are finite `P(evidence | hypothesis)` values in `[0, 1]`; they do not need to sum to 1.
- Finite hypotheses in V1 are exactly `enum`, `bool`, and `entity_ref`; `integer`, `text`, and spaces with fewer than two hypotheses fail closed.
- All behavior-affecting model configuration must be explicit recursively frozen `parameters`; hidden mutable hook state is outside the supported contract.
- Probability-model identity must change when hook identity or canonical parameters change.
- New symbols export from `narrative_dynamics.narrative` only, never top-level `narrative_dynamics`.
- Use `_PROBABILITY_TOLERANCE = 1e-12` and `math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_TOLERANCE)` for normalized distributions.
- The first implementation commit after the design/plan commits is tests only. Observe the intended RED in a fresh PR proof before creating `uncertain.py`.
- If implementation requires GenericNarrative schema changes, discrete epistemic changes, or decision-contract changes, stop and reclassify instead of expanding V1.
- GREEN requires a fresh PR proof with Lean conformance, full Lean, theorem suite, all Python tests, StoryState, Testimony, and exact-tree success.
- Stop at integration-ready. Do not move `proof/narrative-dynamics-v0` or `master` in this plan.

---

## File Structure

- Create `tests/test_narrative_uncertain_belief.py`: all new records/model, behavior/provenance, finite-hypothesis, and probabilistic-error tests.
- Modify `tests/test_narrative_trust_api.py`: add exactly the nine approved narrative-only public symbols to the exact surface expectation.
- Create `narrative_dynamics/narrative/uncertain.py`: records, parameter freezing, identity, hypothesis enumeration, probability validation, sequential Bayesian replay.
- Modify `narrative_dynamics/narrative/__init__.py`: export exactly the nine approved symbols.
- Do not modify `narrative_dynamics/__init__.py`.

Approved symbols:

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

### Task 1: Freeze the complete test-only RED

**Files:**
- Create: `tests/test_narrative_uncertain_belief.py`
- Modify: `tests/test_narrative_trust_api.py`

**Interfaces:**
- Consumes: current narrative IR/replay APIs and existing `tests.narrative_test_support` fixtures.
- Produces: frozen behavioral contract for Tasks 2–6; no production code.

- [ ] **Step 1: Add a direct import of the missing module**

At the top of the new test file:

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
    Entity,
    EntityRef,
    GenericNarrative,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.replay import epistemic_state, objective_state
from tests.narrative_test_support import make_test_domain, make_test_story, target_cell
```

Before `uncertain.py` exists, unittest discovery should report one import error for this module.

- [ ] **Step 2: Add explicit-parameter hooks used by all valid-model tests**

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
```

- [ ] **Step 3: Add the canonical service-health model factory**

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
        likelihood_hook=(ParameterLikelihoodHook() if likelihood_hook is None else likelihood_hook),
    )
```

- [ ] **Step 4: Add record/model identity tests in their own class**

Create `UncertainBeliefRecordModelTests` with these exact behaviors:

```python
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
            distribution.probability_of(TypedValue("HealthState", "healthy"))

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
        changed_hook = make_uncertain_model(likelihood_hook=AlternateLikelihoodHook())
        self.assertEqual(first.content_hash, same.content_hash)
        self.assertNotEqual(first.content_hash, changed_parameters.content_hash)
        self.assertNotEqual(first.content_hash, changed_hook.content_hash)

    def test_parameters_are_detached_and_noncanonical_values_reject(self):
        raw = {
            "prior_by_value": {"healthy": 0.2, "failed": 0.5, "recovered": 0.3},
            "default_likelihood_by_value": {"healthy": 1.0, "failed": 1.0, "recovered": 1.0},
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

Also add direct constructor checks that `BeliefMass`/`BeliefLikelihood` reject non-`TypedValue`, booleans-as-probabilities, negative values, values above 1, NaN, and infinity; `BeliefDistribution` rejects duplicate hypotheses, fewer than two masses, and totals outside `1e-12` of 1.

- [ ] **Step 5: Add narrative behavior/provenance tests in a separate class**

Create `NarrativeUncertainBeliefTests`:

```python
class NarrativeUncertainBeliefTests(unittest.TestCase):
    def test_prior_without_evidence_is_preserved_and_discrete_replay_stays_sparse(self):
        story = make_test_story(receive=False, bob_observes_failure=False)
        domain = make_test_domain()
        state = uncertain.uncertain_epistemic_state(
            story, domain, "bob", make_uncertain_model(), (target_cell(),), at_time=3
        )
        view = state.cells[target_cell()]
        self.assertEqual(view.updates, ())
        self.assertEqual(view.posterior, view.prior)
        self.assertAlmostEqual(
            view.posterior.probability_of(TypedValue("HealthState", "failed")), 0.50
        )
        self.assertNotIn(target_cell(), epistemic_state(story, domain, "bob", at_time=3).cells)

    def test_received_claim_updates_posterior_with_exact_provenance(self):
        story = make_test_story(receive=True, bob_observes_failure=False)
        domain = make_test_domain()
        state = uncertain.uncertain_epistemic_state(
            story, domain, "bob", make_uncertain_model(), (target_cell(),), at_time=3
        )
        step = state.cells[target_cell()].updates[0]
        self.assertEqual(step.evidence.supporting_id, "c1")
        self.assertEqual(step.evidence.source_agent, "alice")
        self.assertEqual(step.evidence.logical_time, 3)
        self.assertEqual(step.evidence.provenance_refs, ("c1", "e2"))
```

For the posterior assertion, compute expected weights with the repository primitive:

```python
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
```

Assert every final `BeliefMass` equals the corresponding `expected` entry.

Add three more methods:

```python
def test_reception_changes_belief_without_changing_objective_state(self):
    received = make_test_story(receive=True, bob_observes_failure=False)
    unreceived = make_test_story(receive=False, bob_observes_failure=False)
    domain = make_test_domain()
    model = make_uncertain_model()
    left = uncertain.uncertain_epistemic_state(received, domain, "bob", model, (target_cell(),), at_time=3)
    right = uncertain.uncertain_epistemic_state(unreceived, domain, "bob", model, (target_cell(),), at_time=3)
    self.assertNotEqual(left.cells[target_cell()].posterior, right.cells[target_cell()].posterior)
    self.assertEqual(objective_state(received, domain, at_time=3), objective_state(unreceived, domain, at_time=3))


def test_conflicting_claims_are_discrete_conflict_but_graded_uncertainty(self):
    story = make_test_story(claim_value="recovered", second_claim_value="failed", bob_observes_failure=False)
    domain = make_test_domain()
    self.assertEqual(epistemic_state(story, domain, "bob", at_time=4).cells[target_cell()].status, "conflicted")
    view = uncertain.uncertain_epistemic_state(story, domain, "bob", make_uncertain_model(), (target_cell(),), at_time=4).cells[target_cell()]
    self.assertEqual(tuple(step.evidence.supporting_id for step in view.updates), ("c1", "c2"))
    self.assertAlmostEqual(sum(mass.probability for mass in view.posterior.masses), 1.0)
    self.assertTrue(all(0.0 < mass.probability < 1.0 for mass in view.posterior.masses))


def test_time_cutoff_excludes_later_claim_update(self):
    story = make_test_story(claim_value="recovered", second_claim_value="failed", bob_observes_failure=False)
    domain = make_test_domain()
    model = make_uncertain_model()
    early = uncertain.uncertain_epistemic_state(story, domain, "bob", model, (target_cell(),), at_time=3)
    late = uncertain.uncertain_epistemic_state(story, domain, "bob", model, (target_cell(),), at_time=4)
    self.assertEqual(tuple(step.evidence.supporting_id for step in early.cells[target_cell()].updates), ("c1",))
    self.assertEqual(tuple(step.evidence.supporting_id for step in late.cells[target_cell()].updates), ("c1", "c2"))
    self.assertNotEqual(early.cells[target_cell()].posterior, late.cells[target_cell()].posterior)
```

- [ ] **Step 6: Add local finite-type fixture and tests**

```python
def make_finite_domain_and_story():
    domain = DomainSpec(
        domain_id="finite-belief-test",
        version="1",
        entity_types=(EntityTypeSpec("Agent"), EntityTypeSpec("Subject"), EntityTypeSpec("Candidate")),
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

Create `FiniteHypothesisTests` with:

```python
def make_finite_model():
    return uncertain.UncertainBeliefModelSpec(
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
```

Assert bool hypotheses are exactly `{False, True}`, entity-ref hypotheses are exactly `{candidate-a, candidate-b}`, and integer/text tracked cells raise `UncertainBeliefResolutionError` before the prior hook is invoked.

- [ ] **Step 7: Add probabilistic-resolution failures in their own class**

Use separate `UncertainBeliefResolutionValidationTests` so Task 2 can make the record/model class green before replay exists.

Add these malformed hooks:

```python
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
        return {_value_hash(value): 0.1 for value in hypotheses}


class ZeroLikelihoodHook:
    def __call__(self, agent_id, evidence, hypotheses, parameters):
        return {_value_hash(value): 0.0 for value in hypotheses}
```

Add explicit tests that missing/extra keys, non-mapping prior, non-normalized prior, negative/>1/NaN/infinite prior or likelihood, and zero posterior mass raise `UncertainBeliefResolutionError`. Add one structural-control test:

```python
bad_story = replace(story, domain_spec_hash="sha256:" + "0" * 64)
with self.assertRaisesRegex(ValueError, "domain identity"):
    uncertain.uncertain_epistemic_state(
        bad_story, domain, "bob", model, (target_cell(),)
    )
```

This structural error must remain the existing `ValueError`, not be relabeled as a probabilistic resolution error.

- [ ] **Step 8: Enlarge the exact narrative public-surface expectation**

Add exactly these strings to `_EXPECTED_PUBLIC_API` in `tests/test_narrative_trust_api.py`:

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

Do not change the root-isolation assertion.

- [ ] **Step 9: Commit RED tests only**

```bash
git add tests/test_narrative_uncertain_belief.py tests/test_narrative_trust_api.py
git diff --cached --name-only
git commit -m "test: define narrative uncertain belief RED"
```

The staged list must contain exactly those two test files. The design/plan docs are pre-existing commits; this RED commit itself contains no production file.

- [ ] **Step 10: Open a draft PR and observe fresh RED**

Use:

```text
base: proof/narrative-dynamics-v0
head: work/narrative-uncertain-belief-v1
draft: true
title: RED: add narrative uncertain belief v1
```

Accept RED only if the fresh proof shows:

```text
Lean conformance: success
Full Lean build: success
Lean theorem suite: success
Python: failure only from
  - test_narrative_uncertain_belief import because narrative_dynamics.narrative.uncertain is missing
  - exact narrative public-surface test because the nine approved exports are missing
No existing-test regression
```

Record run/job/head/tree evidence in the draft PR before production work.

---

### Task 2: Implement immutable records and model identity

**Files:**
- Create: `narrative_dynamics/narrative/uncertain.py`
- Test: `tests/test_narrative_uncertain_belief.py`

**Interfaces:**
- Produces: `BeliefMass`, `BeliefLikelihood`, `BeliefDistribution`, `BeliefUpdateStep`, `UncertainBeliefCellView`, `UncertainBeliefState`, `UncertainBeliefModelSpec`, `UncertainBeliefResolutionError`.

- [ ] **Step 1: Add constants, validation, and parameter freezing**

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
from narrative_dynamics.narrative.ir import EntityRef, GenericNarrative, StateCellRef, TypedValue
from narrative_dynamics.narrative.replay import EpistemicEvidence, epistemic_state

_PROBABILITY_TOLERANCE = 1e-12

class UncertainBeliefResolutionError(ValueError):
    """An explicit uncertain-belief model could not produce a valid state."""
```

Implement `_text`, `_probability`, and recursively detached `_freeze_parameter`. `_probability` rejects booleans, non-numerics, non-finite numbers, and values outside `[0,1]`. `_freeze_parameter` accepts only `None`, bool, int, finite float, str, mappings with non-empty string keys, list, tuple; mappings freeze to `MappingProxyType` and sequences to tuples.

- [ ] **Step 2: Implement `BeliefMass`, `BeliefLikelihood`, and `BeliefDistribution`**

```python
@dataclass(frozen=True)
class BeliefMass:
    value: TypedValue
    probability: float

@dataclass(frozen=True)
class BeliefLikelihood:
    value: TypedValue
    likelihood: float

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

Canonicalize `masses` by stable hash of `value.to_dict()`, reject duplicate/fewer-than-two hypotheses, and require total probability within `1e-12` of 1. Domain value-type compatibility is checked by the replay constructor because a raw `BeliefDistribution` has no `DomainSpec` reference.

- [ ] **Step 3: Implement update/cell/state records with exact chain invariants**

`BeliefUpdateStep` fields are exactly:

```python
@dataclass(frozen=True)
class BeliefUpdateStep:
    evidence: EpistemicEvidence
    prior: BeliefDistribution
    likelihoods: tuple[BeliefLikelihood, ...]
    posterior: BeliefDistribution
```

Require `evidence.cell == prior.cell == posterior.cell` and exact hypothesis-set equality across prior, likelihoods, posterior.

`UncertainBeliefCellView` fields:

```python
@dataclass(frozen=True)
class UncertainBeliefCellView:
    cell: StateCellRef
    prior: BeliefDistribution
    posterior: BeliefDistribution
    updates: tuple[BeliefUpdateStep, ...]
```

Validate chain continuity with the fields that actually exist:

```python
current = self.prior
for step in self.updates:
    if (
        step.prior.cell != self.cell
        or step.posterior.cell != self.cell
        or step.prior != current
    ):
        raise ValueError("uncertain belief update chain is discontinuous")
    current = step.posterior
if current != self.posterior:
    raise ValueError("uncertain belief posterior must equal the update-chain tail")
```

`UncertainBeliefState` fields are `agent_id`, `model_id`, `model_hash`, `logical_time`, `evidence_history`, `cells`; freeze `cells` and require each key equals `view.cell`.

- [ ] **Step 4: Implement `UncertainBeliefModelSpec`**

```python
@dataclass(frozen=True)
class UncertainBeliefModelSpec:
    model_id: str
    version: str
    parameters: Mapping[str, object]
    prior_hook: object = field(compare=False, repr=False)
    likelihood_hook: object = field(compare=False, repr=False)

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

In `__post_init__`, validate IDs, require both hooks callable, detach/freeze parameters, and do not include undocumented hook instance state in identity.

- [ ] **Step 5: Run the isolated record/model class**

```bash
python3 -m unittest -v \
  tests.test_narrative_uncertain_belief.UncertainBeliefRecordModelTests
```

Expected: PASS even though replay/validation classes are not green yet.

- [ ] **Step 6: Commit**

```bash
git add narrative_dynamics/narrative/uncertain.py
git commit -m "feat: add uncertain belief records and model identity"
```

---

### Task 3: Implement finite hypotheses and prior-only replay

**Files:**
- Modify: `narrative_dynamics/narrative/uncertain.py`

**Interfaces:**
- Produces private `_hypotheses_for_cell`, `_validated_vector`, `_distribution_from_vector`; starts public `uncertain_epistemic_state`.

- [ ] **Step 1: Implement finite hypothesis enumeration**

Use this exact behavior:

```python
def _hypotheses_for_cell(story, domain, cell):
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
        hypotheses = tuple(TypedValue(value_type.name, raw) for raw in value_type.allowed_values)
    elif value_type.kind == "bool":
        hypotheses = (TypedValue(value_type.name, False), TypedValue(value_type.name, True))
    elif value_type.kind == "entity_ref":
        hypotheses = tuple(
            TypedValue(value_type.name, EntityRef(entity.id, entity.type_name))
            for entity in sorted(story.entities, key=lambda item: (item.type_name, item.id))
            if entity.type_name == value_type.entity_type
        )
    else:
        raise UncertainBeliefResolutionError("tracked state variable is not finitely enumerable in V1")
    if len(hypotheses) < 2:
        raise UncertainBeliefResolutionError("uncertain belief requires at least two finite hypotheses")
    return tuple(sorted(hypotheses, key=lambda value: stable_content_hash(value.to_dict())))
```

- [ ] **Step 2: Implement exact-key prior validation**

`_validated_vector(raw, hypotheses, label)` requires a mapping with keys exactly equal to `stable_content_hash(value.to_dict())` for the supplied hypotheses and every numeric value finite/in `[0,1]`. For priors, separately require sum within `1e-12` of 1. Never normalize a malformed prior.

- [ ] **Step 3: Implement the public replay signature through the no-evidence path**

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

Start with:

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

For each tracked cell: enumerate hypotheses, call `prior_hook(agent_id, cell, hypotheses, model.parameters)`, validate exact coverage/normalization, construct prior distribution, and initially return `posterior == prior`, `updates == ()`. Preserve the complete `admitted.evidence_history` on the result.

- [ ] **Step 4: Run prior and finite-hypothesis tests**

```bash
python3 -m unittest -v \
  tests.test_narrative_uncertain_belief.NarrativeUncertainBeliefTests.test_prior_without_evidence_is_preserved_and_discrete_replay_stays_sparse \
  tests.test_narrative_uncertain_belief.FiniteHypothesisTests
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add narrative_dynamics/narrative/uncertain.py
git commit -m "feat: add finite uncertain belief priors"
```

---

### Task 4: Add sequential Bayesian evidence updates

**Files:**
- Modify: `narrative_dynamics/narrative/uncertain.py`

**Interfaces:**
- Consumes: `admitted.evidence_history`, Task 3 helpers, `posterior_distribution`.
- Produces: complete provenance-linked posterior updates.

- [ ] **Step 1: Apply evidence only to matching tracked cells**

Inside each cell replay:

```python
current = prior
updates = []
for evidence in admitted.evidence_history:
    if evidence.cell != cell:
        continue
    hypotheses = tuple(mass.value for mass in current.masses)
    raw_likelihoods = model.likelihood_hook(
        agent_id, evidence, hypotheses, model.parameters
    )
    likelihood_vector = _validated_vector(
        raw_likelihoods, hypotheses, label="uncertain likelihood"
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
        BeliefLikelihood(
            value,
            likelihood_vector[stable_content_hash(value.to_dict())],
        )
        for value in hypotheses
    )
    updates.append(BeliefUpdateStep(evidence, current, likelihoods, posterior))
    current = posterior
```

Do not add a second sort. Existing replay owns canonical evidence order.

- [ ] **Step 2: Return the completed cell view**

```python
UncertainBeliefCellView(
    cell=cell,
    prior=prior,
    posterior=current,
    updates=tuple(updates),
)
```

The function never examines objective state and never translates high probability into discrete `resolved`.

- [ ] **Step 3: Run behavior/provenance tests**

```bash
python3 -m unittest -v \
  tests.test_narrative_uncertain_belief.NarrativeUncertainBeliefTests
```

Expected: prior-only, one-claim posterior, exact provenance, reception/objective isolation, discrete conflict + graded posterior, and time cutoff all pass.

- [ ] **Step 4: Commit**

```bash
git add narrative_dynamics/narrative/uncertain.py
git commit -m "feat: update uncertain beliefs from narrative evidence"
```

---

### Task 5: Close all probabilistic fail-closed boundaries

**Files:**
- Modify: `narrative_dynamics/narrative/uncertain.py`

**Interfaces:**
- Produces approved `UncertainBeliefResolutionError` behavior without relabeling structural narrative validation errors.

- [ ] **Step 1: Add narrow hook-call wrappers**

```python
def _call_prior_hook(model, agent_id, cell, hypotheses):
    try:
        return model.prior_hook(agent_id, cell, hypotheses, model.parameters)
    except (TypeError, ValueError) as error:
        raise UncertainBeliefResolutionError(
            "uncertain prior hook could not produce a valid vector"
        ) from error


def _call_likelihood_hook(model, agent_id, evidence, hypotheses):
    try:
        return model.likelihood_hook(
            agent_id, evidence, hypotheses, model.parameters
        )
    except (TypeError, ValueError) as error:
        raise UncertainBeliefResolutionError(
            "uncertain likelihood hook could not produce a valid vector"
        ) from error
```

Do not catch `BaseException`, `KeyboardInterrupt`, or `SystemExit`. Keep `validate_narrative()` outside these wrappers.

- [ ] **Step 2: Make vector validation raise the typed error**

Missing/extra keys, non-mapping values, non-normalized prior, negative/>1/NaN/infinite weights, unsupported hypothesis spaces, and zero posterior mass all raise `UncertainBeliefResolutionError`.

- [ ] **Step 3: Run only the resolution-validation class**

```bash
python3 -m unittest -v \
  tests.test_narrative_uncertain_belief.UncertainBeliefResolutionValidationTests
```

Expected: every probabilistic failure uses the typed error; wrong domain identity remains the pre-existing structural `ValueError`.

- [ ] **Step 4: Run the complete new module**

```bash
python3 -m unittest -v tests.test_narrative_uncertain_belief
```

Expected: PASS.

- [ ] **Step 5: Commit production only**

```bash
git add narrative_dynamics/narrative/uncertain.py
git commit -m "fix: enforce uncertain belief resolution boundaries"
```

If a failing assertion contradicts an existing documented engine contract, diagnose first and put any test-contract correction in a separate test-only commit; do not distort production to satisfy an invalid test.

---

### Task 6: Publish the narrative-only API

**Files:**
- Modify: `narrative_dynamics/narrative/__init__.py`
- Test: `tests/test_narrative_trust_api.py`

**Interfaces:**
- Produces the exact enlarged narrative package surface; root package remains unchanged.

- [ ] **Step 1: Import the nine approved symbols**

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

- [ ] **Step 2: Add exactly those names to `narrative.__all__`**

Do not edit `narrative_dynamics/__init__.py`.

- [ ] **Step 3: Verify package isolation**

```bash
python3 -m unittest -v \
  tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation \
  tests.test_narrative_uncertain_belief
```

Expected: PASS; all nine symbols exist under `narrative_dynamics.narrative` and none exists at the top-level package.

- [ ] **Step 4: Commit**

```bash
git add narrative_dynamics/narrative/__init__.py
git commit -m "feat: export narrative uncertain belief api"
```

---

### Task 7: Run compatibility regression and lock the final diff

**Files:**
- No intended changes.

- [ ] **Step 1: Run the complete Python suite**

```bash
python3 -m unittest discover -s tests -v
```

Expected: all tests pass, including existing replay/decision/analysis/trust and Knives Out, The Matrix, Memento.

- [ ] **Step 2: Run the same Lean gates used by proof**

```bash
export PATH="$HOME/.elan/bin:$PATH"
lake build NarrativeDynamics.Core.Belief NarrativeDynamics.Core.Drive NarrativeDynamics.Core.Learning
lake build
lake env lean NarrativeDynamics/Tests/StoryState.lean
lake env lean NarrativeDynamics/Tests/Testimony.lean
```

Expected: all succeed. Do not modify Lean for this V1.

- [ ] **Step 3: Lock the final path boundary**

```bash
git diff --name-status proof/narrative-dynamics-v0...HEAD
```

Allowed paths:

```text
docs/superpowers/specs/2026-08-25-narrative-uncertain-belief-v1-design.md
docs/superpowers/plans/2026-08-25-narrative-uncertain-belief-v1.md
narrative_dynamics/narrative/uncertain.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_uncertain_belief.py
tests/test_narrative_trust_api.py
```

Any other path requires an explicit explanation before continuing.

---

### Task 8: Obtain final fresh proof and exact-tree lock

**Files:**
- PR metadata only.

- [ ] **Step 1: Confirm final topology before trusting CI**

Record fresh base SHA, feature SHA, master SHA, and feature tree. Base must still be `proof/narrative-dynamics-v0`; master must be unchanged.

- [ ] **Step 2: Observe a fresh proof on the exact final head**

Require success for:

```text
Resolve Lean dependencies
Lean Python conformance vectors
Build Lean library
Lean theorem tests
Python numerical tests
Narrative story theorem tests
Narrative testimony theorem tests
```

Read full job logs and explicitly confirm all Python tests are `OK`, new uncertain tests are individually `ok`, existing Knives Out/The Matrix/Memento tests are `ok`, and StoryState/Testimony are successful. Do not call GREEN from combined status alone.

- [ ] **Step 3: Prove exact-tree**

Read the PR merge-checkout SHA from the successful checkout log, fetch its tree and feature-head tree, and require:

```text
feature HEAD tree == tested PR merge-checkout tree
```

Matching tree is required even though PR merge commit SHA differs from feature SHA.

- [ ] **Step 4: Reconfirm pure ancestry and master isolation**

Compare research base to feature and require `status=ahead`, `ahead_by>0`, `behind_by=0`. Re-fetch master and verify unchanged.

- [ ] **Step 5: Update the draft PR evidence**

Set title:

```text
feat: add narrative uncertain belief v1
```

PR body records RED commit/run/job/failure shape, implementation commits, any test-contract correction commits, final GREEN run/job, Python total/OK, Lean gates, StoryState/Testimony, feature SHA/tree, PR checkout SHA/tree, exact-tree statement, final diff boundary, research SHA, and master SHA.

- [ ] **Step 6: Stop at integration-ready**

Terminal state:

```text
Narrative Uncertain Belief V1 = integration-ready
proof/narrative-dynamics-v0 = unchanged by this plan
master = unchanged
```

A later explicit user authorization is required for a separately verified non-force research-ref fast-forward.

---

## Self-Review Results

- **Spec coverage:** separate uncertain projection, finite hypotheses, explicit normalized priors, conditional likelihoods, evidence provenance/time, reception boundary, conflicting evidence, model identity, typed fail-closed errors, exact package isolation, and full proof are all assigned to explicit tasks.
- **No scope leak:** persuasion, source reliability, motivation, institutional feasibility, memory, counterfactual integration, decision integration, GenericNarrative schema changes, and Lean are excluded.
- **Type consistency:** `BeliefUpdateStep` has only `evidence/prior/likelihoods/posterior`; chain validation uses `step.prior.cell` and `step.posterior.cell`, not a nonexistent `step.cell` field.
- **Task independence:** record/model tests are `UncertainBeliefRecordModelTests`; probabilistic replay failures are `UncertainBeliefResolutionValidationTests`, so Task 2 can achieve a meaningful local GREEN before Tasks 3–5 exist.
- **Identity consistency:** hook attestation and recursively frozen explicit `parameters` both enter model hash; supported behavior does not depend on hidden hook state.
- **TDD discipline:** Task 1 is a tests-only RED commit observed in CI before production; later tests remain frozen except evidence-backed test-contract corrections in isolated commits.
