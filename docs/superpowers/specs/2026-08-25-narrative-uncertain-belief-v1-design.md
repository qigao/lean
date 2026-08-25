# Narrative Uncertain Belief V1 Design

## Status

Proposed architectural increment for the Generic Narrative Engine. This document defines the first probabilistic epistemic layer that later persuasion, source reliability, expectation, and institutional-behavior work will build on.

Implementation is intentionally not part of this design commit. The first implementation step after approval is test-only RED.

## Problem

The current Generic Narrative Engine separates objective state, direct perception, received testimony, epistemic state, and decisions. Its epistemic cell semantics are deliberately discrete: a cell is `unknown`, `resolved`, or `conflicted`, with provenance for the evidence that produced that state.

That representation is correct for questions such as “does this agent have supported access to this fact?”, but it cannot represent graded belief such as:

- B considers proposition X 30% plausible before hearing A;
- after receiving A's claim, B considers X 70% plausible;
- another source later supplies conflicting evidence and B moves back to 45%;
- B's final behavior changes only after the posterior crosses a downstream decision threshold.

Persuasion must not be modeled as an event that directly writes “B is persuaded”. It must eventually be analyzable as an evidence-driven change in an agent's uncertain attitude. Therefore uncertain belief is a prerequisite for persuasion rather than a sub-feature of persuasion.

## Design goals

Narrative Uncertain Belief V1 must:

1. derive a finite probability distribution over possible values of selected narrative state cells for one agent;
2. consume the Generic Narrative Engine's existing `EpistemicEvidence` stream, preserving observation/testimony provenance and time order;
3. use explicit, content-hashed model assumptions for priors and evidence likelihoods rather than silently inventing confidence values;
4. reuse the repository's existing finite Bayesian normalization semantics instead of creating a second probability algorithm;
5. leave existing `objective_state`, `direct_state`, `epistemic_state`, `EpistemicCellView`, decision models, movie conformance fixtures, and GenericNarrative schema behavior unchanged;
6. expose deterministic result/model identity that changes when either probability-model code or explicit probability parameters change;
7. fail closed on unsupported hypothesis spaces, invalid probability inputs, impossible evidence updates, or undeclared cells;
8. retain enough stepwise update provenance that a later persuasion layer can compare prior and posterior belief around a specific communication event.

## Non-goals

V1 does **not** add:

- a `persuade` event that directly mutates belief;
- source trust/reliability state;
- learned reliability from past accuracy;
- preferences, goals, utilities, or motivational attitude updates;
- institutional rules, permissions, prohibitions, or action feasibility;
- first-class memory decay/reset/retrieval;
- probabilistic values in `TypedValue`;
- floats in the Generic Narrative IR;
- a new GenericNarrative schema version;
- changes to existing discrete epistemic resolution;
- automatic probability extraction from natural language;
- empirical or unique causal claims about why a human actor changed their mind;
- Lean formalization of this new narrative layer in V1.

These are later increments built on the V1 boundary once its semantics are stable.

## Existing foundations and compatibility constraints

### Existing narrative representation

The Generic Narrative Engine represents domain state with `StateCellRef` and `TypedValue`. `epistemic_state()` builds an ordered `EpistemicEvidence` history from direct observations and received claims, then reduces each cell to an `EpistemicCellView` with status `unknown`, `resolved`, or `conflicted`.

V1 does not replace or reinterpret this path. It consumes the same evidence history through a separate projection.

### Existing probability primitives

The repository already contains:

- `hypothesis_competition.posterior_distribution(priors, likelihoods)` for normalized finite Bayesian hypothesis weights;
- `epistemic_belief.EpistemicBeliefState` and `bayes_update_belief()` for a single symbolic hypothesis plus confidence;
- `grounded_hypothesis_space.GroundedHypothesisSpace` and `grounded_belief_update.posterior_belief_states()` for probability updates tied to `TypedNode`, `BeliefKB`, `WorldState`, and proof-graph provenance.

The grounded subsystem cannot be imported wholesale into narrative replay because its coordinate system is structurally different from narrative `StateCellRef`/`TypedValue` evidence. V1 therefore reuses the finite posterior-normalization function and the repository's Bayesian semantics while introducing narrative-native records. It does not duplicate `GroundedHypothesisSpace` or pretend the two provenance systems are identical.

### Existing implementation attestation

`measure_implementation()` measures backing module bytes and callable/class identity. It does not bind arbitrary instance or closure configuration. Therefore V1 must not treat hook implementation hashes alone as a complete probability-model identity.

All behavior-affecting prior/likelihood configuration is required to live in an explicit canonical `parameters` payload on `UncertainBeliefModelSpec`. That payload is recursively frozen and included directly in the model content hash. Hooks receive that explicit payload as an argument. Hidden mutable hook state is outside the supported V1 contract.

## Alternatives considered

### A. Add probability/confidence directly to `EpistemicCellView`

Rejected for V1. This would silently change the meaning and serialization of the current discrete epistemic contract, force every existing replay/decision/analysis consumer to interpret confidence, and risk breaking the exact public API and movie conformance cases.

### B. Reuse `GroundedHypothesisSpace` directly inside narrative replay

Rejected. That API requires `TypedNode`, `BeliefKB`, `WorldState`, `InterpretationCandidate`, and proof IDs. Adapting every narrative cell and claim into those types would make the narrative engine depend on a second world representation and create artificial provenance records purely to satisfy an unrelated interface.

### C. Add a parallel narrative-native uncertain projection

Chosen. Existing discrete replay remains canonical for “supported/resolved/conflicted” semantics. A new `uncertain_epistemic_state()` consumes the same ordered evidence stream and produces explicit finite posterior distributions using a content-hashed probability model.

This isolates probability assumptions while preserving backward compatibility.

## Architecture

### New module

Add:

`narrative_dynamics/narrative/uncertain.py`

The module owns only uncertain-belief records, validation, finite hypothesis enumeration, sequential Bayesian updates, and model identity.

It does not own narrative event replay, testimony admission, decision selection, persuasion analysis, or institutional rules.

### Public records

#### `BeliefMass`

Immutable pair:

- `value: TypedValue`
- `probability: float`

Validation:

- probability must be finite;
- probability must be in `[0, 1]`;
- `value` must be a `TypedValue`.

Serialization is deterministic through `to_dict()`.

#### `BeliefLikelihood`

Immutable conditional-likelihood pair:

- `value: TypedValue`
- `likelihood: float`

Validation:

- likelihood must be finite;
- likelihood must be in `[0, 1]`;
- `value` must be a `TypedValue`.

The likelihood vector is not required to sum to 1 because it represents `P(evidence | hypothesis)` independently for each hypothesis.

#### `BeliefDistribution`

Immutable normalized finite distribution:

- `cell: StateCellRef`
- `masses: tuple[BeliefMass, ...]`

Requirements:

- at least two distinct hypothesis values;
- hypothesis values have unique stable content hashes;
- masses are sorted by stable hash for canonical serialization;
- probabilities sum to 1 within a strict numerical tolerance;
- all values match the domain state variable's declared value type.

Convenience lookup may return probability by exact `TypedValue`, but canonical identity uses stable hashes, not Python mapping iteration order.

#### `BeliefUpdateStep`

Immutable provenance record for one admitted narrative evidence item:

- `evidence: EpistemicEvidence`
- `prior: BeliefDistribution`
- `likelihoods: tuple[BeliefLikelihood, ...]`
- `posterior: BeliefDistribution`

The step proves which evidence changed which distribution and preserves the original evidence source, logical time, support ID, and provenance refs.

#### `UncertainBeliefCellView`

Immutable result for one tracked cell:

- `cell: StateCellRef`
- `prior: BeliefDistribution`
- `posterior: BeliefDistribution`
- `updates: tuple[BeliefUpdateStep, ...]`

If the agent has no admitted evidence for the cell before the cutoff, `posterior == prior` and `updates == ()`. This is intentional: uncertainty can exist before evidence, unlike the discrete `epistemic_state()` API, which may simply omit an unknown cell.

#### `UncertainBeliefState`

Immutable result for one agent and cutoff:

- `agent_id: str`
- `model_id: str`
- `model_hash: str`
- `logical_time: int | None`
- `evidence_history: tuple[EpistemicEvidence, ...]`
- `cells: Mapping[StateCellRef, UncertainBeliefCellView]`

Only explicitly tracked cells are returned. The API does not invent beliefs for every state variable in the domain.

### Probability model

#### `UncertainBeliefModelSpec`

A configured, content-hashed model declaration:

- `model_id: str`
- `version: str`
- `parameters: Mapping[str, object]`
- `prior_hook: callable`
- `likelihood_hook: callable`

`parameters` must be recursively canonical JSON-like data: `None`, booleans, integers, finite floats, strings, mappings with non-empty string keys, lists, or tuples. It is detached/frozen at construction.

`to_dict()` includes:

- model ID/version;
- canonical parameters;
- measured prior-hook implementation identity;
- measured likelihood-hook implementation identity.

Therefore changing code **or** changing explicit prior/likelihood parameters changes `content_hash`.

The engine, not the hooks, owns hypothesis enumeration, validation, update ordering, Bayesian normalization, and provenance.

##### Prior hook contract

Conceptual signature:

```python
prior_hook(agent_id, cell, hypotheses, parameters) -> Mapping[str, float]
```

where mapping keys are the stable content hashes of the exact hypotheses supplied by the engine.

Requirements:

- exact hypothesis-key coverage;
- finite probabilities in `[0, 1]`;
- probabilities sum to 1 within the declared numerical tolerance;
- no automatic renormalization of malformed priors;
- no access to objective state, later evidence, or decision outcomes is passed to the hook.

The hook therefore cannot inspect canonical truth through the official interface, and the engine does not silently “fix” an invalid prior into a different model assumption.

##### Likelihood hook contract

Conceptual signature:

```python
likelihood_hook(agent_id, evidence, hypotheses, parameters) -> Mapping[str, float]
```

Requirements:

- exact hypothesis-key coverage;
- finite conditional likelihoods in `[0, 1]`;
- positive total posterior mass when combined with the current prior;
- no evidence outside the supplied `EpistemicEvidence` item is passed to the hook.

V1 deliberately does not hardcode “direct evidence = 1.0” or “testimony = 0.7”. Those are model assumptions. A later source-reliability layer can implement a likelihood hook from explicit trust/reliability state without changing uncertain replay.

### Finite hypothesis enumeration

The engine derives candidate values from the tracked cell's `StateVariableSpec` and `ValueTypeSpec`.

V1 supports only finite types:

- `enum`: every declared enum value as `TypedValue(value_type.name, enum_value)`;
- `bool`: `False` and `True`;
- `entity_ref`: every declared entity of the referenced entity type, sorted canonically.

V1 rejects:

- `integer` state variables;
- `text` state variables;
- any finite domain that yields fewer than two hypotheses.

The rejection is explicit because silently choosing an arbitrary integer/text candidate set would smuggle modeling assumptions into the engine.

### Main replay API

Add:

```python
uncertain_epistemic_state(
    story,
    domain,
    agent_id,
    model,
    tracked_cells,
    *,
    at_time=None,
) -> UncertainBeliefState
```

`tracked_cells` is a non-empty unique tuple of `StateCellRef` values.

Processing:

1. validate narrative/domain identity using the existing validator;
2. validate agent and tracked cells;
3. call existing `epistemic_state(story, domain, agent_id, at_time=...)`;
4. reuse its ordered `evidence_history` as the only admitted evidence stream;
5. enumerate finite hypotheses for each tracked cell;
6. ask the model for the prior and validate it as an already normalized distribution;
7. filter admitted evidence to that exact cell;
8. process evidence in the existing canonical evidence order;
9. for each evidence item, request conditional likelihoods from the model;
10. validate likelihood coverage/range and call existing `posterior_distribution()` with the current distribution as priors and the likelihood vector;
11. record a `BeliefUpdateStep` with prior, likelihoods, posterior, and exact evidence provenance;
12. return the final `UncertainBeliefState`.

The function never reads `objective_state()` and never gives the probability hooks access to objective state through their official arguments.

## Relationship to current discrete epistemic semantics

The two projections answer different questions and coexist:

- `epistemic_state()` answers whether evidence resolves, leaves unknown, or conflicts on a state cell under existing discrete semantics;
- `uncertain_epistemic_state()` answers how an explicitly configured probabilistic model redistributes mass over the finite values of selected state cells given the same admitted evidence.

A conflicted discrete cell may still have a non-uniform probabilistic posterior. This is not a contradiction: one layer says “multiple incompatible equalities are present”; the other says “under this probability model, the competing values have these weights.”

No V1 function converts a posterior above an arbitrary threshold into `resolved`. No threshold is hidden in the uncertain layer.

## Error model

Add `UncertainBeliefResolutionError(ValueError)` for failures that occur after narrative/domain structural validation but before a valid uncertain state can be produced.

Examples:

- tracked cell is undeclared or subject type mismatches the state variable;
- value type is not finitely enumerable in V1;
- prior hook omits or invents hypothesis keys;
- prior is not normalized;
- prior or likelihood contains negative, out-of-range, NaN, or infinite values;
- a Bayesian update has zero total posterior mass;
- hook result is not a mapping;
- model parameters are not canonical/finitely hashable;
- model or hook type is invalid.

Existing narrative/domain validation errors remain their existing types. V1 does not relabel structural errors as probabilistic errors.

## Public API boundary

Export from `narrative_dynamics.narrative` only:

- `BeliefMass`
- `BeliefLikelihood`
- `BeliefDistribution`
- `BeliefUpdateStep`
- `UncertainBeliefCellView`
- `UncertainBeliefState`
- `UncertainBeliefModelSpec`
- `UncertainBeliefResolutionError`
- `uncertain_epistemic_state`

Update the exact narrative public-surface test accordingly.

Do **not** export these symbols from the top-level `narrative_dynamics` package. Existing root isolation remains a hard boundary.

## Data-flow example

For a finite cell `policy.outcome ∈ {passes, fails}`:

```text
explicit configured model
P(passes)=0.40, P(fails)=0.60
        |
        v
received claim from A
EpistemicEvidence(claim-A)
        |
likelihood hook for claim-A
L(E|passes)=0.80
L(E|fails)=0.20
        |
posterior_distribution
        v
P(passes)=0.727...
P(fails)=0.272...
        |
conflicting evidence from C
        v
next Bayesian update
        |
final posterior + complete step trace
```

The engine may later compare a counterfactual where the reception of `claim-A` is removed. In V1, callers can recompute the uncertain state on the modified narrative; direct integration with `evaluate_counterfactual()` is deferred.

## Testing strategy

Implementation must follow test-only RED before production code.

### RED file

Add `tests/test_narrative_uncertain_belief.py` first, with production module still absent.

The initial RED must fail only because `narrative_dynamics.narrative.uncertain` and its declared public API do not yet exist; existing Lean and Python gates must remain green apart from the new tests.

### Required behavior tests

1. **Prior without evidence**
   - tracked enum cell has an explicit non-uniform prior;
   - agent receives no evidence for the cell;
   - posterior equals prior;
   - update trace is empty;
   - discrete `epistemic_state()` remains unchanged and may omit the cell.

2. **Single received claim updates belief**
   - prior is explicit;
   - one received testimony item has model-supplied likelihoods;
   - posterior equals the repository's existing `posterior_distribution()` result;
   - update trace names the exact claim, source agent, time, and provenance refs.

3. **Reception boundary**
   - the same claim not received by B does not enter B's uncertain update;
   - adding/removing a reception changes B's posterior but does not change objective state.

4. **Conflicting evidence is graded, not collapsed**
   - two incompatible claims are both admitted;
   - discrete epistemic replay reports `conflicted`;
   - uncertain replay processes both in canonical order and returns a valid non-degenerate posterior according to configured likelihoods.

5. **Time cutoff**
   - evidence after `at_time` cannot affect the posterior;
   - the same story at a later cutoff includes the evidence and changes the posterior.

6. **Finite hypothesis enumeration**
   - enum, bool, and entity-ref cells enumerate deterministically;
   - text and integer cells fail closed with `UncertainBeliefResolutionError`.

7. **Probability validation**
   - missing/extra hypothesis keys reject;
   - prior that does not sum to 1 rejects rather than being auto-normalized;
   - negative/out-of-range/non-finite prior and likelihood values reject;
   - zero posterior mass rejects;
   - model hooks cannot return malformed types.

8. **Model identity**
   - same model code + same canonical parameters produce a stable content hash;
   - changing either hook implementation changes model identity;
   - changing explicit parameters changes model identity even when hook code is unchanged;
   - result records model ID/hash.

9. **Compatibility and isolation**
   - existing `epistemic_state()` serialization/behavior is unchanged;
   - existing film conformance tests remain unchanged;
   - the narrative package exports exactly the enlarged declared surface;
   - top-level package still exports none of the narrative symbols.

### Full proof gate

GREEN requires a fresh PR proof with:

- Lean conformance build success;
- full Lean build success;
- Lean theorem suite success;
- all Python tests success;
- StoryState theorem gate success;
- Testimony theorem gate success;
- exact-tree proof that the feature tree equals the tested PR checkout tree.

No research-ref integration occurs before that evidence exists.

## Implementation boundary

Expected production changes for V1 are limited to:

- new `narrative_dynamics/narrative/uncertain.py`;
- `narrative_dynamics/narrative/__init__.py` public surface additions;
- exact public-surface test adjustment;
- new uncertain-belief tests.

A small import of `hypothesis_competition.posterior_distribution` is expected. A private canonical-freezing helper may be implemented locally or extracted only if tests prove reuse is necessary; no unrelated contracts refactor is part of V1.

No changes are expected to:

- `narrative_dynamics/narrative/ir.py`;
- `narrative_dynamics/narrative/domain.py`;
- `narrative_dynamics/narrative/replay.py` unless a test proves a missing public evidence accessor that cannot be satisfied without it;
- existing decision machinery;
- existing analysis/intervention machinery;
- film domain fixtures;
- Lean source.

If implementation requires changing GenericNarrative schema, existing discrete epistemic semantics, or decision contracts, the work must stop and be reclassified as a larger follow-up design rather than expanding V1 silently.

## Follow-on architecture enabled by V1

V1 is intentionally the foundation for separate later increments:

1. **Source reliability and conflicting evidence**
   - derive likelihoods from explicit agent-to-source trust/reliability state;
   - preserve source-specific evidence provenance.

2. **Persuasion analysis**
   - compare belief immediately before and after a communication;
   - report prior/posterior change and downstream decision difference;
   - persuasion is an analysis result, not a magic state mutation.

3. **Institutional constraints and action feasibility**
   - represent rule/environment changes objectively;
   - filter feasible actions independently of beliefs/preferences;
   - distinguish `belief change -> behavior change` from `constraint change -> behavior change`.

4. **Belief × motivation × constraints decision models**
   - consume probabilistic beliefs together with explicit preferences/goals and feasible action sets.

This ordering preserves the key scientific distinction: an agent can change belief without changing behavior, change behavior without changing belief, or change both through separate mechanisms.

## Acceptance summary

Narrative Uncertain Belief V1 is complete only when the Generic Narrative Engine can say, under an explicit and content-hashed probabilistic model:

> “Given exactly the evidence available to agent B by time t, B's probability mass over this finite narrative state cell moved from this prior distribution to this posterior distribution through these provenance-linked update steps.”

It must be able to say this without changing canonical objective truth, without changing existing discrete epistemic semantics, and without declaring that any communication has “persuaded” B merely because B received it.
