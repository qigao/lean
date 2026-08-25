# Narrative Intentional Decision V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a narrative-native, deterministic `UncertainBeliefState -> GoalState -> ActionPolicy` model that preserves evidence provenance, separates goal and action stochasticity, and remains independently comparable with deterministic and POMDP models.

**Architecture:** Add one new scoped module, `narrative_dynamics/narrative/intention.py`, above the existing uncertain-belief layer and beside the existing deterministic decision layer. The new module consumes only actor-admitted uncertain belief, computes posterior-weighted goal scores and a softmax goal policy, computes goal-conditioned action softmax policies, marginalizes them into the observable action policy, and returns a fully layered immutable result. Existing `decision.py`, `uncertain.py`, IR, replay, movie fixtures, Lean sources, root exports, runtime registry, prison models, and observational protocol stay unchanged.

**Tech Stack:** Python 3 stdlib dataclasses/mappings/math, existing `narrative_dynamics.narrative.uncertain`, existing `narrative_dynamics.narrative.decision.DecisionResolutionError`, existing `grounded_goal_softmax.finite_softmax`, existing `stable_content_hash`, existing implementation attestation, `unittest`, GitHub Actions Lean/Python proof workflow.

**Spec:** `docs/superpowers/specs/2026-08-25-narrative-intentional-decision-v1-design.md`

## Global Constraints

- Base research SHA is exactly `56d3910918db9f3544636726d4f42a33769029d8`.
- Work branch is `work/narrative-intentional-decision-v1`; `master` must not move.
- Production diff is limited to new `narrative_dynamics/narrative/intention.py` plus scoped export edits in `narrative_dynamics/narrative/__init__.py`.
- Test diff is limited to new `tests/test_narrative_intention.py` plus exact-surface edits in `tests/test_narrative_trust_api.py`.
- Do not modify `decision.py`, `uncertain.py`, `ir.py`, `replay.py`, movie domain modules, Lean source, root `narrative_dynamics/__init__.py`, runtime registry, prison adapters, or observational protocol code.
- Core execution is deterministic: no RNG, seed, stochastic latent-goal sampling, or sampled action in V1.
- `run_intentional_decision()` must track exactly `decision.context_cells` and use `at_time=decision.logical_time`.
- No objective-state read or objective fallback is allowed in the intentional layer.
- Reuse `grounded_goal_softmax.finite_softmax()` for both goal and conditional-action normalization.
- Goal and action inverse temperatures remain separate positive finite fields: `beta_goal` and `beta_action`.
- Observable action policy is the mixture `sum_g P(a|g) P(g|B)`, never `softmax(sum_g P(g|B)V_g(a))`.
- MAP projections use lexicographically smallest ID on exact ties and must not depend on mapping/tuple insertion order.
- Malformed coverage and non-finite model inputs fail closed; missing utilities never default to zero.
- Public additions are exactly ten symbols under `narrative_dynamics.narrative`; none are exported at package root.
- Implementation follows test-only RED -> exact-head CI RED -> minimal GREEN -> full regression -> exact-tree GREEN.

---

## File Structure

- `docs/superpowers/specs/2026-08-25-narrative-intentional-decision-v1-design.md` — approved scientific and API contract; already committed.
- `docs/superpowers/plans/2026-08-25-narrative-intentional-decision-v1.md` — this execution plan.
- `narrative_dynamics/narrative/intention.py` — all intentional model records, validation, scoring, marginalization, deterministic MAP projection, and execution API.
- `narrative_dynamics/narrative/__init__.py` — append only the ten approved scoped exports.
- `tests/test_narrative_intention.py` — all intentional semantic, identity, numeric, capability, tie-break, and exact-coverage tests; keep helpers local to preserve the six-file boundary.
- `tests/test_narrative_trust_api.py` — append only the ten intentional names to the exact narrative surface set and retain root isolation checks.

---

### Task 1: Lock the complete test-only RED surface

**Files:**
- Create: `tests/test_narrative_intention.py`
- Modify: `tests/test_narrative_trust_api.py`
- Do not create: `narrative_dynamics/narrative/intention.py`

**Interfaces:**
- Consumes: existing `make_test_domain()`, `make_test_story()`, `target_cell()` from `tests.narrative_test_support`; existing `UncertainBeliefModelSpec`; existing `stable_content_hash`.
- Produces: executable RED contract for `GoalSpec`, `GoalModelSpec`, `GoalState`, `ChoiceModelSpec`, `IntentionalDecisionModelSpec`, `IntentionalDecisionResult`, `IntentionalDecisionResolutionError`, `GoalResolutionError`, `ChoiceResolutionError`, and `run_intentional_decision`.

- [ ] **Step 1: Add a guarded import boundary and deterministic test helpers**

Create `tests/test_narrative_intention.py` with a guarded import so the entire pre-existing suite can run and every new test reports the missing intentional module/API rather than crashing discovery:

```python
_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.intention import (
        ChoiceModelSpec,
        ChoiceResolutionError,
        GoalModelSpec,
        GoalResolutionError,
        GoalSpec,
        GoalState,
        IntentionalDecisionModelSpec,
        IntentionalDecisionResolutionError,
        IntentionalDecisionResult,
        run_intentional_decision,
    )
except ImportError as error:
    _IMPORT_ERROR = error
```

Add local helpers, not shared-test-support edits:

```python
def value_hash(value):
    return stable_content_hash(value.to_dict())


def prior_hook(agent_id, cell, hypotheses, parameters):
    configured = parameters["prior"]
    return {value_hash(h): float(configured[h.value]) for h in hypotheses}


def likelihood_hook(agent_id, evidence, hypotheses, parameters):
    strength = float(parameters["strength"])
    claimed = evidence.proposition.value if evidence.proposition is not None else None
    return {
        value_hash(h): (
            strength if claimed is not None and h == claimed else (1.0 - strength)
        )
        for h in hypotheses
    }
```

Use a three-value `HealthState` prior with positive mass everywhere, for example `failed=.50`, `recovered=.25`, `healthy=.25`, so testimony can move mass without zero-support artifacts.

- [ ] **Step 2: Add semantic-chain tests with explicit assertions**

Create test methods with these exact responsibilities:

```python
def test_admitted_evidence_changes_belief_goal_and_action():
    # matched receive=False / receive=True stories
    # same belief/goal/choice model
    # assert posterior differs
    # assert goal policy differs in expected direction
    # assert final action policy differs in expected direction


def test_unobserved_objective_change_does_not_leak_into_intention():
    # objective story differs while bob receives/observes no changed fact
    # assert belief_state.to_dict(), goal_state.to_dict(), conditional policies,
    # action_scores, action_policy, selected_action all equal


def test_same_belief_different_goal_model_changes_only_downstream_layers():
    # assert exact same belief_state
    # assert changed goal_state and action_policy


def test_same_belief_and_goal_different_choice_model_preserves_goal_state():
    # assert exact same belief_state and goal_state
    # assert changed conditional_action_policies and action_policy
```

- [ ] **Step 3: Add model-separation and mathematical-identity tests**

Add:

```python
def test_beta_goal_and_beta_action_control_distinct_layers():
    # higher beta_goal concentrates goal policy with identical conditional policies
    # higher beta_action leaves goal_state exactly equal and concentrates conditionals


def test_hypothesis_independent_instrumentality_blocks_belief_effect():
    # admitted evidence changes posterior
    # constant I_g,c(h) across all h keeps goal scores/policy and action policy equal


def test_action_policy_is_latent_goal_mixture_not_softmax_of_action_scores():
    # construct asymmetric goal/action table where the two formulas differ
    # compute expected mixture directly with math.fsum
    # assert result.action_policy == mixture within 1e-12
    # compute finite_softmax(result.action_scores, beta=beta_action)
    # assert at least one action differs materially
```

- [ ] **Step 4: Add deterministic tie, exact-coverage, and numeric fail-closed tests**

Add tests that explicitly cover:

```python
def test_exact_ties_use_lexical_ids_and_ignore_input_order(): ...
def test_goal_context_coverage_fails_closed(): ...
def test_goal_hypothesis_coverage_fails_closed(): ...
def test_choice_goal_coverage_fails_closed(): ...
def test_choice_action_coverage_fails_closed(): ...
def test_invalid_goal_numeric_values_fail_closed(): ...
def test_invalid_choice_numeric_values_fail_closed(): ...
```

Use `subTest` cases for missing and extra keys. Require `GoalResolutionError` for context/hypothesis failures and `ChoiceResolutionError` for goal/action-table failures. Constructor-time malformed finite/positive fields may raise `TypeError`/`ValueError` according to the spec; runtime coverage mismatches must use the typed resolution subclasses.

- [ ] **Step 5: Add identity-lineage and result-typing tests**

Add:

```python
def test_model_identity_binds_belief_goal_choice_configuration():
    # changed belief parameters -> composite hash differs
    # changed beta_goal / instrumentality -> goal and composite hashes differ
    # changed beta_action / action values -> choice and composite hashes differ


def test_result_binds_exact_upstream_belief_payload():
    result = run_intentional_decision(...)
    assert isinstance(result, IntentionalDecisionResult)
    assert result.goal_state.belief_state_hash == stable_content_hash(
        result.belief_state.to_dict()
    )
    assert result.model_hash == model.content_hash
    assert result.content_hash == stable_content_hash(result.to_dict())
```

- [ ] **Step 6: Update exact narrative public API expectation only**

In `tests/test_narrative_trust_api.py`, append exactly these names to `_EXPECTED_PUBLIC_API`:

```python
"GoalSpec",
"GoalModelSpec",
"GoalState",
"ChoiceModelSpec",
"IntentionalDecisionModelSpec",
"IntentionalDecisionResult",
"IntentionalDecisionResolutionError",
"GoalResolutionError",
"ChoiceResolutionError",
"run_intentional_decision",
```

Do not modify the root-isolation loop; it must prove each new name is absent from `narrative_dynamics` package root.

- [ ] **Step 7: Run targeted tests and verify intended RED locally when an execution shell is available**

Run:

```bash
python3 -m unittest tests.test_narrative_intention tests.test_narrative_trust_api
```

Expected: failures only because `narrative_dynamics.narrative.intention` / the ten scoped exports do not yet exist. Existing trust behavior must not fail for another reason.

- [ ] **Step 8: Run the full Python discovery before committing RED when an execution shell is available**

Run:

```bash
python3 -m unittest discover -s tests -p 'test*.py'
```

Expected: all pre-existing tests pass; only the new intentional tests and exact-public-surface assertion fail at the missing API boundary.

- [ ] **Step 9: Commit test-only RED atomically**

```bash
git add tests/test_narrative_intention.py tests/test_narrative_trust_api.py
git commit -m "test: define narrative intentional decision v1"
```

Verify the commit contains no production file.

- [ ] **Step 10: Open/update the feature PR and require exact-head CI RED**

PR base: `proof/narrative-dynamics-v0`.

Record feature head SHA and workflow run. Required RED evidence:

- Lean conformance/build/theorem gates remain green;
- Python reaches the new intentional tests;
- failures are only the missing intentional module/scoped exports;
- CI checkout tree corresponds to the exact test-only feature tree modulo GitHub's merge commit.

Do not implement production code until this RED is observed.

---

### Task 2: Implement immutable model records, validation, and identities

**Files:**
- Create: `narrative_dynamics/narrative/intention.py`
- Test: `tests/test_narrative_intention.py`

**Interfaces:**
- Consumes: `DecisionResolutionError`, `UncertainBeliefModelSpec`, `UncertainBeliefState`, `stable_content_hash`, `measure_implementation`, `StateCellRef`.
- Produces: the nine public record/error types except execution behavior, canonical `to_dict()`/`content_hash`, strict construction validation, stable lexical MAP helper, and internal finite-vector helpers used by Task 3.

- [ ] **Step 1: Define errors and canonical primitive validators**

Start `intention.py` with:

```python
class IntentionalDecisionResolutionError(DecisionResolutionError):
    pass


class GoalResolutionError(IntentionalDecisionResolutionError):
    pass


class ChoiceResolutionError(IntentionalDecisionResolutionError):
    pass
```

Add private helpers for non-empty trimmed text, finite float, non-negative finite float, positive finite float, immutable nested mappings, `sha256:` content-hash validation, cell sort key, normalized probability-vector validation, and lexical MAP selection.

Lexical MAP must be equivalent to:

```python
def _map_choice(policy):
    maximum = max(policy.values())
    return min(key for key, value in policy.items() if value == maximum)
```

Do not use insertion order for ties.

- [ ] **Step 2: Implement `GoalSpec` with canonical nested maps**

Implement exactly:

```python
@dataclass(frozen=True)
class GoalSpec:
    goal_id: str
    pressure: float
    cell_weights: Mapping[StateCellRef, float]
    instrumentality: Mapping[StateCellRef, Mapping[str, float]]
    cost: float = 0.0
    risk: float = 0.0
```

Construction rules:

- `goal_id` non-empty/trimmed;
- pressure/cost/risk finite and non-negative;
- cell mappings non-empty and keyed only by `StateCellRef`;
- cell weights finite and non-negative but normalization is runtime-checked against actual decision context;
- instrumentality inner keys are `sha256:` hashes and values finite (negative instrumentality is allowed);
- recursively detach maps with `MappingProxyType`;
- deterministic `to_dict()` sorts cells by `(entity_type, entity_id, state_variable)` and hypothesis hashes lexically.

- [ ] **Step 3: Implement `GoalModelSpec`, `ChoiceModelSpec`, and composite identity**

Implement:

```python
@dataclass(frozen=True)
class GoalModelSpec:
    model_id: str
    version: str
    beta_goal: float
    goals: tuple[GoalSpec, ...]

@dataclass(frozen=True)
class ChoiceModelSpec:
    model_id: str
    version: str
    beta_action: float
    values: Mapping[str, Mapping[str, float]]

@dataclass(frozen=True)
class IntentionalDecisionModelSpec:
    model_id: str
    version: str
    supported_decision_types: tuple[str, ...]
    belief_model: UncertainBeliefModelSpec
    goal_model: GoalModelSpec
    choice_model: ChoiceModelSpec
```

Rules:

- at least two unique goals;
- `beta_goal > 0`, `beta_action > 0`, both finite;
- non-empty action map per choice goal; finite values;
- composite supported decision types non-empty/unique;
- exact goal-ID equality between goal and choice model at composite construction;
- `content_hash` for each spec is `stable_content_hash(to_dict())`;
- composite `to_dict()` includes submodel content hashes and measured implementation identity for `run_intentional_decision` via a helper whose reference is resolved at call time, avoiding module-construction ordering issues.

- [ ] **Step 4: Implement `GoalState` and `IntentionalDecisionResult` validation**

Implement the exact spec fields:

```python
@dataclass(frozen=True)
class GoalState:
    model_id: str
    model_hash: str
    belief_state_hash: str
    scores: Mapping[str, float]
    policy: Mapping[str, float]
    selected_goal: str

@dataclass(frozen=True)
class IntentionalDecisionResult:
    model_id: str
    model_hash: str
    decision_id: str
    belief_state: UncertainBeliefState
    goal_state: GoalState
    conditional_action_policies: Mapping[str, Mapping[str, float]]
    action_scores: Mapping[str, float]
    action_policy: Mapping[str, float]
    selected_action: str
```

Validation must enforce exact key agreement, finite scores, normalized non-negative policies within `1e-12`, lexical MAP projection correctness, content-hash shape, and exact upstream belief hash equality. `IntentionalDecisionResult.content_hash` is `stable_content_hash(to_dict())`.

- [ ] **Step 5: Run constructor/identity/tie/numeric tests**

Run targeted tests selecting methods from `tests.test_narrative_intention` that do not require full execution, plus full file once enough exists:

```bash
python3 -m unittest tests.test_narrative_intention
```

Expected: record construction/identity validation tests move GREEN; execution-chain tests may still fail because `run_intentional_decision()` is not implemented.

- [ ] **Step 6: Commit the records/identity increment**

```bash
git add narrative_dynamics/narrative/intention.py tests/test_narrative_intention.py
git commit -m "feat: add intentional decision model records"
```

No export file change yet.

---

### Task 3: Implement the Belief -> Goal -> Choice execution semantics

**Files:**
- Modify: `narrative_dynamics/narrative/intention.py`
- Test: `tests/test_narrative_intention.py`

**Interfaces:**
- Consumes: `validate_narrative(story, domain)`, `uncertain_epistemic_state(...)`, `finite_softmax(scores, beta=...)`, `Decision`, `GenericNarrative`, `DomainSpec`.
- Produces: `run_intentional_decision(story, domain, decision_id, model) -> IntentionalDecisionResult` and all semantic guarantees in the approved spec.

- [ ] **Step 1: Implement decision lookup and upstream belief call without objective access**

Implement the public signature:

```python
def run_intentional_decision(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    model: IntentionalDecisionModelSpec,
) -> IntentionalDecisionResult:
```

First steps must be:

```python
validate_narrative(story, domain)
# validate model type, normalized decision id, locate exact decision
# reject unsupported decision type
# require unique non-empty decision.context_cells and at least one declared action
belief_state = uncertain_epistemic_state(
    story,
    domain,
    decision.actor_id,
    model.belief_model,
    decision.context_cells,
    at_time=decision.logical_time,
)
```

Do not import or call `objective_state()` anywhere in `intention.py`.

- [ ] **Step 2: Validate exact goal/context/hypothesis coverage and compute scores**

For each `GoalSpec`:

```python
if set(goal.cell_weights) != set(decision.context_cells):
    raise GoalResolutionError(...)
if set(goal.instrumentality) != set(decision.context_cells):
    raise GoalResolutionError(...)
if not math.isclose(math.fsum(goal.cell_weights.values()), 1.0,
                    rel_tol=0.0, abs_tol=1e-12):
    raise GoalResolutionError(...)
```

For each context cell, derive the exact posterior hypothesis-hash set from `belief_state.cells[cell].posterior.masses` and require exact equality with the configured instrumentality keys.

Compute:

```python
expected = math.fsum(
    goal.cell_weights[cell]
    * math.fsum(
        mass.probability * goal.instrumentality[cell][stable_content_hash(mass.value.to_dict())]
        for mass in posterior.masses
    )
    for cell, posterior in sorted_cells
)
score = goal.pressure * expected - goal.cost - goal.risk
```

Reject non-finite derived scores with `GoalResolutionError`.

- [ ] **Step 3: Compute goal policy and `GoalState`**

Use only:

```python
goal_policy = finite_softmax(goal_scores, beta=model.goal_model.beta_goal)
selected_goal = _map_choice(goal_policy)
```

Create `GoalState` with:

```python
belief_state_hash=stable_content_hash(belief_state.to_dict())
```

Do not sample a goal.

- [ ] **Step 4: Validate exact choice coverage and compute `P(a|g)`**

Require:

```python
set(model.choice_model.values) == set(goal_policy)
```

and for every goal:

```python
set(model.choice_model.values[goal_id]) == {action.id for action in decision.actions}
```

Else raise `ChoiceResolutionError`.

Compute conditional policies with:

```python
conditional[goal_id] = finite_softmax(
    model.choice_model.values[goal_id],
    beta=model.choice_model.beta_action,
)
```

No missing action or goal may be filled with `0.0`.

- [ ] **Step 5: Compute diagnostic action scores and the observable mixture**

For every declared action ID in lexical order:

```python
action_scores[action_id] = math.fsum(
    goal_policy[goal_id] * model.choice_model.values[goal_id][action_id]
    for goal_id in sorted(goal_policy)
)
action_policy[action_id] = math.fsum(
    goal_policy[goal_id] * conditional[goal_id][action_id]
    for goal_id in sorted(goal_policy)
)
```

Do not call `finite_softmax(action_scores, ...)` to build the observable policy.

- [ ] **Step 6: Apply only the approved floating residual rule**

Validate every marginal probability finite/non-negative and total within `1e-12` of `1.0`. If outside tolerance, raise `ChoiceResolutionError`.

If inside tolerance but not exact, compute `residual = 1.0 - total` and add it to the action with the largest current probability, using lexical action ID as the exact-tie breaker. Revalidate non-negativity and normalized total.

This is the only normalization correction in V1.

- [ ] **Step 7: Build deterministic result and run all intentional semantic tests**

```python
selected_action = _map_choice(action_policy)
return IntentionalDecisionResult(...)
```

Run:

```bash
python3 -m unittest tests.test_narrative_intention
```

Expected: all semantic-chain, no-leak, layer-isolation, beta separation, hypothesis-independence, mixture-not-softmax, tie, coverage, numeric, identity, and result-lineage tests pass.

- [ ] **Step 8: Run deterministic-decision and movie regression tests before export work**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_decision \
  tests.test_narrative_knives_out \
  tests.test_narrative_the_matrix \
  tests.test_narrative_memento
```

Expected: unchanged GREEN. Any failure here is a design-boundary violation; stop rather than modifying these modules/tests.

- [ ] **Step 9: Commit execution semantics**

```bash
git add narrative_dynamics/narrative/intention.py tests/test_narrative_intention.py
git commit -m "feat: add belief goal choice execution"
```

---

### Task 4: Publish exactly the scoped narrative API

**Files:**
- Modify: `narrative_dynamics/narrative/__init__.py`
- Test: `tests/test_narrative_trust_api.py`

**Interfaces:**
- Consumes: completed intentional module.
- Produces: exactly ten additional `narrative_dynamics.narrative` public names, with root package isolation unchanged.

- [ ] **Step 1: Import exactly ten intentional symbols**

Append one import block:

```python
from narrative_dynamics.narrative.intention import (
    ChoiceModelSpec,
    ChoiceResolutionError,
    GoalModelSpec,
    GoalResolutionError,
    GoalSpec,
    GoalState,
    IntentionalDecisionModelSpec,
    IntentionalDecisionResolutionError,
    IntentionalDecisionResult,
    run_intentional_decision,
)
```

- [ ] **Step 2: Append exactly the same ten names to `__all__`**

Do not reorder or remove existing names except where the formatter naturally requires wrapping. Do not edit package-root exports.

- [ ] **Step 3: Verify scoped public surface and root isolation**

Run:

```bash
python3 -m unittest tests.test_narrative_trust_api tests.test_narrative_intention
```

Expected: both GREEN; every new name exists on `narrative_dynamics.narrative` and is absent on `narrative_dynamics` root.

- [ ] **Step 4: Commit scoped exports**

```bash
git add narrative_dynamics/narrative/__init__.py tests/test_narrative_trust_api.py
git commit -m "feat: export narrative intentional decision api"
```

---

### Task 5: Full verification, self-review, and exact-tree GREEN

**Files:**
- Review only the six approved paths.
- No new source path is allowed without returning to design review.

**Interfaces:**
- Consumes: Tasks 1-4 feature tree.
- Produces: auditable GREEN evidence and an integration-ready research branch; does not move `proof/narrative-dynamics-v0` or `master` yet.

- [ ] **Step 1: Verify exact diff boundary**

Compare feature branch against `56d3910918db9f3544636726d4f42a33769029d8` and require exactly:

```text
docs/superpowers/specs/2026-08-25-narrative-intentional-decision-v1-design.md
docs/superpowers/plans/2026-08-25-narrative-intentional-decision-v1.md
narrative_dynamics/narrative/intention.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_intention.py
tests/test_narrative_trust_api.py
```

- [ ] **Step 2: Run full Python discovery**

```bash
python3 -m unittest discover -s tests -p 'test*.py'
```

Expected: `OK` with no skip/error/failure introduced by the feature.

- [ ] **Step 3: Run explicit narrative regression subset again**

```bash
python3 -m unittest \
  tests.test_narrative_intention \
  tests.test_narrative_decision \
  tests.test_narrative_trust_api \
  tests.test_narrative_knives_out \
  tests.test_narrative_the_matrix \
  tests.test_narrative_memento
```

Expected: all GREEN.

- [ ] **Step 4: Run Lean conformance/build/theorem gates using the repository's established workflow commands**

At minimum the CI workflow must show:

```text
Lean generated conformance corpus: success
full lake build: success
existing Lean theorem suite: success
NarrativeDynamics/Tests/StoryState.lean: success
NarrativeDynamics/Tests/Testimony.lean: success
```

No Lean source change is permitted to make these pass.

- [ ] **Step 5: Self-review capability isolation**

Inspect `intention.py` and require all of the following:

```text
no import of objective_state
no call to objective_state
no alternative tracked_cells argument on run_intentional_decision
belief cutoff exactly equals decision.logical_time
only uncertain_epistemic_state supplies story-derived probabilistic state
```

- [ ] **Step 6: Self-review mathematical semantics and numeric boundaries**

Require:

```text
goal score uses posterior-weighted instrumentality only
cell weights explicitly sum to 1
beta_goal and beta_action remain separate
P(a|B) is mixture of P(a|g), not softmax(action_scores)
math.fsum used for finite weighted sums
floating correction assigns only tiny residual to largest action coordinate
exact coverage failures never default values
lexical MAP ties are input-order invariant
```

- [ ] **Step 7: Self-review model and result identity**

Require:

```text
belief model content hash contributes to composite model hash
goal config and beta_goal contribute to goal/composite hash
choice table and beta_action contribute to choice/composite hash
intentional implementation identity contributes to composite hash
GoalState binds stable hash of exact UncertainBeliefState payload
IntentionalDecisionResult binds composite model hash and full layered payload
```

- [ ] **Step 8: Push final feature head and require exact-head/exact-tree CI GREEN**

Record:

- feature head SHA;
- feature tree SHA;
- GitHub Actions run/job IDs;
- PR merge-checkout SHA/tree if the workflow tests a merge commit;
- equality of tested tree and feature tree where applicable;
- full Python test count and `OK`;
- Lean build job count/success;
- explicit StoryState/Testimony gates;
- explicit intentional + deterministic + three movie tests.

- [ ] **Step 9: Stop before research-ref integration**

Do not move `proof/narrative-dynamics-v0` yet. Report the final GREEN evidence and wait for the explicit integration choice. If integration is approved, advance the research ref only with a non-force fast-forward and verify post-integration compare is identical. Leave `master` unchanged.
