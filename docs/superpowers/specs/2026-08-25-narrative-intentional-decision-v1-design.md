# Narrative Intentional Decision V1 Design

## Status

Approved architectural increment for the Generic Narrative Engine, built immediately after Narrative Uncertain Belief V1.

This design defines the first narrative-native model that explicitly separates uncertain belief, latent goal competition, and action choice:

```text
admitted evidence
    -> uncertain belief posterior
    -> goal scores / goal policy
    -> conditional action policies
    -> marginal action policy
```

Implementation is intentionally not part of this design commit. The first implementation step after review is test-only RED.

## Problem

The Generic Narrative Engine now has two distinct decision-related capabilities:

1. `narrative_dynamics.narrative.decision` can expose direct-only, epistemic, or omniscient state to a capability-limited deterministic decision hook and returns a one-hot `DecisionResult`.
2. `narrative_dynamics.narrative.uncertain` can project the admitted epistemic evidence stream into explicit finite Bayesian posterior distributions over selected narrative state cells.

These layers are useful but disconnected. A probabilistic belief update currently cannot influence a narrative-native goal model or a non-degenerate action policy. Conversely, the existing deterministic decision API compresses all motivational and action-selection structure into one decision hook.

That prevents the engine from answering several research questions that require separable latent mechanisms:

- Did behavior change because the agent's belief changed?
- Did behavior change because the same belief supported a different goal?
- Did behavior become less predictable because goal commitment weakened, or because action execution became noisy after a goal was selected?
- Can a simpler deterministic or reactive model explain the same held-out actions without the extra latent goal layer?
- Can goal-selection and action-selection parameters be recovered separately from synthetic data?

Narrative Intentional Decision V1 must therefore make the intermediate motivational structure explicit without changing the GenericNarrative factual schema or weakening the existing evidence-access boundary.

## Scientific model

For decision actor `i` at logical time `t`, V1 uses the following generative decomposition:

```text
E_i,<=t -> B_i,t -> G_i,t -> A_i,t
```

where:

- `E_i,<=t` is the existing admitted `EpistemicEvidence` history;
- `B_i,t` is an `UncertainBeliefState` over the decision's declared context cells;
- `G_i,t` is a latent goal distribution;
- `A_i,t` is the resulting action distribution.

The engine does not claim that this decomposition is the uniquely correct psychological explanation. It defines one explicit, falsifiable candidate model family whose observable action predictions can later be compared with deterministic, reactive, POMDP, or other alternatives.

## Design goals

Narrative Intentional Decision V1 must:

1. consume `UncertainBeliefState` generated only from evidence admitted to the decision actor before the decision time;
2. track exactly the decision's declared `context_cells` rather than introducing an independent hidden state scope;
3. represent a finite set of explicit candidate goals with content-hashed, immutable model assumptions;
4. derive each goal score from posterior-weighted instrumentality, explicit goal pressure, cost, and risk;
5. produce a normalized softmax goal policy with its own inverse-temperature parameter `beta_goal`;
6. represent explicit goal-conditioned values for every declared action;
7. produce a normalized conditional action policy for each goal using a separate inverse-temperature parameter `beta_action`;
8. marginalize over the latent goal policy to obtain the final observable action policy;
9. return the complete intermediate result needed to distinguish belief effects, goal effects, and choice effects;
10. use deterministic MAP projections only for human-readable `selected_goal` and `selected_action`; the research object is the complete probability distribution;
11. preserve exact model identity and upstream belief-model lineage;
12. fail closed on incomplete goal/hypothesis/action coverage, malformed numeric values, unsupported decision types, or incompatible belief state;
13. leave `decision.py`, `uncertain.py`, GenericNarrative schema V1, discrete replay, existing movie conformance fixtures, and the top-level `narrative_dynamics` package API unchanged;
14. reuse the repository's existing numerically stable finite softmax implementation rather than introducing another choice normalization algorithm;
15. keep POMDP planning out of this increment so the intentional model remains independently comparable with the existing POMDP family.

## Non-goals

V1 does **not** add:

- a new GenericNarrative schema version;
- first-class goals, drives, utility functions, or psychological traits to the narrative IR;
- a goal event that directly writes an agent's latent goal into objective story state;
- automatic goal extraction from prose;
- learned goal pressure, cost, risk, or action values;
- temporal persistence of latent goals across decisions;
- hierarchical or multi-step planning;
- POMDP belief-state planning;
- Theory-of-Mind recursion over other agents' beliefs/goals;
- social or institutional constraints on action feasibility;
- reinforcement learning of action values;
- stochastic sampling in the core narrative API;
- calibration, synthetic recovery, held-out comparison, or preregistered final-test plumbing in this increment;
- Lean formalization of the new intentional layer;
- empirical human-behavior validity claims;
- unique causal identification of hidden beliefs or goals from observed actions.

Those are later increments after the core decomposition is stable.

## Existing foundations and compatibility constraints

### Generic narrative decision boundary

`Decision` already declares:

- `actor_id`;
- `logical_time`;
- `type_name`;
- `context_cells`;
- declared `ActionOption` values.

V1 uses those declarations directly. The narrative IR remains factual/structural; motivational parameters stay outside it in a model specification.

The existing `DecisionModelSpec` / `run_decision_model()` path remains unchanged and continues to provide deterministic direct-only, epistemic, and omniscient baselines. Its one-hot `DecisionResult` is not generalized into a probabilistic result because changing that contract would silently alter existing model semantics and conformance tests.

### Uncertain belief boundary

`uncertain_epistemic_state()` already:

- validates the narrative/domain;
- consumes the actor's existing admitted epistemic evidence stream;
- enumerates finite hypotheses for enum/bool/entity-ref state variables;
- applies explicit prior and likelihood model assumptions;
- records ordered Bayesian update provenance;
- returns an immutable `UncertainBeliefState` with model ID/hash and posterior distributions.

Intentional Decision V1 treats this as the only supported probabilistic belief source. It does not reconstruct evidence, read objective truth, or call `objective_state()`.

### Existing goal mathematics

The repository already contains the binary epistemic goal equation:

```text
score = pressure * expected_instrumentality - cost - risk
```

and grounded finite-hypothesis generalizations that posterior-weight instrumentality and pass finite goal scores through stable softmax.

V1 generalizes the same semantics to narrative-native finite state cells. It does not add a direct reward for belief itself: belief affects a goal only through declared instrumentality under hypotheses.

### Existing finite softmax

`grounded_goal_softmax.finite_softmax(scores, beta=...)` is the existing numerically stable arbitrary-label softmax implementation.

V1 imports and reuses that function for both goal and conditional-action normalization. This creates a small dependency on the existing grounded-goal utility module but avoids a second normalization implementation. Refactoring `finite_softmax` into a more generic utility module is intentionally deferred because it would broaden this increment and modify already-validated code without changing V1 semantics.

## Alternatives considered

### A. Belief directly to action

Structure:

```text
belief posterior -> action score -> action policy
```

Rejected as the primary V1 model. It would connect uncertain belief to behavior with minimal code, but it would collapse motivation into action values and make it impossible to distinguish a belief change from a goal change.

A direct belief-to-action model may later be useful as a simpler comparison baseline.

### B. Belief to latent goal to action choice

Structure:

```text
belief posterior -> goal policy -> conditional action policy -> action policy
```

Chosen. It creates explicit, separately parameterized belief, goal, and choice layers while remaining finite, deterministic at the core API, and compatible with the current Generic Narrative Engine.

### C. Belief to goal to POMDP planning

Structure:

```text
belief posterior -> goal policy -> planning state -> action policy
```

Rejected for V1. The repository already has an independently specified finite POMDP model family. Folding planning into this increment would remove the clean comparison between intentional one-step choice and POMDP planning.

## Architecture

### New module

Add:

`narrative_dynamics/narrative/intention.py`

The module owns:

- immutable goal/choice model specifications;
- goal scoring from uncertain posterior distributions;
- goal softmax policy;
- goal-conditioned action softmax policies;
- latent-goal marginalization into an observable action policy;
- deterministic MAP projections;
- model/result identity and validation.

The module does not own:

- evidence admission;
- narrative replay;
- Bayesian posterior updates;
- objective state;
- generic deterministic decision hooks;
- stochastic episode sampling;
- model calibration/comparison infrastructure.

### Files unchanged by design

V1 does not modify behavior in:

- `narrative_dynamics/narrative/decision.py`;
- `narrative_dynamics/narrative/uncertain.py`;
- `narrative_dynamics/narrative/ir.py`;
- `narrative_dynamics/narrative/replay.py`;
- existing domain fixtures;
- Lean source.

The only production files expected to change are the new `intention.py` module and `narrative_dynamics/narrative/__init__.py` for scoped public exports.

## Mathematical semantics

### Belief input

For every decision context cell `c`, uncertain replay produces a posterior finite distribution:

```text
b_c(h) = P(X_c = h | admitted evidence for actor through decision time)
```

`run_intentional_decision()` tracks exactly `decision.context_cells` and calls:

```python
uncertain_epistemic_state(
    story,
    domain,
    decision.actor_id,
    model.belief_model,
    decision.context_cells,
    at_time=decision.logical_time,
)
```

No alternative tracked-cell set can be supplied through the intentional API.

### Goal expected instrumentality

Each goal `g` explicitly declares:

- a non-negative finite pressure `rho_g`;
- a non-negative finite cost `C_g`;
- a non-negative finite risk `R_g`;
- a non-negative normalized weight `w_g,c` for every decision context cell;
- a finite instrumentality `I_g,c(h)` for every finite hypothesis value of every decision context cell.

Weights must cover the context-cell set exactly and satisfy:

```text
sum_c w_g,c = 1
```

within the module's strict numerical tolerance. Zero weights are allowed but must be explicit. This avoids silently ignoring a context cell and avoids letting the number of context cells implicitly rescale goal pressure.

The expected instrumentality of goal `g` is:

```text
E[I_g | B] = sum_c w_g,c * sum_h b_c(h) * I_g,c(h)
```

The goal score is:

```text
U_g(B) = rho_g * E[I_g | B] - C_g - R_g
```

Belief has no direct reward term. If `I_g,c(h)` is constant across all hypotheses for every cell, changing belief cannot change the goal score.

### Goal policy

`GoalModelSpec` defines one positive finite inverse temperature:

```text
beta_goal > 0
```

The goal policy is:

```text
P(g | B) = softmax_g(beta_goal * U_g(B))
```

using the existing `finite_softmax()` implementation.

`selected_goal` is a deterministic MAP projection. If two or more goals have exactly equal maximum probability, the lexicographically smallest `goal_id` wins. Goal tuple/mapping insertion order must not affect the result.

### Conditional action policies

`ChoiceModelSpec` defines:

- one positive finite inverse temperature `beta_action`;
- a finite value `V_g(a)` for every pair of configured goal ID and declared decision action ID.

Coverage must be exact:

```text
configured goals == goal policy keys
configured actions for every goal == decision.actions IDs
```

For each goal independently:

```text
P(a | g) = softmax_a(beta_action * V_g(a))
```

using the same existing `finite_softmax()` implementation.

### Observable action policy

The final action policy marginalizes the latent goal:

```text
P(a | B) = sum_g P(a | g) * P(g | B)
```

The implementation uses deterministic sorted goal/action iteration and `math.fsum` for stable accumulation.

The result also reports an interpretable goal-policy-weighted action value:

```text
Q(a) = sum_g P(g | B) * V_g(a)
```

This `Q(a)` is diagnostic only. The observable policy is the mixture of conditional softmax policies above; V1 must not replace the mixture with `softmax(Q)` because those are different models.

`selected_action` is a deterministic MAP projection from the final action policy, with lexicographically smallest action ID as the exact-tie breaker.

## Why `beta_goal` and `beta_action` are separate

The two inverse temperatures represent different model mechanisms:

- `beta_goal` controls concentration of latent goal commitment after belief-conditioned goal scoring;
- `beta_action` controls concentration of action execution conditional on a goal.

They must remain distinct parameters and distinct identity fields.

Examples:

- high `beta_goal`, low `beta_action`: the agent's goal is sharply determined but action execution is diffuse;
- low `beta_goal`, high `beta_action`: goal competition remains unresolved, but behavior is highly consistent once conditioned on each possible goal.

A later synthetic-recovery increment can test whether data can recover these parameters separately. V1 only establishes the correct generative surface.

## Public records

### `GoalSpec`

Immutable configured goal declaration:

```python
GoalSpec(
    goal_id: str,
    pressure: float,
    cell_weights: Mapping[StateCellRef, float],
    instrumentality: Mapping[StateCellRef, Mapping[str, float]],
    cost: float = 0.0,
    risk: float = 0.0,
)
```

The inner `instrumentality` keys are stable content hashes of exact `TypedValue` hypotheses, matching `BeliefDistribution` hypothesis identity.

Construction validates only canonical numeric/mapping shape that can be checked without a narrative instance. Runtime validation checks exact context-cell and hypothesis coverage against the actual `UncertainBeliefState`.

Canonical serialization sorts cells by `(entity_type, entity_id, state_variable)` and inner hypothesis hashes lexicographically.

### `GoalModelSpec`

Immutable model declaration:

```python
GoalModelSpec(
    model_id: str,
    version: str,
    beta_goal: float,
    goals: tuple[GoalSpec, ...],
)
```

Requirements:

- at least two goals in V1;
- unique goal IDs;
- positive finite `beta_goal`;
- immutable canonical goal declarations.

At least two goals are required because a one-goal softmax provides no goal-competition model and would add an unidentifiable latent layer without behavioral consequence.

`content_hash` binds model ID/version, `beta_goal`, and all goal configuration.

### `GoalState`

Immutable computed state:

```python
GoalState(
    model_id: str,
    model_hash: str,
    belief_state_hash: str,
    scores: Mapping[str, float],
    policy: Mapping[str, float],
    selected_goal: str,
)
```

Requirements:

- score/policy keys match exactly;
- at least two goals;
- all scores finite;
- policy finite, non-negative, normalized within tolerance;
- `selected_goal` belongs to the policy and is the deterministic MAP/tie-break result;
- `belief_state_hash` is the stable content hash of the exact upstream `UncertainBeliefState.to_dict()` payload.

`GoalState` therefore binds the motivational result to both the configured goal model and the exact upstream belief state.

### `ChoiceModelSpec`

Immutable configured choice declaration:

```python
ChoiceModelSpec(
    model_id: str,
    version: str,
    beta_action: float,
    values: Mapping[str, Mapping[str, float]],
)
```

where:

```text
values[goal_id][action_id] = V_g(a)
```

Requirements:

- non-empty goal mapping;
- non-empty action mapping for every goal;
- finite values;
- positive finite `beta_action`;
- immutable canonical mapping.

Runtime execution validates exact goal/action coverage. No missing value defaults to zero; callers must declare zero explicitly if zero is the intended model assumption.

`content_hash` binds model ID/version, `beta_action`, and the complete goal/action value table.

### `IntentionalDecisionModelSpec`

Composite immutable model:

```python
IntentionalDecisionModelSpec(
    model_id: str,
    version: str,
    supported_decision_types: tuple[str, ...],
    belief_model: UncertainBeliefModelSpec,
    goal_model: GoalModelSpec,
    choice_model: ChoiceModelSpec,
)
```

Requirements:

- non-empty unique supported decision types;
- typed submodels;
- exact goal-ID agreement between `goal_model.goals` and `choice_model.values` at construction time.

`content_hash` binds:

- composite model ID/version;
- supported decision types;
- belief-model content hash;
- goal-model content hash;
- choice-model content hash;
- measured implementation identity of the native intentional-decision algorithm.

This means either model assumptions or behavior-affecting implementation code changes model identity.

### `IntentionalDecisionResult`

Immutable full result:

```python
IntentionalDecisionResult(
    model_id: str,
    model_hash: str,
    decision_id: str,
    belief_state: UncertainBeliefState,
    goal_state: GoalState,
    conditional_action_policies: Mapping[str, Mapping[str, float]],
    action_scores: Mapping[str, float],
    action_policy: Mapping[str, float],
    selected_action: str,
)
```

The result preserves all model layers rather than returning only the selected action.

Validation requires:

- `model_hash` to match the configured composite model;
- conditional policy goal keys to equal `goal_state.policy` keys;
- every conditional action policy to cover the same declared action set;
- each conditional policy normalized and finite/non-negative;
- `action_scores` and `action_policy` to cover that same action set;
- final action policy normalized and finite/non-negative;
- `selected_action` to be the deterministic MAP/tie-break result;
- `goal_state.belief_state_hash == stable_content_hash(belief_state.to_dict())`.

`to_dict()` serializes the complete layered result and `content_hash` identifies it.

## Error model

Add:

```python
class IntentionalDecisionResolutionError(DecisionResolutionError): ...
class GoalResolutionError(IntentionalDecisionResolutionError): ...
class ChoiceResolutionError(IntentionalDecisionResolutionError): ...
```

The new base subclasses the existing `DecisionResolutionError` so callers that already treat invalid decision resolution generically can catch both deterministic and intentional failures.

Use `GoalResolutionError` for failures such as:

- goal context-cell coverage does not exactly match the decision context;
- goal weights are invalid or do not normalize;
- goal hypothesis keys do not exactly match an upstream posterior;
- a goal score or derived policy is non-finite/invalid;
- fewer than two usable goals remain.

Use `ChoiceResolutionError` for failures such as:

- choice goal IDs do not match the goal model/state;
- action IDs do not exactly match the decision's declared actions;
- a conditional action policy is invalid;
- marginal action mass is invalid or not normalized.

Use `IntentionalDecisionResolutionError` directly for composite execution failures such as:

- decision ID does not exist;
- decision type is unsupported;
- composite model type is invalid.

`UncertainBeliefResolutionError` remains its existing type and is allowed to propagate from the upstream belief layer. Narrative/domain structural validation errors also retain their existing types.

## Main execution API

Add:

```python
run_intentional_decision(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    model: IntentionalDecisionModelSpec,
) -> IntentionalDecisionResult
```

Processing:

1. validate the narrative/domain with the existing validator;
2. validate the composite model and locate the exact decision;
3. reject unsupported decision type;
4. require a non-empty unique decision context-cell tuple and at least one declared action;
5. call `uncertain_epistemic_state()` for the decision actor, exactly the decision context cells, and cutoff `decision.logical_time`;
6. validate every goal's exact context-cell coverage, normalized cell weights, and exact hypothesis coverage against each posterior;
7. compute posterior-weighted expected instrumentality and goal score for every goal;
8. pass the goal-score mapping to existing `finite_softmax(..., beta=model.goal_model.beta_goal)`;
9. construct `GoalState`, including exact belief-state hash and deterministic selected goal;
10. validate choice-model goal IDs and exact declared action-ID coverage;
11. for every goal, compute its conditional action policy with existing `finite_softmax(..., beta=model.choice_model.beta_action)`;
12. compute diagnostic `action_scores` by goal-policy-weighted action values;
13. marginalize conditional action policies by the goal policy using sorted iteration and `math.fsum`;
14. validate and normalize only for floating residual tolerance; malformed model inputs are never silently repaired;
15. derive deterministic MAP `selected_action` with lexical tie-break;
16. return the complete `IntentionalDecisionResult`.

### Floating-point residual rule

`finite_softmax()` produces normalized probabilities but floating arithmetic can leave sums microscopically different from exactly `1.0` after mixture accumulation.

V1 validates sums with the same strict absolute tolerance used by uncertain belief (`1e-12`). If the marginal action sum is within tolerance of `1.0`, the implementation assigns only the final floating residual to the largest action-probability coordinate, preserving non-negativity and deterministic identity. If the sum is outside tolerance, execution fails closed.

This is numerical housekeeping, not semantic renormalization of malformed model assumptions.

## Capability and leak boundary

The intentional model receives no objective-state capability.

The only story-derived uncertainty input is the `UncertainBeliefState` produced through existing epistemic admission for the actor. Official model configuration contains only explicit goal/choice assumptions.

Therefore changing an objective fact that the actor neither observes nor receives must not affect:

- the uncertain posterior;
- goal scores/policy;
- conditional action policies;
- marginal action policy.

If the changed objective fact alters a later admitted observation or received claim, the effect is allowed only through the normal evidence path.

## Deterministic core vs stochastic simulation

The core narrative API is deterministic.

`run_intentional_decision()` returns probability distributions and deterministic MAP projections. It does **not** accept an RNG or seed and does not sample a latent goal or action.

This separation is intentional:

- deterministic core outputs are easy to test, compare, hash, and use in counterfactual analysis;
- a later `SimulationRunner` adapter can sample `G` and `A` with supplied seeds while preserving the same computed policies;
- model comparison can choose whether to score complete predicted action probabilities or sampled traces.

## Relationship to the three model families

After V1, the repository has a cleaner hierarchy:

```text
Family 1: epistemic / hypothesis models
    EpistemicEvidence -> UncertainBeliefState

Family 2: intentional decision models
    UncertainBeliefState -> GoalState -> ActionPolicy

Family 3: behavioral control / model comparison
    reactive baseline <-> POMDP planning <-> later intentional adapters
```

V1 deliberately does not merge Family 2 and Family 3. The scientific comparison remains meaningful only if a one-step intentional model and a POMDP planning model can succeed or fail independently.

## Public API boundary

Export from `narrative_dynamics.narrative` only:

- `GoalSpec`
- `GoalModelSpec`
- `GoalState`
- `ChoiceModelSpec`
- `IntentionalDecisionModelSpec`
- `IntentionalDecisionResult`
- `IntentionalDecisionResolutionError`
- `GoalResolutionError`
- `ChoiceResolutionError`
- `run_intentional_decision`

Do **not** export these symbols from package root `narrative_dynamics`.

Existing narrative exports remain unchanged except for appending these ten symbols.

## Test strategy

Implementation must follow strict test-only RED before production code.

### RED files

Add tests first while `narrative_dynamics/narrative/intention.py` is still absent:

- `tests/test_narrative_intention.py`
- update `tests/test_narrative_trust_api.py` only for the exact narrative public surface / root-isolation assertions required by the new exports.

The first RED must keep all existing Lean/Python behavior green and fail only at the new intentional-decision API boundary.

### Required semantic tests

1. **Belief changes goal and action through the declared chain**
   - use one narrative decision with at least two finite context hypotheses, two goals, and two actions;
   - construct two otherwise matched narratives that differ only in an admitted claim/reception available to the actor;
   - confirm the uncertain posterior differs;
   - confirm the same fixed goal/choice model yields different goal policy;
   - confirm the final action policy changes in the predicted direction.

2. **Unobserved objective change cannot leak into behavior**
   - change objective state without adding actor observation/reception;
   - run the same intentional model;
   - posterior, goal state, conditional action policies, and final action policy must be identical.

3. **Same belief, different goal model changes behavior**
   - keep story, belief model, and choice model identical;
   - alter only explicit goal instrumentality/pressure configuration;
   - upstream belief state must remain identical;
   - goal policy and final action policy must change.

4. **Same belief and goal model, different choice model changes only the choice layer**
   - keep belief and goal model identical;
   - alter only goal-conditioned action values or `beta_action`;
   - belief state and goal state must remain identical;
   - conditional/final action policies must change.

5. **`beta_goal` and `beta_action` are behaviorally distinct**
   - increasing `beta_goal` with fixed goal-score ordering makes goal policy more concentrated while conditional action policies remain exactly unchanged;
   - increasing `beta_action` with fixed goal model leaves goal state exactly unchanged while each non-degenerate conditional action policy becomes more concentrated.

6. **Hypothesis-independent instrumentality blocks belief effect**
   - for every goal, set instrumentality constant across each cell's hypotheses;
   - change admitted evidence so posterior changes;
   - goal scores/policy and action policy must remain unchanged.

7. **Latent-goal marginalization is not softmax of expected action values**
   - choose a fixture where the mixture-of-conditionals policy differs numerically from `softmax(action_scores)`;
   - assert the implementation matches the explicit mixture formula.

8. **Deterministic tie breaking**
   - exact equal goal scores select lexicographically smallest goal ID;
   - exact equal final action probabilities select lexicographically smallest action ID;
   - reversing input mapping/tuple order does not change serialized result identity.

9. **Exact coverage fails closed**
   - missing context-cell weight;
   - extra context-cell weight;
   - missing hypothesis utility;
   - extra hypothesis utility;
   - missing goal in choice table;
   - extra goal in choice table;
   - missing declared action;
   - invented undeclared action;
   - each produces the appropriate typed goal/choice error rather than defaulting a value.

10. **Numeric validation fails closed**
    - negative/non-finite pressure, cost, risk, or cell weight;
    - cell weights not normalized;
    - non-finite instrumentality/action value;
    - non-positive/non-finite beta values;
    - invalid derived probability vector injected through a test hook/patch if needed;
    - typed construction or resolution error is required.

11. **Identity lineage**
    - changing uncertain-belief parameters changes composite model hash;
    - changing `beta_goal` changes goal/composite model identity;
    - changing goal instrumentality changes goal/composite model identity;
    - changing `beta_action` or action values changes choice/composite model identity;
    - changing implementation bytes remains covered by measured implementation identity contract;
    - result goal-state belief hash equals exact upstream belief payload hash.

12. **Existing deterministic decision regression**
    - all existing `test_narrative_decision.py` behavior remains unchanged;
    - direct/epistemic/omniscient reference actions stay exactly one-hot and deterministic.

13. **Existing movie conformance regression**
    - Knives Out, The Matrix, and Memento conformance tests remain unchanged and green;
    - no new intentional model is silently substituted for their existing decision models.

14. **Public API isolation**
    - exactly the ten approved new symbols are added to `narrative_dynamics.narrative`;
    - none appear at package root;
    - no unrelated public symbol is added.

## RED/GREEN integration discipline

The implementation branch is based exactly on research head:

```text
56d3910918db9f3544636726d4f42a33769029d8
```

Work branch:

```text
work/narrative-intentional-decision-v1
```

Implementation sequence after this spec is reviewed:

1. write the implementation plan;
2. add test-only RED and commit it atomically;
3. verify exact-head RED in GitHub Actions;
4. add the smallest production implementation needed for GREEN;
5. update the scoped narrative public export surface only when behavior is implemented;
6. run the complete Lean conformance/build/theorem gates and full Python discovery;
7. verify existing narrative decision and three movie conformance tests explicitly;
8. self-review model identity, capability isolation, exact coverage, and numerical boundaries;
9. require final exact-tree CI success;
10. advance `proof/narrative-dynamics-v0` only by non-force fast-forward after explicit integration choice;
11. leave `master` unchanged.

## Expected implementation diff boundary

The intended implementation diff is limited to:

```text
docs/superpowers/specs/2026-08-25-narrative-intentional-decision-v1-design.md
docs/superpowers/plans/2026-08-25-narrative-intentional-decision-v1.md
narrative_dynamics/narrative/intention.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_intention.py
tests/test_narrative_trust_api.py
```

No Lean source, GenericNarrative IR, replay semantics, existing decision implementation, uncertain-belief implementation, movie domain module, root package export, runtime registry, prison adapter, or observational protocol file is expected to change.

If implementation requires changes outside this boundary, stop and re-evaluate the design rather than silently widening scope.

## Claim boundary

The strongest claim supported by V1 is:

> Narrative Dynamics can compute a deterministic, provenance-linked intentional decision distribution in which an actor's admitted evidence produces an explicit finite Bayesian belief state, that belief state changes explicit goal scores and a latent goal policy through declared instrumentality, and the latent goal policy is marginalized through explicit goal-conditioned action policies into an observable action policy.

V1 does **not** establish:

- that human agents actually use this decomposition;
- that hidden goals are uniquely identifiable from behavior;
- that the chosen goal/action utility tables are empirically valid;
- causal validity beyond the declared synthetic/model assumptions;
- population representativeness;
- general human Theory of Mind;
- superior predictive performance over reactive or POMDP alternatives.

Those claims require later recovery, held-out observational comparison, and model-adequacy work.
