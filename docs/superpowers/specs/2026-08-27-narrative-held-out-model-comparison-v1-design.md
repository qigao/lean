# Narrative Held-Out Three-Model Comparison V1 Design

Date: 2026-08-27

Status: written for review

Roadmap: #27 — P1 `Compare against reactive and intentional models on held-out scenarios`

Integrated base: `proof/narrative-dynamics-v0` at `c9f1b97309ffd3c40fc1e2f6d9cc706dd715f765`

Work branch: `work/narrative-held-out-model-comparison-v1`

## 1. Purpose

The generic narrative engine now has three runtime decision families behind one typed dispatch boundary:

- Reactive;
- Intentional belief → goal → choice;
- Planning/POMDP.

The remaining P1 requirement is not another unit-level equivalence fixture. It is to compare those three runtime families on an actually held-out partition under the repository's already-preregistered observational evaluation protocol.

This feature connects the three generic narrative runtime families to that protocol without changing the protocol itself.

```text
prison_initial_choice_v1.json
        |
        +--> TRAIN ---------------------> beta fitting
        |
        +--> SELECTION_VALIDATION ------> one frozen beta per family
        |
        +--> FINAL_TEST ----------------> preregistered comparison only
                                             |
              +------------------------------+------------------------------+
              |                              |                              |
              v                              v                              v
      Generic Reactive              Generic Intentional             Generic Planning
              |                              |                              |
              +---------------- RuntimeDecisionModelSpec ------------------+
                                             |
                                             v
                                    run_runtime_decision()
                                             |
                                             v
                               RuntimeDecisionDispatchResult
                                             |
                                             v
                             ModelRun.outcome["initial_policy"]
                                             |
                                             v
                             prison_initial_action_metrics
                                             |
                                             v
                          compare_models_on_final_partition()
```

The scientific object is the final-test comparison report. The adapter is only the compatibility boundary that lets the generic narrative runtime participate in the existing research protocol.

## 2. Chosen architecture

Use a benchmark adapter, not a comparison-core modification.

Add one production module:

`narrative_dynamics/adapters/narrative_prison.py`

It exposes three stable model sources/factories, one per runtime family. Each source accepts the existing prison `Scenario`, validates it, builds one canonical generic narrative benchmark instance, dispatches the requested runtime family through `RuntimeDecisionModelSpec` / `run_runtime_decision()`, and returns a normal `ModelRun` whose `initial_policy` has the canonical categories `scout`, `escape`, `submit`.

The following modules remain unchanged:

- `narrative_dynamics/model_comparison.py`;
- `narrative_dynamics/observations/preregistration.py`;
- `narrative_dynamics/observations/comparison.py`;
- `narrative_dynamics/observations/release.py`;
- generic Reactive / Intentional / Planning runtime implementations;
- the committed observation fixture.

This preserves the existing fail-closed rules for dataset identity, partition identity, target construction, metric identity, loss identity, seed plan, candidate identity, selection provenance, and final-test role.

## 3. Non-goals

This P1 feature does not:

- add or change stochastic world dynamics;
- add stochastic observation projection;
- add RNG to the generic narrative decision APIs;
- change scheduler or conflict-resolution semantics;
- change GenericNarrative IR or DomainSpec public contracts;
- change the held-out ranking rule;
- change categorical Brier loss or target construction;
- create a new observational dataset;
- alter the committed counts in `fixtures/observations/prison_initial_choice_v1.json`;
- claim that the synthetic fixture is empirical human data;
- fit `beta_goal` separately;
- recover goal pressure, instrumentality, transition, or observation parameters;
- perform P2 intentional-vs-reactive identification analysis;
- perform P2 intentional-vs-POMDP identification analysis;
- add held-out log-score robustness;
- add non-identifiability diagnostics;
- modify Lean sources.

Those remain separate P2 work.

## 4. Existing protocol invariants to preserve

The feature reuses the existing fixture and protocol machinery exactly.

The committed fixture already has all three data roles:

- train;
- selection validation;
- final test.

The final-test pair `final-1` and `final-2` holds every scenario field fixed except `guard_persistence`, changing it from `0.2` to `0.9`. Their observed initial-choice counts differ materially. This pair is therefore the key held-out discriminator for whether a model family uses future transition structure rather than only current-state evidence or immediate goals.

The comparison protocol must continue to bind:

1. exact dataset content hash;
2. exact train / selection / final partition hashes;
3. exact categorical target specification;
4. exact final target payload and target-construction lineage;
5. exact `prison_initial_action_metrics` identity;
6. exact categorical Brier loss identity;
7. exact final simulation seed tuple;
8. exact frozen candidate set;
9. exact frozen candidate parameters and selection manifest hashes;
10. exact runtime source identities;
11. fixed ranking rule `(mean_loss, worst_loss, model_name)`.

No adapter code is permitted to weaken those checks.

## 5. Canonical prison scenario boundary

The adapter accepts exactly the existing prison scenario schema:

- `prior_weak`;
- `signal_accuracy`;
- `guard_persistence`;
- `escape_reward`;
- `capture_cost`;
- `submit_reward`;
- `scout_cost`;
- `discount`;
- `horizon`.

Validation semantics remain aligned with the current finite prison adapters:

- probabilities must be finite and in `[0, 1]`;
- `signal_accuracy >= 0.5`;
- rewards/costs finite;
- non-negative escape reward, capture cost, and scout cost;
- horizon is exactly `1` or `2`;
- candidate parameter mapping contains exactly one positive finite parameter, `beta`.

For `horizon == 1`, the authored generic narrative decision exposes only `escape` and `submit`. The adapter still emits a three-coordinate output policy with `scout = 0.0` so the observational metric contract remains unchanged.

For `horizon == 2`, the authored root action set is exactly `scout`, `escape`, `submit`.

## 6. Canonical generic narrative benchmark instance

Every prison `Scenario` is translated into a fresh canonical generic narrative benchmark instance. Translation is deterministic and contains no RNG.

The benchmark uses:

- one prisoner actor;
- one guard-state subject;
- a typed hidden guard-status cell with values `weak`, `strong`, and an internal absorbing `terminal` value;
- a typed signal cell with `clear`, `alarm`, and `none` values;
- one initial prison-choice decision;
- the action ids `scout`, `escape`, `submit` as permitted by horizon;
- a decision context containing the guard-status and signal cells;
- a runtime evidence ledger at step zero bound to the exact translated story and domain.

The runtime belief model represents the root guard uncertainty with exactly two positive hypotheses:

```text
P(weak)   = prior_weak
P(strong) = 1 - prior_weak
P(terminal) = 0
```

`terminal` exists only to model action-dependent episode termination inside the finite-horizon generic planner. It is never a positive root hypothesis.

The scenario identity remains the original `Scenario` identity used by `SimulationRunner`; the translated story/domain/runtime hashes are emitted in the model result for auditability but do not replace the outer scenario identity.

## 7. Unified dispatch requirement

All three adapters must execute through the same dispatch API:

```python
RuntimeDecisionModelSpec(model_kind=..., model=...)
run_runtime_decision(story, domain, decision_id, ledger, decision_model)
```

Production benchmark code must not call `run_runtime_reactive_decision`, `run_runtime_intentional_decision`, or `run_runtime_planning_decision` directly.

The returned `RuntimeDecisionDispatchResult` is the authoritative nested result. The adapter derives `initial_policy` only from `dispatch.action_policy`.

The adapter must not independently recompute or replace the selected action or policy after dispatch.

## 8. Reactive candidate semantics

Candidate name:

`generic-narrative-prison-reactive`

The Reactive candidate uses `RuntimeReactiveDecisionModelSpec` and a prison-specific score hook.

Its score hook reproduces the already-established finite prison reactive root action-value semantics:

- direct `escape` value uses `prior_weak`, `escape_reward`, and `capture_cost`;
- `submit` value is `submit_reward`;
- for horizon 2, `scout` uses cue discriminability and terminal reactive soft choice exactly as the current finite reactive adapter;
- `guard_persistence` is not read by the reactive score hook and cannot affect the reactive initial policy;
- the model does not construct a posterior belief state;
- the model does not instantiate goals;
- the model does not run the planning solver.

`beta` is used both by the existing reactive scout-value construction and by the root runtime reactive softmax, matching the legacy finite-reactive semantics.

Reference requirement: for every committed prison fixture scenario and every benchmark beta-grid value, the generic Reactive adapter's `initial_policy` must match `FinitePrisonReactiveModel` to tight numerical tolerance.

This is an equivalence witness for the adapter translation, not the held-out conclusion.

## 9. Intentional candidate semantics

Candidate name:

`generic-narrative-prison-intentional`

The Intentional candidate uses the full runtime belief → goal → choice path.

The runtime belief state tracks the hidden guard-status cell using the scenario root prior. It does not use `guard_persistence`.

Two fixed benchmark goals are defined:

### 9.1 Freedom goal

Instrumentality on the guard-status posterior:

- weak: `+1`;
- strong: `-1`.

Goal-conditional action values:

- `escape = escape_reward`;
- `submit = 0`;
- for horizon 2, `scout = (2 * signal_accuracy - 1) * escape_reward - scout_cost`.

### 9.2 Safety goal

Instrumentality on the guard-status posterior:

- weak: `-1`;
- strong: `+1`.

Goal-conditional action values:

- `escape = -capture_cost`;
- `submit = submit_reward`;
- for horizon 2, `scout = (2 * signal_accuracy - 1) * capture_cost - scout_cost`.

Both goals use unit pressure, zero additional cost/risk, and one guard-status cell with weight `1.0`.

`beta_goal` is fixed at `1.0` for this P1 benchmark and is part of runtime model identity. It is not fitted.

The single fitted `beta` is used as `ChoiceModelSpec.beta_action`.

Consequences:

- the Intentional candidate is genuinely belief-conditioned and goal-mediated;
- it can value information acquisition through the goal-specific `scout` values;
- it does not model future guard-state transition dynamics;
- changing only `guard_persistence` must leave its root policy unchanged;
- identifying `beta_goal`, goal pressure, or instrumentality remains P2.

## 10. Planning candidate semantics

Candidate name:

`generic-narrative-prison-planning`

The Planning candidate uses `RuntimePlanningDecisionModelSpec` and the generic finite-horizon soft Bellman solver.

### 10.1 Hidden states

The hidden state set is exactly:

- `weak`;
- `strong`;
- `terminal`.

The root joint belief preserves the runtime posterior marginals and assigns zero mass to `terminal`.

### 10.2 Horizon 1

Root schedule:

```text
(escape, submit)
```

No continuation depth exists. Root values are the expected immediate utilities under the root weak/strong belief.

### 10.3 Horizon 2

Action schedule:

```text
depth 0: (scout, escape, submit)
depth 1: (escape, submit)
```

At depth 0:

- `scout` keeps weak/strong state unchanged, incurs `-scout_cost`, and exposes the informative `clear`/`alarm` observation model;
- direct `escape` and direct `submit` transition to `terminal` after their immediate reward, so their continuation value is exactly zero.

Observation likelihood after `scout` is:

```text
P(clear | weak)   = signal_accuracy
P(alarm | weak)   = 1 - signal_accuracy
P(clear | strong) = 1 - signal_accuracy
P(alarm | strong) = signal_accuracy
```

Direct root `escape` / `submit` emit only `none` with probability `1` from the absorbing terminal branch.

At depth 1:

- `escape` evolves weak/strong according to `guard_persistence`, then rewards `escape_reward` in weak and `-capture_cost` in strong;
- `submit` rewards `submit_reward` without needing information;
- every action from `terminal` has reward `0` and remains terminal.

The model uses the scenario `discount` and fitted `beta` directly in `RuntimePlanningDecisionModelSpec`.

This order deliberately matches the existing finite prison POMDP semantics:

```text
observe current guard signal
        -> posterior current guard
        -> predict future guard with persistence
        -> choose terminal route
```

Reference requirement: for every committed prison fixture scenario and every benchmark beta-grid value, the generic Planning adapter's root `initial_policy` must match `FinitePrisonPOMDPModel` to tight numerical tolerance.

This reference equality is required before held-out comparison is accepted.

## 11. Model-source identity and lineage

Each of the three outer benchmark model sources has a stable explicit identity containing at least:

- benchmark model name;
- benchmark version;
- model family (`reactive`, `intentional`, `planning`);
- benchmark implementation revision;
- measured implementation identity of the outer adapter;
- measured implementation identity of the family-specific benchmark builder/hook set;
- measured implementation identity of `run_runtime_decision`.

Scenario-specific values such as `prior_weak` and `guard_persistence` are inputs, not candidate identity. They are already bound by the outer `Scenario` manifest identity and are also reflected in the translated nested runtime model/result hashes.

A changed adapter implementation must therefore change `component_identity()` and fail comparison against an already-frozen candidate before final execution.

## 12. ModelRun output contract

Every family returns a `ModelRun` with no sampled behavioral choice required for the metric.

The canonical outcome includes:

```text
model_kind
initial_policy
selected_action
runtime_dispatch_hash
runtime_dispatch
translated_story_hash
translated_domain_hash
runtime_ledger_hash
```

`initial_policy` always contains exactly:

```text
scout
escape
submit
```

For horizon 1, `scout` is exactly `0.0` and the two runtime probabilities are copied to `escape` / `submit`.

The adapter may ignore the supplied RNG because the scientific observable is the predicted action distribution, not one sampled action. `SimulationRunner` still owns and records the preregistered seed plan, so seed drift remains rejected by the existing protocol.

## 13. Training, selection, and final-test flow

The integration test uses the existing committed fixture and existing beta grid:

```text
beta in (0.5, 1.0, 2.0, 4.0)
```

For each family independently:

1. construct TRAIN targets;
2. run `fit_training_target_grid` on only the training partition;
3. convert candidate parameters into `ParameterAcceptanceSet` with training manifest lineage;
4. construct the SELECTION_VALIDATION held-out suite;
5. run `select_on_validation_suite`;
6. freeze exactly one `beta` using `FrozenModelCandidate.from_selection`.

Only after all three candidates are frozen:

7. create one `PreregisteredEvaluationProtocol` with all three candidates;
8. bind the existing FINAL_TEST targets;
9. call `compare_models_on_final_partition` once;
10. attest the resulting report.

The Planning family is the protocol baseline because the roadmap item is explicitly the Planning/POMDP comparison against Reactive and Intentional alternatives.

No final-test result may feed back into training, accepted parameter sets, selection, candidate construction, thresholds, metric choice, or candidate membership.

## 14. Scientific assertions versus software assertions

Tests must prove protocol correctness and deterministic execution, not hard-code a desired scientific winner.

Software assertions include:

- exactly three preregistered candidates;
- one frozen selected beta per family;
- same final cases, metric, loss, and final seeds for all three;
- report lineage is attestable;
- ranking is deterministic;
- Reactive and Planning reference equivalence to the legacy prison adapters;
- final persistence-pair policy invariance for Reactive;
- final persistence-pair policy invariance for Intentional;
- final persistence-pair sensitivity for Planning;
- implementation or candidate identity drift fails before final model execution.

The test suite must not assert `planning is best` merely because that is expected. The observed final ranking and numeric losses belong in PR evidence and roadmap evidence.

## 15. Error handling

The benchmark adapter fails closed.

Before dispatch it rejects:

- wrong scenario shape;
- non-finite or out-of-range scenario values;
- unsupported horizon;
- any parameter mapping other than exactly positive finite `beta`;
- malformed translated narrative/domain bindings.

Runtime family errors remain typed through `RuntimeDecisionDispatchError` internally and are wrapped by the outer adapter as a benchmark-specific `ValueError` only if the normal `SimulationRunner` model-source contract requires that boundary.

No invalid family result may be converted into a partial policy.

## 16. Production scope

Allowed production changes:

- add `narrative_dynamics/adapters/narrative_prison.py`;
- optionally add narrow exports in `narrative_dynamics/adapters/__init__.py` if tests or users require them.

Not allowed without a design amendment:

- `narrative_dynamics/model_comparison.py`;
- `narrative_dynamics/observations/*`;
- `narrative_dynamics/narrative/runtime_reactive.py`;
- `narrative_dynamics/narrative/runtime_intention.py`;
- `narrative_dynamics/narrative/runtime_planning.py`;
- `narrative_dynamics/narrative/runtime_decision_dispatch.py`;
- scheduler/world/conflict/projection production files;
- fixture JSON;
- Lean files.

## 17. Test scope and TDD order

Primary new test file:

`tests/test_narrative_held_out_model_comparison.py`

The first implementation phase is test-only RED. It must establish at least:

1. the three public benchmark model factories/sources are missing;
2. adapter reference-equivalence expectations for Reactive and Planning;
3. unified-dispatch output contract for all three families;
4. Intentional persistence-invariance expectation;
5. the full train → selection → preregistration → final comparison path for exactly three generic narrative candidates;
6. final report attestation and lineage expectations.

The authoritative RED commit contains no production implementation in `narrative_prison.py`.

GREEN then adds only the minimal benchmark adapter required to satisfy those tests.

Existing protocol drift tests remain authoritative for generic partition/loss/seed/extractor/candidate drift; the new tests add only benchmark-specific identity and integration coverage rather than duplicating the entire comparison-core suite.

## 18. Acceptance criteria

The P1 roadmap checkbox can be marked complete only when all are true:

- the work branch is based on the exact integrated baseline recorded above;
- an authoritative test-only RED commit is preserved and CI fails for the intended missing benchmark boundary;
- the generic Reactive adapter matches legacy finite Reactive policies across the committed fixture and beta grid;
- the generic Planning adapter matches legacy finite POMDP policies across the committed fixture and beta grid;
- the Intentional candidate demonstrably runs belief → goal → choice through unified runtime dispatch;
- all three candidates are trained and selected without final-test access;
- all three are frozen into one `PreregisteredEvaluationProtocol`;
- one final-test comparison runs with the exact preregistered metric, loss, seeds, targets, and candidates;
- the final comparison report is attestable;
- the final persistence pair distinguishes Planning sensitivity from Reactive/Intentional non-lookahead invariance;
- full Lean/Python narrative proof CI is GREEN on the exact PR head;
- no P2 identification or stochasticity checkbox is claimed complete.

## 19. Roadmap interpretation after merge

After integration, issue #27 may mark only this P1 item complete:

`Compare against reactive and intentional models on held-out scenarios`

The issue remains open.

P2 remains unchanged, including stochastic world/observation models, separate aleatoric vs decision stochasticity, intentional-vs-reactive identification, intentional-vs-POMDP identification, information interventions, synthetic recovery, Brier/log-score robustness, observational equivalence analysis, and explicit non-identifiability reporting.
