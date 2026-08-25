# Narrative Runtime Percept Admission V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an immutable runtime percept-admission ledger and deterministic runtime cognition replay so authored cognition at one source cutoff can evolve from admitted runtime percepts without objective-world leakage.

**Architecture:** Add two sidecar modules. `runtime_perception.py` owns engine-produced percept admission, provenance, blackout batches, and the immutable world/ledger hash chain. `runtime_cognition.py` recomputes authored epistemic/uncertain seed state at `source_at_time`, then replays only per-agent runtime percept semantics; runtime likelihood hooks receive provenance-free `RuntimePerceptView` values while full lineage remains attached to audit artifacts.

**Tech Stack:** Python standard library (`dataclasses`, `inspect`, `MappingProxyType`, `unittest`), existing `stable_content_hash`, `measure_implementation`, `posterior_distribution`, authored replay/uncertain-belief records, World Transition V1, and Observation Projection V1.

**Spec:** `docs/superpowers/specs/2026-08-25-narrative-runtime-percept-admission-v1-design.md`

## Global Constraints

- Exact base: `2a45278bfaee75c69cdafc8c7abd9516cd41c1c7` on `proof/narrative-dynamics-v0`.
- Work only on `work/narrative-runtime-percept-admission-v1`.
- Deterministic authored seed is exactly `epistemic_state(story, domain, agent_id, at_time=source_at_time)`.
- Authored uncertain seed is exactly `uncertain_epistemic_state(story, domain, agent_id, seed_model, tracked_cells, at_time=source_at_time)`.
- Authored `logical_time` and runtime `step_index` remain distinct; no conversion exists.
- Admission accepts no naked `ProjectedObservation`; it must execute `project_world_observations()` internally for the current world step.
- Every admitted world step appends exactly one `RuntimeEvidenceBatch`, including blackout steps with empty evidence.
- Runtime cognition accepts no `WorldState`, `WorldStepResult`, objective mapping, or projection model object.
- Runtime likelihood hooks receive `RuntimePerceptView`, never provenance-bearing `RuntimeEpistemicEvidence`.
- Hidden provenance changes may change audit artifact hashes but must not change deterministic cell semantics, likelihood inputs, or posterior semantics.
- The simulation-wide ledger is filtered by `observer_id` before cognition; one agent never receives another agent's evidence in V1.
- Runtime V1 supports only percept relations `equals|clear`; no runtime testimony, `not_equals`, attention, forgetting, noise, scheduler, action selection, planning, learning, or RNG.
- Existing authored `Observation`, `EpistemicEvidence`, `EpistemicState`, `UncertainBeliefState`, `UncertainBeliefModelSpec`, `run_intentional_decision`, World Transition, Observation Projection, GenericNarrative IR, DomainSpec, root exports, and Lean sources remain unchanged.
- Final diff must contain exactly eight paths: approved spec, this plan, two sidecar modules, narrative `__init__.py`, two dedicated test modules, and trust API test.
- Follow complete test-only RED -> exact-head CI RED -> staged GREEN commits -> export-only RED -> exact final CI.
- Never merge to `proof/narrative-dynamics-v0` without explicit user instruction.

## File Map

- `docs/superpowers/specs/2026-08-25-narrative-runtime-percept-admission-v1-design.md`: approved architecture.
- `docs/superpowers/plans/2026-08-25-narrative-runtime-percept-admission-v1.md`: this plan.
- `narrative_dynamics/narrative/runtime_perception.py`: runtime percept records, empty ledger, admission execution, continuity, blackout, and provenance.
- `narrative_dynamics/narrative/runtime_cognition.py`: deterministic runtime epistemic replay, runtime belief model/records, and runtime Bayesian replay.
- `narrative_dynamics/narrative/__init__.py`: final 18-name narrative-scoped export only.
- `tests/test_narrative_runtime_perception.py`: runtime admission/ledger contract.
- `tests/test_narrative_runtime_cognition.py`: authored seed inheritance, no-leakage, deterministic replay, runtime belief contract.
- `tests/test_narrative_trust_api.py`: exact public API/root-isolation lock.

---

### Task 1: Commit the complete test-only RED contract

**Files:**
- Create: `tests/test_narrative_runtime_perception.py`
- Create: `tests/test_narrative_runtime_cognition.py`
- Modify: `tests/test_narrative_trust_api.py`

**Interfaces:**
- Consumes: merged World Transition V1, Observation Projection V1, authored `epistemic_state`, authored `uncertain_epistemic_state`, and existing finite belief records.
- Produces: 10 perception test methods + 16 cognition test methods covering all 38 required V1 invariants, plus the exact 18-name public-surface RED.

- [ ] **Step 1: Add guarded runtime-perception imports and reusable fixture hooks**

At the top of `tests/test_narrative_runtime_perception.py`, use:

```python
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
```

Define:

```python
class NarrativeRuntimePerceptionTests(unittest.TestCase):
    def require_perception(self) -> None:
        if _PERCEPTION_IMPORT_ERROR is not None:
            self.fail(
                "narrative runtime perception boundary is missing: "
                f"{_PERCEPTION_IMPORT_ERROR}"
            )
```

Define module-level measurable projection hooks:

```python
class ObserveAgentPhase:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        cell = StateCellRef(EntityRef(observer.id, observer.type_name), "agent.phase")
        value = next_visible.get(cell)
        return () if value is None else (ObservationFact(cell, "equals", value),)


class ObserveServiceAlert:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        cell = StateCellRef(EntityRef("svc", "Service"), "service.alert")
        if cell in next_visible:
            return (ObservationFact(cell, "equals", next_visible[cell]),)
        return (ObservationFact(cell, "clear", None),)


class NoPercepts:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        return ()


class RecordingPhaseHook(ObserveAgentPhase):
    def __init__(self):
        self.calls = []

    def __call__(self, prior_visible, next_visible, observer, step_index):
        self.calls.append((dict(prior_visible), dict(next_visible), observer.id, step_index))
        return super().__call__(prior_visible, next_visible, observer, step_index)
```

Define local helpers named `make_runtime_domain`, `make_runtime_story`, `make_world_transition_model`, `make_projection_model`, and `advance_runtime_world`. The domain must contain entity types `Agent|Service`, finite enum value types `PhaseState={ready,active,waiting}` and `AlertState={on,off}`, and state variables `agent.phase` / `service.alert`. The story must contain `a1:Agent`, `a2:Agent`, `svc:Service`, authored seed events/observations/claims, and two canonical decisions whose action options are covered by the transition model. `advance_runtime_world(story, domain, prior_state, transition_model, *, phase_value, clear_alert=False)` must construct canonical `ActionIntent` values and call existing `advance_world_step()`; it must never fabricate `WorldStepResult`.

- [ ] **Step 2: Add the 10 perception tests with exact method names**

Create these methods in `NarrativeRuntimePerceptionTests`:

```text
test_runtime_perception_record_constructors_are_canonical_data_not_certification
test_empty_ledger_binds_exact_branch_point_and_initial_world_state
test_first_admission_executes_projection_and_binds_exact_lineage
test_blackout_appends_explicit_empty_batch_and_advances_step
test_consecutive_batches_bind_world_and_prior_ledger_hash_chain
test_duplicate_skipped_and_reordered_admission_reject_typed
test_source_domain_story_cutoff_and_prior_state_mismatch_reject_before_projection
test_evidence_and_ledger_identity_ignore_semantically_irrelevant_input_order
test_immutable_old_ledger_can_seed_two_counterfactual_next_steps
test_projection_failure_is_preserved_as_typed_admission_failure
```

The constructor test must assert malformed text/hash/step/tuple/relation/value fields fail, and that a directly constructed syntactically valid `RuntimeEpistemicEvidence` retains caller-supplied hashes without implying certification. It must also assert:

```python
self.assertEqual(evidence.percept_view.observer_id, evidence.observer_id)
self.assertEqual(evidence.percept_view.channel, evidence.channel)
self.assertEqual(evidence.percept_view.cell, evidence.cell)
self.assertEqual(evidence.percept_view.relation, evidence.relation)
self.assertEqual(evidence.percept_view.value, evidence.value)
self.assertEqual(evidence.percept_view.step_index, evidence.step_index)
self.assertFalse(hasattr(evidence.percept_view, "source_world_state_hash"))
self.assertFalse(hasattr(evidence.percept_view, "source_world_step_hash"))
```

The empty-ledger test must compute the expected initial state through existing `world_state_from_story` and assert exact domain/story/cutoff/hash binding and `current_step_index == 0`.

The first-admission test must assert one batch at step 1, `prior_ledger_hash == prior_ledger.content_hash`, exact prior/next world hashes, exact projection result hash/model hash, and for each projected/runtime pair:

```python
self.assertEqual(item.projected_observation_hash, projected.content_hash)
self.assertEqual(item.projection_result_hash, result.projection_result.content_hash)
self.assertEqual(item.source_world_step_hash, world_step.content_hash)
self.assertEqual(item.source_world_state_hash, world_step.next_state.content_hash)
```

The blackout test must use a valid empty `ObservationProjectionModelSpec` or `NoPercepts` projection and assert `evidence == ()`, one new batch, next ledger step advanced, and next ledger current world hash updated.

The continuity test must admit two real consecutive world steps and assert batch steps `(1, 2)`, exact world-state chaining, and each stored `prior_ledger_hash` equals the content hash of the exact immutable prefix before that batch.

The duplicate/skipped/reordered test must attempt admission against a ledger whose current world/step does not equal the supplied world-step prior and assert `RuntimePerceptAdmissionError`.

For the mismatch pre-projection test use `RecordingPhaseHook`; forge domain/story/cutoff/prior-world mismatches using `dataclasses.replace`, then assert:

```python
with self.assertRaises(RuntimePerceptAdmissionError) as caught:
    admit_world_percepts(story, domain, world_step, model, bad_ledger)
self.assertEqual(hook.calls, [])
self.assertNotEqual(
    str(caught.exception),
    "runtime percept admission execution is unavailable in this stage",
)
```

The order test must build equivalent projection specs whose constructor-permitted input order differs, then assert equal batch evidence order and equal ledger/result hashes within one fixed provenance branch.

The counterfactual-branch test must reuse one immutable ledger as prior for two different valid next world steps and assert the prior ledger remains unchanged while next ledgers diverge.

- [ ] **Step 3: Add guarded runtime-cognition imports and model hooks**

At the top of `tests/test_narrative_runtime_cognition.py`, use:

```python
_COGNITION_IMPORT_ERROR: ImportError | None = None
try:
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
```

Define:

```python
class NarrativeRuntimeCognitionTests(unittest.TestCase):
    def require_cognition(self) -> None:
        if _COGNITION_IMPORT_ERROR is not None:
            self.fail(
                "narrative runtime cognition boundary is missing: "
                f"{_COGNITION_IMPORT_ERROR}"
            )
```

Define a finite authored seed model using existing `UncertainBeliefModelSpec`. Define runtime likelihood hooks:

```python
class SemanticRuntimeLikelihood:
    def __init__(self):
        self.inputs = []

    def __call__(self, agent_id, percept_view, hypotheses, parameters):
        self.inputs.append(percept_view)
        favored = percept_view.value
        return {
            stable_content_hash(value.to_dict()): (
                parameters["match"] if value == favored else parameters["mismatch"]
            )
            for value in hypotheses
        }


class ClearRuntimeLikelihood:
    def __call__(self, agent_id, percept_view, hypotheses, parameters):
        weight = parameters["clear"] if percept_view.relation == "clear" else 1.0
        return {
            stable_content_hash(value.to_dict()): weight
            for value in hypotheses
        }


class InvalidRuntimeLikelihood:
    def __init__(self, raw):
        self.raw = raw

    def __call__(self, agent_id, percept_view, hypotheses, parameters):
        return self.raw
```

Create `make_runtime_belief_model(seed_model, hook=None)` with canonical parameters `{"match": 0.9, "mismatch": 0.1, "clear": 0.5}`.

- [ ] **Step 4: Add the 16 cognition tests with exact method names**

Create:

```text
test_runtime_epistemic_seed_equals_authored_seed_and_keeps_clocks_separate
test_runtime_equals_supersedes_stale_authored_cell_semantics
test_runtime_clear_makes_cell_unknown_and_clears_stale_constraints
test_unperceived_cells_preserve_authored_seed_semantics
test_multi_channel_same_cell_is_deterministic_and_keeps_all_runtime_support
test_runtime_epistemic_state_filters_other_agents_evidence
test_same_semantic_percepts_preserve_cognition_semantics_across_provenance_branches
test_runtime_belief_seed_equals_authored_uncertain_seed_without_runtime_evidence
test_runtime_belief_model_identity_binds_seed_parameters_and_hook
test_runtime_likelihood_receives_only_provenance_free_percept_view
test_hidden_provenance_changes_cannot_change_posterior_semantics
test_invalid_runtime_likelihood_shape_keys_and_probabilities_reject_typed
test_zero_runtime_posterior_mass_rejects_typed
test_runtime_belief_updates_form_exact_chain_from_seed_posterior
test_runtime_clear_uses_existing_hypotheses_and_never_invents_absent
test_blackout_only_steps_preserve_cognition_and_posterior_while_advancing_step
```

For the seed test compare `state.seed_state.to_dict()` exactly with:

```python
authored_seed = epistemic_state(
    story,
    domain,
    agent_id,
    at_time=ledger.source_at_time,
)
self.assertEqual(state.seed_state.to_dict(), authored_seed.to_dict())
```

Assert runtime state has `step_index`, authored seed evidence retains `logical_time`, and no runtime evidence object has a `logical_time` field.

For equals/clear/unperceived tests assert exact cell status, resolved value, constraints, basis, runtime support hashes, and `resolved_values` membership. Include one runtime-observed cell that is absent from authored `seed_state.cells`; assert it appears in the runtime state so the implementation must use `seed cells ∪ runtime-observed cells`.

For multi-channel same-cell use two valid projection channels that observe the same exact post-step value. Assert one runtime cell view, all same-step evidence hashes retained in canonical order, and no conflict status.

For multi-agent isolation assert every `runtime_evidence_history` record in agent `a1` state has `observer_id == "a1"` and no `a2` record appears.

For semantic-equality/provenance-difference construct two self-consistent ledgers whose semantic percept tuples are equal but lineage hashes differ. Compare only cognitive semantic payloads:

```python
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

self.assertEqual(cognition_semantics(left), cognition_semantics(right))
self.assertNotEqual(left.ledger_hash, right.ledger_hash)
```

For the runtime-belief seed test use an empty ledger and assert `posterior == seed_posterior`, `updates == ()`, and the embedded seed belief exactly matches authored `uncertain_epistemic_state(story, domain, agent_id, model.seed_model, tracked_cells, at_time=ledger.source_at_time)`.

For model identity compare same model, changed seed model, changed runtime parameters, and a distinct module-level runtime likelihood hook; only exact same semantics/code should share content hash.

For provenance-free hook visibility inspect every captured `RuntimePerceptView` and assert none exposes any world/projection/transition hash field or full ledger object.

For hidden-provenance posterior equality, replay two provenance-different but semantically equal ledgers with the same runtime model and assert each tracked cell's posterior distributions are equal, while full state `ledger_hash` values differ.

For likelihood validation cover: non-mapping return, missing hypothesis key, extra hypothesis key, bool value, negative value, value greater than 1, NaN, and infinity. Each must raise `RuntimeBeliefResolutionError`.

For zero posterior mass return all-zero likelihoods and assert typed rejection.

For update-chain test admit two semantic percepts for one tracked cell and assert:

```python
self.assertEqual(view.updates[0].prior, view.seed_posterior)
self.assertEqual(view.updates[1].prior, view.updates[0].posterior)
self.assertEqual(view.posterior, view.updates[-1].posterior)
```

For clear assert the hook receives relation `clear`, hypothesis set equals the original finite seed hypothesis set, and no value token named `absent` is added.

For blackout-only steps compare cognition semantics before/after blackout and assert equal semantics/posterior while `step_index` advances.

- [ ] **Step 5: Lock API signatures against objective-world inputs**

Within the cognition test module add assertions using `inspect.signature`:

```python
self.assertEqual(
    tuple(inspect.signature(runtime_epistemic_state).parameters),
    ("story", "domain", "agent_id", "ledger"),
)
self.assertEqual(
    tuple(inspect.signature(runtime_uncertain_belief_state).parameters),
    ("story", "domain", "agent_id", "ledger", "model", "tracked_cells"),
)
```

This is the executable lock for invariant 36.

- [ ] **Step 6: Extend only the expected narrative public API set**

Insert exactly these 18 names into `_EXPECTED_PUBLIC_API` in `tests/test_narrative_trust_api.py`:

```python
    # Runtime percept admission.
    "RuntimeEpistemicEvidence",
    "RuntimePerceptView",
    "RuntimeEvidenceBatch",
    "RuntimeEvidenceLedger",
    "RuntimePerceptAdmissionResult",
    "RuntimePerceptAdmissionError",
    "runtime_evidence_ledger_from_story",
    "admit_world_percepts",
    # Runtime cognition replay.
    "RuntimeEpistemicCellView",
    "RuntimeEpistemicState",
    "RuntimeEpistemicResolutionError",
    "RuntimeBeliefModelSpec",
    "RuntimeBeliefUpdateStep",
    "RuntimeUncertainBeliefCellView",
    "RuntimeUncertainBeliefState",
    "RuntimeBeliefResolutionError",
    "runtime_epistemic_state",
    "runtime_uncertain_belief_state",
```

Do not edit production `narrative_dynamics/narrative/__init__.py` yet.

- [ ] **Step 7: Verify the exact test-only RED and commit**

Run:

```bash
python3 -m py_compile \
  tests/test_narrative_runtime_perception.py \
  tests/test_narrative_runtime_cognition.py \
  tests/test_narrative_trust_api.py
python3 -m unittest \
  tests.test_narrative_runtime_perception \
  tests.test_narrative_runtime_cognition \
  tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation \
  -v
```

Expected: syntax exit 0; all 10 perception methods fail only through `require_perception()` because `runtime_perception.py` is absent; all 16 cognition methods fail only through `require_cognition()` because `runtime_cognition.py` is absent; trust API fails only on the 18 missing names.

Commit exactly the three test paths:

```bash
git add \
  tests/test_narrative_runtime_perception.py \
  tests/test_narrative_runtime_cognition.py \
  tests/test_narrative_trust_api.py
git commit -m "test: define narrative runtime percept admission v1"
```

Open a Draft PR to `proof/narrative-dynamics-v0`. Exact-head `proof` CI must show all Lean gates and the previous 487 Python tests green, with exactly 27 feature-contract failures: 10 perception module gates, 16 cognition module gates, and one public-surface gate. Do not begin Task 2 until this RED is observed.

---

### Task 2: Implement runtime-perception records and exact empty-ledger branch point

**Files:**
- Create: `narrative_dynamics/narrative/runtime_perception.py`

**Interfaces:**
- Consumes: `GenericNarrative`, `DomainSpec`, `StateCellRef`, `TypedValue`, `world_state_from_story`, `ObservationProjectionResult`.
- Produces: all eight runtime-perception public names structurally, complete record identity, complete empty-ledger API, and a typed non-empty admission stage boundary.

- [ ] **Step 1: Add canonical structural helpers and public error**

Use exact helpers:

```python
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_RELATIONS = frozenset({"equals", "clear"})


class RuntimePerceptAdmissionError(ValueError):
    """A runtime world percept could not be admitted safely."""


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _step(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _cell_key(cell: StateCellRef) -> tuple[str, str, str]:
    return (cell.subject.entity_type, cell.subject.entity_id, cell.state_variable)
```

- [ ] **Step 2: Implement `RuntimePerceptView` and `RuntimeEpistemicEvidence`**

`RuntimePerceptView` is exactly:

```python
@dataclass(frozen=True)
class RuntimePerceptView:
    observer_id: str
    channel: str
    cell: StateCellRef
    relation: str
    value: TypedValue | None
    step_index: int
```

Validate exact relation/value semantics. Its `to_dict()` contains no provenance fields.

`RuntimeEpistemicEvidence` uses the exact fields from the spec. Add:

```python
@property
def percept_view(self) -> RuntimePerceptView:
    return RuntimePerceptView(
        self.observer_id,
        self.channel,
        self.cell,
        self.relation,
        self.value,
        self.step_index,
    )
```

Canonicalize transition hashes lexically and expose `to_dict()` / `content_hash`.

- [ ] **Step 3: Implement batch, ledger, and admission-result constructors**

Define batch evidence ordering:

```python
def _evidence_key(item: RuntimeEpistemicEvidence):
    return (item.observer_id, item.channel, _cell_key(item.cell))
```

Reject duplicate semantic evidence keys within one batch. Validate every evidence record binds the batch's step, next-world hash, world-step hash, projection model hash, and projection result hash.

For the ledger, define:

```python
@property
def current_step_index(self) -> int:
    return 0 if not self.batches else self.batches[-1].step_index
```

Validate first/later step continuity, prior/next world hash chain, current-world hash, and each batch prior-ledger prefix hash by constructing the exact prefix payload through private `_ledger_payload(domain_id, domain_version, domain_spec_hash, source_story_hash, source_at_time, initial_world_state_hash, current_world_state_hash, batches)` and hashing that canonical payload. The prefix's `current_world_state_hash` is the initial hash for zero batches and the previous batch's next-world hash otherwise.

`RuntimePerceptAdmissionResult` validates that its batch is the last batch in `next_ledger`, that recomputing the prefix before that last batch yields `prior_ledger_hash`, and that projection/batch source hashes match exactly.

- [ ] **Step 4: Implement `runtime_evidence_ledger_from_story` completely**

Use:

```python
def runtime_evidence_ledger_from_story(
    story: GenericNarrative,
    domain: DomainSpec,
    *,
    at_time: int | None = None,
) -> RuntimeEvidenceLedger:
    validate_narrative(story, domain)
    initial = world_state_from_story(story, domain, at_time=at_time)
    return RuntimeEvidenceLedger(
        domain_id=domain.domain_id,
        domain_version=domain.version,
        domain_spec_hash=domain.content_hash,
        source_story_hash=story.content_hash,
        source_at_time=at_time,
        initial_world_state_hash=initial.content_hash,
        current_world_state_hash=initial.content_hash,
        batches=(),
    )
```

Validate `at_time` consistently with existing authored/world APIs; do not synthesize logical time.

- [ ] **Step 5: Add the exact typed stage boundary for admission**

Expose:

```python
def admit_world_percepts(
    story: GenericNarrative,
    domain: DomainSpec,
    world_step: WorldStepResult,
    projection_model: ObservationProjectionModelSpec,
    prior_ledger: RuntimeEvidenceLedger,
) -> RuntimePerceptAdmissionResult:
    raise RuntimePerceptAdmissionError(
        "runtime percept admission execution is unavailable in this stage"
    )
```

- [ ] **Step 6: Verify staged GREEN/RED and commit**

Run only:

```bash
python3 -m unittest \
  tests.test_narrative_runtime_perception.NarrativeRuntimePerceptionTests.test_runtime_perception_record_constructors_are_canonical_data_not_certification \
  tests.test_narrative_runtime_perception.NarrativeRuntimePerceptionTests.test_empty_ledger_binds_exact_branch_point_and_initial_world_state \
  -v
```

Expected: both pass.

Run the full perception module; the remaining eight admission methods must fail only at the exact stage-boundary message. Cognition remains module-missing RED. Public API remains RED.

Commit:

```bash
git add narrative_dynamics/narrative/runtime_perception.py
git commit -m "feat: add runtime percept ledger records"
```

Obtain exact-head CI evidence before Task 3.

---

### Task 3: Implement current-step admission, blackout, and immutable ledger continuity

**Files:**
- Modify: `narrative_dynamics/narrative/runtime_perception.py`

**Interfaces:**
- Consumes: Task 2 records, `project_world_observations`, real `WorldStepResult`, real projection model.
- Produces: complete `admit_world_percepts()` semantics; all 10 runtime-perception methods GREEN.

- [ ] **Step 1: Validate ledger/world branch identity before projection execution**

Implement `_validate_admission_source` with these exact checks before projection:

```python
validate_narrative(story, domain)
if not isinstance(prior_ledger, RuntimeEvidenceLedger):
    raise TypeError("runtime percept admission requires RuntimeEvidenceLedger")
if prior_ledger.domain_id != domain.domain_id:
    raise ValueError("runtime ledger domain id mismatch")
if prior_ledger.domain_version != domain.version:
    raise ValueError("runtime ledger domain version mismatch")
if prior_ledger.domain_spec_hash != domain.content_hash:
    raise ValueError("runtime ledger domain spec hash mismatch")
if prior_ledger.source_story_hash != story.content_hash:
    raise ValueError("runtime ledger source story mismatch")
if prior_ledger.current_world_state_hash != world_step.prior_state.content_hash:
    raise ValueError("runtime ledger current world state does not match world-step prior")
if prior_ledger.current_step_index != world_step.prior_state.step_index:
    raise ValueError("runtime ledger step does not match world-step prior")
if prior_ledger.source_at_time != world_step.prior_state.source_at_time:
    raise ValueError("runtime ledger source cutoff does not match world-step prior")
if world_step.next_state.source_at_time != prior_ledger.source_at_time:
    raise ValueError("runtime ledger source cutoff does not match world-step next state")
if world_step.next_state.step_index != prior_ledger.current_step_index + 1:
    raise ValueError("runtime admission world step must advance exactly one step")
```

Any mismatch must occur before projection hook execution.

- [ ] **Step 2: Execute Observation Projection internally and derive evidence only from its result**

Call exactly:

```python
projection_result = project_world_observations(
    story,
    domain,
    world_step,
    projection_model,
)
```

For each returned `ProjectedObservation`, construct `RuntimeEpistemicEvidence` from its semantic/provenance fields. `projected_observation_hash` is the exact observation `content_hash`; `projection_result_hash` is the exact result `content_hash`; `projection_model_hash` is `projection_result.model_hash`.

No public helper may accept a caller-supplied projected observation as admission input.

- [ ] **Step 3: Construct one batch including blackout**

Create:

```python
batch = RuntimeEvidenceBatch(
    prior_ledger_hash=prior_ledger.content_hash,
    step_index=world_step.next_state.step_index,
    source_prior_world_state_hash=world_step.prior_state.content_hash,
    source_world_state_hash=world_step.next_state.content_hash,
    source_world_step_hash=world_step.content_hash,
    projection_model_hash=projection_result.model_hash,
    projection_result_hash=projection_result.content_hash,
    evidence=tuple(runtime_evidence),
)
```

When projection has no observations, `runtime_evidence` is empty and the batch still exists.

- [ ] **Step 4: Append immutably using the exact ledger-prefix hash chain**

Construct:

```python
next_ledger = RuntimeEvidenceLedger(
    domain_id=prior_ledger.domain_id,
    domain_version=prior_ledger.domain_version,
    domain_spec_hash=prior_ledger.domain_spec_hash,
    source_story_hash=prior_ledger.source_story_hash,
    source_at_time=prior_ledger.source_at_time,
    initial_world_state_hash=prior_ledger.initial_world_state_hash,
    current_world_state_hash=world_step.next_state.content_hash,
    batches=prior_ledger.batches + (batch,),
)
```

Return `RuntimePerceptAdmissionResult(prior_ledger.content_hash, projection_result, batch, next_ledger)`.

- [ ] **Step 5: Wrap operational errors without erasing projection causes**

Re-raise existing `RuntimePerceptAdmissionError`. Catch `ObservationProjectionError`, `TypeError`, and `ValueError`, and wrap as `RuntimePerceptAdmissionError` with exception chaining. Pre-projection continuity errors must remain distinguishable from projection failure by their chained cause/message.

- [ ] **Step 6: Verify all perception semantics GREEN and commit**

Run:

```bash
python3 -m unittest tests.test_narrative_runtime_perception -v
python3 -m unittest \
  tests.test_narrative_observation_projection \
  tests.test_narrative_world_transition \
  -v
```

Expected: all pass.

Commit:

```bash
git add narrative_dynamics/narrative/runtime_perception.py
git commit -m "feat: admit runtime world percepts"
```

Exact-head CI should now have all 10 perception tests GREEN, 16 cognition module-gate failures, and the one exact API gate failure; all pre-existing tests/Lean gates remain green.

---

### Task 4: Implement deterministic runtime epistemic replay and runtime belief record/model boundary

**Files:**
- Create: `narrative_dynamics/narrative/runtime_cognition.py`

**Interfaces:**
- Consumes: `RuntimeEvidenceLedger`, `RuntimeEpistemicEvidence`, `RuntimePerceptView`, authored `epistemic_state`, authored uncertain-belief records/model.
- Produces: deterministic runtime epistemic state semantics plus all runtime belief public records/model identity. `runtime_uncertain_belief_state()` remains an explicit typed stage boundary for Bayesian execution.

- [ ] **Step 1: Add public runtime cognition errors and canonical helpers**

Define:

```python
class RuntimeEpistemicResolutionError(ValueError):
    """Runtime direct-perception evidence could not produce a valid epistemic state."""


class RuntimeBeliefResolutionError(ValueError):
    """Runtime percept evidence could not produce a valid uncertain belief state."""
```

Reuse canonical text/hash/probability/freeze logic locally rather than importing private helpers from `uncertain.py`. Reuse public `BeliefDistribution`, `BeliefLikelihood`, `BeliefMass`, and `UncertainBeliefModelSpec` types.

- [ ] **Step 2: Implement `RuntimeEpistemicCellView` and `RuntimeEpistemicState`**

Use exact fields from the spec. `basis` is exactly `authored_seed|runtime_perception`.

`RuntimeEpistemicState` stores:

```python
agent_id: str
source_at_time: int | None
step_index: int
ledger_hash: str
seed_state: EpistemicState
runtime_evidence_history: tuple[RuntimeEpistemicEvidence, ...]
cells: Mapping[StateCellRef, RuntimeEpistemicCellView]
resolved_values: Mapping[StateCellRef, TypedValue]
```

Freeze mappings with `MappingProxyType`; canonicalize runtime history by `(step_index, channel, cell)`.

- [ ] **Step 3: Implement `runtime_epistemic_state` completely**

Expose exactly:

```python
def runtime_epistemic_state(
    story: GenericNarrative,
    domain: DomainSpec,
    agent_id: str,
    ledger: RuntimeEvidenceLedger,
) -> RuntimeEpistemicState:
```

Validate story/domain/agent and exact ledger source identity. Recompute:

```python
seed = epistemic_state(
    story,
    domain,
    agent_id,
    at_time=ledger.source_at_time,
)
```

Filter runtime evidence to `observer_id == agent_id`. Build:

```python
runtime_cells = {item.cell for item in runtime_history}
all_cells = set(seed.cells) | runtime_cells
```

For each cell in canonical `_cell_key` order, group runtime evidence by step and apply:

- no runtime evidence => copy seed view status/value/constraints, `basis="authored_seed"`, no runtime support hashes;
- runtime evidence for a seed-absent cell => create a new runtime cell view from the latest runtime percept;
- latest runtime `equals` => resolved to the one truth-compatible value, empty stale constraints, `basis="runtime_perception"`;
- latest runtime `clear` => unknown with no resolved value and empty constraints;
- multiple same-step channels for one cell must agree on relation/value semantics or reject as corrupted ledger input; retain all supporting evidence hashes sorted.

The function must not accept any world/projection argument.

- [ ] **Step 4: Implement `RuntimeBeliefModelSpec` and runtime belief record constructors**

Define:

```python
@dataclass(frozen=True)
class RuntimeBeliefModelSpec:
    model_id: str
    version: str
    seed_model: UncertainBeliefModelSpec
    runtime_parameters: Mapping[str, object]
    runtime_likelihood_hook: object = field(compare=False, repr=False)
```

Freeze parameters canonically and bind `seed_model.content_hash` plus `measure_implementation(runtime_likelihood_hook).manifest_identity()` in `to_dict()` / `content_hash`.

Implement `RuntimeBeliefUpdateStep`, `RuntimeUncertainBeliefCellView`, and `RuntimeUncertainBeliefState` with the exact chain invariants from the spec. `RuntimeBeliefUpdateStep` stores full `RuntimeEpistemicEvidence` for audit, but later execution passes only `evidence.percept_view` to the hook.

- [ ] **Step 5: Add exact stage boundary for `runtime_uncertain_belief_state`**

Expose only:

```python
def runtime_uncertain_belief_state(
    story: GenericNarrative,
    domain: DomainSpec,
    agent_id: str,
    ledger: RuntimeEvidenceLedger,
    model: RuntimeBeliefModelSpec,
    tracked_cells: tuple[StateCellRef, ...],
) -> RuntimeUncertainBeliefState:
    raise RuntimeBeliefResolutionError(
        "runtime uncertain belief execution is unavailable in this stage"
    )
```

- [ ] **Step 6: Verify exact Task 4 staged GREEN/RED and commit**

These cognition methods must pass:

```text
test_runtime_epistemic_seed_equals_authored_seed_and_keeps_clocks_separate
test_runtime_equals_supersedes_stale_authored_cell_semantics
test_runtime_clear_makes_cell_unknown_and_clears_stale_constraints
test_unperceived_cells_preserve_authored_seed_semantics
test_multi_channel_same_cell_is_deterministic_and_keeps_all_runtime_support
test_runtime_epistemic_state_filters_other_agents_evidence
test_same_semantic_percepts_preserve_cognition_semantics_across_provenance_branches
test_runtime_belief_model_identity_binds_seed_parameters_and_hook
```

These Bayesian execution methods must remain RED only at `"runtime uncertain belief execution is unavailable in this stage"`:

```text
test_runtime_belief_seed_equals_authored_uncertain_seed_without_runtime_evidence
test_runtime_likelihood_receives_only_provenance_free_percept_view
test_hidden_provenance_changes_cannot_change_posterior_semantics
test_invalid_runtime_likelihood_shape_keys_and_probabilities_reject_typed
test_zero_runtime_posterior_mass_rejects_typed
test_runtime_belief_updates_form_exact_chain_from_seed_posterior
test_runtime_clear_uses_existing_hypotheses_and_never_invents_absent
test_blackout_only_steps_preserve_cognition_and_posterior_while_advancing_step
```

Commit:

```bash
git add narrative_dynamics/narrative/runtime_cognition.py
git commit -m "feat: replay runtime epistemic cognition"
```

Exact-head CI must show 10 perception + 8 deterministic/model cognition tests GREEN, 8 Bayesian staged RED, and one public API RED. Obtain this evidence before Task 5.

---

### Task 5: Implement runtime finite Bayesian replay with provenance-free hook inputs

**Files:**
- Modify: `narrative_dynamics/narrative/runtime_cognition.py`

**Interfaces:**
- Consumes: Task 4 runtime belief model/records, existing authored `uncertain_epistemic_state`, existing finite belief records, existing `posterior_distribution`.
- Produces: complete `runtime_uncertain_belief_state()` semantics; all 16 cognition methods GREEN.

- [ ] **Step 1: Validate tracked cells and reconstruct authored uncertain seed**

Validate `tracked_cells` is a non-empty tuple of unique `StateCellRef` values. Validate story/domain/agent/ledger/model before any runtime likelihood hook call.

Compute:

```python
seed = uncertain_epistemic_state(
    story,
    domain,
    agent_id,
    model.seed_model,
    tracked_cells,
    at_time=ledger.source_at_time,
)
```

This is the only source of the initial posterior for each tracked cell.

- [ ] **Step 2: Filter and canonicalize runtime evidence semantically**

Select only evidence satisfying:

```python
tracked = set(tracked_cells)
runtime_history = tuple(
    item
    for batch in ledger.batches
    for item in batch.evidence
    if item.observer_id == agent_id and item.cell in tracked
)
```

Sort by `(step_index, channel, cell)` and never use provenance hashes to choose semantic order.

- [ ] **Step 3: Call runtime likelihood hook with `RuntimePerceptView` only**

Implement:

```python
def _call_runtime_likelihood_hook(
    model: RuntimeBeliefModelSpec,
    agent_id: str,
    evidence: RuntimeEpistemicEvidence,
    hypotheses: tuple[TypedValue, ...],
) -> object:
    try:
        return model.runtime_likelihood_hook(
            agent_id,
            evidence.percept_view,
            hypotheses,
            model.runtime_parameters,
        )
    except (TypeError, ValueError, KeyError) as error:
        raise RuntimeBeliefResolutionError(
            "runtime likelihood hook could not produce a valid vector"
        ) from error
```

Do not pass ledger hash, evidence hash, world hash, projection hash, transition hash, or any objective state object.

- [ ] **Step 4: Validate exact likelihood vectors and compute canonical posterior updates**

Require a mapping keyed by the exact stable-content hashes of current hypotheses; reject missing/extra keys. Validate every likelihood is numeric, finite, non-bool, and in `[0,1]`.

Use:

```python
posterior_vector = posterior_distribution(
    current_prior_vector,
    likelihood_vector,
)
```

Wrap zero-mass failures as `RuntimeBeliefResolutionError`.

Build `BeliefLikelihood` records, one `RuntimeBeliefUpdateStep`, and advance current distribution. For runtime `clear`, do not alter the finite hypothesis set; the hook receives relation `clear` and decides the likelihood vector over existing hypotheses.

- [ ] **Step 5: Build exact runtime uncertain cell/state artifacts**

For each tracked cell, set:

```python
RuntimeUncertainBeliefCellView(
    cell=cell,
    seed_posterior=seed.cells[cell].posterior,
    posterior=current,
    updates=tuple(updates),
)
```

Return `RuntimeUncertainBeliefState` with exact model hash, source cutoff, ledger current step, ledger hash, full seed belief state, per-agent filtered runtime evidence history, and canonical cell mapping.

With no runtime evidence for a cell, posterior must equal seed posterior and updates must be empty. Blackout-only batches change state `step_index`/ledger hash but not posterior semantics.

- [ ] **Step 6: Verify all cognition semantics GREEN and no-leakage explicitly**

Run:

```bash
python3 -m unittest tests.test_narrative_runtime_cognition -v
python3 -m unittest \
  tests.test_narrative_replay \
  tests.test_narrative_uncertain_belief \
  tests.test_narrative_intention \
  tests.test_narrative_observation_projection \
  tests.test_narrative_runtime_perception \
  -v
```

Expected: all pass.

- [ ] **Step 7: Commit semantic GREEN and obtain export-only RED CI evidence**

```bash
git add narrative_dynamics/narrative/runtime_cognition.py
git commit -m "feat: update runtime belief from admitted percepts"
```

On exact-head CI require all 10 perception + 16 cognition tests GREEN. The only feature-related failure must be `test_exact_public_surface_and_root_isolation` because the 18 exports are still absent. All prior 487 tests and Lean gates remain green.

---

### Task 6: Export the exact 18-name API and close full regression

**Files:**
- Modify: `narrative_dynamics/narrative/__init__.py`

**Interfaces:**
- Consumes: complete Task 3 + Task 5 sidecar public APIs.
- Produces: exactly 18 new narrative-scoped names; root isolation preserved.

- [ ] **Step 1: Add only runtime-perception scoped imports**

Add:

```python
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
```

- [ ] **Step 2: Add only runtime-cognition scoped imports**

Add:

```python
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
```

Add exactly the 18 approved strings to narrative `__all__`; do not edit root `narrative_dynamics/__init__.py`.

- [ ] **Step 3: Run exact API, dedicated semantics, then full Python discovery**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation \
  -v
python3 -m unittest \
  tests.test_narrative_runtime_perception \
  tests.test_narrative_runtime_cognition \
  -v
python3 -m unittest discover -s tests -v
```

Expected: API/root-isolation passes; all 26 new methods pass; full suite exits 0. Baseline before this feature is 487 tests, so unchanged discovery plus 26 methods must report exactly **513 tests**. Any other count requires explicit investigation before integration.

- [ ] **Step 4: Commit export-only change**

```bash
git add narrative_dynamics/narrative/__init__.py
git commit -m "feat: export narrative runtime cognition api"
```

The commit must touch only narrative `__init__.py`.

- [ ] **Step 5: Run and inspect final exact-head `proof` CI**

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

Fetch the full job log and confirm the unittest summary line begins with `Ran 513 tests in`, followed by `OK`. Also confirm every `NarrativeRuntimePerceptionTests`, every `NarrativeRuntimeCognitionTests`, and `test_exact_public_surface_and_root_isolation` is `ok`.

- [ ] **Step 6: Verify exact final eight-path diff**

Compare base `2a45278bfaee75c69cdafc8c7abd9516cd41c1c7` to final feature head. Changed files must be exactly:

```text
docs/superpowers/specs/2026-08-25-narrative-runtime-percept-admission-v1-design.md
docs/superpowers/plans/2026-08-25-narrative-runtime-percept-admission-v1.md
narrative_dynamics/narrative/runtime_perception.py
narrative_dynamics/narrative/runtime_cognition.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_runtime_perception.py
tests/test_narrative_runtime_cognition.py
tests/test_narrative_trust_api.py
```

Any additional path stops integration.

- [ ] **Step 7: Update Draft PR evidence and mark Ready for review**

Record exact base, test-only RED head/run, Task 2 staged head/run, Task 3 perception-GREEN/cognition-RED head/run, Task 4 deterministic-cognition staged head/run, Task 5 semantic head/run proving export-only RED, and final head/run. State explicitly that authored replay/uncertain/intention, GenericNarrative IR, DomainSpec, World Transition, Observation Projection, root package, movie fixtures, model-comparison infrastructure, and Lean sources are unchanged. Mark Ready for review; do not merge.

## Spec Coverage Map

- Invariants 1-2: empty-ledger test.
- Invariants 3, 12-13: first-admission lineage test.
- Invariants 4-6: consecutive-batch hash-chain test.
- Invariants 7-8: blackout admission + blackout cognition tests.
- Invariants 9-10: duplicate/skipped/reordered admission test.
- Invariant 11: pre-projection mismatch test.
- Invariants 14-15: order/canonical-record constructor tests.
- Invariants 16, 18-19: runtime epistemic seed/clock test plus unchanged regression suite.
- Invariants 20-23: equals/clear/unperceived/multi-channel deterministic tests.
- Invariant 24: multi-agent isolation test.
- Invariants 25-26: semantic-equality/provenance-difference test.
- Invariants 17, 34: runtime uncertain seed/no-runtime-evidence test.
- Invariants 27-29: runtime model identity, provenance-free hook, hidden-provenance posterior tests.
- Invariants 30-31: invalid likelihood + zero-posterior tests.
- Invariant 32: exact runtime posterior-chain test.
- Invariant 33: clear-with-existing-hypotheses test.
- Invariant 35: blackout-only runtime cognition/posterior test.
- Invariant 36: exact public function-signature assertions.
- Invariant 37: trust API/root isolation test.
- Invariant 38: full Python + Lean exact-head CI.

## Completion Evidence Checklist

- [ ] All 10 runtime-perception methods pass on exact final head.
- [ ] All 16 runtime-cognition methods pass on exact final head.
- [ ] Exact 18-name narrative API/root-isolation test passes.
- [ ] Full Python discovery reports exactly 513 tests and zero failures.
- [ ] Lean/Python conformance passes.
- [ ] Full Lean build passes.
- [ ] Lean theorem suite passes.
- [ ] StoryState theorem gate passes.
- [ ] Testimony theorem gate passes.
- [ ] Runtime likelihood hook inputs contain no provenance lineage.
- [ ] Equal semantic percept sequences produce equal cognition/posterior semantics across provenance-different branches.
- [ ] Blackout batches advance runtime step metadata without changing cognition semantics.
- [ ] Final diff contains exactly eight approved paths.
- [ ] PR head equals exact CI-tested head.
- [ ] PR remains unmerged until explicit user instruction.
