# Narrative Stochastic World / Observation V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit, seeded, replayable aleatoric stochastic world transitions and truth-preserving stochastic observation projection while preserving deterministic V1/V2 payloads, hashes, decision APIs, conflict semantics, and uncertainty/comparison infrastructure.

**Architecture:** Add one runtime-owned stateless hash-categorical randomness core. World and observation models gain separate additive stochastic lanes whose hooks return finite distributions without receiving RNG; the trusted runtime validates every candidate, derives component-local substreams from a frozen narrative root seed, samples one outcome, and binds complete sample lineage into world/projection/evidence/simulation hashes. `simulate_step` remains seedless after initialization, decision dispatch remains RNG-free, and deterministic records omit every new optional field.

**Tech Stack:** Python 3 stdlib (`dataclasses`, `math`, `re`, `MappingProxyType`, `unittest`), existing `stable_content_hash`, implementation attestation, GenericNarrative/DomainSpec world transition, Conflict Resolution V2, observation projection, runtime percept ledger, multi-step simulation scheduler, outer `SimulationRunner`, existing uncertainty seed-block diagnostics, GitHub Actions `proof.yml`.

**Spec:** `docs/superpowers/specs/2026-08-28-narrative-stochastic-world-observation-v1-design.md`

## Global Constraints

- Integrated base: `proof/narrative-dynamics-v0@e2b505230211a38390e8eda9e92ba25c347703aa`.
- Feature branch: `work/narrative-stochastic-world-observation-v1`.
- Approved written-spec head before planning: `93904c6d5eee05a6bd69011c701f83c1a376cbc8`.
- Docs-only spec/plan commits do not count as feature proof. The first authoritative feature proof is the complete test-only RED commit.
- Strict sequence: test-only RED -> exact-head RED evidence -> minimal GREEN tasks -> exact-head feature GREEN -> PR synthetic-merge GREEN -> merge -> post-merge exact-head GREEN.
- No production code is committed in the authoritative RED commit.
- Production scope is one new module, `narrative_dynamics/narrative/randomness.py`, plus modifications only to `world.py`, `observation_projection.py`, `runtime_perception.py`, and narrative `simulation.py`.
- Do not modify decision-family implementations, runtime decision dispatch, conflict resolver semantics, comparison/preregistration/release code, uncertainty/calibration core, committed fixtures, package-root exports, or Lean sources.
- Existing deterministic hook signatures stay exact: `transition_hook(snapshot, decision, action)` and `projection_hook(prior_visible, next_visible, observer, step_index)`.
- Reactive/Intentional/Planning and `run_runtime_decision` remain RNG-free.
- Conflict Resolution V2 remains deterministic and receives concrete sampled transition records.
- Observation stochasticity is truth-preserving: visibility/dropout/selection among true facts only; no false `equals` values.
- `RANDOM_DERIVATION_VERSION` is exactly `narrative-hash-categorical-v1`.
- V1 random namespaces are exactly `world.transition` and `observation.projection`.
- World sample step is `prior_state.step_index + 1`; observation sample step is `world_step.next_state.step_index`.
- World sample source is exact prior `WorldState.content_hash`; observation sample source is exact `WorldStepResult.content_hash`.
- World component key is `(actor_id, decision_id, action_id)`; observation component key is `(observer_id, channel)`.
- No process-global RNG, trajectory-global mutable RNG, or sibling-consumed RNG state.
- Every finite distribution has at least two outcomes, unique trimmed ids, positive finite non-bool probabilities, lexical id canonicalization, and exact `math.fsum(probabilities) == 1.0`. No renormalization.
- Every candidate world delta is capability-validated before sampling. Every candidate observation tuple is shape/capability/truth-validated before sampling.
- Empty stochastic observation outcomes are valid and still produce projection/evidence-batch sample lineage.
- `WorldState.root_seed` is optional and appended after existing fields; its serialized key is absent when `None`; next states preserve it exactly.
- A selected stochastic world transition or executing stochastic observation projection requires a non-`None` root seed before calling its hook.
- `simulation_state_from_story(..., seed=None)` is the only narrative scheduler entry that accepts the root seed. `simulate_step` and `simulate_trajectory` gain no seed parameter.
- Deterministic V1/V2 payloads and hashes remain exact by omitting all new empty/`None` fields.
- The observation independence invariant is evaluated on the same exact `WorldStepResult`: adding/reordering an unrelated stochastic projection component must not perturb an existing observer/channel sample. Changing the story/entity set changes the source hash and is not an invariance claim.
- Outer `SimulationRunner` compatibility is test-adapter-only: exactly one `rng.getrandbits(64)` call deterministically maps its already-recorded seed to the narrative root seed.
- Do not mark #27 P2 Stochastic World / Observation complete before merge plus post-merge exact-head proof. P2 Identification and Model Comparison remains open.

## File Structure

### Production

- `narrative_dynamics/narrative/randomness.py` — seed validation, namespace validation, stream derivation, deterministic 64-bit draw, categorical selection, `RandomSampleRecord`.
- `narrative_dynamics/narrative/world.py` — stochastic transition distributions/specs/samples, optional root seed, sampled transition lane, realized-write conflict integration.
- `narrative_dynamics/narrative/observation_projection.py` — stochastic observation distributions/specs/samples, truth-preserving prevalidation, dropout lineage.
- `narrative_dynamics/narrative/runtime_perception.py` — projection sample hashes copied into evidence and complete sample hashes retained in each batch.
- `narrative_dynamics/narrative/simulation.py` — root-seed initialization only; step/trajectory APIs stay seedless.

### Tests

- `tests/test_narrative_randomness.py`
- `tests/test_narrative_stochastic_world.py`
- `tests/test_narrative_stochastic_observation.py`
- `tests/test_narrative_stochastic_simulation.py`
- `tests/test_narrative_stochastic_seed_diagnostics.py`

---

### Task 1: Complete the Authoritative Test-Only RED Boundary

**Files:**
- Create the five test modules listed above.

**Interfaces:**
- Consumes existing deterministic world/projection/perception/scheduler fixtures and outer `SimulationRunner`/uncertainty APIs.
- Produces the complete P2 acceptance boundary before any stochastic production symbol exists.

- [ ] **Step 1: Create randomness RED tests**

Use guarded imports so discovery runs and failures are attributable to the missing module:

```python
from __future__ import annotations

from dataclasses import replace
import unittest

_RANDOMNESS_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.randomness import (
        RANDOM_DERIVATION_VERSION,
        derive_stream_hash,
        draw_u64_for_stream,
        sample_categorical,
        validate_root_seed,
    )
except ImportError as error:
    _RANDOMNESS_IMPORT_ERROR = error


class NarrativeRandomnessTests(unittest.TestCase):
    def require_randomness(self) -> None:
        if _RANDOMNESS_IMPORT_ERROR is not None:
            self.fail(f"narrative randomness API is missing: {_RANDOMNESS_IMPORT_ERROR}")
```

Add six methods with these exact assertions:

1. `test_seed_validation_and_namespaces_are_closed`: `0` and `-7` are accepted; `True`, `False`, `1.0`, `"1"`, and `None` are rejected; unsupported namespace `decision.choice` is rejected by `derive_stream_hash`.
2. `test_stream_and_draw_are_exact_and_replay`: identical derivation inputs return the same stream and draw, and the version constant equals `narrative-hash-categorical-v1`.
3. `test_component_key_change_changes_stream_identity`: change only `component_key` and require a different stream hash.
4. `test_outcome_input_order_does_not_change_selection`: call `sample_categorical` with `(("drop", 0.5, hash_a), ("emit", 0.5, hash_b))` and its reverse; require equal `to_dict()`.
5. `test_distribution_validation_rejects_invalid_probabilities`: reject a one-outcome distribution, zero probability, non-unit total mass, duplicate ids, bool probability, NaN, and infinity.
6. `test_random_sample_record_forgery_rejects`: create one valid record via `sample_categorical`, then `dataclasses.replace` its `stream_hash`, `draw_u64`, and `namespace`; each replacement must raise.

Use these canonical test inputs for derivation assertions:

```python
COMMON = {
    "root_seed": 42,
    "namespace": "world.transition",
    "step_index": 1,
    "source_hash": "sha256:" + "1" * 64,
    "component_hash": "sha256:" + "2" * 64,
    "component_key": ("a1", "d1", "act"),
}
```

- [ ] **Step 2: Create stochastic world RED tests**

Reuse `make_world_domain`, `make_world_story`, `_agent_cell`, and `intent` from `tests.test_narrative_world_transition`. Guard-import `StateDeltaOutcome`, `StateDeltaDistribution`, `StochasticActionTransitionSpec`, and `StochasticTransitionSample`.

Define this controlled hook:

```python
class StochasticPhaseHook:
    def __init__(self):
        self.calls = []

    def __call__(self, snapshot, decision, action, parameters):
        self.calls.append((dict(snapshot), decision.id, action.id, dict(parameters)))
        p = parameters["ready_probability"]
        return StateDeltaDistribution((
            StateDeltaOutcome(
                "done",
                1.0 - p,
                StateDelta((StateDeltaOp(
                    "set",
                    decision.actor_id,
                    "agent.phase",
                    TypedValue("PhaseState", "done"),
                ),)),
            ),
            StateDeltaOutcome(
                "ready",
                p,
                StateDelta((StateDeltaOp(
                    "set",
                    decision.actor_id,
                    "agent.phase",
                    TypedValue("PhaseState", "ready"),
                ),)),
            ),
        ))
```

Create seven methods:

- `test_deterministic_world_payloads_and_hashes_remain_v1_shaped`: construct the existing deterministic model/state/step, require `root_seed`, `stochastic_transitions`, and `stochastic_sample` keys are absent, rebuild the pre-P2 payload dict from existing fields, and require `stable_content_hash(expected_payload) == object.content_hash`.
- `test_same_seed_replays_stochastic_transition_exactly`: initialize two independent worlds with seed `17`, execute the same stochastic action, require equal transition/sample/next-state/result dictionaries and hashes.
- `test_actor_order_and_unrelated_actor_do_not_perturb_component_sample`: use one unchanged story/prior/model and two already-declared actors; compare bob's sample for `(bob, alice)`, `(alice, bob)`, and `(bob,)`; bob's `StochasticTransitionSample.to_dict()` must match in all three.
- `test_multiple_seeds_exercise_declared_world_outcomes`: scan `range(256)` and require both `done` and `ready` are realized.
- `test_every_candidate_delta_is_validated_before_sampling`: return one valid candidate and one outside-capability candidate with probability `1e-12`; require `WorldTransitionError` and no successful next state.
- `test_stochastic_action_requires_seed_before_hook`: use `seed=None`, require `WorldTransitionError`, and require hook `calls == []`.
- `test_realized_stochastic_writes_feed_existing_conflict_resolution`: create a controlled stochastic service-health action whose realized delta overlaps another concrete action; require the existing resolver sees concrete transition records and produces the same deterministic resolution semantics for that realized conflict.

- [ ] **Step 3: Create stochastic observation RED tests**

Reuse `make_projection_domain`, `make_projection_story`, `_cell`, and deterministic projection helpers from `tests.test_narrative_observation_projection`. Build seeded world steps by calling `world_state_from_story(..., seed=seed)` followed by `advance_world_step`; do not use the existing unseeded `make_world_step` for stochastic projection cases.

Guard-import `ObservationOutcome`, `ObservationOutcomeDistribution`, `StochasticObservationSample`, and `StochasticObserverProjectionSpec`.

Define:

```python
class StochasticOwnLocationHook:
    def __init__(self):
        self.calls = []

    def __call__(self, prior_visible, next_visible, observer, step_index, parameters):
        self.calls.append((observer.id, step_index, dict(parameters)))
        cell = _cell(observer.id, "Agent", "agent.location")
        fact = ObservationFact(cell, "equals", next_visible[cell])
        p = parameters["emit_probability"]
        return ObservationOutcomeDistribution((
            ObservationOutcome("drop", 1.0 - p, ()),
            ObservationOutcome("emit", p, (fact,)),
        ))
```

Create seven methods:

- `test_deterministic_projection_payloads_and_hashes_remain_v1_shaped`: existing deterministic model/result/observation emit no stochastic keys and hash their exact pre-P2-shaped payloads.
- `test_same_seed_and_world_step_replay_projection_exactly`: project the same seeded world step twice and require equal samples/observations/result hashes.
- `test_spec_order_and_unrelated_projection_do_not_perturb_existing_sample`: keep one exact `WorldStepResult`; compare `(a1, vision)` under a model containing that spec alone and a model also containing an unrelated different-channel stochastic spec; existing sample stays equal. Reverse spec input order and require the same result semantics.
- `test_seed_scan_exercises_emit_and_dropout`: scan `range(256)` until both selected outcome ids appear.
- `test_empty_outcome_keeps_projection_and_evidence_batch_lineage`: find a dropout seed, require zero projected observations, one stochastic projection sample, then after admission require the sample hash in `RuntimeEvidenceBatch.projection_sample_hashes` even though `evidence == ()` for that spec.
- `test_every_candidate_fact_is_validated_before_sampling`: include one valid outcome and one false `equals` fact with probability `1e-12`; require `ObservationProjectionError` before successful projection.
- `test_stochastic_projection_requires_seed_before_hook`: unseeded world step plus stochastic projection must fail with hook `calls == []`.

- [ ] **Step 4: Create stochastic scheduler RED tests**

Reuse `make_scheduler_domain`, `make_scheduler_story`, `make_intentional_models`, `RuntimeAgentSpec`, and existing scheduler decision fixtures from `tests.test_narrative_simulation`. Build one simulation model with a stochastic alert-control world transition, a deterministic response transition, and a stochastic alert projection for agent `a2`.

Create six methods:

- `test_seeded_initialization_binds_world_and_ledger_to_same_initial_hash`: `simulation_state_from_story(..., seed=101)` yields `world_state.root_seed == 101` and `evidence_ledger.initial_world_state_hash == world_state.content_hash`.
- `test_same_seed_replays_exact_multi_round_trajectory`: independently initialize twice with seed `101`, run two rounds, require equal `to_dict()` and content hash.
- `test_different_root_seed_changes_lineage_even_when_extensional_outcome_matches`: scan seeds until two choose the same world outcome, then require different sample/state hashes because root-seed lineage differs.
- `test_step_and_trajectory_signatures_do_not_accept_replacement_seed`: `inspect.signature(simulate_step)` and `inspect.signature(simulate_trajectory)` have no `seed`; `simulation_state_from_story` has keyword-only `seed` defaulting to `None`.
- `test_decisions_are_seed_invariant_until_stochastic_history_diverges`: compare step-1 decision-result dictionaries across two root seeds and require equality; find two seeds whose admitted alert evidence differs after step 1 and require the step-2 evidence-derived intentional result/posterior differs.
- `test_world_and_observation_namespaces_are_distinct`: inspect a world sample and an observation sample from the same round, require namespaces exactly `world.transition` and `observation.projection`, and require different stream hashes.

- [ ] **Step 5: Create outer seed-diagnostics RED tests**

Define a test-only adapter in `tests/test_narrative_stochastic_seed_diagnostics.py`:

```python
class NarrativeAleatoricAdapter:
    name = "narrative-aleatoric-adapter"

    def simulate(self, scenario, parameters, rng):
        root_seed = rng.getrandbits(64)
        value, lineage_hash = run_test_narrative_once(
            root_seed=root_seed,
            level=float(parameters["level"]),
        )
        return ModelRun(
            events=(),
            outcome={
                "value": value,
                "narrative_root_seed": root_seed,
                "narrative_lineage_hash": lineage_hash,
            },
        )
```

`run_test_narrative_once` uses the stochastic world fixture, maps one sampled outcome to `level + 1.0` and the other to `level - 1.0`, and returns the transition sample content hash.

Create two methods:

- `test_outer_seed_deterministically_maps_to_narrative_root_seed_and_manifest`: same outer seed gives same trace seed, root seed, inner lineage hash, outcome, and manifest hash; a different outer seed changes the recorded trace seed. Do not assert injectivity of the 64-bit derived root seed.
- `test_existing_seed_block_variants_detect_aleatoric_acceptance_drift`: scan outer seeds `1..256` to find opposite aleatoric signs, then call `calibrate_seed_block_variants` with `seed_blocks=((seed_a,),)`, `seed_offsets=(0, seed_b-seed_a)`, `parameter_grid={"level": (-1.0, 1.0)}`, and `target={"value": 0.0}`; require different accepted levels and empty accepted intersection.

- [ ] **Step 6: Run and commit the complete RED**

```bash
python3 -m unittest \
  tests.test_narrative_randomness \
  tests.test_narrative_stochastic_world \
  tests.test_narrative_stochastic_observation \
  tests.test_narrative_stochastic_simulation \
  tests.test_narrative_stochastic_seed_diagnostics -v
python3 -m unittest discover -s tests -v
git add \
  tests/test_narrative_randomness.py \
  tests/test_narrative_stochastic_world.py \
  tests/test_narrative_stochastic_observation.py \
  tests/test_narrative_stochastic_simulation.py \
  tests/test_narrative_stochastic_seed_diagnostics.py
git commit -m "test: define stochastic world observation red boundary"
```

Expected before commit: every pre-existing test remains green; failures are confined to the new P2 modules and are caused by missing stochastic APIs.

- [ ] **Step 7: Capture authoritative exact-head RED proof**

Set `RED_HEAD=$(git rev-parse HEAD)`. Require the `proof` workflow for exactly `RED_HEAD` to show exact checkout, all Lean gates green, and only intended new stochastic Python failures. Do not begin Task 2 until that failure signature is clean.

---

### Task 2: Add the Randomness Core and Seeded World-State Boundary

**Files:**
- Create: `narrative_dynamics/narrative/randomness.py`
- Modify: `narrative_dynamics/narrative/world.py`

**Interfaces:**
- Produces `RANDOM_DERIVATION_VERSION`, `validate_root_seed`, `derive_stream_hash`, `draw_u64_for_stream`, `sample_categorical`, `RandomSampleRecord`.
- Extends `WorldState` with `root_seed: int | None = None` and `world_state_from_story(..., seed=None)`.

- [ ] **Step 1: Implement exact stream derivation**

Create `randomness.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
import math
import re

from narrative_dynamics.contracts import stable_content_hash

RANDOM_DERIVATION_VERSION = "narrative-hash-categorical-v1"
_NAMESPACES = frozenset({"world.transition", "observation.projection"})
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def validate_root_seed(seed: object) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("narrative root seed must be an integer")
    return seed


def derive_stream_hash(*, root_seed, namespace, step_index, source_hash, component_hash, component_key) -> str:
    root_seed = validate_root_seed(root_seed)
    if namespace not in _NAMESPACES:
        raise ValueError("narrative random namespace is unsupported")
    if not isinstance(step_index, int) or isinstance(step_index, bool) or step_index < 0:
        raise ValueError("narrative random step index must be non-negative")
    for value in (source_hash, component_hash):
        if not isinstance(value, str) or _HASH.fullmatch(value) is None:
            raise ValueError("narrative random source/component hash must be sha256")
    if not isinstance(component_key, tuple) or not component_key:
        raise TypeError("narrative random component key must be a non-empty tuple")
    if any(not isinstance(item, str) or not item or item != item.strip() for item in component_key):
        raise ValueError("narrative random component key items must be trimmed strings")
    return stable_content_hash({
        "derivation_version": RANDOM_DERIVATION_VERSION,
        "root_seed": root_seed,
        "namespace": namespace,
        "step_index": step_index,
        "source_hash": source_hash,
        "component_hash": component_hash,
        "component_key": list(component_key),
    })


def draw_u64_for_stream(stream_hash: str, *, draw_index: int = 0) -> int:
    if not isinstance(stream_hash, str) or _HASH.fullmatch(stream_hash) is None:
        raise ValueError("narrative random stream hash must be sha256")
    if not isinstance(draw_index, int) or isinstance(draw_index, bool) or draw_index != 0:
        raise ValueError("narrative random V1 draw index must be zero")
    draw_hash = stable_content_hash({
        "derivation_version": RANDOM_DERIVATION_VERSION,
        "stream_hash": stream_hash,
        "draw_index": draw_index,
    })
    return int(draw_hash.removeprefix("sha256:")[:16], 16)
```

- [ ] **Step 2: Implement `RandomSampleRecord` and categorical selection**

Use this record shape:

```python
@dataclass(frozen=True)
class RandomSampleRecord:
    derivation_version: str
    root_seed: int
    namespace: str
    step_index: int
    source_hash: str
    component_hash: str
    component_key: tuple[str, ...]
    stream_hash: str
    draw_index: int
    draw_u64: int
    distribution_hash: str
    selected_outcome_id: str
    selected_outcome_hash: str
```

`__post_init__` must re-run `derive_stream_hash`, re-run `draw_u64_for_stream`, require `0 <= draw_u64 < 2**64`, validate distribution/outcome hashes, and require a trimmed selected id. `to_dict()` emits every field and `content_hash` is `stable_content_hash(to_dict())`.

Implement:

```python
def sample_categorical(
    *,
    root_seed: int,
    namespace: str,
    step_index: int,
    source_hash: str,
    component_hash: str,
    component_key: tuple[str, ...],
    distribution_hash: str,
    outcomes: tuple[tuple[str, float, str], ...],
) -> RandomSampleRecord:
```

Materialize once, require at least two outcomes, validate ids/probabilities/hashes, sort by id, require exact `math.fsum`, derive stream/draw, compute `u = draw_u64 / 2**64`, choose the first non-final cumulative interval where `u < cumulative`, otherwise the final outcome, then build `RandomSampleRecord`.

- [ ] **Step 3: Add optional root seed to `WorldState` without changing unseeded serialization**

Append after `values`:

```python
root_seed: int | None = None
```

Validate only when non-`None`. Change `to_dict()` from immediate return to `payload = {existing exact keys}` and append `payload["root_seed"]` only when seeded. Extend:

```python
def world_state_from_story(
    story: GenericNarrative,
    domain: DomainSpec,
    *,
    at_time: int | None = None,
    seed: int | None = None,
) -> WorldState:
```

Pass `root_seed=None if seed is None else validate_root_seed(seed)`.

- [ ] **Step 4: Preserve seed across next-state construction**

Add to `WorldStepResult.__post_init__`:

```python
if self.next_state.root_seed != self.prior_state.root_seed:
    raise ValueError("world step next state must preserve root seed")
```

Pass `root_seed=prior.root_seed` from `_atomic_result`. `advance_world_step` signature stays unchanged.

- [ ] **Step 5: Verify and commit Task 2**

```bash
python3 -m unittest \
  tests.test_narrative_randomness \
  tests.test_narrative_world_transition \
  tests.test_narrative_stochastic_world.NarrativeStochasticWorldTests.test_deterministic_world_payloads_and_hashes_remain_v1_shaped -v
git add narrative_dynamics/narrative/randomness.py narrative_dynamics/narrative/world.py
git commit -m "feat: add narrative random seed boundary"
```

Expected: randomness and deterministic compatibility green; stochastic world execution tests still RED.

---

### Task 3: Add the Stochastic World Transition Lane

**Files:**
- Modify: `narrative_dynamics/narrative/world.py`

**Interfaces:**
- Consumes Task 2 randomness APIs.
- Produces `StateDeltaOutcome`, `StateDeltaDistribution`, `StochasticTransitionSample`, `StochasticActionTransitionSpec`, `WorldTransitionModelSpec.stochastic_transitions`, `ActionTransitionRecord.stochastic_sample`.

- [ ] **Step 1: Add finite world distribution records**

```python
@dataclass(frozen=True)
class StateDeltaOutcome:
    outcome_id: str
    probability: float
    delta: StateDelta

    def to_dict(self):
        return {
            "outcome_id": self.outcome_id,
            "probability": self.probability,
            "delta": _delta_payload(self.delta),
        }

    @property
    def content_hash(self):
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class StateDeltaDistribution:
    outcomes: tuple[StateDeltaOutcome, ...]
```

Both constructors enforce the global distribution shape except domain capability; distribution canonicalizes outcomes lexically. `StateDeltaDistribution.to_dict()` emits `{"outcomes": [item.to_dict() for item in self.outcomes]}`.

- [ ] **Step 2: Add stochastic transition spec**

```python
@dataclass(frozen=True)
class StochasticActionTransitionSpec:
    action_type: str
    effects: tuple[ActionEffectSpec, ...]
    parameters: Mapping[str, float]
    distribution_hook: object = field(compare=False, repr=False)
```

Reuse deterministic effect validation. Freeze `parameters` in a lexical `MappingProxyType` with finite float values and non-empty trimmed keys. `to_dict()` includes action type, effects, `[[name, value], ...]`, `random_derivation_version`, and `measure_implementation(distribution_hook).manifest_identity()`.

- [ ] **Step 3: Integrate stochastic specs into model declaration and intent resolution**

Append to `WorldTransitionModelSpec` after `conflict_resolver`:

```python
stochastic_transitions: tuple[StochasticActionTransitionSpec, ...] = ()
```

Validate one cross-lane action-type namespace. `_validate_transition_declarations` checks both lanes. `_resolve_intents` builds one action-type map from both lanes. Update `_allowed_cells` and `_attested_transition_hash` annotations to accept `ActionTransitionSpec | StochasticActionTransitionSpec`.

`WorldTransitionModelSpec.to_dict()` emits `stochastic_transitions` only when non-empty; deterministic payload stays exact.

- [ ] **Step 4: Add stochastic sample wrapper and record link**

```python
@dataclass(frozen=True)
class StochasticTransitionSample:
    distribution: StateDeltaDistribution
    sample_record: RandomSampleRecord

    @property
    def selected_outcome(self):
        matches = tuple(
            item for item in self.distribution.outcomes
            if item.outcome_id == self.sample_record.selected_outcome_id
        )
        if len(matches) != 1:
            raise ValueError("stochastic transition selected outcome is missing")
        return matches[0]
```

`__post_init__` requires world namespace, distribution hash equality, and selected outcome content hash equality. Append `stochastic_sample: StochasticTransitionSample | None = None` after `ActionTransitionRecord.delta`. Require sampled delta equality and conditionally serialize the sample.

- [ ] **Step 5: Execute stochastic actions with pre-sample validation**

Add `_execute_one_stochastic`. Before the hook, require `prior.root_seed is not None`. Call the hook with immutable snapshot and frozen parameters. Require `StateDeltaDistribution`. For every outcome, run `_validated_delta(domain, entities, allowed, outcome.delta)` and rebuild a canonical validated distribution. Then:

```python
transition_hash = _attested_transition_hash(transition)
sample_record = sample_categorical(
    root_seed=prior.root_seed,
    namespace="world.transition",
    step_index=prior.step_index + 1,
    source_hash=prior.content_hash,
    component_hash=transition_hash,
    component_key=(decision.actor_id, decision.id, action.id),
    distribution_hash=validated_distribution.content_hash,
    outcomes=tuple(
        (outcome.outcome_id, outcome.probability, outcome.content_hash)
        for outcome in validated_distribution.outcomes
    ),
)
sample = StochasticTransitionSample(validated_distribution, sample_record)
```

Return `ActionTransitionRecord` using `sample.selected_outcome.delta`, the stochastic spec hash, and the sample wrapper.

- [ ] **Step 6: Certify stochastic lineage in `WorldStepResult`**

For each sampled record require root seed, next-step index, prior-state source hash, transition spec hash, and exact actor/decision/action component key equality. Also require `next_state.root_seed == prior_state.root_seed`.

- [ ] **Step 7: Preserve realized-write conflict behavior**

Do not alter resolver hooks. `_conflict_components` continues using concrete `record.delta`; transition batch hashes continue using `record.to_dict()`, so the stochastic sample is naturally lineage-bound.

- [ ] **Step 8: Verify and commit Task 3**

```bash
python3 -m unittest \
  tests.test_narrative_randomness \
  tests.test_narrative_world_transition \
  tests.test_narrative_conflict_resolution \
  tests.test_narrative_conflict_resolution_safety \
  tests.test_narrative_stochastic_world -v
git add narrative_dynamics/narrative/world.py
git commit -m "feat: add stochastic narrative world transitions"
```

Expected: all listed tests PASS without loosening conflict or capability assertions.

---

### Task 4: Add Truth-Preserving Stochastic Observation Projection

**Files:**
- Modify: `narrative_dynamics/narrative/observation_projection.py`

**Interfaces:**
- Consumes seeded `WorldStepResult` and Task 2 categorical sampling.
- Produces `ObservationOutcome`, `ObservationOutcomeDistribution`, `StochasticObservationSample`, `StochasticObserverProjectionSpec`, additive model/result/observation stochastic fields.

- [ ] **Step 1: Add observation distribution records**

```python
@dataclass(frozen=True)
class ObservationOutcome:
    outcome_id: str
    probability: float
    facts: tuple[ObservationFact, ...]


@dataclass(frozen=True)
class ObservationOutcomeDistribution:
    outcomes: tuple[ObservationOutcome, ...]
```

`ObservationOutcome` requires an exact tuple, only `ObservationFact` values, unique cells, and canonical cell order. Empty facts are valid. Distribution applies the global finite-distribution contract and lexical outcome-id order. Both expose `to_dict()` and `content_hash`.

- [ ] **Step 2: Add stochastic projection spec**

```python
@dataclass(frozen=True)
class StochasticObserverProjectionSpec:
    observer_type: str
    channel: str
    read_capabilities: tuple[ObservationCapabilitySpec, ...]
    emit_capabilities: tuple[ObservationCapabilitySpec, ...]
    parameters: Mapping[str, float]
    distribution_hook: object = field(compare=False, repr=False)
```

Reuse deterministic capability/containment validation, freeze parameters, and bind randomness version plus hook implementation identity in `to_dict()`.

- [ ] **Step 3: Integrate stochastic projection declarations**

Append `stochastic_projections: tuple[StochasticObserverProjectionSpec, ...] = ()` after deterministic projections. Enforce `(observer_type, channel)` uniqueness across both lanes. `_validate_projection_declarations` attests both lanes. Emit stochastic projection payload only when non-empty.

- [ ] **Step 4: Add stochastic projection sample and output links**

```python
@dataclass(frozen=True)
class StochasticObservationSample:
    observer_id: str
    channel: str
    step_index: int
    source_world_state_hash: str
    source_world_step_hash: str
    projection_spec_hash: str
    distribution: ObservationOutcomeDistribution
    sample_record: RandomSampleRecord
```

Require observation namespace, step/source/spec/component-key/distribution/selected-outcome equality. Append `stochastic_sample_hash: str | None = None` to `ProjectedObservation`, and `stochastic_samples: tuple[StochasticObservationSample, ...] = ()` to `ObservationProjectionResult`; both serialize only when populated.

- [ ] **Step 5: Prevalidate every candidate fact tuple**

Extract candidate validation so it can run before sampling. For each candidate fact enforce canonical subject/type/state variable, emit capability, `equals == exact post-step value`, `clear == post-step absence plus explicit effective current-step clear`, and unique cell. Rebuild canonical `ObservationOutcomeDistribution` from validated facts before sampling.

- [ ] **Step 6: Execute stochastic projection**

Before each stochastic hook, require `world_step.next_state.root_seed` non-`None`. Build deterministic-style immutable visible views and call the hook with frozen parameters. Then:

```python
sample_record = sample_categorical(
    root_seed=world_step.next_state.root_seed,
    namespace="observation.projection",
    step_index=world_step.next_state.step_index,
    source_hash=world_step.content_hash,
    component_hash=spec_hash,
    component_key=(observer.id, spec.channel),
    distribution_hash=distribution.content_hash,
    outcomes=tuple(
        (outcome.outcome_id, outcome.probability, outcome.content_hash)
        for outcome in distribution.outcomes
    ),
)
```

Create `StochasticObservationSample`. Emit only the selected facts; each emitted `ProjectedObservation` points to that sample wrapper hash. A selected empty tuple emits no observations but retains the sample in the result.

- [ ] **Step 7: Certify projection result lineage and forged world samples**

`ObservationProjectionResult.__post_init__` requires sample source-world-step/source-world-state/step equality, unique observer/channel samples, valid observation sample references, and selected-fact membership. `_validate_source_world_step` independently checks stochastic transition root/step/source/spec/component-key bindings so constructor-bypassing forged world steps fail projection preflight.

- [ ] **Step 8: Verify and commit Task 4**

```bash
python3 -m unittest \
  tests.test_narrative_observation_projection \
  tests.test_narrative_conflict_projection \
  tests.test_narrative_stochastic_world \
  tests.test_narrative_stochastic_observation -v
```

At this checkpoint the observation test that inspects `RuntimeEvidenceBatch.projection_sample_hashes` remains the only intended RED because Task 5 has not added that field. Commit only if every other listed test is green:

```bash
git add narrative_dynamics/narrative/observation_projection.py
git commit -m "feat: add stochastic narrative observation projection"
```

---

### Task 5: Carry Stochastic Projection Lineage Through Runtime Perception

**Files:**
- Modify: `narrative_dynamics/narrative/runtime_perception.py`

**Interfaces:**
- Consumes `ProjectedObservation.stochastic_sample_hash` and `ObservationProjectionResult.stochastic_samples`.
- Produces `RuntimeEpistemicEvidence.projection_sample_hash` and `RuntimeEvidenceBatch.projection_sample_hashes`.
- Leaves `RuntimePerceptView` unchanged.

- [ ] **Step 1: Append optional evidence sample link**

Append `projection_sample_hash: str | None = None` after `RuntimeEpistemicEvidence.projection_spec_hash`. Validate with `_hash` only when non-`None`, serialize only when present, and leave `percept_view` unchanged.

- [ ] **Step 2: Append complete projection sample hashes to batch**

Append `projection_sample_hashes: tuple[str, ...] = ()` after `RuntimeEvidenceBatch.evidence`. Validate unique sha256 hashes, canonicalize lexically, serialize only when non-empty, and require every non-null evidence sample hash to belong to this tuple.

- [ ] **Step 3: Copy and certify lineage during admission**

In `admit_world_percepts`:

```python
projection_sample_hashes = tuple(
    sorted(sample.content_hash for sample in projection_result.stochastic_samples)
)
```

Pass each projected observation's sample hash into `RuntimeEpistemicEvidence` and the complete tuple into `RuntimeEvidenceBatch`, including the zero-evidence dropout case. In `RuntimePerceptAdmissionResult.__post_init__`, require exact equality between result stochastic sample hashes and batch hashes.

- [ ] **Step 4: Verify and commit Task 5**

```bash
python3 -m unittest \
  tests.test_narrative_runtime_perception \
  tests.test_narrative_runtime_cognition \
  tests.test_narrative_stochastic_observation -v
git add narrative_dynamics/narrative/runtime_perception.py
git commit -m "feat: bind stochastic projection lineage to runtime evidence"
```

Expected: all PASS; cognition continues to receive the same provenance-free `RuntimePerceptView`.

---

### Task 6: Freeze the Root Seed Through Multi-Step Simulation

**Files:**
- Modify: `narrative_dynamics/narrative/runtime_perception.py`
- Modify: `narrative_dynamics/narrative/simulation.py`

**Interfaces:**
- Extends initialization APIs only.
- `simulate_step` and `simulate_trajectory` stay seedless and consume the root seed only transitively through `prior_state.world_state`.

- [ ] **Step 1: Extend ledger initialization**

Change the signature to:

```python
def runtime_evidence_ledger_from_story(
    story: GenericNarrative,
    domain: DomainSpec,
    *,
    at_time: int | None = None,
    seed: int | None = None,
) -> RuntimeEvidenceLedger:
```

Construct the initial world with `world_state_from_story(story, domain, at_time=cutoff, seed=seed)`. Do not add an independent seed field to the ledger.

- [ ] **Step 2: Extend simulation initialization only**

Change:

```python
def simulation_state_from_story(
    story: GenericNarrative,
    domain: DomainSpec,
    model: SimulationModelSpec,
    *,
    at_time: int | None = None,
    seed: int | None = None,
) -> SimulationState:
```

Use the same `seed` for both `world_state_from_story` and `runtime_evidence_ledger_from_story`. Existing `SimulationState` hash binding then proves they refer to the same seeded initial state.

- [ ] **Step 3: Preserve decision isolation**

Do not alter the scheduler's `run_runtime_decision(story, domain, decision_id, ledger, decision_model)` call. No decision API receives root seed or a random stream.

- [ ] **Step 4: Verify and commit Task 6**

```bash
python3 -m unittest \
  tests.test_narrative_simulation \
  tests.test_narrative_runtime_decision_dispatch \
  tests.test_narrative_runtime_reactive \
  tests.test_narrative_runtime_intention \
  tests.test_narrative_runtime_planning \
  tests.test_narrative_stochastic_simulation -v
git add \
  narrative_dynamics/narrative/runtime_perception.py \
  narrative_dynamics/narrative/simulation.py
git commit -m "feat: freeze narrative stochastic seed across trajectories"
```

Expected: all PASS, including exact two-round same-seed replay, seedless step/trajectory signatures, and seed-invariant step-1 decisions.

---

### Task 7: Prove Seed-Diagnostics Compatibility and Finish Integration

**Files:**
- Test: `tests/test_narrative_stochastic_seed_diagnostics.py`
- No production uncertainty/comparison changes.

**Interfaces:**
- Consumes completed stochastic runtime plus existing `SimulationRunner` and `calibrate_seed_block_variants`.
- Produces the final evidence needed for the five P2 stochastic-world/observation roadmap checks.

- [ ] **Step 1: Green the test-only outer adapter**

Keep `NarrativeAleatoricAdapter` entirely in the test module and consume exactly one `rng.getrandbits(64)` before narrative initialization. Outcome includes `value`, `narrative_root_seed`, and `narrative_lineage_hash`. No registry or production adapter is added.

- [ ] **Step 2: Verify outer seed and seed-block diagnostics**

```bash
python3 -m unittest \
  tests.test_narrative_stochastic_seed_diagnostics.NarrativeStochasticSeedDiagnosticsTests.test_outer_seed_deterministically_maps_to_narrative_root_seed_and_manifest \
  tests.test_narrative_stochastic_seed_diagnostics.NarrativeStochasticSeedDiagnosticsTests.test_existing_seed_block_variants_detect_aleatoric_acceptance_drift -v
```

Expected: both PASS; same outer seed replays exact root/lineage/manifest; seed-block variants retain different accepted levels with empty intersection.

- [ ] **Step 3: Run all new stochastic tests and full Python suite**

```bash
python3 -m unittest \
  tests.test_narrative_randomness \
  tests.test_narrative_stochastic_world \
  tests.test_narrative_stochastic_observation \
  tests.test_narrative_stochastic_simulation \
  tests.test_narrative_stochastic_seed_diagnostics -v
python3 -m unittest discover -s tests -v
```

Expected: all new tests PASS and full discovery ends `OK`.

- [ ] **Step 4: Commit final test-only diagnostics closure if Task 1 content required implementation-safe edits**

Only if imports/helper wiring changed after the authoritative RED, commit those test-only changes now:

```bash
git add \
  tests/test_narrative_randomness.py \
  tests/test_narrative_stochastic_world.py \
  tests/test_narrative_stochastic_observation.py \
  tests/test_narrative_stochastic_simulation.py \
  tests/test_narrative_stochastic_seed_diagnostics.py
git diff --cached --quiet || git commit -m "test: close stochastic runtime acceptance"
```

Assertions and acceptance thresholds must remain at least as strict as the RED versions.

- [ ] **Step 5: Verify exact diff scope**

```bash
git diff --name-status e2b505230211a38390e8eda9e92ba25c347703aa...HEAD
```

Allowed non-doc paths are exactly:

```text
A  narrative_dynamics/narrative/randomness.py
M  narrative_dynamics/narrative/world.py
M  narrative_dynamics/narrative/observation_projection.py
M  narrative_dynamics/narrative/runtime_perception.py
M  narrative_dynamics/narrative/simulation.py
A  tests/test_narrative_randomness.py
A  tests/test_narrative_stochastic_world.py
A  tests/test_narrative_stochastic_observation.py
A  tests/test_narrative_stochastic_simulation.py
A  tests/test_narrative_stochastic_seed_diagnostics.py
```

The spec and plan are the only allowed doc additions.

- [ ] **Step 6: Capture exact-head feature GREEN proof**

```bash
FEATURE_HEAD=$(git rev-parse HEAD)
echo "$FEATURE_HEAD"
```

Require the `proof` workflow whose `head_sha` is exactly `$FEATURE_HEAD` to reach `completed / success`. Read its Python job log and record the exact `Ran N tests ... OK` line. Require Lean dependency/conformance/full build/theorem, story theorem, and testimony theorem steps all green.

- [ ] **Step 7: Review the complete diff before PR**

Confirm all of these directly from the diff/tests:

```text
no RNG in deterministic hooks
no RNG in decision APIs
no stochastic conflict resolver
no false observation values
all unsampled candidates validated
root seed frozen through world chain
empty observation samples retained in evidence lineage
world/observation namespaces distinct
deterministic optional fields omitted
no uncertainty/comparison/Lean production changes
```

- [ ] **Step 8: Open PR with RED/GREEN evidence**

The PR body records integrated base SHA, spec/plan paths, authoritative test-only RED SHA/run/failure signature, final feature SHA/run, deterministic compatibility evidence, same-seed trajectory replay, seed-diagnostics evidence, exact scope, and explicit statement that P2 Identification and Model Comparison remains open.

- [ ] **Step 9: Require PR synthetic-merge GREEN, then guarded merge**

Require the PR-triggered `proof` synthetic merge for the current head/base to complete successfully. Then merge with `expected_head_sha` set to the current feature head; do not merge if head moved.

- [ ] **Step 10: Require post-merge exact-head GREEN**

Read the new `proof/narrative-dynamics-v0` merge commit SHA. Require its push-triggered `proof` run to complete successfully and read exact Python `Ran N tests ... OK` evidence before updating roadmap status.

- [ ] **Step 11: Update only the five #27 stochastic-world/observation checkboxes**

Mark exactly these `[x]` after post-merge GREEN:

```text
Explicit RNG/seed boundary
Seed included in transition/observation lineage
Replay exactness under fixed seed
Separate aleatoric transition noise from decision stochasticity
Batch/seed diagnostics compatible with existing uncertainty infrastructure
```

Update the integrated baseline/evidence paragraph with PR number, merge commit, and post-merge proof. Keep issue #27 open and leave every P2 Identification and Model Comparison checkbox unchanged.
