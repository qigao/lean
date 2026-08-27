# Narrative Stochastic World / Observation V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit, seeded, replayable aleatoric stochastic world transitions and truth-preserving stochastic observation projection while preserving deterministic V1/V2 payloads, hashes, decision APIs, conflict semantics, and uncertainty/comparison infrastructure.

**Architecture:** Add one runtime-owned stateless hash-categorical randomness core. World and observation models gain separate additive stochastic lanes whose hooks return finite distributions without receiving RNG; the trusted runtime validates every candidate, derives component-local substreams from a frozen narrative root seed, samples one outcome, and binds the complete sample lineage into existing world/projection/evidence/simulation hashes. `simulate_step()` remains seedless after initialization, decision dispatch remains RNG-free, and deterministic records omit all new optional fields.

**Tech Stack:** Python 3 stdlib (`dataclasses`, `math`, `MappingProxyType`, `unittest`), existing `stable_content_hash`, implementation attestation, GenericNarrative/DomainSpec world transition, Conflict Resolution V2, observation projection, runtime percept ledger, multi-step simulation scheduler, outer `SimulationRunner`, existing uncertainty seed-block diagnostics, GitHub Actions `proof.yml`.

**Spec:** `docs/superpowers/specs/2026-08-28-narrative-stochastic-world-observation-v1-design.md`

## Global Constraints

- Integrated base: `proof/narrative-dynamics-v0@e2b505230211a38390e8eda9e92ba25c347703aa`.
- Feature branch: `work/narrative-stochastic-world-observation-v1`.
- Approved written-spec head before planning: `93904c6d5eee05a6bd69011c701f83c1a376cbc8`.
- Docs-only spec/plan commits do not count as feature proof. The first authoritative feature proof must be the complete test-only RED commit.
- Strict sequence: test-only RED -> exact-head RED evidence -> minimal GREEN tasks -> exact-head feature GREEN -> PR synthetic-merge GREEN -> merge -> post-merge exact-head GREEN.
- No production code may be committed in the authoritative RED commit.
- Create only one new production module: `narrative_dynamics/narrative/randomness.py`.
- Modify production only in `narrative_dynamics/narrative/world.py`, `narrative_dynamics/narrative/observation_projection.py`, `narrative_dynamics/narrative/runtime_perception.py`, and `narrative_dynamics/narrative/simulation.py` in addition to the new randomness module.
- Do not modify `runtime_decision_dispatch.py`, `runtime_reactive.py`, `runtime_intention.py`, `runtime_planning.py`, conflict resolver semantics, comparison/preregistration/release code, uncertainty/calibration core, committed fixtures, package-root exports, or Lean sources.
- Do not add RNG arguments to existing deterministic world or projection hooks.
- `run_runtime_decision(...)` and all Reactive/Intentional/Planning APIs remain RNG-free.
- Conflict Resolution V2 remains deterministic and runs only after stochastic world transitions have been sampled into concrete transition records.
- Observation stochasticity is truth-preserving: it may vary visibility/dropout/selection among true facts but may not emit a false `equals` value.
- `RANDOM_DERIVATION_VERSION` is exactly `"narrative-hash-categorical-v1"`.
- Random namespaces are exactly `"world.transition"` and `"observation.projection"` in V1.
- World sample step index is `prior_state.step_index + 1`; observation sample step index is `world_step.next_state.step_index`.
- World sample source hash is the exact prior `WorldState.content_hash`; observation sample source hash is the exact `WorldStepResult.content_hash`.
- World component key is exactly `(actor_id, decision_id, action_id)`.
- Observation component key is exactly `(observer_id, channel)`.
- Sampling is stateless and component-local. No shared mutable RNG or process-global RNG is permitted.
- Every finite distribution contains at least two positive-probability outcomes, unique trimmed outcome ids, `math.fsum(probabilities) == 1.0`, and lexical outcome-id canonicalization. No implicit renormalization.
- Every candidate stochastic world delta is capability-validated before sampling.
- Every candidate stochastic observation fact tuple is shape/capability/truth-validated before sampling.
- Empty stochastic observation outcomes are valid but must still leave projection and evidence-batch sampling lineage.
- `WorldState.root_seed` is optional, appended after existing fields, omitted from `to_dict()` when `None`, and immutable across a state chain.
- A stochastic lane requires a non-`None` root seed before its hook is called.
- `simulation_state_from_story(..., seed=None)` is the only scheduler entry point that accepts the narrative root seed. `simulate_step()` and `simulate_trajectory()` do not gain per-step/per-round seeds.
- Deterministic V1/V2 `to_dict()` payloads and content hashes remain exact because every new optional field is omitted when absent/empty.
- Existing deterministic tests must remain unchanged unless a pre-existing test helper cannot construct the additive optional field; assertions must never be weakened.
- The spec phrase about one observer not perturbing another is interpreted consistently with the declared source-hash contract: on the **same exact world step**, adding/reordering an unrelated stochastic projection component must not perturb an existing observer/channel sample. Changing the story/entity set changes the world-step source hash and is not an invariance claim.
- Outer `SimulationRunner` seed compatibility is test-adapter-only: one fixed `rng.getrandbits(64)` call maps the already-recorded outer simulation seed deterministically to a narrative root seed. No production uncertainty/calibration API changes.
- Do not mark any #27 P2 Stochastic World / Observation checkbox complete until merge plus post-merge exact-head proof. P2 Identification and Model Comparison remains untouched and open.

## File Structure

### Production

- `narrative_dynamics/narrative/randomness.py` — seed validation, closed namespaces, versioned stream derivation, deterministic `draw_u64`, categorical selection, and self-validating `RandomSampleRecord`.
- `narrative_dynamics/narrative/world.py` — additive stochastic transition distribution/spec/sample records, optional world root seed, stochastic execution lane, concrete-sample conflict integration, deterministic compatibility.
- `narrative_dynamics/narrative/observation_projection.py` — additive stochastic projection distribution/spec/sample records, truth-preserving prevalidation, selected-fact projection lineage, empty-outcome lineage, deterministic compatibility.
- `narrative_dynamics/narrative/runtime_perception.py` — copy optional projection-sample lineage into evidence and preserve complete sample hashes in every evidence batch.
- `narrative_dynamics/narrative/simulation.py` — initialize/freeze root seed and preserve seedless step/trajectory APIs.

### Tests

- `tests/test_narrative_randomness.py` — randomness derivation, finite categorical validation, exact replay, forgery rejection.
- `tests/test_narrative_stochastic_world.py` — deterministic world compatibility, stochastic transitions, order-independent substreams, pre-sample validation, realized-write conflicts.
- `tests/test_narrative_stochastic_observation.py` — deterministic projection compatibility, truth-preserving stochastic projection, dropout lineage, independent projection components.
- `tests/test_narrative_stochastic_simulation.py` — seeded initialization, multi-round replay, seed immutability, decision/aleatoric separation, world/observation namespace separation.
- `tests/test_narrative_stochastic_seed_diagnostics.py` — outer `SimulationRunner` seed mapping and existing seed-block diagnostic compatibility without production core changes.

Approved spec and this plan remain docs-only evidence.

---

### Task 1: Complete the Authoritative Test-Only RED Boundary

**Files:**
- Create: `tests/test_narrative_randomness.py`
- Create: `tests/test_narrative_stochastic_world.py`
- Create: `tests/test_narrative_stochastic_observation.py`
- Create: `tests/test_narrative_stochastic_simulation.py`
- Create: `tests/test_narrative_stochastic_seed_diagnostics.py`

**Interfaces:**
- Consumes existing deterministic world/projection/perception/simulation helpers and outer `SimulationRunner`/uncertainty APIs.
- Produces the complete P2 acceptance boundary before `randomness.py` or any stochastic production symbols exist.

- [ ] **Step 1: Add guarded randomness imports and exact randomness tests**

Create `tests/test_narrative_randomness.py` with this import boundary and test names:

```python
from __future__ import annotations

from dataclasses import replace
import unittest

_RANDOMNESS_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.randomness import (
        RANDOM_DERIVATION_VERSION,
        RandomSampleRecord,
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

    def test_seed_validation_and_namespaces_are_closed(self):
        self.require_randomness()
        self.assertEqual(validate_root_seed(0), 0)
        self.assertEqual(validate_root_seed(-7), -7)
        for bad in (True, False, 1.0, "1", None):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    validate_root_seed(bad)

    def test_stream_and_draw_are_exact_and_replay(self):
        self.require_randomness()
        args = dict(
            root_seed=42,
            namespace="world.transition",
            step_index=1,
            source_hash="sha256:" + "1" * 64,
            component_hash="sha256:" + "2" * 64,
            component_key=("a1", "d1", "act"),
        )
        first = derive_stream_hash(**args)
        second = derive_stream_hash(**args)
        self.assertEqual(first, second)
        self.assertEqual(draw_u64_for_stream(first), draw_u64_for_stream(second))
        self.assertEqual(RANDOM_DERIVATION_VERSION, "narrative-hash-categorical-v1")

    def test_component_key_change_changes_stream_identity(self):
        self.require_randomness()
        common = dict(
            root_seed=42,
            namespace="world.transition",
            step_index=1,
            source_hash="sha256:" + "1" * 64,
            component_hash="sha256:" + "2" * 64,
        )
        self.assertNotEqual(
            derive_stream_hash(component_key=("a1", "d1", "act"), **common),
            derive_stream_hash(component_key=("a2", "d2", "act"), **common),
        )

    def test_outcome_input_order_does_not_change_selection(self):
        self.require_randomness()
        common = dict(
            root_seed=9,
            namespace="observation.projection",
            step_index=2,
            source_hash="sha256:" + "3" * 64,
            component_hash="sha256:" + "4" * 64,
            component_key=("a1", "vision"),
            distribution_hash="sha256:" + "5" * 64,
        )
        outcomes = (
            ("emit", 0.5, "sha256:" + "6" * 64),
            ("drop", 0.5, "sha256:" + "7" * 64),
        )
        first = sample_categorical(outcomes=outcomes, **common)
        second = sample_categorical(outcomes=tuple(reversed(outcomes)), **common)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_distribution_validation_rejects_invalid_probabilities(self):
        self.require_randomness()
        common = dict(
            root_seed=1,
            namespace="world.transition",
            step_index=1,
            source_hash="sha256:" + "1" * 64,
            component_hash="sha256:" + "2" * 64,
            component_key=("a1", "d1", "act"),
            distribution_hash="sha256:" + "3" * 64,
        )
        bad_sets = (
            (("only", 1.0, "sha256:" + "4" * 64),),
            (("a", 0.0, "sha256:" + "4" * 64), ("b", 1.0, "sha256:" + "5" * 64)),
            (("a", 0.4, "sha256:" + "4" * 64), ("b", 0.4, "sha256:" + "5" * 64)),
            (("a", 0.5, "sha256:" + "4" * 64), ("a", 0.5, "sha256:" + "5" * 64)),
        )
        for outcomes in bad_sets:
            with self.subTest(outcomes=outcomes):
                with self.assertRaises((TypeError, ValueError)):
                    sample_categorical(outcomes=outcomes, **common)

    def test_random_sample_record_forgery_rejects(self):
        self.require_randomness()
        record = sample_categorical(
            root_seed=4,
            namespace="world.transition",
            step_index=1,
            source_hash="sha256:" + "1" * 64,
            component_hash="sha256:" + "2" * 64,
            component_key=("a1", "d1", "act"),
            distribution_hash="sha256:" + "3" * 64,
            outcomes=(("a", 0.5, "sha256:" + "4" * 64), ("b", 0.5, "sha256:" + "5" * 64)),
        )
        for field, value in (
            ("stream_hash", "sha256:" + "f" * 64),
            ("draw_u64", record.draw_u64 ^ 1),
            ("namespace", "decision.choice"),
        ):
            with self.subTest(field=field):
                with self.assertRaises((TypeError, ValueError)):
                    replace(record, **{field: value})
```

- [ ] **Step 2: Add stochastic world acceptance tests**

Create `tests/test_narrative_stochastic_world.py`. Reuse `make_world_domain`, `make_world_story`, `_agent_cell`, and `intent` from `tests.test_narrative_world_transition`; reuse conflict helpers from `tests.test_narrative_conflict_resolution` only when an exact helper already exists. Guard-import these new production symbols:

```python
_WORLD_STOCHASTIC_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.world import (
        StateDeltaDistribution,
        StateDeltaOutcome,
        StochasticActionTransitionSpec,
        StochasticTransitionSample,
    )
except ImportError as error:
    _WORLD_STOCHASTIC_IMPORT_ERROR = error
```

Define one controlled hook whose candidate deltas have distinct outcomes and which records calls:

```python
class StochasticPhaseHook:
    def __init__(self):
        self.calls = []

    def __call__(self, snapshot, decision, action, parameters):
        self.calls.append((dict(snapshot), decision.id, action.id, dict(parameters)))
        return StateDeltaDistribution((
            StateDeltaOutcome(
                "ready",
                parameters["ready_probability"],
                StateDelta((StateDeltaOp(
                    "set", decision.actor_id, "agent.phase",
                    TypedValue("PhaseState", "ready"),
                ),)),
            ),
            StateDeltaOutcome(
                "done",
                1.0 - parameters["ready_probability"],
                StateDelta((StateDeltaOp(
                    "set", decision.actor_id, "agent.phase",
                    TypedValue("PhaseState", "done"),
                ),)),
            ),
        ))
```

Add these exact tests:

```python
class NarrativeStochasticWorldTests(unittest.TestCase):
    def require_stochastic_world(self):
        if _WORLD_STOCHASTIC_IMPORT_ERROR is not None:
            self.fail(f"stochastic world API is missing: {_WORLD_STOCHASTIC_IMPORT_ERROR}")

    def test_deterministic_world_payloads_and_hashes_remain_v1_shaped(self): ...
    def test_same_seed_replays_stochastic_transition_exactly(self): ...
    def test_actor_order_and_unrelated_actor_do_not_perturb_component_sample(self): ...
    def test_multiple_seeds_exercise_declared_world_outcomes(self): ...
    def test_every_candidate_delta_is_validated_before_sampling(self): ...
    def test_stochastic_action_requires_seed_before_hook(self): ...
    def test_realized_stochastic_writes_feed_existing_conflict_resolution(self): ...
```

For the deterministic compatibility test, explicitly assert `root_seed`/`stochastic_transitions`/`stochastic_sample` are absent from `to_dict()` when unused, and assert `stable_content_hash(expected_old_payload) == current.content_hash` for each reconstructed old payload.

For the order test, use the same story/prior/model/spec and two already-declared actors. Compare the `StochasticTransitionSample.to_dict()` for actor `bob` between `(bob, alice)` and `(alice, bob)`, then compare bob's sample again against a run that omits alice's intent. Do not change story/entity/model/spec identity between those comparisons.

For variation, scan root seeds `range(256)` and stop after both `ready` and `done` have appeared; assert the observed set is exactly `{"ready", "done"}`.

For pre-sample validation, return one valid delta and one outside-capability delta with probability `1e-12`; assert the entire world step fails before any successful state is returned.

For missing seed, call `world_state_from_story(..., seed=None)` and assert failure occurs before `StochasticPhaseHook.calls` is appended.

- [ ] **Step 3: Add stochastic observation acceptance tests**

Create `tests/test_narrative_stochastic_observation.py`. Reuse `make_projection_domain`, `make_projection_story`, `make_world_step`, `_cell`, and deterministic projection helpers from `tests.test_narrative_observation_projection`. Guard-import:

```python
_OBSERVATION_STOCHASTIC_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.observation_projection import (
        ObservationOutcome,
        ObservationOutcomeDistribution,
        StochasticObservationSample,
        StochasticObserverProjectionSpec,
    )
except ImportError as error:
    _OBSERVATION_STOCHASTIC_IMPORT_ERROR = error
```

Define a controlled truth-preserving emit/drop hook:

```python
class StochasticOwnLocationHook:
    def __init__(self):
        self.calls = []

    def __call__(self, prior_visible, next_visible, observer, step_index, parameters):
        self.calls.append((observer.id, step_index, dict(parameters)))
        cell = _cell(observer.id, "Agent", "agent.location")
        fact = ObservationFact(cell, "equals", next_visible[cell])
        return ObservationOutcomeDistribution((
            ObservationOutcome("drop", 1.0 - parameters["emit_probability"], ()),
            ObservationOutcome("emit", parameters["emit_probability"], (fact,)),
        ))
```

Add these exact tests:

```python
class NarrativeStochasticObservationTests(unittest.TestCase):
    def require_stochastic_observation(self): ...
    def test_deterministic_projection_payloads_and_hashes_remain_v1_shaped(self): ...
    def test_same_seed_and_world_step_replay_projection_exactly(self): ...
    def test_spec_order_and_unrelated_projection_do_not_perturb_existing_sample(self): ...
    def test_seed_scan_exercises_emit_and_dropout(self): ...
    def test_empty_outcome_keeps_projection_and_evidence_batch_lineage(self): ...
    def test_every_candidate_fact_is_validated_before_sampling(self): ...
    def test_stochastic_projection_requires_seed_before_hook(self): ...
```

The unrelated-component invariance test must keep the **same `WorldStepResult`**. Compare the `(a1, "vision")` sample under a model with only that stochastic projection against a model that also contains a different channel/spec; do not add/remove story entities.

The invalid-fact test must include a false `equals` outcome with probability `1e-12` and assert failure even when the selected draw would have chosen the valid outcome.

The empty-outcome test must assert `result.observations == ()` for a selected dropout while `result.stochastic_samples` contains one sample and the later `RuntimeEvidenceBatch.projection_sample_hashes` contains that sample hash.

- [ ] **Step 4: Add scheduler/trajectory stochastic tests**

Create `tests/test_narrative_stochastic_simulation.py`. Reuse `make_scheduler_domain`, `make_scheduler_story`, `make_intentional_models`, `alert_cell`, and deterministic scheduler helpers from `tests.test_narrative_simulation`. Guard-import the stochastic world/projection symbols plus existing scheduler public functions.

Add these exact tests:

```python
class NarrativeStochasticSimulationTests(unittest.TestCase):
    def require_stochastic_simulation(self): ...
    def test_seeded_initialization_binds_world_and_ledger_to_same_initial_hash(self): ...
    def test_same_seed_replays_exact_multi_round_trajectory(self): ...
    def test_different_root_seed_changes_lineage_even_when_extensional_outcome_matches(self): ...
    def test_step_and_trajectory_signatures_do_not_accept_replacement_seed(self): ...
    def test_decisions_are_seed_invariant_until_stochastic_history_diverges(self): ...
    def test_world_and_observation_namespaces_are_distinct(self): ...
```

Use `inspect.signature(simulate_step)` and `inspect.signature(simulate_trajectory)` to assert neither contains `seed`. Assert `inspect.signature(simulation_state_from_story)` contains `seed` with default `None`.

For same-seed replay, construct two initial states independently with the same seed and run two rounds; require byte-for-byte equal `to_dict()` and equal trajectory hashes.

For different-seed/same-extensional outcome, scan seeds until two seeds select the same world outcome; assert their stochastic sample records and state hashes still differ because root-seed lineage differs.

For decision separation, compare step-1 `RuntimeDecisionDispatchResult.to_dict()` across two initial root seeds before world/projection stochasticity can affect evidence; require equality. Then use a controlled projection distribution and a finite seed scan to find two branches whose admitted evidence differs by step 1; require the step-2 model-visible posterior/decision result to differ where the existing intentional model responds to that evidence.

- [ ] **Step 5: Add outer seed-diagnostics compatibility tests**

Create `tests/test_narrative_stochastic_seed_diagnostics.py`. Define a test-only `SimulatorModel` adapter that maps the outer runner RNG to one narrative root seed with exactly one call:

```python
class NarrativeAleatoricAdapter:
    name = "narrative-aleatoric-adapter"

    def simulate(self, scenario, parameters, rng):
        root_seed = rng.getrandbits(64)
        value, lineage = run_test_narrative_once(
            root_seed=root_seed,
            level=float(parameters["level"]),
        )
        return ModelRun(
            events=(),
            outcome={
                "value": value,
                "narrative_root_seed": root_seed,
                "narrative_lineage_hash": lineage,
            },
        )
```

`run_test_narrative_once` is a test helper built from the stochastic world fixture in `tests.test_narrative_stochastic_world`; it returns `level + 1.0` for one sampled world outcome and `level - 1.0` for the other, plus the stochastic transition sample hash.

Add:

```python
class NarrativeStochasticSeedDiagnosticsTests(unittest.TestCase):
    def test_outer_seed_deterministically_maps_to_narrative_root_seed_and_manifest(self): ...
    def test_existing_seed_block_variants_detect_aleatoric_acceptance_drift(self): ...
```

For the second test, scan outer seeds `1..256` to find `seed_a` and `seed_b` that produce opposite aleatoric signs. Then call existing `calibrate_seed_block_variants` with:

```python
seed_blocks=((seed_a,),)
seed_offsets=(0, seed_b - seed_a)
parameter_grid={"level": (-1.0, 1.0)}
target={"value": 0.0}
```

Assert the two variants select different accepted `level` values and the accepted intersection is empty. Do not modify uncertainty production code.

- [ ] **Step 6: Run the complete new stochastic suite and record local RED**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_randomness \
  tests.test_narrative_stochastic_world \
  tests.test_narrative_stochastic_observation \
  tests.test_narrative_stochastic_simulation \
  tests.test_narrative_stochastic_seed_diagnostics -v
```

Expected: FAIL only because the new randomness/stochastic production symbols are missing. No existing deterministic test is allowed to fail.

- [ ] **Step 7: Run the full Python suite before committing RED**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: every pre-existing test remains `ok`; all failures/errors are confined to the new P2 test modules and are caused by missing new stochastic APIs.

- [ ] **Step 8: Commit the authoritative test-only RED**

```bash
git add \
  tests/test_narrative_randomness.py \
  tests/test_narrative_stochastic_world.py \
  tests/test_narrative_stochastic_observation.py \
  tests/test_narrative_stochastic_simulation.py \
  tests/test_narrative_stochastic_seed_diagnostics.py
git commit -m "test: define stochastic world observation red boundary"
```

- [ ] **Step 9: Capture exact-head RED proof before any production commit**

Verify the branch SHA, locate the `proof` workflow for that exact SHA, and require:

```text
checkout == exact RED commit
Lean dependency/conformance/build/theorem gates == success
existing Python tests == success
new stochastic tests == intended missing-API failures only
```

Do not begin Task 2 until the RED failure signature is clean.

---

### Task 2: Add the Randomness Core and Seeded World-State Boundary

**Files:**
- Create: `narrative_dynamics/narrative/randomness.py`
- Modify: `narrative_dynamics/narrative/world.py`
- Test: `tests/test_narrative_randomness.py`
- Test: `tests/test_narrative_stochastic_world.py`

**Interfaces:**
- Produces `RANDOM_DERIVATION_VERSION`, `validate_root_seed`, `derive_stream_hash`, `draw_u64_for_stream`, `sample_categorical`, and `RandomSampleRecord`.
- Extends `WorldState` with `root_seed: int | None = None` and `world_state_from_story(..., seed=None)` while keeping unseeded payloads exact.
- Later tasks consume the frozen root seed and generic sample record.

- [ ] **Step 1: Implement closed randomness validation and exact derivation**

Create `narrative_dynamics/narrative/randomness.py` with these public interfaces:

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


def derive_stream_hash(
    *,
    root_seed: int,
    namespace: str,
    step_index: int,
    source_hash: str,
    component_hash: str,
    component_key: tuple[str, ...],
) -> str:
    root_seed = validate_root_seed(root_seed)
    if namespace not in _NAMESPACES:
        raise ValueError("narrative random namespace is unsupported")
    if not isinstance(step_index, int) or isinstance(step_index, bool) or step_index < 0:
        raise ValueError("narrative random step index must be non-negative")
    for value, label in ((source_hash, "source hash"), (component_hash, "component hash")):
        if not isinstance(value, str) or _HASH.fullmatch(value) is None:
            raise ValueError(f"narrative random {label} must be a sha256 hash")
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
    if not isinstance(draw_index, int) or isinstance(draw_index, bool) or draw_index != 0:
        raise ValueError("narrative random V1 draw index must be zero")
    draw_hash = stable_content_hash({
        "derivation_version": RANDOM_DERIVATION_VERSION,
        "stream_hash": stream_hash,
        "draw_index": draw_index,
    })
    return int(draw_hash.removeprefix("sha256:")[:16], 16)
```

- [ ] **Step 2: Implement self-validating sample records and categorical selection**

Add:

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

    def __post_init__(self) -> None:
        if self.derivation_version != RANDOM_DERIVATION_VERSION:
            raise ValueError("narrative random derivation version mismatch")
        expected_stream = derive_stream_hash(
            root_seed=self.root_seed,
            namespace=self.namespace,
            step_index=self.step_index,
            source_hash=self.source_hash,
            component_hash=self.component_hash,
            component_key=self.component_key,
        )
        if self.stream_hash != expected_stream:
            raise ValueError("narrative random stream hash mismatch")
        expected_draw = draw_u64_for_stream(expected_stream, draw_index=self.draw_index)
        if self.draw_u64 != expected_draw:
            raise ValueError("narrative random draw mismatch")
        if not isinstance(self.distribution_hash, str) or _HASH.fullmatch(self.distribution_hash) is None:
            raise ValueError("narrative random distribution hash must be sha256")
        if not isinstance(self.selected_outcome_id, str) or not self.selected_outcome_id.strip():
            raise ValueError("narrative random selected outcome id is invalid")
        if not isinstance(self.selected_outcome_hash, str) or _HASH.fullmatch(self.selected_outcome_hash) is None:
            raise ValueError("narrative random selected outcome hash must be sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "derivation_version": self.derivation_version,
            "root_seed": self.root_seed,
            "namespace": self.namespace,
            "step_index": self.step_index,
            "source_hash": self.source_hash,
            "component_hash": self.component_hash,
            "component_key": list(self.component_key),
            "stream_hash": self.stream_hash,
            "draw_index": self.draw_index,
            "draw_u64": self.draw_u64,
            "distribution_hash": self.distribution_hash,
            "selected_outcome_id": self.selected_outcome_id,
            "selected_outcome_hash": self.selected_outcome_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())
```

Implement `sample_categorical(...)` with exact signature:

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

Inside it, materialize exactly once, sort lexically by outcome id, reject fewer than two outcomes/duplicate ids/non-finite-or-bool/non-positive probabilities/invalid outcome hashes, require `math.fsum(probabilities) == 1.0`, compute `u = draw_u64 / 2**64`, choose the first non-final cumulative interval strictly greater than `u`, otherwise choose the final outcome, then return `RandomSampleRecord`.

- [ ] **Step 3: Extend `WorldState` and initialization without changing unseeded payloads**

In `world.py`, append:

```python
@dataclass(frozen=True)
class WorldState:
    # existing fields unchanged and in existing order
    values: Mapping[StateCellRef, TypedValue]
    root_seed: int | None = None
```

Validation:

```python
if self.root_seed is not None:
    object.__setattr__(self, "root_seed", validate_root_seed(self.root_seed))
```

Serialization must remain conditional:

```python
payload = {  # exact existing V1 payload
    ...
}
if self.root_seed is not None:
    payload["root_seed"] = self.root_seed
return payload
```

Extend initialization only:

```python
def world_state_from_story(
    story: GenericNarrative,
    domain: DomainSpec,
    *,
    at_time: int | None = None,
    seed: int | None = None,
) -> WorldState:
```

Set `root_seed=None if seed is None else validate_root_seed(seed)`.

- [ ] **Step 4: Preserve the root seed across every next world state**

In `WorldStepResult.__post_init__`, add:

```python
if self.next_state.root_seed != self.prior_state.root_seed:
    raise ValueError("world step next state must preserve root seed")
```

In `_atomic_result`, set:

```python
root_seed=prior.root_seed,
```

Do not change `advance_world_step(...)` signature.

- [ ] **Step 5: Run randomness and deterministic world compatibility tests**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_randomness \
  tests.test_narrative_world_transition \
  tests.test_narrative_stochastic_world.NarrativeStochasticWorldTests.test_deterministic_world_payloads_and_hashes_remain_v1_shaped -v
```

Expected: randomness tests PASS; all existing world-transition tests PASS; deterministic compatibility test PASS. Stochastic execution tests remain RED because stochastic world records/specs are not implemented yet.

- [ ] **Step 6: Commit the randomness/seed boundary**

```bash
git add \
  narrative_dynamics/narrative/randomness.py \
  narrative_dynamics/narrative/world.py
git commit -m "feat: add narrative random seed boundary"
```

---

### Task 3: Add the Stochastic World Transition Lane

**Files:**
- Modify: `narrative_dynamics/narrative/world.py`
- Test: `tests/test_narrative_stochastic_world.py`

**Interfaces:**
- Consumes `RandomSampleRecord` and `sample_categorical` from Task 2.
- Produces `StateDeltaOutcome`, `StateDeltaDistribution`, `StochasticTransitionSample`, `StochasticActionTransitionSpec`, additive `WorldTransitionModelSpec.stochastic_transitions`, and additive `ActionTransitionRecord.stochastic_sample`.
- Leaves deterministic `ActionTransitionSpec` and `advance_world_step(...)` signatures unchanged.

- [ ] **Step 1: Add finite state-delta distribution records**

Add these exact public records:

```python
@dataclass(frozen=True)
class StateDeltaOutcome:
    outcome_id: str
    probability: float
    delta: StateDelta

    def to_dict(self) -> dict[str, object]:
        return {
            "outcome_id": self.outcome_id,
            "probability": self.probability,
            "delta": _delta_payload(self.delta),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class StateDeltaDistribution:
    outcomes: tuple[StateDeltaOutcome, ...]

    def to_dict(self) -> dict[str, object]:
        return {"outcomes": [item.to_dict() for item in self.outcomes]}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())
```

Constructors must enforce the global finite-distribution rules and lexical outcome-id canonicalization. `StateDeltaOutcome` must require an actual `StateDelta` but does not perform domain/capability validation; runtime execution performs that for every candidate before sampling.

- [ ] **Step 2: Add stochastic transition spec identity**

Add:

```python
@dataclass(frozen=True)
class StochasticActionTransitionSpec:
    action_type: str
    effects: tuple[ActionEffectSpec, ...]
    parameters: Mapping[str, float]
    distribution_hook: object = field(compare=False, repr=False)
```

Freeze/canonicalize parameters with lexical keys and finite float values; reject bools. Reuse `ActionTransitionSpec` effect validation rules. Require a callable hook. `to_dict()` must include:

```python
{
    "action_type": self.action_type,
    "effects": [effect.to_dict() for effect in self.effects],
    "parameters": [[name, value] for name, value in sorted(self.parameters.items())],
    "random_derivation_version": RANDOM_DERIVATION_VERSION,
    "implementation_identity": measure_implementation(self.distribution_hook).manifest_identity(),
}
```

- [ ] **Step 3: Add stochastic transitions to `WorldTransitionModelSpec` without changing deterministic identity**

Append after `conflict_resolver`:

```python
stochastic_transitions: tuple[StochasticActionTransitionSpec, ...] = ()
```

Validate deterministic and stochastic action types in one uniqueness set. Canonicalize both tuples lexically by action type. In `to_dict()`:

```python
if self.stochastic_transitions:
    payload["stochastic_transitions"] = [
        item.to_dict() for item in self.stochastic_transitions
    ]
```

When empty, emit no new key.

- [ ] **Step 4: Add stochastic transition sampling records**

Add:

```python
@dataclass(frozen=True)
class StochasticTransitionSample:
    distribution: StateDeltaDistribution
    sample_record: RandomSampleRecord

    @property
    def selected_outcome(self) -> StateDeltaOutcome:
        return next(
            item for item in self.distribution.outcomes
            if item.outcome_id == self.sample_record.selected_outcome_id
        )
```

`__post_init__` must require `sample_record.namespace == "world.transition"`, exact distribution hash equality, selected outcome existence, and selected outcome content hash equality. `to_dict()` contains both complete distribution and sample record.

Append to `ActionTransitionRecord`:

```python
stochastic_sample: StochasticTransitionSample | None = None
```

If present, require `self.delta == stochastic_sample.selected_outcome.delta`. Serialize only when non-`None`:

```python
if self.stochastic_sample is not None:
    payload["stochastic_sample"] = self.stochastic_sample.to_dict()
```

- [ ] **Step 5: Resolve deterministic vs stochastic selected action types through one closed execution boundary**

Change the internal resolution tuple to carry either transition spec kind. Do not change the public `ActionIntent` or `advance_world_step` signature.

For stochastic transitions, call this new internal path:

```python
def _execute_one_stochastic(
    snapshot,
    domain,
    entities,
    prior,
    resolved,
) -> ActionTransitionRecord:
    item, decision, action, transition = resolved
    if prior.root_seed is None:
        raise WorldTransitionError("stochastic world transition requires a root seed")
    allowed = _allowed_cells(domain, entities, decision, action, transition)
    try:
        raw = transition.distribution_hook(
            snapshot,
            decision,
            action,
            transition.parameters,
        )
    except Exception as error:
        raise WorldTransitionError("stochastic action transition hook failed") from error
    if not isinstance(raw, StateDeltaDistribution):
        raise WorldTransitionError("stochastic action transition hook must return StateDeltaDistribution")
```

Canonicalize/validate **every** candidate delta with `_validated_delta(...)`, reconstruct a validated `StateDeltaDistribution`, attest the stochastic spec hash, then call:

```python
sample = sample_categorical(
    root_seed=prior.root_seed,
    namespace="world.transition",
    step_index=prior.step_index + 1,
    source_hash=prior.content_hash,
    component_hash=transition_hash,
    component_key=(decision.actor_id, decision.id, action.id),
    distribution_hash=validated_distribution.content_hash,
    outcomes=tuple(
        (item.outcome_id, item.probability, item.content_hash)
        for item in validated_distribution.outcomes
    ),
)
```

Construct the concrete `ActionTransitionRecord` from the selected delta and `StochasticTransitionSample`.

- [ ] **Step 6: Add world-step stochastic lineage certification**

In `WorldStepResult.__post_init__`, for every record with `stochastic_sample`, require:

```python
sample = record.stochastic_sample.sample_record
if self.prior_state.root_seed is None or sample.root_seed != self.prior_state.root_seed:
    raise ValueError("stochastic transition root seed mismatch")
if sample.step_index != self.next_state.step_index:
    raise ValueError("stochastic transition step mismatch")
if sample.source_hash != self.prior_state.content_hash:
    raise ValueError("stochastic transition source mismatch")
if sample.component_hash != record.transition_spec_hash:
    raise ValueError("stochastic transition spec mismatch")
if sample.component_key != (
    record.actor_id,
    record.intent.decision_id,
    record.action.id,
):
    raise ValueError("stochastic transition component key mismatch")
```

This makes constructor-bypassing/forged lineage fail before projection or scheduler success.

- [ ] **Step 7: Keep conflict resolution on realized writes only**

Do not modify conflict resolver hooks or records. `_conflict_components` continues to inspect each concrete `record.delta`. Verify resolved batch hashing naturally binds stochastic sample payload through `record.to_dict()`.

- [ ] **Step 8: Run world GREEN tests**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_randomness \
  tests.test_narrative_world_transition \
  tests.test_narrative_conflict_resolution \
  tests.test_narrative_conflict_resolution_safety \
  tests.test_narrative_stochastic_world -v
```

Expected: all PASS. Do not loosen distribution, capability, replay, order, or conflict assertions.

- [ ] **Step 9: Commit stochastic world GREEN**

```bash
git add narrative_dynamics/narrative/world.py
git commit -m "feat: add stochastic narrative world transitions"
```

---

### Task 4: Add Truth-Preserving Stochastic Observation Projection

**Files:**
- Modify: `narrative_dynamics/narrative/observation_projection.py`
- Test: `tests/test_narrative_stochastic_observation.py`

**Interfaces:**
- Consumes seeded `WorldStepResult` and `sample_categorical`.
- Produces `ObservationOutcome`, `ObservationOutcomeDistribution`, `StochasticObservationSample`, `StochasticObserverProjectionSpec`, additive `ObservationProjectionModelSpec.stochastic_projections`, additive `ProjectedObservation.stochastic_sample_hash`, and additive `ObservationProjectionResult.stochastic_samples`.
- Leaves deterministic `ObserverProjectionSpec` and `project_world_observations(...)` signature unchanged.

- [ ] **Step 1: Add observation distribution records**

Add:

```python
@dataclass(frozen=True)
class ObservationOutcome:
    outcome_id: str
    probability: float
    facts: tuple[ObservationFact, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "outcome_id": self.outcome_id,
            "probability": self.probability,
            "facts": [fact.to_dict() for fact in self.facts],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class ObservationOutcomeDistribution:
    outcomes: tuple[ObservationOutcome, ...]
```

Apply the same finite-distribution rules as Task 3. Within one `ObservationOutcome`, require an exact tuple of `ObservationFact`, unique cells, and canonical fact ordering by `_cell_key`; the empty tuple is legal.

- [ ] **Step 2: Add stochastic projection spec identity**

Add:

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

Reuse deterministic observer/capability containment validation. Freeze parameters identically to stochastic world parameters. `to_dict()` includes capabilities, parameters, `random_derivation_version`, and implementation identity.

- [ ] **Step 3: Add stochastic projections to model identity without changing deterministic identity**

Append:

```python
stochastic_projections: tuple[StochasticObserverProjectionSpec, ...] = ()
```

Validate `(observer_type, channel)` uniqueness across both deterministic and stochastic lanes. Emit `stochastic_projections` only when non-empty.

- [ ] **Step 4: Add projection sample records and selected-observation links**

Add:

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

Require exact consistency between these source fields and the generic sample record (`namespace == "observation.projection"`, step/source/component/component key, distribution hash, selected outcome hash).

Append to `ProjectedObservation`:

```python
stochastic_sample_hash: str | None = None
```

Serialize only when non-`None`.

Append to `ObservationProjectionResult`:

```python
stochastic_samples: tuple[StochasticObservationSample, ...] = ()
```

Canonicalize samples by `(observer_id, channel)`. Serialize only when non-empty.

- [ ] **Step 5: Prevalidate every candidate observation outcome before sampling**

Refactor the deterministic fact checks into a helper that can validate a candidate tuple without constructing output records:

```python
def _validate_candidate_facts(
    facts,
    *,
    domain,
    entities,
    world_step,
    observer,
    spec,
) -> tuple[ObservationFact, ...]:
```

For every fact in every candidate outcome, enforce existing rules exactly:

```text
canonical subject/type/state variable
inside emit capability
`equals` value == exact post-step truth
`clear` == post-step absence + explicit effective current-step clear
unique cell within outcome
```

A malformed unsampled outcome fails the whole projection.

- [ ] **Step 6: Execute stochastic projection with one observer/channel-local sample**

For each stochastic spec/observer:

```python
if world_step.next_state.root_seed is None:
    raise ObservationProjectionError("stochastic observation projection requires a root seed")
```

Require this before calling the hook. Build the same immutable `prior_visible`/`next_visible` views as deterministic projection. Call the hook with frozen parameters, validate/canonicalize the complete distribution, attest the stochastic spec hash, then sample:

```python
sample = sample_categorical(
    root_seed=world_step.next_state.root_seed,
    namespace="observation.projection",
    step_index=world_step.next_state.step_index,
    source_hash=world_step.content_hash,
    component_hash=spec_hash,
    component_key=(observer.id, spec.channel),
    distribution_hash=distribution.content_hash,
    outcomes=tuple(
        (item.outcome_id, item.probability, item.content_hash)
        for item in distribution.outcomes
    ),
)
```

Create one `StochasticObservationSample`, then emit only selected facts as normal `ProjectedObservation` with `stochastic_sample_hash=sample_wrapper.content_hash`. For an empty selected fact tuple, emit zero observations but keep the sample wrapper in the result.

- [ ] **Step 7: Certify result/sample/observation lineage**

In `ObservationProjectionResult.__post_init__`, require:

```text
sample source_world_step_hash == result.source_world_step_hash
sample source_world_state_hash == result.source_world_state_hash
sample step_index == result.step_index
sample ids unique by observer/channel
all non-null ProjectedObservation.stochastic_sample_hash values exist in result samples
referenced observation fact belongs to that sample's selected outcome
observations without stochastic_sample_hash are deterministic
```

Do not make a sample bind the enclosing projection-result hash.

Also extend `_validate_source_world_step` to reject forged stochastic transition lineage using the same root/step/source/spec/component-key checks enforced by `WorldStepResult`, because projection must fail closed even if public record constructors were bypassed.

- [ ] **Step 8: Run projection GREEN tests**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_observation_projection \
  tests.test_narrative_conflict_projection \
  tests.test_narrative_stochastic_world \
  tests.test_narrative_stochastic_observation -v
```

Expected: all projection/world tests PASS except the stochastic observation test that explicitly inspects runtime evidence-batch sample hashes; that one remains RED until Task 5.

- [ ] **Step 9: Commit stochastic observation GREEN**

```bash
git add narrative_dynamics/narrative/observation_projection.py
git commit -m "feat: add stochastic narrative observation projection"
```

---

### Task 5: Carry Stochastic Projection Lineage Through Runtime Perception

**Files:**
- Modify: `narrative_dynamics/narrative/runtime_perception.py`
- Test: `tests/test_narrative_stochastic_observation.py`

**Interfaces:**
- Consumes `ProjectedObservation.stochastic_sample_hash` and `ObservationProjectionResult.stochastic_samples`.
- Produces optional evidence-level sample reference and complete projection-sample hashes in `RuntimeEvidenceBatch`.
- Does not change `RuntimePerceptView` or cognition likelihood APIs.

- [ ] **Step 1: Append optional stochastic lineage to runtime evidence**

Append after existing `projection_spec_hash`:

```python
projection_sample_hash: str | None = None
```

In `RuntimeEpistemicEvidence.__post_init__`, validate it with `_hash(...)` only when non-`None`. In `to_dict()`, conditionally emit:

```python
if self.projection_sample_hash is not None:
    payload["projection_sample_hash"] = self.projection_sample_hash
```

`percept_view` stays unchanged and therefore remains provenance-free.

- [ ] **Step 2: Append complete projection sample hashes to every evidence batch**

Append after existing `evidence`:

```python
projection_sample_hashes: tuple[str, ...] = ()
```

Validate/canonicalize unique hashes. Serialize only when non-empty.

For each evidence item with a non-null `projection_sample_hash`, require it belongs to `projection_sample_hashes`.

- [ ] **Step 3: Copy projection sampling lineage during admission**

In `admit_world_percepts`, compute:

```python
projection_sample_hashes = tuple(
    sorted(item.content_hash for item in projection_result.stochastic_samples)
)
```

When constructing `RuntimeEpistemicEvidence`, pass:

```python
projection_sample_hash=projected.stochastic_sample_hash,
```

When constructing `RuntimeEvidenceBatch`, pass all sample hashes even if `runtime_evidence == ()`.

- [ ] **Step 4: Certify exact projection/batch sample equality**

In `RuntimePerceptAdmissionResult.__post_init__`, compute expected sample hashes from `projection_result.stochastic_samples` and require exact equality with `evidence_batch.projection_sample_hashes`.

This is the lock that preserves stochastic blackout lineage when no percept exists.

- [ ] **Step 5: Run perception + stochastic observation GREEN tests**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_runtime_perception \
  tests.test_narrative_runtime_cognition \
  tests.test_narrative_stochastic_observation -v
```

Expected: all PASS. Existing cognition tests must see unchanged `RuntimePerceptView` payloads.

- [ ] **Step 6: Commit perception lineage GREEN**

```bash
git add narrative_dynamics/narrative/runtime_perception.py
git commit -m "feat: bind stochastic projection lineage to runtime evidence"
```

---

### Task 6: Freeze the Root Seed Through Multi-Step Simulation

**Files:**
- Modify: `narrative_dynamics/narrative/simulation.py`
- Modify: `narrative_dynamics/narrative/runtime_perception.py`
- Test: `tests/test_narrative_stochastic_simulation.py`

**Interfaces:**
- Extends only initialization APIs with optional seed plumbing.
- `simulate_step(...)` and `simulate_trajectory(...)` remain unchanged and derive stochastic execution solely from `prior_state.world_state.root_seed`.
- Decision dispatch remains seedless.

- [ ] **Step 1: Extend runtime ledger initialization to reconstruct the same seeded initial world**

Change:

```python
def runtime_evidence_ledger_from_story(
    story: GenericNarrative,
    domain: DomainSpec,
    *,
    at_time: int | None = None,
    seed: int | None = None,
) -> RuntimeEvidenceLedger:
```

Construct:

```python
initial = world_state_from_story(story, domain, at_time=cutoff, seed=seed)
```

No independent seed field is added to `RuntimeEvidenceLedger`; seed identity remains transitive through `initial_world_state_hash/current_world_state_hash`.

- [ ] **Step 2: Extend simulation initialization and nothing else**

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

Call:

```python
world = world_state_from_story(story, domain, at_time=cutoff, seed=seed)
ledger = runtime_evidence_ledger_from_story(story, domain, at_time=cutoff, seed=seed)
```

Require the resulting ledger/world hashes to match through existing `SimulationState` validation.

Do not add `seed` parameters to `simulate_step` or `simulate_trajectory`.

- [ ] **Step 3: Keep decision dispatch independent from the root seed**

Do not alter this existing call shape:

```python
result = run_runtime_decision(
    story,
    domain,
    agent.decision_template_id,
    prior_state.evidence_ledger,
    agent.decision_model,
)
```

The seed may affect a later decision only through changed world/projection/evidence lineage, never as a direct decision input.

- [ ] **Step 4: Run exact replay and separation tests**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_simulation \
  tests.test_narrative_runtime_decision_dispatch \
  tests.test_narrative_runtime_reactive \
  tests.test_narrative_runtime_intention \
  tests.test_narrative_runtime_planning \
  tests.test_narrative_stochastic_simulation -v
```

Expected: all PASS, including same-seed exact multi-round trajectory replay and unchanged decision public signatures.

- [ ] **Step 5: Commit scheduler seed plumbing GREEN**

```bash
git add \
  narrative_dynamics/narrative/runtime_perception.py \
  narrative_dynamics/narrative/simulation.py
git commit -m "feat: freeze narrative stochastic seed across trajectories"
```

---

### Task 7: Prove Existing Seed Diagnostics Compatibility and Close Feature Verification

**Files:**
- Test: `tests/test_narrative_stochastic_seed_diagnostics.py`
- No production uncertainty/comparison changes.

**Interfaces:**
- Consumes the completed stochastic runtime and existing outer `SimulationRunner`/`calibrate_seed_block_variants` APIs.
- Produces final proof that external experiment seeds can deterministically drive inner narrative aleatoric branches and existing seed diagnostics observe the variation.

- [ ] **Step 1: Make the test-only outer adapter fully GREEN**

Keep the adapter entirely inside `tests/test_narrative_stochastic_seed_diagnostics.py`. It must consume exactly one `rng.getrandbits(64)` call before narrative initialization and must expose `narrative_root_seed` plus a stochastic lineage hash in `ModelRun.outcome`.

No production adapter or registry entry is added.

- [ ] **Step 2: Verify outer seed manifests and narrative lineage**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_stochastic_seed_diagnostics.NarrativeStochasticSeedDiagnosticsTests.test_outer_seed_deterministically_maps_to_narrative_root_seed_and_manifest -v
```

Assert two runs with the same outer seed have the same outer trace seed, same derived narrative root seed, same inner lineage hash, and same manifest hash. A different outer seed must change the outer trace seed and normally changes the derived narrative root seed; the test must not claim the 64-bit mapping is mathematically injective.

- [ ] **Step 3: Verify existing seed-block diagnostics detect aleatoric drift**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_stochastic_seed_diagnostics.NarrativeStochasticSeedDiagnosticsTests.test_existing_seed_block_variants_detect_aleatoric_acceptance_drift -v
```

Expected: PASS with two seed variants retaining different accepted parameter values and an empty accepted intersection.

- [ ] **Step 4: Run all new stochastic tests**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_randomness \
  tests.test_narrative_stochastic_world \
  tests.test_narrative_stochastic_observation \
  tests.test_narrative_stochastic_simulation \
  tests.test_narrative_stochastic_seed_diagnostics -v
```

Expected: all PASS.

- [ ] **Step 5: Run the full Python suite**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: `OK`. No existing deterministic test may be skipped, weakened, or converted to a stochastic expectation.

- [ ] **Step 6: Verify approved production/test scope against the integrated base**

Run:

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

The approved spec and plan are the only allowed doc additions for this workstream.

- [ ] **Step 7: Capture exact-head feature GREEN proof**

Record the exact branch SHA and require the `proof` workflow for that exact SHA to reach `completed / success` with:

```text
Lean dependency resolution == success
Lean/Python conformance == success
full Lean build == success
Lean theorem tests == success
Python numerical/tests == success
Narrative story theorem tests == success
Narrative testimony theorem tests == success
```

Read the Python job log and record the exact `Ran N tests ... OK` line. Do not infer GREEN from a prior head.

- [ ] **Step 8: Review the complete diff before PR**

Confirm:

```text
no RNG in deterministic hooks
no RNG in decision APIs
no stochastic conflict resolver
no false observation values
all unsampled candidates validated
root seed frozen in world chain
empty observation samples retained in ledger lineage
world/observation namespaces distinct
deterministic optional fields omitted
no uncertainty/comparison/Lean changes
```

- [ ] **Step 9: Open the PR with RED/GREEN evidence and explicit P2 exclusions**

PR body must include:

```text
integrated base SHA
spec + plan paths
authoritative test-only RED SHA + proof run + failure signature
final feature SHA + exact-head proof run
same-seed replay evidence
deterministic V1/V2 compatibility evidence
seed diagnostics evidence
production/test scope
explicit statement that P2 Identification and Model Comparison remains open
```

- [ ] **Step 10: Require PR synthetic-merge GREEN before merge**

Verify the PR-triggered proof run is for the current head/base synthetic merge and is `completed / success`. Do not use the feature-head run as a substitute.

- [ ] **Step 11: Guarded merge and post-merge exact-head proof**

Merge only with `expected_head_sha=<current feature head>`. Then require a push-triggered proof on the resulting exact `proof/narrative-dynamics-v0` merge commit to reach `completed / success`, and read its Python `Ran N tests ... OK` evidence.

- [ ] **Step 12: Update roadmap #27 only after post-merge GREEN**

Change only these five P2 Stochastic World / Observation items to `[x]`:

```text
Explicit RNG/seed boundary
Seed included in transition/observation lineage
Replay exactness under fixed seed
Separate aleatoric transition noise from decision stochasticity
Batch/seed diagnostics compatible with existing uncertainty infrastructure
```

Update the integrated baseline/evidence paragraph with PR, merge commit, and post-merge proof. Keep issue #27 open and keep every P2 Identification and Model Comparison checkbox unchanged.
