# Generic Narrative Engine V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, domain-pluggable structured narrative research engine that compiles audited candidate semantics into a typed event-sourced IR, replays objective/direct/communicated evidence, evaluates capability-limited decision mechanisms, and compares baseline, mechanism, and single-variable counterfactual trajectories.

**Architecture:** Add a new `narrative_dynamics.narrative` package alongside the frozen legacy `narrative_dynamics.story` implementation. Generic semantics are driven by immutable typed IR plus a hashed `DomainSpec`; domain hooks may only emit validated objective-state deltas. Compiler provenance, canonical narrative identity, decision-model identity, and analysis lineage remain separate, while Testimony V2 is retained as a conformance reference through an adapter.

**Tech Stack:** Python standard library, existing `narrative_dynamics.contracts.stable_content_hash`, existing `narrative_dynamics.attestation.measure_implementation`, `unittest`, existing GitHub Actions `proof` workflow, existing Lean 4 regressions.

**Spec:** `docs/superpowers/specs/2026-08-24-generic-narrative-engine-v1-design.md`

## Global Constraints

- Work only on `work/generic-narrative-engine-v1`; implementation base is exact research SHA `82096098dda907aec04330b5ddf357ac6f753d2a`.
- Do not move `proof/narrative-dynamics-v0` until Task 13. Never target or move `master` as part of this feature.
- Generic APIs live under `narrative_dynamics.narrative`; package root `narrative_dynamics` gains no Generic Narrative exports.
- Existing `narrative_dynamics.story` V1/V2/V3 schemas, fixture hashes, runtime IDs, payloads, APIs, and behavior remain unchanged.
- LLM/NLP may exist outside this engine only as candidate extractors. Compiler/replay/decision/analysis code performs no LLM/NLP/network call.
- Source text, spans, extractor identity, confidence, and human resolution metadata remain compiler provenance and never enter canonical runtime semantics.
- Canonical narrative data is immutable, deterministic, JSON-serializable, and hashed only through existing `stable_content_hash`.
- Timed events, claims, and decisions use one strict total logical order in V1.
- Objective state, direct evidence, and communicated evidence are separate paths. No direct or epistemic replay may fall back to objective state.
- Claim support is provenance, never a truth predicate. `not_equals` remains a constraint and never infers a unique value.
- Decision models receive an immutable capability-limited `DecisionContext`, never a raw `GenericNarrative`; V1 has no RNG in the decision API.
- Counterfactuals change exactly one upstream canonical variable and recompute validation → replay → decision → analysis. No arbitrary patch or intervention powerset.
- Semantic hooks are callable class instances so `measure_implementation()` can bind their module bytes into identity.
- No new Lean semantics. Every GREEN checkpoint must preserve the full existing Lean build/theorem suite.
- Every production increment uses test-only RED commit → observed exact-head CI RED → minimal GREEN commit → fresh exact-head full `proof` success.
- Open one draft PR from `work/generic-narrative-engine-v1` to `proof/narrative-dynamics-v0` at Task 1 RED; keep it draft through Task 13.
- Claim standalone `python3 -m compileall -q narrative_dynamics` only if independently executed and observed successful.

## File Map

```text
narrative_dynamics/narrative/
  __init__.py
  ir.py
  domain.py
  replay.py
  decision.py
  analysis.py
  interventions.py
  compiler.py
  trust.py
  compat_story_v2.py
  domains/
    __init__.py
    service_incident.py

tests/
  narrative_test_support.py
  narrative_compiler_test_support.py
  test_narrative_ir.py
  test_narrative_domain.py
  test_narrative_validation.py
  test_narrative_replay.py
  test_narrative_decision.py
  test_narrative_analysis.py
  test_narrative_interventions.py
  test_narrative_compat_story_v2.py
  test_narrative_service_incident.py
  test_narrative_compiler_records.py
  test_narrative_compiler.py
  test_narrative_trust_api.py
```

---

### Task 1: Canonical Generic IR

**Files:** Create `narrative_dynamics/narrative/__init__.py`, `narrative_dynamics/narrative/ir.py`, `tests/test_narrative_ir.py`.

**Produces:** `EntityRef`, `StateCellRef`, `TypedValue`, `Entity`, `NarrativeEvent`, `Observation`, `Proposition`, `Claim`, `Reception`, `ActionOption`, `Decision`, `GenericNarrative`, `GENERIC_NARRATIVE_SCHEMA_VERSION = 1`.

- [ ] **Step 1: Write RED test**

```python
from __future__ import annotations
import json
import unittest
from narrative_dynamics.narrative.ir import (
    ActionOption, Claim, Decision, Entity, EntityRef, GenericNarrative,
    NarrativeEvent, Observation, Proposition, Reception, StateCellRef, TypedValue,
)

class GenericNarrativeIRTests(unittest.TestCase):
    def test_records_are_immutable_and_hash_stable(self):
        service = EntityRef("svc", "Service")
        story = GenericNarrative(
            domain_id="service-incident", domain_version="1",
            domain_spec_hash="sha256:" + "1" * 64,
            entities=(Entity("svc", "Service"), Entity("bob", "Agent")),
            events=(NarrativeEvent("e1", 1, "SetHealth", None, {
                "service": TypedValue("ServiceRef", service),
                "health": TypedValue("HealthState", "failed"),
            }),),
            observations=(Observation("o1", "bob", "e1"),),
            claims=(Claim("c1", 2, "bob", Proposition(
                service, "service.health", "equals", TypedValue("HealthState", "failed")
            ), ("e1",)),),
            receptions=(Reception("r1", "c1", "bob"),),
            decisions=(Decision("d1", 3, "bob", "service-response",
                (StateCellRef(service, "service.health"),),
                (ActionOption("restart", "service-action", {}),)),),
        )
        payload = story.to_dict()
        self.assertEqual(json.loads(json.dumps(payload, sort_keys=True)), payload)
        self.assertEqual(story.content_hash, story.content_hash)
        with self.assertRaises(TypeError):
            story.events[0].arguments["health"] = TypedValue("HealthState", "healthy")

if __name__ == "__main__": unittest.main()
```

- [ ] **Step 2: Commit RED and observe exact-head failure**

```bash
git add tests/test_narrative_ir.py
git commit -m "test: define generic narrative IR RED"
git push origin work/generic-narrative-engine-v1
```

Expected Python failure is the missing `narrative_dynamics.narrative.ir` module only; earlier Lean gates remain green. Open the draft PR here.

- [ ] **Step 3: Implement minimal IR**

Use trimmed non-empty text validation, non-negative integer logical times, tuples, and `MappingProxyType`. `TypedValue.value` accepts only `bool | int | str | EntityRef`; proposition relation accepts exactly `equals` or `not_equals`. Every record exposes `to_dict()`.

Core definitions:

```python
GENERIC_NARRATIVE_SCHEMA_VERSION = 1
_ALLOWED_RELATIONS = frozenset({"equals", "not_equals"})

@dataclass(frozen=True)
class EntityRef:
    entity_id: str
    entity_type: str
    def to_dict(self): return {"entity_id": self.entity_id, "entity_type": self.entity_type}

@dataclass(frozen=True)
class TypedValue:
    type_name: str
    value: bool | int | str | EntityRef
    def to_dict(self):
        value = self.value.to_dict() if isinstance(self.value, EntityRef) else self.value
        return {"type_name": self.type_name, "value": value}
```

`GenericNarrative.to_dict()` contains exactly `schema_version`, `domain_id`, `domain_version`, `domain_spec_hash`, `entities`, `events`, `observations`, `claims`, `receptions`, `decisions`, and:

```python
@property
def content_hash(self) -> str:
    return stable_content_hash(self.to_dict())
```

- [ ] **Step 4: Focused GREEN**

```bash
python3 -m unittest tests.test_narrative_ir -v
```

- [ ] **Step 5: Commit GREEN and require fresh full proof**

```bash
git add narrative_dynamics/narrative/__init__.py narrative_dynamics/narrative/ir.py tests/test_narrative_ir.py
git commit -m "feat: add canonical generic narrative IR"
git push origin work/generic-narrative-engine-v1
```

Do not begin Task 2 until an exact-head full `proof` succeeds.

---

### Task 2: DomainSpec and Objective Transition Hooks

**Files:** Create `narrative_dynamics/narrative/domain.py`, `tests/test_narrative_domain.py`.

**Produces:** `EntityTypeSpec`, `ValueTypeSpec`, `ParameterSpec`, `StateVariableSpec`, `StateEffectSpec`, `EventTypeSpec`, `ActionTypeSpec`, `DecisionTypeSpec`, `StateDeltaOp`, `StateDelta`, `SemanticHookBinding`, `DomainSpec`.

- [ ] **Step 1: Write RED test**

```python
class SetHealthHook:
    def __call__(self, prior_state, event):
        service = event.arguments["service"].value
        return StateDelta((StateDeltaOp("set", service.entity_id, "service.health", event.arguments["health"]),))

class IllegalHook:
    def __call__(self, prior_state, event):
        return StateDelta((StateDeltaOp("set", "svc", "service.secret", TypedValue("HealthState", "failed")),))

class DomainSpecTests(unittest.TestCase):
    def test_hook_identity_and_delta_boundary(self):
        domain = make_domain(SetHealthHook())
        self.assertTrue(domain.content_hash.startswith("sha256:"))
        event = NarrativeEvent("e1", 1, "SetHealth", None, {
            "service": TypedValue("ServiceRef", EntityRef("svc", "Service")),
            "health": TypedValue("HealthState", "failed"),
        })
        entities = {"svc": Entity("svc", "Service")}
        self.assertEqual(domain.apply_event({}, event, entities).operations[0].state_variable, "service.health")
        with self.assertRaisesRegex(ValueError, "undeclared state variable"):
            make_domain(IllegalHook()).apply_event({}, event, entities)
```

`make_domain()` constructs: entity `Service`; value types `ServiceRef=entity_ref(Service)` and `HealthState=enum(healthy,failed,recovered)`; state variable `service.health`; event `SetHealth(service, health)` with effect `service.health` whose subject argument is `service`; `health` is intervenable; hook name is `set_health`.

- [ ] **Step 2: Commit RED / observe missing domain module**

```bash
git add tests/test_narrative_domain.py
git commit -m "test: define generic domain semantics RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 3: Implement domain declarations and hook identity**

Allowed value kinds are exactly `enum`, `bool`, `integer`, `text`, `entity_ref`; delta kinds exactly `set`, `clear`. Every declaration/delta record exposes deterministic `to_dict()`.

```python
@dataclass(frozen=True)
class SemanticHookBinding:
    name: str
    declared_purpose: str
    hook: object = field(compare=False, repr=False)
    @property
    def implementation_hash(self) -> str:
        return measure_implementation(self.hook).content_hash
```

`DomainSpec.content_hash` hashes declarations plus ordered `{hook_name, declared_purpose, implementation_hash}` records. `apply_event()` validates event type/arguments/actor, invokes exactly the declared hook, requires `StateDelta`, and rejects operations outside declared effects, wrong subjects, wrong value types, `clear` carrying a value, or `set` lacking a value.

- [ ] **Step 4: GREEN / commit / exact-head proof**

```bash
python3 -m unittest tests.test_narrative_domain -v
git add narrative_dynamics/narrative/domain.py tests/test_narrative_domain.py
git commit -m "feat: add typed narrative domain semantics"
git push origin work/generic-narrative-engine-v1
```

Require fresh full `proof` success.

---

### Task 3: Canonical Narrative Validation

**Files:** Modify `domain.py`; create `tests/narrative_test_support.py`, `tests/test_narrative_validation.py`.

**Produces:** `validate_narrative(story: GenericNarrative, domain: DomainSpec) -> None` plus reusable test helpers `make_test_domain()`, `make_test_story(...)`, `target_cell()`.

- [ ] **Step 1: Create reusable test support**

`make_test_domain()` declares `Agent`, `Service`, `ServiceRef`, `HealthState`, `service.health`, `SetHealth`, `service-action`, and `service-response`. `make_test_story()` creates e1@1 failed observed by Bob, e2@2 recovered observed by Alice, c1@3 Alice claim received by Bob, d1@4 Bob decision. Optional arguments allow claim relation/value, reception removal, Bob observation removal, Alice support observation removal, and a second Carol claim at t4 with decision shifted to t5.

Exact target helper:

```python
def target_cell():
    return StateCellRef(EntityRef("svc", "Service"), "service.health")
```

- [ ] **Step 2: Write RED tests**

```python
class NarrativeValidationTests(unittest.TestCase):
    def test_support_requires_speaker_access_but_not_truth(self):
        domain = make_test_domain()
        story = make_test_story()
        stale = replace(story.claims[0], proposition=replace(
            story.claims[0].proposition, value=TypedValue("HealthState", "healthy")
        ))
        validate_narrative(replace(story, claims=(stale,)), domain)
        with self.assertRaisesRegex(ValueError, "speaker.*support"):
            validate_narrative(make_test_story(alice_observes_recovery=False), domain)

    def test_times_are_globally_unique(self):
        domain = make_test_domain(); story = make_test_story()
        colliding = replace(story.claims[0], logical_time=1)
        with self.assertRaisesRegex(ValueError, "logical times must be globally unique"):
            validate_narrative(replace(story, claims=(colliding,)), domain)
```

- [ ] **Step 3: Commit RED / observe missing validator**

```bash
git add tests/narrative_test_support.py tests/test_narrative_validation.py
git commit -m "test: define canonical narrative validation RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 4: Implement exact validator**

First require exact domain ID/version/hash match. Then validate: unique entity/event/claim/decision/observation/reception IDs; declared entity/event/value/state/action/decision types; globally unique event/claim/decision times; observation refs and unique `(event,agent)`; claim speaker/subject/proposition type; non-empty supports; event support earlier plus speaker observation; prior-claim support earlier plus speaker reception; reception refs and unique `(claim,recipient)`; decision actor type/context/action schema and non-empty unique actions. Never compare claim value to objective truth.

- [ ] **Step 5: GREEN / commit / exact-head proof**

```bash
python3 -m unittest tests.test_narrative_validation -v
git add narrative_dynamics/narrative/domain.py tests/narrative_test_support.py tests/test_narrative_validation.py
git commit -m "feat: validate canonical generic narratives"
git push origin work/generic-narrative-engine-v1
```

Require fresh full `proof` success.

---

### Task 4: Objective, Direct, and Epistemic Replay

**Files:** Create `replay.py`, `tests/test_narrative_replay.py`.

**Produces:** `EpistemicEvidence`, `EpistemicCellView`, `EpistemicState`, `objective_state`, `direct_state`, `epistemic_state`.

- [ ] **Step 1: Write RED tests**

```python
class GenericReplayTests(unittest.TestCase):
    def test_no_objective_fallback(self):
        story=make_test_story(); domain=make_test_domain(); cell=target_cell()
        self.assertEqual(objective_state(story,domain,at_time=2)[cell], TypedValue("HealthState","recovered"))
        self.assertEqual(direct_state(story,domain,"bob",at_time=2).resolved_values[cell], TypedValue("HealthState","failed"))

    def test_received_and_unreceived_claims(self):
        cell=target_cell(); domain=make_test_domain()
        received=epistemic_state(make_test_story(),domain,"bob")
        self.assertEqual(received.cells[cell].evidence_kind,"testimony")
        unreceived=epistemic_state(make_test_story(receive=False),domain,"bob")
        self.assertEqual(unreceived.cells[cell].evidence_kind,"direct_perception")

    def test_not_equals_and_conflict(self):
        cell=target_cell(); domain=make_test_domain()
        constrained=epistemic_state(make_test_story(claim_relation="not_equals",bob_observes_failure=False),domain,"bob")
        self.assertEqual(constrained.cells[cell].status,"unknown")
        conflict=epistemic_state(make_test_story(second_claim_value="failed"),domain,"bob")
        self.assertEqual(conflict.cells[cell].status,"conflicted")
        self.assertIsNone(conflict.cells[cell].resolved_value)
```

- [ ] **Step 2: Commit RED / observe missing replay module**

```bash
git add tests/test_narrative_replay.py
git commit -m "test: define generic epistemic replay RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 3: Implement objective replay**

Call `validate_narrative`, apply event hooks in time order, and apply only returned set/clear deltas to `(subject,state_variable)` cells. Reject invalid `at_time`.

- [ ] **Step 4: Implement evidence replay and resolution**

Direct observation emits evidence only for that observed event's delta. Received claim emits testimony evidence only for its proposition and preserves support refs. `not_equals` is a constraint only. One later testimony can supersede older direct evidence. Two differing testimony equality values after the most recent direct equality produce `conflicted`; a later direct equality resolves again. `latest_evidence_by_state_cell` still records latest evidence; `resolved_values` contains only `resolved` cells. Freeze all outputs.

- [ ] **Step 5: GREEN / commit / proof**

```bash
python3 -m unittest tests.test_narrative_replay -v
git add narrative_dynamics/narrative/replay.py tests/test_narrative_replay.py
git commit -m "feat: add generic narrative replay"
git push origin work/generic-narrative-engine-v1
```

Require fresh full `proof` success.

---

### Task 5: Capability-Limited Deterministic Decision Models

**Files:** Create `decision.py`; modify `tests/narrative_test_support.py`; create `tests/test_narrative_decision.py`.

**Produces:** `EvidenceAccess`, `DecisionCellView`, `DecisionContext`, `DecisionChoice`, `DecisionModelSpec`, `DecisionResult`, `EpistemicResolutionError`, `DecisionResolutionError`, `require_resolved_cell`, `run_decision_model`.

- [ ] **Step 1: Add test hooks**

```python
class HealthActionHook:
    def __call__(self, context):
        value=require_resolved_cell(context,target_cell()).value
        if value=="failed": action="restart"
        elif value=="recovered": action="leave"
        else: raise DecisionResolutionError("resolved health does not map to a declared service response")
        return DecisionChoice(action, tuple(e.supporting_id for e in context.evidence_history), (target_cell(),))
```

Add `make_model(model_id, access, hook)` and `make_health_model(model_id, access)` to shared support.

- [ ] **Step 2: Write RED tests**

```python
class GenericDecisionTests(unittest.TestCase):
    def test_access_boundary(self):
        hook=RecordingFirstActionHook(); model=make_model("direct",EvidenceAccess.DIRECT_ONLY,hook)
        run_decision_model(make_test_story(),make_test_domain(),"d1",model)
        self.assertFalse(hasattr(hook.context,"story")); self.assertFalse(hasattr(hook.context,"objective_state"))
        self.assertTrue(all(e.evidence_kind=="direct_perception" for e in hook.context.evidence_history))

    def test_model_identity_binds_access(self):
        direct=make_model("same",EvidenceAccess.DIRECT_ONLY,RecordingFirstActionHook())
        epistemic=make_model("same",EvidenceAccess.EPISTEMIC,RecordingFirstActionHook())
        self.assertNotEqual(direct.content_hash,epistemic.content_hash)
```

Also test conflicted `DecisionCellView` raises `EpistemicResolutionError` and an undeclared action raises `DecisionResolutionError`.

- [ ] **Step 3: Commit RED / observe missing decision module**

```bash
git add tests/narrative_test_support.py tests/test_narrative_decision.py
git commit -m "test: define capability-limited decision RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 4: Implement contexts and identity**

`direct_only` gets direct cells/history; `epistemic` and `custom_epistemic` get actor epistemic cells/history; `omniscient` gets objective cells wrapped as resolved views and empty evidence history. No context stores story/domain/RNG.

```python
@property
def implementation_hash(self): return measure_implementation(self.decision_hook).content_hash
@property
def content_hash(self):
    return stable_content_hash({"model_id":self.model_id,"version":self.version,
      "supported_decision_types":self.supported_decision_types,"evidence_access":self.evidence_access.value,
      "parameter_schema":tuple(x.to_dict() for x in self.parameter_schema),"implementation_hash":self.implementation_hash})
```

`run_decision_model` validates exact decision type/parameters, invokes hook with context only, validates exactly one declared selected action, and emits deterministic one-hot scores/policy plus basis.

- [ ] **Step 5: GREEN / commit / proof**

```bash
python3 -m unittest tests.test_narrative_decision -v
git add narrative_dynamics/narrative/decision.py tests/narrative_test_support.py tests/test_narrative_decision.py
git commit -m "feat: add capability-limited narrative decisions"
git push origin work/generic-narrative-engine-v1
```

Require fresh full `proof` success.

---

### Task 6: Generic Analysis Scope, Trajectory, and Mechanism Comparison

**Files:** Create `analysis.py`; modify shared test support; create `tests/test_narrative_analysis.py`.

**Produces:** `AnalysisScope`, `TriggerRef`, `AgentEvolutionView`, `EvolutionSnapshot`, `NarrativeTrajectory`, `MechanismPairwiseComparison`, `MechanismComparison`, `NarrativeAnalysis`, `derive_analysis_scope`, `build_trajectory`, `analyze_narrative`.

- [ ] **Step 1: Add shared analysis cases**

`make_analysis_case()` returns test story/domain plus direct, epistemic, omniscient health models. `make_same_action_different_basis_case()` uses stale claim value `failed`, so direct and epistemic both choose restart but use different evidence bases.

- [ ] **Step 2: Write RED tests**

```python
class GenericAnalysisTests(unittest.TestCase):
    def test_scope(self):
        story,domain,direct,epistemic,omniscient=make_analysis_case()
        scope=derive_analysis_scope(story,domain,"d1")
        self.assertEqual(scope.tracked_agents,("bob","alice")); self.assertEqual(scope.snapshot_times,(1,2,3,4))

    def test_same_action_different_basis(self):
        story,domain,direct,epistemic=make_same_action_different_basis_case()
        pair=analyze_narrative(story,domain,"d1",direct,comparison_models=(epistemic,)).mechanism_comparison.pairwise[0]
        self.assertTrue(pair.action_equal); self.assertFalse(pair.basis_equal)
```

- [ ] **Step 3: Commit RED / observe missing analysis module**

```bash
git add tests/narrative_test_support.py tests/test_narrative_analysis.py
git commit -m "test: define generic narrative analysis RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 4: Implement scope/trajectory/mechanism comparison**

Scope starts with decision context cells and actor; adds speakers of relevant claims actually received by actor; relevant event cells come from `EventTypeSpec.effects`; snapshot times are relevant event times + relevant claim times + decision time, sorted/frozen. Every snapshot stores objective tracked cells and direct/epistemic views for tracked agents. Only decision time stores result. Empty trigger tuple is legal for later event-removal counterfactuals. Mechanism pairwise fields are exactly `action_equal` and `basis_equal`; `mechanism_uniqueness_claimed` is structurally `False`.

- [ ] **Step 5: GREEN / commit / proof**

```bash
python3 -m unittest tests.test_narrative_analysis -v
git add narrative_dynamics/narrative/analysis.py tests/narrative_test_support.py tests/test_narrative_analysis.py
git commit -m "feat: add generic narrative analysis"
git push origin work/generic-narrative-engine-v1
```

Require fresh full `proof` success.

---

### Task 7: Single-Variable Interventions and Counterfactual Rejection

**Files:** Create `interventions.py`, `tests/test_narrative_interventions.py`; modify `analysis.py`.

**Produces:** intervention kinds exactly `remove_event`, `change_event_argument`, `remove_observation`, `change_claim_value`, `remove_reception`; `Intervention`, `Divergence`, `CounterfactualResult`, `generate_minimal_interventions`, `evaluate_counterfactual`, `first_divergence`.

- [ ] **Step 1: Write RED tests**

```python
class GenericInterventionTests(unittest.TestCase):
    def test_kinds_and_reception_divergence(self):
        story,domain,direct,epistemic,omniscient=make_analysis_case(); scope=derive_analysis_scope(story,domain,"d1")
        items=generate_minimal_interventions(story,domain,scope)
        self.assertEqual({x.kind for x in items},{"remove_event","change_event_argument","remove_observation","change_claim_value","remove_reception"})
        baseline=build_trajectory(story,domain,scope,epistemic)
        intervention=next(x for x in items if x.kind=="remove_reception")
        result=evaluate_counterfactual(story,domain,scope,epistemic,baseline,intervention)
        self.assertEqual(result.status,"valid"); self.assertEqual(result.first_divergence.logical_time,3)
```

Add tests: removing Alice support observation → `narrative_validation`; changing claim to known but unmapped `healthy` → `decision_resolution`; changing one of two same-valued claims to conflict → `epistemic_resolution`.

- [ ] **Step 2: Commit RED / observe missing interventions module**

```bash
git add tests/test_narrative_interventions.py
git commit -m "test: define generic counterfactual RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 3: Implement legal application and deterministic generation**

`change_event_argument` changes one exact intervenable parameter only; `change_claim_value` changes proposition value only; remove operations target one exact ID; `remove_event` does not repair dependent refs. Generate finite alternates only for enum/bool values. Sort interventions deterministically by kind/time/target/field/hashed target value.

- [ ] **Step 4: Implement rejection and first-divergence pipeline**

Catch `ValueError` during intervention type/domain check as `domain_validation`; canonical validator `ValueError` as `narrative_validation`; `EpistemicResolutionError` as `epistemic_resolution`; `DecisionResolutionError` as `decision_resolution`; all other programming/runtime errors propagate. Counterfactual trajectory reuses baseline `AnalysisScope.snapshot_times`; compare objective/direct/epistemic canonical fields and final selected action only. Identical trajectory returns no divergence.

- [ ] **Step 5: GREEN / commit / proof**

```bash
python3 -m unittest tests.test_narrative_interventions -v
git add narrative_dynamics/narrative/interventions.py narrative_dynamics/narrative/analysis.py tests/test_narrative_interventions.py
git commit -m "feat: add generic narrative counterfactuals"
git push origin work/generic-narrative-engine-v1
```

Require fresh full `proof` success.

---

### Task 8: Testimony V2 Compatibility Adapter and Equivalence Locks

**Files:** Create `compat_story_v2.py`, `tests/test_narrative_compat_story_v2.py`.

**Produces:** `location_testimony_domain`, `adapt_testimony_story_v2`, `legacy_direct_search_model`, `legacy_epistemic_search_model`, `legacy_omniscient_search_model`. These are compatibility-module APIs only, not package `__all__` exports.

- [ ] **Step 1: Write RED equivalence tests**

For truthful and stale committed fixtures, compare legacy V3 baseline to generic path. Lock exact selected-action triples:

```python
expected={"truthful":("search_drawer","search_box","search_box"),
          "stale":("search_drawer","search_drawer","search_box")}
```

Also compare objective/direct/epistemic trajectories, evidence kind/support/source, remove-reception result, claim-content change result, direct-observation removal rejection, support-observation removal rejection, first divergence time, and `action_changed`. Run existing V1/V2 identity regressions unchanged.

- [ ] **Step 2: Commit RED / observe missing compatibility module**

```bash
git add tests/test_narrative_compat_story_v2.py
git commit -m "test: define Testimony V2 generic conformance RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 3: Implement static location domain and adapter**

Entity types are `Agent`, `Object`, `Location`; state `object.location` uses entity-ref `LocationRef`. Relocation hook validates authored continuity and emits one set. Deterministic IDs: observation `obs:<event>:<agent>`, reception `recv:<report>:<recipient>`. A report maps exactly to a generic `Claim` whose proposition is `object.location equals LocationRef` with support ref equal to legacy support event.

- [ ] **Step 4: Implement three compatibility models**

All use the same search decision hook; only evidence access differs: direct, epistemic, omniscient. Hook selects unique action whose location argument equals resolved location, otherwise raises `DecisionResolutionError`.

- [ ] **Step 5: GREEN + legacy regressions / commit / proof**

```bash
python3 -m unittest tests.test_narrative_compat_story_v2 tests.test_story_schema_v2 tests.test_story_scenario_v2 tests.test_story_replay_v2 tests.test_story_testimony_models tests.test_story_evolution_v2 -v
git add narrative_dynamics/narrative/compat_story_v2.py tests/test_narrative_compat_story_v2.py
git commit -m "feat: add Testimony V2 generic conformance"
git push origin work/generic-narrative-engine-v1
```

Require fresh full `proof` success. Legacy production files remain untouched.

---

### Task 9: Non-Location Service Incident Conformance Domain

**Files:** Create `domains/__init__.py`, `domains/service_incident.py`, `tests/test_narrative_service_incident.py`.

**Produces:** `service_incident_domain`, recovered/stale claim stories, and direct/epistemic/omniscient service models.

- [ ] **Step 1: Write RED tests**

Lock matched pair identical except c1 proposition value and exact action contrast:

```python
expected={"recovered":("restart_service","leave_running","leave_running"),
          "stale":("restart_service","restart_service","leave_running")}
```

Also lock stale claim support validity, remove-reception first divergence t3, recovered action change, stale no action change with changed evidence basis.

- [ ] **Step 2: Commit RED / observe missing domain**

```bash
git add tests/test_narrative_service_incident.py
git commit -m "test: define non-location narrative conformance RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 3: Implement exact Service Incident semantics**

Entity types `Agent`,`Service`; `HealthState=healthy|failed|recovered`; `service.health`; events `ServiceFailure` and `ServiceRecovery`; decision `service-response`; actions `restart_service`,`leave_running`. Failure sets failed from absent/healthy; recovery requires failed then sets recovered. Pair: e1@1 failure observed Bob, e2@2 recovery observed Alice, c1@3 Alice says recovered or failed supported by e2 and received Bob, d1@4. Decision rule: failed→restart, recovered→leave, other/unknown/conflicted→typed resolution rejection.

- [ ] **Step 4: GREEN / commit / proof**

```bash
python3 -m unittest tests.test_narrative_service_incident -v
git add narrative_dynamics/narrative/domains tests/test_narrative_service_incident.py
git commit -m "feat: add service incident conformance domain"
git push origin work/generic-narrative-engine-v1
```

Require fresh full `proof` success.

---

### Task 10: Source, Candidate, and Resolution Provenance Records

**Files:** Create `compiler.py`, `tests/narrative_compiler_test_support.py`, `tests/test_narrative_compiler_records.py`.

**Produces:** `SourceDocument`, `SourceSpan`, `SourceBundle`, `ExtractorIdentity`, `CandidateEntityRef`, `CandidateStateCellRef`, `CandidateValue`, `CandidateEntity`, `CandidateEvent`, `CandidateObservation`, `CandidateProposition`, `CandidateClaim`, `CandidateReception`, `CandidateActionOption`, `CandidateDecision`, `CandidateBundle`, `ResolutionRecord`.

- [ ] **Step 1: Create compiler test support**

`make_source_bundle(text)` returns one source document. `make_candidate_bundle(source, first_event_time=1, claim_value="recovered", confidence=0.9)` emits candidates for Bob/Alice/Service, SetHealth failure/recovery, Bob/Alice observations, Alice claim, Bob reception, Bob decision. Candidate references use candidate IDs (`ce-svc`, `cev1`, etc.), never final canonical IDs.

- [ ] **Step 2: Write RED record tests**

```python
class CompilerRecordTests(unittest.TestCase):
    def test_span_and_confidence(self):
        doc=SourceDocument("d1","service failed","text","memory:d1")
        span=SourceSpan.from_document(doc,0,7); SourceBundle((doc,)).validate_span(span)
        self.assertTrue(span.exact_text_hash.startswith("sha256:"))
        with self.assertRaisesRegex(ValueError,"confidence"):
            CandidateEntity("ce","svc","Service",(span,),float("nan"))
```

Also assert source/candidate/resolution hashes are distinct.

- [ ] **Step 3: Commit RED / observe missing compiler module**

```bash
git add tests/narrative_compiler_test_support.py tests/test_narrative_compiler_records.py
git commit -m "test: define narrative compiler provenance RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 4: Implement provenance records**

`SourceDocument.content_hash=stable_content_hash({"content":content})`; span hashes exact source slice and validates offsets. Candidate confidence must be finite in [0,1], every extracted candidate has source spans, nested data freezes. CandidateBundle requires unique candidate IDs and records extractor identity. Resolution decision is exactly accepted/rejected/unresolved and freezes selected fields; field whitelist is Task 11.

- [ ] **Step 5: GREEN / commit / proof**

```bash
python3 -m unittest tests.test_narrative_compiler_records -v
git add narrative_dynamics/narrative/compiler.py tests/narrative_compiler_test_support.py tests/test_narrative_compiler_records.py
git commit -m "feat: add narrative compiler provenance records"
git push origin work/generic-narrative-engine-v1
```

Require fresh full `proof` success.

---

### Task 11: Deterministic Narrative Compiler

**Files:** Modify `compiler.py`; create `tests/test_narrative_compiler.py`.

**Produces:** `CompilationDiagnostic`, `CompilationResult`, `compile_candidates`; statuses exactly canonical/incomplete/rejected.

- [ ] **Step 1: Write RED compiler tests**

Lock: missing first event time → incomplete/no scenario; accepted resolution `{logical_time:1}` → canonical; different source wording/confidence but same semantics → different provenance hashes and same canonical hash; conflicting communicated claims → canonical; unknown selected resolution field → rejected; forged source span → rejected; confidence/source/resolver metadata absent from canonical payload.

- [ ] **Step 2: Commit RED / observe missing compiler function**

```bash
git add tests/test_narrative_compiler.py
git commit -m "test: define deterministic narrative compiler RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 3: Implement exact resolution whitelist**

Allowed selected fields: entity `{entity_id,type_name}`; event `{logical_time,type_name,actor_candidate_id}`; observation `{agent_candidate_id,event_candidate_id}`; claim `{logical_time,speaker_candidate_id}`; reception `{claim_candidate_id,recipient_candidate_id}`; decision `{logical_time,actor_candidate_id,type_name}`. Unknown field/missing candidate resolution rejects. Explicit rejected candidate omitted. Explicit unresolved or accepted-but-required-None creates `unresolved_required_field` diagnostic and incomplete result. No resolution means accept extracted concrete fields.

- [ ] **Step 4: Implement dependency-order compile**

Compile entities→events→observations→claims→receptions→decisions. Resolve every `CandidateEntityRef` through final entity bindings. Multiple mentions may collapse to same entity ID only with identical type. Construct `GenericNarrative(domain_id,domain_version,domain.content_hash,...)`, call `validate_narrative`, and convert expected semantic `ValueError` to rejected result. Programming/attestation/runtime errors propagate.

- [ ] **Step 5: Implement identities**

Result records source bundle hash, candidate bundle hash, stable sorted resolution-bundle hash, and canonical scenario/hash only for canonical status. Canonical hash is only `GenericNarrative.content_hash`, excluding all provenance/diagnostics.

- [ ] **Step 6: GREEN / commit / proof**

```bash
python3 -m unittest tests.test_narrative_compiler_records tests.test_narrative_compiler -v
git add narrative_dynamics/narrative/compiler.py tests/test_narrative_compiler.py
git commit -m "feat: add deterministic narrative compiler"
git push origin work/generic-narrative-engine-v1
```

Require fresh full `proof` success.

---

### Task 12: Domain Trust, Analysis Artifact Lineage, and Final Public API

**Files:** Create `trust.py`; modify narrative `__init__.py`; create `tests/test_narrative_trust_api.py`.

**Produces:** `TrustedDomainPin`, `TrustedDomainPolicy`, `TrustedDomainBinding`, `NarrativeAnalysisArtifact`.

- [ ] **Step 1: Write RED trust/API tests**

Construct pin from external test code using `domain_id`, version, domain hash and `{hook.name:hook.implementation_hash}`. Policy bind must pass exact domain and reject version/domain/hook drift. Create two analysis artifacts with identical semantic fields and different compilation hashes; assert equal `analysis_identity_hash`, different lineage dictionaries. Assert exact narrative package `__all__` and root isolation.

- [ ] **Step 2: Commit RED / observe missing trust + exports**

```bash
git add tests/test_narrative_trust_api.py
git commit -m "test: define narrative trust and API RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 3: Implement external trust policy**

Pin freezes domain ID/version/spec hash/hook hash map. Policy sorts pins, rejects duplicates, and `bind(domain)` requires exact domain hash and exact ordered hook hashes, then returns binding. Domain alone cannot create a trusted binding.

- [ ] **Step 4: Implement analysis artifact**

Semantic identity payload contains exactly canonical narrative hash, domain spec hash, decision model identity, analysis scope hash, intervention manifest hash, analysis payload hash. `analysis_identity_hash=stable_content_hash(identity_payload)`. Optional `compilation_artifact_hash` exists only in lineage output, never semantic identity.

- [ ] **Step 5: Finalize package API**

`narrative_dynamics.narrative.__all__` contains canonical IR types; DomainSpec declaration/delta types and validator; replay types/functions; decision types/functions; analysis types/functions; intervention types/functions; compiler provenance/result types and `compile_candidates`; trust types. Do not export compatibility helpers, Service Incident helpers, implementation hook classes, or private helpers. Do not modify `narrative_dynamics/__init__.py`.

Exact root isolation test:

```python
for name in narrative.__all__:
    self.assertFalse(hasattr(narrative_dynamics,name),name)
```

- [ ] **Step 6: Full Python GREEN**

```bash
python3 -m unittest tests.test_narrative_ir tests.test_narrative_domain tests.test_narrative_validation tests.test_narrative_replay tests.test_narrative_decision tests.test_narrative_analysis tests.test_narrative_interventions tests.test_narrative_compat_story_v2 tests.test_narrative_service_incident tests.test_narrative_compiler_records tests.test_narrative_compiler tests.test_narrative_trust_api -v
python3 -m unittest discover -s tests -v
```

- [ ] **Step 7: Commit GREEN and require final exact-head full proof**

```bash
git add narrative_dynamics/narrative tests/narrative_* tests/test_narrative_*.py
git commit -m "feat: finalize Generic Narrative Engine V1"
git push origin work/generic-narrative-engine-v1
```

Require fresh exact-head `proof` success including Lean build, existing Lean theorem suite, full Python discovery, StoryState gate, and Testimony gate.

---

### Task 13: Final Review, Evidence, PR Update, and Research Integration

**Files:** No production changes expected. Documentation may be corrected only for factual mismatch; semantic changes require a new RED→GREEN increment.

- [ ] **Step 1: Read required completion skills**

At execution time use `requesting-code-review`, `verification-before-completion`, and `finishing-a-development-branch`. If no subagent executor exists, perform review inline.

- [ ] **Step 2: Review exact diff**

```bash
git diff --stat 82096098dda907aec04330b5ddf357ac6f753d2a...work/generic-narrative-engine-v1
git diff --name-only 82096098dda907aec04330b5ddf357ac6f753d2a...work/generic-narrative-engine-v1
```

Expected changed surface is the approved design, this plan, new `narrative_dynamics/narrative/**`, and new `tests/narrative_*` / `tests/test_narrative_*`. No Lean source, legacy story production source, runtime/registry/prison production source, or package-root export change.

- [ ] **Step 3: Boundary scan**

Review final diff for unique-mechanism claims, ToM/deception/population claims, objective fallback, RNG in Generic decision APIs, embedded LLM/API calls, source/confidence leakage into canonical payloads, raw story access by decision hooks, undeclared state writes, and multi-intervention APIs. Any real violation blocks integration.

- [ ] **Step 4: Fresh final verification evidence**

Record exact final feature SHA, proof run number/ID, full Lean build job count, Python test count/OK, StoryState success, Testimony success. If an isolated checkout exists, independently run `python3 -m compileall -q narrative_dynamics`; otherwise explicitly leave this as an unclaimed gap.

- [ ] **Step 5: Update draft PR**

Final title: `feat: add Generic Narrative Engine V1`.

PR body must describe typed event-sourced IR, deterministic compiler/provenance separation, objective/direct/communicated replay, capability-limited decisions, baseline/mechanism/counterfactual analysis, V2 equivalence, Service Incident conformance, exact verification evidence, root isolation, legacy identity preservation, no new Lean semantics, and scientific non-claims.

- [ ] **Step 6: Lock topology immediately before integration**

Capture final feature SHA, current research SHA, and current master SHA. Require research still exact `82096098dda907aec04330b5ddf357ac6f753d2a`, feature ahead and behind=0. If research moved, stop and rebase/revalidate; never force.

- [ ] **Step 7: Non-force fast-forward research only**

Move `proof/narrative-dynamics-v0` to final feature SHA with `force=false`. Do not call PR merge against master.

- [ ] **Step 8: Post-integration verification**

Require research ref == feature ref == final SHA; master unchanged from Step 6; feature branch retained. If GitHub auto-records the draft PR closed/merged because base=head, describe it as a research-ref fast-forward, not a master merge.

The strongest supported claim after integration is:

> Narrative Dynamics can execute a deterministic, domain-pluggable canonical narrative semantics in which objective events, direct observations, communicated claims, recipient-specific evidence, decision mechanisms, and single-variable counterfactuals remain separately represented, replayable, provenance-aware, and comparable across both the legacy testimony domain and an independent non-location conformance domain.

It still does not establish unique hidden-mechanism identification, general human Theory of Mind, deception intent, empirical human causal validity, or population-level validity.
