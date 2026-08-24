# Generic Narrative Engine V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, domain-pluggable structured narrative research engine that compiles audited candidate semantics into a typed event-sourced IR, replays objective/direct/communicated evidence, evaluates capability-limited decision mechanisms, and compares baseline, mechanism, and single-variable counterfactual trajectories.

**Architecture:** Add a new `narrative_dynamics.narrative` package alongside the frozen legacy `narrative_dynamics.story` implementation. Generic semantics are driven by immutable typed IR plus a hashed `DomainSpec`; domain hooks may only emit validated objective-state deltas. Compiler provenance, canonical narrative identity, decision-model identity, and analysis lineage remain separate, while Testimony V2 is retained as a conformance reference through an adapter.

**Tech Stack:** Python standard library (`dataclasses`, `enum`, `types.MappingProxyType`, `json`, `math`, `unittest`), existing `narrative_dynamics.contracts.stable_content_hash`, existing `narrative_dynamics.attestation.measure_implementation`, existing GitHub Actions `proof` workflow, existing Lean 4 regression suite.

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

**Interfaces:**

```text
GENERIC_NARRATIVE_SCHEMA_VERSION = 1
EntityRef(entity_id, entity_type)
StateCellRef(subject, state_variable)
TypedValue(type_name, value)
Entity(id, type_name)
NarrativeEvent(id, logical_time, type_name, actor_id, arguments)
Observation(id, agent_id, event_id, channel="direct")
Proposition(subject, state_variable, relation, value)
Claim(id, logical_time, speaker_id, proposition, support_refs)
Reception(id, claim_id, recipient_id, channel="direct_testimony")
ActionOption(id, type_name, arguments)
Decision(id, logical_time, actor_id, type_name, context_cells, actions)
GenericNarrative(domain_id, domain_version, domain_spec_hash, entities, events,
                 observations, claims, receptions, decisions, schema_version=1)
```

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

    def test_entity_reference_is_not_plain_text(self):
        entity = TypedValue("ServiceRef", EntityRef("svc", "Service"))
        text = TypedValue("ServiceRef", "svc")
        self.assertNotEqual(entity.to_dict(), text.to_dict())

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Commit RED and observe exact-head failure**

```bash
git add tests/test_narrative_ir.py
git commit -m "test: define generic narrative IR RED"
git push origin work/generic-narrative-engine-v1
```

Expected Python failure is the missing `narrative_dynamics.narrative.ir` module only; earlier Lean gates remain green. Open the draft PR here.

- [ ] **Step 3: Implement minimal IR**

Create package marker:

```python
# narrative_dynamics/narrative/__init__.py
"""Domain-pluggable canonical narrative research engine."""
```

Core validation/freeze pattern:

```python
GENERIC_NARRATIVE_SCHEMA_VERSION = 1
_ALLOWED_RELATIONS = frozenset({"equals", "not_equals"})

def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value

@dataclass(frozen=True)
class EntityRef:
    entity_id: str
    entity_type: str
    def __post_init__(self):
        object.__setattr__(self,"entity_id",_text(self.entity_id,label="entity ref id"))
        object.__setattr__(self,"entity_type",_text(self.entity_type,label="entity ref type"))
    def to_dict(self): return {"entity_id":self.entity_id,"entity_type":self.entity_type}

@dataclass(frozen=True)
class TypedValue:
    type_name: str
    value: bool | int | str | EntityRef
    def to_dict(self):
        value=self.value.to_dict() if isinstance(self.value,EntityRef) else self.value
        return {"type_name":self.type_name,"value":value}
```

Implement the remaining interface records with the same strict type checks, tuple conversion, and `MappingProxyType` for mappings. `Proposition` accepts only the two allowed relations. `GenericNarrative.to_dict()` contains exactly schema/domain identity plus entities/events/observations/claims/receptions/decisions, and:

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

Do not begin Task 2 until exact-head full `proof` succeeds.

---

### Task 2: DomainSpec and Objective Transition Hooks

**Files:** Create `narrative_dynamics/narrative/domain.py`, `tests/test_narrative_domain.py`.

**Interfaces:** `EntityTypeSpec`, `ValueTypeSpec`, `ParameterSpec`, `StateVariableSpec`, `StateEffectSpec`, `EventTypeSpec`, `ActionTypeSpec`, `DecisionTypeSpec`, `StateDeltaOp`, `StateDelta`, `SemanticHookBinding`, `DomainSpec`.

- [ ] **Step 1: Write RED test with exact local domain**

```python
from __future__ import annotations
import unittest
from narrative_dynamics.narrative.domain import (
    DomainSpec, EntityTypeSpec, EventTypeSpec, ParameterSpec,
    SemanticHookBinding, StateDelta, StateDeltaOp, StateEffectSpec,
    StateVariableSpec, ValueTypeSpec,
)
from narrative_dynamics.narrative.ir import Entity, EntityRef, NarrativeEvent, TypedValue

class SetHealthHook:
    def __call__(self, prior_state, event):
        service=event.arguments["service"].value
        return StateDelta((StateDeltaOp("set",service.entity_id,"service.health",event.arguments["health"]),))

class IllegalHook:
    def __call__(self, prior_state, event):
        return StateDelta((StateDeltaOp("set","svc","service.secret",TypedValue("HealthState","failed")),))

def make_domain(hook):
    return DomainSpec(
      domain_id="service-test",version="1",
      entity_types=(EntityTypeSpec("Service"),),
      value_types=(ValueTypeSpec("ServiceRef","entity_ref",entity_type="Service"),
                   ValueTypeSpec("HealthState","enum",allowed_values=("healthy","failed","recovered"))),
      state_variables=(StateVariableSpec("service.health","Service","HealthState"),),
      event_types=(EventTypeSpec("SetHealth",None,
          (ParameterSpec("service","ServiceRef"),ParameterSpec("health","HealthState",intervenable=True)),
          (StateEffectSpec("service.health","service"),),"set_health"),),
      action_types=(),decision_types=(),
      semantic_hooks=(SemanticHookBinding("set_health","set service health",hook),))

class DomainSpecTests(unittest.TestCase):
    def test_hook_identity_and_delta_boundary(self):
        event=NarrativeEvent("e1",1,"SetHealth",None,{"service":TypedValue("ServiceRef",EntityRef("svc","Service")),"health":TypedValue("HealthState","failed")})
        entities={"svc":Entity("svc","Service")}
        domain=make_domain(SetHealthHook())
        self.assertTrue(domain.content_hash.startswith("sha256:"))
        self.assertEqual(domain.apply_event({},event,entities).operations[0].state_variable,"service.health")
        with self.assertRaisesRegex(ValueError,"undeclared state variable"):
            make_domain(IllegalHook()).apply_event({},event,entities)
```

- [ ] **Step 2: Commit RED / observe missing domain module**

```bash
git add tests/test_narrative_domain.py
git commit -m "test: define generic domain semantics RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 3: Implement declarations and hook identity**

```python
_VALUE_KINDS=frozenset({"enum","bool","integer","text","entity_ref"})
_DELTA_KINDS=frozenset({"set","clear"})

@dataclass(frozen=True)
class SemanticHookBinding:
    name: str
    declared_purpose: str
    hook: object = field(compare=False,repr=False)
    @property
    def implementation_hash(self) -> str:
        return measure_implementation(self.hook).content_hash
    def to_dict(self):
        return {"hook_name":self.name,"declared_purpose":self.declared_purpose,
                "implementation_hash":self.implementation_hash}
```

Every declaration/delta record exposes deterministic `to_dict()`. `ValueTypeSpec.validate()` enforces exact `TypedValue.type_name` and kind semantics. `DomainSpec.content_hash` hashes declarations plus ordered hook identities. `apply_event()` validates event type/args/actor, invokes only the declared hook, requires `StateDelta`, and rejects operations outside declared effects, wrong subjects, wrong value types, `clear` with value, or `set` without value.

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

**Produces:** `validate_narrative(story, domain)` and shared test helpers.

- [ ] **Step 1: Create exact shared non-location test fixture**

```python
# tests/narrative_test_support.py
class TestSetHealthHook:
    def __call__(self, prior_state, event):
        service=event.arguments["service"].value
        return StateDelta((StateDeltaOp("set",service.entity_id,"service.health",event.arguments["health"]),))

def make_test_domain():
    return DomainSpec(
      domain_id="test-service",version="1",
      entity_types=(EntityTypeSpec("Agent"),EntityTypeSpec("Service")),
      value_types=(ValueTypeSpec("ServiceRef","entity_ref",entity_type="Service"),
                   ValueTypeSpec("HealthState","enum",allowed_values=("healthy","failed","recovered"))),
      state_variables=(StateVariableSpec("service.health","Service","HealthState"),),
      event_types=(EventTypeSpec("SetHealth",None,
          (ParameterSpec("service","ServiceRef"),ParameterSpec("health","HealthState",intervenable=True)),
          (StateEffectSpec("service.health","service"),),"set_health"),),
      action_types=(ActionTypeSpec("service-action",()),),
      decision_types=(DecisionTypeSpec("service-response","Agent","service-action"),),
      semantic_hooks=(SemanticHookBinding("set_health","set one health cell",TestSetHealthHook()),))

def target_cell(): return StateCellRef(EntityRef("svc","Service"),"service.health")

def make_test_story(*,claim_value="recovered",claim_relation="equals",receive=True,
                    second_claim_value=None,bob_observes_failure=True,alice_observes_recovery=True):
    domain=make_test_domain(); observations=[]
    if bob_observes_failure: observations.append(Observation("o1","bob","e1"))
    if alice_observes_recovery: observations.append(Observation("o2","alice","e2"))
    if second_claim_value is not None: observations.append(Observation("o3","carol","e2"))
    claims=[Claim("c1",3,"alice",Proposition(EntityRef("svc","Service"),"service.health",claim_relation,TypedValue("HealthState",claim_value)),("e2",))]
    receptions=[Reception("r1","c1","bob")] if receive else []
    decision_time=4
    if second_claim_value is not None:
        claims.append(Claim("c2",4,"carol",Proposition(EntityRef("svc","Service"),"service.health","equals",TypedValue("HealthState",second_claim_value)),("e2",)))
        if receive: receptions.append(Reception("r2","c2","bob"))
        decision_time=5
    return GenericNarrative(domain.domain_id,domain.version,domain.content_hash,
      (Entity("bob","Agent"),Entity("alice","Agent"),Entity("carol","Agent"),Entity("svc","Service")),
      (NarrativeEvent("e1",1,"SetHealth",None,{"service":TypedValue("ServiceRef",EntityRef("svc","Service")),"health":TypedValue("HealthState","failed")}),
       NarrativeEvent("e2",2,"SetHealth",None,{"service":TypedValue("ServiceRef",EntityRef("svc","Service")),"health":TypedValue("HealthState","recovered")})),
      tuple(observations),tuple(claims),tuple(receptions),
      (Decision("d1",decision_time,"bob","service-response",(target_cell(),),
        (ActionOption("restart","service-action",{}),ActionOption("leave","service-action",{}))),))
```

- [ ] **Step 2: Write RED validator tests**

```python
class NarrativeValidationTests(unittest.TestCase):
    def test_support_requires_access_but_not_truth(self):
        domain=make_test_domain(); story=make_test_story()
        stale=replace(story.claims[0],proposition=replace(story.claims[0].proposition,value=TypedValue("HealthState","healthy")))
        validate_narrative(replace(story,claims=(stale,)),domain)
        with self.assertRaisesRegex(ValueError,"speaker.*support"):
            validate_narrative(make_test_story(alice_observes_recovery=False),domain)
    def test_global_time_uniqueness(self):
        domain=make_test_domain(); story=make_test_story(); claim=replace(story.claims[0],logical_time=1)
        with self.assertRaisesRegex(ValueError,"logical times must be globally unique"):
            validate_narrative(replace(story,claims=(claim,)),domain)
```

- [ ] **Step 3: Commit RED / observe missing validator**

```bash
git add tests/narrative_test_support.py tests/test_narrative_validation.py
git commit -m "test: define canonical narrative validation RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 4: Implement validator**

```python
def validate_narrative(story,domain):
    if not isinstance(story,GenericNarrative): raise TypeError("generic narrative validation requires GenericNarrative")
    if not isinstance(domain,DomainSpec): raise TypeError("generic narrative validation requires DomainSpec")
    if (story.domain_id,story.domain_version,story.domain_spec_hash)!=(domain.domain_id,domain.version,domain.content_hash):
        raise ValueError("generic narrative domain identity does not match DomainSpec")
```

Then enforce unique IDs; declared types; event validation; globally unique event/claim/decision times; observation refs/unique pairs; claim typed proposition and non-empty support; earlier event support + speaker observation; earlier prior-claim support + speaker reception; reception refs/unique pairs; decision actor/context/action schema. Never compare claim content with objective truth.

- [ ] **Step 5: GREEN / commit / proof**

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
        self.assertEqual(objective_state(story,domain,at_time=2)[cell],TypedValue("HealthState","recovered"))
        self.assertEqual(direct_state(story,domain,"bob",at_time=2).resolved_values[cell],TypedValue("HealthState","failed"))
    def test_received_unreceived_constraint_and_conflict(self):
        cell=target_cell(); domain=make_test_domain()
        self.assertEqual(epistemic_state(make_test_story(),domain,"bob").cells[cell].evidence_kind,"testimony")
        self.assertEqual(epistemic_state(make_test_story(receive=False),domain,"bob").cells[cell].evidence_kind,"direct_perception")
        constrained=epistemic_state(make_test_story(claim_relation="not_equals",bob_observes_failure=False),domain,"bob")
        self.assertEqual(constrained.cells[cell].status,"unknown"); self.assertNotIn(cell,constrained.resolved_values)
        conflict=epistemic_state(make_test_story(second_claim_value="failed"),domain,"bob")
        self.assertEqual(conflict.cells[cell].status,"conflicted"); self.assertIsNone(conflict.cells[cell].resolved_value)
```

- [ ] **Step 2: Commit RED / observe missing replay module**

```bash
git add tests/test_narrative_replay.py
git commit -m "test: define generic epistemic replay RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 3: Implement objective replay**

```python
state={}
for event in sorted(story.events,key=lambda item:item.logical_time):
    if at_time is not None and event.logical_time>at_time: continue
    delta=domain.apply_event(state,event,entity_by_id)
    for op in delta.operations:
        subject=entity_by_id[op.subject_id]; cell=StateCellRef(EntityRef(subject.id,subject.type_name),op.state_variable)
        if op.kind=="clear": state.pop(cell,None)
        else: state[cell]=op.value
return MappingProxyType(dict(state))
```

- [ ] **Step 4: Implement direct/communicated evidence resolution**

Direct observation converts only the observed event's delta to direct evidence. Received claim converts only its proposition to testimony evidence. `not_equals` is constraint only. One later testimony may supersede older direct evidence. Two different testimony equalities after the most recent direct equality produce `conflicted`; latest evidence remains recorded but resolved value is absent. A later direct equality resolves again. Freeze all histories/maps.

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

**Files:** Create `decision.py`; modify shared test support; create `tests/test_narrative_decision.py`.

**Produces:** `EvidenceAccess`, `DecisionCellView`, `DecisionContext`, `DecisionChoice`, `DecisionModelSpec`, `DecisionResult`, `EpistemicResolutionError`, `DecisionResolutionError`, `require_resolved_cell`, `run_decision_model`.

- [ ] **Step 1: Add exact shared model hooks**

```python
class RecordingFirstActionHook:
    def __init__(self): self.context=None
    def __call__(self,context):
        self.context=context
        return DecisionChoice(context.decision.actions[0].id,tuple(e.supporting_id for e in context.evidence_history),tuple(context.visible_state))

class HealthActionHook:
    def __call__(self,context):
        value=require_resolved_cell(context,target_cell()).value
        if value=="failed": action="restart"
        elif value=="recovered": action="leave"
        else: raise DecisionResolutionError("resolved health does not map to a declared service response")
        return DecisionChoice(action,tuple(e.supporting_id for e in context.evidence_history),(target_cell(),))

def make_model(model_id,access,hook):
    return DecisionModelSpec(model_id,"1",("service-response",),access,(),hook)
def make_health_model(model_id,access): return make_model(model_id,access,HealthActionHook())
```

- [ ] **Step 2: Write RED tests**

```python
class GenericDecisionTests(unittest.TestCase):
    def test_access_boundary_and_identity(self):
        hook=RecordingFirstActionHook(); direct=make_model("same",EvidenceAccess.DIRECT_ONLY,hook)
        run_decision_model(make_test_story(),make_test_domain(),"d1",direct)
        self.assertFalse(hasattr(hook.context,"story")); self.assertFalse(hasattr(hook.context,"objective_state"))
        self.assertTrue(all(e.evidence_kind=="direct_perception" for e in hook.context.evidence_history))
        epistemic=make_model("same",EvidenceAccess.EPISTEMIC,RecordingFirstActionHook())
        self.assertNotEqual(direct.content_hash,epistemic.content_hash)
    def test_invalid_action_and_conflict_fail_closed(self):
        cell=target_cell(); context=DecisionContext(make_test_story().decisions[0],"bob",4,
            {cell:DecisionCellView("conflicted",None,("c1","c2"))},(),{}, {})
        with self.assertRaises(EpistemicResolutionError): require_resolved_cell(context,cell)
        with self.assertRaises(DecisionResolutionError):
            run_decision_model(make_test_story(),make_test_domain(),"d1",make_model("bad",EvidenceAccess.DIRECT_ONLY,InvalidActionHook()))
```

`InvalidActionHook` returns `DecisionChoice("not-declared",(),())`.

- [ ] **Step 3: Commit RED / observe missing decision module**

```bash
git add tests/narrative_test_support.py tests/test_narrative_decision.py
git commit -m "test: define capability-limited decision RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 4: Implement contexts, identity, execution**

`direct_only` gets direct cells/history; `epistemic` and `custom_epistemic` get epistemic cells/history; `omniscient` gets objective cells wrapped as resolved and empty evidence history. No context stores story/domain/RNG.

```python
@property
def implementation_hash(self): return measure_implementation(self.decision_hook).content_hash
@property
def content_hash(self):
    return stable_content_hash({"model_id":self.model_id,"version":self.version,
      "supported_decision_types":self.supported_decision_types,"evidence_access":self.evidence_access.value,
      "parameter_schema":tuple(x.to_dict() for x in self.parameter_schema),"implementation_hash":self.implementation_hash})
```

`run_decision_model()` validates exact decision/model/parameters, invokes hook with context only, requires one declared action, emits one-hot scores/policy and deterministic basis.

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

- [ ] **Step 1: Add exact analysis helpers**

```python
def make_analysis_case():
    story=make_test_story(); domain=make_test_domain()
    return story,domain,make_health_model("direct-health",EvidenceAccess.DIRECT_ONLY),make_health_model("epistemic-health",EvidenceAccess.EPISTEMIC),make_health_model("omniscient-health",EvidenceAccess.OMNISCIENT)
def make_same_action_different_basis_case():
    story=make_test_story(claim_value="failed"); domain=make_test_domain()
    return story,domain,make_health_model("direct-health",EvidenceAccess.DIRECT_ONLY),make_health_model("epistemic-health",EvidenceAccess.EPISTEMIC)
```

- [ ] **Step 2: Write RED tests**

```python
class GenericAnalysisTests(unittest.TestCase):
    def test_scope_and_same_action_different_basis(self):
        story,domain,direct,epistemic,omniscient=make_analysis_case(); scope=derive_analysis_scope(story,domain,"d1")
        self.assertEqual(scope.tracked_agents,("bob","alice")); self.assertEqual(scope.snapshot_times,(1,2,3,4))
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

- [ ] **Step 4: Implement scope, trajectory, mechanism comparison**

```python
def derive_analysis_scope(story,domain,decision_id):
    decision=_one_decision(story,decision_id)
    cells=tuple(decision.context_cells); agents=[decision.actor_id]
    received={(r.claim_id,r.recipient_id) for r in story.receptions}
    for claim in sorted(story.claims,key=lambda x:x.logical_time):
        cell=StateCellRef(claim.proposition.subject,claim.proposition.state_variable)
        if cell in cells and (claim.id,decision.actor_id) in received and claim.speaker_id not in agents:
            agents.append(claim.speaker_id)
    times=_relevant_times(story,domain,cells,decision)
    return AnalysisScope(decision.id,cells,tuple(agents),tuple(sorted(times)))
```

Build snapshots at frozen scope times from objective/direct/epistemic replay; empty triggers are legal. Only decision time stores model result. Pairwise fields exactly `action_equal` and `basis_equal`; mechanism uniqueness flags are structurally false.

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

**Produces:** kinds exactly `remove_event`, `change_event_argument`, `remove_observation`, `change_claim_value`, `remove_reception`; `Intervention`, `Divergence`, `CounterfactualResult`, `generate_minimal_interventions`, `evaluate_counterfactual`, `first_divergence`.

- [ ] **Step 1: Write RED tests with all rejection stages**

```python
class GenericInterventionTests(unittest.TestCase):
    def test_kinds_and_reception_divergence(self):
        story,domain,direct,epistemic,omniscient=make_analysis_case(); scope=derive_analysis_scope(story,domain,"d1")
        items=generate_minimal_interventions(story,domain,scope)
        self.assertEqual({x.kind for x in items},{"remove_event","change_event_argument","remove_observation","change_claim_value","remove_reception"})
        baseline=build_trajectory(story,domain,scope,epistemic)
        result=evaluate_counterfactual(story,domain,scope,epistemic,baseline,next(x for x in items if x.kind=="remove_reception"))
        self.assertEqual(result.status,"valid"); self.assertEqual(result.first_divergence.logical_time,3)
    def test_rejection_stages(self):
        story,domain,direct,model,omniscient=make_analysis_case(); scope=derive_analysis_scope(story,domain,"d1"); baseline=build_trajectory(story,domain,scope,model)
        support=next(x for x in generate_minimal_interventions(story,domain,scope) if x.kind=="remove_observation" and x.target_ref=="o2")
        self.assertEqual(evaluate_counterfactual(story,domain,scope,model,baseline,support).rejection_stage,"narrative_validation")
        healthy=next(x for x in generate_minimal_interventions(story,domain,scope) if x.kind=="change_claim_value" and x.to_value.value=="healthy")
        self.assertEqual(evaluate_counterfactual(story,domain,scope,model,baseline,healthy).rejection_stage,"decision_resolution")
        conflict_story=make_test_story(claim_value="recovered",second_claim_value="recovered"); conflict_scope=derive_analysis_scope(conflict_story,domain,"d1"); conflict_base=build_trajectory(conflict_story,domain,conflict_scope,model)
        conflict=next(x for x in generate_minimal_interventions(conflict_story,domain,conflict_scope) if x.kind=="change_claim_value" and x.target_ref=="c1" and x.to_value.value=="failed")
        self.assertEqual(evaluate_counterfactual(conflict_story,domain,conflict_scope,model,conflict_base,conflict).rejection_stage,"epistemic_resolution")
```

- [ ] **Step 2: Commit RED / observe missing interventions module**

```bash
git add tests/test_narrative_interventions.py
git commit -m "test: define generic counterfactual RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 3: Implement single-variable application/generation**

```python
_ALLOWED_INTERVENTIONS=frozenset({"remove_event","change_event_argument","remove_observation","change_claim_value","remove_reception"})
```

Each call applies one intervention only. Event argument changes require exact `intervenable=True`; claim change alters value only; removals target one exact ID; event removal never repairs refs. Generate enum/bool alternates only and sort deterministically by kind/time/target/field/value hash.

- [ ] **Step 4: Implement typed rejection and divergence**

```python
try: candidate=_apply_intervention(story,domain,intervention)
except ValueError as error: return _rejected(intervention,"domain_validation",error)
try: validate_narrative(candidate,domain)
except ValueError as error: return _rejected(intervention,"narrative_validation",error)
try: trajectory=build_trajectory(candidate,domain,scope,model)
except EpistemicResolutionError as error: return _rejected(intervention,"epistemic_resolution",error)
except DecisionResolutionError as error: return _rejected(intervention,"decision_resolution",error)
```

All other exceptions propagate. Reuse baseline snapshot times and compare canonical objective/direct/epistemic/action fields only; identical trajectory returns no divergence.

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

**Produces:** `location_testimony_domain`, `adapt_testimony_story_v2`, `legacy_direct_search_model`, `legacy_epistemic_search_model`, `legacy_omniscient_search_model`; compatibility-module only.

- [ ] **Step 1: Write exact RED equivalence tests**

```python
class StoryV2CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.truthful=NarrativeScenarioV2.from_case(load_narrative_case_v2("fixtures/stories/key_location_truthful_testimony_v2.json"))
        self.stale=NarrativeScenarioV2.from_case(load_narrative_case_v2("fixtures/stories/key_location_stale_testimony_v2.json"))
    def test_three_model_actions_and_timeline(self):
        expected={"truthful":("search_drawer","search_box","search_box"),"stale":("search_drawer","search_drawer","search_box")}
        for name,story in (("truthful",self.truthful),("stale",self.stale)):
            generic=adapt_testimony_story_v2(story); domain=location_testimony_domain()
            actions=tuple(analyze_narrative(generic,domain,"d1",m).baseline.selected_action for m in (legacy_direct_search_model(),legacy_epistemic_search_model(),legacy_omniscient_search_model()))
            self.assertEqual(actions,expected[name])
            legacy=analyze_testimony_evolution(story).baseline
            generic_base=analyze_narrative(generic,domain,"d1",legacy_epistemic_search_model()).baseline
            self.assertEqual(tuple(s.logical_time for s in generic_base.snapshots),tuple(s.logical_time for s in legacy.snapshots))
```

Add explicit assertions in this test file for objective/direct/epistemic cell values t1–t4; evidence kind/support/source at t3; remove reception; claim-content change; Bob direct-observation removal rejection; Alice support-observation removal rejection; first divergence and action_changed. Run legacy identity tests unchanged.

- [ ] **Step 2: Commit RED / observe missing compatibility module**

```bash
git add tests/test_narrative_compat_story_v2.py
git commit -m "test: define Testimony V2 generic conformance RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 3: Implement static location domain/adapter**

```python
def _observation_id(event_id,agent_id): return f"obs:{event_id}:{agent_id}"
def _reception_id(report_id,recipient_id): return f"recv:{report_id}:{recipient_id}"
```

Use `Agent`,`Object`,`Location`, `LocationRef`, and `object.location`. Relocation hook validates legacy continuity. Report maps to `Claim(report.id,report.logical_time,report.speaker,Proposition(EntityRef(report.object,"Object"),"object.location","equals",TypedValue("LocationRef",EntityRef(report.location,"Location"))),(report.support_event,))`.

- [ ] **Step 4: Implement three compatibility models**

One search hook resolves target cell and selects unique declared action whose `location` argument equals resolved `LocationRef`; access modes direct/epistemic/omniscient create three specs. Missing/non-unique action raises `DecisionResolutionError`.

- [ ] **Step 5: GREEN + legacy regressions / commit / proof**

```bash
python3 -m unittest tests.test_narrative_compat_story_v2 tests.test_story_schema_v2 tests.test_story_scenario_v2 tests.test_story_replay_v2 tests.test_story_testimony_models tests.test_story_evolution_v2 -v
git add narrative_dynamics/narrative/compat_story_v2.py tests/test_narrative_compat_story_v2.py
git commit -m "feat: add Testimony V2 generic conformance"
git push origin work/generic-narrative-engine-v1
```

Require fresh full `proof` success; legacy production files remain untouched.

---

### Task 9: Non-Location Service Incident Conformance Domain

**Files:** Create `domains/__init__.py`, `domains/service_incident.py`, `tests/test_narrative_service_incident.py`.

**Produces:** `service_incident_domain`, `service_incident_recovered_claim_story`, `service_incident_stale_claim_story`, direct/epistemic/omniscient service models.

- [ ] **Step 1: Write exact RED tests**

```python
class ServiceIncidentConformanceTests(unittest.TestCase):
    def test_pair_and_model_contrast(self):
        recovered=service_incident_recovered_claim_story(); stale=service_incident_stale_claim_story(); domain=service_incident_domain()
        self.assertEqual(recovered.events,stale.events); self.assertEqual(recovered.observations,stale.observations); self.assertEqual(recovered.receptions,stale.receptions); self.assertEqual(recovered.decisions,stale.decisions)
        self.assertNotEqual(recovered.claims[0].proposition.value,stale.claims[0].proposition.value)
        expected={"recovered":("restart_service","leave_running","leave_running"),"stale":("restart_service","restart_service","leave_running")}
        for name,story in (("recovered",recovered),("stale",stale)):
            actual=tuple(analyze_narrative(story,domain,"d1",m).baseline.selected_action for m in (service_direct_model(),service_epistemic_model(),service_omniscient_model()))
            self.assertEqual(actual,expected[name])
    def test_reception_ablation(self):
        domain=service_incident_domain(); model=service_epistemic_model()
        for story,changed in ((service_incident_recovered_claim_story(),True),(service_incident_stale_claim_story(),False)):
            scope=derive_analysis_scope(story,domain,"d1"); base=build_trajectory(story,domain,scope,model)
            intervention=next(x for x in generate_minimal_interventions(story,domain,scope) if x.kind=="remove_reception")
            result=evaluate_counterfactual(story,domain,scope,model,base,intervention)
            self.assertEqual(result.first_divergence.logical_time,3); self.assertEqual(result.first_divergence.action_changed,changed)
```

- [ ] **Step 2: Commit RED / observe missing domain**

```bash
git add tests/test_narrative_service_incident.py
git commit -m "test: define non-location narrative conformance RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 3: Implement exact Service Incident semantics**

```python
HEALTH_VALUES=("healthy","failed","recovered")
class ServiceFailureHook:
    def __call__(self,prior,event):
        service=event.arguments["service"].value; cell=StateCellRef(service,"service.health")
        current=prior.get(cell)
        if current is not None and current.value!="healthy": raise ValueError("service failure requires healthy prior state")
        return StateDelta((StateDeltaOp("set",service.entity_id,"service.health",TypedValue("HealthState","failed")),))
class ServiceRecoveryHook:
    def __call__(self,prior,event):
        service=event.arguments["service"].value; cell=StateCellRef(service,"service.health")
        if prior.get(cell)!=TypedValue("HealthState","failed"): raise ValueError("service recovery requires failed prior state")
        return StateDelta((StateDeltaOp("set",service.entity_id,"service.health",TypedValue("HealthState","recovered")),))
```

Pair is exactly e1@1 failure Bob observes; e2@2 recovery Alice observes; c1@3 Alice says recovered or failed supported by e2 and Bob receives; d1@4. Decision rule failed→restart_service, recovered→leave_running, otherwise typed resolution error.

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

- [ ] **Step 1: Create exact compiler test support**

```python
def make_source_bundle(text="service failed then recovered; Alice told Bob recovered"):
    return SourceBundle((SourceDocument("d1",text,"text","memory:d1"),))
def whole_span(source):
    doc=source.documents[0]; return SourceSpan.from_document(doc,0,len(doc.content))
def make_candidate_bundle(source,first_event_time=1,claim_value="recovered",confidence=0.9):
    span=whole_span(source)
    return CandidateBundle(ExtractorIdentity("test-extractor","1"),(
      CandidateEntity("ce-bob","bob","Agent",(span,),confidence),CandidateEntity("ce-alice","alice","Agent",(span,),confidence),CandidateEntity("ce-svc","svc","Service",(span,),confidence),
      CandidateEvent("cev1","e1","SetHealth",first_event_time,None,{"service":CandidateValue("ServiceRef",CandidateEntityRef("ce-svc")),"health":CandidateValue("HealthState","failed")},(span,),confidence),
      CandidateEvent("cev2","e2","SetHealth",2,None,{"service":CandidateValue("ServiceRef",CandidateEntityRef("ce-svc")),"health":CandidateValue("HealthState","recovered")},(span,),confidence),
      CandidateObservation("co1","o1","ce-bob","cev1",(span,),confidence),CandidateObservation("co2","o2","ce-alice","cev2",(span,),confidence),
      CandidateClaim("cc1","c1",3,"ce-alice",CandidateProposition("ce-svc","service.health","equals",CandidateValue("HealthState",claim_value)),("cev2",),(span,),confidence),
      CandidateReception("cr1","r1","cc1","ce-bob",(span,),confidence),
      CandidateDecision("cd1","d1",4,"ce-bob","service-response",(CandidateStateCellRef("ce-svc","service.health"),),(CandidateActionOption("restart","service-action",{}),CandidateActionOption("leave","service-action",{})),(span,),confidence),))
```

- [ ] **Step 2: Write RED record tests**

```python
class CompilerRecordTests(unittest.TestCase):
    def test_span_confidence_and_distinct_identity_layers(self):
        doc=SourceDocument("d1","service failed","text","memory:d1"); span=SourceSpan.from_document(doc,0,7); source=SourceBundle((doc,)); source.validate_span(span)
        with self.assertRaisesRegex(ValueError,"confidence"): CandidateEntity("ce","svc","Service",(span,),float("nan"))
        candidates=make_candidate_bundle(source)
        resolution=ResolutionRecord("ce-bob","accepted",{},"human:test","entity confirmed")
        self.assertNotEqual(source.content_hash,candidates.content_hash)
        self.assertTrue(stable_content_hash((resolution.to_dict(),)).startswith("sha256:"))
```

- [ ] **Step 3: Commit RED / observe missing compiler module**

```bash
git add tests/narrative_compiler_test_support.py tests/test_narrative_compiler_records.py
git commit -m "test: define narrative compiler provenance RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 4: Implement provenance records**

```python
@property
def content_hash(self): return stable_content_hash({"content":self.content})
@classmethod
def from_document(cls,document,start,end):
    if not isinstance(start,int) or isinstance(start,bool) or not isinstance(end,int) or isinstance(end,bool) or start<0 or end<start or end>len(document.content): raise ValueError("source span offsets are invalid")
    return cls(document.document_id,start,end,stable_content_hash({"text":document.content[start:end]}))
```

Candidate confidence is finite [0,1], every extracted candidate has source spans, nested data freezes, CandidateBundle candidate IDs are unique, and candidate references use candidate IDs. Resolution decision exactly accepted/rejected/unresolved; freeze selected fields.

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

**Produces:** `CompilationDiagnostic`, `CompilationResult`, `compile_candidates`; statuses exactly `canonical`, `incomplete`, `rejected`.

- [ ] **Step 1: Write exact RED compiler tests**

```python
class DeterministicCompilerTests(unittest.TestCase):
    def test_incomplete_then_resolved(self):
        source=make_source_bundle(); candidates=make_candidate_bundle(source,first_event_time=None); domain=make_test_domain()
        incomplete=compile_candidates(source,candidates,(),domain)
        self.assertEqual(incomplete.status,"incomplete"); self.assertIsNone(incomplete.canonical_scenario)
        resolved=compile_candidates(source,candidates,(ResolutionRecord("cev1","accepted",{"logical_time":1},"human:test","first event"),),domain)
        self.assertEqual(resolved.status,"canonical"); self.assertEqual(resolved.canonical_hash,resolved.canonical_scenario.content_hash)
    def test_provenance_can_change_without_semantic_identity_change(self):
        source_a=make_source_bundle("wording A"); source_b=make_source_bundle("wording B")
        first=compile_candidates(source_a,make_candidate_bundle(source_a,confidence=0.8),(),make_test_domain())
        second=compile_candidates(source_b,make_candidate_bundle(source_b,confidence=0.99),(),make_test_domain())
        self.assertNotEqual(first.source_bundle_hash,second.source_bundle_hash); self.assertNotEqual(first.candidate_bundle_hash,second.candidate_bundle_hash); self.assertEqual(first.canonical_hash,second.canonical_hash)
    def test_unknown_resolution_field_is_rejected(self):
        source=make_source_bundle(); candidates=make_candidate_bundle(source)
        result=compile_candidates(source,candidates,(ResolutionRecord("cev1","accepted",{"unsupported_field":"x"},"human:test","probe"),),make_test_domain())
        self.assertEqual(result.status,"rejected")
```

Add a local two-claim candidate helper by appending Carol entity/observation, c2@4 supported by cev2, Bob reception, and moving decision to t5; assert conflicting claims compile as canonical. Forge one `SourceSpan.exact_text_hash` and assert compiler rejects before semantic construction. Assert canonical payload contains none of `content`, `confidence`, extractor name, resolver identity, or diagnostics.

- [ ] **Step 2: Commit RED / observe missing compiler function**

```bash
git add tests/test_narrative_compiler.py
git commit -m "test: define deterministic narrative compiler RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 3: Implement exact resolution whitelist**

```python
_ALLOWED_RESOLUTION_FIELDS={
 CandidateEntity:frozenset({"entity_id","type_name"}),
 CandidateEvent:frozenset({"logical_time","type_name","actor_candidate_id"}),
 CandidateObservation:frozenset({"agent_candidate_id","event_candidate_id"}),
 CandidateClaim:frozenset({"logical_time","speaker_candidate_id"}),
 CandidateReception:frozenset({"claim_candidate_id","recipient_candidate_id"}),
 CandidateDecision:frozenset({"logical_time","actor_candidate_id","type_name"}),
}
```

Unknown field/missing candidate resolution rejects. Rejected candidate omitted. Explicit unresolved or required None after resolution produces diagnostic `{code:"unresolved_required_field",candidate_id,field,message:"required semantic field is unresolved"}` and incomplete result. No resolution means accept concrete extracted fields.

- [ ] **Step 4: Implement dependency-order compilation and identities**

```python
# dependency order
entities=_compile_entities(...)
events=_compile_events(...)
observations=_compile_observations(...)
claims=_compile_claims(...)
receptions=_compile_receptions(...)
decisions=_compile_decisions(...)
story=GenericNarrative(domain.domain_id,domain.version,domain.content_hash,entities,events,observations,claims,receptions,decisions)
validate_narrative(story,domain)
```

Resolve candidate refs through final entity bindings; duplicate final entity ID allowed only with same type. Expected semantic `ValueError` becomes rejected; programming/attestation/runtime errors propagate. Record source/candidate/sorted-resolution hashes; canonical hash is only story content hash and exists only for canonical status.

- [ ] **Step 5: GREEN / commit / proof**

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

- [ ] **Step 1: Write exact RED trust/API tests**

```python
def pin_for(domain):
    return TrustedDomainPin(domain.domain_id,domain.version,domain.content_hash,{h.name:h.implementation_hash for h in domain.semantic_hooks})

class NarrativeTrustTests(unittest.TestCase):
    def test_policy_and_lineage(self):
        domain=make_test_domain(); policy=TrustedDomainPolicy("research","1",(pin_for(domain),)); self.assertEqual(policy.bind(domain).pin.domain_spec_hash,domain.content_hash)
        with self.assertRaisesRegex(ValueError,"trusted domain"): policy.bind(replace(domain,version="2"))
        story=make_test_story(); model=make_health_model("epistemic",EvidenceAccess.EPISTEMIC); analysis=analyze_narrative(story,domain,"d1",model); scope=derive_analysis_scope(story,domain,"d1")
        common=dict(canonical_narrative_hash=story.content_hash,domain_spec_hash=domain.content_hash,decision_model_identity=model.content_hash,analysis_scope_hash=stable_content_hash(scope.to_dict()),intervention_manifest_hash=stable_content_hash(()),analysis_payload_hash=stable_content_hash(analysis.to_dict()))
        first=NarrativeAnalysisArtifact(**common,compilation_artifact_hash="sha256:"+"1"*64); second=NarrativeAnalysisArtifact(**common,compilation_artifact_hash="sha256:"+"2"*64)
        self.assertEqual(first.analysis_identity_hash,second.analysis_identity_hash); self.assertNotEqual(first.lineage_dict(),second.lineage_dict())
```

Add exact export assertion: build expected set from all canonical IR/domain/replay/decision/analysis/intervention/compiler/trust names fixed in the spec/this plan, assert `set(narrative.__all__)==expected`, and for every name assert `not hasattr(narrative_dynamics,name)`.

- [ ] **Step 2: Commit RED / observe missing trust + exports**

```bash
git add tests/test_narrative_trust_api.py
git commit -m "test: define narrative trust and API RED"
git push origin work/generic-narrative-engine-v1
```

- [ ] **Step 3: Implement external trust policy and analysis artifact**

```python
class TrustedDomainPolicy:
    def bind(self,domain):
        pin=self.pin_for(domain.domain_id,domain.version)
        actual={h.name:h.implementation_hash for h in domain.semantic_hooks}
        if pin.domain_spec_hash!=domain.content_hash or dict(pin.hook_hashes)!=actual:
            raise ValueError("trusted domain identity does not match policy")
        return TrustedDomainBinding(domain,self,pin)
```

Analysis identity payload exactly canonical narrative hash, domain spec hash, decision model identity, analysis scope hash, intervention manifest hash, analysis payload hash. `analysis_identity_hash=stable_content_hash(identity_payload)`. Optional compilation artifact hash appears only in lineage output.

- [ ] **Step 4: Finalize public surface**

Import/export the canonical names produced by Tasks 1–12 from `narrative_dynamics.narrative.__init__`. Do not export compatibility helpers, service helper constructors, hook implementation classes, or private compiler helpers. Do not modify `narrative_dynamics/__init__.py`.

- [ ] **Step 5: Full Python GREEN / commit / final exact-head proof**

```bash
python3 -m unittest tests.test_narrative_ir tests.test_narrative_domain tests.test_narrative_validation tests.test_narrative_replay tests.test_narrative_decision tests.test_narrative_analysis tests.test_narrative_interventions tests.test_narrative_compat_story_v2 tests.test_narrative_service_incident tests.test_narrative_compiler_records tests.test_narrative_compiler tests.test_narrative_trust_api -v
python3 -m unittest discover -s tests -v
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

Expected changed surface is approved design, this plan, new `narrative_dynamics/narrative/**`, and new narrative tests only. No Lean source, legacy story production source, runtime/registry/prison production source, or package-root export change.

- [ ] **Step 3: Boundary scan**

Review final diff for unique-mechanism claims, ToM/deception/population claims, objective fallback, RNG in Generic decision APIs, embedded LLM/API calls, source/confidence leakage into canonical payloads, raw story access by decision hooks, undeclared state writes, and multi-intervention APIs. Any real violation blocks integration.

- [ ] **Step 4: Fresh final verification evidence**

Record exact final feature SHA, proof run number/ID, full Lean build job count, Python test count/OK, StoryState success, Testimony success. If isolated checkout exists, independently run `python3 -m compileall -q narrative_dynamics`; otherwise explicitly leave this unclaimed.

- [ ] **Step 5: Update draft PR**

Final title `feat: add Generic Narrative Engine V1`. Body records architecture, V2 equivalence, Service Incident conformance, exact final evidence, root isolation, legacy identity preservation, no new Lean semantics, and scientific non-claims.

- [ ] **Step 6: Lock topology immediately before integration**

Capture final feature SHA, current research SHA, current master SHA. Require research still exact `82096098dda907aec04330b5ddf357ac6f753d2a`, feature ahead and behind=0. If research moved, stop and rebase/revalidate; never force.

- [ ] **Step 7: Non-force fast-forward research only**

Move `proof/narrative-dynamics-v0` to final feature SHA with `force=false`. Do not call PR merge against master.

- [ ] **Step 8: Post-integration verification**

Require research ref == feature ref == final SHA; master unchanged from Step 6; feature branch retained. If GitHub auto-records draft PR closed/merged because base=head, describe it as a research-ref fast-forward, not a master merge.

The strongest supported claim after integration is:

> Narrative Dynamics can execute a deterministic, domain-pluggable canonical narrative semantics in which objective events, direct observations, communicated claims, recipient-specific evidence, decision mechanisms, and single-variable counterfactuals remain separately represented, replayable, provenance-aware, and comparable across both the legacy testimony domain and an independent non-location conformance domain.

It still does not establish unique hidden-mechanism identification, general human Theory of Mind, deception intent, empirical human causal validity, or population-level validity.
