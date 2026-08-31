# Population Lifecycle V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make active network membership change through deterministic entry, exit, and death events while preserving auditable identities and synchronous local propagation.

**Architecture:** A lifecycle wrapper owns persistent status and information state for the complete fixed catalog. Each round applies lifecycle events first, induces a V1 model over active agents, delegates propagation to the existing synchronous kernel, and recombines active output with inactive/dead members. A separate metric module exposes population-level lifecycle outcomes.

**Tech Stack:** Python 3 standard library, frozen dataclasses, V1 ABM contracts and propagation, existing stable content hashing, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-population-lifecycle-v3-design.md`

## Global Constraints

- The catalogued identities, profiles, and social edge definitions remain fixed.
- Lifecycle status is persistent and exactly one of inactive, active, or dead.
- Events apply before propagation; at most one event may target an agent per round.
- Dead agents cannot re-enter, and inactive/dead agents never broadcast.
- Propagation uses only active-to-active edges and remains synchronous.
- Zero active agents is a valid lifecycle state and round outcome.
- Canonical ordering and stable hashes are required for every public artifact.

---

### Task 1: Lifecycle model and state contracts

**Files:**
- Create: `narrative_dynamics/abm/lifecycle_contracts.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Test: `tests/test_network_abm_lifecycle_contracts.py`

**Interfaces:**
- Consumes: `NetworkABMModel`, validation helpers, and stable content hashing.
- Produces: `LifecycleStatus`, `LifecycleEventKind`, `PopulationLifecycleEvent`, `PopulationLifecycleModel`, `LifecycleMemberState`, `PopulationLifecycleState`, and `initialize_lifecycle_population`.

- [ ] **Step 1: Write failing lifecycle contract tests**

```python
def test_initialization_covers_catalog_and_marks_effective_population():
    model = lifecycle_model(initial_active_agent_ids=("a",))
    state = initialize_lifecycle_population(model, beliefs={"a": 1.0})
    self.assertEqual(member(state, "a").status, LifecycleStatus.ACTIVE)
    self.assertEqual(member(state, "a").entry_count, 1)
    self.assertEqual(member(state, "b").status, LifecycleStatus.INACTIVE)
```

- [ ] **Step 2: Run contract tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_lifecycle_contracts -v`

Expected: import failure because lifecycle contracts do not exist.

- [ ] **Step 3: Implement immutable lifecycle contracts**

```python
class LifecycleStatus(str, Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    DEAD = "dead"

@dataclass(frozen=True)
class LifecycleMemberState:
    agent_id: str
    belief: float
    exposure_count: int
    broadcasting: bool
    status: LifecycleStatus
    entry_count: int
    exit_count: int
```

Validate catalog coverage, initial-active membership, status/broadcast consistency, event payload rules, parent chains, canonical tuples, and `to_dict()`/`content_hash` identities.

- [ ] **Step 4: Run contract tests and confirm GREEN**

Run: `python -m unittest tests.test_network_abm_lifecycle_contracts -v`

Expected: all lifecycle contract tests pass.

- [ ] **Step 5: Commit lifecycle contracts**

```text
git add docs/superpowers/specs/2026-08-31-population-lifecycle-v3-design.md docs/superpowers/plans/2026-08-31-population-lifecycle-v3.md narrative_dynamics/abm tests/test_network_abm_lifecycle_contracts.py
git commit -m "feat: define population lifecycle state"
```

### Task 2: Boundary events and active-subgraph propagation

**Files:**
- Create: `narrative_dynamics/abm/lifecycle.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Test: `tests/test_network_abm_lifecycle.py`

**Interfaces:**
- Consumes: Task 1 contracts, V1 model contracts, and `simulate_round`.
- Produces: `LifecycleRoundResult`, `LifecycleTrajectory`, `simulate_lifecycle_round`, `simulate_lifecycle_population`, and `active_population_view`.

- [ ] **Step 1: Write failing event-timing and propagation tests**

```python
def test_entry_applies_before_one_hop_propagation():
    first = simulate_lifecycle_round(model, initial, events=(enter("b"),))
    self.assertEqual(member(first.next_state, "b").belief, 1.0)
    self.assertEqual(member(first.next_state, "c").belief, 0.0)
    second = simulate_lifecycle_round(model, first.next_state, events=(enter("c"),))
    self.assertEqual(member(second.next_state, "c").belief, 1.0)
```

- [ ] **Step 2: Run lifecycle simulation tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_lifecycle -v`

Expected: import failure for lifecycle simulation interfaces.

- [ ] **Step 3: Implement lifecycle transitions and induced-model projection**

Apply canonical events to a temporary complete-catalog snapshot, create an induced `NetworkABMModel` and `PopulationState` when active membership is non-empty, delegate one round to V1, translate transmissions and resulting agent state back to lifecycle identity, and bypass V1 for an empty active set.

- [ ] **Step 4: Implement multi-round schedules and replay validation**

```python
trajectory = simulate_lifecycle_population(
    model,
    initial,
    event_schedule=((enter("b"),), (enter("c"),)),
)
self.assertEqual(trajectory.final_state.round_index, 2)
self.assertEqual(trajectory.rounds[0].next_state, trajectory.rounds[1].prior_state)
```

Run: `python -m unittest tests.test_network_abm_lifecycle -v`

Expected: entry, exit, death, empty-population, canonical replay, and schedule tests all pass.

- [ ] **Step 5: Commit lifecycle execution**

```text
git add narrative_dynamics/abm tests/test_network_abm_lifecycle.py
git commit -m "feat: simulate population lifecycle"
```

### Task 3: Lifecycle metrics, public API, and documentation

**Files:**
- Create: `narrative_dynamics/abm/lifecycle_metrics.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `README.md`
- Modify: `tests/test_network_abm_public_api.py`
- Test: `tests/test_network_abm_lifecycle_metrics.py`

**Interfaces:**
- Consumes: lifecycle model and state.
- Produces: `PopulationLifecycleMetrics` and `measure_population_lifecycle`, plus the complete V3 public surface.

- [ ] **Step 1: Write hand-computed lifecycle metric tests**

```python
def test_lifecycle_metrics_match_catalog_state():
    metrics = measure_population_lifecycle(model, state)
    self.assertEqual(metrics.catalog_population, 3)
    self.assertEqual(metrics.active_population, 1)
    self.assertEqual(metrics.dead_population, 1)
    self.assertEqual(metrics.active_share, 1 / 3)
    self.assertEqual(metrics.cumulative_entries, 2)
```

- [ ] **Step 2: Run metric tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_lifecycle_metrics -v`

Expected: import failure for lifecycle metrics.

- [ ] **Step 3: Implement lifecycle metrics and extend public exports**

```python
return PopulationLifecycleMetrics(
    catalog_population=len(state.members),
    active_population=active,
    inactive_population=inactive,
    dead_population=dead,
    active_share=active / len(state.members),
    dead_share=dead / len(state.members),
    cumulative_entries=sum(item.entry_count for item in state.members),
    cumulative_exits=sum(item.exit_count for item in state.members),
)
```

Extend `narrative_dynamics.abm.__all__` and its exact public API test with every V3 contract and function.

- [ ] **Step 4: Document and execute the lifecycle example**

Add a README example showing B entering before propagation, receiving A's information, and the resulting active population metric. Execute every README Python block independently.

- [ ] **Step 5: Run V1/V2/V3 and targeted narrative regressions**

Run:

```text
python -m unittest discover -s tests -p "test_network_abm*.py" -v
python -m unittest tests.test_narrative_simulation tests.test_narrative_stochastic_simulation tests.test_narrative_runtime_cognition -v
```

Expected: all network ABM and targeted narrative regression tests pass.

- [ ] **Step 6: Commit V3 metrics and docs**

```text
git add README.md narrative_dynamics/abm tests/test_network_abm_public_api.py tests/test_network_abm_lifecycle_metrics.py
git commit -m "docs: expose population lifecycle"
```
