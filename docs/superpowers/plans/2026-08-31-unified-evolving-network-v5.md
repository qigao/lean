# Unified Evolving Network V5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate lifecycle, adaptive trust, and endogenous topology into one deterministic model, state, round transition, trajectory, and macro snapshot.

**Architecture:** A unified contract composes existing lifecycle member, edge trust, and edge topology value types over one fixed V1 catalog. The runtime applies lifecycle events, derives an active trusted topology for V1 propagation, learns trust from actual transmissions, rewires active endpoint pairs from resulting beliefs, and commits every dimension to one parent-linked state. A focused metric module projects cross-layer population, cognition, trust, and structure observables.

**Tech Stack:** Python 3 standard library, frozen dataclasses, V1 propagation, V2 trust value contracts, V3 lifecycle value contracts, V4 topology value contracts, stable content hashing, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-unified-evolving-network-v5-design.md`

## Global Constraints

- Agent identity/profile and candidate-edge catalogs remain fixed.
- Initial active membership is canonical and non-empty.
- Every unified state covers the complete member, trust, and topology catalogs exactly once.
- Lifecycle events apply before propagation.
- Current propagation uses prior trust and prior topology over post-event active agents.
- Trust learning and topology rewiring affect only later rounds.
- Edges incident to inactive/dead agents retain trust and topology without rewiring.
- Zero active agents is a valid post-event state.
- Canonical ordering and stable hashes are required for every public artifact.

---

### Task 1: Unified model and state contracts

**Files:**
- Create: `narrative_dynamics/abm/evolving_contracts.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Create: `tests/evolving_fixtures.py`
- Test: `tests/test_network_abm_evolving_contracts.py`

**Interfaces:**
- Consumes: `NetworkABMModel`, `LifecycleMemberState`, `EdgeTrustState`, `EdgeTopologyState`, `initialize_population`, and stable content hashing.
- Produces: `EvolvingNetworkModel`, `EvolvingPopulationState`, and `initialize_evolving_population`.

- [ ] **Step 1: Write failing unified initialization tests**

```python
def test_initialization_covers_all_catalog_dimensions():
    model = evolving_model(initial_active_agent_ids=("a",))
    state = initialize_evolving_population(model, beliefs={"a": 1.0})
    self.assertEqual(tuple(item.agent_id for item in state.members), ("a", "b", "c"))
    self.assertEqual(tuple(item.trust for item in state.edge_trust), (0.5, 0.5))
    self.assertEqual(tuple(item.active for item in state.edge_topology), (True, False))
```

- [ ] **Step 2: Run contract tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_evolving_contracts -v`

Expected: import failure because unified evolving contracts do not exist.

- [ ] **Step 3: Implement immutable integrated contracts**

```python
@dataclass(frozen=True)
class EvolvingNetworkModel:
    model_id: str
    version: str
    base_model: NetworkABMModel
    initial_active_agent_ids: tuple[str, ...]
    learning_rate: float
    initial_trust: float
    dissolution_similarity: float
    formation_similarity: float

@dataclass(frozen=True)
class EvolvingPopulationState:
    model_id: str
    model_hash: str
    round_index: int
    parent_state_hash: str | None
    members: tuple[LifecycleMemberState, ...]
    edge_trust: tuple[EdgeTrustState, ...]
    edge_topology: tuple[EdgeTopologyState, ...]
```

Validate threshold ordering, initial membership, parent chains, unique member/edge identities, and tuple types. Initialize all dimensions from one V1 population snapshot; force inactive broadcasting off while retaining seeded belief and exposure.

- [ ] **Step 4: Run contract tests and confirm GREEN**

Run: `python -m unittest tests.test_network_abm_evolving_contracts -v`

Expected: all unified contract tests pass.

- [ ] **Step 5: Commit unified contracts**

```text
git add docs/superpowers/specs/2026-08-31-unified-evolving-network-v5-design.md docs/superpowers/plans/2026-08-31-unified-evolving-network-v5.md narrative_dynamics/abm tests/evolving_fixtures.py tests/test_network_abm_evolving_contracts.py
git commit -m "feat: define unified evolving network state"
```

### Task 2: Atomic unified round and trajectory

**Files:**
- Create: `narrative_dynamics/abm/evolving.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Test: `tests/test_network_abm_evolving.py`

**Interfaces:**
- Consumes: Task 1 contracts, `PopulationLifecycleEvent`, `TruthFeedback`, `EdgeTrustUpdate`, `EdgeRewiringUpdate`, `InformationTransmission`, and V1 `simulate_round`.
- Produces: `EvolvingRoundResult`, `EvolvingTrajectory`, `evolving_active_population_view`, `simulate_evolving_round`, and `simulate_evolving_population`.

- [ ] **Step 1: Write failing composition and timing tests**

```python
def test_entry_propagation_learning_and_rewiring_share_one_next_state():
    model, initial = evolving_case(
        initial_active_agent_ids=("a", "c"),
        beliefs={"a": 1.0, "c": 0.5},
    )
    result = simulate_evolving_round(
        model,
        initial,
        events=(PopulationLifecycleEvent("b", LifecycleEventKind.ENTER),),
        feedback=(TruthFeedback(EdgeSelector("a", "b", "peer"), 1.0),),
    )
    self.assertEqual(member(result.next_state, "b").belief, 0.5)
    self.assertEqual(trust(result.next_state, "a", "b"), 0.75)
    self.assertTrue(topology(result.next_state, "b", "c").active)
```

- [ ] **Step 2: Run unified runtime tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_evolving -v`

Expected: import failure for unified runtime interfaces.

- [ ] **Step 3: Implement lifecycle-first active trusted propagation**

Canonicalize and validate one event per agent, apply allowed status transitions, induce the post-event active roster, retain only prior-active active-to-active edges, and scale base influence by prior trust. Bypass V1 when no agent is active.

```python
effective_influence = base_edge.influence * prior_trust[base_edge.identity]
```

- [ ] **Step 4: Implement post-propagation trust learning and active-edge rewiring**

Reject feedback without an exact same-round transmission. Apply the existing prediction-error rule and update only feedback edges. Rewire only candidate edges whose endpoints remain active; preserve incident edge state for inactive/dead endpoints.

```python
accuracy = 1.0 - abs(transmission.signal - feedback.observed_truth)
next_trust = prior.trust + model.learning_rate * (accuracy - prior.trust)
similarity = 1.0 - abs(source.belief - target.belief)
```

- [ ] **Step 5: Implement and test paired multi-round schedules**

```python
trajectory = simulate_evolving_population(
    model,
    initial,
    event_schedule=((enter_b,), ()),
    feedback_schedule=((feedback_ab,), ()),
)
self.assertEqual(trajectory.final_state.round_index, 2)
self.assertEqual(trajectory.rounds[0].next_state, trajectory.rounds[1].prior_state)
```

Require both schedules to be non-empty tuples with equal lengths and tuple entries. Test zero-active rounds, invalid feedback after exit, dead re-entry, canonical replay, exact coverage, and delayed trust/topology effects.

Run: `python -m unittest tests.test_network_abm_evolving -v`

Expected: all composition, ordering, retention, validation, and trajectory tests pass.

- [ ] **Step 6: Commit unified execution**

```text
git add narrative_dynamics/abm tests/test_network_abm_evolving.py
git commit -m "feat: run unified evolving network rounds"
```

### Task 3: Cross-layer metrics, public API, and documentation

**Files:**
- Create: `narrative_dynamics/abm/evolving_metrics.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `README.md`
- Modify: `tests/test_network_abm_public_api.py`
- Test: `tests/test_network_abm_evolving_metrics.py`

**Interfaces:**
- Consumes: evolving model/state and Task 2 exact model-state validation.
- Produces: `EvolvingSystemMetrics` and `measure_evolving_system`, plus the complete V5 public surface.

- [ ] **Step 1: Write a hand-computed cross-layer metric test**

```python
def test_metrics_cover_membership_cognition_trust_and_structure():
    metrics = measure_evolving_system(model, state)
    self.assertEqual(metrics.active_population, 3)
    self.assertEqual(metrics.mean_active_belief, 2 / 3)
    self.assertEqual(metrics.mean_trust, 0.625)
    self.assertEqual(metrics.learned_edge_rate, 0.5)
    self.assertEqual(metrics.effective_active_edge_count, 2)
    self.assertEqual(metrics.cumulative_rewirings, 1)
```

- [ ] **Step 2: Run metric tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_evolving_metrics -v`

Expected: import failure for evolving system metrics.

- [ ] **Step 3: Implement validated cross-layer metrics**

Compute active adoption against base agent thresholds, return `None` for active belief/adoption when active population is empty, measure trust over all candidate edges, count an effective edge only when topology is active and both endpoints are active, and use the complete edge catalog as the rate denominator.

```python
effective = tuple(
    edge for edge in state.edge_topology
    if edge.active
    and edge.edge.source_agent_id in active_ids
    and edge.edge.target_agent_id in active_ids
)
```

- [ ] **Step 4: Extend public exports and add an executable V5 README example**

Document one relay entering, receiving a source signal, producing accurate feedback learning, and forming its inactive candidate edge in one unified round. Assert next-state belief, trust, topology, and cross-layer metrics. Extend the exact `narrative_dynamics.abm.__all__` test and execute every README Python block independently.

- [ ] **Step 5: Run V1–V5 and targeted narrative regressions**

Run:

```text
python -m unittest discover -s tests -p "test_network_abm*.py"
python -m unittest tests.test_narrative_simulation tests.test_narrative_stochastic_simulation tests.test_narrative_runtime_cognition
python -m compileall -q narrative_dynamics tests
git diff origin/proof/narrative-dynamics-v0 --check
```

Expected: all network and targeted narrative tests pass; compilation and diff checks exit zero.

- [ ] **Step 6: Commit, push, and update the existing PR**

```text
git add README.md narrative_dynamics/abm tests/test_network_abm_public_api.py tests/test_network_abm_evolving_metrics.py
git commit -m "docs: expose unified evolving network engine"
git push origin feature/network-interaction-emergence-v1
```
