# Endogenous Network Rewiring V4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make catalogued social relationships form and dissolve deterministically from post-propagation belief similarity, feeding structural change into later information rounds.

**Architecture:** A rewiring wrapper owns the existing V1 agent snapshot plus persistent activity state for every catalogued edge. Each round derives an effective V1 topology from prior edge state, delegates synchronous propagation, computes post-round similarity, and applies hysteretic formation/dissolution rules for the next round. A focused metric module measures density, similarity, churn, and weak fragmentation.

**Tech Stack:** Python 3 standard library, frozen dataclasses, V1 ABM contracts and propagation, existing stable content hashing, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-endogenous-network-rewiring-v4-design.md`

## Global Constraints

- The identity roster, profiles, and candidate directed-edge identities remain fixed.
- Dissolution and formation thresholds satisfy `0 <= dissolution < formation <= 1`.
- Current-round propagation uses only the prior topology snapshot.
- Rewiring uses post-propagation beliefs and affects only later rounds.
- Every state contains every catalogued agent and candidate edge exactly once.
- Zero active edges is a valid topology.
- Canonical ordering and stable hashes are required for every public artifact.

---

### Task 1: Rewiring model and persistent topology contracts

**Files:**
- Create: `narrative_dynamics/abm/rewiring_contracts.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Create: `tests/rewiring_fixtures.py`
- Test: `tests/test_network_abm_rewiring_contracts.py`

**Interfaces:**
- Consumes: `NetworkABMModel`, `NetworkAgentState`, `EdgeSelector`, `initialize_population`, validation helpers, and stable content hashing.
- Produces: `EndogenousRewiringModel`, `EdgeTopologyState`, `RewiringPopulationState`, and `initialize_rewiring_population`.

- [ ] **Step 1: Write failing canonical contract tests**

```python
def test_initialization_covers_every_agent_and_candidate_edge():
    model = rewiring_model()
    state = initialize_rewiring_population(model, beliefs={"a": 1.0})
    self.assertEqual(tuple(item.agent_id for item in state.agents), ("a", "b", "c"))
    self.assertEqual(tuple(item.active for item in state.edge_topology), (True, False))
    self.assertEqual(tuple(item.rewiring_count for item in state.edge_topology), (0, 0))
```

- [ ] **Step 2: Run contract tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_rewiring_contracts -v`

Expected: import failure because rewiring contracts do not exist.

- [ ] **Step 3: Implement immutable rewiring contracts**

```python
@dataclass(frozen=True)
class EdgeTopologyState:
    edge: EdgeSelector
    active: bool
    last_similarity: float
    rewiring_count: int = 0

@dataclass(frozen=True)
class EndogenousRewiringModel:
    model_id: str
    version: str
    base_model: NetworkABMModel
    dissolution_similarity: float
    formation_similarity: float
```

Initialize agents through V1, copy each base edge's activity, compute initial endpoint similarity, reject empty edge catalogs and invalid threshold order, canonicalize tuples, and implement `to_dict()` plus `content_hash` on every public contract.

- [ ] **Step 4: Run contract tests and confirm GREEN**

Run: `python -m unittest tests.test_network_abm_rewiring_contracts -v`

Expected: all rewiring contract tests pass.

- [ ] **Step 5: Commit rewiring contracts**

```text
git add docs/superpowers/specs/2026-08-31-endogenous-network-rewiring-v4-design.md docs/superpowers/plans/2026-08-31-endogenous-network-rewiring-v4.md narrative_dynamics/abm tests/rewiring_fixtures.py tests/test_network_abm_rewiring_contracts.py
git commit -m "feat: define endogenous topology state"
```

### Task 2: Post-propagation rewiring execution

**Files:**
- Create: `narrative_dynamics/abm/rewiring.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Test: `tests/test_network_abm_rewiring.py`

**Interfaces:**
- Consumes: Task 1 contracts, V1 `simulate_round`, base `SocialEdge` identities, and `InformationTransmission`.
- Produces: `EdgeRewiringUpdate`, `RewiringRoundResult`, `RewiringTrajectory`, `rewiring_population_view`, `simulate_rewiring_round`, and `simulate_rewiring_population`.

- [ ] **Step 1: Write failing timing and hysteresis tests**

```python
def test_dissolution_affects_following_round_not_current_transmission():
    model, initial = dissimilar_active_case()
    first = simulate_rewiring_round(model, initial)
    self.assertEqual(len(first.transmissions), 1)
    self.assertFalse(topology(first.next_state, "a", "b").active)
    second = simulate_rewiring_round(model, first.next_state)
    self.assertEqual(second.transmissions, ())
```

- [ ] **Step 2: Run execution tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_rewiring -v`

Expected: import failure for rewiring execution interfaces.

- [ ] **Step 3: Implement prior-topology propagation and post-round rewiring**

```python
similarity = 1.0 - abs(source.belief - target.belief)
if similarity >= model.formation_similarity:
    next_active = True
elif similarity <= model.dissolution_similarity:
    next_active = False
else:
    next_active = prior.active
```

Build an effective `NetworkABMModel` whose edge-active flags come from prior topology state, project the prior agents to a V1 state, call `simulate_round`, update every candidate edge from resulting beliefs, and increment `rewiring_count` exactly when activity changes.

- [ ] **Step 4: Implement trajectory schedules and exact replay validation**

```python
trajectory = simulate_rewiring_population(model, initial, rounds=2)
self.assertEqual(trajectory.final_state.round_index, 2)
self.assertEqual(trajectory.rounds[0].next_state, trajectory.rounds[1].prior_state)
self.assertEqual(trajectory.final_state, trajectory.rounds[-1].next_state)
```

Run: `python -m unittest tests.test_network_abm_rewiring -v`

Expected: dissolution timing, formation timing, hysteresis, zero-active-edge, counters, replay, and validation tests all pass.

- [ ] **Step 5: Commit rewiring execution**

```text
git add narrative_dynamics/abm tests/test_network_abm_rewiring.py
git commit -m "feat: rewire network from belief similarity"
```

### Task 3: Structural metrics, public API, and documentation

**Files:**
- Create: `narrative_dynamics/abm/rewiring_metrics.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `README.md`
- Modify: `tests/test_network_abm_public_api.py`
- Test: `tests/test_network_abm_rewiring_metrics.py`

**Interfaces:**
- Consumes: rewiring model/state and Task 2 exact model-state validation.
- Produces: `NetworkStructureMetrics` and `measure_network_structure`, plus the complete V4 public surface.

- [ ] **Step 1: Write hand-computed structural metric tests**

```python
def test_fragmented_topology_metrics_match_by_hand():
    metrics = measure_network_structure(model, state)
    self.assertEqual(metrics.catalog_edge_count, 2)
    self.assertEqual(metrics.active_edge_count, 1)
    self.assertEqual(metrics.active_edge_rate, 0.5)
    self.assertEqual(metrics.weak_component_count, 2)
    self.assertEqual(metrics.largest_component_share, 2 / 3)
```

- [ ] **Step 2: Run metric tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_rewiring_metrics -v`

Expected: import failure for network structure metrics.

- [ ] **Step 3: Implement dependency-free weak-component and churn metrics**

Build an undirected adjacency map only for component traversal, retain directed edges for density and similarity, count isolated agents as components, and validate metric count/share consistency in the frozen metric contract.

```python
while unseen:
    stack = [unseen.pop()]
    component_size = 0
    while stack:
        agent_id = stack.pop()
        component_size += 1
        neighbors = adjacency[agent_id] & unseen
        unseen.difference_update(neighbors)
        stack.extend(neighbors)
```

- [ ] **Step 4: Extend public exports and add an executable README example**

Document an initially inactive `B -> C` candidate edge that forms after beliefs converge, then transmits in the next round. Assert the active-edge count and cumulative rewiring metric. Extend the exact `narrative_dynamics.abm.__all__` test with every V4 symbol and execute every README Python block independently.

- [ ] **Step 5: Run network and targeted narrative regression suites**

Run:

```text
python -m unittest discover -s tests -p "test_network_abm*.py" -v
python -m unittest tests.test_narrative_simulation tests.test_narrative_stochastic_simulation tests.test_narrative_runtime_cognition
python -m compileall -q narrative_dynamics tests
git diff origin/proof/narrative-dynamics-v0 --check
```

Expected: all V1–V4 network tests and targeted narrative regressions pass; compilation and diff checks exit zero.

- [ ] **Step 6: Commit and push V4**

```text
git add README.md narrative_dynamics/abm tests/test_network_abm_public_api.py tests/test_network_abm_rewiring_metrics.py
git commit -m "docs: expose endogenous network rewiring"
git push origin feature/network-interaction-emergence-v1
```
