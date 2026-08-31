# Network Adaptive Trust V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add feedback-driven edge trust learning so source reliability changes the strength of future network information.

**Architecture:** Focused adaptive-contract, learning-runtime, and trust-metric modules wrap the V1 model without changing V1 contracts. Adaptive states own edge trust, each round derives effective edge influence from the prior trust snapshot, V1 synchronous propagation updates beliefs, and post-round truth feedback updates trust for the following round.

**Tech Stack:** Python 3 standard library, frozen dataclasses, V1 ABM contracts and propagation, existing stable content hashing, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-network-adaptive-trust-v2-design.md`

## Global Constraints

- V2 remains dependency-free and deterministic.
- The population roster, role labels, and topology identities remain fixed.
- Trust is edge-local, persistent, finite, and bounded in `[0, 1]`.
- Feedback updates occur after current-round propagation and affect only later rounds.
- Feedback can target only an edge that transmitted in the same round.
- Canonical ordering and stable hashes are required for every public artifact.

---

### Task 1: Adaptive model and state contracts

**Files:**
- Create: `narrative_dynamics/abm/adaptive_contracts.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Test: `tests/test_network_abm_learning_contracts.py`

**Interfaces:**
- Consumes: `NetworkABMModel`, `NetworkAgentState`, `PopulationState`, `EdgeSelector`, and `initialize_population`.
- Produces: `AdaptiveTrustModel`, `EdgeTrustState`, `AdaptivePopulationState`, `TruthFeedback`, and `initialize_adaptive_population`.

- [ ] **Step 1: Write failing canonical contract tests**

```python
def test_adaptive_initialization_covers_every_edge_and_agent():
    base, _ = learning_case()
    model = AdaptiveTrustModel("adaptive", "1", base, 0.5, 0.5)
    state = initialize_adaptive_population(model, beliefs={"accurate": 1.0})
    self.assertEqual(tuple(item.trust for item in state.edge_trust), (0.5, 0.5))
    self.assertEqual(tuple(item.feedback_count for item in state.edge_trust), (0, 0))
```

- [ ] **Step 2: Run contract tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_learning_contracts -v`

Expected: import failure because adaptive trust contracts do not exist.

- [ ] **Step 3: Implement immutable adaptive contracts**

```python
@dataclass(frozen=True)
class EdgeTrustState:
    edge: EdgeSelector
    trust: float
    feedback_count: int = 0

@dataclass(frozen=True)
class AdaptivePopulationState:
    model_id: str
    model_hash: str
    round_index: int
    parent_state_hash: str | None
    agents: tuple[NetworkAgentState, ...]
    edge_trust: tuple[EdgeTrustState, ...]
```

Validate exact roster and edge coverage in `initialize_adaptive_population`, canonicalize tuples, and implement `to_dict()` plus `content_hash` for every public contract.

- [ ] **Step 4: Run contract tests and confirm GREEN**

Run: `python -m unittest tests.test_network_abm_learning_contracts -v`

Expected: all adaptive contract tests pass.

- [ ] **Step 5: Commit adaptive contracts**

```text
git add docs/superpowers/specs/2026-08-31-network-adaptive-trust-v2-design.md docs/superpowers/plans/2026-08-31-network-adaptive-trust-v2.md narrative_dynamics/abm tests/test_network_abm_learning_contracts.py
git commit -m "feat: define adaptive trust state"
```

### Task 2: Feedback-driven adaptive rounds

**Files:**
- Create: `narrative_dynamics/abm/learning.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Test: `tests/test_network_abm_learning.py`

**Interfaces:**
- Consumes: Task 1 contracts and V1 `simulate_round`.
- Produces: `EdgeTrustUpdate`, `AdaptiveRoundResult`, `AdaptiveTrajectory`, `simulate_adaptive_round`, `simulate_adaptive_population`, and `population_view`.

- [ ] **Step 1: Write failing timing and learning tests**

```python
def test_feedback_changes_next_round_influence_not_current_round():
    model, initial = adaptive_case()
    first = simulate_adaptive_round(model, initial, feedback=truth_feedback())
    self.assertEqual(agent(first.next_state, "target").belief, 0.5)
    self.assertEqual(trust(first.next_state, "accurate", "target"), 0.75)
    self.assertEqual(trust(first.next_state, "inaccurate", "target"), 0.25)
    second = simulate_adaptive_round(model, first.next_state)
    self.assertEqual(agent(second.next_state, "target").belief, 0.75)
```

- [ ] **Step 2: Run learning tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_learning -v`

Expected: import failure for adaptive simulation interfaces.

- [ ] **Step 3: Implement effective influence and post-round learning**

```python
accuracy = 1.0 - abs(transmission.signal - feedback.observed_truth)
new_trust = old.trust + model.learning_rate * (accuracy - old.trust)
```

Build effective edges with `base influence * prior trust`, call the V1 synchronous round, reject feedback without a matching transmission, update trust after belief propagation, and bind the adaptive next state to the prior adaptive hash.

- [ ] **Step 4: Implement and test multi-round schedules**

```python
trajectory = simulate_adaptive_population(
    model,
    initial,
    feedback_schedule=(truth_feedback(), ()),
)
self.assertEqual(trajectory.final_state.round_index, 2)
self.assertEqual(trajectory.rounds[0].next_state, trajectory.rounds[1].prior_state)
```

Run: `python -m unittest tests.test_network_abm_learning -v`

Expected: timing, validation, replay, and schedule tests all pass.

- [ ] **Step 5: Commit adaptive execution**

```text
git add narrative_dynamics/abm tests/test_network_abm_learning.py
git commit -m "feat: learn trust from truth feedback"
```

### Task 3: Trust metrics, public API, and documentation

**Files:**
- Create: `narrative_dynamics/abm/adaptive_metrics.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `README.md`
- Modify: `tests/test_network_abm_public_api.py`
- Test: `tests/test_network_abm_learning_metrics.py`

**Interfaces:**
- Consumes: adaptive model and state.
- Produces: `AdaptiveTrustMetrics` and `measure_adaptive_trust`, plus the complete V2 public surface.

- [ ] **Step 1: Write hand-computed trust metric tests**

```python
def test_adaptive_trust_metrics_match_feedback_outcome():
    result = learned_round()
    metrics = measure_adaptive_trust(result.model, result.next_state)
    self.assertEqual(metrics.mean_trust, 0.5)
    self.assertEqual(metrics.min_trust, 0.25)
    self.assertEqual(metrics.max_trust, 0.75)
    self.assertEqual(metrics.learned_edge_rate, 1.0)
```

- [ ] **Step 2: Run metric tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_learning_metrics -v`

Expected: import failure for adaptive trust metrics.

- [ ] **Step 3: Implement bounded trust metrics and extend public exports**

```python
return AdaptiveTrustMetrics(
    mean_trust=sum(trusts) / len(trusts),
    min_trust=min(trusts),
    max_trust=max(trusts),
    learned_edge_rate=sum(item.feedback_count > 0 for item in state.edge_trust) / len(trusts),
)
```

Extend `narrative_dynamics.abm.__all__` and its exact public API test with every V2 contract and function.

- [ ] **Step 4: Document and execute the adaptive example**

Add a README example with accurate and inaccurate sources, round-one feedback, and round-two target belief `0.75`. Execute every README Python block independently.

- [ ] **Step 5: Run V1/V2 and narrative regression suites**

Run:

```text
python -m unittest discover -s tests -p "test_network_abm*.py" -v
python -m unittest tests.test_narrative_simulation tests.test_narrative_stochastic_simulation tests.test_narrative_runtime_cognition -v
```

Expected: all V1, V2, and targeted narrative regression tests pass.

- [ ] **Step 6: Commit V2 metrics and docs**

```text
git add README.md narrative_dynamics/abm tests/test_network_abm_public_api.py tests/test_network_abm_learning_metrics.py
git commit -m "docs: expose adaptive trust learning"
```
