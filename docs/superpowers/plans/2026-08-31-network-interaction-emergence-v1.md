# Network Interaction and Emergence V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic network-constrained population ABM with persistent agent belief changes, synchronous multi-round propagation, emergence metrics, and paired interventions.

**Architecture:** Add a focused `narrative_dynamics.abm` package instead of widening the existing narrative scheduler. Immutable contracts own canonical identity, the simulation module owns synchronous state transitions and lineage, the metrics module owns the micro-to-macro projection, and the interventions module owns pure baseline/treatment construction and comparison.

**Tech Stack:** Python 3 standard library, frozen dataclasses, existing `stable_content_hash`, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-network-interaction-emergence-v1-design.md`

## Global Constraints

- The implementation remains dependency-free.
- V1 uses a fixed population roster and fixed role labels.
- All rounds are synchronous and deterministic.
- Information travels only over active positive-influence directed edges.
- Every public artifact has canonical ordering, strict validation, and a stable content hash.
- Population lifecycle, network rewiring, parameter learning, and stochastic contact are deferred.

---

### Task 1: Immutable ABM contracts

**Files:**
- Create: `narrative_dynamics/abm/contracts.py`
- Create: `narrative_dynamics/abm/__init__.py`
- Test: `tests/test_network_abm_contracts.py`

**Interfaces:**
- Consumes: `narrative_dynamics.contracts.stable_content_hash`.
- Produces: `NetworkAgentSpec`, `SocialEdge`, `SocialNetwork`, `NetworkABMModel`, `NetworkAgentState`, `PopulationState`, and `initialize_population`.

- [ ] **Step 1: Write contract tests**

```python
def test_model_canonicalizes_agent_and_edge_order():
    first = make_model(reverse=False)
    second = make_model(reverse=True)
    self.assertEqual(first, second)
    self.assertEqual(first.content_hash, second.content_hash)

def test_contracts_reject_duplicate_unknown_and_invalid_values():
    with self.assertRaises(ValueError):
        SocialEdge("a", "a", "peer", 1.0)
    with self.assertRaises(ValueError):
        NetworkAgentSpec("a", "source", float("nan"), 0.5, 0.5)
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m unittest tests.test_network_abm_contracts -v`

Expected: import failure because `narrative_dynamics.abm` does not exist.

- [ ] **Step 3: Implement canonical contracts**

```python
@dataclass(frozen=True)
class SocialEdge:
    source_agent_id: str
    target_agent_id: str
    relation_type: str
    influence: float = 1.0
    active: bool = True

@dataclass(frozen=True)
class PopulationState:
    model_id: str
    model_hash: str
    round_index: int
    parent_state_hash: str | None
    agents: tuple[NetworkAgentState, ...]
```

Validate scalar types and ranges, reject duplicate identities, canonicalize all tuples, implement `to_dict()` and `content_hash`, and ensure `initialize_population(model, beliefs={})` binds every model agent exactly once.

- [ ] **Step 4: Run contract tests to verify GREEN**

Run: `python -m unittest tests.test_network_abm_contracts -v`

Expected: all tests pass.

- [ ] **Step 5: Commit contracts**

```text
git add .gitignore narrative_dynamics/abm tests/test_network_abm_contracts.py docs/superpowers/specs/2026-08-31-network-interaction-emergence-v1-design.md docs/superpowers/plans/2026-08-31-network-interaction-emergence-v1.md
git commit -m "feat: define network ABM contracts"
```

### Task 2: Synchronous local propagation

**Files:**
- Create: `narrative_dynamics/abm/simulation.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Test: `tests/test_network_abm_simulation.py`

**Interfaces:**
- Consumes: `NetworkABMModel`, `PopulationState`, `NetworkAgentState`.
- Produces: `InformationTransmission`, `NetworkRoundResult`, `PopulationTrajectory`, `simulate_round`, and `simulate_population`.

- [ ] **Step 1: Write line-network and replay tests**

```python
def test_information_moves_at_most_one_hop_per_round():
    model, initial = line_case()
    first = simulate_round(model, initial)
    self.assertEqual(state(first.next_state, "b").belief, 1.0)
    self.assertEqual(state(first.next_state, "c").belief, 0.0)
    second = simulate_round(model, first.next_state)
    self.assertEqual(state(second.next_state, "c").belief, 1.0)

def test_replay_and_input_order_are_hash_identical():
    first = simulate_population(*line_case(reverse=False), rounds=2)
    second = simulate_population(*line_case(reverse=True), rounds=2)
    self.assertEqual(first.content_hash, second.content_hash)
```

- [ ] **Step 2: Run simulation tests to verify RED**

Run: `python -m unittest tests.test_network_abm_simulation -v`

Expected: import failure for the missing simulation interfaces.

- [ ] **Step 3: Implement synchronous rounds**

```python
def _updated_belief(profile, prior, incoming):
    total = sum(item.influence for item in incoming)
    signal = sum(item.influence * item.signal for item in incoming) / total
    assimilation = profile.receptivity * min(1.0, total)
    return prior.belief + assimilation * (signal - prior.belief)
```

Build all transmissions from the prior snapshot, group them by target, then construct all next states. Validate exact model binding and parent lineage before returning immutable round and trajectory artifacts.

- [ ] **Step 4: Run simulation tests to verify GREEN**

Run: `python -m unittest tests.test_network_abm_simulation -v`

Expected: all tests pass.

- [ ] **Step 5: Commit synchronous simulation**

```text
git add narrative_dynamics/abm tests/test_network_abm_simulation.py
git commit -m "feat: add synchronous network propagation"
```

### Task 3: Macro emergence metrics

**Files:**
- Create: `narrative_dynamics/abm/metrics.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Test: `tests/test_network_abm_metrics.py`

**Interfaces:**
- Consumes: `NetworkABMModel`, `PopulationState`.
- Produces: `EmergenceMetrics` and `measure_emergence`.

- [ ] **Step 1: Write hand-computed metric tests**

```python
def test_split_population_metrics_are_exact():
    model, state = four_agent_state((1.0, 1.0, 0.0, 0.0))
    metrics = measure_emergence(model, state)
    self.assertEqual(metrics.mean_belief, 0.5)
    self.assertEqual(metrics.adoption_rate, 0.5)
    self.assertEqual(metrics.consensus, 0.0)
    self.assertEqual(metrics.polarization, 1.0)
```

- [ ] **Step 2: Run metric tests to verify RED**

Run: `python -m unittest tests.test_network_abm_metrics -v`

Expected: import failure for `EmergenceMetrics`.

- [ ] **Step 3: Implement the metric projection**

```python
mean = sum(beliefs) / len(beliefs)
polarization = 4.0 * sum((value - mean) ** 2 for value in beliefs) / len(beliefs)
consensus = 1.0 - (max(beliefs) - min(beliefs))
```

Compute all six specified metrics, validate exact model/state binding, expose canonical serialization and content hashing, and clamp floating boundary noise to `[0, 1]`.

- [ ] **Step 4: Run metric tests to verify GREEN**

Run: `python -m unittest tests.test_network_abm_metrics -v`

Expected: all tests pass.

- [ ] **Step 5: Commit metrics**

```text
git add narrative_dynamics/abm tests/test_network_abm_metrics.py
git commit -m "feat: measure network emergence"
```

### Task 4: Paired interventions and null control

**Files:**
- Create: `narrative_dynamics/abm/interventions.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Test: `tests/test_network_abm_interventions.py`

**Interfaces:**
- Consumes: model/state contracts, `simulate_population`, and `measure_emergence`.
- Produces: `EdgeSelector`, `EdgeInfluenceChange`, `BeliefSeed`, `NetworkIntervention`, `AppliedIntervention`, `InterventionComparison`, `apply_intervention`, `compare_intervention`, and `no_propagation_intervention`.

- [ ] **Step 1: Write paired intervention tests**

```python
def test_disabling_bridge_blocks_second_hop_and_reports_delta():
    model, initial = line_case()
    intervention = NetworkIntervention(
        "cut-bridge",
        disabled_edges=(EdgeSelector("b", "c", "peer"),),
    )
    comparison = compare_intervention(model, initial, intervention, rounds=2)
    self.assertEqual(agent(comparison.baseline.final_state, "c").belief, 1.0)
    self.assertEqual(agent(comparison.treatment.final_state, "c").belief, 0.0)
    self.assertEqual(comparison.metric_deltas["adoption_rate"], -1 / 3)
```

- [ ] **Step 2: Run intervention tests to verify RED**

Run: `python -m unittest tests.test_network_abm_interventions -v`

Expected: import failure for intervention interfaces.

- [ ] **Step 3: Implement pure intervention application and comparison**

```python
def compare_intervention(model, initial_state, intervention, *, rounds):
    applied = apply_intervention(model, initial_state, intervention)
    baseline = simulate_population(model, initial_state, rounds=rounds)
    treatment = simulate_population(applied.model, applied.initial_state, rounds=rounds)
    baseline_metrics = measure_emergence(model, baseline.final_state)
    treatment_metrics = measure_emergence(applied.model, treatment.final_state)
    return InterventionComparison.from_runs(
        intervention, baseline, treatment, baseline_metrics, treatment_metrics
    )
```

Reject unknown, duplicate, and conflicting edge operations. Implement `no_propagation_intervention` by disabling every active edge. Preserve baseline objects unchanged and report treatment-minus-baseline deltas for every metric.

- [ ] **Step 4: Run intervention tests to verify GREEN**

Run: `python -m unittest tests.test_network_abm_interventions -v`

Expected: all tests pass.

- [ ] **Step 5: Commit interventions**

```text
git add narrative_dynamics/abm tests/test_network_abm_interventions.py
git commit -m "feat: compare network interventions"
```

### Task 5: Public documentation and regression verification

**Files:**
- Modify: `README.md`
- Modify: `narrative_dynamics/abm/__init__.py`
- Test: `tests/test_network_abm_public_api.py`

**Interfaces:**
- Consumes: every V1 public contract and function.
- Produces: stable `narrative_dynamics.abm` public API and an executable README example.

- [ ] **Step 1: Write the public API test**

```python
def test_public_api_exports_the_complete_v1_surface():
    import narrative_dynamics.abm as abm
    self.assertEqual(
        set(abm.__all__),
        {"NetworkAgentSpec", "SocialEdge", "SocialNetwork", "NetworkABMModel",
         "NetworkAgentState", "PopulationState", "initialize_population",
         "InformationTransmission", "NetworkRoundResult", "PopulationTrajectory",
         "simulate_round", "simulate_population", "EmergenceMetrics",
         "measure_emergence", "EdgeSelector", "EdgeInfluenceChange", "BeliefSeed",
         "NetworkIntervention", "AppliedIntervention", "InterventionComparison",
         "apply_intervention", "compare_intervention", "no_propagation_intervention"},
    )
```

- [ ] **Step 2: Run the public API test to verify RED**

Run: `python -m unittest tests.test_network_abm_public_api -v`

Expected: failure until `__all__` is complete.

- [ ] **Step 3: Document the executable line-network example**

Add a README section that constructs `A -> B -> C`, seeds `A`, runs two rounds, prints final emergence metrics, and compares a disabled `B -> C` intervention. Ensure every imported name belongs to `narrative_dynamics.abm.__all__`.

- [ ] **Step 4: Run feature and targeted regression suites**

Run:

```text
python -m unittest discover -s tests -p "test_network_abm*.py" -v
python -m unittest tests.test_narrative_simulation tests.test_narrative_stochastic_simulation tests.test_narrative_runtime_cognition -v
```

Expected: all feature and targeted regression tests pass.

- [ ] **Step 5: Run full verification and record pre-existing failures separately**

Run: `python -m unittest discover -s tests -v`

Expected: all tests unrelated to already observed measurement-validity evidence/fixture failures pass; any baseline failure must be reported verbatim and must not be attributed to this feature.

- [ ] **Step 6: Commit documentation and public surface**

```text
git add README.md narrative_dynamics/abm tests/test_network_abm_public_api.py
git commit -m "docs: expose network ABM V1"
```
