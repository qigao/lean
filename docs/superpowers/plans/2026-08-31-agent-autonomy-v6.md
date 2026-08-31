# Agent Autonomy V6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make active agents deterministically choose sharing, silence, verification, and voluntary exit from local information, belief, trust, role policy, and finite verification budgets.

**Architecture:** An autonomy wrapper binds a complete role-policy catalog to V5 and persists one resource/choice state per agent alongside the exact evolving state. Each round applies environment entry/death, propagates through prior sharing choices, derives one local action intent per active agent, learns only from selected truth observations, applies generated exits, rewires remaining active endpoints, and commits all layers to one chain. State-derived metrics summarize cumulative behavior and resource use.

**Tech Stack:** Python 3 standard library, frozen dataclasses, V1 propagation, V5 evolving value contracts, stable content hashing, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-agent-autonomy-v6-design.md`

## Global Constraints

- Every base role has exactly one deterministic role policy.
- Environment events may be `ENTER` or `DEATH`, never `EXIT`.
- Current propagation uses prior sharing choices, prior trust, and prior topology.
- Each active agent receives only its targeted transmissions and their prior trust.
- Each agent may verify at most one received edge per round and only with remaining budget.
- Truth observations cover exactly the edges selected for verification.
- Decisions, trust learning, autonomous exits, and rewiring affect only later propagation.
- Inactive/dead agents retain budgets/history, do not share, and produce no intent.
- Canonical ordering and stable hashes are required for every public artifact.

---

### Task 1: Autonomy policies, resources, and audit contracts

**Files:**
- Create: `narrative_dynamics/abm/autonomy_contracts.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Create: `tests/autonomy_fixtures.py`
- Test: `tests/test_network_abm_autonomy_contracts.py`

**Interfaces:**
- Consumes: `EvolvingNetworkModel`, `EvolvingPopulationState`, `EdgeSelector`, validation helpers, and stable content hashing.
- Produces: `SharingDecision`, `RoleDecisionPolicy`, `TruthObservation`, `AgentActionIntent`, `AutonomousAgentState`, `AutonomousNetworkModel`, `AutonomousPopulationState`, and `initialize_autonomous_population`.

- [ ] **Step 1: Write failing complete-policy and resource tests**

```python
def test_initialization_assigns_role_policy_budget_and_initial_choice():
    model = autonomous_model()
    state = initialize_autonomous_population(model, beliefs={"a": 1.0, "c": 0.5})
    self.assertTrue(resource(state, "a").sharing)
    self.assertEqual(resource(state, "a").remaining_verification_budget, 1)
    self.assertFalse(resource(state, "b").sharing)
    self.assertEqual(resource(state, "b").remaining_verification_budget, 2)
```

- [ ] **Step 2: Run contract tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_autonomy_contracts -v`

Expected: import failure because autonomy contracts do not exist.

- [ ] **Step 3: Implement immutable policy, intent, resource, model, and state contracts**

```python
@dataclass(frozen=True)
class RoleDecisionPolicy:
    role: str
    sharing_enabled: bool
    share_belief_threshold: float
    verify_trust_threshold: float
    exit_belief_threshold: float | None
    initial_verification_budget: int

@dataclass(frozen=True)
class AutonomousAgentState:
    agent_id: str
    sharing: bool
    remaining_verification_budget: int
    verification_count: int
    decision_count: int
    silent_decision_count: int
```

Validate numeric bounds, optional exit threshold, budget/counter consistency, complete role coverage, embedded/wrapper round identity, complete agent resource coverage, canonical tuples, and `to_dict()`/`content_hash` for every public artifact.

- [ ] **Step 4: Run contract tests and confirm GREEN**

Run: `python -m unittest tests.test_network_abm_autonomy_contracts -v`

Expected: all autonomy contract tests pass.

- [ ] **Step 5: Commit autonomy contracts**

```text
git add docs/superpowers/specs/2026-08-31-agent-autonomy-v6-design.md docs/superpowers/plans/2026-08-31-agent-autonomy-v6.md narrative_dynamics/abm tests/autonomy_fixtures.py tests/test_network_abm_autonomy_contracts.py
git commit -m "feat: define agent autonomy contracts"
```

### Task 2: Local autonomous decision runtime

**Files:**
- Create: `narrative_dynamics/abm/autonomy.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Test: `tests/test_network_abm_autonomy.py`

**Interfaces:**
- Consumes: Task 1 contracts, lifecycle events, V1 propagation, trust/topology update contracts, and V5 evolving contracts.
- Produces: `AutonomousRoundResult`, `AutonomousTrajectory`, `simulate_autonomous_round`, and `simulate_autonomous_population`.

- [ ] **Step 1: Write failing local verification and delayed-action tests**

```python
def test_relay_verifies_low_trust_and_spends_budget_after_receiving():
    result = simulate_autonomous_round(
        model,
        initial,
        environment_events=(enter_b,),
        truth_observations=(TruthObservation(edge_ab, 1.0),),
    )
    intent = intent_for(result, "b")
    self.assertEqual(intent.verification_edge, edge_ab)
    self.assertEqual(intent.verification_budget_before, 2)
    self.assertEqual(intent.verification_budget_after, 1)
    self.assertEqual(trust(result.next_state, edge_ab), 0.75)
```

- [ ] **Step 2: Run autonomy runtime tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_autonomy -v`

Expected: import failure for autonomous runtime interfaces.

- [ ] **Step 3: Implement environment filtering and sharing-constrained propagation**

Reject environment `EXIT`, apply `ENTER`/`DEATH`, recompute entrant sharing, induce active edges, suppress every outgoing edge whose source resource has `sharing=False`, scale remaining influence by prior trust, and delegate synchronous belief updates to V1. Bypass V1 for zero active agents.

- [ ] **Step 4: Implement deterministic local action selection**

```python
incoming = tuple(item for item in transmissions if item.target_agent_id == agent_id)
candidate = min(
    incoming,
    key=lambda item: (prior_trust[edge_identity(item)], edge_identity(item)),
    default=None,
)
verify = (
    candidate is not None
    and resource.remaining_verification_budget > 0
    and prior_trust[edge_identity(candidate)] <= policy.verify_trust_threshold
)
```

Use only post-round belief, incoming transmissions, corresponding prior trust, fixed role policy, and current budget. Emit exactly one intent for each post-environment active agent.

- [ ] **Step 5: Apply selected truth, autonomous exits, and remaining-active rewiring**

Require exact selected-observation coverage, update selected edge trust by the V2 learning rule, decrement budgets, generate canonical `EXIT` events for exit intents, mark those members inactive, retain incident edge state, and rewire only endpoints still active. Persist sharing choices and counters for next propagation.

- [ ] **Step 6: Implement paired multi-round schedules and replay tests**

```python
trajectory = simulate_autonomous_population(
    model,
    initial,
    environment_event_schedule=((enter_b,), ()),
    truth_observation_schedule=((truth_ab,), ()),
)
self.assertEqual(trajectory.final_state.round_index, 2)
self.assertEqual(trajectory.rounds[0].next_state, trajectory.rounds[1].prior_state)
```

Test exhausted budgets, silence affecting only the following round, autonomous exit, rejected external exit, exact truth coverage, incoming-only observation, canonical replay, zero-active rounds, and schedule validation.

Run: `python -m unittest tests.test_network_abm_autonomy -v`

Expected: all autonomy timing, locality, resource, exit, truth, and replay tests pass.

- [ ] **Step 7: Commit autonomy runtime**

```text
git add narrative_dynamics/abm tests/test_network_abm_autonomy.py
git commit -m "feat: execute autonomous agent decisions"
```

### Task 3: Behavior metrics, public API, and documentation

**Files:**
- Create: `narrative_dynamics/abm/autonomy_metrics.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `README.md`
- Modify: `tests/test_network_abm_public_api.py`
- Test: `tests/test_network_abm_autonomy_metrics.py`

**Interfaces:**
- Consumes: autonomy model/state and Task 2 exact validation.
- Produces: `AgentAutonomyMetrics` and `measure_agent_autonomy`, plus the complete V6 public surface.

- [ ] **Step 1: Write a hand-computed behavior metric test**

```python
def test_behavior_metrics_cover_decisions_resources_and_current_sharing():
    metrics = measure_agent_autonomy(model, state)
    self.assertEqual(metrics.decision_count, 3)
    self.assertEqual(metrics.verification_count, 1)
    self.assertEqual(metrics.verification_rate, 1 / 3)
    self.assertEqual(metrics.silent_decision_count, 1)
    self.assertEqual(metrics.autonomous_exit_count, 1)
    self.assertEqual(metrics.remaining_verification_budget, 3)
```

- [ ] **Step 2: Run behavior metric tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_autonomy_metrics -v`

Expected: import failure for autonomy behavior metrics.

- [ ] **Step 3: Implement validated cumulative and current behavior metrics**

Derive cumulative counters from agent resources, autonomous exits from embedded member exit counts, initial total budget from role policies applied to the fixed agent catalog, and active sharing from resources joined to active membership. Define zero-denominator rates as `0.0`.

```python
verification_rate = verification_count / decision_count if decision_count else 0.0
budget_utilization = (
    (initial_budget - remaining_budget) / initial_budget
    if initial_budget
    else 0.0
)
```

- [ ] **Step 4: Extend public exports and add an executable V6 README example**

Document a relay entering, receiving a low-trust source signal, selecting verification, spending one budget unit, and learning source trust. Assert its intent, next budget, learned trust, and behavior metrics. Extend the exact `narrative_dynamics.abm.__all__` test and execute every README Python block independently.

- [ ] **Step 5: Run V1–V6 and targeted narrative regressions**

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
git add README.md narrative_dynamics/abm tests/test_network_abm_public_api.py tests/test_network_abm_autonomy_metrics.py
git commit -m "docs: expose agent autonomy"
git push origin feature/network-interaction-emergence-v1
```
