# Dynamic Roles V7 Implementation Plan

> **For agentic workers:** Execute task-by-task with strict red/green tests and frequent commits. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let agents deterministically change behavioral roles from local post-round evidence while preserving V6 timing, finite resources, replay, and auditability.

**Architecture:** A V7 wrapper stores current role and tenure beside the exact V6 state. V6 receives a validated internal role map for entrant sharing and local decisions. After the embedded autonomous round, V7 evaluates canonical directed rules for agents still active, emits evidence-rich transition records, and commits role changes for the following round. State-derived metrics expose role composition, diversity, transition rate, and tenure.

**Tech Stack:** Python 3 standard library, frozen dataclasses, V6 autonomy runtime, stable content hashing, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-dynamic-roles-v7-design.md`

## Global Constraints

- Immutable catalog identity and mutable role assignment remain separate.
- The prior dynamic role governs all decisions in the current round.
- Role transitions use only post-round local belief, local cumulative counters, and local role tenure.
- At most one transition occurs per active agent per round; smallest unique priority wins.
- Exited/dead/inactive agents do not transition and do not accrue tenure.
- Role changes never replenish verification budget.
- V6 public behavior remains unchanged when no role override is supplied.
- Every public tuple, result, state, and hash is canonical and input-order invariant.

---

### Task 1: Dynamic-role policy, state, and audit contracts

**Files:**
- Create: `narrative_dynamics/abm/role_contracts.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Create: `tests/dynamic_role_fixtures.py`
- Test: `tests/test_network_abm_role_contracts.py`

**Interfaces:**
- Consumes: `AutonomousNetworkModel`, `AutonomousPopulationState`, V6 initialization, lifecycle status, validation helpers, and stable hashing.
- Produces: `RoleTransitionRule`, `RoleTransitionRecord`, `DynamicRoleAgentState`, `DynamicRoleModel`, `DynamicRolePopulationState`, and `initialize_dynamic_role_population`.

- [ ] **Step 1: Write failing initialization, canonicalization, and validation tests**

```python
def test_initial_roles_come_from_profiles_with_zero_tenure_and_transitions():
    model = dynamic_role_model()
    state = initialize_dynamic_role_population(model, beliefs={"a": 1.0, "c": 0.5})
    self.assertEqual(role_state(state, "b").current_role, "relay")
    self.assertEqual(role_state(state, "b").rounds_in_role, 0)
    self.assertEqual(role_state(state, "b").transition_count, 0)
```

- [ ] **Step 2: Run contract tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_role_contracts -v`

Expected: import failure because V7 role contracts do not exist.

- [ ] **Step 3: Implement immutable rules, records, role state, model, and initialization**

Validate role references, non-self transitions, belief intervals, substantive conditions, unique `(from_role, priority)` pairs, exact agent coverage, wrapper/embedded round equality, initial profile roles, and canonical tuple order. Add `to_dict()` and `content_hash` to every public value.

- [ ] **Step 4: Run contract tests and confirm GREEN**

Run: `python -m unittest tests.test_network_abm_role_contracts -v`

- [ ] **Step 5: Commit V7 contracts and design**

```text
git add docs/superpowers/specs/2026-08-31-dynamic-roles-v7-design.md docs/superpowers/plans/2026-08-31-dynamic-roles-v7.md narrative_dynamics/abm tests/dynamic_role_fixtures.py tests/test_network_abm_role_contracts.py
git commit -m "feat: define dynamic role contracts"
```

### Task 2: Role-aware autonomy and transition runtime

**Files:**
- Modify: `narrative_dynamics/abm/autonomy.py`
- Create: `narrative_dynamics/abm/roles.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Test: `tests/test_network_abm_roles.py`
- Test: `tests/test_network_abm_autonomy.py`

**Interfaces:**
- Consumes: Task 1 contracts, V6 autonomous schedules/results, and the V6 local decision runtime.
- Produces: `DynamicRoleRoundResult`, `DynamicRoleTrajectory`, `simulate_dynamic_role_round`, and `simulate_dynamic_role_population`.

- [ ] **Step 1: Write failing delayed-role-effect and evidence tests**

```python
def test_verified_relay_transitions_after_relay_policy_governs_round():
    result = simulate_dynamic_role_round(
        model,
        initial,
        environment_events=(enter_b,),
        truth_observations=(truth_ab,),
    )
    self.assertEqual(intent_for(result.autonomy_result, "b").role, "relay")
    self.assertEqual(transition_for(result, "b").to_role, "source")
    self.assertEqual(role_state(result.next_state, "b").current_role, "source")
```

- [ ] **Step 2: Run runtime tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_roles -v`

- [ ] **Step 3: Add the validated internal V6 role-override seam**

Refactor the internal round implementation to accept an exact role map. Apply it only to entrant sharing and role-policy lookup in `_decide`; retain catalogued susceptibility, broadcast threshold, budget history, and the unchanged public V6 function. Rerun `tests.test_network_abm_autonomy` to prove V6 compatibility.

- [ ] **Step 4: Implement deterministic post-round transitions**

Evaluate only agents active in the embedded next state. Use `rounds_in_role + 1` as completed tenure for the just-finished round, plus post-round belief and updated V6 counters. Select the smallest-priority matching rule, emit one record, set new-role tenure to zero, or increment unchanged-role tenure. Preserve inactive/dead role state unchanged.

- [ ] **Step 5: Implement paired multi-round schedules and replay validation**

Test next-round policy effect, budget preservation, exit/death suppression, retained-role re-entry, priority, canonical input order, no-match tenure, zero-active rounds, model/state mismatch, and schedule lengths/types.

Run:

```text
python -m unittest tests.test_network_abm_roles tests.test_network_abm_autonomy -v
```

- [ ] **Step 6: Commit the V7 runtime**

```text
git add narrative_dynamics/abm tests/test_network_abm_roles.py tests/test_network_abm_autonomy.py
git commit -m "feat: evolve agent roles"
```

### Task 3: Role-emergence metrics, API, and documentation

**Files:**
- Create: `narrative_dynamics/abm/role_metrics.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `README.md`
- Modify: `tests/test_network_abm_public_api.py`
- Test: `tests/test_network_abm_role_metrics.py`

**Interfaces:**
- Consumes: exact V7 model/state validation and V6 cumulative decision counters.
- Produces: `RolePopulationCount`, `RoleDynamicsMetrics`, and `measure_role_dynamics`.

- [ ] **Step 1: Write failing hand-computed composition and diversity tests**

```python
def test_role_metrics_cover_composition_diversity_transitions_and_tenure():
    metrics = measure_role_dynamics(model, transitioned_state)
    self.assertEqual(role_count(metrics, "source").count, 2)
    self.assertEqual(metrics.transition_count, 1)
    self.assertEqual(metrics.transition_rate, 1 / 3)
```

- [ ] **Step 2: Run metric tests and confirm RED**

Run: `python -m unittest tests.test_network_abm_role_metrics -v`

- [ ] **Step 3: Implement validated role-level emergence metrics**

Emit all configured roles in canonical order. Compute active shares, normalized Shannon entropy, cumulative transitions per cumulative decision, and mean active tenure with explicit empty-population and one-role behavior.

- [ ] **Step 4: Extend exact public exports and add executable V7 README example**

Show a relay receiving/verifying information, transitioning to source, retaining its remaining budget, and being governed by the source role on the following round. Assert the transition record and role metrics. Execute every README Python block independently.

- [ ] **Step 5: Run V1–V7 and targeted narrative regressions**

Run:

```text
python -m unittest discover -s tests -p "test_network_abm*.py"
python -m unittest tests.test_narrative_simulation tests.test_narrative_stochastic_simulation tests.test_narrative_runtime_cognition
python -m compileall -q narrative_dynamics tests
git diff origin/proof/narrative-dynamics-v0 --check
```

- [ ] **Step 6: Commit, push, and update PR #44**

```text
git add README.md narrative_dynamics/abm tests/test_network_abm_public_api.py tests/test_network_abm_role_metrics.py
git commit -m "docs: expose dynamic roles"
git push origin feature/network-interaction-emergence-v1
```
