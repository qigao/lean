# Finite Fault Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that any finite collection of faults selects one order-independent, duplicate-insensitive, least sufficient recovery action.

**Architecture:** Add a small algebraic module above `joinRecovery`. It folds per-fault `minimumRecovery` values with the existing join operation, defines a self-contained finite permutation relation, and proves coverage/minimality without introducing Browser runtime state. Existing `FaultSupervisor` remains unchanged except for optional executable comparison tests.

**Tech Stack:** Lean 4.33.0, Lake 5.0.0, existing `Browser.Interaction.AdversarialRecovery` contracts.

**Spec:** `docs/superpowers/specs/2026-08-21-finite-fault-aggregation-design.md`

## Global Constraints

- Do not add Browser/Page/Frame state to this proof layer.
- Reuse `minimumRecovery`, `recoveryRank`, and `joinRecovery`.
- Use TDD: failing spec first, then implementation.
- Keep PR #1 Draft and do not merge.

---

### Task 1: Finite aggregation RED/GREEN

**Files:**
- Create: `Browser/Interaction/FaultAggregationSpec.lean`
- Create: `Browser/Interaction/FaultAggregation.lean`
- Modify: `Browser.lean`

**Interfaces:**
- Consumes: `joinRecovery : RecoveryAction → RecoveryAction → RecoveryAction`, `minimumRecovery : FaultClass → RecoveryAction`.
- Produces: `aggregateRecovery`, `CoversFaults`, `MinimalCombined`, `FaultPermutation` and their theorems.

- [ ] **Step 1: Write the failing spec**

The spec must import nonexistent `Browser.Interaction.FaultAggregation` and require examples for permutation invariance, duplicate invariance, combined coverage, and leastness.

- [ ] **Step 2: Verify RED in GitHub Actions**

Expected: `lake build --wfail` fails because `Browser/Interaction/FaultAggregation.lean` does not exist.

- [ ] **Step 3: Implement the minimal algebraic module**

```lean
def aggregateRecovery : List FaultClass → RecoveryAction
  | [] => .retry
  | fault :: rest => joinRecovery (minimumRecovery fault) (aggregateRecovery rest)
```

Add recursive `CoversFaults`, `MinimalCombined`, and `FaultPermutation` with `nil`, `cons`, `swap`, and `trans` constructors.

- [ ] **Step 4: Prove the required laws**

Required theorem names:

```text
aggregate_duplicate_invariant
aggregate_permutation_invariant
aggregate_covers_faults
aggregate_least_upper
aggregate_is_minimal_combined
```

- [ ] **Step 5: Verify GREEN**

Run the PR workflow and require `lake build --wfail` plus all existing executable regressions to succeed.

### Task 2: Executable finite-fault contract

**Files:**
- Create: `FaultAggregationMain.lean`
- Modify: `lakefile.toml`
- Modify: `.github/workflows/verify.yml`

**Interfaces:**
- Consumes: `aggregateRecovery`, `FaultPermutation`, `MinimalCombined`.
- Produces: executable `browser-fault-aggregation-check`.

- [ ] **Step 1: Add executable scenarios**

Check at least:

```text
[elementStale, sessionLost, pageLost] -> recreatePage
[pageLost, elementStale, sessionLost] -> recreatePage
[pageLost, pageLost, sessionLost] -> recreatePage
[runtimeLost, contextLost, sessionLost] -> recreateContext
[pageLost, timeoutFault] -> fail
```

- [ ] **Step 2: Add executable to Lake and CI**

```text
lake exe browser-fault-aggregation-check
```

- [ ] **Step 3: Run full CI**

Require build, finite-fault executable, adversarial recovery executable, and all prior integration checks to pass on the exact final head.
