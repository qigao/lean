# Small-World Finite Metrics V23.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define finite small-world metrics and prove that a deterministic small-world certificate bounds clustering, diameter, and average shortest-path length.

**Architecture:** A new Lean module enumerates finite neighborhoods and ordered node pairs while retaining `ReachWithin` as the sole path relation. `Nat.find` supplies exact minimum hop counts and diameter; rational finite sums supply clustering and average-path metrics.

**Tech Stack:** Lean 4.32, Mathlib 4.32, Lake, GitHub Actions

**Spec:** `docs/superpowers/specs/2026-09-03-small-world-finite-metrics-v23-2-design.md`

## Global Constraints

- Continue on `feature/mesh-feasibility-v23` and update PR #58 without force-push.
- Use finite ordered pairs and rational-valued ratios.
- Reuse `ReachWithin` and `GlobalHopBound`; do not introduce another walk relation.
- Keep stochastic WS/BA claims out of V23.2.
- Add no `sorry`, `admit`, axiom, or unsafe declaration.

---

### Task 1: Finite local clustering coefficient

**Files:**
- Create: `NarrativeDynamics/Core/SmallWorldMetrics.lean`
- Create: `NarrativeDynamics/Tests/SmallWorldMetrics.lean`

**Interfaces:**
- Consumes: `MeshGraph` and `PerfectLocalClustering`.
- Produces: `neighborSet`, `orderedNeighborPairs`, `closedNeighborPairs`, `localClusteringCoefficient`, `closedNeighborPairs_subset`, `closedNeighborPairs_eq_of_perfect`, and `localClusteringCoefficient_eq_one_of_perfect`.

- [ ] **Step 1: Write a failing coefficient-one theorem-use test**

```lean
import NarrativeDynamics.Core.SmallWorldMetrics

open NarrativeDynamics

example {Node : Type*} [Fintype Node] [DecidableEq Node]
    (g : MeshGraph Node) [DecidableRel g] (center : Node)
    (clustered : PerfectLocalClustering g)
    (pairsExist : (orderedNeighborPairs g center).Nonempty) :
    localClusteringCoefficient g center = 1 := by
  exact localClusteringCoefficient_eq_one_of_perfect clustered pairsExist
```

- [ ] **Step 2: Run RED**

Run: `lake env lean NarrativeDynamics/Tests/SmallWorldMetrics.lean`

Expected: the `SmallWorldMetrics` module does not exist.

- [ ] **Step 3: Implement finite pair sets and coefficient proofs**

Create the module with these definitions and proofs:

```lean
import NarrativeDynamics.Core.SmallWorld

namespace NarrativeDynamics

def neighborSet {Node : Type*} [Fintype Node] [DecidableEq Node]
    (g : MeshGraph Node) [DecidableRel g] (center : Node) : Finset Node :=
  Finset.univ.filter (g center)

def orderedNeighborPairs {Node : Type*} [Fintype Node] [DecidableEq Node]
    (g : MeshGraph Node) [DecidableRel g] (center : Node) : Finset (Node × Node) :=
  ((neighborSet g center).product (neighborSet g center)).filter
    (fun pair => pair.1 ≠ pair.2)

def closedNeighborPairs {Node : Type*} [Fintype Node] [DecidableEq Node]
    (g : MeshGraph Node) [DecidableRel g] (center : Node) : Finset (Node × Node) :=
  (orderedNeighborPairs g center).filter (fun pair => g pair.1 pair.2)

def localClusteringCoefficient {Node : Type*} [Fintype Node] [DecidableEq Node]
    (g : MeshGraph Node) [DecidableRel g] (center : Node) : ℚ :=
  ((closedNeighborPairs g center).card : ℚ) /
    ((orderedNeighborPairs g center).card : ℚ)
```

Prove the filter subset, use `Finset.filter_eq_self.2` plus the
`PerfectLocalClustering` witness to prove equality, and simplify equal nonzero
rational numerator/denominator to one.

```lean
theorem closedNeighborPairs_subset {Node : Type*} [Fintype Node]
    [DecidableEq Node] (g : MeshGraph Node) [DecidableRel g] (center : Node) :
    closedNeighborPairs g center ⊆ orderedNeighborPairs g center := by
  exact Finset.filter_subset _ _

theorem closedNeighborPairs_eq_of_perfect {Node : Type*} [Fintype Node]
    [DecidableEq Node] (g : MeshGraph Node) [DecidableRel g] (center : Node)
    (clustered : PerfectLocalClustering g) :
    closedNeighborPairs g center = orderedNeighborPairs g center := by
  rw [closedNeighborPairs]
  apply Finset.filter_eq_self.2
  intro pair pairIn
  rw [orderedNeighborPairs] at pairIn
  rcases Finset.mem_filter.mp pairIn with ⟨pairMembers, different⟩
  rcases Finset.mem_product.mp pairMembers with ⟨leftMember, rightMember⟩
  rw [neighborSet] at leftMember rightMember
  exact clustered different (Finset.mem_filter.mp leftMember).2
    (Finset.mem_filter.mp rightMember).2

theorem localClusteringCoefficient_eq_one_of_perfect {Node : Type*}
    [Fintype Node] [DecidableEq Node] (g : MeshGraph Node) [DecidableRel g]
    (center : Node) (clustered : PerfectLocalClustering g)
    (pairsExist : (orderedNeighborPairs g center).Nonempty) :
    localClusteringCoefficient g center = 1 := by
  rw [localClusteringCoefficient,
    closedNeighborPairs_eq_of_perfect g center clustered]
  exact div_self (by exact_mod_cast Finset.card_ne_zero.mpr pairsExist)
```

- [ ] **Step 4: Run GREEN**

Run: `lake build NarrativeDynamics.Core.SmallWorldMetrics` followed by
`lake env lean NarrativeDynamics/Tests/SmallWorldMetrics.lean`.

Expected: exit 0.

- [ ] **Step 5: Commit**

Run: `git add NarrativeDynamics/Core/SmallWorldMetrics.lean NarrativeDynamics/Tests/SmallWorldMetrics.lean && git commit -m "feat(lean): measure finite local clustering"`

### Task 2: Exact shortest hop count and diameter

**Files:**
- Modify: `NarrativeDynamics/Core/SmallWorldMetrics.lean`
- Modify: `NarrativeDynamics/Tests/SmallWorldMetrics.lean`

**Interfaces:**
- Consumes: `ReachWithin`, `GlobalHopBound`, and `SmallWorldCertificate`.
- Produces: `shortestHopCount`, `shortestHopCount_spec`, `shortestHopCount_minimal`, `meshDiameter`, `meshDiameter_spec`, `meshDiameter_minimal`, and `SmallWorldCertificate.meshDiameter_le`.

- [ ] **Step 1: Add failing shortest-path and diameter tests**

```lean
example {Node : Type*} (g : MeshGraph Node)
    (bounded : ∃ limit, GlobalHopBound g limit) (source target : Node) :
    ReachWithin g (shortestHopCount g bounded source target) source target := by
  exact shortestHopCount_spec g bounded source target

example {Node : Type*} {g : MeshGraph Node} {limit : Nat}
    (certificate : SmallWorldCertificate g limit) :
    meshDiameter g ⟨limit, certificate.shortPaths⟩ ≤ limit := by
  exact certificate.meshDiameter_le
```

- [ ] **Step 2: Run RED**

Run: `lake env lean NarrativeDynamics/Tests/SmallWorldMetrics.lean`.

Expected: unknown identifiers `shortestHopCount` and `meshDiameter`.

- [ ] **Step 3: Implement `Nat.find` metrics and proof interfaces**

Construct pair reachability from the supplied existential global bound. Define
`shortestHopCount` as `Nat.find` of that pair witness and `meshDiameter` as
`Nat.find bounded`. Use `Nat.find_spec` for reachability and `Nat.find_min'` for
minimality. Derive `SmallWorldCertificate.meshDiameter_le` by supplying
`certificate.shortPaths` as a candidate global bound.

```lean
theorem exists_reachWithin_of_globalHopBound {Node : Type*}
    (g : MeshGraph Node) (bounded : ∃ limit, GlobalHopBound g limit)
    (source target : Node) : ∃ limit, ReachWithin g limit source target := by
  rcases bounded with ⟨limit, globalBound⟩
  exact ⟨limit, globalBound source target⟩

noncomputable def shortestHopCount {Node : Type*} (g : MeshGraph Node)
    (bounded : ∃ limit, GlobalHopBound g limit) (source target : Node) : Nat :=
  by
    classical
    exact Nat.find (exists_reachWithin_of_globalHopBound g bounded source target)

theorem shortestHopCount_spec {Node : Type*} (g : MeshGraph Node)
    (bounded : ∃ limit, GlobalHopBound g limit) (source target : Node) :
    ReachWithin g (shortestHopCount g bounded source target) source target := by
  classical
  exact Nat.find_spec (exists_reachWithin_of_globalHopBound g bounded source target)

theorem shortestHopCount_minimal {Node : Type*} (g : MeshGraph Node)
    (bounded : ∃ limit, GlobalHopBound g limit) {source target : Node}
    {candidate : Nat} (reachable : ReachWithin g candidate source target) :
    shortestHopCount g bounded source target ≤ candidate := by
  classical
  exact Nat.find_min' (exists_reachWithin_of_globalHopBound g bounded source target)
    reachable

noncomputable def meshDiameter {Node : Type*} (g : MeshGraph Node)
    (bounded : ∃ limit, GlobalHopBound g limit) : Nat := by
  classical
  exact Nat.find bounded

theorem meshDiameter_spec {Node : Type*} (g : MeshGraph Node)
    (bounded : ∃ limit, GlobalHopBound g limit) :
    GlobalHopBound g (meshDiameter g bounded) := by
  classical
  exact Nat.find_spec bounded

theorem meshDiameter_minimal {Node : Type*} (g : MeshGraph Node)
    (bounded : ∃ limit, GlobalHopBound g limit) {candidate : Nat}
    (candidateBound : GlobalHopBound g candidate) :
    meshDiameter g bounded ≤ candidate := by
  classical
  exact Nat.find_min' bounded candidateBound

theorem SmallWorldCertificate.meshDiameter_le {Node : Type*}
    {g : MeshGraph Node} {limit : Nat}
    (certificate : SmallWorldCertificate g limit) :
    meshDiameter g ⟨limit, certificate.shortPaths⟩ ≤ limit :=
  meshDiameter_minimal g ⟨limit, certificate.shortPaths⟩ certificate.shortPaths
```

- [ ] **Step 4: Run GREEN**

Run the focused module build and theorem-use file. Expected: exit 0.

- [ ] **Step 5: Commit**

Run: `git add NarrativeDynamics/Core/SmallWorldMetrics.lean NarrativeDynamics/Tests/SmallWorldMetrics.lean && git commit -m "feat(lean): define shortest hops and mesh diameter"`

### Task 3: Average shortest-path length

**Files:**
- Modify: `NarrativeDynamics/Core/SmallWorldMetrics.lean`
- Modify: `NarrativeDynamics/Tests/SmallWorldMetrics.lean`

**Interfaces:**
- Consumes: `shortestHopCount_minimal` and a certificate's `shortPaths` field.
- Produces: `orderedDistinctNodePairs`, `averageShortestPathLength`, `averageShortestPathLength_eq_zero_of_no_pairs`, and `SmallWorldCertificate.averageShortestPathLength_le`.

- [ ] **Step 1: Add failing zero-case and certificate-bound tests**

Use `PUnit` for a one-node graph and prove its average is zero. Add a generic finite
certificate theorem-use example requiring the average to be at most the certificate
limit.

```lean
example :
    let g : MeshGraph PUnit := fun _ _ => False
    let bounded : ∃ limit, GlobalHopBound g limit := ⟨0, by
      intro source target
      cases source
      cases target
      exact ⟨0, by omega, MeshWalk.refl _⟩⟩
    averageShortestPathLength g bounded = 0 := by
  dsimp
  apply averageShortestPathLength_eq_zero_of_no_pairs
  ext pair
  simp [orderedDistinctNodePairs]

example {Node : Type*} [Fintype Node] [DecidableEq Node]
    {g : MeshGraph Node} {limit : Nat}
    (certificate : SmallWorldCertificate g limit) :
    averageShortestPathLength g ⟨limit, certificate.shortPaths⟩ ≤ limit := by
  exact certificate.averageShortestPathLength_le
```

- [ ] **Step 2: Run RED**

Run the focused theorem-use file. Expected: unknown identifier
`averageShortestPathLength`.

- [ ] **Step 3: Implement average metric and bound**

Enumerate distinct ordered pairs from `Finset.univ.product Finset.univ`. Sum the
rational casts of `shortestHopCount`, divide by the pair count, and prove the empty
case by simplification. For a nonempty pair set, use `Finset.sum_le_sum`,
`shortestHopCount_minimal`, positivity of the rational pair count, and
`div_le_iff₀` to establish the certificate limit.

```lean
def orderedDistinctNodePairs (Node : Type*) [Fintype Node] [DecidableEq Node] :
    Finset (Node × Node) :=
  (Finset.univ.product Finset.univ).filter (fun pair => pair.1 ≠ pair.2)

noncomputable def averageShortestPathLength {Node : Type*}
    [Fintype Node] [DecidableEq Node] (g : MeshGraph Node)
    (bounded : ∃ limit, GlobalHopBound g limit) : ℚ :=
  (∑ pair in orderedDistinctNodePairs Node,
      (shortestHopCount g bounded pair.1 pair.2 : ℚ)) /
    ((orderedDistinctNodePairs Node).card : ℚ)

theorem averageShortestPathLength_eq_zero_of_no_pairs {Node : Type*}
    [Fintype Node] [DecidableEq Node] (g : MeshGraph Node)
    (bounded : ∃ limit, GlobalHopBound g limit)
    (noPairs : orderedDistinctNodePairs Node = ∅) :
    averageShortestPathLength g bounded = 0 := by
  simp [averageShortestPathLength, noPairs]

theorem SmallWorldCertificate.averageShortestPathLength_le {Node : Type*}
    [Fintype Node] [DecidableEq Node] {g : MeshGraph Node} {limit : Nat}
    (certificate : SmallWorldCertificate g limit) :
    averageShortestPathLength g ⟨limit, certificate.shortPaths⟩ ≤ limit := by
  classical
  by_cases noPairs : orderedDistinctNodePairs Node = ∅
  · simp [averageShortestPathLength, noPairs]
  · have pairNonempty : (orderedDistinctNodePairs Node).Nonempty :=
      Finset.nonempty_iff_ne_empty.mpr noPairs
    have cardPositiveNat : 0 < (orderedDistinctNodePairs Node).card :=
      Finset.card_pos.mpr pairNonempty
    have cardPositiveRat : (0 : ℚ) <
        ((orderedDistinctNodePairs Node).card : ℚ) := by
      exact_mod_cast cardPositiveNat
    apply (div_le_iff₀ cardPositiveRat).2
    calc
      (∑ pair in orderedDistinctNodePairs Node,
          (shortestHopCount g ⟨limit, certificate.shortPaths⟩
            pair.1 pair.2 : ℚ))
          ≤ ∑ _pair in orderedDistinctNodePairs Node, (limit : ℚ) := by
              apply Finset.sum_le_sum
              intro pair _
              exact_mod_cast shortestHopCount_minimal g
                ⟨limit, certificate.shortPaths⟩
                (certificate.shortPaths pair.1 pair.2)
      _ = ((orderedDistinctNodePairs Node).card : ℚ) * limit := by simp
      _ = limit * ((orderedDistinctNodePairs Node).card : ℚ) := by ring
```

- [ ] **Step 4: Run GREEN**

Run the focused module build and theorem-use file. Expected: exit 0.

- [ ] **Step 5: Commit**

Run: `git add NarrativeDynamics/Core/SmallWorldMetrics.lean NarrativeDynamics/Tests/SmallWorldMetrics.lean && git commit -m "feat(lean): bound average shortest paths"`

### Task 4: Root, CI, documentation, and PR integration

**Files:**
- Modify: `NarrativeDynamics.lean`
- Modify: `.github/workflows/proof.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: the complete finite-metrics theorem surface.
- Produces: root build inclusion, CI execution, and explicit V23.2 scope.

- [ ] **Step 1: Add root and CI imports**

Import `NarrativeDynamics.Core.SmallWorldMetrics` after `SmallWorld` and execute
`NarrativeDynamics/Tests/SmallWorldMetrics.lean` after the SmallWorld test.

- [ ] **Step 2: Document V23.2**

Explain ordered-pair clustering, zero-denominator convention, minimum hop/diameter,
average shortest path, certificate bounds, and the continued exclusion of
stochastic WS/BA claims.

- [ ] **Step 3: Run fresh verification**

Run:

```text
lake env lean NarrativeDynamics/Tests/SocialMesh.lean
lake env lean NarrativeDynamics/Tests/SmallWorld.lean
lake env lean NarrativeDynamics/Tests/SmallWorldMetrics.lean
lake build
rg -n "\b(sorry|admit)\b" NarrativeDynamics/Core/SmallWorldMetrics.lean NarrativeDynamics/Tests/SmallWorldMetrics.lean
git diff --check
```

Expected: all Lean commands and diff check exit 0; forbidden-token search has no
matches.

- [ ] **Step 4: Commit and push**

Commit root, workflow, and README changes as
`docs: document finite small-world metrics`, push without force, and update PR #58
to include V23.2 scope and verification.
