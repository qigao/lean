# Watts--Strogatz Foundation V23.3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define a finite regular ring and prove deterministic shortcut effects on clustering examples and path metrics.

**Architecture:** Adapt Mathlib's `cycleGraph` into the existing `MeshGraph` path semantics rather than duplicating cycle connectivity. A focused Watts--Strogatz module owns ring construction and shortcut augmentation while reusing V23.2 numerical metrics.

**Tech Stack:** Lean 4.32, Mathlib 4.32, Lake, GitHub Actions

**Spec:** `docs/superpowers/specs/2026-09-03-watts-strogatz-foundation-v23-3-design.md`

## Global Constraints

- Continue on `feature/mesh-feasibility-v23` and update PR #58 without force-push.
- Reuse `MeshWalk`, `GlobalHopBound`, and V23.2 metrics.
- Treat WS V23.3 as deterministic; add no probability or random generator.
- Preserve the distinction between path monotonicity and non-monotone clustering.
- Add no `sorry`, `admit`, axiom, or unsafe declaration.

---

### Task 1: Construct the finite regular ring

**Files:**
- Create: `NarrativeDynamics/Core/WattsStrogatz.lean`
- Create: `NarrativeDynamics/Tests/WattsStrogatz.lean`

**Interfaces:**
- Consumes: `MeshGraph`, `SymmetricMesh`, and finite clustering metrics.
- Produces: `LooplessMesh`, `wattsStrogatzRing`, its decidable relation instance, `wattsStrogatzRing_symmetric`, `wattsStrogatzRing_loopless`, and `WattsStrogatzParameters`.

- [ ] **Step 1: Write failing ring tests**

```lean
import NarrativeDynamics.Core.WattsStrogatz

open NarrativeDynamics

example (n radius : Nat) : SymmetricMesh (wattsStrogatzRing n radius) := by
  exact wattsStrogatzRing_symmetric n radius

example (n radius : Nat) : LooplessMesh (wattsStrogatzRing n radius) := by
  exact wattsStrogatzRing_loopless n radius

example : localClusteringCoefficient (wattsStrogatzRing 6 2) (0 : Fin 6) =
    (2 : ℚ) / 3 := by
  native_decide
```

- [ ] **Step 2: Run RED**

Run `lake env lean NarrativeDynamics/Tests/WattsStrogatz.lean` and verify module
absence.

- [ ] **Step 3: Implement the ring**

```lean
import Mathlib.Combinatorics.SimpleGraph.CycleGraph
import NarrativeDynamics.Core.SmallWorldMetrics

namespace NarrativeDynamics

def LooplessMesh {Node : Type*} (g : MeshGraph Node) : Prop :=
  ∀ node, ¬g node node

def wattsStrogatzRing (nodeCount radius : Nat) : MeshGraph (Fin nodeCount) :=
  fun source target => source ≠ target ∧
    ((source - target).val ≤ radius ∨ (target - source).val ≤ radius)

instance wattsStrogatzRingDecidable (nodeCount radius : Nat) :
    DecidableRel (wattsStrogatzRing nodeCount radius) := by
  intro source target
  unfold wattsStrogatzRing
  infer_instance

theorem wattsStrogatzRing_symmetric (nodeCount radius : Nat) :
    SymmetricMesh (wattsStrogatzRing nodeCount radius) := by
  intro source target edge
  exact ⟨edge.1.symm, edge.2.elim Or.inr Or.inl⟩

theorem wattsStrogatzRing_loopless (nodeCount radius : Nat) :
    LooplessMesh (wattsStrogatzRing nodeCount radius) := by
  intro node edge
  exact edge.1 rfl

structure WattsStrogatzParameters where
  nodeCount : Nat
  radius : Nat
  radiusPositive : 0 < radius
  localNotComplete : 2 * radius < nodeCount

def WattsStrogatzParameters.ringGraph (parameters : WattsStrogatzParameters) :
    MeshGraph (Fin parameters.nodeCount) :=
  wattsStrogatzRing parameters.nodeCount parameters.radius
```

- [ ] **Step 4: Run GREEN**

Build `NarrativeDynamics.Core.WattsStrogatz`, then run its theorem-use file.

- [ ] **Step 5: Commit**

Commit the core, tests, and any exact proof-plan correction as
`feat(lean): construct Watts-Strogatz ring lattice`.

### Task 2: Prove connectivity through the cycle subgraph

**Files:**
- Modify: `NarrativeDynamics/Core/WattsStrogatz.lean`
- Modify: `NarrativeDynamics/Tests/WattsStrogatz.lean`

**Interfaces:**
- Consumes: Mathlib `cycleGraph_preconnected`, `Reachable.exists_isPath`, and `IsPath.length_lt`.
- Produces: `MeshWalk.ofSimpleGraphWalk`, `cycleGraph_globalHopBound`, `cycleGraph_subgraph_wattsStrogatzRing`, `wattsStrogatzRing_globalHopBound`, and `WattsStrogatzParameters.globalHopBound`.

- [ ] **Step 1: Add failing global-bound tests**

```lean
example (n radius : Nat) (positive : 1 ≤ radius) :
    GlobalHopBound (wattsStrogatzRing n radius) (n - 1) := by
  exact wattsStrogatzRing_globalHopBound n radius positive

example (parameters : WattsStrogatzParameters) :
    GlobalHopBound parameters.ringGraph (parameters.nodeCount - 1) := by
  exact parameters.globalHopBound
```

- [ ] **Step 2: Run RED**

Run the theorem-use file and verify the global-bound names are missing.

- [ ] **Step 3: Implement cycle adaptation and bound**

```lean
theorem MeshWalk.ofSimpleGraphWalk {Node : Type*} {g : SimpleGraph Node}
    {source target : Node} (walk : g.Walk source target) :
    MeshWalk g.Adj walk.length source target := by
  induction walk with
  | nil => exact MeshWalk.refl _
  | cons edge rest ih => exact MeshWalk.step edge ih

theorem cycleGraph_globalHopBound (nodeCount : Nat) :
    GlobalHopBound (SimpleGraph.cycleGraph nodeCount).Adj (nodeCount - 1) := by
  intro source target
  obtain ⟨walk, isPath⟩ :=
    (SimpleGraph.cycleGraph_preconnected source target).exists_isPath
  refine ⟨walk.length, ?_, walk.ofSimpleGraphWalk⟩
  have lengthLt : walk.length < nodeCount := by
    simpa using isPath.length_lt
  omega

theorem cycleGraph_subgraph_wattsStrogatzRing (nodeCount radius : Nat)
    (positive : 1 ≤ radius) :
    MeshSubgraph (SimpleGraph.cycleGraph nodeCount).Adj
      (wattsStrogatzRing nodeCount radius) := by
  intro source target adjacent
  refine ⟨adjacent.ne, ?_⟩
  rcases SimpleGraph.cycleGraph_adj'.mp adjacent with forward | backward
  · exact Or.inl (by simpa [forward] using positive)
  · exact Or.inr (by simpa [backward] using positive)

theorem wattsStrogatzRing_globalHopBound (nodeCount radius : Nat)
    (positive : 1 ≤ radius) :
    GlobalHopBound (wattsStrogatzRing nodeCount radius) (nodeCount - 1) :=
  globalHopBound_of_subgraph
    (cycleGraph_subgraph_wattsStrogatzRing nodeCount radius positive)
    (cycleGraph_globalHopBound nodeCount)

theorem WattsStrogatzParameters.globalHopBound
    (parameters : WattsStrogatzParameters) :
    GlobalHopBound parameters.ringGraph (parameters.nodeCount - 1) :=
  wattsStrogatzRing_globalHopBound parameters.nodeCount parameters.radius
    parameters.radiusPositive
```

- [ ] **Step 4: Run GREEN**

Build the focused module and run the theorem-use file; both must exit 0.

- [ ] **Step 5: Commit**

Commit as `feat(lean): bound ring lattice paths`.

### Task 3: Add shortcuts and prove path-metric monotonicity

**Files:**
- Modify: `NarrativeDynamics/Core/WattsStrogatz.lean`
- Modify: `NarrativeDynamics/Tests/WattsStrogatz.lean`

**Interfaces:**
- Consumes: V23.2 shortest hop, diameter, and average-path definitions.
- Produces: `addUndirectedShortcut`, `base_subgraph_addUndirectedShortcut`, symmetry/loopless preservation, `shortestHopCount_mono_edges`, `meshDiameter_mono_edges`, and `averageShortestPathLength_mono_edges`.

- [ ] **Step 1: Add failing shortcut metric tests**

Write separate examples asserting base subgraph inclusion and the three generic
metric inequalities under `MeshSubgraph g h`; specialize inclusion to
`addUndirectedShortcut g left right`.

```lean
example {Node : Type*} (g : MeshGraph Node) (left right : Node) :
    MeshSubgraph g (addUndirectedShortcut g left right) := by
  exact base_subgraph_addUndirectedShortcut g left right

example {Node : Type*} {g h : MeshGraph Node}
    (included : MeshSubgraph g h)
    (gBounded : ∃ limit, GlobalHopBound g limit)
    (hBounded : ∃ limit, GlobalHopBound h limit) (source target : Node) :
    shortestHopCount h hBounded source target ≤
      shortestHopCount g gBounded source target := by
  exact shortestHopCount_mono_edges included gBounded hBounded source target

example {Node : Type*} {g h : MeshGraph Node}
    (included : MeshSubgraph g h)
    (gBounded : ∃ limit, GlobalHopBound g limit)
    (hBounded : ∃ limit, GlobalHopBound h limit) :
    meshDiameter h hBounded ≤ meshDiameter g gBounded := by
  exact meshDiameter_mono_edges included gBounded hBounded

example {Node : Type*} [Fintype Node] [DecidableEq Node]
    {g h : MeshGraph Node} (included : MeshSubgraph g h)
    (gBounded : ∃ limit, GlobalHopBound g limit)
    (hBounded : ∃ limit, GlobalHopBound h limit) :
    averageShortestPathLength h hBounded ≤
      averageShortestPathLength g gBounded := by
  exact averageShortestPathLength_mono_edges included gBounded hBounded
```

- [ ] **Step 2: Run RED**

Run the theorem-use file and verify shortcut/monotonicity identifiers are missing.

- [ ] **Step 3: Implement shortcut and monotonicity proofs**

```lean
def addUndirectedShortcut {Node : Type*} (g : MeshGraph Node)
    (left right : Node) : MeshGraph Node :=
  fun source target => g source target ∨
    (source = left ∧ target = right) ∨ (source = right ∧ target = left)

theorem base_subgraph_addUndirectedShortcut {Node : Type*}
    (g : MeshGraph Node) (left right : Node) :
    MeshSubgraph g (addUndirectedShortcut g left right) := by
  intro source target edge
  exact Or.inl edge

theorem shortestHopCount_mono_edges {Node : Type*} {g h : MeshGraph Node}
    (included : MeshSubgraph g h)
    (gBounded : ∃ limit, GlobalHopBound g limit)
    (hBounded : ∃ limit, GlobalHopBound h limit) (source target : Node) :
    shortestHopCount h hBounded source target ≤
      shortestHopCount g gBounded source target := by
  apply shortestHopCount_minimal h hBounded
  exact reachWithin_of_subgraph included
    (shortestHopCount_spec g gBounded source target)

theorem meshDiameter_mono_edges {Node : Type*} {g h : MeshGraph Node}
    (included : MeshSubgraph g h)
    (gBounded : ∃ limit, GlobalHopBound g limit)
    (hBounded : ∃ limit, GlobalHopBound h limit) :
    meshDiameter h hBounded ≤ meshDiameter g gBounded := by
  apply meshDiameter_minimal h hBounded
  exact globalHopBound_of_subgraph included (meshDiameter_spec g gBounded)

theorem averageShortestPathLength_mono_edges {Node : Type*}
    [Fintype Node] [DecidableEq Node] {g h : MeshGraph Node}
    (included : MeshSubgraph g h)
    (gBounded : ∃ limit, GlobalHopBound g limit)
    (hBounded : ∃ limit, GlobalHopBound h limit) :
    averageShortestPathLength h hBounded ≤
      averageShortestPathLength g gBounded := by
  unfold averageShortestPathLength
  apply div_le_div_of_nonneg_right
  · apply Finset.sum_le_sum
    intro pair _
    exact_mod_cast shortestHopCount_mono_edges included gBounded hBounded
      pair.1 pair.2
  · positivity
```

Add direct proofs that a distinct-endpoint shortcut preserves `SymmetricMesh` and
`LooplessMesh` by case analysis over the three disjuncts.

```lean
theorem addUndirectedShortcut_symmetric {Node : Type*} {g : MeshGraph Node}
    (symmetric : SymmetricMesh g) (left right : Node) :
    SymmetricMesh (addUndirectedShortcut g left right) := by
  intro source target edge
  rcases edge with old | added | added
  · exact Or.inl (symmetric old)
  · exact Or.inr (Or.inr ⟨added.2, added.1⟩)
  · exact Or.inr (Or.inl ⟨added.2, added.1⟩)

theorem addUndirectedShortcut_loopless {Node : Type*} {g : MeshGraph Node}
    (loopless : LooplessMesh g) {left right : Node} (distinct : left ≠ right) :
    LooplessMesh (addUndirectedShortcut g left right) := by
  intro node edge
  rcases edge with old | added | added
  · exact loopless node old
  · exact distinct (added.1.symm.trans added.2)
  · exact distinct (added.2.symm.trans added.1)
```

- [ ] **Step 4: Run GREEN**

Build the focused module and run the theorem-use file; both must exit 0.

- [ ] **Step 5: Commit**

Commit as `feat(lean): prove shortcut metric monotonicity`.

### Task 4: Integrate and document V23.3

**Files:**
- Modify: `NarrativeDynamics.lean`
- Modify: `.github/workflows/proof.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: complete deterministic WS foundation.
- Produces: root import, CI theorem test, documented claims/non-claims.

- [ ] **Step 1: Add root import and CI command**

Import `NarrativeDynamics.Core.WattsStrogatz` after `SmallWorldMetrics`; run
`NarrativeDynamics/Tests/WattsStrogatz.lean` immediately after the metrics test.

```text
import NarrativeDynamics.Core.WattsStrogatz
lake env lean NarrativeDynamics/Tests/WattsStrogatz.lean
```

- [ ] **Step 2: Document V23.3**

Add a README section stating the ring definition, concrete six-node `2 / 3`
clustering result, cycle-derived `n - 1` hop bound, shortcut metric monotonicity,
and the absence of stochastic claims.

- [ ] **Step 3: Verify and push**

Run all four V23 theorem-use files, `lake build`, forbidden-token search, and
`git diff --check`. Commit integration as
`docs: document Watts-Strogatz proof foundation`, push without force, and update
PR #58.

```text
lake env lean NarrativeDynamics/Tests/SocialMesh.lean
lake env lean NarrativeDynamics/Tests/SmallWorld.lean
lake env lean NarrativeDynamics/Tests/SmallWorldMetrics.lean
lake env lean NarrativeDynamics/Tests/WattsStrogatz.lean
lake build
rg -n "\b(sorry|admit)\b" NarrativeDynamics/Core/WattsStrogatz.lean NarrativeDynamics/Tests/WattsStrogatz.lean
git diff --check
```
