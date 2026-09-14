# Mesh Feasibility Lean V23 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove in Lean that isolated local societies embed conservatively into a directed mesh and that one explicit bridge composes bounded local paths into a six-degree global path without weakening unique physical residence.

**Architecture:** Add one dependency-light Lean module over `Mathlib` containing exact-length walks, bounded reachability, local graph restriction, closure, bridge composition, and separated physical/social Agent placement. Add one theorem-use test module, import the production module from the root library, and register the test in the proof workflow.

**Tech Stack:** Lean 4.32.0, mathlib v4.32.0, Lake.

**Spec:** `docs/superpowers/specs/2026-09-03-mesh-feasibility-lean-v23-design.md`

## Global Constraints

- Lean mathematical proof precedes every Python, NetworkX, distributed, P2P, WSS, or Raft implementation.
- The change is additive and must not change existing V1--V22 definitions or theorem statements.
- Production proofs contain no `sorry` or `admit`.
- Tests are written and observed failing before the corresponding production definitions are added.
- A graph snapshot is directed; no symmetry is assumed.
- Six-degree reachability is a conditional theorem, never an unconditional empirical claim.

---

### Task 1: Exact Mesh Walks and Bounded Reachability

**Files:**
- Create: `NarrativeDynamics/Core/SocialMesh.lean`
- Create: `NarrativeDynamics/Tests/SocialMesh.lean`

**Interfaces:**
- Consumes: Lean `Nat`, relations, and arithmetic from `Mathlib`.
- Produces: `MeshGraph`, `MeshWalk`, `MeshWalk.single`, `MeshWalk.append`, `ReachWithin`, and `reachWithin_mono`.

- [ ] **Step 1: Write the failing theorem-use test**

```lean
import NarrativeDynamics.Core.SocialMesh

open NarrativeDynamics

example {Node : Type*} (g : MeshGraph Node) {a b c : Node}
    (hab : g a b) (hbc : g b c) : ReachWithin g 2 a c := by
  exact ⟨2, by omega, (MeshWalk.single hab).append (MeshWalk.single hbc)⟩
```

- [ ] **Step 2: Run the test and verify RED**

Run: `lake env lean NarrativeDynamics/Tests/SocialMesh.lean`

Expected: failure because `NarrativeDynamics.Core.SocialMesh` does not exist.

- [ ] **Step 3: Implement the minimal walk core**

```lean
import Mathlib

namespace NarrativeDynamics

abbrev MeshGraph (Node : Type*) := Node -> Node -> Prop

inductive MeshWalk {Node : Type*} (g : MeshGraph Node) : Nat -> Node -> Node -> Prop where
  | refl (node : Node) : MeshWalk g 0 node node
  | step {n : Nat} {source next target : Node} :
      g source next -> MeshWalk g n next target -> MeshWalk g (Nat.succ n) source target

def MeshWalk.single {Node : Type*} {g : MeshGraph Node} {source target : Node}
    (edge : g source target) : MeshWalk g 1 source target :=
  .step edge (.refl target)

theorem MeshWalk.append {Node : Type*} {g : MeshGraph Node}
    {leftLength rightLength : Nat} {source middle target : Node}
    (left : MeshWalk g leftLength source middle)
    (right : MeshWalk g rightLength middle target) :
    MeshWalk g (leftLength + rightLength) source target := by
  induction left with
  | refl => simpa using right
  | step edge rest ih =>
      simpa [Nat.succ_add] using MeshWalk.step edge (ih right)

def ReachWithin {Node : Type*} (g : MeshGraph Node) (limit : Nat)
    (source target : Node) : Prop :=
  Exists fun length => length <= limit /\ MeshWalk g length source target

theorem reachWithin_mono {Node : Type*} {g : MeshGraph Node}
    {smaller larger : Nat} {source target : Node}
    (reachable : ReachWithin g smaller source target)
    (largerBound : smaller <= larger) : ReachWithin g larger source target := by
  rcases reachable with ⟨length, bound, walk⟩
  exact ⟨length, Nat.le_trans bound largerBound, walk⟩

end NarrativeDynamics
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `lake env lean NarrativeDynamics/Tests/SocialMesh.lean`

Expected: success.

- [ ] **Step 5: Commit**

```text
git add NarrativeDynamics/Core/SocialMesh.lean NarrativeDynamics/Tests/SocialMesh.lean
git commit -m "feat(lean): define bounded social mesh walks"
```

### Task 2: Conservative Local Projection and Isolation

**Files:**
- Modify: `NarrativeDynamics/Core/SocialMesh.lean`
- Modify: `NarrativeDynamics/Tests/SocialMesh.lean`

**Interfaces:**
- Consumes: `MeshGraph`, `MeshWalk`, and `ReachWithin` from Task 1.
- Produces: `restrictMesh`, `ClosedSociety`, `restricted_walk_lifts`, `closed_walk_restricts`, and `closed_no_cross_reach`.

- [ ] **Step 1: Add failing projection and isolation examples**

```lean
example {Node : Type*} (g : MeshGraph Node) (inside : Node -> Prop)
    {n : Nat} {a b : Node} (walk : MeshWalk (restrictMesh g inside) n a b) :
    MeshWalk g n a b := by
  exact restricted_walk_lifts walk

example {Node : Type*} (g : MeshGraph Node) (inside : Node -> Prop)
    (closed : ClosedSociety g inside) {limit : Nat} {a outsider : Node}
    (ha : inside a) (hout : ¬ inside outsider) :
    ¬ ReachWithin g limit a outsider := by
  exact @closed_no_cross_reach Node g inside closed limit a outsider ha hout
```

- [ ] **Step 2: Run the test and verify RED**

Run: `lake env lean NarrativeDynamics/Tests/SocialMesh.lean`

Expected: unknown constants `restrictMesh` and `ClosedSociety`.

- [ ] **Step 3: Implement restriction, lifting, and closure proofs**

```lean
def restrictMesh {Node : Type*} (g : MeshGraph Node) (inside : Node -> Prop) :
    MeshGraph Node := fun source target => inside source /\ inside target /\ g source target

def ClosedSociety {Node : Type*} (g : MeshGraph Node) (inside : Node -> Prop) : Prop :=
  forall {source target}, inside source -> g source target -> inside target

theorem restricted_walk_lifts {Node : Type*} {g : MeshGraph Node}
    {inside : Node -> Prop} {length : Nat} {source target : Node}
    (walk : MeshWalk (restrictMesh g inside) length source target) :
    MeshWalk g length source target := by
  induction walk with
  | refl => exact .refl _
  | step edge rest ih => exact .step edge.2.2 ih

theorem closed_walk_restricts {Node : Type*} {g : MeshGraph Node}
    {inside : Node -> Prop} (closed : ClosedSociety g inside)
    {length : Nat} {source target : Node} (source_inside : inside source)
    (walk : MeshWalk g length source target) :
    MeshWalk (restrictMesh g inside) length source target := by
  induction walk with
  | refl => exact .refl _
  | step edge rest ih =>
      have hnext := closed source_inside edge
      exact .step ⟨source_inside, hnext, edge⟩ (ih hnext)

theorem closed_walk_target_inside {Node : Type*} {g : MeshGraph Node}
    {inside : Node -> Prop} (closed : ClosedSociety g inside)
    {length : Nat} {source target : Node} (source_inside : inside source)
    (walk : MeshWalk g length source target) : inside target := by
  induction walk with
  | refl => exact source_inside
  | step edge rest ih => exact ih (closed source_inside edge)

theorem closed_no_cross_reach {Node : Type*} {g : MeshGraph Node}
    {inside : Node -> Prop} (closed : ClosedSociety g inside)
    {limit : Nat} (source outsider : Node) (source_inside : inside source)
    (outsider_not_inside : ¬ inside outsider) :
    ¬ ReachWithin g limit source outsider := by
  intro reachable
  rcases reachable with ⟨_, _, walk⟩
  exact outsider_not_inside (closed_walk_target_inside closed source_inside walk)
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `lake env lean NarrativeDynamics/Tests/SocialMesh.lean`

Expected: success.

- [ ] **Step 5: Commit**

```text
git add NarrativeDynamics/Core/SocialMesh.lean NarrativeDynamics/Tests/SocialMesh.lean
git commit -m "feat(lean): prove mesh projection isolation"
```

### Task 3: Bridge Composition and Six Degrees

**Files:**
- Modify: `NarrativeDynamics/Core/SocialMesh.lean`
- Modify: `NarrativeDynamics/Tests/SocialMesh.lean`

**Interfaces:**
- Consumes: `MeshWalk.append` and `ReachWithin` from Task 1.
- Produces: `reachWithin_bridge` and `six_degrees_of_two_three_bridge`.

- [ ] **Step 1: Add the failing six-degree example**

```lean
example {Node : Type*} (g : MeshGraph Node) {a x y b : Node}
    (left : ReachWithin g 2 a x) (bridge : g x y)
    (right : ReachWithin g 3 y b) : ReachWithin g 6 a b := by
  exact six_degrees_of_two_three_bridge left bridge right
```

- [ ] **Step 2: Run the test and verify RED**

Run: `lake env lean NarrativeDynamics/Tests/SocialMesh.lean`

Expected: unknown constant `six_degrees_of_two_three_bridge`.

- [ ] **Step 3: Implement general bridge composition and the corollary**

```lean
theorem reachWithin_bridge {Node : Type*} {g : MeshGraph Node}
    {leftLimit rightLimit : Nat} {source leftGate rightGate target : Node}
    (left : ReachWithin g leftLimit source leftGate)
    (bridge : g leftGate rightGate)
    (right : ReachWithin g rightLimit rightGate target) :
    ReachWithin g (leftLimit + 1 + rightLimit) source target := by
  rcases left with ⟨leftLength, leftBound, leftWalk⟩
  rcases right with ⟨rightLength, rightBound, rightWalk⟩
  refine ⟨leftLength + 1 + rightLength, ?_, ?_⟩
  · omega
  · exact (leftWalk.append (MeshWalk.single bridge)).append rightWalk

theorem six_degrees_of_two_three_bridge {Node : Type*} {g : MeshGraph Node}
    {source leftGate rightGate target : Node}
    (left : ReachWithin g 2 source leftGate)
    (bridge : g leftGate rightGate)
    (right : ReachWithin g 3 rightGate target) : ReachWithin g 6 source target := by
  exact reachWithin_mono (reachWithin_bridge left bridge right) (by omega)
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `lake env lean NarrativeDynamics/Tests/SocialMesh.lean`

Expected: success.

- [ ] **Step 5: Commit**

```text
git add NarrativeDynamics/Core/SocialMesh.lean NarrativeDynamics/Tests/SocialMesh.lean
git commit -m "feat(lean): prove six-degree mesh bridges"
```

### Task 4: Physical Uniqueness, Integration, and Documentation

**Files:**
- Modify: `NarrativeDynamics/Core/SocialMesh.lean`
- Modify: `NarrativeDynamics/Tests/SocialMesh.lean`
- Modify: `NarrativeDynamics.lean`
- Modify: `.github/workflows/proof.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: all Task 1--3 public theorems.
- Produces: `AgentMeshPlacement`, `physical_world_unique`, `multiple_social_memberships_compatible`, root-library import, CI theorem test, and user documentation.

- [ ] **Step 1: Add failing placement examples**

```lean
example {Agent World Society : Type*}
    (placement : AgentMeshPlacement Agent World Society)
    (agent : Agent) {first second : World}
    (hfirst : placement.physicalWorld agent = first)
    (hsecond : placement.physicalWorld agent = second) : first = second := by
  exact physical_world_unique placement hfirst hsecond

example {Agent World Society : Type*}
    (placement : AgentMeshPlacement Agent World Society)
    (agent : Agent) (first second : Society)
    (hfirst : placement.socialMember agent first)
    (hsecond : placement.socialMember agent second) :
    exists world, placement.physicalWorld agent = world /\
      placement.socialMember agent first /\ placement.socialMember agent second := by
  exact multiple_social_memberships_compatible placement hfirst hsecond
```

- [ ] **Step 2: Run the test and verify RED**

Run: `lake env lean NarrativeDynamics/Tests/SocialMesh.lean`

Expected: unknown constant `AgentMeshPlacement`.

- [ ] **Step 3: Implement placement and proofs**

```lean
structure AgentMeshPlacement (Agent World Society : Type*) where
  physicalWorld : Agent -> World
  socialMember : Agent -> Society -> Prop

theorem physical_world_unique {Agent World Society : Type*}
    (placement : AgentMeshPlacement Agent World Society) {agent : Agent}
    {first second : World} (hfirst : placement.physicalWorld agent = first)
    (hsecond : placement.physicalWorld agent = second) : first = second := by
  exact hfirst.symm.trans hsecond

theorem multiple_social_memberships_compatible {Agent World Society : Type*}
    (placement : AgentMeshPlacement Agent World Society) {agent : Agent}
    {first second : Society} (hfirst : placement.socialMember agent first)
    (hsecond : placement.socialMember agent second) :
    exists world, placement.physicalWorld agent = world /\
      placement.socialMember agent first /\ placement.socialMember agent second := by
  exact ⟨placement.physicalWorld agent, rfl, hfirst, hsecond⟩
```

- [ ] **Step 4: Integrate module, workflow, and README**

Add `import NarrativeDynamics.Core.SocialMesh` to `NarrativeDynamics.lean`.
Add `lake env lean NarrativeDynamics/Tests/SocialMesh.lean` beside the existing
Lean theorem tests in `.github/workflows/proof.yml`. Add a README V23 section
stating the exact conditional theorems and all statistical/distributed non-goals.

- [ ] **Step 5: Run focused verification**

Run:

```text
lake build +NarrativeDynamics.Core.SocialMesh
lake env lean NarrativeDynamics/Tests/SocialMesh.lean
rg -n "sorry|admit" NarrativeDynamics/Core/SocialMesh.lean NarrativeDynamics/Tests/SocialMesh.lean
git diff --check
```

Expected: both Lean commands succeed; the `rg` command returns no matches; diff check succeeds.

- [ ] **Step 6: Run root verification after serially warming changed targets**

Run: `lake build`

Expected: success. On the known Windows cold-cache environment, first build
`+NarrativeDynamics.Core.SocialMesh` as in Step 5 before the root build.

- [ ] **Step 7: Commit**

```text
git add NarrativeDynamics/Core/SocialMesh.lean NarrativeDynamics/Tests/SocialMesh.lean NarrativeDynamics.lean .github/workflows/proof.yml README.md docs/superpowers/specs/2026-09-03-mesh-feasibility-lean-v23-design.md docs/superpowers/plans/2026-09-03-mesh-feasibility-lean-v23.md
git commit -m "docs: document Lean mesh feasibility proof"
```
