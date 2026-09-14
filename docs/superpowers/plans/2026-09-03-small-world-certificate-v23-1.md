# Small-World Certificate V23.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove a deterministic small-world certificate by lifting bounded society paths through Agent gateways and combining six-hop global reachability with explicit strong local clustering.

**Architecture:** Reuse `MeshGraph`, `MeshWalk`, and `ReachWithin` as the only path model. A focused `SmallWorld` module supplies edge monotonicity, gateway lifting, and certificate construction; theorem-use tests pin every public result before the root build and CI import the module.

**Tech Stack:** Lean 4.32, Mathlib 4.32, Lake, GitHub Actions

**Spec:** `docs/superpowers/specs/2026-09-03-small-world-certificate-v23-1-design.md`

## Global Constraints

- Stay on `feature/mesh-feasibility-v23`; do not create another branch or worktree.
- Reuse the V23.0 directed `MeshGraph`, exact `MeshWalk`, and inclusive `ReachWithin` semantics.
- Keep all results conditional; do not claim empirical six degrees, WS expectations, or alternate attachment-model power laws.
- Do not add Python, NetworkX, stochastic, transport, or distributed-runtime dependencies.
- Add no `sorry`, `admit`, axiom, or unsafe declaration.

---

### Task 1: Edge-preserving path algebra

**Files:**
- Create: `NarrativeDynamics/Core/SmallWorld.lean`
- Create: `NarrativeDynamics/Tests/SmallWorld.lean`

**Interfaces:**
- Consumes: `MeshGraph`, `MeshWalk`, `ReachWithin` from `NarrativeDynamics.Core.SocialMesh`.
- Produces: `MeshSubgraph`, `MeshWalk.mapNodes`, `reachWithin_append`, `reachWithin_of_subgraph`, `GlobalHopBound`, and `globalHopBound_of_subgraph`.

- [ ] **Step 1: Write the failing edge-monotonicity test**

```lean
import NarrativeDynamics.Core.SmallWorld

open NarrativeDynamics

example {Node : Type*} (g h : MeshGraph Node)
    (included : MeshSubgraph g h) {limit : Nat} {source target : Node}
    (reachable : ReachWithin g limit source target) :
    ReachWithin h limit source target := by
  exact reachWithin_of_subgraph included reachable
```

- [ ] **Step 2: Run the test and verify RED**

Run: `lake env lean NarrativeDynamics/Tests/SmallWorld.lean`

Expected: failure because `NarrativeDynamics.Core.SmallWorld` does not exist.

- [ ] **Step 3: Implement minimal path algebra**

```lean
import NarrativeDynamics.Core.SocialMesh

namespace NarrativeDynamics

def MeshSubgraph {Node : Type*} (g h : MeshGraph Node) : Prop :=
  ∀ {source target}, g source target → h source target

theorem MeshWalk.mapNodes {Source Target : Type*}
    {sourceGraph : MeshGraph Source} {targetGraph : MeshGraph Target}
    (mapNode : Source → Target)
    (mapEdge : ∀ {source target}, sourceGraph source target →
      targetGraph (mapNode source) (mapNode target))
    {length : Nat} {source target : Source}
    (walk : MeshWalk sourceGraph length source target) :
    MeshWalk targetGraph length (mapNode source) (mapNode target) := by
  induction walk with
  | refl => exact MeshWalk.refl _
  | step edge rest ih => exact MeshWalk.step (mapEdge edge) ih

theorem reachWithin_append {Node : Type*} {g : MeshGraph Node}
    {leftLimit rightLimit : Nat} {source middle target : Node}
    (left : ReachWithin g leftLimit source middle)
    (right : ReachWithin g rightLimit middle target) :
    ReachWithin g (leftLimit + rightLimit) source target := by
  rcases left with ⟨leftLength, leftBound, leftWalk⟩
  rcases right with ⟨rightLength, rightBound, rightWalk⟩
  exact ⟨leftLength + rightLength, Nat.add_le_add leftBound rightBound,
    leftWalk.append rightWalk⟩

theorem reachWithin_of_subgraph {Node : Type*} {g h : MeshGraph Node}
    (included : MeshSubgraph g h) {limit : Nat} {source target : Node}
    (reachable : ReachWithin g limit source target) :
    ReachWithin h limit source target := by
  rcases reachable with ⟨length, bound, walk⟩
  exact ⟨length, bound, walk.mapNodes id (fun edge => included edge)⟩

def GlobalHopBound {Node : Type*} (g : MeshGraph Node) (limit : Nat) : Prop :=
  ∀ source target, ReachWithin g limit source target

theorem globalHopBound_of_subgraph {Node : Type*} {g h : MeshGraph Node}
    (included : MeshSubgraph g h) {limit : Nat}
    (bounded : GlobalHopBound g limit) : GlobalHopBound h limit := by
  intro source target
  exact reachWithin_of_subgraph included (bounded source target)
```

- [ ] **Step 4: Run the test and verify GREEN**

Run: `lake env lean NarrativeDynamics/Tests/SmallWorld.lean`

Expected: exit 0.

- [ ] **Step 5: Commit the path algebra**

Run: `git add NarrativeDynamics/Core/SmallWorld.lean NarrativeDynamics/Tests/SmallWorld.lean && git commit -m "feat(lean): preserve mesh paths under shortcuts"`

### Task 2: Lift society paths through Agent gateways

**Files:**
- Modify: `NarrativeDynamics/Core/SmallWorld.lean`
- Modify: `NarrativeDynamics/Tests/SmallWorld.lean`

**Interfaces:**
- Consumes: `MeshWalk.mapNodes` and `reachWithin_append` from Task 1.
- Produces: `SocietyGatewayModel`, `SocietyGatewayModel.walk_lifts`, `SocietyGatewayModel.reach_lifts`, and `reachWithin_via_societies`.

- [ ] **Step 1: Add the failing gateway-lifting test**

```lean
example {Agent Society : Type*} {agentGraph : MeshGraph Agent}
    {societyGraph : MeshGraph Society}
    (model : SocietyGatewayModel agentGraph societyGraph)
    {limit : Nat} {source target : Society}
    (reachable : ReachWithin societyGraph limit source target) :
    ReachWithin agentGraph limit (model.gateway source) (model.gateway target) := by
  exact model.reach_lifts reachable
```

- [ ] **Step 2: Run the test and verify RED**

Run: `lake env lean NarrativeDynamics/Tests/SmallWorld.lean`

Expected: unknown identifier `SocietyGatewayModel`.

- [ ] **Step 3: Implement the gateway model and lifting theorems**

```lean
structure SocietyGatewayModel {Agent Society : Type*}
    (agentGraph : MeshGraph Agent) (societyGraph : MeshGraph Society) where
  member : Agent → Society → Prop
  gateway : Society → Agent
  gatewayMember : ∀ society, member (gateway society) society
  bridge : ∀ {source target}, societyGraph source target →
    agentGraph (gateway source) (gateway target)

theorem SocietyGatewayModel.walk_lifts {Agent Society : Type*}
    {agentGraph : MeshGraph Agent} {societyGraph : MeshGraph Society}
    (model : SocietyGatewayModel agentGraph societyGraph)
    {length : Nat} {source target : Society}
    (walk : MeshWalk societyGraph length source target) :
    MeshWalk agentGraph length (model.gateway source) (model.gateway target) := by
  exact walk.mapNodes model.gateway (fun edge => model.bridge edge)

theorem SocietyGatewayModel.reach_lifts {Agent Society : Type*}
    {agentGraph : MeshGraph Agent} {societyGraph : MeshGraph Society}
    (model : SocietyGatewayModel agentGraph societyGraph)
    {limit : Nat} {source target : Society}
    (reachable : ReachWithin societyGraph limit source target) :
    ReachWithin agentGraph limit (model.gateway source) (model.gateway target) := by
  rcases reachable with ⟨length, bound, walk⟩
  exact ⟨length, bound, model.walk_lifts walk⟩

theorem reachWithin_via_societies {Agent Society : Type*}
    {agentGraph : MeshGraph Agent} {societyGraph : MeshGraph Society}
    (model : SocietyGatewayModel agentGraph societyGraph)
    {entryLimit societyLimit exitLimit : Nat}
    {source target : Agent} {sourceSociety targetSociety : Society}
    (entry : ReachWithin agentGraph entryLimit source
      (model.gateway sourceSociety))
    (between : ReachWithin societyGraph societyLimit sourceSociety targetSociety)
    (exit : ReachWithin agentGraph exitLimit
      (model.gateway targetSociety) target) :
    ReachWithin agentGraph (entryLimit + societyLimit + exitLimit) source target := by
  exact reachWithin_append (reachWithin_append entry (model.reach_lifts between)) exit
```

- [ ] **Step 4: Run the test and verify GREEN**

Run: `lake env lean NarrativeDynamics/Tests/SmallWorld.lean`

Expected: exit 0.

- [ ] **Step 5: Commit gateway lifting**

Run: `git add NarrativeDynamics/Core/SmallWorld.lean NarrativeDynamics/Tests/SmallWorld.lean && git commit -m "feat(lean): lift society paths through gateways"`

### Task 3: Derive a six-hop global bound

**Files:**
- Modify: `NarrativeDynamics/Core/SmallWorld.lean`
- Modify: `NarrativeDynamics/Tests/SmallWorld.lean`

**Interfaces:**
- Consumes: `SocietyGatewayModel`, `GlobalHopBound`, and `reachWithin_via_societies`.
- Produces: `OneHopGatewayCover` and `global_six_hop_bound`.

- [ ] **Step 1: Add the failing six-hop theorem-use test**

```lean
example {Agent Society : Type*} {agentGraph : MeshGraph Agent}
    {societyGraph : MeshGraph Society}
    (model : SocietyGatewayModel agentGraph societyGraph)
    (cover : OneHopGatewayCover model)
    (societyBound : GlobalHopBound societyGraph 4) :
    GlobalHopBound agentGraph 6 := by
  exact global_six_hop_bound model cover societyBound
```

- [ ] **Step 2: Run the test and verify RED**

Run: `lake env lean NarrativeDynamics/Tests/SmallWorld.lean`

Expected: unknown identifier `OneHopGatewayCover`.

- [ ] **Step 3: Implement the cover and global theorem**

```lean
structure OneHopGatewayCover {Agent Society : Type*}
    {agentGraph : MeshGraph Agent} {societyGraph : MeshGraph Society}
    (model : SocietyGatewayModel agentGraph societyGraph) where
  societyOf : Agent → Society
  assignedMember : ∀ agent, model.member agent (societyOf agent)
  toGateway : ∀ agent,
    ReachWithin agentGraph 1 agent (model.gateway (societyOf agent))
  fromGateway : ∀ agent,
    ReachWithin agentGraph 1 (model.gateway (societyOf agent)) agent

theorem global_six_hop_bound {Agent Society : Type*}
    {agentGraph : MeshGraph Agent} {societyGraph : MeshGraph Society}
    (model : SocietyGatewayModel agentGraph societyGraph)
    (cover : OneHopGatewayCover model)
    (societyBound : GlobalHopBound societyGraph 4) :
    GlobalHopBound agentGraph 6 := by
  intro source target
  have routed := reachWithin_via_societies model
    (cover.toGateway source)
    (societyBound (cover.societyOf source) (cover.societyOf target))
    (cover.fromGateway target)
  simpa using routed
```

- [ ] **Step 4: Run the test and verify GREEN**

Run: `lake env lean NarrativeDynamics/Tests/SmallWorld.lean`

Expected: exit 0 and Lean normalizes `1 + 4 + 1` to `6`.

- [ ] **Step 5: Commit the six-hop derivation**

Run: `git add NarrativeDynamics/Core/SmallWorld.lean NarrativeDynamics/Tests/SmallWorld.lean && git commit -m "feat(lean): derive six-hop gateway cover bound"`

### Task 4: Construct the deterministic small-world certificate

**Files:**
- Modify: `NarrativeDynamics/Core/SmallWorld.lean`
- Modify: `NarrativeDynamics/Tests/SmallWorld.lean`

**Interfaces:**
- Consumes: `global_six_hop_bound` and `globalHopBound_of_subgraph`.
- Produces: `SymmetricMesh`, `PerfectLocalClustering`, `SmallWorldCertificate`, `SmallWorldCertificate.closes_neighbor_wedge`, `SmallWorldCertificate.short_paths_survive_edge_addition`, and `smallWorld_of_gateway_cover`.

- [ ] **Step 1: Add the failing certificate-construction test**

```lean
example {Agent Society : Type*} {agentGraph : MeshGraph Agent}
    {societyGraph : MeshGraph Society}
    (model : SocietyGatewayModel agentGraph societyGraph)
    (cover : OneHopGatewayCover model)
    (societyBound : GlobalHopBound societyGraph 4)
    (symmetric : SymmetricMesh agentGraph)
    (clustered : PerfectLocalClustering agentGraph) :
    SmallWorldCertificate agentGraph 6 := by
  exact smallWorld_of_gateway_cover model cover societyBound symmetric clustered
```

- [ ] **Step 2: Run the test and verify RED**

Run: `lake env lean NarrativeDynamics/Tests/SmallWorld.lean`

Expected: unknown identifier `SmallWorldCertificate`.

- [ ] **Step 3: Implement the certificate and projections**

```lean
def SymmetricMesh {Node : Type*} (g : MeshGraph Node) : Prop :=
  ∀ {source target}, g source target → g target source

def PerfectLocalClustering {Node : Type*} (g : MeshGraph Node) : Prop :=
  ∀ {center left right}, left ≠ right → g center left → g center right → g left right

structure SmallWorldCertificate {Node : Type*}
    (g : MeshGraph Node) (limit : Nat) : Prop where
  symmetric : SymmetricMesh g
  shortPaths : GlobalHopBound g limit
  clustered : PerfectLocalClustering g

theorem SmallWorldCertificate.closes_neighbor_wedge {Node : Type*}
    {g : MeshGraph Node} {limit : Nat}
    (certificate : SmallWorldCertificate g limit)
    {center left right : Node} (different : left ≠ right)
    (leftNeighbor : g center left) (rightNeighbor : g center right) :
    g left right := by
  exact certificate.clustered different leftNeighbor rightNeighbor

theorem SmallWorldCertificate.short_paths_survive_edge_addition {Node : Type*}
    {g h : MeshGraph Node} {limit : Nat}
    (certificate : SmallWorldCertificate g limit)
    (included : MeshSubgraph g h) : GlobalHopBound h limit := by
  exact globalHopBound_of_subgraph included certificate.shortPaths

theorem smallWorld_of_gateway_cover {Agent Society : Type*}
    {agentGraph : MeshGraph Agent} {societyGraph : MeshGraph Society}
    (model : SocietyGatewayModel agentGraph societyGraph)
    (cover : OneHopGatewayCover model)
    (societyBound : GlobalHopBound societyGraph 4)
    (symmetric : SymmetricMesh agentGraph)
    (clustered : PerfectLocalClustering agentGraph) :
    SmallWorldCertificate agentGraph 6 := by
  exact ⟨symmetric, global_six_hop_bound model cover societyBound, clustered⟩
```

- [ ] **Step 4: Run certificate tests and verify GREEN**

Run: `lake env lean NarrativeDynamics/Tests/SmallWorld.lean`

Expected: exit 0.

- [ ] **Step 5: Commit the certificate**

Run: `git add NarrativeDynamics/Core/SmallWorld.lean NarrativeDynamics/Tests/SmallWorld.lean && git commit -m "feat(lean): certify deterministic small-world meshes"`

### Task 5: Integrate, document, and verify V23.1

**Files:**
- Modify: `NarrativeDynamics.lean`
- Modify: `.github/workflows/proof.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: the complete `NarrativeDynamics.Core.SmallWorld` public theorem surface.
- Produces: root-library inclusion, CI execution, and documented scope/non-claims.

- [ ] **Step 1: Import the module from the root library**

Add `import NarrativeDynamics.Core.SmallWorld` immediately after the SocialMesh
import in `NarrativeDynamics.lean`.

- [ ] **Step 2: Add the theorem-use test to CI**

Add this command immediately after the SocialMesh theorem test:

```text
lake env lean NarrativeDynamics/Tests/SmallWorld.lean
```

- [ ] **Step 3: Document the exact V23.1 claim**

Add a `Lean deterministic small-world certificate V23.1` README section explaining
the gateway lifting theorem, the conditional `1 + 4 + 1 = 6` result, perfect local
clustering, shortcut reachability monotonicity, and the exclusion of stochastic
WS or alternate attachment-model claims.

- [ ] **Step 4: Run fresh verification**

Run:

```text
lake env lean NarrativeDynamics/Tests/SocialMesh.lean
lake env lean NarrativeDynamics/Tests/SmallWorld.lean
lake build
rg -n "\b(sorry|admit)\b" NarrativeDynamics/Core/SmallWorld.lean NarrativeDynamics/Tests/SmallWorld.lean
git diff --check
```

Expected: both theorem-use files and the root build exit 0; the forbidden-token
search returns no matches; `git diff --check` exits 0.

- [ ] **Step 5: Commit and push the integrated increment**

Run:

```text
git add NarrativeDynamics.lean .github/workflows/proof.yml README.md
git commit -m "docs: document deterministic small-world certificate"
git push
```

Expected: PR #58 updates without force-push.
