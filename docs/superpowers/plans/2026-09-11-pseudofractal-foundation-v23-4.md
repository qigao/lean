# Deterministic Pseudofractal Mesh V23.4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Read the specification and this plan together. The current change records the plan only; execution starts after its review.

**Goal:** Construct the deterministic pseudofractal graph family, prove its finite degree and clustering identities, and certify the actual generation-five graph's exact six-hop diameter with a generation-six counterexample.

**Architecture:** Keep an unordered-edge mathematical expansion and a canonical numbered executable view, connected by a proved graph isomorphism. Derive counts and clustering from the construction. A checked finite distance table, not the search algorithm or a claimed histogram, supplies exact-distance evidence for the existing mesh metrics.

**Tech Stack:** Lean `leanprover/lean4:v4.32.0`, mathlib `v4.32.0`, existing `MeshGraph`/`MeshWalk`/finite metrics, GitHub Actions `proof.yml`. No new Python dependencies or runtime modules.

**Spec:** `docs/superpowers/specs/2026-09-11-pseudofractal-foundation-v23-4-design.md`, approved in the conversation by “keep going”; specification commit `a3938dfff05badbf182a11d9eb5a4d75104c9d8e`, blob `b8a63e15b6f9c1f3f5acc17b5dcf0dc04a406313`.

## Global Constraints

- Continue `feature/pseudofractal-foundation-v23-4`, Draft PR #60, based on #58 at `c268e5355b14fb408f30814ed1c2bee2f41f1144`; do not modify or merge #58 or #59.
- “Index the triangle as G_0, not a single edge.”
- “The entire old edge set is frozen before expansion. New edges do not participate until the following generation.”
- “The two orientations of one edge must not generate two vertices.”
- “No independently assumed adjacency matrix may replace the recursively defined graph.”
- “Use total finite indexing. Certificate dimension, endpoint bounds, and parent indices must be checked; no missing entry defaults to a valid zero distance.”
- “Do not require or weaken the existing `SmallWorldCertificate`.”
- “Structural proofs and checker soundness use ordinary Lean proof terms, without `sorry`, `admit`, or new user axioms.”
- Compiler-generated native-evaluation assumptions are permitted only for reported concrete computations, as allowed by the spec; audit actual `#print axioms` output on the pinned toolchain. Do not hard-code an expected axiom name from a different Lean version.
- “Do not create Python runtime modules, package-root exports, network generators, transport, scheduler behavior, or automatic mappings to V24 societies.” The Lean finite constructor in this plan is the mathematical model, not a runtime network generator.
- “Preserve the full Python job, even if its baseline failure is unrelated.” No research-lock changes or failure suppression.
- Finite dyadic counts are the scale-free result. Do not claim an exact continuous power law, an asymptotic distance theorem, or six-hop reachability for every generation.

## Execution context and dependency order

The existing source supplies `MeshWalk.mapNodes`, `reachWithin_append`, `shortestHopCount_spec`, `shortestHopCount_minimal`, `meshDiameter_spec`, and `meshDiameter_minimal`. Reuse them; do not change their definitions. `LooplessMesh` currently lives in `WattsStrogatz.lean`; importing it is acceptable, moving it is not part of this plan.

Use namespace `NarrativeDynamics.Pseudofractal`. Put representation and certificate helpers under its `Internal` namespace. These helpers are importable for tests but are not a new stable public graph/certificate framework. Do not use Lean `private` for helpers that the separate test file must reference.

The executable view must not depend on `Classical.choice` for enumeration. Mathematical existence proofs may use the repository's usual classical reasoning. Their noncomputability must not flow into executable numbering or certificate checks.

| File | Responsibility |
| --- | --- |
| `NarrativeDynamics/Core/Pseudofractal.lean` | Expansion, bundled finite family, counts, numbering isomorphism, degrees, coarse reachability. |
| `NarrativeDynamics/Core/PseudofractalMetrics.lean` | Actual rational clustering, internal certificate checker/soundness, finite evaluator, links to semantic metrics. |
| `NarrativeDynamics/Tests/Pseudofractal.lean` | Generic theorem applications, generations 0–3, adversarial certificate tests. |
| `NarrativeDynamics/Tests/PseudofractalSixHop.lean` | G5 certificate and exact aggregates, G6 single-source counterexample, concrete axiom audit. |
| `NarrativeDynamics.lean` | Additive imports as core modules become available; no imports of test modules. |
| `.github/workflows/proof.yml` | Named structural and finite checks, provenance logging; retain every existing check. |
| `README.md` | Final verified scope and trust boundary only. |

Task dependencies are `1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9`. Some proofs are mathematically independent, but keep implementation serial to avoid shared-file conflicts. No additional public module or fixture format is needed. If a representation cannot satisfy the spec, document the precise obstruction before changing the architecture.

### RED/GREEN protocol used by every task

Write the task's test first and push a RED commit. A missing new import/declaration or a demonstrated false certificate acceptance is an expected RED; a dependency download failure, a test typo, or the unrelated Python failure is not. Record the actual checked-out SHA, run attempt, event, job and failing step before adding its implementation. After the implementation commit, require the named tests and root Lean build to pass at that revision before the next task.

For Task 1, add the structural-test workflow step before the core import exists, so the existing root build still completes and the missing module is the visible RED. Add the core root import in GREEN. Add the finite-test step only when Task 8 introduces its file. Never mask an intended RED with `continue-on-error`, an expected-failure wrapper, a stub axiom, or a skipped test.

All code blocks below are planned definitions, test bodies or interface contracts, not a claim that they have already elaborated. The first execution must check them with the pinned toolchain. Ordinary proof obligations listed with their derivations must be implemented as proofs, not converted into assumed structure fields.

---

## Task 1: One old unordered edge creates one newborn

**Files:** Create `NarrativeDynamics/Core/Pseudofractal.lean` and `NarrativeDynamics/Tests/Pseudofractal.lean`; modify `NarrativeDynamics.lean` and `.github/workflows/proof.yml` only for the new import/test step.

**Consumes:** `SimpleGraph`, its `edgeSet : Set (Sym2 V)`, and `MeshWalk.mapNodes`.

**Produces:** `ExpansionVertex G := V ⊕ G.edgeSet`; `expandGraph G : SimpleGraph (ExpansionVertex G)`; a computable adjacency instance when the old adjacency is decidable; `oldVertex G : V ↪ ExpansionVertex G`; `old_walk_lifts`.

- [ ] **1. Write RED tests for all four adjacency cases.** The new module exports the definitions named here.

```lean
import NarrativeDynamics.Core.Pseudofractal
open NarrativeDynamics NarrativeDynamics.Pseudofractal

example {V : Type} (G : SimpleGraph V) (u v : V) :
    (expandGraph G).Adj (Sum.inl u) (Sum.inl v) ↔ G.Adj u v := Iff.rfl
example {V : Type} (G : SimpleGraph V) (v : V) (e : G.edgeSet) :
    (expandGraph G).Adj (Sum.inl v) (Sum.inr e) ↔ v ∈ e.val := Iff.rfl
example {V : Type} (G : SimpleGraph V) (e : G.edgeSet) (v : V) :
    (expandGraph G).Adj (Sum.inr e) (Sum.inl v) ↔ v ∈ e.val := Iff.rfl
example {V : Type} (G : SimpleGraph V) (e f : G.edgeSet) :
    ¬ (expandGraph G).Adj (Sum.inr e) (Sum.inr f) := by simp [expandGraph, expansionAdj]
```

Add a named `Pseudofractal structural theorem tests` step that runs `lake env lean NarrativeDynamics/Tests/Pseudofractal.lean`, after the existing Lean build. The first failure must be the intentionally missing module, not a failure in an earlier old test.

- [ ] **2. Commit/push the tests and inspect GitHub RED.** Record the log's exact missing module/declaration. Do not create the implementation until that evidence exists.

- [ ] **3. Implement the expansion by the following adjacency definition.** Supply symmetry and looplessness by case analysis; old–old uses `G.symm`/`G.loopless`, mixed cases are identical membership tests, new–new is false.

```lean
abbrev ExpansionVertex {V : Type} (G : SimpleGraph V) := V ⊕ G.edgeSet

def expansionAdj {V : Type} (G : SimpleGraph V) :
    ExpansionVertex G → ExpansionVertex G → Prop
  | .inl u, .inl v => G.Adj u v
  | .inl v, .inr e => v ∈ e.val
  | .inr e, .inl v => v ∈ e.val
  | .inr _, .inr _ => False
```

Expose this relation through `expandGraph`. Define `oldVertex` using `Sum.inl`. The proof of `old_walk_lifts G w` is `w.mapNodes Sum.inl` with the old–old adjacency implication. Also export bounded-walk lifting without changing its budget. Do not use a same-carrier `MeshSubgraph` between generations.

- [ ] **4. Add and run the walk test, then obtain GREEN.** Add the root import now; run root `lake build` and the structural test step.

```lean
example {V : Type} (G : SimpleGraph V) {u v : V} {n : Nat}
    (w : MeshWalk G.Adj n u v) :
    MeshWalk (expandGraph G).Adj n (Sum.inl u) (Sum.inl v) :=
  old_walk_lifts G w
```

- [ ] **5. Commit checkpoint:** `feat(lean): define pseudofractal edge expansion`. Confirm the diff contains no duplicated oriented-edge births and no changes to older graph semantics.

## Task 2: Finite family, actual counts, and a non-tight reachability bound

**Files:** Extend the two Task 1 Lean files.

**Consumes:** `expandGraph`, `old_walk_lifts`, finite unordered edges, `reachWithin_append`.

**Produces:** `family`, `Vertex t`, `graph t`, finite/decidable instances, `nodeCount`, `edgeCount`, `nodeCount_formula`, `edgeCount_formula`, `coarseBound`, `bounded`.

- [ ] **1. Add RED tests for the family and generic identities.**

```lean
example : nodeCount 0 = 3 := nodeCount_zero
example : edgeCount 0 = 3 := edgeCount_zero
example (t : Nat) : edgeCount t = 3 ^ (t + 1) := edgeCount_formula t
example (t : Nat) : 2 * nodeCount t = 3 ^ (t + 1) + 3 := nodeCount_formula t
example (t : Nat) : GlobalHopBound (graph t).Adj (2 * t + 1) := coarseBound t
```

- [ ] **2. Push and inspect the structural-test RED; then implement a bundled family.** Use the following bundle so the graph and its carrier recurse together instead of creating a circular type/graph definition.

```lean
structure FiniteStage where
  Vertex : Type
  vertices : Fintype Vertex
  eqDec : DecidableEq Vertex
  graph : SimpleGraph Vertex
  adjDec : DecidableRel graph.Adj
```

`triangleStage : FiniteStage` has carrier `Fin 3` and adjacency inequality. `expandStage : FiniteStage -> FiniteStage` installs the input bundle's instances locally and uses `expandGraph`. Define `family 0 := triangleStage` and `family (t+1) := expandStage (family t)`. Define `Vertex t := (family t).Vertex`, `graph t := (family t).graph`, with their projected instances. Define `nodeCount` from `Fintype.card (Vertex t)` and `edgeCount` from actual `graph t` unordered edges, not recurrence counters.

- [ ] **3. Prove the counting bijections, then solve the recurrences.** For any finite old graph, partition the expanded edges into old edges and incidences `(e,v)` where `v` is one of `e`'s two distinct endpoints. Their images are respectively old–old edges and old–new edges; tags prove disjointness and unique inverse decoding.

```text
Edges(expand G) ≃ Edges(G) ⊕ (Σ e : Edges(G), {v // v ∈ e})
Vertices(expand G) ≃ Vertices(G) ⊕ Edges(G)
card({v // v ∈ e}) = 2
E(t+1) = 3*E(t)
N(t+1) = N(t)+E(t)
```

Prove these identities from finite cardinalities, then prove the two requested formulas by induction. Use the division-free vertex formula in the public theorem. Do not derive adjacency from the recurrence counts.

- [ ] **4. Prove coarse reachability and connectedness.** Base triangle pairs have length zero or one. Each expanded vertex is an old vertex or has an old endpoint at distance one, in both directions. Lift the old path and compose entry + old path + exit to get `L+2`; instantiate `L=2*t+1`. Export `bounded t : ∃ k, GlobalHopBound (graph t).Adj k := ⟨2*t+1, coarseBound t⟩`. Derive graph connectedness from the resulting walks and the nonempty triangle embedding. The bound is deliberately not an exact diameter formula.

- [ ] **5. Run root build and structural tests on GREEN; commit:** `feat(lean): prove pseudofractal counts and connectivity`.

## Task 3: Canonical numbered view with adjacency-preserving equivalence

**Files:** Extend `Pseudofractal.lean` and `Tests/Pseudofractal.lean` under the `Internal` namespace for encoding helpers.

**Consumes:** Task 2 family and Task 1 expansion. Counts may simplify dimensions but cannot establish equivalence.

**Produces:** `Internal.EncodedGraph`, `Internal.encoded`, `Internal.numberedGraph`, `Internal.encodingWellFormed`, `Internal.numbering`, `Internal.numberingIso`, `Internal.numbered_adj_iff`, `Internal.numberedBounded`.

- [ ] **1. Add RED exact-construction tests.**

```lean
open NarrativeDynamics.Pseudofractal.Internal
example : (encoded 0).nodeCount = 3 := by decide
example : (encoded 1).nodeCount = 6 := by decide
example : (encoded 1).edges =
    #[(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,5),(2,4),(2,5)] := by decide
example (t : Nat) (u v : Vertex t) :
    (numberedGraph t).Adj (numbering t u) (numbering t v) ↔
      (graph t).Adj u v := numbered_adj_iff t u v
```

The last test consumes the local `numbered_adj_iff` theorem produced in this task. Prove it from `numberingIso`; do not implement it as an assumed adjacency assertion. Check the pinned mathlib isomorphism accessor during implementation rather than exposing that moving accessor in every test.

- [ ] **2. Observe RED, then implement the raw encoding and its well-formedness predicate.**

```lean
structure EncodedGraph where
  nodeCount : Nat
  edges : Array (Nat × Nat)
  deriving DecidableEq, Repr
```

`encodingWellFormed : EncodedGraph -> Prop` requires every edge `(u,v)` to satisfy `u<v<nodeCount` and the entire list to be strictly lexicographically sorted. Provide its executable decision procedure. Define `toGraph (g : EncodedGraph) : SimpleGraph (Fin g.nodeCount)` by `u != v` and membership of `(min u.val v.val, max u.val v.val)` in `g.edges`; prove decidability, symmetry and looplessness.

`encoded : Nat -> EncodedGraph` is separate executable recursion. Its base is `⟨3, #[(0,1),(0,2),(1,2)]⟩`. For an old graph with N nodes and frozen sorted array E, enumerate E once. For entry i with endpoints (u,v), add (u,N+i) and (v,N+i), retain every old edge, sort the combined result, and set nodeCount to N+E.size. Sort using a proved lexicographic ordering; derive strictness from no duplicates, not from the sort alone. Do not mutate E during enumeration.

- [ ] **3. Prove numbered/mathematical equivalence by induction.** Let the old equivalence be phi. Map `.inl v` to the unchanged ID `phi(v)`. Map `.inr e` to `N + rank(sort(min(phi(u),phi(v)),max(phi(u),phi(v))))`. The unordered-pair map is independent of endpoint orientation. Prove membership, unique rank and exhaustive enumeration of the old canonical edge array. The inverse splits IDs at N and decodes either the old vertex or the ranked old edge.

Prove both inverse equations and all four adjacency cases. Package the result as `numberingIso t : graph t ≃g numberedGraph t`. Export `numbering` as its underlying equivalence and `numbered_adj_iff` with the exact test signature above. Prove numeric well-formedness for every t, and transfer bounded reachability using the isomorphism and `MeshWalk.mapNodes` to obtain `numberedBounded t`.

- [ ] **4. Verify all pairs of small numbered generations and cardinality transfer.** Add executable checks for t=0,1,2,3: node counts 3,6,15,42; edge counts 3,9,27,81; endpoints in range; sorted unique edges; old IDs retained. Instantiate the isomorphism theorem for each small generation, rather than using a numerical graph comparison as its proof. Explicit finite IDs use bounded constructors or validated decoding, not `Fin.ofNat`/modular coercions that could silently wrap an invalid witness ID.

- [ ] **5. GREEN and commit:** `feat(lean): prove canonical pseudofractal numbering`. Stop this task if count tests pass but adjacency equivalence is unproved. That is not a usable distance baseline.

## Task 4: Degree doubling, birth cohorts, exact spectrum and tails

**Files:** Extend the core and structural test modules.

**Consumes:** Actual finite graph degrees, expansion adjacency and finite-edge incidences; numbering only for small executable checks.

**Produces:** `old_degree`, `new_degree`, `birth`, `degreeCount`, `tailCount`, `degreeCount_dyadic`, `degreeCount_top`, `degreeCount_other`, `tailCount_dyadic`, and the lower bound `two_le_degree`.

- [ ] **1. Write the following RED theorem applications.**

```lean
example (t j : Nat) (hj : 1 ≤ j) (hjt : j ≤ t) :
    degreeCount t (2 ^ j) = 3 ^ (t - j + 1) := degreeCount_dyadic t j hj hjt
example (t : Nat) : degreeCount t (2 ^ (t+1)) = 3 := degreeCount_top t
example (t j : Nat) (hj : 1 ≤ j) (hjt : j ≤ t+1) :
    2 * tailCount t (2 ^ j) = 3 ^ (t+2-j) + 3 := tailCount_dyadic t j hj hjt
example : degreeCount 5 3 = 0 := by
  apply degreeCount_other 5 3
  rintro ⟨j, hj, hjt, h⟩
  interval_cases j <;> norm_num at h
```

`degreeCount_other t k` consumes `¬ ∃ j, 1 ≤ j ∧ j ≤ t+1 ∧ k=2^j`. The displayed test bounds j using the hypotheses before `interval_cases`; do not expect `decide` to enumerate an unbounded existential over naturals.

- [ ] **2. Inspect RED and implement actual count definitions.**

```lean
def degreeCount (t k : Nat) : Nat :=
  (Finset.univ.filter (fun v : Vertex t => (graph t).degree v = k)).card

def tailCount (t k : Nat) : Nat :=
  (Finset.univ.filter (fun v : Vertex t => k ≤ (graph t).degree v)).card
```

An old vertex's expanded neighbor set is the disjoint union of its old neighbors and its incident-edge newborns. Identify the incident edges with its old neighbors to prove `old_degree : degree_new(inl v)=2*degree_old(v)`. A newborn has exactly its two parent endpoints, proving `new_degree=2`.

- [ ] **3. Prove birth-cohort identities, then obtain the degree spectrum.** Define `birth 0 v=0`, `birth (t+1) (.inl v)=birth t v`, and `birth (t+1) (.inr e)=t+1`. Prove all births are at most t, seed cohort size is 3, and each b with `1≤b≤t` has exactly `3^b` vertices. For every vertex prove `degree(v)=2^(t-birth(v)+1)` and `2≤degree(v)`. Distinguish the seed cohort from newborn cohorts when computing the top degree.

Derive the dyadic spectrum using the unique b=t-j+1 and strict monotonicity of natural powers of 2. Derive the tail by the finite geometric sum of cohorts; keep the division-free count `2*tail=3^(t+2-j)+3`. Include j=1 and j=t+1 explicitly to catch range/seed errors.

**Natural-subtraction normalization:** the spec writes the algebraic tail exponent as `t-j+2`, with `j<=t+1`. In Lean naturals, `(t-j)+2` is incorrect at `j=t+1`: subtraction saturates at zero. Encode the intended exponent as `t+2-j`. In particular at t=0,j=1 the required doubled tail is 6, not 12; at t=5,j=6 it is also 6. This is normalization of the intended mathematical identity, not a different tail law. Add explicit tests for these endpoints and use proved inequalities before every other natural subtraction.

```lean
example : 2 * tailCount 0 (2 ^ 1) = 6 := by
  simpa using tailCount_dyadic 0 1 (by omega) (by omega)
example : 2 * tailCount 5 (2 ^ 6) = 6 := by
  simpa using tailCount_dyadic 5 6 (by omega) (by omega)
```

- [ ] **4. GREEN: instantiate t=0,1,2,3 and 5 using the generic proofs.** Check G5 degrees `{2:243,4:81,8:27,16:9,32:3,64:3}`; total 366 and degree sum 1458. These are corollaries of actual-degree results, not a replacement stored histogram.

- [ ] **5. Commit:** `feat(lean): prove pseudofractal dyadic degree spectrum`.

## Task 5: Measured clustering and exact rational mean

**Files:** Create `NarrativeDynamics/Core/PseudofractalMetrics.lean`; extend structural tests and add its root import.

**Consumes:** `closedNeighborPairs`, `orderedNeighborPairs`, `localClusteringCoefficient`, Task 4 actual degrees.

**Produces:** `localClustering_formula`, `meanLocalClustering`, `meanLocalClustering_formula`, and its G5 corollary. Internal finite sum helpers stay model-scoped.

- [ ] **1. Add RED applications of the actual measured coefficient.**

```lean
import NarrativeDynamics.Core.PseudofractalMetrics
example (t : Nat) (v : Vertex t) :
    localClusteringCoefficient (graph t).Adj v = 2 / ((graph t).degree v : ℚ) :=
  localClustering_formula t v
example : meanLocalClustering 5 = (51 : ℚ) / 64 := by
  rw [meanLocalClustering_formula]
  norm_num
```

- [ ] **2. Observe RED and prove the neighbor-pair counts.** Prove that the candidate ordered pair count is k*(k-1), where k is the actual degree. For closed pairs, prove induction invariant `closedPairs(v)=2*(k-1)`.

For a surviving old vertex, old closed pairs remain and each old incident edge contributes exactly the two orientations of the edge between its old opposite endpoint and its newborn. Thus closedPairs' = closedPairs + 2*k and degree' = 2*k. A newborn's two endpoints are joined by its retained parent edge, giving two closed ordered pairs. Exclude all other combinations using expansion adjacency, especially new–new edges. Use k>=2 before cancelling the positive rational denominator.

- [ ] **3. Define the measured mean and prove its closed form.**

```lean
def meanLocalClustering (t : Nat) : ℚ :=
  (∑ v : Vertex t, localClusteringCoefficient (graph t).Adj v) /
    (nodeCount t : ℚ)
```

Let S(t) be that numerator. Old coefficients halve and every newborn has coefficient 1. Prove `S(0)=3` and `S(t+1)=S(t)/2+edgeCount t`. An induction-friendly identity is `10*2^t*S(t)=12*6^t+18` over rationals. Combine it with the proved vertex count and nonzero denominators to derive:

```text
meanLocalClustering t =
  ((12*6^t+18 : Nat) : ℚ) /
  ((5*2^t*(3^(t+1)+3) : Nat) : ℚ)
```

Do not compute this as natural-number division. Prove invariance of the local coefficient under `numberingIso` using neighbor/pair bijections so the same measured mean applies to the numbered graph.

- [ ] **4. GREEN tests:** generic coefficient identity, t=0 mean 1, t=1 mean 3/4, t=5 mean 51/64, and equality of mathematical/numbered measurements. No global-transitivity theorem is substituted for the mean-local theorem.

- [ ] **5. Commit:** `feat(lean): prove pseudofractal clustering identities`.

## Task 6: Fail-closed distance certificates and proof of exactness

**Files:** Extend `PseudofractalMetrics.lean` and structural tests; helpers live under `Internal`.

**Consumes:** `EncodedGraph`, its actual `toGraph`, finite indices, the existing shortest-hop specifications. No BFS algorithm is needed yet.

**Produces:** `RawDistanceRow`, typed `DistanceRow n`, `decodeRow`, `ValidRow`, `checkRow`, `checkRow_sound`, `row_walk`, `label_le_walk`, `row_shortest_eq`, and `checkTable_sound`.

- [ ] **1. Add RED checker tests, including a case that isolates shortestness.**

```lean
open NarrativeDynamics.Pseudofractal.Internal

def tri : EncodedGraph := ⟨3, #[(0,1),(0,2),(1,2)]⟩
def line : EncodedGraph := ⟨3, #[(0,1),(1,2)]⟩
def lineRow : RawDistanceRow := ⟨0, #[0,1,2], #[none,some 0,some 1]⟩
def triRow : RawDistanceRow := ⟨0, #[0,1,1], #[none,some 0,some 0]⟩
example : checkRow line lineRow = true := by decide
example : checkRow tri triRow = true := by decide
example : checkRow tri lineRow = false := by decide
```

The last row has correct zero and decreasing adjacent parents even in the triangle, but violates the 0–2 edge inequality. This must be rejected specifically by the lower-bound condition; a parent-only checker would wrongly accept it.

- [ ] **2. Inspect RED and implement strict decoding, independent of search.**

```lean
structure RawDistanceRow where
  source : Nat
  labels : Array Nat
  parents : Array (Option Nat)
  deriving DecidableEq, Repr

structure DistanceRow (n : Nat) where
  source : Fin n
  label : Fin n → Nat
  parent : Fin n → Option (Fin n)
```

`decodeRow (n : Nat) (raw : RawDistanceRow) : Option (DistanceRow n)` rejects source>=n, either array length different from n, and any supplied parent>=n. Use proven bounded indexing after the length checks. A missing parent remains `none`; never map it or an absent label to zero. The source must have no parent; every other vertex must have one. n=0 has no valid source and is rejected.

`ValidRow (G : SimpleGraph (Fin n)) (r : DistanceRow n) : Prop` means: `(r.label v=0 ↔ v=r.source)` for all v; root parent is none; every non-root has parent p, G.Adj p v and label(p)+1=label(v); and `G.Adj u v -> label(v)<=label(u)+1` for every orientation. Derive a decidable predicate.

`checkRow (g : EncodedGraph) (raw : RawDistanceRow) : Bool` first rejects malformed graph encodings, then decodes. Test vertices for zero/parent conditions and iterate every canonical edge in both orientations for the inequalities. Prove this sparse test equivalent to `ValidRow g.toGraph r`; never verify a caller-supplied partial edge list in place of g's actual edges.

- [ ] **3. Prove the three semantic lemmas by ordinary induction.**

```text
row_walk: ValidRow G r -> MeshWalk G.Adj (r.label v) r.source v
label_le_walk: ValidRow G r -> MeshWalk G.Adj n u v ->
               r.label v <= r.label u + n
row_shortest_eq: ValidRow G r -> (bounded : ∃ k, GlobalHopBound G.Adj k) ->
                 shortestHopCount G.Adj bounded r.source v = r.label v
```

For `row_walk`, use strong induction on label(v); zero forces v=source, otherwise the parent has a smaller label and its edge appends one step. For `label_le_walk`, induct on the walk and compose edge inequalities. For `row_shortest_eq`, use the constructed walk for one inequality via `shortestHopCount_minimal`, and use `shortestHopCount_spec` plus the zero-source label for the other. Do not assume shortestness, global distances, or the certificate's advertised bound inside `ValidRow`.

- [ ] **4. Prove all-source coverage with checked row identities.** Define `checkTable (g : EncodedGraph) (rows : Array RawDistanceRow) : Bool`. Require exactly g.nodeCount rows, row i has source i, and every row passes `checkRow`. Derive a valid row and exact distance for every bounded source index. Duplicate or missing sources must not hide behind a correct row count. The lower-level typed result may use total finite functions; the executable input boundary remains fail-closed.

- [ ] **5. GREEN tests and axiom audit; commit:** `feat(lean): prove finite distance certificate soundness`. Print axioms of `checkRow_sound`, `label_le_walk` and `row_shortest_eq`; these generic proofs must not depend on native evaluation or admitted proof holes.

## Task 7: Deterministic certificate generation and adversarial coverage

**Files:** Extend the metrics and structural test modules. No external fixture or Python generator.

**Consumes:** Canonical numbered graph and Task 6 checker.

**Produces:** `bfsRow : EncodedGraph -> Nat -> Option RawDistanceRow`, `bfsTable : EncodedGraph -> Option (Array RawDistanceRow)`, proved adjacency-list reflection if a cache is used, and the complete negative test set.

- [ ] **1. Add RED tests for deterministic generation and rejection.**

```lean
example : bfsRow line 0 = some lineRow := by decide
example : bfsRow tri 0 = some triRow := by decide
example : bfsRow tri 3 = none := by decide
example : checkRow tri ⟨0, #[0,0,1], #[none,some 0,some 0]⟩ = false := by decide
example : checkRow tri ⟨0, #[0,1,1], #[none,none,some 0]⟩ = false := by decide
example : checkRow tri ⟨0, #[0,1,1], #[none,some 3,some 0]⟩ = false := by decide
example : checkRow line ⟨0, #[0,1,2], #[none,some 0,some 0]⟩ = false := by decide
example : checkRow tri ⟨0, #[0,1,1], #[none,some 2,some 0]⟩ = false := by decide
example : checkRow tri ⟨0, #[0,1], #[none,some 0]⟩ = false := by decide
example : checkTable tri #[triRow,triRow,triRow] = false := by decide
```

Also add cases with mismatched parent-array length, invalid source, a nonempty parent at the root, dropped/extra table rows and an empty graph. The already-defined triangle/line comparison tests the same-size wrong-topology case, not just a dimension mismatch.

- [ ] **2. Observe RED, then implement total BFS.** Construct ascending neighbor lists by folding the actual canonical edge array in both directions; prove membership iff `toGraph.Adj` if this derived cache is shared with checking. Queue starts with the source; labels are `Option Nat`, initially none except source=some 0; parents start none. Process queue FIFO, neighbors ascending, and assign a parent only on first discovery. Use a nodeCount processing budget with visited-on-enqueue, so the executable recursion terminates. Return none for malformed encodings, invalid sources or remaining undiscovered vertices; otherwise decode a complete row with exact dimensions.

No BFS shortestness proof is required for soundness: the checker must reject any bad output. For the finite acceptance computations, assert existence of a produced row/table and acceptance of that very output. Do not use `get!`, `getD`, `unsafe`, or a fallback row when generation returns none.

- [ ] **3. Exercise all rows for generations 0–3.** Define a total Boolean result by matching `bfsTable (encoded t)` and returning false on none. `decide`/reported `native_decide` checks that the output exists and `checkTable` accepts it. Recompute small scalar degree/clustering counts independently from numbered adjacency, and connect them to Task 3/5 isomorphism results. A producer/checker agreement alone is not a proof of graph correspondence.

- [ ] **4. GREEN and commit:** `test(lean): exercise pseudofractal distance certificates`. Record which small tests used kernel reduction and which used native evaluation. Preserve the triangle edge-inequality negative test even when BFS always emits valid labels.

## Task 8: Certify the 366-node graph and transfer every metric

**Files:** Create `NarrativeDynamics/Tests/PseudofractalSixHop.lean`; extend metrics with model-scoped semantic aggregate lemmas; add the named finite-check workflow step.

**Consumes:** Numbering isomorphism, `numberedBounded 5`, checked all-source rows, mean-clustering theorem.

**Produces:** Named concrete theorems `g5_globalHopBound`, `g5_diameter`, `g5_distance_109_362`, `g5_distance_histogram`, `g5_averageShortestPathLength`, and `g5_sixHopCoverage`. Concrete definitions and claims live in namespace `NarrativeDynamics.Pseudofractal.Tests` in the finite test file, not the root public proof API. With `g5 := numberedGraph 5` and `g5Bounded := numberedBounded 5`, the key conclusions are:

```text
g5_globalHopBound : GlobalHopBound g5.Adj 6
g5_diameter : meshDiameter g5.Adj g5Bounded = 6
g5_distance_109_362 : shortestHopCount g5.Adj g5Bounded u109 u362 = 6
g5_averageShortestPathLength : averageShortestPathLength g5.Adj g5Bounded = (76373 : ℚ)/22265
```

Construct `u109` and `u362` as bounded `Fin (encoded 5).nodeCount` values after proving that node count equals 366 from the cardinality-transfer theorem. They are not modular numeral coercions.

- [ ] **1. Write the finite RED file with exact targets.** Use `open ...` for established namespaces and define `g5 := numberedGraph 5`, `g5Bounded := numberedBounded 5`. Use the raw IDs 109 and 362 only after proving the exact numbered count and constructing bounded vertices. Add tests for a path of six edges and absence of any at-most-five path. Add the `Pseudofractal six-hop certificate tests` step after the structural step. Capture this feature's specific RED.

```text
G5: N=366, E=729, orderedDistinctPairs=133590
histogram (distances 1..6) = [1458,16944,52692,48432,13296,768]
sum of ordered distinct distances = 458238
mean = (458238 : ℚ)/133590 = 76373/22265
mean local clustering = 51/64
path IDs = [109,8,0,1,5,40,362]
```

- [ ] **2. Generate once, check once, and establish the measured interpretation.** Define `checkG5Evidence : Bool` in the finite test namespace: generate the canonical G5 and its BFS table; reject none; check `checkTable`; check every label<=6, exact dimensions, the concrete target label=6, and the stated histogram/sum. Aggregate only distinct source/target indices. Every matched table is the same table consumed by both validation and summarization; do not check one table and measure another.

Prove by ordinary finite sums that a checked table's labels equal semantic shortest hops (Task 6), so its distance histogram, sum, mean and coverage coincide with expressions over `shortestHopCount`. For the histogram define the semantic count as a filter of `orderedDistinctNodePairs`; for the mean use `averageShortestPathLength`. These bridge lemmas must exist before the concrete Boolean result is used. Pure finite summary equalities can then be discharged using the one accepted evidence computation.

- [ ] **3. Evaluate the finite evidence without bypassing the checker.**

```lean
theorem g5_evidence_accepted : checkG5Evidence = true := by
  native_decide
```

Native evaluation is permitted here and must be reported. Keep graph generation, adjacency derivation and row validation inside the evaluated dependency chain. Avoid evaluating `Nat.find`-based noncomputable metrics directly; rewrite them with the proved label equivalence. If computation is expensive, cache the actual graph/neighbor lists and fuse scans, with reflection proofs, rather than replacing the graph, importing a Python assertion, or weakening the goals.

- [ ] **4. Derive exact distance, diameter, average and coverage.** Project valid rows and bounds out of the accepted Boolean. `g5_globalHopBound` follows from row walks and labels<=6. `meshDiameter_minimal` gives diameter<=6; the target pair's proved shortest count=6 and `meshDiameter_spec` give diameter>=6. Derive `¬ReachWithin g5.Adj 5 u109 u362` from exact shortestness. Compute the rational mean from 458238/133590 only after its semantic sum equality is established. Check that exactly 768 ordered pairs need six hops, not 768 unordered pairs.

Use `numberingIso 5` in both directions to transfer the six-hop and exact-diameter statements to the mathematical `graph 5`, and transfer the arithmetic mean by the induced distinct-pair bijection. Export the corresponding mathematical-family test theorems; do not stop at the numbered view. The local clustering result transfers through Task 5, not through distance labels.

- [ ] **5. GREEN on root build, structural tests and named finite checks; commit:** `test(lean): certify the pseudofractal six-hop instance`. Print axioms for the concrete evidence and each exported metric theorem and identify native dependencies explicitly.

## Task 9: Seven-hop growth counterexample and final review evidence

**Files:** Extend `Tests/PseudofractalSixHop.lean`, finalize provenance logging in `proof.yml`, and update only the verified scope paragraph in `README.md`.

**Consumes:** All prior tasks; only one G6 BFS row is necessary.

**Produces:** `g6_distance_354_1087`, `g6_not_globalSix`, adversarial G5/G6 tests, and the exact-revision review report in the PR conversation.

- [ ] **1. Add the G6 RED assertions before implementing its concrete evidence.** Target numbered N=1095, source354, target1087, path `[354,36,3,0,2,13,119,1087]`, exact shortest count7. Also corrupt the accepted G5 target label from6 to5 and require `checkRow=false`; submit an unmodified G5 row against G6 and require rejection of the dimension mismatch. Never prove that every possible edited certificate is invalid: only the specified mutation or a false reachability claim is being tested.

- [ ] **2. Observe RED, then compute and check one G6 source row.** Match `bfsRow (encoded 6) 354`, reject none, check that row on the actual G6, and verify label(1087)=7. Apply `row_shortest_eq` with `numberedBounded 6`. Refute `GlobalHopBound ... 6` by specializing to these endpoints and applying `shortestHopCount_minimal`; transfer the refutation to the mathematical graph through `numberingIso 6`. Do not turn the existence of one seven-hop pair into an unproved statement that the entire G6 diameter is exactly7. No full G6 histogram is required.

- [ ] **3. Run final axiom and source audits.** Print axioms of construction/count/spectrum/clustering theorems, `checkRow_sound`, `row_shortest_eq`, `g5_evidence_accepted`, all concrete G5 metric theorems, and the G6 counterexample. Generic proofs must be ordinary proofs; concrete native assumptions must be identified and their propositions inspectable. Reject any admitted-proof dependency or handwritten axiom. Record output rather than treating a source-word search as sufficient.

- [ ] **4. Preserve all existing checks and log the actual checkout in the new named steps.** Add this provenance prefix to the structural step; leave the existing full build/conformance/old-theorem/Python commands intact.

```yaml
      - name: Pseudofractal structural theorem tests
        run: |
          export PATH="$HOME/.elan/bin:$PATH"
          git rev-parse HEAD
          printf 'event=%s github_sha=%s head_ref=%s base_ref=%s\n' \
            "$GITHUB_EVENT_NAME" "$GITHUB_SHA" "$GITHUB_HEAD_REF" "$GITHUB_BASE_REF"
          lake env lean --version
          lake env lean NarrativeDynamics/Tests/Pseudofractal.lean
      - name: Pseudofractal six-hop certificate tests
        run: |
          export PATH="$HOME/.elan/bin:$PATH"
          git rev-parse HEAD
          lake env lean NarrativeDynamics/Tests/PseudofractalSixHop.lean
```

Both steps run after the root Lean library build. Keep push and pull_request coverage. A push event checks the actual head; the ordinary PR checkout may be a synthetic merge ref. Record actual `git rev-parse HEAD` for each, and do not relabel one as the other. A docs-only skipped workflow earns no passing result.

- [ ] **5. Final scope/compatibility verification, documentation and commit.** Run:

```bash
lake build
lake env lean NarrativeDynamics/Tests/Pseudofractal.lean
lake env lean NarrativeDynamics/Tests/PseudofractalSixHop.lean
git diff --check
```

Use the unchanged workflow for the full conformance, all old theorem tests and Python suite. Update README only after the new proof/test results are known: triangle indexing, exact finite degree spectrum, mean-local metric, G5 exact six-hop/mean results, G6 seven-hop counterexample, and native trust boundary. Explicitly exclude stochastic models, asymptotic small-world laws, decentralized routing and six-round information delivery. Commit `docs: record verified pseudofractal scope and evidence`, push, and obtain final exact-head Lean evidence again because this commit changes the head. Report any failed or unresolved required check and retain draft status until review; no merge or automatic merge is authorized.

---

## Acceptance map and checkpoints

| Specification obligation | Owning task and decisive evidence |
| --- | --- |
| Triangle/frozen unordered-edge rule; one fresh vertex per old edge | 1–2; four adjacency cases, edge/vertex bijections. |
| Persistent sorted IDs and actual graph correspondence | 3; executable small graphs plus a generic graph isomorphism. |
| Finite, decidable, symmetric, loopless, connected | 1–2; construction and coarse bound, not certificate fields. |
| Actual counts, degree doubling, full spectrum and exact tail | 2,4; generic cardinality/cohort proofs with boundary cases. |
| Measured local clustering and rational mean | 5; actual neighbor counts and finite-sum recurrence. |
| Source-zero, parent and edge inequalities; shortestness | 6; decoder/reflection and two induction proofs. |
| Malformed indices, dimensions, parents, rows and wrong graph | 6–7,9; executable rejection tests including same-size triangle/path. |
| G5 counts, histogram, sum, mean, full six-hop coverage, exact diameter | 4–5,8; one checked table linked to existing metrics. |
| G6 refutes a global six-hop guarantee | 9; one exact seven-hop pair, not an assumed diameter. |
| Mathematical rather than only numbered claims | 3,5,8–9; adjacency/pair bijection transfer. |
| Native evaluation and axiom transparency | 6,8–9; generic/concrete audits recorded separately. |
| Preserve V24 and all existing checks | Every task; final changed-file review and unchanged checks. |

Review checkpoints: after Task 3 (representation correctness), Task 6 (certificate soundness), Task 8 (G5 mathematics), and Task 9 (complete scope/trust/CI evidence). A task is not complete just because an unrelated old workflow step is green.

## Baseline, execution evidence, and limitations of this plan commit

The approved spec records #58 run `33702197135`, attempt2, with Lean job `103113321382` successful and Python job `103113321545` failed. Re-read current run details at execution time. Do not claim the Python failure's root cause from its job conclusion alone and do not repair it opportunistically here.

At each checkpoint record: implementation SHA; actual tested checkout SHA; event type; run ID/attempt; Lean job ID; named new step results; full Python result; expected RED log; GREEN log; axiom output; and remaining review limitations. Keep this record in the PR conversation, not an expanding new runtime artifact subsystem. A failed old Python check does not invalidate a completed Lean theorem, but it prevents calling the overall required workflow green or the PR unconditionally merge-ready.

This plan changes no Lean source, tests or CI configuration. Its signatures and tactics are not presented as already compiled. The execution plan is based on source review, the committed mathematical spec and explicitly checked arithmetic targets, not on a fresh local Lean build. Review this plan before Task 1; no production execution is authorized by the plan-only commit itself.

## Source anchors for implementation

- Repository baseline: `c268e5355b14fb408f30814ed1c2bee2f41f1144`; spec revision: `a3938dfff05badbf182a11d9eb5a4d75104c9d8e`.
- Read `NarrativeDynamics/Core/SocialMesh.lean`, `SmallWorld.lean`, `SmallWorldMetrics.lean`, `WattsStrogatz.lean`, and `Tests/WattsStrogatz.lean` at that baseline for exact existing signatures and the existing native-evaluation convention.
- Pinned dependency source: `https://github.com/leanprover-community/mathlib4/blob/v4.32.0/Mathlib/Combinatorics/SimpleGraph/Finite.lean`. Prefer that checkout over names in the moving generated documentation when resolving helper APIs.
- API navigation: `https://leanprover-community.github.io/mathlib4_docs/Mathlib/Combinatorics/SimpleGraph/Finite.html` describes `edgeFinset`, `neighborFinset`, `incidenceFinset` and graph-isomorphism cardinality/degree transfer. Recheck signatures in v4.32.0 during elaboration.
- Trust guidance: `https://lean-lang.org/doc/reference/latest/Axioms/` and `https://lean-lang.org/doc/reference/latest/Tactic-Proofs/Tactic-Reference/`. These are moving references; the pinned compiler's actual `#print axioms` output is the evidence to record.
- Model source and all finite targets are recorded in the approved spec. Neither the literature nor the exploratory Python evidence substitutes for any Lean obligation above.
