# Fitness Attachment V23.5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, or superpowers:subagent-driven-development when a reviewer/worker facility is available. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an exact finite BB attachment model whose constant-fitness specialization is BA, with checked conditional probabilities, atomic graph growth, and deterministic replay.

**Architecture:** Separate rational normalization, the law of ordered choices without replacement, and the graph update. Derive weights from actual adjacency and immutable fitness; validate complete raw inputs before constructing a successor. Reuse existing mesh walks and metrics without importing `Pseudofractal.Internal` or creating a general probability framework.

**Tech Stack:** Lean `leanprover/lean4:v4.32.0`, mathlib `v4.32.0`, `SimpleGraph (Fin n)`, exact `Rat`, GitHub Actions `proof.yml`.

**Spec:** `docs/superpowers/specs/2026-09-11-fitness-attachment-v23-5-design.md`, commit `90e5055093a2d7b68095d5dcec07c5e3a851f102`, blob `f64ef994dd78532f0fb08f5aaa36a27908f92208`. The conversation's subsequent `go` approves moving from the committed design to this plan. This commit is plan-only; implementation awaits the plan review gate.

## Global Constraints

- Continue `feature/fitness-attachment-v23-5`, Draft PR #61, from design head `90e5055093a2d7b68095d5dcec07c5e3a851f102` and #58 base `c268e5355b14fb408f30814ed1c2bee2f41f1144`.
- “This is a sibling of #59 and #60, not stacked on their unfinished work.” Do not modify, close, merge, rebase, or copy their internal implementations.
- “Degree is computed from actual adjacency; there is no independent authoritative degree counter.”
- “Stored fitness is strictly positive and immutable for an existing vertex.”
- “Fix `m` with `1 <= m <= n0` for a growth trace.” The seed is connected and `n0 >= 2`.
- “Every draw is from remaining old vertices. The newborn is never eligible during its own birth.” Freeze graph, degrees, and fitness during a birth; renormalize after each selected target.
- “All-zero weights produce an explicit error, never a zero-sum ‘probability distribution’ or a uniform fallback.” Low-level normalization may accept zero individual weights; valid growth fitness may not be zero.
- “Validate the entire request before producing a successor.” No partial graph/fitness mutation or unchecked/defaulted array access.
- “An unordered target set's probability is the sum over its orderings, not one selected ordering's product.”
- “The executable boundary is deterministic replay of a supplied target trace, accompanied by its exact conditional probability.” No PRNG, entropy, continuous fitness sampler, or implicit quantization.
- “Generic proofs must not use `sorry`, `admit`, new user axioms, `native_decide`, or unsafe escape hatches.” No unlimited resource settings or suppression of failed checks.
- “Existing `MeshWalk`, `ReachWithin`, and `SmallWorldMetrics` semantics remain unchanged.” Do not weaken or require `SmallWorldCertificate`.
- Preserve Python/V24 runtime, dependency pins, research locks, and every existing workflow check. No automatic society mapping, transport, scheduler, belief, or observation change.
- No universal six-hop, pure-power-law exponent, condensation, clustering, asymptotic distance, or local-routing guarantee. No merge or auto-merge.

## Repository reconnaissance and file map

At the inspected design head, `lakefile.toml` has root target `NarrativeDynamics` and mathlib tag `v4.32.0`; `lean-toolchain` pins Lean 4.32.0. `proof.yml` ignores spec/plan-only diffs, builds the root, compares conformance vectors, runs the existing theorem files, and retains an independent full Python job. Do not treat the absence of a plan-only proof run as GREEN. No review comments were returned for #61 during this plan review.

`Core/SmallWorld.lean` already supplies `MeshWalk.mapNodes`, `reachWithin_append`, and `GlobalHopBound`; `Core/SmallWorldMetrics.lean` supplies actual neighbor sets, shortest-hop and diameter specifications/minimality. Reuse these semantics, not a second definition of distance. Before execution refresh the branch, applicable repository instructions, and pinned library APIs; do not assume another branch's CI status applies here.

| Path | Planned responsibility |
| --- | --- |
| `NarrativeDynamics/Core/FitnessAttachment.lean` | One scoped foundation: finite rows, graph-derived weights, choice law, growth, raw validation, replay. |
| `NarrativeDynamics/Tests/FitnessAttachment.lean` | Exact fixtures, generic theorem applications, invalid-input checks, positive-probability seven-hop counterexample, axiom audits. |
| `NarrativeDynamics.lean` | Add the core import only when Task 1 implementation exists. Never import tests. |
| `.github/workflows/proof.yml` | Add a named fitness contract step and checkout/dependency provenance; retain all existing checks. |
| `README.md` | Task 7: describe only the verified finite model and its limits. |

Use namespace `NarrativeDynamics.FitnessAttachment`; keep low-level helpers in `Internal`. Tests may access those helpers, but they are not a promised reusable public framework. `Fixtures` below belongs only to the test file. Seven serial tasks share these two source/test files; no parallel writes. Review checkpoints follow Tasks 3, 5, and 7. Every task still requires its own exact-head RED/GREEN review before the next task.

## Type and trust conventions

The interface blocks below specify required signatures; they are not axioms, stubs, or a claim of successful elaboration. Prove theorem bodies using the derivations given here and check concrete syntax on the pinned compiler. No production code is added by this plan.

- A rational row carries computed masses plus proved nonnegativity and total mass one. Its constructor is an output of normalization, never an input that assumes the desired answer.
- Executable enumeration uses `Fin`, finite functions/embeddings, lists, and checked arrays. Do not use `Fintype.ofFinite`, choice-based enumeration, or noncomputable selection in an executable path.
- Mathematical proofs may use classical reasoning; proof erasure must keep it out of data computation. Audit actual dependencies without describing classical theorems as axiom-free.
- Constant fitness c versus fitness 1 gives equal probabilities and topology, not identical stored fitness. Whole-replay scale invariance requires scaling both seed fitness AND every future newborn fitness.
- An empty remaining target sequence has mass one and performs no draw. A request for another row after exhausting candidates is an error. This includes the last draw when `m=n`.

## RED/GREEN protocol

For each task, add its tests first, commit/push, and inspect the exact checkout SHA and failing step. A missing new module/declaration is an intended RED; a typo, dependency failure, or unrelated Python failure is not. Implement only after that RED is observed. Preserve earlier tests. Never use failure wrappers, skipped tests, or `continue-on-error` to disguise the boundary.

Task 1 adds the fitness test step before the module exists; the old root build must pass and the new named step must fail on the missing import. Add the root import in GREEN. Execute these commands for every GREEN revision:

```sh
lake env lean --version
lake build
lake env lean NarrativeDynamics/Tests/FitnessAttachment.lean
git diff --check
```

Also require all original Lean/conformance/Story/Testimony steps on that same revision, and record the independent Python and World Studio outcomes honestly. Evidence records include event, requested SHA, actual checkout, attempt, job ID, failing/passing step, versions, and `#print axioms` output. A PR merge-ref success is not exact-head push evidence. No automatic rerun loop.

Add the following after checkout in both proof jobs, without changing checkout/security permissions:

```yaml
- name: Record checked revision
  run: |
    printf 'event=%s ref=%s requested_sha=%s run=%s attempt=%s\n' \
      "$GITHUB_EVENT_NAME" "$GITHUB_REF" "$GITHUB_SHA" \
      "$GITHUB_RUN_ID" "$GITHUB_RUN_ATTEMPT"
    git rev-parse HEAD
```

After successful dependency resolution, log `cat lean-toolchain`, `lake env lean --version`, and `git -C .lake/packages/mathlib rev-parse HEAD`. After the existing root build add:

```yaml
- name: Fitness attachment contract tests
  run: |
    export PATH="$HOME/.elan/bin:$PATH"
    lake env lean NarrativeDynamics/Tests/FitnessAttachment.lean
```

Do not transplant #60's workflow file; make these additive edits against #61's actual base.

## Task 1: Checked finite rational normalization

**Files:** Create the core and test files; add the named workflow step/provenance, then the root import in GREEN.

**Consumes:** Finite sums over `Fin n`, rational order/division, `Except`.

**Produces (inside `Internal`):**

```text
Error := negativeWeight | zeroMass | invalidNodeCount | fitnessSizeMismatch
       | nonpositiveFitness | invalidEdge | duplicateEdge | disconnectedSeed
       | invalidM | targetCountMismatch | targetOutOfRange | duplicateTarget
PosFitness := { eta : Rat // 0 < eta }
Row (n : Nat) := { mass : Fin n -> Rat, nonneg : forall i, 0 <= mass i,
                  total_one : (sum i, mass i) = 1 }
total (w : Fin n -> Rat) : Rat
remaining (w : Fin n -> Rat) (selected : Finset (Fin n)) : Fin n -> Rat
normalizePositive (w : Fin n -> Rat) (h : forall i, 0 <= w i)
                  (hz : 0 < total w) : Row n
normalize (w : Fin n -> Rat) : Except Error (Row n)
rowValues (r : Except Error (Row n)) : Except Error (List Rat)
normalizeValues (xs : List Rat) : Except Error (List Rat)
normalize_sum_one, normalize_mass_eq, normalize_support, normalize_nonneg
```

- [ ] **1. Add exact RED tests and the workflow step.** Define `Error` with `DecidableEq, Repr`; `rowValues` makes tests compare data, not proof-carrying functions. The tested helpers themselves belong to the implementation, not substitutes defined in the test.

```lean
import NarrativeDynamics.Core.FitnessAttachment
open NarrativeDynamics NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal

example : normalizeValues [2,4,8] = .ok [1/7,2/7,4/7] := by decide_cbv
example : normalizeValues [0,5,0] = .ok [0,1,0] := by decide_cbv
example : normalizeValues [0,0] = .error .zeroMass := by decide_cbv
example : normalizeValues [] = .error .zeroMass := by decide_cbv
example : normalizeValues [-1,2] = .error .negativeWeight := by decide_cbv
```

- [ ] **2. Commit/push `test(lean): specify exact fitness normalization` and record missing-module RED after the old root build.** Do not add implementation or root import beforehand.

- [ ] **3. Implement the arithmetic and checked construction.** Use these definitions, with explicit finite instances:

```lean
def total {n : Nat} (w : Fin n -> Rat) : Rat := Finset.univ.sum w

def remaining {n : Nat} (w : Fin n -> Rat)
    (selected : Finset (Fin n)) (i : Fin n) : Rat :=
  if i ∈ selected then 0 else w i
```

`normalize` first checks `forall i, 0 <= w i`, then `0 < total w`; return `.negativeWeight` or `.zeroMass` on failure. On success return `normalizePositive w h hz`, whose mass is `w i / total w`. Prove nonnegativity by division by a positive denominator; distribute the finite sum through division to prove total one. Under nonnegativity, zero total is exactly all-zero weights. `normalize_mass_eq` identifies every successful mass with the actual input ratio, not merely with some normalized row. Support is `mass i > 0 iff w i > 0`. Build `normalizeValues` from checked list indexing `Fin xs.length`; preserve length and ordering.

- [ ] **4. Add generic tests applying each produced theorem and print its axioms.** For `normalize w = .ok row`, test sum-one, the ratio equation for arbitrary i, support equivalence, and nonnegativity. These conclusions must follow from the constructor's success, not extra assumed output invariants.
- [ ] **5. Add the root import, obtain full exact-head GREEN, and commit `feat(lean): prove checked rational normalization`.** This is still only a finite row, not BB graph growth.

## Task 2: Actual graph weights, BA reduction, and fitness laws

**Files:** Extend the two core/test files.

**Consumes:** Task 1 rows, `neighborSet`, `SimpleGraph`, finite graph connectedness.

**Produces:**

```text
Snapshot (n : Nat): graph : SimpleGraph (Fin n), adjDec : DecidableRel graph.Adj,
                  fitness : Fin n -> Rat
Snapshot.Valid s := 2 <= n AND s.graph.Connected AND forall i, 0 < s.fitness i
State (n : Nat): snapshot : Snapshot n, valid : snapshot.Valid
degree (s : Snapshot n) (i : Fin n) : Nat
weights (s : Snapshot n) : Fin n -> Rat
attachmentRow (s : State n) (S : Finset (Fin n)) (hS : S.card < n) : Internal.Row n
asBA (s : State n) : State n
scaleFitness (s : State n) (c : Internal.PosFitness) : State n
baRow (s : State n) (S : Finset (Fin n)) (hS : S.card < n) : Internal.Row n
state_degree_pos, remaining_total_pos, attachment_ba, attachment_scale
attachment_ratio, attachment_mono, attachment_strict_mono
```

- [ ] **1. Add RED graph/kernel tests.** Add a test-local triangle `Fixtures.triangle : State 3` with actual top-graph adjacency and fitness `(1,2,4)`, proved valid by zero/one-hop paths. Add `Fixtures.constantTriangle : State 3` with fitness one. Define the fixture proofs ordinarily; do not assume valid outputs.

```lean
example : degree Fixtures.triangle.snapshot 0 = 2 := by decide_cbv
example : weights Fixtures.triangle.snapshot 2 = 8 := by decide_cbv
example : rowValues (.ok (attachmentRow Fixtures.triangle ∅ (by decide))) =
    .ok [1/7,2/7,4/7] := by decide_cbv
example : rowValues (.ok (attachmentRow Fixtures.triangle {2} (by decide))) =
    .ok [1/3,2/3,0] := by decide_cbv
example : rowValues (.ok (attachmentRow Fixtures.constantTriangle ∅ (by decide))) =
    .ok [1/3,1/3,1/3] := by decide_cbv
```

- [ ] **2. Push `test(lean): specify graph-derived BB and BA rows`; inspect missing-declaration RED.** Retain Task 1 checks.
- [ ] **3. Compute actual degree and prove normalization cannot fail.** Install `s.adjDec` locally; define `degree s i := (neighborSet s.graph.Adj i).card` and `weights s i := s.fitness i * (degree s i : Rat)`. Connectedness with `n>=2` gives a distinct reachable vertex and therefore a first incident edge, so every degree is positive. `S.card<n` supplies an unselected vertex; one strictly positive remaining weight plus all nonnegative weights gives positive total. Construct the row with Task 1's `normalizePositive` and these proved premises; no choice-based extraction of data is needed.
- [ ] **4. Prove the laws pointwise in mass.** `asBA` replaces fitness by one and keeps the graph; `baRow := attachmentRow (asBA s)`. Factor a common positive c out of the sum for BA reduction and scaling. Cancel a shared positive denominator for the odds ratio. For monotonicity isolate `x=eta_i*k_i` and competing mass `B`; cross-multiply `x/(x+B) <= x'/(x'+B)` to reduce it to `(x'-x)*B>=0`. Strictness requires both a strict fitness increase and `B>0`; only one eligible vertex always has probability one. Include zero-degree weights as low-level normalization examples, not as valid connected states.
- [ ] **5. Extend tests with generic law applications, scale `3/2`, and the singleton-candidate case.** The vector `(100,20)` with fitness `(1/10,1)` must yield `(1/3,2/3)`; BA vector weights yield `(5/6,1/6)`. Label this vector fixture explicitly as not a two-vertex graph. Audit all generic laws, obtain GREEN, commit `feat(lean): prove BA-compatible fitness attachment laws`.

## Task 3: Complete ordered-choice law without replacement

**Files:** Extend the two core/test files, probability helpers under `Internal`.

**Consumes:** Task 2 positive weights and rows; computable finite embeddings.

**Produces:**

```text
Targets (n m : Nat) := Fin m embedding_into Fin n
Targets.ordered (T : Targets n m) : List (Fin n)
Targets.selected (T : Targets n m) : Finset (Fin n)
traceMass (w : Fin n -> Rat) (S : Finset (Fin n)) (xs : List (Fin n)) : Rat
orderedMass (s : State n) (T : Targets n m) : Rat
setMass (s : State n) (m : Nat) (A : Finset (Fin n)) : Rat
orderedMass_pos, orderedMass_sum_one, setMass_sum_one
orderedMass_ba, orderedMass_scale, selected_card
```

Here `embedding_into` means Lean's `↪`, so `Targets n m` is `Fin m ↪ Fin n`, not a new structure with assumed length/nodup laws. Use its computable finite instance, and prove `Targets.selected.card=m` from injectivity. Define `ordered` by `List.ofFn T` and `selected` by the image of `Finset.univ`.

The normalization theorem must retain its dimension premise:

```text
orderedMass_sum_one (s : State n) (hm : m <= n) :
  (sum T : Targets n m, orderedMass s T) = 1
orderedMass_pos (s : State n) (T : Targets n m) : 0 < orderedMass s T
setMass_sum_one (s : State n) (hm : m <= n) :
  (sum A in univ.powersetCard m, setMass s m A) = 1
```

When m>n there are no embeddings and the corresponding sum is zero. Do not turn that impossible-choice family into a normalized law; raw growth requests reject it.

- [ ] **1. Add RED tests with test-local embeddings `Fixtures.t21`, `Fixtures.t12 : Targets 3 2`.** Their functions are `![2,1]` and `![1,2]`, with finite injectivity proofs. Include the empty embedding and all-target boundary.

```lean
example : Fintype.card (Targets 3 2) = 6 := by decide
example : orderedMass Fixtures.triangle Fixtures.t21 = 8/21 := by decide_cbv
example : orderedMass Fixtures.triangle Fixtures.t12 = 8/35 := by decide_cbv
example : setMass Fixtures.triangle 2 {1,2} = 64/105 := by decide_cbv
example : (∑ T : Targets 3 2, orderedMass Fixtures.triangle T) = 1 := by decide_cbv
example : (∑ T : Targets 3 3, orderedMass Fixtures.triangle T) = 1 := by decide_cbv
example : (∑ T : Targets 3 0, orderedMass Fixtures.triangle T) = 1 := by decide_cbv
```

- [ ] **2. Push `test(lean): specify complete without-replacement laws`; record RED.** These are new declarations, not a reason to modify earlier probabilities.
- [ ] **3. Define the finite mass and prove normalization by prefix induction.** Use this total mathematical helper:

```text
traceMass(w,S,[]) = 1
traceMass(w,S,i::rest) =
  if i is in S or total(remaining(w,S)) <= 0 then 0
  else (w(i) / total(remaining(w,S))) * traceMass(w,insert(i,S),rest)
orderedMass(s,T) = traceMass(weights(s.snapshot),empty,T.ordered)
setMass(s,m,A) = sum over T : Targets n m with T.selected=A of orderedMass(s,T)
```

Zero on an impossible *mathematical event* is not acceptance of an invalid API request. Raw repeated/out-of-range targets must return errors in Task 5. For positive w and `S.card+r<=n`, prove the sum of all legal r-step continuations is one. The base r=0 is exactly one, even when `S.card=n`; do not request a probability row then. For r+1, partition by the first unselected target, apply the induction hypothesis to the remaining r choices, and sum the first normalized row. Supply the bijection from an embedding to its head plus injective tail; no duplicate enumeration or missing first choices.

- [ ] **4. Derive complete laws and distinguish order from set.** Obtain ordered positivity and normalization from the empty-prefix theorem. Partition embeddings by their selected set for `setMass_sum_one` over subsets of cardinal m. Wrong-size sets have mass zero, not a renormalized distribution. Lift Task 2's common-factor cancellation through the conditional product for BA and scaling. Do not multiply m copies of the initial row.
- [ ] **5. Test each constant-fitness triangle pair has mass `1/6`, all three unordered pairs have mass `1/3`, and include m=1, m=n, m=0 laws.** m=0 is valid for the internal empty-choice law but remains invalid for a growth request. Print axioms; obtain GREEN and commit `feat(lean): prove finite BB target laws`. Stop at the probability-law review checkpoint before graph growth.

## Task 4: Atomic graph extension with exact structural invariants

**Files:** Extend the two core/test files.

**Consumes:** `State n`, `Targets n m`, positive newborn fitness, `MeshWalk.mapNodes`.

**Produces:**

```text
oldId (n : Nat) : Fin n embedding_into Fin (n+1)
newId (n : Nat) : Fin (n+1)
applyBirth (s : State n) (T : Targets n m) (hm : 0 < m)
           (etaNew : Internal.PosFitness) : State (n+1)
actualNodeCount (s : Snapshot n) : Nat
actualEdgeCount (s : Snapshot n) : Nat
birth_old_adj_iff, birth_new_adj_iff, birth_fitness_old, birth_fitness_new
birth_nodes, birth_edges, birth_degree_old, birth_degree_new, birth_degree_sum
birth_connected, birth_order_irrelevant, birth_new_fitness_independent
birth_walk_lifts, birth_reachWithin_lifts
```

- [ ] **1. Add RED tests using the already defined triangle and target embeddings.** Define test-local `Fixtures.after21 := applyBirth Fixtures.triangle Fixtures.t21 (by decide) ⟨3/2, by norm_num⟩` and the analogous `after12`. Add all four adjacency cases as generic examples, not just counts.

```lean
example : actualNodeCount Fixtures.after21.snapshot = 4 := by decide_cbv
example : actualEdgeCount Fixtures.after21.snapshot = 5 := by decide_cbv
example : (List.ofFn (degree Fixtures.after21.snapshot)) = [2,3,3,2] := by decide_cbv
example : (List.ofFn Fixtures.after21.snapshot.fitness) = [1,2,4,3/2] := by decide_cbv
example (i j : Fin 4) :
    Fixtures.after21.snapshot.graph.Adj i j ↔
    Fixtures.after12.snapshot.graph.Adj i j := by
  revert i j
  decide_cbv
```

- [ ] **2. Push `test(lean): specify atomic fitness graph growth`; inspect RED.** No raw API exists yet; typed inputs express the valid case.
- [ ] **3. Construct the new relation directly, not from a probability histogram.** `oldId` is the natural `Fin.castSucc` embedding and `newId=Fin.last n`. Case-split both vertices into old/new. Keep old adjacency; a mixed pair is adjacent exactly when its old vertex is in `T.selected`; new/new is false. Old fitness is read unchanged, newborn fitness is `etaNew.val`. Supply symmetry and looplessness by cases. `hm` and injectivity give a nonempty target set, which connects the newborn to the connected old graph; derive the new `State.valid`, do not accept it as an input assumption.
- [ ] **4. Prove actual counts and degrees.** Define node count by `Fintype.card (Fin n)` and edge count by the finite set of adjacent pairs with `u<v`; prove its bijection with actual unordered graph edges. Decompose successor edges as old edges plus `T.selected`, and successor neighbors as old neighbors plus the newborn iff selected. Derive +1 vertex, +m edges, degree m for the newborn, +1 for selected old vertices and +0 for others, then +2m for degree sum. Never use separately updated degree/edge counters.
- [ ] **5. Prove identity, ordering, and path contracts.** Same selected set and same newborn fitness give equal topology and fitness regardless of target order; the trace probabilities may still differ. Current-birth target probabilities do not consume `etaNew`; future rows do. Lift old exact walks using `mapNodes oldId` and old adjacency, then lift bounded walks with the same budget. Test generic statements, m=1 and m=n births, and that changing only newborn fitness leaves topology unchanged but changes a later row when the newborn competes with others.
- [ ] **6. Obtain GREEN and commit `feat(lean): prove atomic BB growth invariants`.** No general shortcut or old-carrier subgraph theorem may be misapplied to different vertex carriers.

## Task 5: Checked raw seeds and requests, without partial success

**Files:** Extend the two core/test files; raw helpers stay inside `Internal`.

**Consumes:** Typed growth, finite graph walks, actual degree/weight definitions.

**Produces:**

```text
RawSeed: nodeCount : Nat, fitness : Array Rat, edges : Array (Nat times Nat)
RawBirth: fitness : Rat, targets : Array Nat
ValidatedBirth (n m : Nat): targets : Targets n m, fitness : PosFitness
parseSeed (raw : RawSeed) : Except Error (State raw.nodeCount)
validateBirth (s : State n) (m : Nat) (raw : RawBirth) : Except Error (ValidatedBirth n m)
rawAttachmentRow (s : State n) (prefix : Array Nat) : Except Error (Row n)
step (s : State n) (m : Nat) (raw : RawBirth) : Except Error (State (n+1) times Rat)
reached (s : Snapshot n) (source : Fin n) (budget : Nat) : Finset (Fin n)
reached_iff, state_bounded, parseSeed_sound, parseSeed_complete
validateBirth_sound, validateBirth_complete, step_spec
```

`times` means Lean's product `×`. Raw dimensions are input declarations to validate, not trusted graph counters. `State.valid` returned by `parseSeed` must come from the checks and their soundness proofs.

- [ ] **1. Add RED request/seed tests with test-local `Fixtures.rawTriangle := ⟨3, #[1,2,4], #[(0,1),(0,2),(1,2)]⟩`.** Add `errorOf : Except Error α -> Option Error` in the test file by matching error/ok, so tests do not compare proof records.

```lean
example : errorOf (validateBirth Fixtures.triangle 0 ⟨1, #[]⟩) =
    some .invalidM := by decide_cbv
example : errorOf (validateBirth Fixtures.triangle 2 ⟨1, #[1,1]⟩) =
    some .duplicateTarget := by decide_cbv
example : errorOf (validateBirth Fixtures.triangle 2 ⟨1, #[1,3]⟩) =
    some .targetOutOfRange := by decide_cbv
example : errorOf (validateBirth Fixtures.triangle 2 ⟨0, #[1,2]⟩) =
    some .nonpositiveFitness := by decide_cbv
example : errorOf (parseSeed (⟨3, #[1,1], #[]⟩ : RawSeed)) =
    some .fitnessSizeMismatch := by decide_cbv
example : errorOf (parseSeed (⟨3, #[1,1,1], #[(0,1)]⟩ : RawSeed)) =
    some .disconnectedSeed := by decide_cbv
```

- [ ] **2. Push `test(lean): specify BB validation failures`; observe RED.** Include wrong count, m>n, negative fitness, loops, repeated edges, and invalid seeds before implementation.
- [ ] **3. Implement finite connectivity with an explicit soundness theorem.** Define `reached s a 0={a}`; each successor is the previous set union all actual neighbors of its members. Prove membership iff `ReachWithin s.graph.Adj budget a b`. A valid finite connected graph has simple paths of length at most n-1: derive this with mathlib path/erase-cycle facts on the pinned version, mapping to/from `MeshWalk`. Conversely, the checked bound supplies connectedness by actual walks. This gives `state_bounded s : GlobalHopBound s.snapshot.graph.Adj (n-1)` and makes the raw seed test both sound and complete. Keep this helper internal; do not add a general BFS certificate subsystem or import #60 internals.
- [ ] **4. Validate seeds in deterministic order.** Check n>=2; fitness array length exactly n; each fitness>0; every raw edge has two in-range distinct endpoints; normalize each pair by min/max and reject duplicate normalized pairs, including reversed duplicates; construct the simple graph from these validated pairs; check every vertex belongs to `reached graph 0 (n-1)`. Input edge order/orientation does not change the graph. Do not silently discard duplicates with `toFinset` before checking them. `parseSeed_complete` must show every raw seed satisfying these documented conditions is accepted, not only that returned seeds are valid.
- [ ] **5. Validate births in deterministic order.** Check `1<=m<=n`, positive incoming fitness, target array size=m, each target<n, then distinctness. Only now create `Targets n m` by bounded indexing and invoke `applyBirth`. `step` returns its successor together with `orderedMass` for the complete validated ordering. A late invalid target must return an error with no successor, not a valid partial birth. `rawAttachmentRow` checks index bounds and distinctness before converting a prefix to a set; a fully exhausted prefix returns `.zeroMass`. Completed m=n births succeed because they never ask for this extra row.
- [ ] **6. Add all error cases, positive boundary cases, and validator completeness tests.** Cover n=0/1, empty graph, disconnected seed, mismatched fitness dimensions, zero/negative fitness, loop/out-of-range/reversed-duplicate edges, wrong target count, m=0/>n, duplicate/out-of-range targets, and selecting ID n. Ensure malformed late entries cannot expose an intermediate state. Test m=n success, a valid one-element remaining row, arbitrary ordering of valid raw seed edges, and successful triangle step output. Audit the validators' soundness/completeness and step specification; obtain GREEN, commit `feat(lean): validate BB inputs before graph growth`. Stop at the validation/atomicity review checkpoint.

## Task 6: Finite replay, whole-trace laws, and compatibility

**Files:** Extend the two core/test files.

**Consumes:** Task 3 complete conditional laws and Task 5 checked step.

**Produces:**

```text
RunState: nodeCount : Nat, state : State nodeCount
ReplayResult: final : RunState, probability : Rat
ReplayError := seed(Error) | initialM | atBirth(index : Nat, cause : Error)
replay (seed : RawSeed) (m : Nat) (births : List RawBirth) : Except ReplayError ReplayResult
continuationMass (s : State n) (m : Nat) (hm : 0<m) (hbound : m<=n)
                 (schedule : List PosFitness) : Rat
replay_empty, replay_step, replay_valid, replay_nodes, replay_edges
replay_fitness_prefix, replay_probability_pos, continuationMass_one
replay_ba_topology, replay_ba_probability, replay_scale_topology, replay_scale_probability
```

`continuationMass` is the total law over all legal choices for a fixed supplied positive fitness schedule; it is a finite mathematical sum, not a new entropy source.

- [ ] **1. Add RED tests through the raw replay API.** Define test-local `summaryOf` mapping a successful result to `(actual nodes, actual edges, degree list, fitness list, probability)`, preserving errors; it reads actual final adjacency. Add these complete fixture assertions:

```text
replay(rawTriangle,2,[birth(3/2,[2,1])])
  -> nodes=4, edges=5, degrees=[2,3,3,2], fitness=[1,2,4,3/2], probability=8/21
replay(rawTriangle,2,[birth(3/2,[1,2])])
  -> same graph/fitness, probability=8/35
replay(rawTriangle,2,[birth(3/2,[2,1]), birth(1/3,[3,2])])
  -> nodes=5, edges=7, degrees=[2,3,4,3,2], probability=24/805
replay(rawTriangle,0,[]) -> error initialM
replay(rawTriangle,2,[birth(3/2,[2,1]), birth(1,[3,3])])
  -> error atBirth(1,duplicateTarget), no partial result
```

Translate these into `example : summaryOf (...) = .ok (...) := by decide_cbv`; use arrays for raw target literals, exact rational literals, and zero-based error positions. No default reads from failed results.

- [ ] **2. Push `test(lean): specify finite BB replay contracts`; record RED.** Do not substitute a test-only simulator for replay.
- [ ] **3. Implement pure left-to-right replay with initial validation even for an empty list.** Parse seed and check fixed initial m before processing births. Each successful `step` replaces the entire local state and multiplies the accumulated exact probability by its conditional mass. Wrap the first error with its birth index and return only the error. Start the product at one. The functions expose no external effects and no intermediate graph on failure.
- [ ] **4. Prove whole-trace statements by induction on the supplied birth list.** Each accepted step gives a valid successor. Iterate +1 nodes and +m edges to obtain actual counts `N0+t` and `E0+m*t`. Compose natural old-ID embeddings to preserve seed fitness and each newborn's fitness at its assigned stable ID. All accepted trace probabilities are positive. Expand `continuationMass` as a sum over `Targets n m` of `orderedMass(s,T) * continuationMass(applyBirth(s,T,eta),rest)`. The empty schedule is one. Apply induction to the continuation and Task 3 normalization to prove total one; updated graph weights must be used at the next birth.
- [ ] **5. Prove BA and scaling over entire runs, not just one row.** With a constant positive c for seed and every birth, topology and trace probabilities equal the degree-only BA specialization; fitness storage is c, not necessarily one. For scaling, multiply every seed fitness and every newborn fitness by c, replay the same target IDs, and inductively show equal graphs, scaled fitness vectors, and equal probabilities. Scaling only seed fitness is deliberately NOT this theorem.
- [ ] **6. Add a negative scope fixture for incorrect scaling.** In the two-birth example above, scaling seed fitness by 2 but leaving the first newborn at 3/2 changes trace probability to `24/1505`; scaling both seed and births retains `24/805`. This is a regression test against applying a single-row cancellation theorem too broadly. Add empty valid replay, a non-triangle seed, and multi-birth m=n0 cases. Print all replay/normalization axioms, obtain GREEN, commit `feat(lean): prove finite BB replay and BA compatibility`.

## Task 7: Positive-probability seven-hop counterexample and final audit

**Files:** Extend tests; update README after verified results. Core changes only for a missing helper required by the same contract.

**Consumes:** Typed/replayed growth, `state_bounded`, existing shortest-hop/diameter specifications and minimality.

**Produces (test-local named theorems):** `path8_replay`, `path8_positive_mass`, `path8_adj`, `path8_walk_seven`, `path8_no_six`, `path8_shortest_seven`, `path8_diameter_seven`.

- [ ] **1. Add RED acceptance targets for the actual replayed path.** Start with two nodes, edge (0,1), fitness (1,1), m=1. Births have fitness 1 and targets `[1],[2],[3],[4],[5],[6]`. Require the raw replay to match the typed six-successor fixture, not an independently supplied eight-node path graph.

```text
actual nodes = 8; actual edges = 7
actual degrees = [1,2,2,2,2,2,2,1]
trace probability = (1/2)*(1/4)*(1/6)*(1/8)*(1/10)*(1/12) = 1/46080 > 0
adj(u,v) iff u.val+1=v.val OR v.val+1=u.val
ReachWithin graph.Adj 7 0 7 AND NOT ReachWithin graph.Adj 6 0 7
shortestHopCount graph.Adj bounded 0 7 = 7
meshDiameter graph.Adj bounded = 7
```

- [ ] **2. Observe RED at the new named theorem/contract, not an unrelated prior failure.** Add exact fixture equality/count/probability tests with `decide_cbv` and generic lower-bound theorem applications; no expected-failure wrapper.
- [ ] **3. Prove both sides of exact distance.** Verify every finite adjacency pair against the real replay result and construct the explicit walk 0→1→2→3→4→5→6→7. Prove by `MeshWalk` induction that any walk in this graph satisfies `target.val <= source.val + length`, since every edge changes the label by at most one. A budget of six for endpoints 0 and 7 contradicts this inequality. Combine the seven-hop witness, the lower bound, and the existing shortest-hop minimality/specification to obtain exact distance. `state_bounded` gives all-pairs upper bound 7; the endpoint lower bound forces diameter exactly 7.
- [ ] **4. Complete the trust and scope review.** Audit normalization, support, BA/scaling, ordered/set laws, growth/validation/replay, and counterexample dependencies using actual `#print axioms`. Scan source for forbidden placeholders/axioms/native evaluation, unchecked array defaults, noncomputable executable definitions, silent fallback normalization, and changed old tests. `decide_cbv` is equation-based proof evaluation, not permission to disable guards or switch to a native oracle. Preserve finite resource budgets and report remaining warnings.
- [ ] **5. Run final exact-head verification and record actual outcomes.** Retain all old checks; inspect the successful complete Lean job and independent Python/World Studio results, not just a middle build step. Update README with fixed rational-fitness, simple-graph, without-replacement semantics, BA compatibility, exact replay scope, and the positive-probability non-six-hop example. State that power-law statistics and six-hop results require separate analysis. Commit `test(lean): verify BB scope counterexample and trust boundary`, then stop for final review with PR still Draft and no merge authorization.

## Coverage checklist and execution handoff

| Spec sections | Implementation tasks |
| --- | --- |
| 1: Sibling branches, no pseudofractal replacement | Global constraints, file map, every review checkpoint |
| 2: Exact fixed-fitness profile and valid seed | 2, 4, 5, 6 |
| 3: Conditional kernel, zero/error handling | 1, 2, 3, 5 |
| 4: Finite law, unordered events, no entropy | 3, 6 |
| 5: Atomic growth, actual counts, path lifting | 4, 5, 6 |
| 6: BA reduction, scaling, odds and monotonicity | 2, 3, 4, 6 |
| 7: Seven-hop counterexample and epistemic limits | 7; preserved globally |
| 8: Positive and adversarial fixtures | 1 through 7 |
| 9–10: Module scope, RED/GREEN, trust, unchanged CI | File map, protocol, each task, final audit |
| 11: References and finite-claim boundaries | Approved spec remains authoritative; no new asymptotic claims |

Plan review must check especially the full-support/empty-choice boundary, conditional versus initial probabilities, same-topology versus full-state equality, raw-validator completeness, and the lower-bound half of the path counterexample. None of these are solved by degree histograms alone.

At this document's commit, no task is executed and no BB theorem is reported as compiled. Source inspection and independent rational fixture arithmetic are plan checks only. After plan approval, enter Task 1 test-first RED on the current #61 branch; do not reopen the already approved model choice or begin Task 2 before Task 1 GREEN. Use inline execution with checkpoints when no subagent facility is available; do not claim an independent review that did not occur.
