# BB finite-path convergence implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Execute one task at a time, preserve each RED/GREEN boundary, and review before advancing.

**Goal:** Generalize the verified Path4/Path5 BB belief dynamics to every finite path `Fin n` with `2 ≤ n`, proving coordinatewise convergence of the actual `NetworkPropagation.propagate` trajectory to the degree-weighted stationary mean without an explicit spectral closed form.

**Architecture:** Keep executable BB dynamics unchanged. Add a proof-only `FiniteConsensus` layer over finite rational kernels and a path-specific `FitnessABMPathN` layer that derives its kernel from actual propagation inside the all-broadcast region. Prove a path-specific `(n - 1)`-step common-column mass lower bound `δ(n) = (1/4)^(n-1)`, use it for geometric coordinate-range contraction, and cast only the final convergence argument to `Real`.

**Tech Stack:** Lean 4.32.0, mathlib v4.32.0, exact `Rat`, `Matrix`, existing `NetworkPropagation`, existing fitness trust audit, existing GitHub proof workflow.

**Spec:** `docs/superpowers/specs/2026-09-15-bb-pathn-convergence-design.md`, approved at `c3ba593215f551fb9473ce854c508870755a0077`.

## Global constraints

- `beliefStep` remains defined through `NetworkPropagation.propagate`; matrices/kernels are proof-only.
- Keep existing Path4 and Path5 production modules and theorem surfaces intact.
- Do not add a Path6 production module; `n = 6` is only a fixed generic smoke test.
- Do not prove convergence for arbitrary connected graphs, arbitrary stochastic matrices, arbitrary receptivity, or arbitrary thresholds.
- Profiles remain receptivity `1/2`, inclusive threshold `1/2`.
- Generic convergence requires explicit `2 ≤ n` and explicit initial `allBroadcast`.
- Stay over `Rat` until the final analytic layer; cast to `Real` only for the limit proof.
- No `sorry`, `admit`, `native_decide`, new `axiom`, `unsafe`, `unlock_limits`, or unlimited resource settings.
- Allowed transitive axioms remain exactly `propext`, `Classical.choice`, `Quot.sound`.
- Every focused build/test in the final gate uses `timeout --kill-after=10s 240s` and GNU `time` peak RSS.
- Do not import `FiniteConsensus` or `FitnessABMPathN` from `NarrativeDynamics.lean` in this PR. Explicit gate builds must prove these specialized modules compile.
- A resource failure is a proof-structure signal. Factor the proof before changing limits.

## File map

| File | Responsibility | Tasks |
| --- | --- | --- |
| `NarrativeDynamics/Core/FiniteConsensus.lean` | Generic rational averaging, stationary weights, range contraction, generic convergence | 1, 6 |
| `NarrativeDynamics/Tests/FiniteConsensus.lean` | Generic RED/GREEN consumers and axiom reports | 1, 6, 8 |
| `NarrativeDynamics/Core/FitnessABMPathN.lean` | Generic path, actual propagation bridge, stationary degree weights, common mass, final theorem | 2–6 |
| `NarrativeDynamics/Tests/FitnessABMPathN.lean` | Generic contracts, compatibility, n=6 smoke, axiom reports | 2–7 |
| `tools/check_fitness_abm_pathn.sh` | Dedicated bounded build/test/trust gate | 8 |
| `.github/workflows/proof.yml` | Dedicated PathN step after existing Path4/Path5 gate | 8 |
| Existing Path4/Path5/conformance files | Regression evidence only | 7–8 |

## Preparation before Task 1

- [ ] Create `feature/bb-pathn-convergence-v1` from the committed plan head, not from `master`.

```bash
git checkout design/bb-pathn-convergence-v1
git pull --ff-only
git checkout -b feature/bb-pathn-convergence-v1
git rev-parse HEAD
```

- [ ] Record the implementation base and verify the starting diff contains only the approved spec and this plan.

```bash
git merge-base HEAD proof/narrative-dynamics-v0
git diff --stat proof/narrative-dynamics-v0...HEAD
```

- [ ] Before editing, read `NetworkPropagation.lean`, Path4, Path4Convergence, Path5, Path5Vectors, `check_fitness_abm_path4.sh`, and `audit_fitness_trust.py`.

---

## Task 1: FiniteConsensus RED and exact averaging foundation

**Files:** create `NarrativeDynamics/Core/FiniteConsensus.lean`; create `NarrativeDynamics/Tests/FiniteConsensus.lean`.

**Purpose:** Establish a generic finite rational kernel vocabulary, coordinate extrema/range, stationary weights, and one-step preservation/bounds. Do not prove asymptotic convergence yet.

### Public interfaces

```lean
namespace NarrativeDynamics.FiniteConsensus

open scoped BigOperators

abbrev Kernel (ι : Type*) := Matrix ι ι Rat

def applyKernel [Fintype ι] (K : Kernel ι) (x : ι → Rat) : ι → Rat :=
  fun i => ∑ j, K i j * x j

structure AveragingKernel [Fintype ι] (K : Kernel ι) : Prop where
  nonneg : ∀ i j, 0 ≤ K i j
  row_sum : ∀ i, ∑ j, K i j = 1

structure StationaryWeights [Fintype ι] (K : Kernel ι) (π : ι → Rat) : Prop where
  nonneg : ∀ i, 0 ≤ π i
  sum_one : ∑ i, π i = 1
  stationary : ∀ j, ∑ i, π i * K i j = π j

def weightedMean [Fintype ι] (π x : ι → Rat) : Rat :=
  ∑ i, π i * x i

def kernelTrajectory [Fintype ι] (K : Kernel ι) (x : ι → Rat) (k : Nat) : ι → Rat :=
  (applyKernel K)^[k] x

def CommonColumnMass [Fintype ι] (K : Kernel ι) (δ : Rat) : Prop :=
  ∃ c : ι, ∀ i, δ ≤ K i c
```

Under `[Fintype ι] [Nonempty ι] [DecidableEq ι]`, define `coordMin`, `coordMax`, and `coordRange x := coordMax x - coordMin x`. Consumers must not depend on the internal finite-min/max representation.

Required Task-1 theorems:

```text
coordMin_le
le_coordMax
coordRange_nonneg
applyKernel_between
coordRange_apply_le
weightedMean_apply
weightedMean_between
coordinate_dist_weightedMean_le_range
kernelTrajectory_zero
kernelTrajectory_succ
applyKernel_mul
kernelPow_apply
```

`weightedMean_apply` consumes `StationaryWeights K π` and proves `weightedMean π (applyKernel K x) = weightedMean π x` by finite-sum rearrangement. `kernelPow_apply` fixes matrix/application orientation once:

```lean
applyKernel (K ^ k) x = kernelTrajectory K x k
```

### TDD

- [ ] **RED:** create the consumer before the core module.

```lean
import NarrativeDynamics.Core.FiniteConsensus

open NarrativeDynamics.FiniteConsensus

private def pairKernel : Kernel (Fin 2) := fun _ _ => 1/2
private def pairWeights : Fin 2 → Rat := fun _ => 1/2

example : applyKernel pairKernel ![0, 1] = ![1/2, 1/2] := by decide_cbv
example : weightedMean pairWeights ![0, 1] = 1/2 := by decide_cbv
example (x : Fin 2 → Rat) : 0 ≤ coordRange x := coordRange_nonneg x
```

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FiniteConsensus.lean
```

Expected RED: missing `NarrativeDynamics.Core.FiniteConsensus`. A missing Lean toolchain is not RED evidence.

- [ ] **GREEN:** implement only the interfaces and basic lemmas. Prove output bounds from nonnegative row coefficients summing to one; derive one-step range nonincrease. Prove stationary mean preservation with sum rearrangement, not coordinate enumeration.

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FiniteConsensus
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FiniteConsensus.lean
```

- [ ] Commit.

```bash
git add NarrativeDynamics/Core/FiniteConsensus.lean \
        NarrativeDynamics/Tests/FiniteConsensus.lean
git commit -m "feat(lean): add finite consensus kernel foundation"
```

**Review gate:** `FiniteConsensus` imports no BB/path module and contains no graph/connectivity theorem.

---

## Task 2: Generic finite-path structure and proof kernel

**Files:** create `NarrativeDynamics/Core/FitnessABMPathN.lean`; create `NarrativeDynamics/Tests/FitnessABMPathN.lean`.

### Required interfaces

```lean
namespace NarrativeDynamics.FitnessABMPathN

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FiniteConsensus
open scoped BigOperators

abbrev Beliefs (n : Nat) := Fin n → Rat

def pathAdj (n : Nat) (i j : Fin n) : Prop :=
  i.val + 1 = j.val ∨ j.val + 1 = i.val

instance (n : Nat) : DecidableRel (pathAdj n) := fun i j => inferInstance

def neighbors (n : Nat) (i : Fin n) : Finset (Fin n) :=
  Finset.univ.filter fun j => pathAdj n j i

def degree (n : Nat) (i : Fin n) : Nat := (neighbors n i).card

def population (n : Nat) (x : Beliefs n) (e : Fin n → Nat) : Population n :=
  ⟨fun _ => ⟨1/2, 1/2⟩, fun i => ⟨x i, e i⟩⟩

def project (n : Nat) (p : Population n) : Beliefs n :=
  fun i => (p.agents i).belief

def allBroadcast (n : Nat) (x : Beliefs n) : Prop :=
  ∀ i, 1/2 ≤ x i ∧ x i ≤ 1

def beliefStep (n : Nat) (x : Beliefs n) : Beliefs n :=
  project n (propagate (pathAdj n) (population n x (fun _ => 0)))

def trajectory (n : Nat) (x : Beliefs n) (k : Nat) : Beliefs n :=
  (beliefStep n)^[k] x

def pathKernel (n : Nat) : Kernel (Fin n) := fun i j =>
  if i = j then 1/2
  else if pathAdj n j i then 1 / (2 * (degree n i : Rat))
  else 0
```

Required structural theorems for `hn : 2 ≤ n`:

```text
neighbors_nonempty
degree_pos
degree_le_two
pathKernel_nonneg
pathKernel_row_sum
pathKernel_averaging
pathKernel_self
pathKernel_self_lower
pathKernel_adj_lower
```

### TDD

- [ ] **RED:** fixed n=6 consumers force generic declarations without proving anything by generic enumeration.

```lean
import NarrativeDynamics.Core.FitnessABMPathN

open NarrativeDynamics.FitnessABMPathN

example : pathAdj 6 (0 : Fin 6) 1 := by decide
example : degree 6 0 = 1 := by decide_cbv
example : degree 6 3 = 2 := by decide_cbv
example : pathKernel 6 0 0 = 1/2 := by decide_cbv
example : pathKernel 6 0 1 = 1/2 := by decide_cbv
example : pathKernel 6 3 2 = 1/4 := by decide_cbv
```

Run the test before creating the core module; expected missing-import RED.

- [ ] **GREEN:** implement path structure. Prove degree facts structurally from `Fin` inequalities. Endpoint/interior splitting is allowed; `fin_cases i` over generic `Fin n` is not.
- [ ] Prove row sum by reducing the neighbor contribution to `degree * (1/(2*degree))` and using `degree_pos`.

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMPathN
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathN.lean
```

- [ ] Commit.

```bash
git add NarrativeDynamics/Core/FitnessABMPathN.lean \
        NarrativeDynamics/Tests/FitnessABMPathN.lean
git commit -m "feat(lean): define generic finite path kernel"
```

**Review gate:** `beliefStep` visibly calls actual `propagate`; `pathKernel` is not the executable definition.

---

## Task 3: Actual-propagation bridge and all-broadcast invariance

**Files:** modify PathN core/test.

### Required theorems

```lean
theorem propagate_independent_exposures
    (n : Nat) (x : Beliefs n) (e : Fin n → Nat) :
  project n (propagate (pathAdj n) (population n x e)) = beliefStep n x

theorem incoming_eq_neighbors
    (n : Nat) (x : Beliefs n) (e : Fin n → Nat)
    (hx : allBroadcast n x) (i : Fin n) :
  incoming (pathAdj n) (population n x e) i = neighbors n i

theorem propagate_eq_kernel
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n) (e : Fin n → Nat)
    (hx : allBroadcast n x) :
  project n (propagate (pathAdj n) (population n x e)) =
    applyKernel (pathKernel n) x

theorem allBroadcast_step
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : allBroadcast n x) :
  allBroadcast n (beliefStep n x)

theorem allBroadcast_iterate
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : allBroadcast n x) (k : Nat) :
  allBroadcast n (trajectory n x k)

theorem trajectory_eq_kernelTrajectory
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : allBroadcast n x) (k : Nat) :
  trajectory n x k = kernelTrajectory (pathKernel n) x k
```

### TDD

- [ ] **RED:** add consumers for `propagate_independent_exposures` and `trajectory_eq_kernelTrajectory`, then run the focused PathN test and record unknown declarations.
- [ ] **GREEN:** prove exposure independence outside the broadcast region by the same locality used by Path4/Path5: profiles/beliefs determine incoming and signal; exposures do not.
- [ ] Prove `incoming_eq_neighbors`: under `allBroadcast`, the incoming filter reduces exactly to adjacency.
- [ ] Derive `propagate_eq_kernel` from `nextAgent`; self `1/2` and neighbor `1/(2*degree)` coefficients must come from actual receptivity/neighbor averaging.
- [ ] Prove `allBroadcast_step` from `pathKernel_averaging` + generic convex bounds, then iterate it.
- [ ] Prove actual `trajectory = kernelTrajectory` by induction, using the invariant to reapply the bridge at every step.

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMPathN
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathN.lean
git add NarrativeDynamics/Core/FitnessABMPathN.lean \
        NarrativeDynamics/Tests/FitnessABMPathN.lean
git commit -m "feat(lean): bridge generic paths to actual propagation"
```

**Review gate:** no generic theorem may prove `Fin n` cases by `fin_cases`.

---

## Task 4: Stationary degree weights and invariant mean

**Files:** modify PathN core/test; modify FiniteConsensus only if a truly generic helper is missing.

### Required interfaces

```lean
def weightSum (n : Nat) : Rat :=
  ∑ i : Fin n, (degree n i : Rat)

def stationaryWeight (n : Nat) (i : Fin n) : Rat :=
  (degree n i : Rat) / weightSum n

def mean (n : Nat) (x : Beliefs n) : Rat :=
  weightedMean (stationaryWeight n) x
```

Required theorems for `hn : 2 ≤ n`:

```text
path_degree_sum
weightSum_pos
stationaryWeight_nonneg
stationaryWeight_sum_one
pathKernel_detailed_balance
path_stationary_weights
mean_kernel_step
mean_kernel_iterate
mean_step
mean_iterate
mean_between
```

`path_degree_sum` must use one explicit rational typing convention consistently; e.g. `weightSum n = ((2 * (n - 1) : Nat) : Rat)`.

### TDD

- [ ] **RED:** add both consumers before implementation.

```lean
example (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : allBroadcast n x) :
    mean n (beliefStep n x) = mean n x :=
  mean_step n hn x hx

example (n : Nat) (hn : 2 ≤ n) (x : Beliefs n) (k : Nat) :
    mean n (kernelTrajectory (pathKernel n) x k) = mean n x :=
  mean_kernel_iterate n hn x k
```

- [ ] **GREEN:** prove degree sum structurally (endpoint/interior Finset decomposition or a narrow finite path handshake argument).
- [ ] Prove detailed balance by `i=j`, adjacent-distinct, nonadjacent cases; do not generalize graph classes.
- [ ] Derive normalized stationarity from detailed balance + row sum.
- [ ] Derive `mean_kernel_step` from generic `weightedMean_apply`; induct for `mean_kernel_iterate`. Derive actual `mean_step` via `propagate_eq_kernel` and actual `mean_iterate` via trajectory induction. `mean_between` comes from normalized nonnegative weights.

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FiniteConsensus
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMPathN
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FiniteConsensus.lean
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPathN.lean
```

- [ ] Commit changed files only.

```bash
git add NarrativeDynamics/Core/FitnessABMPathN.lean \
        NarrativeDynamics/Tests/FitnessABMPathN.lean
git add NarrativeDynamics/Core/FiniteConsensus.lean \
        NarrativeDynamics/Tests/FiniteConsensus.lean 2>/dev/null || true
git commit -m "feat(lean): prove generic path stationary mean"
```

---

## Task 5: `(n - 1)`-step common-column mass

**Files:** modify PathN core/test.

### Required definitions/theorems

```lean
def block (n : Nat) : Nat := n - 1
def delta (n : Nat) : Rat := (1/4 : Rat) ^ block n
```

For `hn : 2 ≤ n`:

```text
block_pos
delta_pos
delta_lt_one
one_sub_delta_pos
one_sub_delta_lt_one
path_block_common_mass
```

Main theorem:

```lean
theorem path_block_common_mass
    (n : Nat) (hn : 2 ≤ n) :
  CommonColumnMass ((pathKernel n) ^ block n) (delta n)
```

The common column witness is node `0`.

### Proof construction

Do not expand the entire matrix power.

1. `left_reach_mass`: after `i.val` steps, coefficient `(pathKernel n ^ i.val) i 0` is at least `(1/4)^i.val`; induct along `0 → 1 → ... → i`, retaining only the predecessor term from a nonnegative multiplication sum.
2. `self_pad_mass`: an additional self transition preserves at least another factor `1/4`; iterate for `block n - i.val` steps.
3. Combine `i.val ≤ n-1` and exponent arithmetic to reach exactly `(1/4)^(n-1)`.

If matrix-power elaboration is the bottleneck, introduce a local recursive coefficient helper and prove equivalence to matrix power. Do not raise global limits.

### TDD

- [ ] **RED:**

```lean
example (n : Nat) (hn : 2 ≤ n) :
    CommonColumnMass ((pathKernel n) ^ block n) (delta n) :=
  path_block_common_mass n hn
```

Then add fixed sanity checks:

```lean
example : block 6 = 5 := by decide
example : delta 6 = 1/1024 := by norm_num [delta, block]
```

- [ ] **GREEN:** implement the path-specific proof and run PathN build/test under the 240-second bound.

```bash
git add NarrativeDynamics/Core/FitnessABMPathN.lean \
        NarrativeDynamics/Tests/FitnessABMPathN.lean
git commit -m "feat(lean): prove finite path common mass"
```

**Review gate:** no fixed-size case analysis occurs in `path_block_common_mass`.

---

## Task 6: Generic block contraction and coordinate convergence

**Files:** modify FiniteConsensus core/test and PathN core/test.

### Generic FiniteConsensus additions

```text
coordRange_apply_le_of_commonColumn
coordRange_block_iterate_le
block_geometric_bound
block_contraction_tendsto
```

Target theorem:

```lean
theorem block_contraction_tendsto
    [Fintype ι] [Nonempty ι] [DecidableEq ι]
    (K : Kernel ι) (π : ι → Rat)
    (hK : AveragingKernel K)
    (hπ : StationaryWeights K π)
    (b : Nat) (hb : 0 < b)
    (δ : Rat) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hcommon : CommonColumnMass (K ^ b) δ)
    (x : ι → Rat) (i : ι) :
  Tendsto
    (fun k : Nat => (kernelTrajectory K x k i : Real))
    atTop
    (nhds (weightedMean π x : Real))
```

Any helper showing `K^b` is averaging must be derived from `hK` inside FiniteConsensus.

### Required proof route

1. For the common column `c`, subtract `δ` from that column in every row and divide the residual by `1-δ`; prove the residual is an averaging kernel.
2. The common term `δ*x(c)` cancels between outputs, giving `coordRange (applyKernel (K^b) x) ≤ (1-δ) * coordRange x`.
3. Use `kernelPow_apply` to convert the block matrix to `b` iterations.
4. Induct: `coordRange (kernelTrajectory K x (q*b)) ≤ (1-δ)^q * coordRange x`.
5. Cast to `Real` and use the v4.32 geometric-power limit theorem with `0 ≤ 1-δ < 1`.
6. For arbitrary `k ≥ q*b`, use single-step range nonincrease rather than a `Nat.div` formula.
7. Stationarity preserves `weightedMean`; normalized weights keep it between current extrema, yielding `|x_k i - weightedMean π x| ≤ coordRange x_k`; squeeze to zero.

### PathN final theorem

```lean
theorem trajectory_tendsto
    (n : Nat) (hn : 2 ≤ n)
    (x : Beliefs n)
    (hx : allBroadcast n x)
    (i : Fin n) :
  Tendsto
    (fun k : Nat => (trajectory n x k i : Real))
    atTop
    (nhds (mean n x : Real))
```

Instantiate generic consensus with `pathKernel n`, `stationaryWeight n`, `block n`, `delta n`, then rewrite actual trajectory via Task 3.

### TDD

- [ ] **RED:** add a `pairKernel` consumer for `block_contraction_tendsto`; separately add a PathN consumer for `trajectory_tendsto`. Run both tests and record missing-theorem RED.
- [ ] **GREEN:** prove generic convergence first, then the thin PathN theorem.
- [ ] Add exact required axiom reports:

```lean
#print axioms NarrativeDynamics.FiniteConsensus.coordRange_apply_le_of_commonColumn
#print axioms NarrativeDynamics.FiniteConsensus.block_contraction_tendsto
#print axioms NarrativeDynamics.FitnessABMPathN.propagate_independent_exposures
#print axioms NarrativeDynamics.FitnessABMPathN.propagate_eq_kernel
#print axioms NarrativeDynamics.FitnessABMPathN.allBroadcast_iterate
#print axioms NarrativeDynamics.FitnessABMPathN.path_stationary_weights
#print axioms NarrativeDynamics.FitnessABMPathN.mean_step
#print axioms NarrativeDynamics.FitnessABMPathN.path_block_common_mass
#print axioms NarrativeDynamics.FitnessABMPathN.trajectory_tendsto
```

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FiniteConsensus
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMPathN
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FiniteConsensus.lean
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPathN.lean
```

- [ ] Commit.

```bash
git add NarrativeDynamics/Core/FiniteConsensus.lean \
        NarrativeDynamics/Core/FitnessABMPathN.lean \
        NarrativeDynamics/Tests/FiniteConsensus.lean \
        NarrativeDynamics/Tests/FitnessABMPathN.lean
git commit -m "feat(lean): prove finite path consensus convergence"
```

**Stop-and-fix rule:** if focused proof exceeds 240 seconds, stop before compatibility/CI and factor whichever of finite extrema, matrix power, or analytic conversion is responsible.

---

## Task 7: Path4/Path5 compatibility and n=6 smoke

**Files:** primarily `NarrativeDynamics/Tests/FitnessABMPathN.lean`; only expose a small PathN helper if the test cannot state compatibility cleanly. Do not change Path4/Path5 production modules.

Import PathN, Path4, Path4Convergence, and Path5 using namespace aliases `PN`, `P4`, `P5` in the test.

### Compatibility theorems

```text
path4_adj_compat
path4_step_compat
path4_trajectory_compat
path4_mean_compat
path5_adj_compat
path5_step_compat
path5_trajectory_compat
path5_mean_compat
```

Use `fin_cases` only for these fixed-size compatibility proofs when necessary.

Prove that the generic theorem independently reproduces the old Path4 convergence statement; do not call `P4.trajectory_tendsto`:

```lean
example (x : P4.Beliefs) (hx : P4.allBroadcast x) (i : Fin 4) :
    Tendsto
      (fun k : Nat => (P4.trajectory x k i : Real))
      atTop
      (nhds (P4.mean x : Real)) := by
  have hxN : PN.allBroadcast 4 x := by
    simpa [PN.allBroadcast, P4.allBroadcast] using hx
  simpa only [path4_trajectory_compat, path4_mean_compat] using
    PN.trajectory_tendsto 4 (by decide) x hxN i
```

### n=6 smoke

```lean
def smoke6 : PN.Beliefs 6 := ![1, 7/8, 3/4, 5/8, 3/4, 1]

theorem smoke6_allBroadcast : PN.allBroadcast 6 smoke6 := by
  intro j
  fin_cases j <;> norm_num [smoke6, PN.allBroadcast]

example (i : Fin 6) :
    Tendsto
      (fun k : Nat => (PN.trajectory 6 smoke6 k i : Real))
      atTop
      (nhds (PN.mean 6 smoke6 : Real)) :=
  PN.trajectory_tendsto 6 (by decide) smoke6 smoke6_allBroadcast i
```

This is test-only; do not create `FitnessABMPath6.lean`.

### TDD/regression

- [ ] **RED:** add compatibility consumers before compatibility helpers; intended failures are unknown declarations/unclosed equalities.
- [ ] **GREEN:** prove compatibility and smoke, then run:

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPathN.lean
timeout --kill-after=10s 900s bash tools/check_fitness_abm_path4.sh
```

Expected: generic tests pass and the existing Path4 closed-form/convergence plus Path5 generated replay golden remain untouched and green.

- [ ] Commit.

```bash
git add NarrativeDynamics/Tests/FitnessABMPathN.lean
git commit -m "test(lean): certify PathN compatibility and smoke"
```

---

## Task 8: Dedicated bounded gate and exact-head review readiness

**Files:** create `tools/check_fitness_abm_pathn.sh`; modify `.github/workflows/proof.yml`.

Do not fold PathN into `check_fitness_abm_path4.sh`.

### Gate script

Create `tools/check_fitness_abm_pathn.sh` with `set -euo pipefail`, repo-root normalization, GNU `time`, two logs, and a cleanup trap.

Source audit:

```bash
python3 tools/audit_fitness_trust.py source \
  NarrativeDynamics/Core/FiniteConsensus.lean \
  NarrativeDynamics/Tests/FiniteConsensus.lean \
  NarrativeDynamics/Core/FitnessABMPathN.lean \
  NarrativeDynamics/Tests/FitnessABMPathN.lean
```

Explicit builds:

```bash
for module in \
    NarrativeDynamics.Core.FiniteConsensus \
    NarrativeDynamics.Core.FitnessABMPathN; do
  /usr/bin/time -f "$module elapsed=%e s peak_rss=%M KiB" \
    timeout --kill-after=10s 240s lake build "$module"
done
```

Tests/logs:

```bash
/usr/bin/time -f 'FiniteConsensus tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FiniteConsensus.lean \
  2>&1 | tee "$consensus_log"

/usr/bin/time -f 'FitnessABMPathN tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPathN.lean \
  2>&1 | tee "$pathn_log"
```

Mandatory trust reports:

```bash
python3 tools/audit_fitness_trust.py log "$consensus_log" \
  --require NarrativeDynamics.FiniteConsensus.coordRange_apply_le_of_commonColumn \
  --require NarrativeDynamics.FiniteConsensus.block_contraction_tendsto

python3 tools/audit_fitness_trust.py log "$pathn_log" \
  --require NarrativeDynamics.FitnessABMPathN.propagate_independent_exposures \
  --require NarrativeDynamics.FitnessABMPathN.propagate_eq_kernel \
  --require NarrativeDynamics.FitnessABMPathN.allBroadcast_iterate \
  --require NarrativeDynamics.FitnessABMPathN.path_stationary_weights \
  --require NarrativeDynamics.FitnessABMPathN.mean_step \
  --require NarrativeDynamics.FitnessABMPathN.path_block_common_mass \
  --require NarrativeDynamics.FitnessABMPathN.trajectory_tendsto
```

### Workflow integration

Immediately after existing `BB path-four convergence`:

```yaml
      - name: BB finite-path convergence
        timeout-minutes: 15
        run: bash tools/check_fitness_abm_pathn.sh
```

Do not alter event selection, exact-head checkout, existing Path4/Path5 gate, Path5 golden, or later regression steps.

### Final verification

- [ ] Hygiene:

```bash
bash -n tools/check_fitness_abm_pathn.sh
git diff --check
git status --short
```

- [ ] Dedicated gate and existing Path4/Path5 gate on the same head:

```bash
timeout --kill-after=10s 900s bash tools/check_fitness_abm_pathn.sh
timeout --kill-after=10s 900s bash tools/check_fitness_abm_path4.sh
```

- [ ] Commit CI integration.

```bash
git add tools/check_fitness_abm_pathn.sh .github/workflows/proof.yml
git commit -m "ci: gate generic finite path convergence"
```

- [ ] Push `feature/bb-pathn-convergence-v1`; open a draft PR against `proof/narrative-dynamics-v0`; record the exact head SHA.
- [ ] Require one exact-head PR proof run where Python tests, full Lean build, all prior BB gates, existing Path4/Path5 generated replay gate, new PathN gate, naming/attachment/replay/scope/distribution/general theorem/story/testimony tails all succeed.
- [ ] Require World Studio success on that same head.
- [ ] Record PathN build/test elapsed time and peak RSS, source audit, and mandatory axiom-audit results.
- [ ] Request code review; fix Critical/Important findings before marking ready.
- [ ] Update #83 with factual evidence and explicit scope boundaries.
- [ ] Stop at review/merge readiness. Do not merge or close #83 unless the user explicitly asks after exact-head review.

## Final acceptance checklist

- [ ] `beliefStep` is actual `NetworkPropagation.propagate`.
- [ ] Exposure independence is generic and does not require `allBroadcast`.
- [ ] `propagate_eq_kernel` holds for all `n ≥ 2` in the all-broadcast region.
- [ ] All-broadcast invariance is generic and convexity-based.
- [ ] Path degree sum and normalized stationary degree weights are proved.
- [ ] Actual trajectory preserves the degree-weighted mean.
- [ ] `(n-1)`-step common-column mass is at least `(1/4)^(n-1)`.
- [ ] Generic block range contraction is proved.
- [ ] Coordinatewise `Tendsto` to the stationary mean is proved for every `n ≥ 2`.
- [ ] Path4 adjacency/step/trajectory/mean compatibility passes.
- [ ] The generic theorem independently reproduces the Path4 convergence statement.
- [ ] Path5 adjacency/step/trajectory/mean compatibility passes.
- [ ] Existing Path5 runtime/model golden conformance remains green.
- [ ] Fixed `n=6` smoke instantiates the generic theorem with no Path6 production module.
- [ ] No arbitrary connected-graph/stochastic-matrix convergence theorem was added.
- [ ] No arbitrary receptivity/threshold generalization was added.
- [ ] Source audit and required theorem axiom reports pass.
- [ ] No resource limit is disabled or raised merely to hide proof structure.
- [ ] Exact-head proof and World Studio workflows pass.
