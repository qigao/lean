import NarrativeDynamics.Core.FitnessABMPathN

/-!
# Parameterized finite-path BB belief dynamics

This module keeps the executable step on the existing
`NetworkPropagation.propagate` path while making receptivity and threshold
explicit exact-rational parameters.
-/

namespace NarrativeDynamics.FitnessABMPathNParameters

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FiniteConsensus
open scoped BigOperators

abbrev Beliefs (n : Nat) := FitnessABMPathN.Beliefs n

structure ResponseParameters where
  receptivity : Rat
  threshold : Rat
  deriving Repr, DecidableEq

namespace ResponseParameters

def Valid (p : ResponseParameters) : Prop :=
  0 ≤ p.receptivity ∧ p.receptivity ≤ 1 ∧
    0 ≤ p.threshold ∧ p.threshold ≤ 1

end ResponseParameters

def population (params : ResponseParameters)
    (n : Nat) (x : Beliefs n) (e : Fin n → Nat) : Population n :=
  ⟨fun _ => ⟨params.receptivity, params.threshold⟩,
    fun i => ⟨x i, e i⟩⟩

def project (n : Nat) (p : Population n) : Beliefs n :=
  fun i => (p.agents i).belief

def allBroadcast (params : ResponseParameters)
    (n : Nat) (x : Beliefs n) : Prop :=
  ∀ i, params.threshold ≤ x i ∧ x i ≤ 1

def beliefStep (params : ResponseParameters)
    (n : Nat) (x : Beliefs n) : Beliefs n :=
  project n
    (propagate (FitnessABMPathN.pathAdj n)
      (population params n x (fun _ => 0)))

def half : ResponseParameters :=
  ⟨1/2, 1/2⟩

/-- The canonical half-response parameterization is exactly the existing fixed
finite-path population, including supplied exposure counters. -/
theorem population_half (n : Nat) (x : Beliefs n) (e : Fin n → Nat) :
    population half n x e = FitnessABMPathN.population n x e := by
  rfl

/-- The canonical half-response executable step is exactly the existing fixed
finite-path step. No second propagation implementation is introduced. -/
theorem beliefStep_half (n : Nat) (x : Beliefs n) :
    beliefStep half n x = FitnessABMPathN.beliefStep n x := by
  rfl

/-- The canonical half threshold has the same inclusive all-broadcast region
as the existing fixed finite-path model. -/
theorem allBroadcast_half (n : Nat) (x : Beliefs n) :
    allBroadcast half n x ↔ FitnessABMPathN.allBroadcast n x := by
  rfl

/-- Broadcasting decisions, received beliefs, and therefore projected updated
beliefs do not depend on the stored exposure counters. -/
theorem propagate_independent_exposures
    (params : ResponseParameters) (n : Nat) (x : Beliefs n)
    (e : Fin n → Nat) :
    project n
        (propagate (FitnessABMPathN.pathAdj n) (population params n x e)) =
      beliefStep params n x := by
  change project n
      (propagate (FitnessABMPathN.pathAdj n) (population params n x e)) =
    project n
      (propagate (FitnessABMPathN.pathAdj n)
        (population params n x (fun _ => 0)))
  funext i
  have hin :
      incoming (FitnessABMPathN.pathAdj n) (population params n x e) i =
        incoming (FitnessABMPathN.pathAdj n)
          (population params n x (fun _ => 0)) i := by
    rfl
  change
    (nextAgent (FitnessABMPathN.pathAdj n) (population params n x e) i).belief =
      (nextAgent (FitnessABMPathN.pathAdj n)
        (population params n x (fun _ => 0)) i).belief
  simp only [nextAgent, hin]
  split_ifs <;> rfl

/-- Proof-only lazy averaging kernel for arbitrary response parameters.
The self mass is `1 - α`; broadcasting-neighbor mass is `α / degree`. -/
def pathKernel (params : ResponseParameters) (n : Nat) : Kernel (Fin n) :=
  fun i j =>
    (if i = j then 1 - params.receptivity else 0) +
      (if j ∈ FitnessABMPathN.neighbors n i then
        params.receptivity / (FitnessABMPathN.degree n i : Rat)
      else 0)

/-- The parameter kernel at `α = 1/2` is exactly the existing fixed PathN
kernel. -/
theorem pathKernel_half (n : Nat) :
    pathKernel half n = FitnessABMPathN.pathKernel n := by
  funext i j
  by_cases hij : i = j
  · subst j
    have hself : i ∉ FitnessABMPathN.neighbors n i := by
      intro h
      exact FitnessABMPathN.pathAdj_self i
        ((FitnessABMPathN.mem_neighbors_iff i i).mp h)
    simp [pathKernel, FitnessABMPathN.pathKernel, half, hself]
    norm_num
  · by_cases hmem : j ∈ FitnessABMPathN.neighbors n i
    · simp [pathKernel, FitnessABMPathN.pathKernel, half, hij, hmem]
      rw [div_eq_mul_inv]
      ring
    · simp [pathKernel, FitnessABMPathN.pathKernel, half, hij, hmem]

/-- Valid response parameters make every path-kernel entry nonnegative. -/
theorem pathKernel_nonneg
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (i j : Fin n) :
    0 ≤ pathKernel params n i j := by
  classical
  have hdPos := FitnessABMPathN.degree_pos n hn i
  have hdLe := FitnessABMPathN.degree_le_two n i
  have hd : FitnessABMPathN.degree n i = 1 ∨
      FitnessABMPathN.degree n i = 2 := by
    omega
  rcases hd with hd | hd <;>
    by_cases hij : i = j <;>
    by_cases hmem : j ∈ FitnessABMPathN.neighbors n i <;>
    simp [pathKernel, hij, hmem, hd] <;>
    linarith [hvalid.1, hvalid.2.1]

private theorem self_mass_sum
    (params : ResponseParameters) (n : Nat) (i : Fin n) :
    (∑ j : Fin n, if i = j then 1 - params.receptivity else 0) =
      1 - params.receptivity := by
  classical
  simp

private theorem neighbor_mass_sum
    (params : ResponseParameters) (n : Nat) (hn : 2 ≤ n) (i : Fin n) :
    (∑ j : Fin n,
      if j ∈ FitnessABMPathN.neighbors n i then
        params.receptivity / (FitnessABMPathN.degree n i : Rat)
      else 0) = params.receptivity := by
  classical
  let c : Rat :=
    params.receptivity / (FitnessABMPathN.degree n i : Rat)
  have hsum :
      (∑ j : Fin n,
        if j ∈ FitnessABMPathN.neighbors n i then c else 0) =
        ∑ j ∈ FitnessABMPathN.neighbors n i, c := by
    simpa using
      (Finset.sum_ite_mem (Finset.univ : Finset (Fin n))
        (FitnessABMPathN.neighbors n i) (fun _ => c))
  rw [hsum]
  have hdPos := FitnessABMPathN.degree_pos n hn i
  have hdLe := FitnessABMPathN.degree_le_two n i
  have hd : FitnessABMPathN.degree n i = 1 ∨
      FitnessABMPathN.degree n i = 2 := by
    omega
  rcases hd with hd | hd
  · have hcard : (FitnessABMPathN.neighbors n i).card = 1 := by
      simpa [FitnessABMPathN.degree] using hd
    simp [hcard, c, hd]
  · have hcard : (FitnessABMPathN.neighbors n i).card = 2 := by
      simpa [FitnessABMPathN.degree] using hd
    simp [hcard, c, hd] <;> ring

/-- Every valid parameterized path-kernel row has exact mass one. -/
theorem pathKernel_rowsum
    (params : ResponseParameters) (_hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (i : Fin n) :
    ∑ j, pathKernel params n i j = 1 := by
  classical
  rw [show (∑ j : Fin n, pathKernel params n i j) =
      (∑ j : Fin n, if i = j then 1 - params.receptivity else 0) +
      (∑ j : Fin n,
        if j ∈ FitnessABMPathN.neighbors n i then
          params.receptivity / (FitnessABMPathN.degree n i : Rat)
        else 0) by
    simp only [pathKernel, Finset.sum_add_distrib]]
  rw [self_mass_sum params n i, neighbor_mass_sum params n hn i]
  ring

private theorem pathKernel_averaging
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) :
    AveragingKernel (pathKernel params n) := by
  exact ⟨pathKernel_nonneg params hvalid n hn,
    pathKernel_rowsum params hvalid n hn⟩

private theorem incoming_eq_neighbors
    (params : ResponseParameters) (n : Nat) (x : Beliefs n)
    (e : Fin n → Nat) (hx : allBroadcast params n x) (i : Fin n) :
    incoming (FitnessABMPathN.pathAdj n) (population params n x e) i =
      FitnessABMPathN.neighbors n i := by
  ext j
  have hb : broadcasting ((population params n x e).profiles j)
      ((population params n x e).agents j) = true := by
    change decide (params.threshold ≤ x j) = true
    exact decide_eq_true (hx j).1
  simp only [incoming, Finset.mem_filter, Finset.mem_univ, true_and, hb,
    and_true, FitnessABMPathN.mem_neighbors_iff]

private theorem applyKernel_pathKernel
    (params : ResponseParameters) (n : Nat) (x : Beliefs n) (i : Fin n) :
    applyKernel (pathKernel params n) x i =
      (1 - params.receptivity) * x i +
        (params.receptivity / (FitnessABMPathN.degree n i : Rat)) *
          (∑ j ∈ FitnessABMPathN.neighbors n i, x j) := by
  classical
  simp only [applyKernel, Matrix.mulVec_apply, dotProduct]
  change
    (∑ j : Fin n, pathKernel params n i j * x j) =
      (1 - params.receptivity) * x i +
        (params.receptivity / (FitnessABMPathN.degree n i : Rat)) *
          (∑ j ∈ FitnessABMPathN.neighbors n i, x j)
  simp only [pathKernel]
  rw [show
      (∑ j : Fin n,
        ((if i = j then 1 - params.receptivity else 0) +
          if j ∈ FitnessABMPathN.neighbors n i then
            params.receptivity / (FitnessABMPathN.degree n i : Rat)
          else 0) * x j) =
        (∑ j : Fin n,
          (if i = j then 1 - params.receptivity else 0) * x j) +
        (∑ j : Fin n,
          (if j ∈ FitnessABMPathN.neighbors n i then
            params.receptivity / (FitnessABMPathN.degree n i : Rat)
          else 0) * x j) by
    rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro j _
    ring]
  have hself :
      (∑ j : Fin n,
        (if i = j then 1 - params.receptivity else 0) * x j) =
        (1 - params.receptivity) * x i := by
    simp
  rw [hself]
  let c : Rat :=
    params.receptivity / (FitnessABMPathN.degree n i : Rat)
  have hneighbor :
      (∑ j : Fin n,
        (if j ∈ FitnessABMPathN.neighbors n i then c else 0) * x j) =
        c * (∑ j ∈ FitnessABMPathN.neighbors n i, x j) := by
    calc
      (∑ j : Fin n,
        (if j ∈ FitnessABMPathN.neighbors n i then c else 0) * x j) =
          ∑ j ∈ FitnessABMPathN.neighbors n i, c * x j := by
        simpa using
          (Finset.sum_ite_mem (Finset.univ : Finset (Fin n))
            (FitnessABMPathN.neighbors n i) (fun j => c * x j))
      _ = c * (∑ j ∈ FitnessABMPathN.neighbors n i, x j) := by
        rw [Finset.mul_sum]
  change
    (1 - params.receptivity) * x i +
      (∑ j : Fin n,
        (if j ∈ FitnessABMPathN.neighbors n i then c else 0) * x j) =
      (1 - params.receptivity) * x i +
        c * (∑ j ∈ FitnessABMPathN.neighbors n i, x j)
  rw [hneighbor]

/-- Inside the inclusive all-broadcast region, the real executable propagation
step agrees with the proof-only parameterized kernel. -/
theorem propagate_eq_kernel
    (params : ResponseParameters) (_hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : allBroadcast params n x) :
    beliefStep params n x = applyKernel (pathKernel params n) x := by
  funext i
  have hd : FitnessABMPathN.degree n i ≠ 0 :=
    Nat.ne_of_gt (FitnessABMPathN.degree_pos n hn i)
  have hcard : (FitnessABMPathN.neighbors n i).card ≠ 0 := by
    simpa [FitnessABMPathN.degree] using hd
  have hdq : (FitnessABMPathN.degree n i : Rat) ≠ 0 := by
    exact_mod_cast hd
  change
    (nextAgent (FitnessABMPathN.pathAdj n)
      (population params n x (fun _ => 0)) i).belief =
      applyKernel (pathKernel params n) x i
  rw [applyKernel_pathKernel params n x i]
  simp only [nextAgent,
    incoming_eq_neighbors params n x (fun _ => 0) hx i, hcard, if_false]
  change
    (1 - params.receptivity) * x i +
      params.receptivity *
        ((∑ j ∈ FitnessABMPathN.neighbors n i, x j) /
          (FitnessABMPathN.degree n i : Rat)) =
      (1 - params.receptivity) * x i +
        (params.receptivity / (FitnessABMPathN.degree n i : Rat)) *
          (∑ j ∈ FitnessABMPathN.neighbors n i, x j)
  field_simp [hdq] <;> ring

private theorem kernel_preserves_allBroadcast
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : allBroadcast params n x) :
    allBroadcast params n (applyKernel (pathKernel params n) x) := by
  let i0 : Fin n := ⟨0, by omega⟩
  letI : Nonempty (Fin n) := ⟨i0⟩
  intro i
  have hbetween :=
    applyKernel_between (pathKernel_averaging params hvalid n hn) x i
  have hmin : params.threshold ≤ coordMin x := by
    unfold coordMin
    apply Finset.le_inf'
    intro j _
    exact (hx j).1
  have hmax : coordMax x ≤ (1 : Rat) := by
    unfold coordMax
    apply Finset.sup'_le
    intro j _
    exact (hx j).2
  exact ⟨hmin.trans hbetween.1, hbetween.2.trans hmax⟩

/-- The inclusive all-broadcast region is invariant under one parameterized
executable step. -/
theorem allBroadcast_step
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : allBroadcast params n x) :
    allBroadcast params n (beliefStep params n x) := by
  rw [propagate_eq_kernel params hvalid n hn x hx]
  exact kernel_preserves_allBroadcast params hvalid n hn x hx

/-- The inclusive all-broadcast region is invariant under every finite iterate
of the parameterized executable step. -/
theorem allBroadcast_iterate
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : allBroadcast params n x) (k : Nat) :
    allBroadcast params n ((beliefStep params n)^[k] x) := by
  induction k with
  | zero => simpa using hx
  | succ k ih =>
      simpa [Function.iterate_succ_apply'] using
        allBroadcast_step params hvalid n hn
          ((beliefStep params n)^[k] x) ih

end NarrativeDynamics.FitnessABMPathNParameters
