import NarrativeDynamics.Core.FitnessReplay

/-!
# Exact finite BB target-trace distribution

The finite carrier enumerates every legal ordered target trace while the exact
probability reuses the existing `orderedMass`, `applyBirth`, and
`continuationMass` kernel. No second normalization algorithm is introduced.
-/

namespace NarrativeDynamics.FitnessAttachment

open Internal
open scoped BigOperators

/-- A finite sequence of legal ordered target choices whose node type grows after
    each birth. The empty trace has one inhabitant. -/
def TargetTrace (n m : Nat) : Nat → Type
  | 0 => PUnit
  | steps + 1 => Targets n m × TargetTrace (n + 1) m steps

/-- Target traces have executable equality inherited recursively from target
    embeddings and products. -/
instance TargetTrace.instDecidableEq (n m steps : Nat) :
    DecidableEq (TargetTrace n m steps) := by
  induction steps generalizing n with
  | zero =>
      change DecidableEq PUnit
      infer_instance
  | succ steps ih =>
      change DecidableEq (Targets n m × TargetTrace (n + 1) m steps)
      letI : DecidableEq (TargetTrace (n + 1) m steps) := ih (n + 1)
      infer_instance

/-- Enumerate every target trace recursively from the existing executable target
    enumeration. No choice-based `Fintype.ofFinite` bridge is used. -/
instance TargetTrace.instFintype (n m steps : Nat) :
    Fintype (TargetTrace n m steps) := by
  induction steps generalizing n with
  | zero =>
      change Fintype PUnit
      infer_instance
  | succ steps ih =>
      change Fintype (Targets n m × TargetTrace (n + 1) m steps)
      letI : Fintype (TargetTrace (n + 1) m steps) := ih (n + 1)
      infer_instance

/-- Exact probability of one legal typed trace under a fixed positive newborn
    fitness schedule. Each factor is the existing BB ordered-target law on the
    actual successor state. -/
def traceProbability {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) :
    (schedule : List PosFitness) → TargetTrace n m schedule.length → Rat
  | [], _ => 1
  | eta :: rest, (T, tail) =>
      orderedMass s T *
        traceProbability (applyBirth s T hm eta) m hm
          (Nat.le_trans hb (Nat.le_succ n)) rest tail

/-- Every legal typed trace has strictly positive exact mass. -/
theorem traceProbability_pos {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (trace : TargetTrace n m schedule.length) :
    0 < traceProbability s m hm hb schedule trace := by
  induction schedule generalizing n with
  | nil =>
      cases trace
      norm_num [traceProbability]
  | cons eta rest ih =>
      rcases trace with ⟨T, tail⟩
      simp only [traceProbability]
      exact mul_pos (orderedMass_pos s T)
        (ih (s := applyBirth s T hm eta)
          (hb := Nat.le_trans hb (Nat.le_succ n)) tail)

/-- Trace masses are nonnegative as a direct corollary of strict support. -/
theorem traceProbability_nonneg {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (trace : TargetTrace n m schedule.length) :
    0 ≤ traceProbability s m hm hb schedule trace :=
  (traceProbability_pos s m hm hb schedule trace).le

/-- Summing the typed carrier is definitionally the same recursive finite law as
    the existing replay-side continuation mass. -/
theorem traceProbability_sum_continuationMass {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness) :
    (∑ trace : TargetTrace n m schedule.length,
      traceProbability s m hm hb schedule trace) =
      continuationMass s m hm hb schedule := by
  induction schedule generalizing n with
  | nil =>
      simp [TargetTrace, traceProbability, continuationMass]
  | cons eta rest ih =>
      change
        (∑ trace : Targets n m × TargetTrace (n + 1) m rest.length,
          orderedMass s trace.1 *
            traceProbability (applyBirth s trace.1 hm eta) m hm
              (Nat.le_trans hb (Nat.le_succ n)) rest trace.2) =
        ∑ T : Targets n m,
          orderedMass s T *
            continuationMass (applyBirth s T hm eta) m hm
              (Nat.le_trans hb (Nat.le_succ n)) rest
      rw [Fintype.sum_prod_type]
      apply Finset.sum_congr rfl
      intro T _
      change (∑ y, orderedMass s T *
        traceProbability (applyBirth s T hm eta) m hm
          (Nat.le_trans hb (Nat.le_succ n)) rest y) = _
      rw [← Finset.mul_sum, ih]

/-- The complete finite typed trace law is normalized exactly to one. -/
theorem traceProbability_sum_one {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness) :
    (∑ trace : TargetTrace n m schedule.length,
      traceProbability s m hm hb schedule trace) = 1 := by
  rw [traceProbability_sum_continuationMass, continuationMass_one]

/-- No individual legal trace can carry more than the complete normalized mass. -/
theorem traceProbability_le_one {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (trace : TargetTrace n m schedule.length) :
    traceProbability s m hm hb schedule trace ≤ 1 := by
  calc
    traceProbability s m hm hb schedule trace ≤
        ∑ t : TargetTrace n m schedule.length,
          traceProbability s m hm hb schedule t := by
      exact Finset.single_le_sum
        (fun t _ => traceProbability_nonneg s m hm hb schedule t)
        (Finset.mem_univ trace)
    _ = 1 := traceProbability_sum_one s m hm hb schedule

/-- Evaluate one typed target trace through the authoritative BB birth transition. -/
def traceFinal {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) :
    (schedule : List PosFitness) → TargetTrace n m schedule.length → RunState
  | [], _ => ⟨n, s⟩
  | eta :: rest, (T, tail) =>
      traceFinal (applyBirth s T hm eta) m hm
        (Nat.le_trans hb (Nat.le_succ n)) rest tail

/-- The existential run-state index follows the one-vertex-per-birth carrier growth. -/
theorem traceFinal_nodeCount {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (trace : TargetTrace n m schedule.length) :
    (traceFinal s m hm hb schedule trace).nodeCount = n + schedule.length := by
  induction schedule generalizing n with
  | nil =>
      cases trace
      rfl
  | cons eta rest ih =>
      rcases trace with ⟨T, tail⟩
      change
        (traceFinal (applyBirth s T hm eta) m hm
          (Nat.le_trans hb (Nat.le_succ n)) rest tail).nodeCount =
        n + (rest.length + 1)
      rw [ih (s := applyBirth s T hm eta)
        (hb := Nat.le_trans hb (Nat.le_succ n)) (trace := tail)]
      omega

/-- Actual final node count is derived from the real state carrier, not stored separately. -/
theorem traceFinal_nodes {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (trace : TargetTrace n m schedule.length) :
    actualNodeCount (traceFinal s m hm hb schedule trace).state.snapshot =
      actualNodeCount s.snapshot + schedule.length := by
  simpa [actualNodeCount] using traceFinal_nodeCount s m hm hb schedule trace

/-- Every birth contributes exactly m actual edges along the evaluated trace. -/
theorem traceFinal_edges {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (trace : TargetTrace n m schedule.length) :
    actualEdgeCount (traceFinal s m hm hb schedule trace).state.snapshot =
      actualEdgeCount s.snapshot + m * schedule.length := by
  induction schedule generalizing n with
  | nil =>
      cases trace
      simp [traceFinal]
  | cons eta rest ih =>
      rcases trace with ⟨T, tail⟩
      change
        actualEdgeCount
            (traceFinal (applyBirth s T hm eta) m hm
              (Nat.le_trans hb (Nat.le_succ n)) rest tail).state.snapshot =
          actualEdgeCount s.snapshot + m * (rest.length + 1)
      rw [ih (s := applyBirth s T hm eta)
        (hb := Nat.le_trans hb (Nat.le_succ n)) (trace := tail), birth_edges]
      rw [Nat.mul_add, Nat.mul_one]
      omega

/-- Final stable-ID fitness values are exactly the seed values followed by the schedule. -/
theorem traceFinal_fitness {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (trace : TargetTrace n m schedule.length) :
    List.ofFn (traceFinal s m hm hb schedule trace).state.snapshot.fitness =
      List.ofFn s.snapshot.fitness ++ schedule.map (fun eta => eta.val) := by
  induction schedule generalizing n with
  | nil =>
      cases trace
      simp [traceFinal]
  | cons eta rest ih =>
      rcases trace with ⟨T, tail⟩
      change
        List.ofFn
            (traceFinal (applyBirth s T hm eta) m hm
              (Nat.le_trans hb (Nat.le_succ n)) rest tail).state.snapshot.fitness =
          List.ofFn s.snapshot.fitness ++ eta.val :: rest.map (fun x => x.val)
      rw [ih (s := applyBirth s T hm eta)
        (hb := Nat.le_trans hb (Nat.le_succ n)) (trace := tail), birth_fitness_list]
      simp only [List.append_assoc, List.singleton_append]

/-- The evaluated final state is valid because every transition is `applyBirth`. -/
theorem traceFinal_valid {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (trace : TargetTrace n m schedule.length) :
    (traceFinal s m hm hb schedule trace).state.snapshot.Valid :=
  (traceFinal s m hm hb schedule trace).state.valid

/-- Exact finite probability of a decidable property of the authoritative final state.
    Distinct ordered traces are summed separately even when they reach the same graph. -/
def eventProbability {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (event : RunState → Prop) [DecidablePred event] : Rat :=
  ∑ trace : TargetTrace n m schedule.length,
    if event (traceFinal s m hm hb schedule trace)
    then traceProbability s m hm hb schedule trace
    else 0

/-- The certain event has the complete normalized mass. -/
theorem eventProbability_true {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness) :
    eventProbability s m hm hb schedule (fun _ => True) = 1 := by
  simp [eventProbability, traceProbability_sum_one]

/-- The impossible event has zero mass. -/
theorem eventProbability_false {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness) :
    eventProbability s m hm hb schedule (fun _ => False) = 0 := by
  simp [eventProbability]

/-- Event mass is nonnegative because it is a finite sum of retained trace masses. -/
theorem eventProbability_nonneg {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (event : RunState → Prop) [DecidablePred event] :
    0 ≤ eventProbability s m hm hb schedule event := by
  unfold eventProbability
  apply Finset.sum_nonneg
  intro trace _
  by_cases h : event (traceFinal s m hm hb schedule trace)
  · simp [h, traceProbability_nonneg s m hm hb schedule trace]
  · simp [h]

/-- Filtering the normalized trace law cannot increase total mass above one. -/
theorem eventProbability_le_one {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (event : RunState → Prop) [DecidablePred event] :
    eventProbability s m hm hb schedule event ≤ 1 := by
  calc
    eventProbability s m hm hb schedule event ≤
        ∑ trace : TargetTrace n m schedule.length,
          traceProbability s m hm hb schedule trace := by
      unfold eventProbability
      apply Finset.sum_le_sum
      intro trace _
      by_cases h : event (traceFinal s m hm hb schedule trace)
      · simp [h]
      · simp [h, traceProbability_nonneg s m hm hb schedule trace]
    _ = 1 := traceProbability_sum_one s m hm hb schedule

/-- A decidable event and its complement partition the complete finite trace mass. -/
theorem eventProbability_compl {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (event : RunState → Prop) [DecidablePred event] :
    eventProbability s m hm hb schedule (fun out => ¬ event out) =
      1 - eventProbability s m hm hb schedule event := by
  have hsum :
      eventProbability s m hm hb schedule event +
          eventProbability s m hm hb schedule (fun out => ¬ event out) = 1 := by
    rw [← traceProbability_sum_one s m hm hb schedule]
    unfold eventProbability
    rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro trace _
    by_cases h : event (traceFinal s m hm hb schedule trace)
    · simp [h]
    · simp [h]
  linarith

/-- Event inclusion gives monotonicity of exact finite event probability. -/
theorem eventProbability_mono {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (event₁ event₂ : RunState → Prop) [DecidablePred event₁] [DecidablePred event₂]
    (hsub : ∀ out, event₁ out → event₂ out) :
    eventProbability s m hm hb schedule event₁ ≤
      eventProbability s m hm hb schedule event₂ := by
  unfold eventProbability
  apply Finset.sum_le_sum
  intro trace _
  let out := traceFinal s m hm hb schedule trace
  by_cases h₁ : event₁ out
  · have h₂ : event₂ out := hsub out h₁
    simp [out, h₁, h₂]
  · by_cases h₂ : event₂ out
    · simp [out, h₁, h₂, traceProbability_nonneg s m hm hb schedule trace]
    · simp [out, h₁, h₂]

/-- Exact rational expectation of an observable of the authoritative final state. -/
def expectation {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (observable : RunState → Rat) : Rat :=
  ∑ trace : TargetTrace n m schedule.length,
    traceProbability s m hm hb schedule trace *
      observable (traceFinal s m hm hb schedule trace)

/-- A constant observable has that same expectation. -/
theorem expectation_const {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness) (c : Rat) :
    expectation s m hm hb schedule (fun _ => c) = c := by
  unfold expectation
  rw [← Finset.sum_mul, traceProbability_sum_one]
  simp

/-- Exact expectation is additive. -/
theorem expectation_add {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (f g : RunState → Rat) :
    expectation s m hm hb schedule (fun out => f out + g out) =
      expectation s m hm hb schedule f + expectation s m hm hb schedule g := by
  unfold expectation
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro trace _
  ring

/-- Exact expectation commutes with rational scalar multiplication. -/
theorem expectation_smul {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (c : Rat) (f : RunState → Rat) :
    expectation s m hm hb schedule (fun out => c * f out) =
      c * expectation s m hm hb schedule f := by
  unfold expectation
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro trace _
  ring

/-- The expectation of an event indicator is exactly its event probability. -/
theorem expectation_indicator {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (event : RunState → Prop) [DecidablePred event] :
    expectation s m hm hb schedule (fun out => if event out then 1 else 0) =
      eventProbability s m hm hb schedule event := by
  unfold expectation eventProbability
  apply Finset.sum_congr rfl
  intro trace _
  by_cases h : event (traceFinal s m hm hb schedule trace)
  · simp [h]
  · simp [h]

end NarrativeDynamics.FitnessAttachment
