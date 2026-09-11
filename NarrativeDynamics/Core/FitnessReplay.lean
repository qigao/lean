import NarrativeDynamics.Core.FitnessValidation

/-!
# Finite checked BB replay

Replay consumes supplied ordered births; it does not draw random numbers.
Each step uses the actual successor graph and its current degree weights.
Failure returns only its first zero-based position and cause, never a prefix.
-/

namespace NarrativeDynamics.FitnessAttachment

open Internal
open scoped BigOperators

/-- The size indexes the actual state, rather than an independent graph counter. -/
structure RunState where
  nodeCount : Nat
  state : State nodeCount

structure ReplayResult where
  final : RunState
  probability : Rat

inductive ReplayError where
  | seed (cause : Error)
  | initialM
  | atBirth (index : Nat) (cause : Error)
  deriving DecidableEq, Repr

namespace Internal

/-- Traverse left to right; only a fully successful suffix yields a result. -/
def runBirths (m index : Nat) (s : RunState) (births : List RawBirth) :
    Except ReplayError ReplayResult :=
  match births with
  | [] => .ok ⟨s, 1⟩
  | raw :: rest =>
    match step s.state m raw with
    | .error e => .error (.atBirth index e)
    | .ok next =>
      match runBirths m (index + 1) ⟨s.nodeCount + 1, next.1⟩ rest with
      | .error e => .error e
      | .ok result => .ok ⟨result.final, next.2 * result.probability⟩
termination_by structural births

/-- A stable-ID fitness list is extended by precisely the supplied new value. -/
private theorem birth_fitness_list {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) :
    List.ofFn (applyBirth s T hm eta).snapshot.fitness =
      List.ofFn s.snapshot.fitness ++ [eta.val] := by
  rw [List.ofFn_succ']
  simp [applyBirth, birthSnapshot, List.concat_eq_append]

/-- Counts, stored values and mass follow from each actual checked step. -/
theorem runBirths_properties (m index : Nat) (s : RunState) (bs : List RawBirth)
    (out : ReplayResult) (h : runBirths m index s bs = .ok out) :
    out.final.nodeCount = s.nodeCount + bs.length ∧
    actualEdgeCount out.final.state.snapshot =
      actualEdgeCount s.state.snapshot + m * bs.length ∧
    List.ofFn out.final.state.snapshot.fitness =
      List.ofFn s.state.snapshot.fitness ++ bs.map RawBirth.fitness ∧
    0 < out.probability := by
  induction bs generalizing index s out with
  | nil =>
    simp only [runBirths, Except.ok.injEq] at h
    cases h
    simp
  | cons raw bs ih =>
    cases ht : step s.state m raw with
    | error e => simp [runBirths, ht] at h
    | ok next =>
      cases hr : runBirths m (index + 1) ⟨s.nodeCount + 1, next.1⟩ bs with
      | error e => simp [runBirths, ht, hr] at h
      | ok tail =>
        have hout : (⟨tail.final, next.2 * tail.probability⟩ : ReplayResult) = out := by
          simpa only [runBirths, ht, hr, Except.ok.injEq] using h
        cases hout
        have props := ih (index := index + 1)
          (s := ⟨s.nodeCount + 1, next.1⟩) (out := tail) hr
        obtain ⟨v, hv, rfl⟩ := (step_spec s.state raw next).mp ht
        rcases props with ⟨hN, hE, hF, hP⟩
        have hvalue := (validateBirth_values s.state raw v hv).1
        refine ⟨?_, ?_, ?_, ?_⟩
        · change tail.final.nodeCount = s.nodeCount + (bs.length + 1)
          change tail.final.nodeCount = (s.nodeCount + 1) + bs.length at hN
          omega
        · change actualEdgeCount tail.final.state.snapshot =
            actualEdgeCount s.state.snapshot + m * (bs.length + 1)
          change actualEdgeCount tail.final.state.snapshot =
            actualEdgeCount (applyBirth s.state v.targets v.positive v.fitness).snapshot +
              m * bs.length at hE
          rw [birth_edges] at hE
          rw [Nat.mul_add, Nat.mul_one]
          omega
        · change List.ofFn tail.final.state.snapshot.fitness =
            List.ofFn s.state.snapshot.fitness ++ (raw.fitness :: bs.map RawBirth.fitness)
          change List.ofFn tail.final.state.snapshot.fitness =
            List.ofFn (applyBirth s.state v.targets v.positive v.fitness).snapshot.fitness ++
              bs.map RawBirth.fitness at hF
          rw [hF, birth_fitness_list, hvalue]
          simp only [List.append_assoc, List.cons_append, List.nil_append]
        · exact mul_pos (orderedMass_pos s.state v.targets) hP

/-- Parsing preserves the full original array, in stable numeric-ID order. -/
private theorem seed_fitness_list (seed : RawSeed) (s : State seed.nodeCount)
    (h : parseSeed seed = .ok s) : List.ofFn s.snapshot.fitness = seed.fitness.toList := by
  obtain ⟨hs, hstate⟩ := parseSeed_snapshot seed s h
  rw [hstate]
  apply List.ext_getElem
  · simpa using hs.symm
  · intro i hi hj
    simp [seedSnapshot]

end Internal

/-- Validate the initial state and fixed m even when there are no births. -/
def replay (seed : RawSeed) (m : Nat) (births : List RawBirth) :
    Except ReplayError ReplayResult :=
  match parseSeed seed with
  | .error e => .error (.seed e)
  | .ok s =>
    if 0 < m ∧ m ≤ seed.nodeCount then runBirths m 0 ⟨seed.nodeCount, s⟩ births
    else .error .initialM

private theorem replay_success (seed : RawSeed) (m : Nat) (bs : List RawBirth)
    (out : ReplayResult) (h : replay seed m bs = .ok out) :
    ∃ s : State seed.nodeCount, parseSeed seed = .ok s ∧
      (0 < m ∧ m ≤ seed.nodeCount) ∧ runBirths m 0 ⟨seed.nodeCount, s⟩ bs = .ok out := by
  unfold replay at h
  cases hs : parseSeed seed with
  | error e => simp [hs] at h
  | ok s =>
    rw [hs] at h
    split at h
    · rename_i hm
      exact ⟨s, hs, hm, h⟩
    · contradiction

/-- An empty accepted trace returns the parsed input and multiplicative identity. -/
theorem replay_empty (seed : RawSeed) (m : Nat) (s : State seed.nodeCount)
    (hs : parseSeed seed = .ok s) (hm : 0 < m ∧ m ≤ seed.nodeCount) :
    replay seed m [] = .ok ⟨⟨seed.nodeCount, s⟩, 1⟩ := by
  simp [replay, hs, hm, runBirths]

/-- Expose the exact checked-step recurrence, including both error boundaries. -/
theorem replay_step (m index : Nat) (s : RunState) (raw : RawBirth) (bs : List RawBirth) :
    runBirths m index s (raw :: bs) =
      match step s.state m raw with
      | .error e => .error (.atBirth index e)
      | .ok next =>
        match runBirths m (index + 1) ⟨s.nodeCount + 1, next.1⟩ bs with
        | .error e => .error e
        | .ok result => .ok ⟨result.final, next.2 * result.probability⟩ := rfl

/-- Successful replay contains a state proved valid by the existing constructors. -/
theorem replay_valid (seed : RawSeed) (m : Nat) (bs : List RawBirth) (out : ReplayResult)
    (_h : replay seed m bs = .ok out) : out.final.state.snapshot.Valid := out.final.state.valid

/-- One actual new vertex per successful birth. -/
theorem replay_nodes (seed : RawSeed) (m : Nat) (bs : List RawBirth) (out : ReplayResult)
    (h : replay seed m bs = .ok out) :
    actualNodeCount out.final.state.snapshot = seed.nodeCount + bs.length := by
  obtain ⟨s, _, _, hr⟩ := replay_success seed m bs out h
  simpa [actualNodeCount] using (runBirths_properties m 0 ⟨seed.nodeCount, s⟩ bs out hr).1

/-- Actual edge counts, not a parallel recurrence stored in ReplayResult. -/
theorem replay_edges (seed : RawSeed) (m : Nat) (bs : List RawBirth) (out : ReplayResult)
    (s : State seed.nodeCount) (hs : parseSeed seed = .ok s)
    (h : replay seed m bs = .ok out) :
    actualEdgeCount out.final.state.snapshot = actualEdgeCount s.snapshot + m * bs.length := by
  obtain ⟨t, ht, _, hr⟩ := replay_success seed m bs out h
  have he : t = s := Except.ok.inj (ht.symm.trans hs)
  subst t
  exact (runBirths_properties m 0 ⟨seed.nodeCount, s⟩ bs out hr).2.1

/-- Every original and newborn fitness value is retained at its assigned numeric ID. -/
theorem replay_fitness_prefix (seed : RawSeed) (m : Nat) (bs : List RawBirth)
    (out : ReplayResult) (h : replay seed m bs = .ok out) :
    List.ofFn out.final.state.snapshot.fitness = seed.fitness.toList ++ bs.map RawBirth.fitness := by
  obtain ⟨s, hs, _, hr⟩ := replay_success seed m bs out h
  have hf := (runBirths_properties m 0 ⟨seed.nodeCount, s⟩ bs out hr).2.2.1
  rw [seed_fitness_list seed s hs] at hf
  exact hf

/-- Every accepted finite trace has strictly positive conditional-product mass. -/
theorem replay_probability_pos (seed : RawSeed) (m : Nat) (bs : List RawBirth)
    (out : ReplayResult) (h : replay seed m bs = .ok out) : 0 < out.probability := by
  obtain ⟨s, _, _, hr⟩ := replay_success seed m bs out h
  exact (runBirths_properties m 0 ⟨seed.nodeCount, s⟩ bs out hr).2.2.2

/-- Total mass of all legal ordered choices for a fixed positive fitness schedule.
This exact finite sum is not a scalable sampler or a source of randomness. -/
def continuationMass {n : Nat} (s : State n) (m : Nat) (hm : 0 < m) (hb : m ≤ n)
    (schedule : List PosFitness) : Rat :=
  match schedule with
  | [] => 1
  | eta :: rest => ∑ T : Targets n m,
      orderedMass s T * continuationMass (applyBirth s T hm eta) m hm
        (Nat.le_trans hb (Nat.le_succ n)) rest
termination_by structural schedule

/-- Normalize at every updated graph, then compose the complete conditional laws. -/
theorem continuationMass_one {n : Nat} (s : State n) (m : Nat) (hm : 0 < m) (hb : m ≤ n)
    (schedule : List PosFitness) : continuationMass s m hm hb schedule = 1 := by
  induction schedule generalizing n with
  | nil => rfl
  | cons eta rest ih =>
    simp only [continuationMass, ih, mul_one]
    exact orderedMass_sum_one s hb

end NarrativeDynamics.FitnessAttachment
