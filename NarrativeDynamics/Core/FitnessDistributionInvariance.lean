import NarrativeDynamics.Core.FitnessDistribution

/-!
# BB finite-distribution invariances

A single positive factor may scale every stored initial fitness and every
scheduled newborn fitness without changing the exact target-trace law. The
final graph is unchanged while stored fitness values are scaled.
-/

namespace NarrativeDynamics.FitnessAttachment

open Internal
open scoped BigOperators

/-- Scale one positive fitness by one common positive factor. -/
def scalePosFitness (c eta : PosFitness) : PosFitness :=
  ⟨c.val * eta.val, mul_pos c.property eta.property⟩

/-- Apply one common positive factor to the complete future fitness schedule. -/
def scaleSchedule (c : PosFitness) (schedule : List PosFitness) : List PosFitness :=
  schedule.map (scalePosFitness c)

/-- The target carrier depends only on schedule length, so common fitness scaling
    preserves the exact ordered choices at every birth. -/
def scaleTargetTrace {n m : Nat} (c : PosFitness) :
    (schedule : List PosFitness) →
    TargetTrace n m schedule.length →
    TargetTrace n m (scaleSchedule c schedule).length
  | [], trace => trace
  | _eta :: rest, (T, tail) =>
      (T, scaleTargetTrace (n := n + 1) (m := m) c rest tail)

private theorem state_ext_scale {n : Nat} (s t : State n)
    (hg : s.snapshot.graph = t.snapshot.graph)
    (hf : s.snapshot.fitness = t.snapshot.fitness) : s = t := by
  rcases s with ⟨⟨sg, sd, sf⟩, sv⟩
  rcases t with ⟨⟨tg, td, tf⟩, tv⟩
  dsimp only at hg hf
  cases hg
  cases hf
  have hd : sd = td := Subsingleton.elim _ _
  cases hd
  rfl

/-- Common scaling commutes with the authoritative BB birth constructor. -/
private theorem applyBirth_scale_distribution {n m : Nat} (s : State n)
    (T : Targets n m) (hm : 0 < m) (eta c : PosFitness) :
    applyBirth (scaleFitness s c) T hm (scalePosFitness c eta) =
      scaleFitness (applyBirth s T hm eta) c := by
  apply state_ext_scale
  · rfl
  · funext i
    cases i using Fin.lastCases <;>
      simp [applyBirth, birthSnapshot, scaleFitness, scalePosFitness]

/-- Scaling every initial and future fitness value preserves the exact mass of
    the corresponding typed target trace. -/
theorem traceProbability_scale {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (trace : TargetTrace n m schedule.length) (c : PosFitness) :
    traceProbability (scaleFitness s c) m hm hb (scaleSchedule c schedule)
        (scaleTargetTrace c schedule trace) =
      traceProbability s m hm hb schedule trace := by
  induction schedule generalizing n with
  | nil =>
      cases trace
      rfl
  | cons eta rest ih =>
      rcases trace with ⟨T, tail⟩
      change
        orderedMass (scaleFitness s c) T *
            traceProbability
              (applyBirth (scaleFitness s c) T hm (scalePosFitness c eta))
              m hm (Nat.le_trans hb (Nat.le_succ n))
              (scaleSchedule c rest) (scaleTargetTrace c rest tail) =
          orderedMass s T *
            traceProbability (applyBirth s T hm eta) m hm
              (Nat.le_trans hb (Nat.le_succ n)) rest tail
      rw [orderedMass_scale, applyBirth_scale_distribution,
        ih (s := applyBirth s T hm eta)
          (hb := Nat.le_trans hb (Nat.le_succ n)) (trace := tail)]

/-- The scaled experiment reaches exactly the scaled final run state for the
    corresponding target trace. In particular, graph topology is unchanged. -/
theorem traceFinal_scale {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (trace : TargetTrace n m schedule.length) (c : PosFitness) :
    traceFinal (scaleFitness s c) m hm hb (scaleSchedule c schedule)
        (scaleTargetTrace c schedule trace) =
      scaleRunState (traceFinal s m hm hb schedule trace) c := by
  induction schedule generalizing n with
  | nil =>
      cases trace
      rfl
  | cons eta rest ih =>
      rcases trace with ⟨T, tail⟩
      change
        traceFinal
            (applyBirth (scaleFitness s c) T hm (scalePosFitness c eta))
            m hm (Nat.le_trans hb (Nat.le_succ n))
            (scaleSchedule c rest) (scaleTargetTrace c rest tail) =
          scaleRunState
            (traceFinal (applyBirth s T hm eta) m hm
              (Nat.le_trans hb (Nat.le_succ n)) rest tail) c
      rw [applyBirth_scale_distribution]
      exact ih (s := applyBirth s T hm eta)
        (hb := Nat.le_trans hb (Nat.le_succ n)) (trace := tail)

private theorem eventProbability_cons {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (eta : PosFitness) (rest : List PosFitness)
    (event : RunState → Prop) [DecidablePred event] :
    eventProbability s m hm hb (eta :: rest) event =
      ∑ T : Targets n m, orderedMass s T *
        eventProbability (applyBirth s T hm eta) m hm
          (Nat.le_trans hb (Nat.le_succ n)) rest event := by
  unfold eventProbability
  change
    (∑ trace : Targets n m × TargetTrace (n + 1) m rest.length,
      if event
          (traceFinal (applyBirth s trace.1 hm eta) m hm
            (Nat.le_trans hb (Nat.le_succ n)) rest trace.2)
      then orderedMass s trace.1 *
        traceProbability (applyBirth s trace.1 hm eta) m hm
          (Nat.le_trans hb (Nat.le_succ n)) rest trace.2
      else 0) =
      ∑ T : Targets n m, orderedMass s T *
        (∑ tail : TargetTrace (n + 1) m rest.length,
          if event
              (traceFinal (applyBirth s T hm eta) m hm
                (Nat.le_trans hb (Nat.le_succ n)) rest tail)
          then traceProbability (applyBirth s T hm eta) m hm
            (Nat.le_trans hb (Nat.le_succ n)) rest tail
          else 0)
  rw [Fintype.sum_prod_type]
  apply Finset.sum_congr rfl
  intro T _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro tail _
  by_cases h : event
      (traceFinal (applyBirth s T hm eta) m hm
        (Nat.le_trans hb (Nat.le_succ n)) rest tail)
  · simp [h]
  · simp [h]

/-- Every final-state event invariant under `scaleRunState` has the same exact
    probability after common positive scaling of the complete BB experiment. -/
theorem eventProbability_scale {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (c : PosFitness) (event : RunState → Prop) [DecidablePred event]
    (hinv : ∀ out, event (scaleRunState out c) ↔ event out) :
    eventProbability (scaleFitness s c) m hm hb (scaleSchedule c schedule) event =
      eventProbability s m hm hb schedule event := by
  induction schedule generalizing n with
  | nil =>
      simp only [scaleSchedule, List.map_nil]
      unfold eventProbability
      simp only [traceFinal, traceProbability]
      by_cases h : event ⟨n, s⟩
      · have hs : event ⟨n, scaleFitness s c⟩ := (hinv ⟨n, s⟩).2 h
        simp [h, hs]
      · have hs : ¬ event ⟨n, scaleFitness s c⟩ := by
          intro hs
          exact h ((hinv ⟨n, s⟩).1 (by simpa [scaleRunState] using hs))
        simp [h, hs]
  | cons eta rest ih =>
      simp only [scaleSchedule, List.map_cons]
      rw [eventProbability_cons, eventProbability_cons]
      apply Finset.sum_congr rfl
      intro T _
      rw [orderedMass_scale, applyBirth_scale_distribution]
      change
        orderedMass s T *
            eventProbability (scaleFitness (applyBirth s T hm eta) c)
              m hm (Nat.le_trans hb (Nat.le_succ n))
              (scaleSchedule c rest) event =
          orderedMass s T *
            eventProbability (applyBirth s T hm eta) m hm
              (Nat.le_trans hb (Nat.le_succ n)) rest event
      rw [ih (s := applyBirth s T hm eta)
        (hb := Nat.le_trans hb (Nat.le_succ n))]

end NarrativeDynamics.FitnessAttachment
