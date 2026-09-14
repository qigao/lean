import NarrativeDynamics.Core.FitnessABM
import NarrativeDynamics.Core.FitnessDistribution

/-!
# Exact joint BB / agent outcome probabilities

The calendar fixes births and propagation rounds. Every existing ordered target
trace contributes its BB mass exactly once; deterministic agent updates add no
probability factors and never remove traces from the carrier.
-/

namespace NarrativeDynamics.FitnessABM

open NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal
open scoped BigOperators

abbrev Calendar := List (Option BirthData)

/-- Filter idle ticks and retain the supplied newborn fitnesses in order. -/
def fitnessSchedule : Calendar → List PosFitness
  | [] => []
  | none :: rest => fitnessSchedule rest
  | some data :: rest => data.fitness :: fitnessSchedule rest

/-- Attach the authoritative target trace to the fixed birth calendar. -/
def scheduleOfTrace {n m : Nat} (calendar : Calendar) :
    TargetTrace n m (fitnessSchedule calendar).length → Schedule n m :=
  match calendar with
  | [] => fun _ => .nil
  | none :: rest => fun trace => .idle (scheduleOfTrace rest trace)
  | some data :: rest => fun trace =>
      .birth trace.1 data (scheduleOfTrace rest trace.2)

/-- Actual composed execution, including exactly one propagation per tick. -/
def jointFinal {n : Nat} (s : JointState n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (calendar : Calendar)
    (trace : TargetTrace n m (fitnessSchedule calendar).length) : RunState :=
  (runTyped s hm hb (scheduleOfTrace calendar trace)).final

/-- The existing normalized BB law is the entire joint trace probability. -/
def jointProbability {n : Nat} (s : JointState n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (calendar : Calendar)
    (trace : TargetTrace n m (fitnessSchedule calendar).length) : Rat :=
  traceProbability s.network m hm hb (fitnessSchedule calendar) trace

/-- Sum the existing executable carrier, retaining precisely the event's mass. -/
def eventProbability {n : Nat} (s : JointState n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (calendar : Calendar)
    (event : RunState → Prop) [DecidablePred event] : Rat :=
  ∑ trace : TargetTrace n m (fitnessSchedule calendar).length,
    if event (jointFinal s m hm hb calendar trace)
    then jointProbability s m hm hb calendar trace else 0

private theorem probability_bridge {n : Nat} (s : JointState n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (calendar : Calendar)
    (trace : TargetTrace n m (fitnessSchedule calendar).length) (roundIndex : Nat) :
    traceProbability s.network m hm hb (fitnessSchedule calendar) trace =
      (runTyped s hm hb (scheduleOfTrace calendar trace) roundIndex).probability := by
  induction calendar generalizing n roundIndex with
  | nil => rfl
  | cons entry rest ih =>
      cases entry with
      | none =>
          simpa only [fitnessSchedule, scheduleOfTrace, runTyped, advance_projection]
            using ih (s := advance s) (hb := hb) (trace := trace)
              (roundIndex := roundIndex + 1)
      | some data =>
          rcases trace with ⟨T, tail⟩
          change orderedMass s.network T *
              traceProbability (applyBirth s.network T hm data.fitness) m hm
                (Nat.le_trans hb (Nat.le_succ n)) (fitnessSchedule rest) tail =
            orderedMass s.network T *
              (runTyped (advance (grow s T hm data)) hm
                (Nat.le_trans hb (Nat.le_succ n))
                (scheduleOfTrace rest tail) (roundIndex + 1)).probability
          have h := ih (s := advance (grow s T hm data))
            (hb := Nat.le_trans hb (Nat.le_succ n)) (trace := tail)
            (roundIndex := roundIndex + 1)
          simpa only [advance_projection, grow_projection] using
            congrArg (fun mass => orderedMass s.network T * mass) h

/-- The trace law agrees with the mass returned by the actual composed run. -/
theorem jointProbability_eq_runTyped {n : Nat} (s : JointState n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (calendar : Calendar)
    (trace : TargetTrace n m (fitnessSchedule calendar).length) :
    jointProbability s m hm hb calendar trace =
      (runTyped s hm hb (scheduleOfTrace calendar trace)).probability :=
  probability_bridge s m hm hb calendar trace 0

private theorem final_bridge {n : Nat} (s : JointState n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (calendar : Calendar)
    (trace : TargetTrace n m (fitnessSchedule calendar).length) (roundIndex : Nat) :
    (⟨(runTyped s hm hb (scheduleOfTrace calendar trace) roundIndex).final.nodeCount,
      (runTyped s hm hb (scheduleOfTrace calendar trace) roundIndex).final.state.network⟩ :
        FitnessAttachment.RunState) =
      traceFinal s.network m hm hb (fitnessSchedule calendar) trace := by
  induction calendar generalizing n roundIndex with
  | nil => rfl
  | cons entry rest ih =>
      cases entry with
      | none =>
          simpa only [fitnessSchedule, scheduleOfTrace, runTyped, advance_projection]
            using ih (s := advance s) (hb := hb) (trace := trace)
              (roundIndex := roundIndex + 1)
      | some data =>
          rcases trace with ⟨T, tail⟩
          simpa only [fitnessSchedule, scheduleOfTrace, runTyped, traceFinal,
            advance_projection, grow_projection]
            using ih (s := advance (grow s T hm data))
              (hb := Nat.le_trans hb (Nat.le_succ n)) (trace := tail)
              (roundIndex := roundIndex + 1)

private theorem state_observations (a b : FitnessAttachment.RunState) (h : a = b) :
    ∃ hn : a.nodeCount = b.nodeCount,
      (∀ i j, a.state.snapshot.graph.Adj i j ↔
        b.state.snapshot.graph.Adj (Fin.cast hn i) (Fin.cast hn j)) ∧
      (∀ i, a.state.snapshot.fitness i = b.state.snapshot.fitness (Fin.cast hn i)) ∧
      actualEdgeCount a.state.snapshot = actualEdgeCount b.state.snapshot := by
  cases h
  exact ⟨rfl, fun _ _ => Iff.rfl, fun _ => rfl, rfl⟩

/-- Stable-ID graph and fitness observations equal the authoritative BB trace. -/
theorem jointFinal_projection {n : Nat} (s : JointState n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (calendar : Calendar)
    (trace : TargetTrace n m (fitnessSchedule calendar).length) :
    let joint := jointFinal s m hm hb calendar trace
    let bb := traceFinal s.network m hm hb (fitnessSchedule calendar) trace
    ∃ hn : joint.nodeCount = bb.nodeCount,
      (∀ i j, joint.state.network.snapshot.graph.Adj i j ↔
        bb.state.snapshot.graph.Adj (Fin.cast hn i) (Fin.cast hn j)) ∧
      (∀ i, joint.state.network.snapshot.fitness i =
        bb.state.snapshot.fitness (Fin.cast hn i)) ∧
      actualEdgeCount joint.state.network.snapshot = actualEdgeCount bb.state.snapshot :=
  state_observations _ _ (final_bridge s m hm hb calendar trace 0)

/-- Every ordered trace is counted once by the existing normalized carrier. -/
theorem jointProbability_sum_one {n : Nat} (s : JointState n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (calendar : Calendar) :
    (∑ trace : TargetTrace n m (fitnessSchedule calendar).length,
      jointProbability s m hm hb calendar trace) = 1 :=
  traceProbability_sum_one s.network m hm hb (fitnessSchedule calendar)

theorem eventProbability_true {n : Nat} (s : JointState n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (calendar : Calendar) :
    eventProbability s m hm hb calendar (fun _ => True) = 1 := by
  simp [eventProbability, jointProbability_sum_one]

theorem eventProbability_false {n : Nat} (s : JointState n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (calendar : Calendar) :
    eventProbability s m hm hb calendar (fun _ => False) = 0 := by
  simp [eventProbability]

theorem eventProbability_nonneg {n : Nat} (s : JointState n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (calendar : Calendar)
    (event : RunState → Prop) [DecidablePred event] :
    0 ≤ eventProbability s m hm hb calendar event := by
  unfold eventProbability
  apply Finset.sum_nonneg
  intro trace _
  split
  · exact traceProbability_nonneg s.network m hm hb (fitnessSchedule calendar) trace
  · exact le_rfl

theorem eventProbability_le_one {n : Nat} (s : JointState n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (calendar : Calendar)
    (event : RunState → Prop) [DecidablePred event] :
    eventProbability s m hm hb calendar event ≤ 1 := by
  calc
    eventProbability s m hm hb calendar event ≤
        ∑ trace : TargetTrace n m (fitnessSchedule calendar).length,
          jointProbability s m hm hb calendar trace := by
      unfold eventProbability
      apply Finset.sum_le_sum
      intro trace _
      split
      · exact le_rfl
      · exact traceProbability_nonneg s.network m hm hb (fitnessSchedule calendar) trace
    _ = 1 := jointProbability_sum_one s m hm hb calendar

end NarrativeDynamics.FitnessABM
