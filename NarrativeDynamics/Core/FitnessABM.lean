import NarrativeDynamics.Core.FitnessBirth
import NarrativeDynamics.Core.NetworkPropagation

/-!
# Typed BB births and finite synchronous agent evolution

Fitness stays in the existing BB state. Agent profiles are fixed after birth,
and each tick propagates once from the complete post-birth snapshot. Only BB
target choices contribute probability; supplied agent data and ticks are fixed.
-/

namespace NarrativeDynamics.FitnessABM

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal

structure JointState (n : Nat) where
  network : FitnessAttachment.State n
  population : Population n
  populationValid : population.Valid

structure NewAgent where
  profile : AgentProfile
  initialBelief : Rat
  profileValid : profile.Valid
  beliefValid : 0 ≤ initialBelief ∧ initialBelief ≤ 1

structure BirthData where
  fitness : PosFitness
  agent : NewAgent

structure RunState where
  nodeCount : Nat
  roundIndex : Nat
  state : JointState nodeCount

structure Result where
  final : RunState
  probability : Rat

inductive Schedule : Nat → Nat → Type where
  | nil {n m : Nat} : Schedule n m
  | idle {n m : Nat} (rest : Schedule n m) : Schedule n m
  | birth {n m : Nat} (targets : Targets n m) (data : BirthData)
      (rest : Schedule (n + 1) m) : Schedule n m

inductive TickInput (n m : Nat) : Type where
  | idle
  | birth (targets : Targets n m) (data : BirthData)

/-- Old IDs retain both fields; the newborn starts with zero exposure. -/
def extendPopulation {n : Nat} (p : Population n) (a : NewAgent) : Population (n + 1) :=
  { profiles := Fin.lastCases a.profile p.profiles
    agents := Fin.lastCases ⟨a.initialBelief, 0⟩ p.agents }

theorem extendPopulation_valid {n : Nat} (p : Population n) (hp : p.Valid)
    (a : NewAgent) : (extendPopulation p a).Valid := by
  constructor
  · intro i
    refine Fin.lastCases ?_ (fun j => ?_) i
    · simpa [extendPopulation] using a.profileValid
    · simpa [extendPopulation] using hp.1 j
  · intro i
    refine Fin.lastCases ?_ (fun j => ?_) i
    · simpa [extendPopulation, AgentState.Valid] using a.beliefValid
    · simpa [extendPopulation] using hp.2 j

/-- Network growth is exactly the existing atomic BB birth. -/
def grow {n m : Nat} (s : JointState n) (T : Targets n m) (hm : 0 < m)
    (b : BirthData) : JointState (n + 1) :=
  { network := applyBirth s.network T hm b.fitness
    population := extendPopulation s.population b.agent
    populationValid := extendPopulation_valid s.population s.populationValid b.agent }

@[simp] theorem grow_projection {n m : Nat} (s : JointState n) (T : Targets n m)
    (hm : 0 < m) (b : BirthData) :
    (grow s T hm b).network = applyBirth s.network T hm b.fitness := rfl

@[simp] theorem grow_old_state {n m : Nat} (s : JointState n) (T : Targets n m)
    (hm : 0 < m) (b : BirthData) (i : Fin n) :
    (grow s T hm b).population.agents (oldId n i) = s.population.agents i := by
  simp [grow, extendPopulation, oldId]

@[simp] theorem grow_new_state {n m : Nat} (s : JointState n) (T : Targets n m)
    (hm : 0 < m) (b : BirthData) :
    (grow s T hm b).population.agents (newId n) = ⟨b.agent.initialBelief, 0⟩ := by
  simp [grow, extendPopulation, newId]

@[simp] theorem grow_old_profile {n m : Nat} (s : JointState n) (T : Targets n m)
    (hm : 0 < m) (b : BirthData) (i : Fin n) :
    (grow s T hm b).population.profiles (oldId n i) = s.population.profiles i := by
  simp [grow, extendPopulation, oldId]

/-- Undirected BB edges give both unit-influence transmission directions. -/
def advance {n : Nat} (s : JointState n) : JointState n :=
  letI := s.network.snapshot.adjDec
  { network := s.network
    population := propagate s.network.snapshot.graph.Adj s.population
    populationValid :=
      propagate_valid s.network.snapshot.graph.Adj s.population s.populationValid }

@[simp] theorem advance_projection {n : Nat} (s : JointState n) :
    (advance s).network = s.network := rfl

@[simp] theorem advance_profiles {n : Nat} (s : JointState n) :
    (advance s).population.profiles = s.population.profiles := rfl

/-- Same selected sets produce equal topology and all one-round observations.
No equality of the ordered target probabilities is asserted. -/
theorem grow_order_irrelevant {n m : Nat} (s : JointState n) (T U : Targets n m)
    (hm : 0 < m) (b : BirthData) (same : T.selected = U.selected) :
    (grow s T hm b).network.snapshot.graph = (grow s U hm b).network.snapshot.graph ∧
    ∀ i,
      (advance (grow s T hm b)).population.profiles i =
        (advance (grow s U hm b)).population.profiles i ∧
      ((advance (grow s T hm b)).population.agents i).belief =
        ((advance (grow s U hm b)).population.agents i).belief ∧
      ((advance (grow s T hm b)).population.agents i).exposures =
        ((advance (grow s U hm b)).population.agents i).exposures := by
  have hg : grow s T hm b = grow s U hm b := by
    unfold grow
    rw [birth_order_irrelevant s.network T U hm b.fitness same]
  rw [hg]
  exact ⟨rfl, fun _ => ⟨rfl, rfl, rfl⟩⟩

def Schedule.tickCount {n m : Nat} : Schedule n m → Nat
  | .nil => 0
  | .idle rest => rest.tickCount + 1
  | .birth _ _ rest => rest.tickCount + 1

def Schedule.birthCount {n m : Nat} : Schedule n m → Nat
  | .nil => 0
  | .idle rest => rest.birthCount
  | .birth _ _ rest => rest.birthCount + 1

/-- Structural recursion evaluates each finite tick once. The initial bound is
required even for empty/idle schedules and transported through every birth. -/
def runTyped {n m : Nat} (s : JointState n) (hm : 0 < m) (hb : m ≤ n)
    (schedule : Schedule n m) (roundIndex : Nat := 0) : Result :=
  match schedule with
  | .nil => ⟨⟨n, roundIndex, s⟩, 1⟩
  | .idle rest => runTyped (advance s) hm hb rest (roundIndex + 1)
  | .birth T b rest =>
      let tail := runTyped (advance (grow s T hm b)) hm
        (Nat.le_trans hb (Nat.le_succ n)) rest (roundIndex + 1)
      ⟨tail.final, orderedMass s.network T * tail.probability⟩
termination_by structural schedule

def tick {n m : Nat} (s : JointState n) (hm : 0 < m) (hb : m ≤ n)
    (input : TickInput n m) (roundIndex : Nat := 0) : Result :=
  match input with
  | .idle => runTyped s hm hb (.idle .nil) roundIndex
  | .birth T b => runTyped s hm hb (.birth T b .nil) roundIndex

/-- Node/round counts and actual graph edges follow the supplied finite schedule. -/
theorem runTyped_counts {n m : Nat} (s : JointState n) (hm : 0 < m) (hb : m ≤ n)
    (schedule : Schedule n m) (roundIndex : Nat := 0) :
    (runTyped s hm hb schedule roundIndex).final.nodeCount = n + schedule.birthCount ∧
    (runTyped s hm hb schedule roundIndex).final.roundIndex =
      roundIndex + schedule.tickCount ∧
    actualEdgeCount (runTyped s hm hb schedule roundIndex).final.state.network.snapshot =
      actualEdgeCount s.network.snapshot + m * schedule.birthCount := by
  revert s hm hb roundIndex
  induction schedule with
  | nil =>
      intro s hm hb roundIndex
      simp [runTyped, Schedule.birthCount, Schedule.tickCount]
  | idle rest ih =>
      intro s hm hb roundIndex
      have h := ih (advance s) hm hb (roundIndex + 1)
      simpa [runTyped, Schedule.birthCount, Schedule.tickCount,
        Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using h
  | birth T b rest ih =>
      intro s hm hb roundIndex
      have h := ih (advance (grow s T hm b)) hm
        (Nat.le_trans hb (Nat.le_succ _)) (roundIndex + 1)
      simpa [runTyped, Schedule.birthCount, Schedule.tickCount, birth_edges,
        Nat.mul_add, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using h

/-- Every supplied legal target trace has positive exact BB mass. -/
theorem runTyped_probability_pos {n m : Nat} (s : JointState n) (hm : 0 < m)
    (hb : m ≤ n) (schedule : Schedule n m) (roundIndex : Nat := 0) :
    0 < (runTyped s hm hb schedule roundIndex).probability := by
  revert s hm hb roundIndex
  induction schedule with
  | nil =>
      intro s hm hb roundIndex
      norm_num [runTyped]
  | idle rest ih =>
      intro s hm hb roundIndex
      exact ih (advance s) hm hb (roundIndex + 1)
  | birth T b rest ih =>
      intro s hm hb roundIndex
      exact mul_pos (orderedMass_pos s.network T)
        (ih (advance (grow s T hm b)) hm
          (Nat.le_trans hb (Nat.le_succ _)) (roundIndex + 1))

end NarrativeDynamics.FitnessABM
