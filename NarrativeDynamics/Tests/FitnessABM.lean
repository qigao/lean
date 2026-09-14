import NarrativeDynamics.Core.FitnessABM

namespace NarrativeDynamics.Tests.FitnessABM

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal
open NarrativeDynamics.FitnessABM

-- Temporary CI diagnostics: force proof dependencies, then flush stdout directly.
-- This observes elaboration only and supplies no proof of any fixture assertion.
private def fixtureCheckpoint (name : Lean.Name) : Lean.Elab.Command.CommandElabM Unit := do
  let out ← IO.getStdout
  out.putStrLn s!"FitnessABM checkpoint waiting: {name}"
  out.flush
  let dependencies ← Lean.collectAxioms name
  out.putStrLn s!"FitnessABM checkpoint complete: {name}; dependencies={dependencies}"
  out.flush

private theorem seed2_connected : (⊤ : SimpleGraph (Fin 2)).Connected where
  preconnected := by
    intro u v
    by_cases h : u = v
    · subst v; exact ⟨.nil⟩
    · exact ⟨.cons h .nil⟩
  nonempty := inferInstance

def seed2 : JointState 2 where
  network :=
    { snapshot := { graph := ⊤, adjDec := inferInstance, fitness := ![1, 1] }
      valid := ⟨by decide, seed2_connected, by
        intro i
        fin_cases i <;> norm_num⟩ }
  population := ⟨fun _ => ⟨1, 1/2⟩, ![⟨1, 0⟩, ⟨0, 0⟩]⟩
  populationValid := by
    constructor
    · intro i
      fin_cases i <;> norm_num [AgentProfile.Valid]
    · intro i
      fin_cases i <;> norm_num [AgentState.Valid]

run_cmd fixtureCheckpoint ``seed2

def newbornZero : NewAgent :=
  { profile := ⟨1, 1/2⟩, initialBelief := 0
    profileValid := by norm_num [AgentProfile.Valid]
    beliefValid := by norm_num }
def birthOne : BirthData := ⟨⟨1, by norm_num⟩, newbornZero⟩
def target1 : Targets 2 1 := ⟨![1], by decide⟩
def joined := grow seed2 target1 (by decide) birthOne

-- Catches loss of old IDs, nonzero newborn exposure, and in-round relay cascades.
example : joined.population.agents (oldId 2 0) = seed2.population.agents 0 := by
  simpa [joined] using grow_old_state seed2 target1 (by decide) birthOne 0
example : joined.population.agents (newId 2) = ⟨0,0⟩ := by decide_cbv
theorem diagnosticFirstRound : List.ofFn (fun i => ((advance joined).population.agents i).belief) =
    [1,1,0] := by decide_cbv

run_cmd fixtureCheckpoint ``diagnosticFirstRound

-- Profiles survive embedding, independently of their agents' changing beliefs.
example : joined.population.profiles (oldId 2 0) = seed2.population.profiles 0 := by
  simpa [joined] using grow_old_profile seed2 target1 (by decide) birthOne 0

def successiveBirths : Schedule 2 1 :=
  .birth target1 birthOne (.birth ⟨![2], by decide⟩ birthOne .nil)
def twoBirths := runTyped seed2 (by decide) (by decide) successiveBirths

-- Catches stale attachment weights, omitted rounds, and asynchronous forwarding.
theorem diagnosticTwoBirthBeliefs : List.ofFn (fun i => (twoBirths.final.state.population.agents i).belief) =
    [1,1,1,0] := by decide_cbv
run_cmd fixtureCheckpoint ``diagnosticTwoBirthBeliefs

theorem diagnosticTwoBirthExposures : List.ofFn (fun i => (twoBirths.final.state.population.agents i).exposures) =
    [1,2,1,0] := by decide_cbv
run_cmd fixtureCheckpoint ``diagnosticTwoBirthExposures

example : twoBirths.final.nodeCount = 4 := by decide_cbv
example : actualEdgeCount twoBirths.final.state.network.snapshot = 3 := by decide_cbv
example : twoBirths.final.roundIndex = 2 := by decide_cbv
theorem diagnosticTwoBirthMass : twoBirths.probability = 1/8 := by decide_cbv
run_cmd fixtureCheckpoint ``diagnosticTwoBirthMass

def birthThenIdle : Schedule 2 1 :=
  .birth target1 birthOne (.birth ⟨![2], by decide⟩ birthOne (.idle .nil))
def thirdRound := runTyped seed2 (by decide) (by decide) birthThenIdle

-- The last newborn receives on the following round; idle adds no probability.
theorem diagnosticThirdBeliefs : List.ofFn (fun i => (thirdRound.final.state.population.agents i).belief) =
    [1,1,1,1] := by decide_cbv
run_cmd fixtureCheckpoint ``diagnosticThirdBeliefs

theorem diagnosticThirdExposures : List.ofFn (fun i => (thirdRound.final.state.population.agents i).exposures) =
    [2,4,2,1] := by decide_cbv
run_cmd fixtureCheckpoint ``diagnosticThirdExposures

example : thirdRound.final.roundIndex = 3 := by decide_cbv
theorem diagnosticThirdMass : thirdRound.probability = 1/8 := by decide_cbv
run_cmd fixtureCheckpoint ``diagnosticThirdMass

def newbornOne : NewAgent :=
  { newbornZero with initialBelief := 1, beliefValid := by norm_num }
def broadcastingBirth : BirthData := ⟨⟨1, by norm_num⟩, newbornOne⟩
def immediate := tick seed2 (by decide) (by decide)
  (.birth target1 broadcastingBirth)

-- Birth precedes propagation: a newborn above threshold broadcasts immediately.
example : List.ofFn (fun i => (immediate.final.state.population.agents i).belief) =
    [1,1,1] := by decide_cbv
theorem diagnosticImmediate : List.ofFn (fun i => (immediate.final.state.population.agents i).exposures) =
    [0,2,0] := by decide_cbv
run_cmd fixtureCheckpoint ``diagnosticImmediate

def idleOnly := runTyped (m := 1) seed2 (by decide) (by decide)
  (.idle (.idle .nil)) 7
example : idleOnly.probability = 1 := by decide_cbv
example : idleOnly.final.roundIndex = 9 := by decide_cbv
example : idleOnly.final.nodeCount = 2 := by decide_cbv
theorem diagnosticIdle : List.ofFn (fun i => (idleOnly.final.state.population.agents i).belief) =
    [1,1] := by decide_cbv
run_cmd fixtureCheckpoint ``diagnosticIdle
example : (tick (m := 1) seed2 (by decide) (by decide) .idle 7).final.roundIndex =
    8 := by decide_cbv
example : (runTyped (m := 1) seed2 (by decide) (by decide) .nil 7).final.roundIndex =
    7 := by decide_cbv

-- Reordering a full target set changes neither topology nor the observed round.
def target01 : Targets 2 2 := ⟨![0,1], by decide⟩
def target10 : Targets 2 2 := ⟨![1,0], by decide⟩
theorem diagnosticOrder :
    (grow seed2 target01 (by decide) birthOne).network.snapshot.graph =
      (grow seed2 target10 (by decide) birthOne).network.snapshot.graph ∧
    ∀ i,
      (advance (grow seed2 target01 (by decide) birthOne)).population.profiles i =
        (advance (grow seed2 target10 (by decide) birthOne)).population.profiles i ∧
      ((advance (grow seed2 target01 (by decide) birthOne)).population.agents i).belief =
        ((advance (grow seed2 target10 (by decide) birthOne)).population.agents i).belief ∧
      ((advance (grow seed2 target01 (by decide) birthOne)).population.agents i).exposures =
        ((advance (grow seed2 target10 (by decide) birthOne)).population.agents i).exposures :=
  grow_order_irrelevant seed2 target01 target10 (by decide) birthOne (by decide_cbv)
run_cmd fixtureCheckpoint ``diagnosticOrder

#print axioms NarrativeDynamics.FitnessABM.extendPopulation_valid
#print axioms NarrativeDynamics.FitnessABM.grow_order_irrelevant
#print axioms NarrativeDynamics.FitnessABM.runTyped_counts
#print axioms NarrativeDynamics.FitnessABM.runTyped_probability_pos

end NarrativeDynamics.Tests.FitnessABM
