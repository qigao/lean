import NarrativeDynamics.Core.FitnessABM

namespace NarrativeDynamics.Tests.FitnessABM

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal
open NarrativeDynamics.FitnessABM

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
example : List.ofFn (fun i => ((advance joined).population.agents i).belief) =
    [1,1,0] := by decide_cbv

end NarrativeDynamics.Tests.FitnessABM
