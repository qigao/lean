import NarrativeDynamics.Core.FitnessABMDistribution

namespace NarrativeDynamics.FitnessABM.DistributionFixtures

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessAttachment

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

def birthOne : BirthData :=
  { fitness := ⟨1, by norm_num⟩
    agent :=
      { profile := ⟨1, 1/2⟩, initialBelief := 0
        profileValid := by norm_num [AgentProfile.Valid]
        beliefValid := by norm_num } }

def newbornBroadcasts (out : RunState) : Bool :=
  if h : 2 < out.nodeCount then
    NetworkPropagation.broadcasting
      (out.state.population.profiles ⟨2, h⟩)
      (out.state.population.agents ⟨2, h⟩)
  else false

theorem newborn_mass_unit : eventProbability seed2 1 (by decide) (by decide)
    [some birthOne] (fun out => newbornBroadcasts out = true) = 1/2 := by
  decide_cbv

end NarrativeDynamics.FitnessABM.DistributionFixtures
