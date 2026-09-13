import NarrativeDynamics.Core.FitnessDistribution

open NarrativeDynamics
open NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal

namespace NarrativeDynamics.FitnessAttachment.DistributionFixtures

example : Fintype.card (TargetTrace 3 2 0) = 1 := by decide_cbv
example : Fintype.card (TargetTrace 3 2 1) = 6 := by decide_cbv
example : Fintype.card (TargetTrace 2 1 2) = 6 := by decide_cbv

def edgeSeed : RawSeed := ⟨2, #[1, 1], #[(0, 1)]⟩

private theorem edgeSeedConnected : (seedGraph edgeSeed).Connected where
  preconnected := by
    intro u v
    by_cases h : u = v
    · subst v
      exact ⟨.nil⟩
    · have huv : (seedGraph edgeSeed).Adj u v := by
        fin_cases u <;> fin_cases v <;>
          simp_all [seedGraph, edgeSeed, canonicalEdge]
      exact ⟨.cons huv .nil⟩
  nonempty := ⟨⟨0, by decide⟩⟩

def edge2State : State 2 :=
  ⟨seedSnapshot edgeSeed rfl, by
    exact ⟨by decide, edgeSeedConnected, by decide⟩⟩

private def unitFitness : PosFitness := ⟨1, by norm_num⟩

private def zeroTarget2 : Targets 2 1 :=
  ⟨fun _ => ⟨0, by decide⟩, fun i j _ => Subsingleton.elim i j⟩

private def zeroTarget3 : Targets 3 1 :=
  ⟨fun _ => ⟨0, by decide⟩, fun i j _ => Subsingleton.elim i j⟩

def edge3State : State 3 :=
  applyBirth edge2State zeroTarget2 (by decide) unitFitness

def edge4State : State 4 :=
  applyBirth edge3State zeroTarget3 (by decide) unitFitness

private theorem birthDegreeFn {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) :
    degree (applyBirth s T hm eta).snapshot =
      Fin.lastCases m (fun u => degree s.snapshot u + if u ∈ T.selected then 1 else 0) := by
  funext v
  refine Fin.lastCases ?_ (fun u => ?_) v
  · rw [Fin.lastCases_last]
    exact birth_degree_new s T hm eta
  · rw [Fin.lastCases_castSucc]
    exact birth_degree_old s T hm eta u

private theorem birthFitnessFn {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) :
    (applyBirth s T hm eta).snapshot.fitness =
      Fin.lastCases eta.val s.snapshot.fitness := rfl

private theorem edge2Degrees :
    degree edge2State.snapshot = (![1, 1] : Fin 2 → Nat) := by
  funext i
  fin_cases i <;> decide_cbv

private theorem edge2Fitness :
    edge2State.snapshot.fitness = (![1, 1] : Fin 2 → Rat) := by
  funext i
  fin_cases i <;> decide_cbv

private theorem edge2FitnessList :
    List.ofFn edge2State.snapshot.fitness = [1, 1] := by
  rw [edge2Fitness]
  decide_cbv

private theorem edge2Edges : actualEdgeCount edge2State.snapshot = 1 := by
  decide_cbv

private theorem edge2Weights :
    weights edge2State.snapshot = (![1, 1] : Fin 2 → Rat) := by
  funext i
  unfold weights
  rw [edge2Fitness, edge2Degrees]
  fin_cases i <;> decide_cbv

private theorem edge2Mass : orderedMass edge2State zeroTarget2 = 1 / 2 := by
  unfold orderedMass
  rw [edge2Weights]
  decide_cbv

private theorem edge3Degrees :
    degree edge3State.snapshot = (![2, 1, 1] : Fin 3 → Nat) := by
  rw [edge3State, birthDegreeFn, edge2Degrees]
  funext i
  fin_cases i <;> decide_cbv

private theorem edge3Fitness :
    edge3State.snapshot.fitness = (![1, 1, 1] : Fin 3 → Rat) := by
  rw [edge3State, birthFitnessFn, edge2Fitness]
  funext i
  fin_cases i <;> decide_cbv

private theorem edge3Weights :
    weights edge3State.snapshot = (![2, 1, 1] : Fin 3 → Rat) := by
  funext i
  unfold weights
  rw [edge3Fitness, edge3Degrees]
  fin_cases i <;> decide_cbv

private theorem edge3Mass : orderedMass edge3State zeroTarget3 = 1 / 2 := by
  unfold orderedMass
  rw [edge3Weights]
  decide_cbv

private def twoBirthSchedule : List PosFitness := [unitFitness, unitFitness]

private def zeroZeroTrace : TargetTrace 2 1 twoBirthSchedule.length :=
  (zeroTarget2, (zeroTarget3, PUnit.unit))

example :
    traceProbability edge2State 1 (by decide) (by decide)
      twoBirthSchedule zeroZeroTrace = 1 / 4 := by
  change orderedMass edge2State zeroTarget2 *
    (orderedMass edge3State zeroTarget3 * 1) = 1 / 4
  rw [edge2Mass, edge3Mass]
  norm_num

example :
    (∑ trace : TargetTrace 2 1 twoBirthSchedule.length,
      traceProbability edge2State 1 (by decide) (by decide)
        twoBirthSchedule trace) = 1 := by
  exact traceProbability_sum_one edge2State 1 (by decide) (by decide) twoBirthSchedule

example :
    traceFinal edge2State 1 (by decide) (by decide)
      twoBirthSchedule zeroZeroTrace = ⟨4, edge4State⟩ := by
  rfl

example :
    actualNodeCount
        (traceFinal edge2State 1 (by decide) (by decide)
          twoBirthSchedule zeroZeroTrace).state.snapshot = 4 := by
  rw [traceFinal_nodes]
  norm_num [actualNodeCount, twoBirthSchedule]

example :
    actualEdgeCount
        (traceFinal edge2State 1 (by decide) (by decide)
          twoBirthSchedule zeroZeroTrace).state.snapshot = 3 := by
  rw [traceFinal_edges, edge2Edges]
  norm_num [twoBirthSchedule]

example :
    List.ofFn
        (traceFinal edge2State 1 (by decide) (by decide)
          twoBirthSchedule zeroZeroTrace).state.snapshot.fitness = [1, 1, 1, 1] := by
  rw [traceFinal_fitness, edge2FitnessList]
  rfl

example :
    (traceFinal edge2State 1 (by decide) (by decide)
      twoBirthSchedule zeroZeroTrace).state.snapshot.Valid := by
  exact traceFinal_valid edge2State 1 (by decide) (by decide)
    twoBirthSchedule zeroZeroTrace

#print axioms NarrativeDynamics.FitnessAttachment.traceProbability_pos
#print axioms NarrativeDynamics.FitnessAttachment.traceProbability_sum_continuationMass
#print axioms NarrativeDynamics.FitnessAttachment.traceProbability_sum_one
#print axioms NarrativeDynamics.FitnessAttachment.traceProbability_le_one
#print axioms NarrativeDynamics.FitnessAttachment.traceFinal_nodes
#print axioms NarrativeDynamics.FitnessAttachment.traceFinal_edges
#print axioms NarrativeDynamics.FitnessAttachment.traceFinal_fitness
#print axioms NarrativeDynamics.FitnessAttachment.traceFinal_valid

end NarrativeDynamics.FitnessAttachment.DistributionFixtures
