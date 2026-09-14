import NarrativeDynamics.Conformance.FitnessABMRuntimeVectors

namespace NarrativeDynamics.Tests.FitnessABMRuntime

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal
open NarrativeDynamics.FitnessABM
open NarrativeDynamics.Conformance.BBRuntime
open scoped BigOperators

-- This observation declaration is intentionally required by the RED fixture.
#check observeTransition

def summary (input : RuntimeCaseInput) :
    Except JointError (Nat × Nat × Nat × List Rat × List Nat × Rat) :=
  (FitnessABM.replay input.seed input.m input.agents input.ticks).map fun out =>
    (out.final.nodeCount, out.final.roundIndex,
      actualEdgeCount out.final.state.network.snapshot,
      List.ofFn (fun i => (out.final.state.population.agents i).belief),
      List.ofFn (fun i => (out.final.state.population.agents i).exposures),
      out.probability)

private instance {α : Type} [DecidableEq α] : DecidableEq (Except JointError α) :=
  fun x y => match x, y with
  | .error x, .error y =>
      if h : x = y then .isTrue (by cases h; rfl)
      else .isFalse (fun he => h (Except.error.inj he))
  | .ok x, .ok y =>
      if h : x = y then .isTrue (by cases h; rfl)
      else .isFalse (fun he => h (Except.ok.inj he))
  | .error _, .ok _ => .isFalse (by intro h; cases h)
  | .ok _, .error _ => .isFalse (by intro h; cases h)

section FiniteLiterals
set_option maxRecDepth 4096

private theorem empty_literal : summary empty =
    .ok (2, 0, 1, [1, 0], [0, 0], 1) := by
  decide_cbv

private theorem idleTwo_literal : summary idleTwo =
    .ok (2, 2, 1, [1, 1], [1, 2], 1) := by
  decide_cbv

private theorem attachSource_literal : summary attachSource =
    .ok (3, 1, 2, [1, 1, 1], [0, 1, 1], 1/2) := by
  decide_cbv

private theorem attachRelay_literal : summary attachRelay =
    .ok (3, 1, 2, [1, 1, 0], [0, 1, 0], 1/2) := by
  decide_cbv

private theorem relayIdle_literal : summary relayIdle =
    .ok (3, 2, 2, [1, 1, 1], [1, 2, 1], 1/2) := by
  decide_cbv

private theorem successiveBirths_literal : summary successiveBirths =
    .ok (4, 2, 3, [1, 1, 1, 0], [1, 2, 1, 0], 1/8) := by
  decide_cbv

private theorem successiveIdle_literal : summary successiveIdle =
    .ok (4, 3, 3, [1, 1, 1, 1], [2, 4, 2, 1], 1/8) := by
  decide_cbv

private theorem weightedSource_literal : summary weightedSource =
    .ok (3, 1, 2, [1, 1, 1], [0, 1, 1], 1/4) := by
  decide_cbv

private theorem ordered01_literal : summary ordered01 =
    .ok (3, 1, 3, [1, 1, 1], [0, 1, 1], 1/4) := by
  decide_cbv

private theorem ordered10_literal : summary ordered10 =
    .ok (3, 1, 3, [1, 1, 1], [0, 1, 1], 3/4) := by
  decide_cbv

private theorem zeroReceptive_literal : summary zeroReceptive =
    .ok (3, 1, 2, [1, 0, 1], [0, 1, 1], 1/2) := by
  decide_cbv

private theorem silent_literal : summary silent =
    .ok (3, 1, 2, [0, 0, 0], [0, 0, 0], 1/2) := by
  decide_cbv

private theorem zeroThreshold_literal : summary zeroThreshold =
    .ok (3, 1, 2, [0, 0, 0], [0, 1, 1], 1/2) := by
  decide_cbv

private theorem broadcastNewborn_literal : summary broadcastNewborn =
    .ok (3, 1, 2, [1, 1, 1], [0, 2, 0], 1/2) := by
  decide_cbv

private theorem retainedExposures_literal : summary retainedExposures =
    .ok (3, 1, 2, [1, 1, 0], [7, 4, 0], 1/2) := by
  decide_cbv

private theorem halfReceptive_literal : summary halfReceptive =
    .ok (3, 1, 2, [1, 1/2, 1], [0, 1, 1], 1/2) := by
  decide_cbv

private theorem seedNodes_literal : summary seedNodes =
    .error (.seedNetwork .invalidNodeCount) := by
  decide_cbv

private theorem seedSize_literal : summary seedSize =
    .error (.seedNetwork .fitnessSizeMismatch) := by
  decide_cbv

private theorem seedFitness_literal : summary seedFitness =
    .error (.seedNetwork .nonpositiveFitness) := by
  decide_cbv

private theorem seedEdge_literal : summary seedEdge =
    .error (.seedNetwork .invalidEdge) := by
  decide_cbv

private theorem seedDuplicate_literal : summary seedDuplicate =
    .error (.seedNetwork .duplicateEdge) := by
  decide_cbv

private theorem seedDisconnected_literal : summary seedDisconnected =
    .error (.seedNetwork .disconnectedSeed) := by
  decide_cbv

private theorem initialMZero_literal : summary initialMZero =
    .error (.initialM) := by
  decide_cbv

private theorem initialMTooLarge_literal : summary initialMTooLarge =
    .error (.initialM) := by
  decide_cbv

private theorem seedAgentCount_literal : summary seedAgentCount =
    .error (.seedAgentCount 2 1) := by
  decide_cbv

private theorem seedAgentR_literal : summary seedAgentR =
    .error (.seedAgent 0 .receptivity) := by
  decide_cbv

private theorem seedAgentThreshold_literal : summary seedAgentThreshold =
    .error (.seedAgent 0 .threshold) := by
  decide_cbv

private theorem seedAgentBelief_literal : summary seedAgentBelief =
    .error (.seedAgent 1 .belief) := by
  decide_cbv

private theorem birthFitness_literal : summary birthFitness =
    .error (.tickNetwork 0 0 .nonpositiveFitness) := by
  decide_cbv

private theorem birthTargetCount_literal : summary birthTargetCount =
    .error (.tickNetwork 0 0 .targetCountMismatch) := by
  decide_cbv

private theorem birthTargetRange_literal : summary birthTargetRange =
    .error (.tickNetwork 0 0 .targetOutOfRange) := by
  decide_cbv

private theorem birthTargetDuplicate_literal : summary birthTargetDuplicate =
    .error (.tickNetwork 0 0 .duplicateTarget) := by
  decide_cbv

private theorem birthAgentThreshold_literal : summary birthAgentThreshold =
    .error (.tickAgent 0 0 .threshold) := by
  decide_cbv

private theorem lateBirth_literal : summary lateBirth =
    .error (.tickNetwork 3 1 .targetOutOfRange) := by
  decide_cbv

private theorem firstFailure_literal : summary firstFailure =
    .error (.tickAgent 0 0 .threshold) := by
  decide_cbv

private theorem lateFirstBirth_literal : summary lateFirstBirth =
    .error (.tickNetwork 2 0 .targetOutOfRange) := by
  decide_cbv

private theorem successive_growth_literal :
    (observeTransition successiveBirths ⟨1, by decide⟩).map
      (fun t => (t.postGrowth.beliefs, t.postGrowth.exposures, t.tickMass, t.transmissions)) =
      .ok ([1, 1, 0, 0], [0, 1, 0, 0], 1/4, [(0, 1, 1), (1, 0, 1), (1, 2, 1)]) := by
  decide_cbv

private theorem retained_growth_literal :
    (observeTransition retainedExposures ⟨0, by decide⟩).map
      (fun t => t.postGrowth.exposures) = .ok [7, 3, 0] := by
  decide_cbv

private theorem newborn_transmissions_literal :
    (observeTransition broadcastNewborn ⟨0, by decide⟩).map
      (fun t => t.transmissions) = .ok [(0, 1, 1), (2, 1, 1)] := by
  decide_cbv

theorem success_literals :
  summary empty = .ok (2, 0, 1, [1, 0], [0, 0], 1) ∧
  summary idleTwo = .ok (2, 2, 1, [1, 1], [1, 2], 1) ∧
  summary attachSource = .ok (3, 1, 2, [1, 1, 1], [0, 1, 1], 1/2) ∧
  summary attachRelay = .ok (3, 1, 2, [1, 1, 0], [0, 1, 0], 1/2) ∧
  summary relayIdle = .ok (3, 2, 2, [1, 1, 1], [1, 2, 1], 1/2) ∧
  summary successiveBirths = .ok (4, 2, 3, [1, 1, 1, 0], [1, 2, 1, 0], 1/8) ∧
  summary successiveIdle = .ok (4, 3, 3, [1, 1, 1, 1], [2, 4, 2, 1], 1/8) ∧
  summary weightedSource = .ok (3, 1, 2, [1, 1, 1], [0, 1, 1], 1/4) ∧
  summary ordered01 = .ok (3, 1, 3, [1, 1, 1], [0, 1, 1], 1/4) ∧
  summary ordered10 = .ok (3, 1, 3, [1, 1, 1], [0, 1, 1], 3/4) ∧
  summary zeroReceptive = .ok (3, 1, 2, [1, 0, 1], [0, 1, 1], 1/2) ∧
  summary silent = .ok (3, 1, 2, [0, 0, 0], [0, 0, 0], 1/2) ∧
  summary zeroThreshold = .ok (3, 1, 2, [0, 0, 0], [0, 1, 1], 1/2) ∧
  summary broadcastNewborn = .ok (3, 1, 2, [1, 1, 1], [0, 2, 0], 1/2) ∧
  summary retainedExposures = .ok (3, 1, 2, [1, 1, 0], [7, 4, 0], 1/2) ∧
  summary halfReceptive = .ok (3, 1, 2, [1, 1/2, 1], [0, 1, 1], 1/2) :=
  ⟨empty_literal, idleTwo_literal, attachSource_literal, attachRelay_literal, relayIdle_literal, successiveBirths_literal, successiveIdle_literal, weightedSource_literal, ordered01_literal, ordered10_literal, zeroReceptive_literal, silent_literal, zeroThreshold_literal, broadcastNewborn_literal, retainedExposures_literal, halfReceptive_literal⟩

theorem shared_error_literals :
  summary seedNodes = .error (.seedNetwork .invalidNodeCount) ∧
  summary seedSize = .error (.seedNetwork .fitnessSizeMismatch) ∧
  summary seedFitness = .error (.seedNetwork .nonpositiveFitness) ∧
  summary seedEdge = .error (.seedNetwork .invalidEdge) ∧
  summary seedDuplicate = .error (.seedNetwork .duplicateEdge) ∧
  summary seedDisconnected = .error (.seedNetwork .disconnectedSeed) ∧
  summary initialMZero = .error (.initialM) ∧
  summary initialMTooLarge = .error (.initialM) ∧
  summary seedAgentCount = .error (.seedAgentCount 2 1) ∧
  summary seedAgentR = .error (.seedAgent 0 .receptivity) ∧
  summary seedAgentThreshold = .error (.seedAgent 0 .threshold) ∧
  summary seedAgentBelief = .error (.seedAgent 1 .belief) ∧
  summary birthFitness = .error (.tickNetwork 0 0 .nonpositiveFitness) ∧
  summary birthTargetCount = .error (.tickNetwork 0 0 .targetCountMismatch) ∧
  summary birthTargetRange = .error (.tickNetwork 0 0 .targetOutOfRange) ∧
  summary birthTargetDuplicate = .error (.tickNetwork 0 0 .duplicateTarget) ∧
  summary birthAgentThreshold = .error (.tickAgent 0 0 .threshold) ∧
  summary lateBirth = .error (.tickNetwork 3 1 .targetOutOfRange) ∧
  summary firstFailure = .error (.tickAgent 0 0 .threshold) ∧
  summary lateFirstBirth = .error (.tickNetwork 2 0 .targetOutOfRange) :=
  ⟨seedNodes_literal, seedSize_literal, seedFitness_literal, seedEdge_literal, seedDuplicate_literal, seedDisconnected_literal, initialMZero_literal, initialMTooLarge_literal, seedAgentCount_literal, seedAgentR_literal, seedAgentThreshold_literal, seedAgentBelief_literal, birthFitness_literal, birthTargetCount_literal, birthTargetRange_literal, birthTargetDuplicate_literal, birthAgentThreshold_literal, lateBirth_literal, firstFailure_literal, lateFirstBirth_literal⟩

theorem transition_literals :
    (observeTransition successiveBirths ⟨1, by decide⟩).map
      (fun t => (t.postGrowth.beliefs, t.postGrowth.exposures, t.tickMass, t.transmissions)) =
      .ok ([1, 1, 0, 0], [0, 1, 0, 0], 1/4, [(0, 1, 1), (1, 0, 1), (1, 2, 1)]) ∧
    (observeTransition retainedExposures ⟨0, by decide⟩).map
      (fun t => t.postGrowth.exposures) = .ok [7, 3, 0] ∧
    (observeTransition broadcastNewborn ⟨0, by decide⟩).map
      (fun t => t.transmissions) = .ok [(0, 1, 1), (2, 1, 1)] :=
  ⟨successive_growth_literal, retained_growth_literal, newborn_transmissions_literal⟩

end FiniteLiterals

#print axioms NarrativeDynamics.Tests.FitnessABMRuntime.success_literals
#print axioms NarrativeDynamics.Tests.FitnessABMRuntime.shared_error_literals
#print axioms NarrativeDynamics.Tests.FitnessABMRuntime.transition_literals

end NarrativeDynamics.Tests.FitnessABMRuntime
