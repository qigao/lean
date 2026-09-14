import NarrativeDynamics.Core.FitnessABMReplay

namespace NarrativeDynamics.Conformance.BBRuntime

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessABM

structure RuntimeCaseInput where
  id : String
  seed : FitnessAttachment.RawSeed
  m : Nat
  agents : Array FitnessABM.RawAgent
  ticks : List FitnessABM.RawTick

structure StateObservation where
  nodeCount : Nat
  edges : List (Nat × Nat)
  fitness : List Rat
  receptivity : List Rat
  thresholds : List Rat
  beliefs : List Rat
  exposures : List Nat
  broadcasting : List Bool
  deriving DecidableEq, Repr

structure PrefixObservation where
  state : StateObservation
  tickCount : Nat
  birthCount : Nat
  traceMass : Rat
  deriving DecidableEq, Repr

structure TransitionObservation where
  tickIndex : Nat
  birthIndex : Nat
  tickMass : Rat
  postGrowth : StateObservation
  transmissions : List (Nat × Nat × Rat)
  deriving DecidableEq, Repr

structure HistoryObservation where
  prefixes : List PrefixObservation
  transitions : List TransitionObservation
  deriving DecidableEq, Repr

def replayPrefix (input : RuntimeCaseInput) (count : Nat) :
    Except FitnessABM.JointError FitnessABM.Result :=
  FitnessABM.replay input.seed input.m input.agents (input.ticks.take count)

def unitSeed : RawSeed := ⟨2, #[1, 1], #[(0, 1)]⟩
def weightedSeed : RawSeed := ⟨2, #[1, 3], #[(0, 1)]⟩
def agentsRaw : Array RawAgent := #[⟨1, 1/2, 1, 0⟩, ⟨1, 1/2, 0, 0⟩]
def birth (target : Nat) : RawBirthInput := ⟨⟨1, #[target]⟩, 1, 1/2, 0⟩

def empty : RuntimeCaseInput := ⟨"empty", unitSeed, 1, agentsRaw, []⟩
def idleTwo : RuntimeCaseInput := { empty with id := "idle-two", ticks := [none, none] }
def attachSource : RuntimeCaseInput :=
  { empty with id := "attach-source", ticks := [some (birth 0)] }
def attachRelay : RuntimeCaseInput :=
  { empty with id := "attach-relay", ticks := [some (birth 1)] }
def relayIdle : RuntimeCaseInput :=
  { empty with id := "relay-idle", ticks := [some (birth 1), none] }
def successiveBirths : RuntimeCaseInput :=
  { empty with id := "successive-births", ticks := [some (birth 1), some (birth 2)] }
def successiveIdle : RuntimeCaseInput :=
  { empty with id := "successive-idle", ticks := [some (birth 1), some (birth 2), none] }
def weightedSource : RuntimeCaseInput :=
  { attachSource with id := "weighted-source", seed := weightedSeed }
def ordered01 : RuntimeCaseInput :=
  { empty with id := "ordered-01", seed := weightedSeed, m := 2,
    ticks := [some ⟨⟨1, #[0, 1]⟩, 1, 1/2, 0⟩] }
def ordered10 : RuntimeCaseInput :=
  { empty with id := "ordered-10", seed := weightedSeed, m := 2,
    ticks := [some ⟨⟨1, #[1, 0]⟩, 1, 1/2, 0⟩] }
def zeroReceptive : RuntimeCaseInput :=
  { attachSource with id := "zero-receptive",
    agents := #[⟨1, 1/2, 1, 0⟩, ⟨0, 1/2, 0, 0⟩] }
def silent : RuntimeCaseInput :=
  { attachSource with id := "silent", agents := #[⟨1, 1/2, 0, 0⟩, ⟨1, 1/2, 0, 0⟩] }
def zeroThreshold : RuntimeCaseInput :=
  { attachSource with id := "zero-threshold", agents := #[⟨1, 0, 0, 0⟩, ⟨1, 1/2, 0, 0⟩] }
def broadcastNewborn : RuntimeCaseInput :=
  { attachRelay with id := "broadcast-newborn", ticks := [some ⟨⟨1, #[1]⟩, 1, 1/2, 1⟩] }
def retainedExposures : RuntimeCaseInput :=
  { attachRelay with id := "retained-exposures",
    agents := #[⟨1, 1/2, 1, 7⟩, ⟨1, 1/2, 0, 3⟩] }
def halfReceptive : RuntimeCaseInput :=
  { attachSource with id := "half-receptive",
    agents := #[⟨1, 1/2, 1, 0⟩, ⟨1/2, 1/2, 0, 0⟩] }

def successInputs : List RuntimeCaseInput :=
  [empty, idleTwo, attachSource, attachRelay, relayIdle, successiveBirths,
    successiveIdle, weightedSource, ordered01, ordered10, zeroReceptive,
    silent, zeroThreshold, broadcastNewborn, retainedExposures, halfReceptive]

def seedNodes : RuntimeCaseInput :=
  { empty with id := "seed-nodes", seed := ⟨0, #[], #[]⟩, agents := #[⟨-1, 2, 2, 0⟩] }
def seedSize : RuntimeCaseInput :=
  { empty with id := "seed-size", seed := ⟨2, #[0], #[(0, 1)]⟩ }
def seedFitness : RuntimeCaseInput :=
  { empty with id := "seed-fitness", seed := ⟨2, #[0, 1], #[(0, 2)]⟩ }
def seedEdge : RuntimeCaseInput :=
  { empty with id := "seed-edge", seed := ⟨2, #[1, 1], #[(0, 1), (1, 0), (0, 2)]⟩ }
def seedDuplicate : RuntimeCaseInput :=
  { empty with id := "seed-duplicate", seed := ⟨2, #[1, 1], #[(0, 1), (1, 0)]⟩ }
def seedDisconnected : RuntimeCaseInput :=
  { empty with id := "seed-disconnected", seed := ⟨3, #[1, 1, 1], #[(0, 1)]⟩ }
def initialMZero : RuntimeCaseInput := { empty with id := "initial-m-zero", m := 0 }
def initialMTooLarge : RuntimeCaseInput :=
  { idleTwo with id := "initial-m-too-large", m := 3 }
def seedAgentCount : RuntimeCaseInput :=
  { empty with id := "seed-agent-count", agents := #[⟨1, 1/2, 1, 0⟩] }
def seedAgentR : RuntimeCaseInput :=
  { empty with id := "seed-agent-r", agents := #[⟨-1, 2, 1, 0⟩, ⟨1, 1/2, 0, 0⟩] }
def seedAgentThreshold : RuntimeCaseInput :=
  { empty with id := "seed-agent-threshold", agents := #[⟨1, 2, 2, 0⟩, ⟨1, 1/2, 0, 0⟩] }
def seedAgentBelief : RuntimeCaseInput :=
  { empty with id := "seed-agent-belief", agents := #[⟨1, 1/2, 1, 0⟩, ⟨1, 1/2, 2, 0⟩] }
def birthFitness : RuntimeCaseInput :=
  { empty with id := "birth-fitness", ticks := [some ⟨⟨0, #[0]⟩, 1, 1/2, 2⟩] }
def birthTargetCount : RuntimeCaseInput :=
  { empty with id := "birth-target-count", ticks := [some ⟨⟨1, #[0, 0]⟩, 1, 1/2, 0⟩] }
def birthTargetRange : RuntimeCaseInput :=
  { empty with id := "birth-target-range", ticks := [some (birth 2)] }
def birthTargetDuplicate : RuntimeCaseInput :=
  { birthTargetCount with id := "birth-target-duplicate", m := 2 }
def birthAgentThreshold : RuntimeCaseInput :=
  { empty with id := "birth-agent-threshold", ticks := [some ⟨⟨1, #[0]⟩, 1, 2, 2⟩] }
def lateBirth : RuntimeCaseInput :=
  { empty with id := "late-birth", ticks := [none, some (birth 1), none, some (birth 3)] }
def firstFailure : RuntimeCaseInput :=
  { empty with id := "first-failure",
    ticks := [some ⟨⟨1, #[0]⟩, 1, 2, 0⟩, some ⟨⟨0, #[0]⟩, 1, 1/2, 0⟩] }
def lateFirstBirth : RuntimeCaseInput :=
  { empty with id := "late-first-birth", ticks := [none, none, some (birth 2)] }

def errorInputs : List RuntimeCaseInput :=
  [seedNodes, seedSize, seedFitness, seedEdge, seedDuplicate, seedDisconnected,
    initialMZero, initialMTooLarge, seedAgentCount, seedAgentR, seedAgentThreshold,
    seedAgentBelief, birthFitness, birthTargetCount, birthTargetRange,
    birthTargetDuplicate, birthAgentThreshold, lateBirth, firstFailure, lateFirstBirth]

end NarrativeDynamics.Conformance.BBRuntime
