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
  { empty with id := "ordered-01", seed := weightedSeed, m := 2, ticks := [some ⟨⟨1, #[0, 1]⟩, 1, 1/2, 0⟩] }
def ordered10 : RuntimeCaseInput :=
  { empty with id := "ordered-10", seed := weightedSeed, m := 2, ticks := [some ⟨⟨1, #[1, 0]⟩, 1, 1/2, 0⟩] }
def zeroReceptive : RuntimeCaseInput :=
  { attachSource with id := "zero-receptive", agents := #[⟨1, 1/2, 1, 0⟩, ⟨0, 1/2, 0, 0⟩] }
def silent : RuntimeCaseInput :=
  { attachSource with id := "silent", agents := #[⟨1, 1/2, 0, 0⟩, ⟨1, 1/2, 0, 0⟩] }
def zeroThreshold : RuntimeCaseInput :=
  { attachSource with id := "zero-threshold", agents := #[⟨1, 0, 0, 0⟩, ⟨1, 1/2, 0, 0⟩] }
def broadcastNewborn : RuntimeCaseInput :=
  { attachRelay with id := "broadcast-newborn", ticks := [some ⟨⟨1, #[1]⟩, 1, 1/2, 1⟩] }
def retainedExposures : RuntimeCaseInput :=
  { attachRelay with id := "retained-exposures", agents := #[⟨1, 1/2, 1, 7⟩, ⟨1, 1/2, 0, 3⟩] }
def halfReceptive : RuntimeCaseInput :=
  { attachSource with id := "half-receptive", agents := #[⟨1, 1/2, 1, 0⟩, ⟨1/2, 1/2, 0, 0⟩] }

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
  { empty with id := "first-failure", ticks := [some ⟨⟨1, #[0]⟩, 1, 2, 0⟩, some ⟨⟨0, #[0]⟩, 1, 1/2, 0⟩] }
def lateFirstBirth : RuntimeCaseInput :=
  { empty with id := "late-first-birth", ticks := [none, none, some (birth 2)] }

def errorInputs : List RuntimeCaseInput :=
  [seedNodes, seedSize, seedFitness, seedEdge, seedDuplicate, seedDisconnected,
    initialMZero, initialMTooLarge, seedAgentCount, seedAgentR, seedAgentThreshold,
    seedAgentBelief, birthFitness, birthTargetCount, birthTargetRange,
    birthTargetDuplicate, birthAgentThreshold, lateBirth, firstFailure, lateFirstBirth]

/-- Numeric ID order and actual adjacency, including every completed birth. -/
def observeState {n : Nat} (state : JointState n) : StateObservation :=
  letI := state.network.snapshot.adjDec
  let ids : List (Fin n) := List.ofFn id
  { nodeCount := n
    edges := ids.flatMap fun source => ids.filterMap fun target =>
      if source < target ∧ state.network.snapshot.graph.Adj source target then
        some (source.val, target.val) else none
    fitness := List.ofFn fun i => state.network.snapshot.fitness i
    receptivity := List.ofFn fun i => (state.population.profiles i).receptivity
    thresholds := List.ofFn fun i => (state.population.profiles i).threshold
    beliefs := List.ofFn fun i => (state.population.agents i).belief
    exposures := List.ofFn fun i => (state.population.agents i).exposures
    broadcasting := List.ofFn fun i =>
      NetworkPropagation.broadcasting (state.population.profiles i) (state.population.agents i) }

def observePrefix (input : RuntimeCaseInput) (count : Nat) :
    Except JointError PrefixObservation :=
  (replayPrefix input count).map fun result =>
    ⟨observeState result.final.state, result.final.roundIndex,
      (inputBirths (input.ticks.take count)).length, result.probability⟩

def observeTransmissions {n : Nat} (state : JointState n) :
    List (Nat × Nat × Rat) :=
  letI := state.network.snapshot.adjDec
  let delivered := NetworkPropagation.transmissions
    state.network.snapshot.graph.Adj state.population
  let ids : List (Fin n) := List.ofFn id
  ids.flatMap fun source => ids.filterMap fun target =>
    if (source, target) ∈ delivered then
      some (source.val, target.val, (state.population.agents source).belief) else none

/-- A real prior replay plus the current checked birth gives the pre-round state.
The successor prefix is computed independently by replay, never by a second formula. -/
def observeTransition (input : RuntimeCaseInput) (index : Fin input.ticks.length) :
    Except JointError TransitionObservation :=
  let birthIndex := (inputBirths (input.ticks.take index.val)).length
  match replayPrefix input index.val with
  | .error error => .error error
  | .ok prior =>
    match input.ticks[index.val] with
    | none => .ok ⟨index.val, birthIndex, 1,
        observeState prior.final.state, observeTransmissions prior.final.state⟩
    | some raw =>
      match checkedBirth prior.final.state input.m raw with
      | .error (.network error) => .error (.tickNetwork index.val birthIndex error)
      | .error (.agent field) => .error (.tickAgent index.val birthIndex field)
      | .ok next => .ok ⟨index.val, birthIndex, next.2,
          observeState next.1, observeTransmissions next.1⟩

/-- Validate the entire authored history before exposing any prefix observation. -/
def observeCase (input : RuntimeCaseInput) : Except JointError HistoryObservation := do
  let _ ← replayPrefix input input.ticks.length
  let prefixes ← (List.range (input.ticks.length + 1)).mapM (observePrefix input)
  let transitions ← (List.ofFn (fun i : Fin input.ticks.length => i)).mapM
    (observeTransition input)
  pure ⟨prefixes, transitions⟩

private def ratJson (value : Rat) : Lean.Json := .str (toString value)
private def natJson (value : Nat) : Lean.Json := Lean.toJson value
private def listJson {α : Type} (render : α → Lean.Json) (values : List α) : Lean.Json :=
  .arr (values.map render).toArray
private def pairJson (pair : Nat × Nat) : Lean.Json :=
  .arr #[natJson pair.1, natJson pair.2]

private def inputJson (input : RuntimeCaseInput) : Lean.Json :=
  Lean.Json.mkObj [
    ("seed", Lean.Json.mkObj [
      ("node_count", natJson input.seed.nodeCount),
      ("fitness", .arr (input.seed.fitness.map ratJson)),
      ("edges", .arr (input.seed.edges.map pairJson))]),
    ("m", natJson input.m),
    ("agents", .arr (input.agents.map fun a => Lean.Json.mkObj [
      ("receptivity", ratJson a.receptivity), ("threshold", ratJson a.threshold),
      ("belief", ratJson a.belief), ("exposures", natJson a.exposures)])),
    ("ticks", listJson (fun tick => match tick with
      | none => .null
      | some raw => Lean.Json.mkObj [
          ("fitness", ratJson raw.birth.fitness),
          ("targets", .arr (raw.birth.targets.map natJson)),
          ("receptivity", ratJson raw.receptivity),
          ("threshold", ratJson raw.threshold), ("belief", ratJson raw.belief)]) input.ticks)]

private def stateJson (state : StateObservation) : Lean.Json :=
  Lean.Json.mkObj [
    ("node_count", natJson state.nodeCount), ("edges", listJson pairJson state.edges),
    ("fitness", listJson ratJson state.fitness),
    ("receptivity", listJson ratJson state.receptivity),
    ("thresholds", listJson ratJson state.thresholds),
    ("beliefs", listJson ratJson state.beliefs),
    ("exposures", listJson natJson state.exposures),
    ("broadcasting", listJson Lean.Json.bool state.broadcasting)]

private def prefixJson (observation : PrefixObservation) : Lean.Json :=
  Lean.Json.mkObj [
    ("state", stateJson observation.state), ("tick_count", natJson observation.tickCount),
    ("birth_count", natJson observation.birthCount), ("trace_mass", ratJson observation.traceMass)]

private def transitionJson (transition : TransitionObservation) : Lean.Json :=
  Lean.Json.mkObj [
    ("tick_index", natJson transition.tickIndex),
    ("birth_index", natJson transition.birthIndex),
    ("tick_mass", ratJson transition.tickMass),
    ("post_growth", stateJson transition.postGrowth),
    ("transmissions", listJson (fun (source, target, signal) => Lean.Json.mkObj [
      ("source", natJson source), ("target", natJson target), ("signal", ratJson signal)])
        transition.transmissions)]

private def causeName : FitnessAttachment.Internal.Error → String
  | .negativeWeight => "negativeWeight"
  | .zeroMass => "zeroMass"
  | .invalidNodeCount => "invalidNodeCount"
  | .fitnessSizeMismatch => "fitnessSizeMismatch"
  | .nonpositiveFitness => "nonpositiveFitness"
  | .invalidEdge => "invalidEdge"
  | .duplicateEdge => "duplicateEdge"
  | .disconnectedSeed => "disconnectedSeed"
  | .invalidM => "invalidM"
  | .targetCountMismatch => "targetCountMismatch"
  | .targetOutOfRange => "targetOutOfRange"
  | .duplicateTarget => "duplicateTarget"

private def causeField : FitnessAttachment.Internal.Error → String
  | .invalidNodeCount => "node_count"
  | .negativeWeight | .fitnessSizeMismatch | .nonpositiveFitness => "fitness"
  | .invalidEdge | .duplicateEdge | .disconnectedSeed => "edges"
  | .invalidM => "m"
  | .zeroMass | .targetCountMismatch | .targetOutOfRange | .duplicateTarget => "targets"

private def agentField : AgentField → String
  | .receptivity => "receptivity"
  | .threshold => "broadcast_threshold"
  | .belief => "belief"

private def errorObject (stage code field : String)
    (agentIndex tickIndex birthIndex : Option Nat := none)
    (cause : Option String := none) (expected actual : Option Nat := none) : Lean.Json :=
  let optionalNat := fun (value : Option Nat) => value.elim Lean.Json.null natJson
  Lean.Json.mkObj [
    ("stage", .str stage), ("code", .str code), ("field", .str field),
    ("agent_index", optionalNat agentIndex), ("tick_index", optionalNat tickIndex),
    ("birth_index", optionalNat birthIndex), ("bb_cause", cause.elim .null Lean.Json.str),
    ("expected", optionalNat expected), ("actual", optionalNat actual)]

private def errorJson : JointError → Lean.Json
  | .seedNetwork cause => errorObject "seed_network" (causeName cause) (causeField cause)
      none none none (some (causeName cause))
  | .initialM => errorObject "initial_m" "initialM" "m"
  | .seedAgentCount expected actual => errorObject "seed_agents" "seedAgentCount" "agents"
      none none none none (some expected) (some actual)
  | .seedAgent index field => errorObject "seed_agent" "invalidAgentValue" (agentField field)
      (some index)
  | .tickNetwork tickIndex birthIndex cause =>
      errorObject "tick_network" (causeName cause) (causeField cause)
        none (some tickIndex) (some birthIndex) (some (causeName cause))
  | .tickAgent tickIndex birthIndex field =>
      errorObject "tick_agent" "invalidAgentValue" (agentField field)
        none (some tickIndex) (some birthIndex)

private def renderSuccess (input : RuntimeCaseInput) : Except String Lean.Json :=
  match observeCase input with
  | .error error => .error s!"{input.id}: unexpected replay error {reprStr error}"
  | .ok history => .ok <| Lean.Json.mkObj [
      ("id", .str input.id), ("input", inputJson input),
      ("prefixes", listJson prefixJson history.prefixes),
      ("transitions", listJson transitionJson history.transitions)]

private def renderError (input : RuntimeCaseInput) : Except String Lean.Json :=
  match observeCase input with
  | .ok _ => .error s!"{input.id}: unexpected replay success"
  | .error error => .ok <| Lean.Json.mkObj [
      ("id", .str input.id), ("input", inputJson input), ("expected_error", errorJson error)]

def renderCorpus : Except String Lean.Json := do
  let success ← successInputs.mapM renderSuccess
  let errors ← errorInputs.mapM renderError
  pure <| Lean.Json.mkObj [
    ("schema", .str "bb-abm-runtime-v1"),
    ("success", .arr success.toArray), ("errors", .arr errors.toArray)]

def main : IO Unit :=
  match renderCorpus with
  | .ok corpus => IO.println corpus.compress
  | .error error => throw (IO.userError error)

end NarrativeDynamics.Conformance.BBRuntime

def main : IO Unit := NarrativeDynamics.Conformance.BBRuntime.main
