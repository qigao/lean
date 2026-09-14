import NarrativeDynamics.Core.FitnessABMReplay

namespace NarrativeDynamics.Conformance.BBPropagation

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessABM

structure VectorInput where
  id : String
  seed : FitnessAttachment.RawSeed
  agents : Array FitnessABM.RawAgent

def attachSource : VectorInput :=
  ⟨"attach-source", ⟨3, #[1, 1, 1], #[(0, 1), (0, 2)]⟩,
    #[⟨1, 1/2, 1, 0⟩, ⟨1, 1/2, 0, 0⟩, ⟨1, 1/2, 0, 0⟩]⟩

def attachRelay : VectorInput :=
  ⟨"attach-relay", ⟨3, #[1, 1, 1], #[(0, 1), (1, 2)]⟩,
    #[⟨1, 1/2, 1, 0⟩, ⟨1, 1/2, 0, 0⟩, ⟨1, 1/2, 0, 0⟩]⟩

def relayNextRound : VectorInput :=
  ⟨"relay-next-round", ⟨3, #[1, 1, 1], #[(0, 1), (1, 2)]⟩,
    #[⟨1, 1/2, 1, 0⟩, ⟨1, 1/2, 1, 1⟩, ⟨1, 1/2, 0, 0⟩]⟩

def secondBirth : VectorInput :=
  ⟨"second-birth", ⟨4, #[1, 1, 1, 1], #[(0, 1), (1, 2), (2, 3)]⟩,
    #[⟨1, 1/2, 1, 0⟩, ⟨1, 1/2, 1, 1⟩, ⟨1, 1/2, 0, 0⟩,
      ⟨1, 1/2, 0, 0⟩]⟩

def halfReceptive : VectorInput :=
  ⟨"half-receptive", ⟨2, #[1, 1], #[(0, 1)]⟩,
    #[⟨1, 1/2, 1, 0⟩, ⟨1/2, 1/2, 0, 0⟩]⟩

def zeroReceptive : VectorInput :=
  ⟨"zero-receptive", ⟨2, #[1, 1], #[(0, 1)]⟩,
    #[⟨1, 1/2, 1, 0⟩, ⟨0, 1/2, 0, 0⟩]⟩

def silent : VectorInput :=
  ⟨"silent", ⟨2, #[1, 1], #[(0, 1)]⟩,
    #[⟨1, 1/2, 0, 0⟩, ⟨1, 1/2, 0, 0⟩]⟩

def zeroThreshold : VectorInput :=
  ⟨"zero-threshold", ⟨2, #[1, 1], #[(0, 1)]⟩,
    #[⟨1, 0, 0, 0⟩, ⟨1, 1/2, 0, 0⟩]⟩

def vectorInputs : List VectorInput :=
  [attachSource, attachRelay, relayNextRound, secondBirth,
    halfReceptive, zeroReceptive, silent, zeroThreshold]

private def ratJson (value : Rat) : Lean.Json :=
  .str (toString value)

private def natJson (value : Nat) : Lean.Json :=
  Lean.toJson value

private def pairJson (source target : Nat) : Lean.Json :=
  .arr #[natJson source, natJson target]

private def orderedPairsJson {n : Nat}
    (keepPair : Fin n → Fin n → Bool) : Lean.Json :=
  let ids : Array (Fin n) := Array.ofFn id
  .arr <| ids.flatMap fun source =>
    ids.filterMap fun target =>
      if keepPair source target then some (pairJson source.val target.val) else none

private def graphEdgesJson {n : Nat} (state : JointState n) : Lean.Json :=
  letI := state.network.snapshot.adjDec
  orderedPairsJson fun source target => decide
    (source < target ∧ state.network.snapshot.graph.Adj source target)

private def transmissionsJson {n : Nat} (state : JointState n) : Lean.Json :=
  letI := state.network.snapshot.adjDec
  let delivered := transmissions state.network.snapshot.graph.Adj state.population
  orderedPairsJson fun source target => decide ((source, target) ∈ delivered)

def renderCase (input : VectorInput) : Except String Lean.Json :=
  match FitnessABM.replay input.seed 1 input.agents [] with
  | .error error => .error s!"{input.id}: {reprStr error}"
  | .ok result =>
    let prior := result.final.state
    let next := advance prior
    .ok <| Lean.Json.mkObj [
      ("id", .str input.id),
      ("input", Lean.Json.mkObj [
        ("edges", graphEdgesJson prior),
        ("beliefs", .arr <| Array.ofFn fun i =>
          ratJson (prior.population.agents i).belief),
        ("exposures", .arr <| Array.ofFn fun i =>
          natJson (prior.population.agents i).exposures),
        ("receptivity", .arr <| Array.ofFn fun i =>
          ratJson (prior.population.profiles i).receptivity),
        ("thresholds", .arr <| Array.ofFn fun i =>
          ratJson (prior.population.profiles i).threshold)
      ]),
      ("expected", Lean.Json.mkObj [
        ("beliefs", .arr <| Array.ofFn fun i =>
          ratJson (next.population.agents i).belief),
        ("exposures", .arr <| Array.ofFn fun i =>
          natJson (next.population.agents i).exposures),
        ("broadcasting", .arr <| Array.ofFn fun i =>
          .bool (broadcasting (next.population.profiles i) (next.population.agents i))),
        ("transmissions", transmissionsJson prior)
      ])
    ]

def renderCorpus : Except String Lean.Json := do
  let vectors ← vectorInputs.mapM renderCase
  pure <| Lean.Json.mkObj [
    ("schema", .str "bb-abm-v1"),
    ("vectors", .arr vectors.toArray)
  ]

def main : IO Unit :=
  match renderCorpus with
  | .ok corpus => IO.println corpus.compress
  | .error error => throw (IO.userError error)

end NarrativeDynamics.Conformance.BBPropagation

def main : IO Unit :=
  NarrativeDynamics.Conformance.BBPropagation.main
