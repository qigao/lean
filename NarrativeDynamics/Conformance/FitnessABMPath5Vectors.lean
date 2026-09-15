import NarrativeDynamics.Core.FitnessABMReplay
import NarrativeDynamics.Core.FitnessABMPath5

/-!
# Generated exact-rational Path5 replay vectors

This conformance slice runs the real `FitnessABM.replay` runtime on a five-node
path for two idle rounds and compares every generated belief prefix against the
concrete Path5 `trajectory`. It emits the runtime values only after the two
implementations agree.
-/

namespace NarrativeDynamics.Conformance.FitnessABMPath5Vectors

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessABM
open NarrativeDynamics.FitnessABMPath5

def rawSeed : RawSeed :=
  ⟨5, #[1, 1, 1, 1, 1], #[(0, 1), (1, 2), (2, 3), (3, 4)]⟩

def rawAgents : Array RawAgent :=
  #[⟨1/2, 1/2, 1, 0⟩,
    ⟨1/2, 1/2, 3/4, 0⟩,
    ⟨1/2, 1/2, 1/2, 0⟩,
    ⟨1/2, 1/2, 3/4, 0⟩,
    ⟨1/2, 1/2, 1, 0⟩]

def ticks : List RawTick := [none, none]

def initialBeliefs : Beliefs := ![1, 3/4, 1/2, 3/4, 1]

def runtimeBeliefs (count : Nat) : Except JointError (List Rat) :=
  (FitnessABM.replay rawSeed 1 rawAgents (ticks.take count)).map fun result =>
    List.ofFn fun i => (result.final.state.population.agents i).belief

def modelBeliefs (count : Nat) : List Rat :=
  List.ofFn fun i => trajectory initialBeliefs count i

def checkedBeliefs (count : Nat) : Except String (List Rat) :=
  match runtimeBeliefs count with
  | .error error => .error s!"runtime replay failed at prefix {count}: {reprStr error}"
  | .ok runtime =>
      let model := modelBeliefs count
      if runtime = model then .ok runtime
      else .error s!"runtime/model mismatch at prefix {count}: runtime={runtime}, model={model}"

private def ratJson (value : Rat) : Lean.Json := .str (toString value)
private def beliefsJson (values : List Rat) : Lean.Json :=
  .arr (values.map ratJson).toArray

def render : Except String Lean.Json := do
  let zero ← checkedBeliefs 0
  let one ← checkedBeliefs 1
  let two ← checkedBeliefs 2
  pure <| Lean.Json.mkObj [
    ("schema", .str "bb-path5-runtime-v1"),
    ("beliefs", .arr #[beliefsJson zero, beliefsJson one, beliefsJson two])]

def main : IO Unit :=
  match render with
  | .ok corpus => IO.println corpus.compress
  | .error error => throw (IO.userError error)

end NarrativeDynamics.Conformance.FitnessABMPath5Vectors

def main : IO Unit := NarrativeDynamics.Conformance.FitnessABMPath5Vectors.main
