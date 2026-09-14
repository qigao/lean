import NarrativeDynamics.Core.FitnessABMReplay

namespace NarrativeDynamics.Tests.FitnessABMReplay

open NarrativeDynamics.FitnessABM

def seedRaw : FitnessAttachment.RawSeed := ⟨2, #[1, 1], #[(0, 1)]⟩
def agentsRaw : Array RawAgent := #[⟨1, 1/2, 1, 0⟩, ⟨1, 1/2, 0, 0⟩]
def attach0 : RawBirthInput := ⟨⟨1, #[0]⟩, 1, 1/2, 0⟩
def attach1 : RawBirthInput := ⟨⟨1, #[1]⟩, 1, 1/2, 0⟩

def summary (r : Except JointError Result) :
    Except JointError (Nat × Nat × Nat × List Rat × List Nat × Rat) :=
  r.map fun out =>
    (out.final.nodeCount, out.final.roundIndex,
     FitnessAttachment.actualEdgeCount out.final.state.network.snapshot,
     List.ofFn (fun i => (out.final.state.population.agents i).belief),
     List.ofFn (fun i => (out.final.state.population.agents i).exposures),
     out.probability)

section FiniteReplayFixtures
-- Match the established finite BB fixture budget; keep default heartbeats.
set_option maxRecDepth 4096

-- Catch propagation before birth or a newborn omitted from the first round.
example : summary (replay seedRaw 1 agentsRaw [some attach0]) =
    .ok (3, 1, 2, [1, 1, 1], [0, 1, 1], 1/2) := by decide_cbv

-- Catch in-place propagation: the newborn sees the previous belief at ID 1.
example : summary (replay seedRaw 1 agentsRaw [some attach1]) =
    .ok (3, 1, 2, [1, 1, 0], [0, 1, 0], 1/2) := by decide_cbv

-- Catch skipped idle rounds or a spurious probability factor for propagation.
example : summary (replay seedRaw 1 agentsRaw [some attach1, none]) =
    .ok (3, 2, 2, [1, 1, 1], [1, 2, 1], 1/2) := by decide_cbv

end FiniteReplayFixtures

end NarrativeDynamics.Tests.FitnessABMReplay
