import NarrativeDynamics.Core.FitnessAttachment

/-!
# Exact finite BB target-trace carrier

This module introduces only the finite executable carrier for ordered BB target
traces. Probability, state evolution, events, and expectations are added by later
V23.6 tasks.
-/

namespace NarrativeDynamics.FitnessAttachment

/-- A finite sequence of legal ordered target choices whose node type grows after
    each birth. The empty trace has one inhabitant. -/
def TargetTrace (n m : Nat) : Nat → Type
  | 0 => PUnit
  | steps + 1 => Targets n m × TargetTrace (n + 1) m steps

/-- Target traces have executable equality inherited recursively from target
    embeddings and products. -/
instance TargetTrace.instDecidableEq (n m steps : Nat) :
    DecidableEq (TargetTrace n m steps) := by
  induction steps generalizing n with
  | zero =>
      change DecidableEq PUnit
      infer_instance
  | succ steps ih =>
      change DecidableEq (Targets n m × TargetTrace (n + 1) m steps)
      letI : DecidableEq (TargetTrace (n + 1) m steps) := ih (n + 1)
      infer_instance

/-- Enumerate every target trace recursively from the existing executable target
    enumeration. No choice-based `Fintype.ofFinite` bridge is used. -/
instance TargetTrace.instFintype (n m steps : Nat) :
    Fintype (TargetTrace n m steps) := by
  induction steps generalizing n with
  | zero =>
      change Fintype PUnit
      infer_instance
  | succ steps ih =>
      change Fintype (Targets n m × TargetTrace (n + 1) m steps)
      letI : Fintype (TargetTrace (n + 1) m steps) := ih (n + 1)
      infer_instance

end NarrativeDynamics.FitnessAttachment
