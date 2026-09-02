import Mathlib

namespace NarrativeDynamics

/-- A directed social-mesh snapshot. Time-dependent models select one such
relation at a logical instant. -/
abbrev MeshGraph (Node : Type*) := Node → Node → Prop

/-- An exact-length directed walk through a mesh snapshot. -/
inductive MeshWalk {Node : Type*} (g : MeshGraph Node) : Nat → Node → Node → Prop where
  | refl (node : Node) : MeshWalk g 0 node node
  | step {n : Nat} {source next target : Node} :
      g source next → MeshWalk g n next target → MeshWalk g (Nat.succ n) source target

/-- One mesh edge is a walk of length one. -/
theorem MeshWalk.single {Node : Type*} {g : MeshGraph Node} {source target : Node}
    (edge : g source target) : MeshWalk g 1 source target := by
  simpa using MeshWalk.step edge (MeshWalk.refl target)

/-- Exact-length mesh walks compose and their lengths add. -/
theorem MeshWalk.append {Node : Type*} {g : MeshGraph Node}
    {leftLength rightLength : Nat} {source middle target : Node}
    (left : MeshWalk g leftLength source middle)
    (right : MeshWalk g rightLength middle target) :
    MeshWalk g (leftLength + rightLength) source target := by
  induction left with
  | refl => simpa using right
  | step edge rest ih =>
      simpa [Nat.succ_add] using MeshWalk.step edge (ih right)

/-- Reachability under a finite inclusive hop budget. -/
def ReachWithin {Node : Type*} (g : MeshGraph Node) (limit : Nat)
    (source target : Node) : Prop :=
  ∃ length, length ≤ limit ∧ MeshWalk g length source target

/-- Increasing a hop budget cannot destroy an existing bounded path. -/
theorem reachWithin_mono {Node : Type*} {g : MeshGraph Node}
    {smaller larger : Nat} {source target : Node}
    (reachable : ReachWithin g smaller source target)
    (largerBound : smaller ≤ larger) : ReachWithin g larger source target := by
  rcases reachable with ⟨length, bound, walk⟩
  exact ⟨length, Nat.le_trans bound largerBound, walk⟩

end NarrativeDynamics
