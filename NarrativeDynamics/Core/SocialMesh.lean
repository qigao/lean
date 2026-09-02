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

/-- The graph visible inside one society retains only internal edges. -/
def restrictMesh {Node : Type*} (g : MeshGraph Node) (inside : Node → Prop) :
    MeshGraph Node :=
  fun source target => inside source ∧ inside target ∧ g source target

/-- A society is closed when no active mesh edge leaves it. -/
def ClosedSociety {Node : Type*} (g : MeshGraph Node) (inside : Node → Prop) : Prop :=
  ∀ {source target}, inside source → g source target → inside target

/-- Every walk in a local projection is also a walk in the global mesh. -/
theorem restricted_walk_lifts {Node : Type*} {g : MeshGraph Node}
    {inside : Node → Prop} {length : Nat} {source target : Node}
    (walk : MeshWalk (restrictMesh g inside) length source target) :
    MeshWalk g length source target := by
  induction walk with
  | refl => exact MeshWalk.refl _
  | step edge rest ih => exact MeshWalk.step edge.2.2 ih

/-- A walk beginning inside a closed society also ends inside it. -/
theorem closed_walk_target_inside {Node : Type*} {g : MeshGraph Node}
    {inside : Node → Prop} (closed : ClosedSociety g inside)
    {length : Nat} {source target : Node} (sourceInside : inside source)
    (walk : MeshWalk g length source target) : inside target := by
  induction walk with
  | refl => exact sourceInside
  | step edge rest ih => exact ih (closed sourceInside edge)

/-- A global walk beginning inside a closed society can be reconstructed in
that society's local projection. -/
theorem closed_walk_restricts {Node : Type*} {g : MeshGraph Node}
    {inside : Node → Prop} (closed : ClosedSociety g inside)
    {length : Nat} {source target : Node} (sourceInside : inside source)
    (walk : MeshWalk g length source target) :
    MeshWalk (restrictMesh g inside) length source target := by
  induction walk with
  | refl => exact MeshWalk.refl _
  | step edge rest ih =>
      have nextInside := closed sourceInside edge
      exact MeshWalk.step ⟨sourceInside, nextInside, edge⟩ (ih nextInside)

/-- No bounded path can cross from a closed society to an outside node. -/
theorem closed_no_cross_reach {Node : Type*} {g : MeshGraph Node}
    {inside : Node → Prop} (closed : ClosedSociety g inside)
    {limit : Nat} (source outsider : Node) (sourceInside : inside source)
    (outsiderNotInside : ¬ inside outsider) :
    ¬ ReachWithin g limit source outsider := by
  intro reachable
  rcases reachable with ⟨_, _, walk⟩
  exact outsiderNotInside
    (@closed_walk_target_inside Node g inside closed _ source outsider sourceInside walk)

/-- Bounded paths on both sides of one explicit bridge compose into a bounded
global path. -/
theorem reachWithin_bridge {Node : Type*} {g : MeshGraph Node}
    {leftLimit rightLimit : Nat} {source leftGate rightGate target : Node}
    (left : ReachWithin g leftLimit source leftGate)
    (bridge : g leftGate rightGate)
    (right : ReachWithin g rightLimit rightGate target) :
    ReachWithin g (leftLimit + 1 + rightLimit) source target := by
  rcases left with ⟨leftLength, leftBound, leftWalk⟩
  rcases right with ⟨rightLength, rightBound, rightWalk⟩
  refine ⟨leftLength + 1 + rightLength, by omega, ?_⟩
  exact (leftWalk.append (MeshWalk.single bridge)).append rightWalk

/-- A two-hop local path, one bridge, and a three-hop local path yield the
conditional six-degree reachability bound. -/
theorem six_degrees_of_two_three_bridge {Node : Type*} {g : MeshGraph Node}
    {source leftGate rightGate target : Node}
    (left : ReachWithin g 2 source leftGate)
    (bridge : g leftGate rightGate)
    (right : ReachWithin g 3 rightGate target) : ReachWithin g 6 source target := by
  have composed : ReachWithin g (2 + 1 + 3) source target :=
    reachWithin_bridge left bridge right
  simpa using composed

end NarrativeDynamics
