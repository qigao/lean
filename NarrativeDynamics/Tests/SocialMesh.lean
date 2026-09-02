import NarrativeDynamics.Core.SocialMesh

open NarrativeDynamics

example {Node : Type*} (g : MeshGraph Node) {a b c : Node}
    (hab : g a b) (hbc : g b c) : ReachWithin g 2 a c := by
  exact ⟨2, by omega, (MeshWalk.single hab).append (MeshWalk.single hbc)⟩

example {Node : Type*} (g : MeshGraph Node) (inside : Node → Prop)
    {n : Nat} {a b : Node} (walk : MeshWalk (restrictMesh g inside) n a b) :
    MeshWalk g n a b := by
  exact restricted_walk_lifts walk

example {Node : Type*} (g : MeshGraph Node) (inside : Node → Prop)
    (closed : ClosedSociety g inside) {limit : Nat} {a outsider : Node}
    (ha : inside a) (hout : ¬ inside outsider) :
    ¬ ReachWithin g limit a outsider := by
  exact @closed_no_cross_reach Node g inside closed limit a outsider ha hout
