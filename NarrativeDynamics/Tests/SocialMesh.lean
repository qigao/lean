import NarrativeDynamics.Core.SocialMesh

open NarrativeDynamics

example {Node : Type*} (g : MeshGraph Node) {a b c : Node}
    (hab : g a b) (hbc : g b c) : ReachWithin g 2 a c := by
  exact ⟨2, by omega, (MeshWalk.single hab).append (MeshWalk.single hbc)⟩
