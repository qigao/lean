import NarrativeDynamics.Core.NetworkPropagation

namespace NarrativeDynamics.Tests.NetworkPropagation

open NarrativeDynamics.NetworkPropagation
open scoped BigOperators

def line3 : MeshGraph (Fin 3) := fun i j => i.val + 1 = j.val
instance : DecidableRel line3 := fun i j =>
  show Decidable (i.val + 1 = j.val) from inferInstance

def initial3 : Population 3 :=
  ⟨fun _ => ⟨1, 1/2⟩, ![⟨1, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩]⟩

-- A relay can only broadcast its newly acquired belief on the next round.
example : List.ofFn (fun i => ((propagate line3 initial3).agents i).belief) =
    [1, 1, 0] := by decide_cbv
example : List.ofFn (fun i =>
    ((propagate line3 (propagate line3 initial3)).agents i).belief) =
    [1, 1, 1] := by decide_cbv
example : transmissions line3 initial3 = {(0, 1)} := by decide_cbv

def halfReceptive : Population 3 :=
  { initial3 with profiles := fun i =>
      if i = 1 then ⟨1/2, 1/2⟩ else initial3.profiles i }

-- Exact convex averaging and the inclusive broadcasting threshold.
example : (nextAgent line3 halfReceptive 1).belief = 1/2 := by decide_cbv
example : (nextAgent line3 halfReceptive 1).exposures = 1 := by decide_cbv
example : broadcasting (halfReceptive.profiles 1)
    ((propagate line3 halfReceptive).agents 1) = true := by decide_cbv

def unreceptive : Population 3 :=
  { initial3 with profiles := fun i =>
      if i = 1 then ⟨0, 1/2⟩ else initial3.profiles i }

-- Receiving a signal increments exposure even when belief cannot change.
example : (nextAgent line3 unreceptive 1).belief = 0 := by decide_cbv
example : (nextAgent line3 unreceptive 1).exposures = 1 := by decide_cbv

def silent : Population 3 :=
  { initial3 with agents := fun _ => ⟨0, 0⟩ }

example : transmissions line3 silent = ∅ := by decide_cbv
example : ∀ i, (propagate line3 silent).agents i = silent.agents i := by
  intro i
  fin_cases i <;> decide_cbv

def zeroSource : Population 3 :=
  { initial3 with
    profiles := fun i => if i = 0 then ⟨1, 0⟩ else initial3.profiles i
    agents := fun _ => ⟨0, 0⟩ }

-- Threshold zero transmits a zero-valued signal; it is still an exposure.
example : transmissions line3 zeroSource = {(0, 1)} := by decide_cbv
example : nextAgent line3 zeroSource 1 = ⟨0, 1⟩ := by decide_cbv

def middleSource : Population 3 :=
  { initial3 with agents := ![⟨0, 0⟩, ⟨1, 0⟩, ⟨0, 0⟩] }

-- Mesh direction is source to receiver; there is no reverse edge.
example : transmissions line3 middleSource = {(1, 2)} := by decide_cbv
example : (1, 0) ∉ transmissions line3 middleSource := by decide_cbv

def twoSources : MeshGraph (Fin 3) := fun j i => j.val < 2 ∧ i = 2
instance : DecidableRel twoSources := fun j i =>
  show Decidable (j.val < 2 ∧ i = 2) from inferInstance

def mixedSignals : Population 3 :=
  { initial3 with agents := ![⟨1, 0⟩, ⟨1/2, 0⟩, ⟨0, 4⟩] }

-- Both incoming beliefs contribute to the mean and to the exposure count.
example : nextAgent twoSources mixedSignals 2 = ⟨3/4, 6⟩ := by decide_cbv

example {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) (hp : p.Valid) : (propagate g p).Valid :=
  propagate_valid g p hp

example {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) (j i : Fin n) :
    (j, i) ∈ transmissions g p ↔
      g j i ∧ broadcasting (p.profiles j) (p.agents j) = true :=
  transmission_iff g p j i

example {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) (i : Fin n) (h : incoming g p i = ∅) :
    nextAgent g p i = p.agents i := nextAgent_no_incoming g p i h

example {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) (i : Fin n) :
    (p.agents i).exposures ≤ (nextAgent g p i).exposures := exposures_mono g p i

example {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p q : Population n) (i : Fin n)
    (hstate : p.agents i = q.agents i)
    (hprofile : p.profiles i = q.profiles i)
    (hbelief : ∀ j, g j i → (p.agents j).belief = (q.agents j).belief)
    (hbroadcast : ∀ j, g j i →
      broadcasting (p.profiles j) (p.agents j) =
        broadcasting (q.profiles j) (q.agents j)) :
    nextAgent g p i = nextAgent g q i :=
  nextAgent_locality g p q i hstate hprofile hbelief hbroadcast

#print axioms NarrativeDynamics.NetworkPropagation.propagate_valid
#print axioms NarrativeDynamics.NetworkPropagation.transmission_iff
#print axioms NarrativeDynamics.NetworkPropagation.nextAgent_no_incoming
#print axioms NarrativeDynamics.NetworkPropagation.exposures_mono
#print axioms NarrativeDynamics.NetworkPropagation.nextAgent_locality

end NarrativeDynamics.Tests.NetworkPropagation
