import NarrativeDynamics.Core.FitnessBirth

open NarrativeDynamics NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal
open scoped BigOperators

namespace NarrativeDynamics.FitnessAttachment.BirthFixtures

private theorem triangle_connected : (⊤ : SimpleGraph (Fin 3)).Connected where
  preconnected := by
    intro u v
    by_cases h : u = v
    · subst v; exact ⟨.nil⟩
    · exact ⟨.cons h .nil⟩
  nonempty := inferInstance

def triangle : State 3 where
  snapshot := { graph := ⊤, adjDec := inferInstance, fitness := ![1, 2, 4] }
  valid := ⟨by decide, triangle_connected, by decide⟩

def t21 : Targets 3 2 := ⟨![2, 1], by decide⟩
def t12 : Targets 3 2 := ⟨![1, 2], by decide⟩
def t0 : Targets 3 1 := ⟨![0], by decide⟩
def tAll : Targets 3 3 := Function.Embedding.refl _

def after21 := applyBirth triangle t21 (by decide) ⟨3/2, by norm_num⟩
def after12 := applyBirth triangle t12 (by decide) ⟨3/2, by norm_num⟩
def afterHigh := applyBirth triangle t21 (by decide) ⟨3, by norm_num⟩
def afterOne := applyBirth triangle t0 (by decide) ⟨1, by norm_num⟩
def afterAll := applyBirth triangle tAll (by decide) ⟨1, by norm_num⟩

end NarrativeDynamics.FitnessAttachment.BirthFixtures

section FiniteBirthFixtures
-- Local equation-evaluation budget, not a budget on generic structural proofs.
set_option maxRecDepth 4096

example : List.ofFn (oldId 3) = [0, 1, 2] := by decide_cbv
example : newId 3 = (3 : Fin 4) := by decide
example : actualNodeCount BirthFixtures.after21.snapshot = 4 := by decide_cbv
example : actualEdgeCount BirthFixtures.triangle.snapshot = 3 := by decide_cbv
example : actualEdgeCount BirthFixtures.after21.snapshot = 5 := by decide_cbv
theorem birth_degree_fixture : List.ofFn (degree BirthFixtures.after21.snapshot) =
    [2, 3, 3, 2] := by decide_cbv
example : List.ofFn BirthFixtures.after21.snapshot.fitness = [1, 2, 4, 3/2] := by
  decide_cbv
example : ∀ i j : Fin 4, BirthFixtures.after21.snapshot.graph.Adj i j ↔
    BirthFixtures.after12.snapshot.graph.Adj i j := by decide_cbv
example : List.ofFn BirthFixtures.after21.snapshot.fitness =
    List.ofFn BirthFixtures.after12.snapshot.fitness := by decide_cbv
example : ∀ i j : Fin 4, BirthFixtures.after21.snapshot.graph.Adj i j ↔
    BirthFixtures.afterHigh.snapshot.graph.Adj i j := by decide_cbv
example : List.ofFn (degree BirthFixtures.afterOne.snapshot) = [3, 2, 2, 1] := by
  decide_cbv
example : actualEdgeCount BirthFixtures.afterOne.snapshot = 4 := by decide_cbv
example : List.ofFn (degree BirthFixtures.afterAll.snapshot) = [3, 3, 3, 3] := by
  decide_cbv
example : actualEdgeCount BirthFixtures.afterAll.snapshot = 6 := by decide_cbv
-- Newborn fitness affects the NEXT row, not which edges this birth adds.
example : (attachmentRow BirthFixtures.after21 ∅ (by decide)).mass 3 = 3/23 := by
  decide_cbv
example : (attachmentRow BirthFixtures.afterHigh ∅ (by decide)).mass 3 = 3/13 := by
  decide_cbv

end FiniteBirthFixtures

-- All adjacency cases and identity laws are generic, not just count fixtures.
example {n m : Nat} (s : State n) (T : Targets n m) (hm : 0 < m)
    (eta : PosFitness) (u v : Fin n) :
    (applyBirth s T hm eta).snapshot.graph.Adj (oldId n u) (oldId n v) ↔
      s.snapshot.graph.Adj u v := birth_old_adj_iff s T hm eta u v
example {n m : Nat} (s : State n) (T : Targets n m) (hm : 0 < m)
    (eta : PosFitness) (u : Fin n) :
    (applyBirth s T hm eta).snapshot.graph.Adj (oldId n u) (newId n) ↔
      u ∈ T.selected := birth_new_adj_iff s T hm eta u
example {n m : Nat} (s : State n) (T : Targets n m) (hm : 0 < m)
    (eta : PosFitness) (u : Fin n) :
    (applyBirth s T hm eta).snapshot.graph.Adj (newId n) (oldId n u) ↔
      u ∈ T.selected := birth_new_adj_iff_rev s T hm eta u
example {n m : Nat} (s : State n) (T : Targets n m) (hm : 0 < m)
    (eta : PosFitness) :
    ¬ (applyBirth s T hm eta).snapshot.graph.Adj (newId n) (newId n) :=
  (applyBirth s T hm eta).snapshot.graph.loopless.irrefl _
example {n m : Nat} (s : State n) (T : Targets n m) (hm : 0 < m)
    (eta : PosFitness) (u : Fin n) :
    (applyBirth s T hm eta).snapshot.fitness (oldId n u) = s.snapshot.fitness u :=
  birth_fitness_old s T hm eta u
example {n m : Nat} (s : State n) (T : Targets n m) (hm : 0 < m)
    (eta : PosFitness) :
    (applyBirth s T hm eta).snapshot.fitness (newId n) = eta.val :=
  birth_fitness_new s T hm eta
example {n m : Nat} (s : State n) (T : Targets n m) (hm : 0 < m)
    (eta : PosFitness) :
    actualNodeCount (applyBirth s T hm eta).snapshot = actualNodeCount s.snapshot + 1 :=
  birth_nodes s T hm eta
example {n m : Nat} (s : State n) (T : Targets n m) (hm : 0 < m)
    (eta : PosFitness) :
    actualEdgeCount (applyBirth s T hm eta).snapshot = actualEdgeCount s.snapshot + m :=
  birth_edges s T hm eta
example {n m : Nat} (s : State n) (T : Targets n m) (hm : 0 < m)
    (eta : PosFitness) (u : Fin n) :
    degree (applyBirth s T hm eta).snapshot (oldId n u) =
      degree s.snapshot u + if u ∈ T.selected then 1 else 0 :=
  birth_degree_old s T hm eta u
example {n m : Nat} (s : State n) (T : Targets n m) (hm : 0 < m)
    (eta : PosFitness) : degree (applyBirth s T hm eta).snapshot (newId n) = m :=
  birth_degree_new s T hm eta
example {n m : Nat} (s : State n) (T : Targets n m) (hm : 0 < m)
    (eta : PosFitness) :
    (∑ i, degree (applyBirth s T hm eta).snapshot i) =
      (∑ i, degree s.snapshot i) + 2*m := birth_degree_sum s T hm eta
example {n m : Nat} (s : State n) (T : Targets n m) (hm : 0 < m)
    (eta : PosFitness) : (applyBirth s T hm eta).snapshot.graph.Connected :=
  birth_connected s T hm eta
example {n m : Nat} (s : State n) (T U : Targets n m) (hm : 0 < m)
    (eta : PosFitness) (same : T.selected = U.selected) :
    applyBirth s T hm eta = applyBirth s U hm eta :=
  birth_order_irrelevant s T U hm eta same
example {n m : Nat} (s : State n) (T : Targets n m) (hm : 0 < m)
    (eta eta' : PosFitness) :
    (applyBirth s T hm eta).snapshot.graph = (applyBirth s T hm eta').snapshot.graph :=
  birth_new_fitness_independent s T hm eta eta'
example {n m k : Nat} (s : State n) (T : Targets n m) (hm : 0 < m)
    (eta : PosFitness) {u v : Fin n} (walk : MeshWalk s.snapshot.graph.Adj k u v) :
    MeshWalk (applyBirth s T hm eta).snapshot.graph.Adj k (oldId n u) (oldId n v) :=
  birth_walk_lifts s T hm eta walk
example {n m k : Nat} (s : State n) (T : Targets n m) (hm : 0 < m)
    (eta : PosFitness) {u v : Fin n} (path : ReachWithin s.snapshot.graph.Adj k u v) :
    ReachWithin (applyBirth s T hm eta).snapshot.graph.Adj k (oldId n u) (oldId n v) :=
  birth_reachWithin_lifts s T hm eta path
example {n : Nat} (s : Snapshot n) :
    Function.Bijective (edgePairToUnordered s) := edgePairToUnordered_bijective s

#print axioms birth_old_adj_iff
#print axioms birth_new_adj_iff
#print axioms birth_fitness_old
#print axioms birth_nodes
#print axioms birth_edges
#print axioms birth_degree_old
#print axioms birth_degree_new
#print axioms birth_degree_sum
#print axioms birth_connected
#print axioms birth_order_irrelevant
#print axioms birth_new_fitness_independent
#print axioms birth_walk_lifts
#print axioms birth_reachWithin_lifts
#print axioms edgePairToUnordered_bijective
#print axioms birth_degree_fixture
