import NarrativeDynamics.Core.FitnessAttachment

/-!
# Atomic birth in the finite BB profile

A validated target embedding adds one fresh last vertex and its incident edges.
The probability layer is unchanged. Counts and degrees are derived from actual
adjacency, and all old fitness values and old IDs are retained exactly.
-/

namespace NarrativeDynamics.FitnessAttachment

open Internal
open scoped BigOperators

/-- Existing finite IDs keep their numeric value in the extended carrier. -/
def oldId (n : Nat) : Fin n ↪ Fin (n + 1) :=
  ⟨Fin.castSucc, fun _ _ h => Fin.ext (congrArg (fun x : Fin (n + 1) => x.val) h)⟩

/-- The sole newborn ID is distinct from every embedded old ID. -/
def newId (n : Nat) : Fin (n + 1) := Fin.last n

namespace Internal

/-- Four explicit adjacency cases, with no independently stored edge counter. -/
def birthAdj {n : Nat} (s : Snapshot n) (A : Finset (Fin n)) :
    Fin (n + 1) → Fin (n + 1) → Prop :=
  Fin.lastCases (Fin.lastCases False (fun v => v ∈ A))
    (fun u => Fin.lastCases (u ∈ A) (fun v => s.graph.Adj u v))

/-- An undirected simple graph constructed from the old graph and exact targets. -/
def birthGraph {n : Nat} (s : Snapshot n) (A : Finset (Fin n)) :
    SimpleGraph (Fin (n + 1)) where
  Adj := birthAdj s A
  symm := ⟨by
    intro u v
    refine Fin.lastCases ?_ (fun i => ?_) u
    · refine Fin.lastCases ?_ (fun j => ?_) v <;> simp [birthAdj]
    · refine Fin.lastCases ?_ (fun j => ?_) v
      · simp [birthAdj]
      · simpa [birthAdj] using s.graph.symm.symm i j⟩
  loopless := ⟨by
    intro u
    refine Fin.lastCases ?_ (fun i => ?_) u
    · simp [birthAdj]
    · simpa only [birthAdj, Fin.lastCases_castSucc] using s.graph.loopless.irrefl i⟩

/-- Decidable adjacency is constructed by finite cases, not classical choice. -/
@[reducible] def birthAdjDec {n : Nat} (s : Snapshot n) (A : Finset (Fin n)) :
    DecidableRel (birthGraph s A).Adj := by
  letI := s.adjDec
  intro u v
  cases u using Fin.lastCases <;> cases v using Fin.lastCases <;>
    simp only [birthGraph, birthAdj, Fin.lastCases_last, Fin.lastCases_castSucc] <;>
    infer_instance

/-- New fitness is appended; no old fitness is recomputed or rescaled. -/
def birthSnapshot {n : Nat} (s : Snapshot n) (A : Finset (Fin n))
    (eta : PosFitness) : Snapshot (n + 1) where
  graph := birthGraph s A
  adjDec := birthAdjDec s A
  fitness := Fin.lastCases eta.val s.fitness

/-- A nonempty target set joins the newborn to the connected old graph. -/
theorem birthGraph_connected {n : Nat} (s : State n) (A : Finset (Fin n))
    (hA : A.Nonempty) : (birthGraph s.snapshot A).Connected := by
  obtain ⟨a, ha⟩ := hA
  let f : s.snapshot.graph →g birthGraph s.snapshot A :=
    ⟨Fin.castSucc, by
      intro u v h
      simpa [birthGraph, birthAdj] using h⟩
  have oldReach (u v : Fin n) :
      (birthGraph s.snapshot A).Reachable u.castSucc v.castSucc := by
    obtain ⟨p⟩ := s.valid.2.1 u v
    exact ⟨p.map f⟩
  have toAnchor (v : Fin (n + 1)) :
      (birthGraph s.snapshot A).Reachable v a.castSucc := by
    refine Fin.lastCases ?_ (fun u => oldReach u a) v
    exact ⟨.cons (by simpa [birthGraph, birthAdj] using ha) .nil⟩
  exact { preconnected := fun u v => (toAnchor u).trans (toAnchor v).symm
          nonempty := inferInstance }

end Internal

/-- A complete typed birth: validity of the successor is derived, never supplied. -/
def applyBirth {n m : Nat} (s : State n) (T : Targets n m) (hm : 0 < m)
    (eta : PosFitness) : State (n + 1) where
  snapshot := birthSnapshot s.snapshot T.selected eta
  valid := ⟨by have hn := s.valid.1; omega,
    birthGraph_connected s T.selected (Finset.card_pos.mp (by
      rw [selected_card]; exact hm)), by
      intro v
      refine Fin.lastCases ?_ (fun u => ?_) v
      · simpa [birthSnapshot] using eta.property
      · simpa [birthSnapshot] using s.valid.2.2 u⟩

@[simp] theorem birth_old_adj_iff {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) (u v : Fin n) :
    (applyBirth s T hm eta).snapshot.graph.Adj (oldId n u) (oldId n v) ↔
      s.snapshot.graph.Adj u v := by
  simp [applyBirth, birthSnapshot, birthGraph, birthAdj, oldId]

@[simp] theorem birth_new_adj_iff {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) (u : Fin n) :
    (applyBirth s T hm eta).snapshot.graph.Adj (oldId n u) (newId n) ↔
      u ∈ T.selected := by
  simp [applyBirth, birthSnapshot, birthGraph, birthAdj, oldId, newId]

@[simp] theorem birth_new_adj_iff_rev {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) (u : Fin n) :
    (applyBirth s T hm eta).snapshot.graph.Adj (newId n) (oldId n u) ↔
      u ∈ T.selected := by
  simp [applyBirth, birthSnapshot, birthGraph, birthAdj, oldId, newId]

@[simp] theorem birth_fitness_old {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) (u : Fin n) :
    (applyBirth s T hm eta).snapshot.fitness (oldId n u) = s.snapshot.fitness u := by
  simp [applyBirth, birthSnapshot, oldId]

@[simp] theorem birth_fitness_new {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) :
    (applyBirth s T hm eta).snapshot.fitness (newId n) = eta.val := by
  simp [applyBirth, birthSnapshot, newId]

/-- Stable-ID fitness values are extended by exactly the newborn value. -/
theorem birth_fitness_list {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) :
    List.ofFn (applyBirth s T hm eta).snapshot.fitness =
      List.ofFn s.snapshot.fitness ++ [eta.val] := by
  rw [List.ofFn_succ']
  simp [applyBirth, birthSnapshot, List.concat_eq_append]

/-- Count the actual finite vertex carrier. -/
def actualNodeCount {n : Nat} (_s : Snapshot n) : Nat := Fintype.card (Fin n)

namespace Internal

/-- Each actual undirected edge has exactly one increasing-endpoint representative. -/
def edgePairs {n : Nat} (s : Snapshot n) : Finset (Fin n × Fin n) :=
  letI := s.adjDec
  Finset.univ.filter (fun p => p.1 < p.2 ∧ s.graph.Adj p.1 p.2)

@[simp] theorem mem_edgePairs {n : Nat} (s : Snapshot n) (p : Fin n × Fin n) :
    p ∈ edgePairs s ↔ p.1 < p.2 ∧ s.graph.Adj p.1 p.2 := by
  simp [edgePairs]

/-- No edge orientation or parallel counter is used as a second source of truth. -/
def edgePairToUnordered {n : Nat} (s : Snapshot n) : ↥(edgePairs s) → s.graph.edgeSet :=
  fun p => ⟨s(p.val.1, p.val.2), ((mem_edgePairs s p.val).mp p.property).2⟩

/-- Increasing representatives are in bijection with the actual unordered edges. -/
theorem edgePairToUnordered_bijective {n : Nat} (s : Snapshot n) :
    Function.Bijective (edgePairToUnordered s) := by
  constructor
  · intro a b hab
    have h := congrArg Subtype.val hab
    change s(a.val.1, a.val.2) = s(b.val.1, b.val.2) at h
    rcases Sym2.eq_iff.mp h with h | h
    · apply Subtype.ext
      exact Prod.ext h.1 h.2
    · have ha := ((mem_edgePairs s a.val).mp a.property).1
      have hb := ((mem_edgePairs s b.val).mp b.property).1
      rw [h.1, h.2] at ha
      exact False.elim (lt_asymm ha hb)
  · rintro ⟨e, he⟩
    revert he
    refine Sym2.inductionOn e ?_
    intro u v huv
    have hadj : s.graph.Adj u v := huv
    by_cases hlt : u < v
    · refine ⟨⟨(u, v), (mem_edgePairs s _).mpr ⟨hlt, hadj⟩⟩, ?_⟩
      rfl
    · have hne : u ≠ v := by
        intro heq
        subst v
        exact s.graph.loopless.irrefl u hadj
      have hgt : v < u := lt_of_le_of_ne (le_of_not_gt hlt) (Ne.symm hne)
      refine ⟨⟨(v, u), (mem_edgePairs s _).mpr
        ⟨hgt, s.graph.symm.symm u v hadj⟩⟩, ?_⟩
      apply Subtype.ext
      exact Sym2.eq_swap

end Internal

/-- Count actual adjacent increasing pairs, not a recurrence field in State. -/
def actualEdgeCount {n : Nat} (s : Snapshot n) : Nat := (edgePairs s).card

/-- The executable pair count agrees with the graph's unordered edge cardinality. -/
theorem actualEdgeCount_eq_edgeSet_card {n : Nat} (s : Snapshot n) :
    letI := s.adjDec
    actualEdgeCount s = Fintype.card s.graph.edgeSet := by
  letI := s.adjDec
  change (edgePairs s).card = _
  simpa only [Fintype.card_coe] using Fintype.card_congr
    (Equiv.ofBijective (edgePairToUnordered s) (edgePairToUnordered_bijective s))

namespace Internal

private theorem degree_sum_indicator {n : Nat} (s : Snapshot n) (u : Fin n) :
    letI := s.adjDec
    degree s u = ∑ v, if s.graph.Adj u v then 1 else 0 := by
  letI := s.adjDec
  simp only [degree, NarrativeDynamics.neighborSet, Finset.card_filter]

private theorem edgeCount_sum_indicator {n : Nat} (s : Snapshot n) :
    letI := s.adjDec
    actualEdgeCount s = ∑ u, ∑ v, if u < v ∧ s.graph.Adj u v then 1 else 0 := by
  letI := s.adjDec
  simp only [actualEdgeCount, edgePairs, Finset.card_filter, Fintype.sum_prod_type]

private theorem target_sum {n m : Nat} (T : Targets n m) :
    (∑ u : Fin n, if u ∈ T.selected then 1 else 0 : Nat) = m := by
  simpa using selected_card T

end Internal

/-- Exactly one actual vertex is added. -/
theorem birth_nodes {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) :
    actualNodeCount (applyBirth s T hm eta).snapshot = actualNodeCount s.snapshot + 1 := by
  simp [actualNodeCount]

/-- Old neighbor sets gain the newborn exactly for selected vertices. -/
theorem birth_degree_old {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) (u : Fin n) :
    degree (applyBirth s T hm eta).snapshot (oldId n u) =
      degree s.snapshot u + if u ∈ T.selected then 1 else 0 := by
  letI := s.snapshot.adjDec
  letI := (applyBirth s T hm eta).snapshot.adjDec
  rw [degree_sum_indicator, Fin.sum_univ_castSucc, degree_sum_indicator]
  simp [applyBirth, birthSnapshot, birthGraph, birthAdj, oldId]

/-- Injectivity of the target embedding gives newborn degree exactly m. -/
theorem birth_degree_new {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) :
    degree (applyBirth s T hm eta).snapshot (newId n) = m := by
  letI := (applyBirth s T hm eta).snapshot.adjDec
  rw [degree_sum_indicator, Fin.sum_univ_castSucc]
  simpa [applyBirth, birthSnapshot, birthGraph, birthAdj, newId] using target_sum T

/-- Split the actual increasing edge pairs into old pairs and newborn incidences. -/
theorem birth_edges {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) :
    actualEdgeCount (applyBirth s T hm eta).snapshot = actualEdgeCount s.snapshot + m := by
  letI := s.snapshot.adjDec
  letI := (applyBirth s T hm eta).snapshot.adjDec
  rw [edgeCount_sum_indicator, edgeCount_sum_indicator]
  simp only [Fin.sum_univ_castSucc]
  have notLast (v : Fin (n + 1)) : ¬ Fin.last n < v := not_lt_of_ge (Fin.le_last v)
  simp [applyBirth, birthSnapshot, birthGraph, birthAdj, notLast,
    Finset.sum_add_distrib, selected_card]

/-- Handshaking increment is derived from neighbor updates, not assumed. -/
theorem birth_degree_sum {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) :
    (∑ i, degree (applyBirth s T hm eta).snapshot i) =
      (∑ i, degree s.snapshot i) + 2*m := by
  rw [Fin.sum_univ_castSucc]
  change (∑ i : Fin n, degree (applyBirth s T hm eta).snapshot (oldId n i)) +
    degree (applyBirth s T hm eta).snapshot (newId n) = _
  simp_rw [birth_degree_old, birth_degree_new]
  rw [Finset.sum_add_distrib, target_sum]
  omega

/-- Connectedness is established during construction of the successor. -/
theorem birth_connected {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) : (applyBirth s T hm eta).snapshot.graph.Connected :=
  (applyBirth s T hm eta).valid.2.1

/-- Order affects trace probabilities, but not a birth with the same target set. -/
theorem birth_order_irrelevant {n m : Nat} (s : State n) (T U : Targets n m)
    (hm : 0 < m) (eta : PosFitness) (same : T.selected = U.selected) :
    applyBirth s T hm eta = applyBirth s U hm eta := by
  simp only [applyBirth, same]

/-- Incoming fitness influences future attractiveness, not this birth's topology. -/
theorem birth_new_fitness_independent {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta eta' : PosFitness) :
    (applyBirth s T hm eta).snapshot.graph = (applyBirth s T hm eta').snapshot.graph := rfl

/-- Old exact walks lift across the carrier extension with unchanged lengths. -/
theorem birth_walk_lifts {n m k : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) {u v : Fin n}
    (walk : MeshWalk s.snapshot.graph.Adj k u v) :
    MeshWalk (applyBirth s T hm eta).snapshot.graph.Adj k (oldId n u) (oldId n v) :=
  walk.mapNodes (oldId n) (fun h => (birth_old_adj_iff s T hm eta _ _).mpr h)

/-- Lift the same witness; no equality of shortest distances is asserted. -/
theorem birth_reachWithin_lifts {n m k : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) {u v : Fin n}
    (path : ReachWithin s.snapshot.graph.Adj k u v) :
    ReachWithin (applyBirth s T hm eta).snapshot.graph.Adj k (oldId n u) (oldId n v) := by
  obtain ⟨length, bound, walk⟩ := path
  exact ⟨length, bound, birth_walk_lifts s T hm eta walk⟩

end NarrativeDynamics.FitnessAttachment
