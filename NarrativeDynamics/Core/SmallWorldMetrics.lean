import NarrativeDynamics.Core.SmallWorld

namespace NarrativeDynamics

/-- The finite set of outgoing neighbors of one center node. -/
def neighborSet {Node : Type*} [Fintype Node] [DecidableEq Node]
    (g : MeshGraph Node) [DecidableRel g] (center : Node) : Finset Node :=
  Finset.univ.filter (g center)

/-- Distinct ordered pairs of outgoing neighbors. On a symmetric graph, using
ordered instead of unordered pairs leaves the resulting ratio unchanged. -/
def orderedNeighborPairs {Node : Type*} [Fintype Node] [DecidableEq Node]
    (g : MeshGraph Node) [DecidableRel g] (center : Node) : Finset (Node × Node) :=
  ((neighborSet g center).product (neighborSet g center)).filter
    (fun pair => pair.1 ≠ pair.2)

/-- Candidate neighbor pairs whose endpoints are joined by an edge. -/
def closedNeighborPairs {Node : Type*} [Fintype Node] [DecidableEq Node]
    (g : MeshGraph Node) [DecidableRel g] (center : Node) : Finset (Node × Node) :=
  (orderedNeighborPairs g center).filter (fun pair => g pair.1 pair.2)

/-- Rational local clustering coefficient. A zero candidate-pair denominator
evaluates to zero under the rational-field convention. -/
def localClusteringCoefficient {Node : Type*} [Fintype Node] [DecidableEq Node]
    (g : MeshGraph Node) [DecidableRel g] (center : Node) : ℚ :=
  ((closedNeighborPairs g center).card : ℚ) /
    ((orderedNeighborPairs g center).card : ℚ)

/-- Every closed neighbor pair is one of the candidate neighbor pairs. -/
theorem closedNeighborPairs_subset {Node : Type*} [Fintype Node]
    [DecidableEq Node] (g : MeshGraph Node) [DecidableRel g] (center : Node) :
    closedNeighborPairs g center ⊆ orderedNeighborPairs g center := by
  exact Finset.filter_subset _ _

/-- Perfect local clustering closes every candidate neighbor pair. -/
theorem closedNeighborPairs_eq_of_perfect {Node : Type*} [Fintype Node]
    [DecidableEq Node] (g : MeshGraph Node) [DecidableRel g] (center : Node)
    (clustered : PerfectLocalClustering g) :
    closedNeighborPairs g center = orderedNeighborPairs g center := by
  rw [closedNeighborPairs]
  apply Finset.filter_eq_self.2
  intro pair pairIn
  rw [orderedNeighborPairs] at pairIn
  rcases Finset.mem_filter.mp pairIn with ⟨pairMembers, different⟩
  rcases Finset.mem_product.mp pairMembers with ⟨leftMember, rightMember⟩
  rw [neighborSet] at leftMember rightMember
  exact clustered different (Finset.mem_filter.mp leftMember).2
    (Finset.mem_filter.mp rightMember).2

/-- A nontrivial perfectly clustered neighborhood has clustering coefficient one. -/
theorem localClusteringCoefficient_eq_one_of_perfect {Node : Type*}
    [Fintype Node] [DecidableEq Node] (g : MeshGraph Node) [DecidableRel g]
    (center : Node) (clustered : PerfectLocalClustering g)
    (pairsExist : (orderedNeighborPairs g center).Nonempty) :
    localClusteringCoefficient g center = 1 := by
  rw [localClusteringCoefficient,
    closedNeighborPairs_eq_of_perfect g center clustered]
  exact div_self (by exact_mod_cast Finset.card_ne_zero.mpr pairsExist)

/-- A global hop witness supplies a bounded path for each particular node pair. -/
theorem exists_reachWithin_of_globalHopBound {Node : Type*}
    (g : MeshGraph Node) (bounded : ∃ limit, GlobalHopBound g limit)
    (source target : Node) : ∃ limit, ReachWithin g limit source target := by
  rcases bounded with ⟨limit, globalBound⟩
  exact ⟨limit, globalBound source target⟩

/-- The least inclusive hop budget under which the selected node pair is
reachable. -/
noncomputable def shortestHopCount {Node : Type*} (g : MeshGraph Node)
    (bounded : ∃ limit, GlobalHopBound g limit) (source target : Node) : Nat :=
  by
    classical
    exact Nat.find (exists_reachWithin_of_globalHopBound g bounded source target)

/-- The least hop count returned by `shortestHopCount` is itself reachable. -/
theorem shortestHopCount_spec {Node : Type*} (g : MeshGraph Node)
    (bounded : ∃ limit, GlobalHopBound g limit) (source target : Node) :
    ReachWithin g (shortestHopCount g bounded source target) source target := by
  classical
  exact Nat.find_spec (exists_reachWithin_of_globalHopBound g bounded source target)

/-- No witnessed hop budget is smaller than `shortestHopCount`. -/
theorem shortestHopCount_minimal {Node : Type*} (g : MeshGraph Node)
    (bounded : ∃ limit, GlobalHopBound g limit) {source target : Node}
    {candidate : Nat} (reachable : ReachWithin g candidate source target) :
    shortestHopCount g bounded source target ≤ candidate := by
  classical
  exact Nat.find_min' (exists_reachWithin_of_globalHopBound g bounded source target)
    reachable

/-- The least uniform hop budget that reaches every ordered node pair. -/
noncomputable def meshDiameter {Node : Type*} (g : MeshGraph Node)
    (bounded : ∃ limit, GlobalHopBound g limit) : Nat := by
  classical
  exact Nat.find bounded

/-- The selected mesh diameter really is a global hop bound. -/
theorem meshDiameter_spec {Node : Type*} (g : MeshGraph Node)
    (bounded : ∃ limit, GlobalHopBound g limit) :
    GlobalHopBound g (meshDiameter g bounded) := by
  classical
  exact Nat.find_spec bounded

/-- Every other global hop bound is at least the selected mesh diameter. -/
theorem meshDiameter_minimal {Node : Type*} (g : MeshGraph Node)
    (bounded : ∃ limit, GlobalHopBound g limit) {candidate : Nat}
    (candidateBound : GlobalHopBound g candidate) :
    meshDiameter g bounded ≤ candidate := by
  classical
  exact Nat.find_min' bounded candidateBound

/-- A small-world certificate's advertised limit bounds its exact mesh diameter. -/
theorem SmallWorldCertificate.meshDiameter_le {Node : Type*}
    {g : MeshGraph Node} {limit : Nat}
    (certificate : SmallWorldCertificate g limit) :
    meshDiameter g ⟨limit, certificate.shortPaths⟩ ≤ limit :=
  meshDiameter_minimal g ⟨limit, certificate.shortPaths⟩ certificate.shortPaths

end NarrativeDynamics
