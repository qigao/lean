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

end NarrativeDynamics
