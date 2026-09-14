import NarrativeDynamics.Core.SocialMesh

namespace NarrativeDynamics.NetworkPropagation

open scoped BigOperators

/-- Fixed response parameters, with validity checked separately from storage. -/
structure AgentProfile where
  receptivity : Rat
  threshold : Rat
  deriving DecidableEq, Repr

/-- The scalar belief and cumulative count of received signals. -/
structure AgentState where
  belief : Rat
  exposures : Nat
  deriving DecidableEq, Repr

def AgentProfile.Valid (p : AgentProfile) : Prop :=
  0 ≤ p.receptivity ∧ p.receptivity ≤ 1 ∧
    0 ≤ p.threshold ∧ p.threshold ≤ 1

def AgentState.Valid (a : AgentState) : Prop :=
  0 ≤ a.belief ∧ a.belief ≤ 1

structure Population (n : Nat) where
  profiles : Fin n → AgentProfile
  agents : Fin n → AgentState

def Population.Valid {n : Nat} (p : Population n) : Prop :=
  (∀ i, (p.profiles i).Valid) ∧ (∀ i, (p.agents i).Valid)

/-- Equality at the threshold is sufficient to broadcast. -/
def broadcasting (p : AgentProfile) (a : AgentState) : Bool :=
  decide (p.threshold ≤ a.belief)

def incoming {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) (i : Fin n) : Finset (Fin n) :=
  Finset.univ.filter fun j => g j i ∧ broadcasting (p.profiles j) (p.agents j) = true

def transmissions {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) : Finset (Fin n × Fin n) :=
  (Finset.univ.product Finset.univ).filter fun pair =>
    g pair.1 pair.2 ∧ broadcasting (p.profiles pair.1) (p.agents pair.1) = true

/-- Every received belief is read from the supplied snapshot. -/
def nextAgent {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) (i : Fin n) : AgentState :=
  let received := incoming g p i
  let count := received.card
  if count = 0 then p.agents i
  else
    let signal := (∑ j ∈ received, (p.agents j).belief) / (count : Rat)
    { belief := (1 - (p.profiles i).receptivity) * (p.agents i).belief +
        (p.profiles i).receptivity * signal
      exposures := (p.agents i).exposures + count }

/-- One synchronous round preserves all profiles. -/
def propagate {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) : Population n :=
  { p with agents := nextAgent g p }

theorem transmission_iff {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) (j i : Fin n) :
    (j, i) ∈ transmissions g p ↔
      g j i ∧ broadcasting (p.profiles j) (p.agents j) = true := by
  simp [transmissions]

theorem nextAgent_no_incoming {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) (i : Fin n) (h : incoming g p i = ∅) :
    nextAgent g p i = p.agents i := by
  simp [nextAgent, h]

theorem exposures_mono {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) (i : Fin n) :
    (p.agents i).exposures ≤ (nextAgent g p i).exposures := by
  unfold nextAgent
  dsimp only
  split <;> simp

/-- A finite mean of valid beliefs is valid; the update is a convex combination. -/
theorem nextAgent_valid {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) (hp : p.Valid) (i : Fin n) : (nextAgent g p i).Valid := by
  unfold nextAgent
  dsimp only
  split
  · exact hp.2 i
  · rename_i hcount
    have hpositive : 0 < ((incoming g p i).card : Rat) := by
      exact_mod_cast Nat.pos_of_ne_zero hcount
    have hsum_nonneg : 0 ≤ ∑ j ∈ incoming g p i, (p.agents j).belief :=
      Finset.sum_nonneg fun j _ => (hp.2 j).1
    have hsum_le : (∑ j ∈ incoming g p i, (p.agents j).belief) ≤
        ((incoming g p i).card : Rat) := by
      calc
        (∑ j ∈ incoming g p i, (p.agents j).belief) ≤
            ∑ _j ∈ incoming g p i, (1 : Rat) :=
          Finset.sum_le_sum fun j _ => (hp.2 j).2
        _ = ((incoming g p i).card : Rat) := by simp
    have hsignal_nonneg := div_nonneg hsum_nonneg hpositive.le
    have hsignal_le : (∑ j ∈ incoming g p i, (p.agents j).belief) /
        ((incoming g p i).card : Rat) ≤ 1 := by
      apply (div_le_iff₀ hpositive).2
      simpa using hsum_le
    have hr := hp.1 i
    have ha := hp.2 i
    dsimp only [AgentState.Valid]
    constructor
    · exact add_nonneg (mul_nonneg (sub_nonneg.mpr hr.2.1) ha.1)
        (mul_nonneg hr.1 hsignal_nonneg)
    · have hleft := mul_le_mul_of_nonneg_left ha.2 (sub_nonneg.mpr hr.2.1)
      have hright := mul_le_mul_of_nonneg_left hsignal_le hr.1
      nlinarith

theorem propagate_valid {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) (hp : p.Valid) : (propagate g p).Valid := by
  exact ⟨hp.1, fun i => nextAgent_valid g p hp i⟩

/-- A receiver reads only its own profile/state and adjacent source beliefs and
broadcast decisions. In particular, source exposure counts are irrelevant. -/
theorem nextAgent_locality {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p q : Population n) (i : Fin n)
    (hstate : p.agents i = q.agents i)
    (hprofile : p.profiles i = q.profiles i)
    (hbelief : ∀ j, g j i → (p.agents j).belief = (q.agents j).belief)
    (hbroadcast : ∀ j, g j i →
      broadcasting (p.profiles j) (p.agents j) =
        broadcasting (q.profiles j) (q.agents j)) :
    nextAgent g p i = nextAgent g q i := by
  have hreceived : incoming g p i = incoming g q i := by
    ext j
    by_cases hedge : g j i
    · simp [incoming, hedge, hbroadcast j hedge]
    · simp [incoming, hedge]
  have hsum : (∑ j ∈ incoming g q i, (p.agents j).belief) =
      ∑ j ∈ incoming g q i, (q.agents j).belief := by
    apply Finset.sum_congr rfl
    intro j hj
    exact hbelief j (Finset.mem_filter.mp hj).2.1
  simp only [nextAgent, hreceived, hsum, hstate, hprofile]

end NarrativeDynamics.NetworkPropagation
