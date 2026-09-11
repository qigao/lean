import NarrativeDynamics.Core.FitnessBirth

/-!
# Checked raw inputs for finite BB growth

Raw arrays are validated before finite IDs or a successor are constructed.
Connectivity is a computed finite closure, related to existing mesh walks.
No parser branch repairs invalid input or returns a partial successor.
-/

namespace NarrativeDynamics.FitnessAttachment

open Internal

structure RawSeed where
  nodeCount : Nat
  fitness : Array Rat
  edges : Array (Nat × Nat)
  deriving Repr

structure RawBirth where
  fitness : Rat
  targets : Array Nat
  deriving Repr

/-- Positive m is evidence obtained by validation, not a caller assertion. -/
structure ValidatedBirth (n m : Nat) where
  targets : Targets n m
  fitness : PosFitness
  positive : 0 < m

namespace Internal

/-- The old reached set plus all its actual neighbors, iterated by a finite budget. -/
def reached {n : Nat} (s : Snapshot n) (source : Fin n) : Nat → Finset (Fin n)
  | 0 => {source}
  | k + 1 =>
    letI := s.adjDec
    let prev := reached s source k
    prev ∪ prev.biUnion (fun v => neighborSet s.graph.Adj v)

private theorem reached_mono {n : Nat} (s : Snapshot n) (a : Fin n)
    {k l : Nat} (h : k ≤ l) : reached s a k ⊆ reached s a l := by
  induction h with
  | refl => exact fun _ hx => hx
  | step h ih => exact fun _ hx => Finset.mem_union_left _ (ih hx)

private theorem mesh_last {n k : Nat} (s : Snapshot n) {a b : Fin n}
    (p : MeshWalk s.graph.Adj (k + 1) a b) :
    ∃ v, MeshWalk s.graph.Adj k a v ∧ s.graph.Adj v b := by
  induction k generalizing a b with
  | zero =>
    cases p with
    | step edge rest =>
      cases rest
      exact ⟨_, .refl _, edge⟩
  | succ k ih =>
    cases p with
    | step edge rest =>
      obtain ⟨v, pv, hv⟩ := ih rest
      exact ⟨v, .step edge pv, hv⟩

private theorem reached_of_walk {n k : Nat} (s : Snapshot n) {a b : Fin n}
    (p : MeshWalk s.graph.Adj k a b) : b ∈ reached s a k := by
  letI := s.adjDec
  induction k generalizing b with
  | zero => cases p; simp [reached]
  | succ k ih =>
    obtain ⟨v, pv, hv⟩ := mesh_last s p
    apply Finset.mem_union_right
    exact Finset.mem_biUnion.mpr ⟨v, ih pv, by simpa [neighborSet] using hv⟩

private theorem reached_sound {n : Nat} (s : Snapshot n) (a : Fin n)
    (k : Nat) {b : Fin n} (h : b ∈ reached s a k) :
    ReachWithin s.graph.Adj k a b := by
  letI := s.adjDec
  induction k generalizing b with
  | zero =>
    have he : b = a := by simpa [reached] using h
    subst b
    exact ⟨0, le_refl _, .refl _⟩
  | succ k ih =>
    rcases Finset.mem_union.mp h with old | fresh
    · exact reachWithin_mono (ih old) (Nat.le_succ _)
    · obtain ⟨v, hv, edge⟩ := Finset.mem_biUnion.mp fresh
      obtain ⟨len, bound, walk⟩ := ih hv
      have adj : s.graph.Adj v b := by simpa [neighborSet] using edge
      exact ⟨len + 1, by omega, walk.append (MeshWalk.single adj)⟩

/-- The finite computation accepts exactly the existing bounded-walk semantics. -/
theorem reached_iff {n : Nat} (s : Snapshot n) (a b : Fin n) (k : Nat) :
    b ∈ reached s a k ↔ ReachWithin s.graph.Adj k a b := by
  constructor
  · exact reached_sound s a k
  · rintro ⟨len, bound, walk⟩
    exact reached_mono s a bound (reached_of_walk s walk)

private theorem walk_to_mesh {n : Nat} (s : Snapshot n) {a b : Fin n}
    (p : s.graph.Walk a b) : MeshWalk s.graph.Adj p.length a b := by
  induction p with
  | nil => exact .refl _
  | cons edge rest ih => exact .step edge ih

private theorem mesh_to_reachable {n k : Nat} (s : Snapshot n) {a b : Fin n}
    (p : MeshWalk s.graph.Adj k a b) : s.graph.Reachable a b := by
  induction p with
  | refl => exact ⟨.nil⟩
  | step edge rest ih =>
    obtain ⟨tail⟩ := ih
    exact ⟨.cons edge tail⟩

/-- A simple path visits distinct finite vertices, hence has at most n-1 edges. -/
theorem connected_bounded {n : Nat} (s : Snapshot n) (h : s.graph.Connected) :
    GlobalHopBound s.graph.Adj (n - 1) := by
  intro a b
  obtain ⟨p, hp⟩ := (h a b).exists_isPath
  have bound : p.length < n := by simpa using hp.length_lt
  exact ⟨p.length, by omega, walk_to_mesh s p⟩

/-- No assumption of a precomputed diameter or connectivity oracle is used. -/
theorem state_bounded {n : Nat} (s : State n) :
    GlobalHopBound s.snapshot.graph.Adj (n - 1) :=
  connected_bounded s.snapshot s.valid.2.1

private theorem connected_of_reached {n : Nat} (s : Snapshot n) (a : Fin n)
    (h : ∀ b, b ∈ reached s a (n - 1)) : s.graph.Connected := by
  have hr (b : Fin n) : s.graph.Reachable a b := by
    obtain ⟨_, _, p⟩ := (reached_iff s a b (n - 1)).mp (h b)
    exact mesh_to_reachable s p
  exact { preconnected := fun u v => (hr u).symm.trans (hr v)
          nonempty := ⟨a⟩ }

/-- Normalize orientation for duplicate detection, but never remove duplicates. -/
def canonicalEdge (e : Nat × Nat) : Nat × Nat := (min e.1 e.2, max e.1 e.2)

def seedGraph (raw : RawSeed) : SimpleGraph (Fin raw.nodeCount) where
  Adj u v := u ≠ v ∧ canonicalEdge (u.val, v.val) ∈ raw.edges.toList.map canonicalEdge
  symm := ⟨by
    intro u v
    simp [canonicalEdge, min_comm, max_comm, ne_comm]⟩
  loopless := ⟨by intro u; simp⟩

@[reducible] def seedAdjDec (raw : RawSeed) : DecidableRel (seedGraph raw).Adj := by
  intro u v
  unfold seedGraph
  infer_instance

/-- Array access uses its checked exact dimension, without a default element. -/
def seedSnapshot (raw : RawSeed) (size : raw.fitness.size = raw.nodeCount) :
    Snapshot raw.nodeCount where
  graph := seedGraph raw
  adjDec := seedAdjDec raw
  fitness i := raw.fitness[i.val]'(by have := i.isLt; omega)

def positiveSeedFitness (raw : RawSeed) : Prop :=
  ∀ i : Fin raw.fitness.size, 0 < raw.fitness[i.val]

def validSeedEdges (raw : RawSeed) : Prop :=
  ∀ i : Fin raw.edges.size,
    (raw.edges[i.val]).1 < raw.nodeCount ∧
    (raw.edges[i.val]).2 < raw.nodeCount ∧
    (raw.edges[i.val]).1 ≠ (raw.edges[i.val]).2

instance (raw : RawSeed) : Decidable (positiveSeedFitness raw) :=
  inferInstanceAs (Decidable (∀ i : Fin raw.fitness.size, 0 < raw.fitness[i.val]))
instance (raw : RawSeed) : Decidable (validSeedEdges raw) :=
  inferInstanceAs (Decidable (∀ i : Fin raw.edges.size,
    (raw.edges[i.val]).1 < raw.nodeCount ∧ (raw.edges[i.val]).2 < raw.nodeCount ∧
      (raw.edges[i.val]).1 ≠ (raw.edges[i.val]).2))

end Internal

/-- Semantic raw seed conditions; independent of the parser's result. -/
structure RawSeed.Valid (raw : RawSeed) : Prop where
  nodes : 2 ≤ raw.nodeCount
  size : raw.fitness.size = raw.nodeCount
  fitness : positiveSeedFitness raw
  edges : validSeedEdges raw
  distinct : (raw.edges.toList.map canonicalEdge).Nodup
  connected : (seedGraph raw).Connected

/-- Validate in deterministic order, then derive validity from finite reachability. -/
def parseSeed (raw : RawSeed) : Except Error (State raw.nodeCount) :=
  if hn : 2 ≤ raw.nodeCount then
    if hs : raw.fitness.size = raw.nodeCount then
      if hf : positiveSeedFitness raw then
        if he : validSeedEdges raw then
          if hd : (raw.edges.toList.map canonicalEdge).Nodup then
            let s := seedSnapshot raw hs
            let a : Fin raw.nodeCount := ⟨0, by omega⟩
            if hc : ∀ b, b ∈ reached s a (raw.nodeCount - 1) then
              .ok ⟨s, hn, connected_of_reached s a hc, by
                intro i
                exact hf ⟨i.val, by have := i.isLt; omega⟩⟩
            else .error .disconnectedSeed
          else .error .duplicateEdge
        else .error .invalidEdge
      else .error .nonpositiveFitness
    else .error .fitnessSizeMismatch
  else .error .invalidNodeCount

/-- Every successful seed satisfies all documented raw conditions. -/
theorem parseSeed_sound (raw : RawSeed) (s : State raw.nodeCount)
    (h : parseSeed raw = .ok s) : raw.Valid := by
  unfold parseSeed at h
  split at h <;> try contradiction
  rename_i hn
  split at h <;> try contradiction
  rename_i hs
  split at h <;> try contradiction
  rename_i hf
  split at h <;> try contradiction
  rename_i he
  split at h <;> try contradiction
  rename_i hd
  dsimp only at h
  split at h <;> try contradiction
  rename_i hc
  exact ⟨hn, hs, hf, he, hd, connected_of_reached _ _ hc⟩

/-- Every semantically valid raw seed is accepted; no hidden stronger bound is assumed. -/
theorem parseSeed_complete (raw : RawSeed) (h : raw.Valid) :
    ∃ s, parseSeed raw = .ok s := by
  have hc : ∀ b, b ∈ reached (seedSnapshot raw h.size)
      (⟨0, by have := h.nodes; omega⟩ : Fin raw.nodeCount) (raw.nodeCount - 1) := by
    intro b
    apply (reached_iff _ _ _ _).mpr
    exact connected_bounded (seedSnapshot raw h.size) h.connected _ b
  simp only [parseSeed, dif_pos h.nodes, dif_pos h.size, dif_pos h.fitness,
    dif_pos h.edges, dif_pos h.distinct, dif_pos hc]
  exact ⟨_, rfl⟩

/-- Successful graph construction uses exactly the raw undirected edge relation. -/
theorem parseSeed_graph (raw : RawSeed) (s : State raw.nodeCount)
    (h : parseSeed raw = .ok s) : s.snapshot.graph = seedGraph raw := by
  unfold parseSeed at h
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h <;> try contradiction
  dsimp only at h
  split at h <;> try contradiction
  cases h
  rfl

namespace Internal

/-- The original array remains authoritative for both order and numeric identity. -/
def targetsBounded (n : Nat) (xs : Array Nat) : Prop :=
  ∀ i : Fin xs.size, xs[i.val] < n

def targetsDistinct (xs : Array Nat) : Prop :=
  Function.Injective (fun i : Fin xs.size => xs[i.val])

instance (n : Nat) (xs : Array Nat) : Decidable (targetsBounded n xs) :=
  inferInstanceAs (Decidable (∀ i : Fin xs.size, xs[i.val] < n))
instance (xs : Array Nat) : Decidable (targetsDistinct xs) :=
  inferInstanceAs (Decidable (Function.Injective (fun i : Fin xs.size => xs[i.val])))

structure CheckedTargets (n : Nat) (xs : Array Nat) : Type where
  bounded : targetsBounded n xs
  distinct : targetsDistinct xs

def CheckedTargets.embedding {n : Nat} {xs : Array Nat} (h : CheckedTargets n xs) :
    Targets n xs.size :=
  ⟨fun i => ⟨xs[i.val], h.bounded i⟩,
    fun _ _ eq => h.distinct (congrArg Fin.val eq)⟩

/-- Check bounds before distinctness, without converting to a set first. -/
def checkTargets (n : Nat) (xs : Array Nat) : Except Error (CheckedTargets n xs) :=
  if hb : targetsBounded n xs then
    if hd : targetsDistinct xs then .ok ⟨hb, hd⟩
    else .error .duplicateTarget
  else .error .targetOutOfRange

end Internal

/-- Raw request validity is a predicate on inputs, not on parser success. -/
def RawBirth.Valid (raw : RawBirth) (n m : Nat) : Prop :=
  (0 < m ∧ m ≤ n) ∧ 0 < raw.fitness ∧ raw.targets.size = m ∧
    targetsBounded n raw.targets ∧ targetsDistinct raw.targets

/-- Validate all fields before creating the existing typed target embedding. -/
def validateBirth {n : Nat} (_s : State n) (m : Nat) (raw : RawBirth) :
    Except Error (ValidatedBirth n m) :=
  if hm : 0 < m ∧ m ≤ n then
    if hf : 0 < raw.fitness then
      if hs : raw.targets.size = m then
        match checkTargets n raw.targets with
        | .error e => .error e
        | .ok checked => .ok ⟨hs ▸ checked.embedding, ⟨raw.fitness, hf⟩, hm.1⟩
      else .error .targetCountMismatch
    else .error .nonpositiveFitness
  else .error .invalidM

theorem validateBirth_sound {n m : Nat} (s : State n) (raw : RawBirth)
    (v : ValidatedBirth n m) (h : validateBirth s m raw = .ok v) : raw.Valid n m := by
  unfold validateBirth at h
  split at h <;> try contradiction
  rename_i hm
  split at h <;> try contradiction
  rename_i hf
  split at h <;> try contradiction
  rename_i hs
  cases hc : checkTargets n raw.targets with
  | error e => simp [hc] at h
  | ok checked => exact ⟨hm, hf, hs, checked.bounded, checked.distinct⟩

theorem validateBirth_complete {n m : Nat} (s : State n) (raw : RawBirth)
    (h : raw.Valid n m) : ∃ v, validateBirth s m raw = .ok v := by
  rcases h with ⟨hm, hf, hs, hb, hd⟩
  simp only [validateBirth, dif_pos hm, dif_pos hf, dif_pos hs,
    checkTargets, dif_pos hb, dif_pos hd]
  exact ⟨_, rfl⟩

/-- A successful validator preserves the supplied order, IDs, and fitness exactly. -/
theorem validateBirth_values {n m : Nat} (s : State n) (raw : RawBirth)
    (v : ValidatedBirth n m) (h : validateBirth s m raw = .ok v) :
    v.fitness.val = raw.fitness ∧
      ∃ hs : raw.targets.size = m, ∀ i : Fin raw.targets.size,
        (v.targets (Fin.cast hs i)).val = raw.targets[i.val] := by
  unfold validateBirth at h
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h <;> try contradiction
  rename_i hs
  cases hc : checkTargets n raw.targets with
  | error e => simp [hc] at h
  | ok checked =>
    simp only [hc] at h
    cases hs
    cases h
    exact ⟨rfl, rfl, fun _ => rfl⟩

/-- A raw prefix is not a growth request: the empty prefix is legal, exhaustion is not. -/
def rawAttachmentRow {n : Nat} (s : State n) (selectedPrefix : Array Nat) : Except Error (Row n) :=
  match checkTargets n selectedPrefix with
  | .error e => .error e
  | .ok checked =>
    let T := checked.embedding
    if h : T.selected.card < n then .ok (attachmentRow s T.selected h)
    else .error .zeroMass

/-- There is one success boundary, after complete validation of the entire request. -/
def step {n : Nat} (s : State n) (m : Nat) (raw : RawBirth) :
    Except Error (State (n + 1) × Rat) :=
  match validateBirth s m raw with
  | .error e => .error e
  | .ok v => .ok (applyBirth s v.targets v.positive v.fitness, orderedMass s v.targets)

/-- Success is precisely the existing typed birth paired with its exact ordered law. -/
theorem step_spec {n m : Nat} (s : State n) (raw : RawBirth)
    (out : State (n + 1) × Rat) :
    step s m raw = .ok out ↔
      ∃ v : ValidatedBirth n m, validateBirth s m raw = .ok v ∧
        out = (applyBirth s v.targets v.positive v.fitness, orderedMass s v.targets) := by
  unfold step
  cases hv : validateBirth s m raw with
  | error e => simp
  | ok v => simp [Except.ok.injEq, eq_comm]

/-- Invalid requests return only the same error, with no successor or probability. -/
theorem step_error {n m : Nat} (s : State n) (raw : RawBirth) (e : Error)
    (h : validateBirth s m raw = .error e) : step s m raw = .error e := by
  simp [step, h]

end NarrativeDynamics.FitnessAttachment
