import NarrativeDynamics.Core.Epistemic

namespace NarrativeDynamics

/-- Minimal world graph needed to state causal/epistemic/action invariants.
`Node` is the shared information-graph node type; events and agents are mapped
into it through `eventNode` and `agentNode`. -/
structure WorldGraph (Agent Event Node : Type*) where
  alive : Agent → Prop
  canAct : Agent → Prop
  info : InfoGraph Node
  eventNode : Event → Node
  agentNode : Agent → Node
  observed : Event → Agent → Prop
  causal : Event → Event → Prop
  eventTime : Event → Nat

/-- Global well-formedness conditions for the minimal temporal causal world.
1. dead agents cannot act;
2. every observation has an information path;
3. causal edges strictly increase time. -/
def WorldInvariant {Agent Event Node : Type*} (w : WorldGraph Agent Event Node) : Prop :=
  (∀ a, ¬ w.alive a → ¬ w.canAct a) ∧
  (∀ e a, w.observed e a →
    infoReachable w.info (w.eventNode e) (w.agentNode a)) ∧
  (∀ u v, w.causal u v → w.eventTime u < w.eventTime v)

/-- Killing an agent removes both liveness and action capability while leaving
all causal and information structure unchanged. -/
def killAgent {Agent Event Node : Type*} [DecidableEq Agent]
    (w : WorldGraph Agent Event Node) (target : Agent) : WorldGraph Agent Event Node :=
  { w with
    alive := fun a => a ≠ target ∧ w.alive a
    canAct := fun a => a ≠ target ∧ w.canAct a }

/-- `killAgent` preserves all world invariants. -/
theorem killAgent_preserves_invariant {Agent Event Node : Type*} [DecidableEq Agent]
    (w : WorldGraph Agent Event Node) (h : WorldInvariant w) (target : Agent) :
    WorldInvariant (killAgent w target) := by
  rcases h with ⟨hdead, hobs, hcausal⟩
  refine ⟨?_, ?_, ?_⟩
  · intro a hnotalive hcan
    have hnotOldAlive : ¬ w.alive a := by
      intro halive
      exact hnotalive ⟨hcan.1, halive⟩
    exact hdead a hnotOldAlive hcan.2
  · intro e a hobserved
    exact hobs e a hobserved
  · intro u v hc
    exact hcausal u v hc

/-- Add an observation only after providing a proof that an information path
exists from the event node to the agent node. -/
def addObservation {Agent Event Node : Type*}
    [DecidableEq Agent] [DecidableEq Event]
    (w : WorldGraph Agent Event Node) (event : Event) (agent : Agent)
    (_reachable : infoReachable w.info (w.eventNode event) (w.agentNode agent)) :
    WorldGraph Agent Event Node :=
  { w with
    observed := fun e a => w.observed e a ∨ (e = event ∧ a = agent) }

/-- A proof-carrying observation insertion preserves epistemic isolation. -/
theorem addObservation_preserves_invariant {Agent Event Node : Type*}
    [DecidableEq Agent] [DecidableEq Event]
    (w : WorldGraph Agent Event Node) (h : WorldInvariant w)
    (event : Event) (agent : Agent)
    (reachable : infoReachable w.info (w.eventNode event) (w.agentNode agent)) :
    WorldInvariant (addObservation w event agent reachable) := by
  rcases h with ⟨hdead, hobs, hcausal⟩
  refine ⟨hdead, ?_, hcausal⟩
  intro e a hobserved
  rcases hobserved with hold | hnew
  · exact hobs e a hold
  · rcases hnew with ⟨rfl, rfl⟩
    exact reachable

/-- Add a causal edge only after providing a proof that source time is
strictly earlier than destination time. -/
def addCausalEdge {Agent Event Node : Type*}
    [DecidableEq Event]
    (w : WorldGraph Agent Event Node) (source target : Event)
    (_timeOrder : w.eventTime source < w.eventTime target) :
    WorldGraph Agent Event Node :=
  { w with
    causal := fun u v => w.causal u v ∨ (u = source ∧ v = target) }

/-- A proof-carrying causal insertion preserves temporal acyclicity locally. -/
theorem addCausalEdge_preserves_invariant {Agent Event Node : Type*}
    [DecidableEq Event]
    (w : WorldGraph Agent Event Node) (h : WorldInvariant w)
    (source target : Event)
    (timeOrder : w.eventTime source < w.eventTime target) :
    WorldInvariant (addCausalEdge w source target timeOrder) := by
  rcases h with ⟨hdead, hobs, hcausal⟩
  refine ⟨hdead, hobs, ?_⟩
  intro u v hc
  rcases hc with hold | hnew
  · exact hcausal u v hold
  · rcases hnew with ⟨rfl, rfl⟩
    exact timeOrder

/-- Generic property used only after concrete rewrites have separately proved
that they preserve `WorldInvariant`. -/
def PreservesWorldInvariant {Agent Event Node : Type*}
    (rewrite : WorldGraph Agent Event Node → WorldGraph Agent Event Node) : Prop :=
  ∀ w, WorldInvariant w → WorldInvariant (rewrite w)

/-- Any finite sequence of already-proved invariant-preserving rewrites also
preserves the invariant. -/
theorem rewriteSequence_preserves_invariant {Agent Event Node : Type*}
    (rewrites : List (WorldGraph Agent Event Node → WorldGraph Agent Event Node))
    (hpres : ∀ r ∈ rewrites, PreservesWorldInvariant r)
    (w : WorldGraph Agent Event Node) (hw : WorldInvariant w) :
    WorldInvariant (rewrites.foldl (fun state r => r state) w) := by
  induction rewrites generalizing w with
  | nil =>
      simpa using hw
  | cons r rs ih =>
      simp only [List.foldl_cons]
      apply ih
      · intro r' hr'
        exact hpres r' (List.mem_cons_of_mem r hr')
      · exact hpres r (by simp) w hw

end NarrativeDynamics
